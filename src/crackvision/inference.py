"""Chaine d'inference commune a l'evaluation et a l'API.

Une image entiere suit exactement le pretraitement de l'entrainement : reduction
d'echelle `prepare.scale`, decoupage en tuiles `prepare.tile_size`, puis les
resultats des tuiles sont recomposes dans le repere de l'image d'origine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from crackvision.data.tiling import merge_tile_detections, pad_to_min, stitch_maps, tile_grid
from crackvision.models.autoencoder import AnomalyDetector
from crackvision.models.hybrid import HybridModel
from crackvision.models.yolox_model import YoloxDetector
from crackvision.utils import read_json

MODEL_NAMES = ("yolox", "autoencoder", "hybrid")


@dataclass
class TilePrediction:
    yolox: np.ndarray | None
    amap: np.ndarray | None
    ae_score: float | None
    ae_score_z: float | None
    hybrid: dict | None = None


class CrackVisionPipeline:
    def __init__(self, detector: YoloxDetector, anomaly: AnomalyDetector,
                 hybrid: HybridModel | None = None, det_conf_floor: float = 0.01,
                 prepare: dict | None = None):
        self.detector = detector
        self.anomaly = anomaly
        self.hybrid = hybrid
        self.det_conf_floor = det_conf_floor
        self.prepare = prepare or {"scale": 1.0, "tile_size": detector.input_size[0]}

    @classmethod
    def from_dir(cls, models_dir: str | Path, device: str = "cpu",
                 with_hybrid: bool = True) -> CrackVisionPipeline:
        models_dir = Path(models_dir)
        detector = YoloxDetector.from_dir(models_dir / "yolox", device)
        anomaly = AnomalyDetector.from_dir(models_dir / "autoencoder", device)
        hybrid, floor = None, 0.01
        if with_hybrid and (models_dir / "hybrid" / "fusion.json").exists():
            hybrid = HybridModel.from_dir(models_dir / "hybrid")
            floor = float(hybrid.params["det_conf_floor"])
        prepare = read_json(models_dir / "yolox" / "meta.json").get("prepare")
        return cls(detector, anomaly, hybrid, floor, prepare)

    # ----------------------------------------------------------------- tuiles
    def predict_tiles(self, tiles: list[np.ndarray], mode: str = "hybrid",
                      batch_size: int = 16) -> list[TilePrediction]:
        """`mode` : 'yolox', 'autoencoder' ou 'hybrid' (les deux modeles + fusion)."""
        if mode not in MODEL_NAMES:
            raise ValueError(f"Modele inconnu : {mode}")
        if mode == "hybrid" and self.hybrid is None:
            raise RuntimeError("Fusion hybride non calibree (models/hybrid/fusion.json absent)")
        n = len(tiles)
        dets = [None] * n
        maps, scores, scores_z = [None] * n, [None] * n, [None] * n
        if mode in ("yolox", "hybrid"):
            dets = self.detector.predict(tiles, conf=self.det_conf_floor, batch_size=batch_size)
        if mode in ("autoencoder", "hybrid"):
            maps, raw = self.anomaly.predict(tiles, batch_size=batch_size)
            scores, scores_z = raw.tolist(), self.anomaly.score_z(raw).tolist()
        predictions = []
        for det, amap, score, score_z in zip(dets, maps, scores, scores_z, strict=True):
            prediction = TilePrediction(det, amap, score, score_z)
            if mode == "hybrid":
                prediction.hybrid = self.hybrid.predict(det, amap, float(score_z))
            predictions.append(prediction)
        return predictions

    # ----------------------------------------------------------------- image entiere
    def predict_image(self, image_bgr: np.ndarray, mode: str = "hybrid") -> dict:
        scale = float(self.prepare["scale"])
        tile_size = int(self.prepare["tile_size"])
        original_h, original_w = image_bgr.shape[:2]
        work = image_bgr
        if scale != 1.0:
            work = cv2.resize(image_bgr, (max(int(round(original_w * scale)), 1),
                                          max(int(round(original_h * scale)), 1)),
                              interpolation=cv2.INTER_AREA)
        padded, (height, width) = pad_to_min(work, tile_size)
        grid = tile_grid(padded.shape[0], padded.shape[1], tile_size)
        positions = [(y0, x0) for _, _, y0, x0 in grid]
        tiles = [padded[y0:y0 + tile_size, x0:x0 + tile_size] for y0, x0 in positions]
        predictions = self.predict_tiles(tiles, mode=mode)

        def gather(boxes_per_tile: list[np.ndarray]) -> np.ndarray:
            """Boites de toutes les tuiles, fusionnees, dans le repere d'origine."""
            shifted = []
            for boxes, (y0, x0) in zip(boxes_per_tile, positions, strict=True):
                if len(boxes):
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    shifted.append(boxes)
            if not shifted:
                return np.zeros((0, 5), dtype=np.float32)
            merged = merge_tile_detections(np.concatenate(shifted))
            merged[:, [0, 2]] = merged[:, [0, 2]].clip(0, width) / scale
            merged[:, [1, 3]] = merged[:, [1, 3]].clip(0, height) / scale
            return merged

        result = {
            "mode": mode,
            "image_size": [original_w, original_h],
            "work_size": [width, height],
            "scale": scale,
            "n_tiles": len(tiles),
        }
        if mode in ("yolox", "hybrid"):
            result["yolox"] = gather([p.yolox for p in predictions])
        if mode in ("autoencoder", "hybrid"):
            amap = stitch_maps([p.amap for p in predictions], positions, padded.shape[:2])
            best = int(np.argmax([p.ae_score for p in predictions]))
            result.update({"anomaly_map": amap[:height, :width],
                           "ae_score": predictions[best].ae_score,
                           "ae_score_z": predictions[best].ae_score_z})
        if mode == "hybrid":
            hybrid = [p.hybrid for p in predictions]
            levels = [h["level"] for h in hybrid]
            regions = []
            for h, (y0, x0) in zip(hybrid, positions, strict=True):
                for region in h["unknown_regions"]:
                    x1, y1, x2, y2 = region["box"]
                    regions.append({**region, "box": [(x1 + x0) / scale, (y1 + y0) / scale,
                                                      min(x2 + x0, width) / scale,
                                                      min(y2 + y0, height) / scale]})
            result["hybrid"] = {
                "proba": max(h["proba"] for h in hybrid),
                "decision": any(h["decision"] for h in hybrid),
                "level": ("fissure" if "fissure" in levels else
                          "anomalie_non_classee" if "anomalie_non_classee" in levels else "sain"),
                "boxes": gather([h["confirmed_boxes"] for h in hybrid]),
                "unknown_regions": regions,
            }
        return result
