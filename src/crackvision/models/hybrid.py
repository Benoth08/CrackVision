"""Modele hybride : fusion du detecteur supervise et de l'autoencodeur.

Repartition des roles (voir l'article de reference cite dans le README) :
- le supervise couvre le volume : il localise et nomme les fissures connues ;
- le non supervise couvre le risque : il signale ce qui s'ecarte du beton sain,
  y compris des defauts absents des annotations.

La fusion se fait a deux niveaux, par regressions logistiques ajustees sur la
validation (jamais sur le blind test) :
1. boite : chaque detection YOLOX est re-notee avec l'evidence de l'autoencodeur
   a l'interieur de la boite (une fausse alarme sur du beton sain est penalisee) ;
2. tuile : la probabilite de defaut combine le meilleur score YOLOX, le nombre de
   boites et le score d'anomalie de l'autoencodeur.

Filet de securite : une zone tres anormale (au-dela du quantile 99,9 % des pixels
sains) sans aucune boite YOLOX est remontee comme anomalie non classee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from crackvision.utils import read_json

IMAGE_FEATURES = ["det_max", "det_count", "ae_score_z", "ae_area_frac"]
BOX_FEATURES = ["det_logit", "ae_box_z", "ae_box_frac", "box_log_area"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


@dataclass
class LogisticFusion:
    """Regression logistique sur variables centrees-reduites, sans dependance a sklearn."""

    features: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[0] == 0:
            return np.zeros(0)
        z = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        return sigmoid(z @ np.asarray(self.coef) + self.intercept)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> LogisticFusion | None:
        return cls(**payload) if payload else None


# --------------------------------------------------------------------------- variables
def box_features(dets: np.ndarray, amap: np.ndarray, pixel_stats: dict,
                 topk: float) -> np.ndarray:
    """Variables par boite : logit du score YOLOX, evidence de l'autoencodeur, taille."""
    features = np.zeros((len(dets), len(BOX_FEATURES)), dtype=np.float64)
    height, width = amap.shape
    std = max(pixel_stats["std"], 1e-6)
    for index, (x1, y1, x2, y2, score) in enumerate(dets):
        x1i, y1i = int(np.clip(np.floor(x1), 0, width - 1)), int(np.clip(np.floor(y1), 0, height - 1))
        x2i, y2i = int(np.clip(np.ceil(x2), x1i + 1, width)), int(np.clip(np.ceil(y2), y1i + 1, height))
        values = amap[y1i:y2i, x1i:x2i].ravel()
        count = max(1, int(round(values.size * topk)))
        top = np.partition(values, values.size - count)[-count:].mean()
        features[index] = [
            logit(score),
            (top - pixel_stats["mean"]) / std,
            float((values > pixel_stats["threshold"]).mean()),
            np.log(max((x2 - x1) * (y2 - y1), 1.0) / float(height * width)),
        ]
    return features


def image_features(dets: np.ndarray, amap: np.ndarray, ae_score_z: float, pixel_threshold: float,
                   box_threshold: float) -> np.ndarray:
    det_max = float(dets[:, 4].max()) if len(dets) else 0.0
    det_count = float(np.log1p((dets[:, 4] >= box_threshold).sum())) if len(dets) else 0.0
    return np.array([det_max, det_count, float(ae_score_z),
                     float((amap > pixel_threshold).mean())], dtype=np.float64)


def unknown_regions(amap: np.ndarray, threshold: float, min_area: int, boxes: np.ndarray,
                    max_overlap: float) -> list[dict]:
    """Zones tres anormales qui ne recouvrent (presque) aucune boite YOLOX."""
    binary = (amap > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
        w, h = int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])
        overlap = 0.0
        for bx1, by1, bx2, by2 in boxes[:, :4] if len(boxes) else []:
            inter_w = max(0.0, min(x + w, bx2) - max(x, bx1))
            inter_h = max(0.0, min(y + h, by2) - max(y, by1))
            overlap = max(overlap, inter_w * inter_h / float(w * h))
        if overlap <= max_overlap:
            regions.append({"box": [x, y, x + w, y + h], "area": area,
                            "score": float(amap[y:y + h, x:x + w].max())})
    return regions


# --------------------------------------------------------------------------- modele
class HybridModel:
    def __init__(self, fusion: dict):
        self.fusion = fusion
        self.image_model = LogisticFusion.from_dict(fusion["image_model"])
        self.box_model = LogisticFusion.from_dict(fusion.get("box_model"))
        self.params = fusion["params"]
        self.pixel_stats = fusion["pixel_stats"]
        self.points = fusion["operating_points"]

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> HybridModel:
        return cls(read_json(Path(model_dir) / "fusion.json"))

    def rescore_boxes(self, dets: np.ndarray, amap: np.ndarray) -> np.ndarray:
        if len(dets) == 0:
            return dets.reshape(0, 5)
        if self.box_model is None:
            return dets
        features = box_features(dets, amap, self.pixel_stats, self.params["box_evidence_topk"])
        fused = dets.copy()
        fused[:, 4] = self.box_model.predict_proba(features)
        return fused

    def predict(self, dets: np.ndarray, amap: np.ndarray, ae_score_z: float) -> dict:
        """Sortie hybride pour une tuile (ou une image) deja traitee par les deux modeles."""
        yolox_box_threshold = self.points["yolox"]["box_threshold"]
        features = image_features(dets, amap, ae_score_z, self.pixel_stats["threshold"],
                                  yolox_box_threshold)
        proba = float(self.image_model.predict_proba(features)[0])
        fused = self.rescore_boxes(dets, amap)
        box_threshold = self.points["hybrid"]["box_threshold"]
        confirmed = fused[fused[:, 4] >= box_threshold] if len(fused) else fused
        regions = unknown_regions(amap, self.params["unknown_threshold"],
                                  int(self.params["unknown_min_area"]), confirmed,
                                  float(self.params["unknown_max_overlap"]))
        decision = proba >= self.points["hybrid"]["image_threshold"]
        if decision and len(confirmed):
            level = "fissure"
        elif decision or regions:
            level = "anomalie_non_classee"
        else:
            level = "sain"
        return {
            "proba": proba,
            "decision": bool(decision),
            "level": level,
            "boxes": fused,
            "confirmed_boxes": confirmed,
            "unknown_regions": regions,
        }
