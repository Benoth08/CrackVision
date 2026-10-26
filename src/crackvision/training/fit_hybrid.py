"""Etape `calibrate` : points de fonctionnement et ajustement de la fusion hybride.

Tout se fait sur la validation :
1. seuils de decision de YOLOX (tuile et boite) et de l'autoencodeur (tuile),
   choisis pour maximiser le F1 ;
2. regressions logistiques de la fusion (tuile et boite), avec ponderation
   equilibree des classes ; leurs seuils sont choisis sur des predictions
   hors echantillon (validation croisee groupee par photo), pour ne pas choisir
   un seuil sur des predictions que le modele a deja vues.

Le blind test n'intervient a aucun moment.
"""

from __future__ import annotations

import shutil

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from crackvision.data.prepare import read_tiles, tile_boxes
from crackvision.evaluation.metrics import (
    best_box_threshold,
    best_threshold,
    box_prf,
    image_metrics,
    match_detections,
)
from crackvision.inference import CrackVisionPipeline
from crackvision.models.hybrid import (
    BOX_FEATURES,
    IMAGE_FEATURES,
    LogisticFusion,
    box_features,
)
from crackvision.paths import Paths
from crackvision.utils import (
    ensure_dir,
    get_logger,
    select_device,
    sha256_file,
    write_json,
)

logger = get_logger(__name__)


def collect(pipeline: CrackVisionPipeline, paths: Paths, split: str, pixel_stats: dict,
            box_topk: float, batch_size: int = 32) -> list[dict]:
    """Passe les deux modeles sur un jeu ; ne garde que des resumes (pas les cartes)."""
    tiles = read_tiles(paths, split)
    image_dir = paths.split_dir(split) / "images"
    records = []
    for start in range(0, len(tiles), batch_size):
        batch = tiles.iloc[start:start + batch_size]
        images = [cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR)
                  for name in batch["file_name"]]
        dets = pipeline.detector.predict(images, conf=pipeline.det_conf_floor,
                                         batch_size=batch_size)
        maps, scores = pipeline.anomaly.predict(images, batch_size=batch_size)
        scores_z = pipeline.anomaly.score_z(scores)
        for (_, tile), det, amap, score, score_z in zip(batch.iterrows(), dets, maps, scores,
                                                        scores_z, strict=True):
            records.append({
                "tile_id": tile["tile_id"],
                "photo_id": str(tile["photo_id"]),
                "label": int(tile["label"] == "positive"),
                "gts": tile_boxes(tile),
                "dets": det,
                "ae_score": float(score),
                "ae_score_z": float(score_z),
                "ae_area_frac": float((amap > pixel_stats["threshold"]).mean()),
                "box_features": box_features(det, amap, pixel_stats, box_topk),
            })
    return records


def fit_logistic(x: np.ndarray, y: np.ndarray, groups: np.ndarray, features: list[str],
                 folds: int, c_value: float, seed: int) -> tuple[LogisticFusion, np.ndarray]:
    """Ajuste la fusion et renvoie aussi les probabilites hors echantillon."""

    def make():
        return LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000,
                                  random_state=seed)

    oof = np.zeros(len(y))
    n_folds = min(folds, len(np.unique(groups)))
    if n_folds >= 2:
        for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(x, y, groups):
            if len(np.unique(y[train_idx])) < 2:
                oof[test_idx] = y[train_idx].mean()
                continue
            scaler = StandardScaler().fit(x[train_idx])
            model = make().fit(scaler.transform(x[train_idx]), y[train_idx])
            oof[test_idx] = model.predict_proba(scaler.transform(x[test_idx]))[:, 1]
    scaler = StandardScaler().fit(x)
    model = make().fit(scaler.transform(x), y)
    if n_folds < 2:
        oof = model.predict_proba(scaler.transform(x))[:, 1]
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    fusion = LogisticFusion(features=list(features), mean=scaler.mean_.tolist(),
                            scale=scale.tolist(), coef=model.coef_[0].tolist(),
                            intercept=float(model.intercept_[0]))
    return fusion, oof


def calibrate(params: dict, paths: Paths) -> dict:
    cfg, iou = params["hybrid"], float(params["evaluation"]["iou_threshold"])
    seed = int(params["seed"])
    model_dir = paths.model_dir("hybrid")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    ensure_dir(model_dir)

    pipeline = CrackVisionPipeline.from_dir(paths.models, select_device(), with_hybrid=False)
    pipeline.det_conf_floor = float(cfg["det_conf_floor"])
    calibration = pipeline.anomaly.calibration
    pixel_stats = {"mean": calibration["pixel_mean"], "std": calibration["pixel_std"],
                   "threshold": calibration["pixel_threshold"]}
    records = collect(pipeline, paths, "val", pixel_stats, float(cfg["box_evidence_topk"]))
    labels = np.array([r["label"] for r in records])
    groups = np.array([r["photo_id"] for r in records])
    per_image = [(r["dets"], r["gts"]) for r in records]

    # 1. Points de fonctionnement des modeles seuls
    det_max = np.array([float(r["dets"][:, 4].max()) if len(r["dets"]) else 0.0 for r in records])
    ae_scores = np.array([r["ae_score"] for r in records])
    yolox_box_t = best_box_threshold(per_image, iou)
    points = {
        "yolox": {"image_threshold": best_threshold(labels, det_max),
                  "box_threshold": yolox_box_t},
        "autoencoder": {"image_threshold": best_threshold(labels, ae_scores),
                        "pixel_threshold": pixel_stats["threshold"]},
    }

    # 2. Fusion au niveau de la tuile
    x_image = np.array([[dm, np.log1p((r["dets"][:, 4] >= yolox_box_t).sum()) if len(r["dets"])
                         else 0.0, r["ae_score_z"], r["ae_area_frac"]]
                        for dm, r in zip(det_max, records, strict=True)])
    image_model, image_oof = fit_logistic(x_image, labels, groups, IMAGE_FEATURES,
                                          int(cfg["cv_folds"]), float(cfg["logistic_c"]), seed)

    # 3. Fusion au niveau de la boite (VP / FP a IoU 0,5)
    box_rows, box_labels, box_groups, owners = [], [], [], []
    for index, record in enumerate(records):
        flags = match_detections(record["dets"], record["gts"], iou)
        for features, flag in zip(record["box_features"], flags, strict=True):
            box_rows.append(features)
            box_labels.append(int(flag))
            box_groups.append(record["photo_id"])
            owners.append(index)
    box_model, hybrid_box_t, box_note = None, yolox_box_t, "repli sur le score YOLOX"
    box_labels_np = np.array(box_labels)
    if len(box_rows) >= 20 and min(box_labels_np.sum(), (1 - box_labels_np).sum()) >= 5:
        box_model, box_oof = fit_logistic(np.array(box_rows), box_labels_np,
                                          np.array(box_groups), BOX_FEATURES,
                                          int(cfg["cv_folds"]), float(cfg["logistic_c"]), seed)
        rescored = [r["dets"].copy() for r in records]
        cursor = {i: 0 for i in range(len(records))}
        for owner, proba in zip(owners, box_oof, strict=True):
            rescored[owner][cursor[owner], 4] = proba
            cursor[owner] += 1
        hybrid_box_t = best_box_threshold(
            [(d, r["gts"]) for d, r in zip(rescored, records, strict=True)], iou)
        box_note = "regression logistique"
    points["hybrid"] = {"image_threshold": best_threshold(labels, image_oof),
                        "box_threshold": hybrid_box_t}

    unknown_key = str(float(cfg["unknown_quantile"]))
    unknown_threshold = calibration["pixel_quantiles"].get(
        unknown_key, max(calibration["pixel_quantiles"].values()))
    fusion = {
        "version": 1,
        "operating_points": points,
        "image_model": image_model.to_dict(),
        "box_model": box_model.to_dict() if box_model else None,
        "pixel_stats": pixel_stats,
        "params": {
            "det_conf_floor": float(cfg["det_conf_floor"]),
            "box_evidence_topk": float(cfg["box_evidence_topk"]),
            "unknown_threshold": float(unknown_threshold),
            "unknown_quantile": float(cfg["unknown_quantile"]),
            "unknown_min_area": int(cfg["unknown_min_area"]),
            "unknown_max_overlap": float(cfg["unknown_max_overlap"]),
        },
        "components": {
            "yolox_sha256": sha256_file(paths.model_dir("yolox") / "model.pth"),
            "autoencoder_sha256": sha256_file(paths.model_dir("autoencoder") / "model.pth"),
        },
    }
    write_json(model_dir / "fusion.json", fusion)

    report = {
        "n_val_tiles": len(records),
        "n_val_positive": int(labels.sum()),
        "box_fusion": box_note,
        "n_val_boxes": len(box_rows),
        "operating_points": points,
        # YOLOX et autoencodeur : seuils choisis sur ces memes tuiles (optimiste) ;
        # hybride : probabilites hors echantillon (validation croisee par photo).
        "val_image_metrics": {
            "yolox": image_metrics(labels, det_max, points["yolox"]["image_threshold"]),
            "autoencoder": image_metrics(labels, ae_scores, points["autoencoder"]["image_threshold"]),
            "hybrid": image_metrics(labels, image_oof, points["hybrid"]["image_threshold"]),
        },
        "val_box_metrics_yolox": box_prf(per_image, yolox_box_t, iou),
        "image_model": {name: round(coef, 4) for name, coef in
                        zip(IMAGE_FEATURES, image_model.coef, strict=True)},
        "box_model": ({name: round(coef, 4) for name, coef in
                       zip(BOX_FEATURES, box_model.coef, strict=True)} if box_model else None),
    }
    write_json(model_dir / "calibration_report.json", report)
    logger.info("Seuils : %s", points)
    return fusion
