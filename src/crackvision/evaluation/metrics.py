"""Metriques de detection, de classification d'image et de localisation au pixel.

Niveaux d'evaluation :
- boite : AP COCO (IoU 0,5 et 0,5:0,95), precision / rappel / F1 a un seuil ;
- tuile : AUROC, AP (aire sous la courbe precision-rappel), F1, MCC, exactitude
  equilibree... Les trois dernieres sont peu sensibles au desequilibre des classes ;
- pixel : AUROC et AP de la carte d'anomalie contre le masque de fissure.

Les intervalles de confiance sont obtenus par bootstrap groupe par photo : les
tuiles d'une meme photo ne sont pas independantes, on reechantillonne donc des
photos entieres.
"""

from __future__ import annotations

import contextlib
import io

import cv2
import numpy as np
from sklearn import metrics as skm

from crackvision.data.tiling import iou_matrix


# --------------------------------------------------------------------------- boites
def coco_detection_metrics(gt: dict, predictions: dict[int, np.ndarray]) -> dict:
    """AP COCO ; `predictions` associe un image_id a ses detections (N, 5)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    results = []
    for image_id, dets in predictions.items():
        for x1, y1, x2, y2, score in dets:
            results.append({"image_id": int(image_id), "category_id": 1,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score)})
    empty = {"ap50_95": 0.0, "ap50": 0.0, "ap75": 0.0, "ar100": 0.0,
             "pr_curve": {"recall": [], "precision": []}}
    if not results or not gt["annotations"]:
        return empty
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {k: gt[k] for k in ("images", "annotations", "categories")}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    precision = evaluator.eval["precision"][0, :, 0, 0, 2]
    recall = evaluator.params.recThrs
    valid = precision > -1
    return {
        "ap50_95": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "ar100": float(stats[8]),
        "pr_curve": {"recall": recall[valid].round(3).tolist(),
                     "precision": precision[valid].round(4).tolist()},
    }


def match_detections(dets: np.ndarray, gts: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Appariement glouton par score decroissant ; renvoie un drapeau VP par detection.

    Les detections sont renvoyees dans l'ordre d'entree.
    """
    flags = np.zeros(len(dets), dtype=bool)
    if len(dets) == 0 or len(gts) == 0:
        return flags
    ious = iou_matrix(dets[:, :4], gts[:, :4])
    used = np.zeros(len(gts), dtype=bool)
    for index in np.argsort(-dets[:, 4]):
        candidates = np.where(~used & (ious[index] >= iou_threshold))[0]
        if len(candidates):
            best = candidates[np.argmax(ious[index, candidates])]
            used[best] = True
            flags[index] = True
    return flags


def ap_from_matches(scores: np.ndarray, flags: np.ndarray, n_gt: int) -> float:
    """AP par interpolation sur 101 points de rappel (comme COCO)."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(flags[order])
    fp = np.cumsum(~flags[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    thresholds = np.linspace(0, 1, 101)
    indices = np.searchsorted(recall, thresholds, side="left")
    sampled = np.where(indices < len(precision), precision[np.minimum(indices, len(precision) - 1)], 0)
    return float(sampled.mean())


def box_prf(per_image: list[tuple[np.ndarray, np.ndarray]], threshold: float,
            iou_threshold: float) -> dict:
    tp = fp = fn = 0
    for dets, gts in per_image:
        kept = dets[dets[:, 4] >= threshold] if len(dets) else dets
        flags = match_detections(kept, gts, iou_threshold)
        tp += int(flags.sum())
        fp += int((~flags).sum())
        fn += int(len(gts) - flags.sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "threshold": float(threshold)}


def best_box_threshold(per_image: list[tuple[np.ndarray, np.ndarray]],
                       iou_threshold: float) -> float:
    """Seuil de score qui maximise le F1 au niveau des boites."""
    candidates = np.unique(np.concatenate([d[:, 4] for d, _ in per_image if len(d)] or [[0.5]]))
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    scores = [(box_prf(per_image, t, iou_threshold)["f1"], t) for t in candidates]
    return float(max(scores)[1])


# --------------------------------------------------------------------------- tuiles
def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Seuil qui maximise le F1 de la classe 'defaut'."""
    precision, recall, thresholds = skm.precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1))])


def image_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "auroc": float(skm.roc_auc_score(y_true, scores)) if both else float("nan"),
        "average_precision": float(skm.average_precision_score(y_true, scores)) if both else float("nan"),
        "f1": float(skm.f1_score(y_true, pred, zero_division=0)),
        "precision": float(skm.precision_score(y_true, pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, pred)) if both else float("nan"),
        "mcc": float(skm.matthews_corrcoef(y_true, pred)) if both else float("nan"),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def curves(y_true: np.ndarray, scores: np.ndarray, points: int = 200) -> dict:
    precision, recall, _ = skm.precision_recall_curve(y_true, scores)
    fpr, tpr, _ = skm.roc_curve(y_true, scores)

    def thin(*arrays):
        step = max(1, len(arrays[0]) // points)
        return [np.round(a[::step], 4).tolist() for a in arrays]

    precision, recall = thin(precision, recall)
    fpr, tpr = thin(fpr, tpr)
    return {"pr": {"precision": precision, "recall": recall}, "roc": {"fpr": fpr, "tpr": tpr}}


def cluster_bootstrap(groups: np.ndarray, statistic, n_resamples: int, seed: int) -> dict:
    """Intervalle a 95 % d'une statistique, en reechantillonnant des groupes entiers.

    `statistic(indices)` recoit les indices des elements tires (avec repetitions).
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in unique}
    values = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[g] for g in drawn])
        value = statistic(indices)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return {"low": float("nan"), "high": float("nan"), "n": 0}
    low, high = np.percentile(values, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "n": len(values)}


# --------------------------------------------------------------------------- pixels
def pixel_metrics(masks: list[np.ndarray], maps: list[np.ndarray], size: int) -> dict:
    """AUROC et AP au pixel, sur des cartes sous-echantillonnees a size x size."""
    labels, values = [], []
    for mask, amap in zip(masks, maps, strict=True):
        small_mask = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        labels.append((small_mask > 0.1).ravel())
        values.append(cv2.resize(amap, (size, size), interpolation=cv2.INTER_AREA).ravel())
    y = np.concatenate(labels).astype(int)
    v = np.concatenate(values)
    if len(np.unique(y)) < 2:
        return {"pixel_auroc": float("nan"), "pixel_ap": float("nan")}
    return {"pixel_auroc": float(skm.roc_auc_score(y, v)),
            "pixel_ap": float(skm.average_precision_score(y, v))}
