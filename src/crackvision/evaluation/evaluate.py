"""Etapes `evaluate` (validation ou blind test) et `data_report`.

Les trois modeles sont evalues sur les memes tuiles, avec les seuils fixes a
l'etape `calibrate`. Pour le blind test, l'empreinte de chaque fichier est
d'abord verifiee : le jeu doit etre identique a celui cree par `tile`.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from crackvision.data.prepare import load_mask, read_tiles, tile_boxes, verify_manifest
from crackvision.evaluation import figures, stress
from crackvision.evaluation.metrics import (
    ap_from_matches,
    box_prf,
    cluster_bootstrap,
    coco_detection_metrics,
    curves,
    image_metrics,
    match_detections,
    pixel_metrics,
)
from crackvision.inference import MODEL_NAMES, CrackVisionPipeline
from crackvision.paths import SPLITS, Paths
from crackvision.utils import ensure_dir, get_logger, read_json, select_device, write_json

logger = get_logger(__name__)

CI_METRICS = ("auroc", "average_precision", "f1", "mcc", "balanced_accuracy")
METRIC_NAMES = {"f1": "F1", "mcc": "MCC", "balanced_accuracy": "Exactitude équilibrée",
                "auroc": "AUROC", "average_precision": "AP (précision-rappel)"}


def _read(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible : {path}")
    return image


def run_models(pipeline: CrackVisionPipeline, paths: Paths, split: str, pixel_size: int,
               batch_size: int = 32) -> tuple[list[dict], dict]:
    """Inference des trois modeles ; chronometre chaque modele separement."""
    tiles = read_tiles(paths, split)
    split_dir = paths.split_dir(split)
    records, timings = [], {"yolox": 0.0, "autoencoder": 0.0, "fusion": 0.0}
    for start in range(0, len(tiles), batch_size):
        batch = tiles.iloc[start:start + batch_size]
        images = [_read(split_dir / "images" / name) for name in batch["file_name"]]
        t0 = time.perf_counter()
        dets = pipeline.detector.predict(images, conf=pipeline.det_conf_floor,
                                         batch_size=batch_size)
        t1 = time.perf_counter()
        maps, scores = pipeline.anomaly.predict(images, batch_size=batch_size)
        scores_z = pipeline.anomaly.score_z(scores)
        t2 = time.perf_counter()
        hybrids = [pipeline.hybrid.predict(d, m, float(z))
                   for d, m, z in zip(dets, maps, scores_z, strict=True)]
        t3 = time.perf_counter()
        timings["yolox"] += t1 - t0
        timings["autoencoder"] += t2 - t1
        timings["fusion"] += t3 - t2
        for (_, tile), det, amap, score, hybrid in zip(batch.iterrows(), dets, maps, scores,
                                                       hybrids, strict=True):
            mask = cv2.imread(str(split_dir / "masks" / f"{tile['tile_id']}.png"),
                              cv2.IMREAD_GRAYSCALE) > 0
            records.append({
                "tile_id": tile["tile_id"],
                "photo_id": str(tile["photo_id"]),
                "label": int(tile["label"] == "positive"),
                "crack_pixels": int(tile["crack_pixels"]),
                "gts": tile_boxes(tile),
                "yolox": det,
                "ae_score": float(score),
                "hybrid": hybrid,
                "mask_small": cv2.resize(mask.astype(np.float32), (pixel_size, pixel_size),
                                         interpolation=cv2.INTER_AREA),
                "amap_small": cv2.resize(amap, (pixel_size, pixel_size),
                                         interpolation=cv2.INTER_AREA),
            })
    n = max(len(records), 1)
    latency = {
        "yolox_ms": 1000 * timings["yolox"] / n,
        "autoencoder_ms": 1000 * timings["autoencoder"] / n,
        "hybrid_ms": 1000 * (timings["yolox"] + timings["autoencoder"] + timings["fusion"]) / n,
        "device": pipeline.detector.device,
        "batch_size": batch_size,
    }
    return records, latency


def _ap50_bootstrap(per_tile: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray,
                    iou: float, n: int, seed: int) -> dict:
    flags = [match_detections(d, g, iou) for d, g in per_tile]
    scores = [d[:, 4] if len(d) else np.zeros(0) for d, _ in per_tile]
    n_gts = np.array([len(g) for _, g in per_tile])

    def statistic(indices):
        return ap_from_matches(np.concatenate([scores[i] for i in indices]),
                               np.concatenate([flags[i] for i in indices]),
                               int(n_gts[indices].sum()))

    point = statistic(np.arange(len(per_tile)))
    return {"value": point, **cluster_bootstrap(groups, statistic, n, seed)}


def _select_examples(records: list[dict], n: int, seed: int) -> list[int]:
    """Tuiles d'illustration : fissures de tailles variees, saines, cas difficiles."""
    rng = np.random.default_rng(seed)
    positives = sorted((r["crack_pixels"], i) for i, r in enumerate(records) if r["label"])
    healthy = [i for i, r in enumerate(records) if not r["label"]]
    chosen = []
    if positives:
        n_pos = max(1, n - 2)
        for q in np.linspace(0.2, 0.95, n_pos):
            chosen.append(positives[int(q * (len(positives) - 1))][1])
    if healthy:
        hardest = max(healthy, key=lambda i: records[i]["hybrid"]["proba"])
        chosen.append(hardest)
        chosen.append(int(rng.choice(healthy)))
    return list(dict.fromkeys(chosen))[:n]


def evaluate_split(params: dict, paths: Paths, split: str) -> dict:
    cfg = params["evaluation"]
    iou, seed, n_boot = float(cfg["iou_threshold"]), int(params["seed"]), int(cfg["bootstrap"])
    split_dir = paths.split_dir(split)
    digest = verify_manifest(split_dir)
    pipeline = CrackVisionPipeline.from_dir(paths.models, select_device())
    points = pipeline.hybrid.points
    records, latency = run_models(pipeline, paths, split, int(cfg["pixel_eval_size"]))
    labels = np.array([r["label"] for r in records])
    groups = np.array([r["photo_id"] for r in records])

    # --- detection (boites)
    gt = read_json(split_dir / "annotations" / "instances.json")
    id_of = {img["tile_id"]: img["id"] for img in gt["images"]}
    detection, detection_curves = {}, {}
    for model, key in (("yolox", "yolox"), ("hybrid", "boxes")):
        per_tile = [(r[key] if model == "yolox" else r["hybrid"][key], r["gts"]) for r in records]
        coco = coco_detection_metrics(gt, {id_of[r["tile_id"]]: d
                                           for r, (d, _) in zip(records, per_tile, strict=True)})
        detection_curves[model] = {**coco.pop("pr_curve"), "ap50": coco["ap50"]}
        coco["box_at_threshold"] = box_prf(per_tile, points[model]["box_threshold"], iou)
        coco["ap50_ci"] = _ap50_bootstrap(per_tile, groups, iou, min(n_boot, 1000), seed)
        detection[model] = coco

    # --- classification des tuiles
    scores = {
        "yolox": np.array([float(r["yolox"][:, 4].max()) if len(r["yolox"]) else 0.0
                           for r in records]),
        "autoencoder": np.array([r["ae_score"] for r in records]),
        "hybrid": np.array([r["hybrid"]["proba"] for r in records]),
    }
    image_level, ci_summary, image_curves = {}, {}, {}
    for model in MODEL_NAMES:
        threshold = points[model]["image_threshold"]
        result = image_metrics(labels, scores[model], threshold)
        result["ci95"] = {}
        ci_summary[model] = {}
        for metric in CI_METRICS:
            def statistic(indices, metric=metric, model=model, threshold=threshold):
                if len(np.unique(labels[indices])) < 2:
                    return None
                return image_metrics(labels[indices], scores[model][indices], threshold)[metric]
            interval = cluster_bootstrap(groups, statistic, n_boot, seed)
            result["ci95"][metric] = interval
            ci_summary[model][metric] = {"value": result[metric], **interval}
        image_level[model] = result
        image_curves[model] = curves(labels, scores[model])

    levels = pd.crosstab(pd.Series(labels, name="verite").map({0: "saine", 1: "fissuree"}),
                         pd.Series([r["hybrid"]["level"] for r in records], name="niveau"))
    pixel = pixel_metrics([r["mask_small"] for r in records], [r["amap_small"] for r in records],
                          int(cfg["pixel_eval_size"]))

    metrics = {
        "split": split,
        "manifest_digest": digest,
        "n_tiles": len(records),
        "n_positive": int(labels.sum()),
        "n_photos": int(len(np.unique(groups))),
        "operating_points": points,
        "detection": detection,
        "image": image_level,
        "pixel_autoencoder": pixel,
        "hybrid_levels": {k: {kk: int(vv) for kk, vv in v.items()}
                          for k, v in levels.to_dict(orient="index").items()},
        "unknown_regions": int(sum(len(r["hybrid"]["unknown_regions"]) for r in records)),
        "latency": latency,
        "bootstrap": {"resamples": n_boot, "unit": "photo", "seed": seed},
    }
    stress_examples = []
    if split == "blind_test":
        # Defauts hors classe incrustes dans des tuiles saines (voir stress.py)
        healthy = [i for i, r in enumerate(records) if not r["label"]]
        rng = np.random.default_rng(seed)
        picked = rng.choice(healthy, size=min(int(cfg["stress_tiles"]), len(healthy)),
                            replace=False)
        images = [_read(split_dir / "images" / f"{records[i]['tile_id']}.jpg") for i in picked]
        metrics["stress_test_hors_classe"], stress_examples = stress.run(pipeline, images, seed)
    out_dir = ensure_dir(paths.reports / split)
    write_json(out_dir / "metrics.json", metrics)
    pd.DataFrame([{
        "tile_id": r["tile_id"], "photo_id": r["photo_id"], "label": r["label"],
        "yolox_score": scores["yolox"][i], "ae_score": scores["autoencoder"][i],
        "hybrid_proba": scores["hybrid"][i], "hybrid_level": r["hybrid"]["level"],
        "n_gt_boxes": len(r["gts"]), "n_yolox_boxes": int(
            (r["yolox"][:, 4] >= points["yolox"]["box_threshold"]).sum()) if len(r["yolox"]) else 0,
    } for i, r in enumerate(records)]).to_csv(out_dir / "predictions.csv", index=False)
    plot_dir = ensure_dir(out_dir / "plots")
    for model in MODEL_NAMES:
        pd.DataFrame(image_curves[model]["pr"]).to_csv(plot_dir / f"pr_{model}.csv", index=False)
        pd.DataFrame(image_curves[model]["roc"]).to_csv(plot_dir / f"roc_{model}.csv", index=False)

    _figures(params, paths, split, pipeline, records, metrics, image_curves, detection_curves,
             ci_summary)
    if stress_examples:
        calibration = pipeline.anomaly.calibration
        heat_range = (calibration["pixel_quantiles"]["0.99"],
                      calibration["pixel_quantiles"]["0.999"] * 1.6)
        fig_dir = paths.reports / split / "figures"
        figures.plot_stress_examples(stress_examples, stress.DEFECTS, points, heat_range,
                                     fig_dir / "stress_examples.png",
                                     "Défauts jamais annotés, incrustés dans des tuiles saines")
        figures.plot_stress_rates(metrics["stress_test_hors_classe"], stress.DEFECTS,
                                  fig_dir / "stress_rates.png")
    logger.info("%s : AP50 YOLOX %.3f / hybride %.3f ; F1 tuile YOLOX %.3f / AE %.3f / hybride %.3f",
                split, detection["yolox"]["ap50"], detection["hybrid"]["ap50"],
                image_level["yolox"]["f1"], image_level["autoencoder"]["f1"],
                image_level["hybrid"]["f1"])
    return metrics


def _figures(params, paths, split, pipeline, records, metrics, image_curves, detection_curves,
             ci_summary) -> None:
    fig_dir = ensure_dir(paths.reports / split / "figures")
    name = "blind test" if split == "blind_test" else "validation"
    for model in MODEL_NAMES:
        image_curves[model]["summary"] = metrics["image"][model]["average_precision"]
    figures.plot_curves(image_curves, "pr", fig_dir / "image_pr_curves.png",
                        f"Tuiles fissurées et saines - {name}", "AP")
    for model in MODEL_NAMES:
        image_curves[model]["summary"] = metrics["image"][model]["auroc"]
    figures.plot_curves(image_curves, "roc", fig_dir / "image_roc_curves.png",
                        f"Courbes ROC - {name}", "AUROC")
    figures.plot_detection_pr(detection_curves, fig_dir / "detection_pr_curves.png",
                              f"Détection des fissures (boîtes) - {name}")
    figures.plot_metric_bars(ci_summary, METRIC_NAMES, fig_dir / "image_metrics_ci.png",
                             f"Classification des tuiles - {name} (IC 95 % par photo)")
    figures.plot_confusions({m: metrics["image"][m]["confusion"] for m in MODEL_NAMES},
                            fig_dir / "confusion_matrices.png",
                            f"Matrices de confusion aux seuils de la validation - {name}")

    calibration = pipeline.anomaly.calibration
    # Affichage : transparent jusqu'au quantile 99 % des pixels sains, opaque a 1,6 x q99,9
    heat_range = (calibration["pixel_quantiles"]["0.99"],
                  calibration["pixel_quantiles"]["0.999"] * 1.6)
    split_dir = paths.split_dir(split)
    examples = []
    for index in _select_examples(records, int(params["evaluation"]["n_examples"]),
                                  int(params["seed"])):
        record = records[index]
        image = _read(split_dir / "images" / f"{record['tile_id']}.jpg")
        maps, _ = pipeline.anomaly.predict([image])
        mask = cv2.imread(str(split_dir / "masks" / f"{record['tile_id']}.png"),
                          cv2.IMREAD_GRAYSCALE) > 0
        examples.append({**record, "image": image, "amap": maps[0], "mask": mask,
                         "yolox_score": float(record["yolox"][:, 4].max())
                         if len(record["yolox"]) else 0.0})
    if examples:
        figures.plot_examples(examples, pipeline.hybrid.points, heat_range,
                              fig_dir / "examples.png", f"Exemples de tuiles - {name}")

    # Photo entiere : meme chaine que l'API (reduction, tuiles, recomposition).
    photos = pd.read_csv(paths.photos_index, dtype={"photo_id": str})
    split_photos = pd.read_csv(paths.photo_split, dtype={"photo_id": str})
    candidates = photos.merge(split_photos[split_photos["split"] == split][["photo_id"]],
                              on="photo_id")
    if len(candidates):
        photo = candidates.sort_values("crack_ratio").iloc[len(candidates) // 2]
        image = _read(paths.root / photo["image_path"])
        result = pipeline.predict_image(image)
        result["yolox_box_threshold"] = pipeline.hybrid.points["yolox"]["box_threshold"]
        figures.plot_full_photo(image, result, heat_range, fig_dir / "full_photo.png",
                                f"Photo {photo['photo_id']} ({name}) : inférence sur l'image entière")

    if split == "val":
        figures.plot_training(pd.read_csv(paths.model_dir("yolox") / "history.csv"),
                              pd.read_csv(paths.model_dir("autoencoder") / "history.csv"),
                              fig_dir / "training_curves.png")


def data_report(params: dict, paths: Paths) -> dict:
    """Etape `data_report` : figures et resume des jeux de donnees."""
    out_dir = ensure_dir(paths.reports / "data")
    counts, summaries = {}, {}
    for split in SPLITS:
        tiles = read_tiles(paths, split, include_ambiguous=True)
        counts[split] = tiles["label"].value_counts().to_dict()
        summaries[split] = read_json(paths.split_dir(split) / "summary.json")
    figures.plot_split_distribution(counts, out_dir / "split_distribution.png")
    balance = read_json(paths.train_balanced / "balance_report.json")
    figures.plot_balance(balance, out_dir / "balance.png")

    train = read_tiles(paths, "train")
    rng = np.random.default_rng(int(params["seed"]))
    picks = []
    positives = train[train["label"] == "positive"].sort_values("crack_ratio")
    for q in (0.1, 0.4, 0.7, 0.95):
        if len(positives):
            picks.append(positives.iloc[int(q * (len(positives) - 1))])
    healthy = train[train["label"] == "healthy"]
    picks += [healthy.iloc[int(i)] for i in rng.choice(len(healthy), size=min(4, len(healthy)),
                                                       replace=False)]
    samples = [{"image": _read(paths.split_dir("train") / "images" / row["file_name"]),
                "gts": tile_boxes(row), "tile_id": row["tile_id"], "label": row["label"]}
               for row in picks]
    figures.plot_data_samples(samples, out_dir / "samples.png",
                              "Tuiles d'entraînement : fissurées (boîtes de vérité) et saines")

    photos = pd.read_csv(paths.photos_index, dtype={"photo_id": str})
    photo = photos.sort_values("crack_ratio").iloc[len(photos) // 2]
    image = _read(paths.root / photo["image_path"])
    mask = load_mask(paths.root / photo["mask_path"], image.shape[:2],
                     {**params["prepare"], "scale": 1.0})
    overlay = image.copy()
    overlay[mask > 0] = (52, 104, 235)
    figures.plot_data_samples([
        {"image": image, "gts": np.zeros((0, 4)), "tile_id": f"photo {photo['photo_id']}",
         "label": "positive"},
        {"image": overlay, "gts": np.zeros((0, 4)), "tile_id": "masque de fissure",
         "label": "positive"},
    ], out_dir / "photo_and_mask.png", "Une photo du jeu et son masque d'annotation")

    summary = {"splits": summaries, "balance": balance,
               "photo_split": read_json(paths.splits_dir / "split_report.json")}
    write_json(out_dir / "summary.json", summary)
    return summary
