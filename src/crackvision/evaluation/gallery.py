"""Commande `gallery` : predictions du modele hybride sur des photos entieres.

Les photos sont choisies par des criteres fixes (et non a la main), pour montrer
des cas varies, y compris des echecs :
- la fissure la plus etendue et la plus fine (taux de pixels de fissure) ;
- la photo qui compte le plus de fissures distinctes ;
- un cas median ;
- la photo ou le modele hybride se trompe sur le plus de tuiles ;
- une photo tiree au hasard (graine fixe).

Chaque photo suit la meme chaine que l'API (reduction, tuiles, recomposition).
Cette commande ne modifie aucun modele ni aucun seuil : elle ne fait que montrer.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from crackvision.data.prepare import load_mask
from crackvision.evaluation import figures
from crackvision.inference import CrackVisionPipeline
from crackvision.paths import Paths
from crackvision.utils import ensure_dir, get_logger, select_device, write_json

logger = get_logger(__name__)


def select_photos(paths: Paths, split: str, threshold: float, seed: int) -> list[tuple[str, str]]:
    photos = pd.read_csv(paths.photos_index, dtype={"photo_id": str})
    assignment = pd.read_csv(paths.photo_split, dtype={"photo_id": str})
    photos = photos.merge(assignment.loc[assignment["split"] == split, ["photo_id"]], on="photo_id")
    predictions = pd.read_csv(paths.reports / split / "predictions.csv", dtype={"photo_id": str})
    predictions["error"] = (predictions["hybrid_proba"] >= threshold).astype(int) != predictions["label"]
    errors = predictions.groupby("photo_id")["error"].sum()
    photos["errors"] = photos["photo_id"].map(errors).fillna(0).astype(int)

    by_ratio = photos.sort_values("crack_ratio")
    candidates = [
        (by_ratio.iloc[-1]["photo_id"], "fissure la plus étendue"),
        (by_ratio.iloc[0]["photo_id"], "fissure la plus fine"),
        (photos.sort_values(["n_components", "crack_ratio"]).iloc[-1]["photo_id"],
         "le plus de fissures distinctes"),
        (by_ratio.iloc[len(by_ratio) // 2]["photo_id"], "cas médian"),
        (photos.sort_values(["errors", "crack_ratio"]).iloc[-1]["photo_id"],
         "le plus d'erreurs du modèle hybride"),
    ]
    remaining = [p for p in photos["photo_id"] if p not in {c[0] for c in candidates}]
    if remaining:  # petit jeu : toutes les photos peuvent deja etre retenues
        rng = np.random.default_rng(seed)
        candidates.append((str(rng.choice(remaining)), "tirée au hasard"))
    selected, seen = [], set()
    for photo_id, reason in candidates:
        if photo_id not in seen:
            selected.append((str(photo_id), reason))
            seen.add(photo_id)
    return selected


def run_gallery(params: dict, paths: Paths, split: str = "blind_test") -> list[dict]:
    pipeline = CrackVisionPipeline.from_dir(paths.models, select_device())
    points = pipeline.hybrid.points
    photos = pd.read_csv(paths.photos_index, dtype={"photo_id": str}).set_index("photo_id")
    predictions = pd.read_csv(paths.reports / split / "predictions.csv", dtype={"photo_id": str})
    out_dir = ensure_dir(paths.reports / split / "gallery")
    calibration = pipeline.anomaly.calibration
    heat_range = (calibration["pixel_quantiles"]["0.99"], calibration["pixel_quantiles"]["0.999"] * 1.6)

    items, summary = [], []
    for photo_id, reason in select_photos(paths, split, points["hybrid"]["image_threshold"],
                                          int(params["seed"])):
        row = photos.loc[photo_id]
        image = cv2.imread(str(paths.root / row["image_path"]), cv2.IMREAD_COLOR)
        result = pipeline.predict_image(image)
        mask = load_mask(paths.root / row["mask_path"], image.shape[:2], params["prepare"])
        tiles = predictions[predictions["photo_id"] == photo_id]
        flagged = tiles["hybrid_proba"] >= points["hybrid"]["image_threshold"]
        item = {
            "photo_id": photo_id,
            "reason": reason,
            "image": image,
            "mask": mask,
            "result": result,
            "n_boxes": int(len(result["hybrid"]["boxes"])),
            "n_unknown": int(len(result["hybrid"]["unknown_regions"])),
            "crack_ratio": float(row["crack_ratio"]),
            "n_components": int(row["n_components"]),
            "tiles_fn": int(((~flagged) & (tiles["label"] == 1)).sum()),
            "tiles_fp": int((flagged & (tiles["label"] == 0)).sum()),
        }
        items.append(item)
        summary.append({k: v for k, v in item.items() if k not in ("image", "mask", "result")})
        logger.info("Photo %s (%s) : %d fissures, %d anomalies non classees", photo_id, reason,
                    item["n_boxes"], item["n_unknown"])

    half = (len(items) + 1) // 2
    name = split.replace("_", " ")
    for index, chunk in enumerate((items[:half], items[half:]), start=1):
        if chunk:
            title = f"Prédictions du modèle hybride sur des photos entières du {name} ({index}/2)"
            figures.plot_gallery(chunk, heat_range, out_dir / f"galerie_{index}.png", title)
    write_json(out_dir / "galerie.json", summary)
    return summary
