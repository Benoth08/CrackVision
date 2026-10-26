"""Annotations au format COCO (lu par YOLOX et par pycocotools)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crackvision.utils import write_json

CATEGORIES = [{"id": 1, "name": "fissure", "supercategory": "defaut"}]


def build_coco(tiles: pd.DataFrame, repeats: list[int] | None = None) -> dict:
    """Construit un dictionnaire COCO ; `repeats` duplique certaines tuiles.

    Une tuile repetee recoit un nouvel identifiant d'image mais garde son fichier :
    c'est ainsi que le jeu d'entrainement reequilibre sur-echantillonne les strates
    rares sans copier d'images.
    """
    if repeats is None:
        repeats = [1] * len(tiles)
    images, annotations = [], []
    image_id, annotation_id = 0, 0
    for (_, tile), count in zip(tiles.iterrows(), repeats, strict=True):
        boxes = json.loads(tile["boxes"])
        for _ in range(int(count)):
            image_id += 1
            images.append({
                "id": image_id,
                "file_name": tile["file_name"],
                "width": int(tile["size"]),
                "height": int(tile["size"]),
                "tile_id": tile["tile_id"],
                "photo_id": str(tile["photo_id"]),
                "label": tile["label"],
            })
            for x, y, w, h in boxes:
                annotation_id += 1
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                })
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def write_coco(tiles: pd.DataFrame, path: Path, repeats: list[int] | None = None) -> Path:
    return write_json(path, build_coco(tiles, repeats))
