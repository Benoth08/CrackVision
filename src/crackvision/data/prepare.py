"""Lecture des photos et des masques, boites englobantes et decoupage en tuiles.

Etapes `index` (statistiques par photo) et `tile` (tuiles d'un jeu).

Une tuile est :
- `positive` si elle contient au moins une composante de fissure assez grande
  pour devenir une boite ;
- `healthy` (saine) si aucun pixel de fissure n'est present dans la tuile ni dans
  une marge autour d'elle ;
- `ambiguous` sinon (fragment trop petit, fissure qui frole le bord) : ces tuiles
  sont exclues des jeux pour ne pas melanger les deux classes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from crackvision.data import coco
from crackvision.data.sources import source_dir
from crackvision.data.tiling import tile_grid
from crackvision.paths import Paths
from crackvision.utils import ensure_dir, get_logger, sha256_files, write_json

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_DIR_NAMES = {"bw", "mask", "masks", "gt", "groundtruth", "ground_truth", "label", "labels"}


# --------------------------------------------------------------------------- lecture
def discover_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    """Associe chaque photo a son masque par nom de fichier (sans extension)."""
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = [part.lower() for part in path.relative_to(root).parts]
        if path.name.startswith("._") or "__macosx" in parts:
            continue
        key = path.stem.lower()
        target = masks if any(part in MASK_DIR_NAMES for part in parts[:-1]) else images
        if key in target:
            raise ValueError(f"Nom en double : {path} et {target[key]}")
        target[key] = path
    unmatched = sorted(set(images) ^ set(masks))
    if unmatched:
        logger.warning("%d fichiers sans correspondance ignores : %s", len(unmatched),
                       unmatched[:10])
    return [(key, images[key], masks[key]) for key in sorted(set(images) & set(masks))]


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)


def load_image(path: Path, scale: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible : {path}")
    if scale != 1.0:
        size = scaled_size(image.shape[1], image.shape[0], scale)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def load_mask(path: Path, image_hw: tuple[int, int], cfg: dict) -> np.ndarray:
    """Masque binaire {0, 1} a l'echelle de travail.

    Les masques d'origine sont des JPEG : on binarise d'abord a pleine resolution,
    puis on reduit en gardant les pixels couverts a plus de `mask_downscale_threshold`
    (sinon les fissures fines disparaitraient a la reduction).
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Masque illisible : {path}")
    height, width = image_hw
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (mask > int(cfg["mask_threshold"])).astype(np.float32)
    scale = float(cfg["scale"])
    if scale != 1.0:
        binary = cv2.resize(binary, scaled_size(width, height, scale), interpolation=cv2.INTER_AREA)
        binary = binary >= float(cfg["mask_downscale_threshold"])
    return binary.astype(np.uint8)


def close_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def components(mask: np.ndarray, min_area: int) -> list[dict]:
    """Composantes connexes (8-voisinage) : boite [x, y, w, h], aire, orientation."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
        w, h = int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])
        roi = (labels[y:y + h, x:x + w] == index).astype(np.uint8)
        moments = cv2.moments(roi, binaryImage=True)
        angle = 0.5 * np.degrees(np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"]))
        found.append({"box": [x, y, w, h], "area": area, "angle": float(angle)})
    return found


def orientation_label(angle_deg: float) -> str:
    angle = abs(angle_deg)
    if angle <= 30:
        return "horizontale"
    if angle >= 60:
        return "verticale"
    return "oblique"


# --------------------------------------------------------------------------- index
def build_index(params: dict, paths: Paths) -> pd.DataFrame:
    """Etape `index` : une ligne par photo avec ses statistiques de fissures."""
    cfg = params["prepare"]
    root = source_dir(params, paths)
    rows = []
    for photo_id, image_path, mask_path in discover_pairs(root):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Photo illisible ignoree : %s", image_path)
            continue
        height, width = image.shape[:2]
        mask = load_mask(mask_path, (height, width), cfg)
        comps = components(close_mask(mask, int(cfg["closing_kernel"])),
                           int(cfg["min_component_area"]))
        rows.append({
            "photo_id": photo_id,
            "image_path": image_path.relative_to(paths.root).as_posix(),
            "mask_path": mask_path.relative_to(paths.root).as_posix(),
            "width": width,
            "height": height,
            "crack_ratio": float(mask.mean()),
            "n_components": len(comps),
            "layout": "paysage" if width >= height else "portrait",
            "mean_gray": float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()),
        })
    if not rows:
        raise RuntimeError(f"Aucune paire photo/masque trouvee sous {root}")
    frame = pd.DataFrame(rows)
    ensure_dir(paths.interim)
    frame.to_csv(paths.photos_index, index=False)
    logger.info("%d photos indexees (taux de fissure moyen %.2f %%)", len(frame),
                100 * frame["crack_ratio"].mean())
    return frame


# --------------------------------------------------------------------------- tuiles
def label_tile(boxes: list[dict], crack_pixels: int, margin_clean: bool) -> str:
    if boxes:
        return "positive"
    if crack_pixels == 0 and margin_clean:
        return "healthy"
    return "ambiguous"


def tile_split(params: dict, paths: Paths, split: str) -> pd.DataFrame:
    """Etape `tile` : decoupe les photos d'un jeu en tuiles et ecrit les annotations."""
    cfg = params["prepare"]
    tile_size, margin = int(cfg["tile_size"]), int(cfg["healthy_margin"])
    photos = pd.read_csv(paths.photos_index, dtype={"photo_id": str})
    assignment = pd.read_csv(paths.photo_split, dtype={"photo_id": str})
    photos = photos.merge(assignment[["photo_id", "split"]], on="photo_id")
    photos = photos[photos["split"] == split]

    out_dir = paths.split_dir(split)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    image_dir, mask_dir = ensure_dir(out_dir / "images"), ensure_dir(out_dir / "masks")

    rows = []
    for photo in photos.itertuples(index=False):
        image = load_image(paths.root / photo.image_path, float(cfg["scale"]))
        mask = load_mask(paths.root / photo.mask_path, (photo.height, photo.width), cfg)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        closed = close_mask(mask, int(cfg["closing_kernel"]))
        height, width = mask.shape
        if height < tile_size or width < tile_size:
            raise ValueError(f"Photo {photo.photo_id} plus petite qu'une tuile ({width}x{height})")
        for row_index, col_index, y0, x0 in tile_grid(height, width, tile_size):
            tile_id = f"{photo.photo_id}_r{row_index}_c{col_index}"
            tile_mask = mask[y0:y0 + tile_size, x0:x0 + tile_size]
            comps = components(closed[y0:y0 + tile_size, x0:x0 + tile_size],
                               int(cfg["min_component_area"]))
            crack_pixels = int(tile_mask.sum())
            window = mask[max(y0 - margin, 0):y0 + tile_size + margin,
                          max(x0 - margin, 0):x0 + tile_size + margin]
            label = label_tile(comps, crack_pixels, margin_clean=int(window.sum()) == 0)
            largest = max(comps, key=lambda c: c["box"][2] * c["box"][3]) if comps else None
            rows.append({
                "tile_id": tile_id,
                "photo_id": photo.photo_id,
                "split": split,
                "x0": x0,
                "y0": y0,
                "size": tile_size,
                "label": label,
                "crack_pixels": crack_pixels,
                "crack_ratio": crack_pixels / float(tile_size * tile_size),
                "n_boxes": len(comps),
                "largest_box_ratio": (largest["box"][2] * largest["box"][3] / tile_size ** 2
                                      if largest else 0.0),
                "orientation": orientation_label(largest["angle"]) if largest else "",
                "boxes": json.dumps([c["box"] for c in comps]),
                "file_name": f"{tile_id}.jpg",
            })
            if label == "ambiguous":
                continue
            cv2.imwrite(str(image_dir / f"{tile_id}.jpg"),
                        image[y0:y0 + tile_size, x0:x0 + tile_size],
                        [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg_quality"])])
            cv2.imwrite(str(mask_dir / f"{tile_id}.png"), tile_mask * 255)

    tiles = pd.DataFrame(rows)
    tiles.to_csv(out_dir / "tiles.csv", index=False)
    kept = tiles[tiles["label"] != "ambiguous"]
    coco.write_coco(kept, out_dir / "annotations" / "instances.json")

    # Empreinte de chaque fichier : permet de verifier qu'un jeu (le blind test
    # surtout) n'a pas ete modifie entre sa creation et son evaluation.
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    entries, digest = sha256_files(files, out_dir)
    with open(out_dir / "MANIFEST.sha256", "w", encoding="utf-8") as handle:
        handle.writelines(f"{sha}  {name}\n" for name, sha in entries)
    counts = tiles["label"].value_counts().to_dict()
    write_json(out_dir / "summary.json", {
        "split": split,
        "n_photos": int(photos.shape[0]),
        "n_tiles": int(tiles.shape[0]),
        "labels": {k: int(v) for k, v in counts.items()},
        "n_boxes": int(kept["n_boxes"].sum()),
        "digest": digest,
    })
    logger.info("Jeu %s : %d photos, %d tuiles %s", split, len(photos), len(tiles), counts)
    return tiles


def read_tiles(paths: Paths, split: str, include_ambiguous: bool = False) -> pd.DataFrame:
    tiles = pd.read_csv(paths.split_dir(split) / "tiles.csv", dtype={"photo_id": str},
                        keep_default_na=False)
    if not include_ambiguous:
        tiles = tiles[tiles["label"] != "ambiguous"].reset_index(drop=True)
    return tiles


def tile_boxes(tile_row) -> np.ndarray:
    """Boites d'une tuile au format [x1, y1, x2, y2]."""
    boxes = np.array(json.loads(tile_row["boxes"]), dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def verify_manifest(split_dir: Path) -> str:
    """Recalcule les empreintes d'un jeu et echoue si un fichier a change."""
    manifest = split_dir / "MANIFEST.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        expected[name] = sha
    files = [split_dir / name for name in expected]
    missing = [name for name, path in zip(expected, files, strict=True) if not path.exists()]
    if missing:
        raise RuntimeError(f"Fichiers manquants dans {split_dir} : {missing[:5]}")
    entries, digest = sha256_files(files, split_dir)
    changed = [name for name, sha in entries if expected[name] != sha]
    if changed:
        raise RuntimeError(f"Fichiers modifies dans {split_dir} : {changed[:5]}")
    return digest
