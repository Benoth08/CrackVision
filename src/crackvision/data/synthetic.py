"""Photos synthetiques de beton fissure, pour les tests et le pipeline de fumee.

La texture imite un beton (bruit multi-echelle, pores sombres) et chaque fissure
est une marche aleatoire sombre d'epaisseur variable. Le masque est trace en meme
temps que la fissure : la verite terrain est donc exacte. Les masques sont ecrits
en JPEG, comme dans le jeu reel, pour exercer la binarisation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from crackvision.utils import ensure_dir, get_logger, write_json

logger = get_logger(__name__)


def concrete_texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(120, 170)
    texture = np.full((height, width), base, dtype=np.float32)
    for cell, amplitude in ((64, 10.0), (16, 6.0), (4, 4.0), (1, 3.0)):
        small = rng.normal(0.0, 1.0, (max(height // cell, 1), max(width // cell, 1)))
        texture += amplitude * cv2.resize(small.astype(np.float32), (width, height),
                                          interpolation=cv2.INTER_CUBIC)
    image = np.stack([texture * tint for tint in rng.uniform(0.95, 1.05, 3)], axis=-1)
    n_pores = int(height * width * 0.0004)
    for _ in range(n_pores):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(1, 4))
        cv2.circle(image, center, radius, (base - 50.0,) * 3, -1, lineType=cv2.LINE_AA)
    return np.clip(image, 0, 255).astype(np.uint8)


def draw_crack(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = mask.shape
    point = np.array([rng.uniform(0.1, 0.9) * width, rng.uniform(0.1, 0.9) * height])
    angle = rng.uniform(0, np.pi)
    n_steps = int(rng.integers(30, 90))
    step = rng.uniform(4.0, 8.0)
    points = [point.copy()]
    for _ in range(n_steps):
        angle += rng.normal(0.0, 0.25)
        point = point + step * np.array([np.cos(angle), np.sin(angle)])
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            break
        points.append(point.copy())
    if len(points) < 3:
        return
    polyline = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    thickness = int(rng.integers(2, 5))
    darkness = float(rng.uniform(60, 100))
    color = tuple(float(max(c - darkness, 0)) for c in image[int(points[0][1]), int(points[0][0])])
    cv2.polylines(image, [polyline], False, color, thickness, lineType=cv2.LINE_AA)
    cv2.polylines(mask, [polyline], False, 255, thickness, lineType=cv2.LINE_8)


def generate(cfg: dict, output_dir: Path, seed: int) -> Path:
    """Ecrit `rgb/NNN.jpg` et `BW/NNN.jpg` dans output_dir."""
    rng = np.random.default_rng(seed)
    rgb_dir, mask_dir = ensure_dir(output_dir / "rgb"), ensure_dir(output_dir / "BW")
    width, height = int(cfg["width"]), int(cfg["height"])
    for index in range(int(cfg["n_photos"])):
        image = concrete_texture(height, width, rng)
        mask = np.zeros((height, width), dtype=np.uint8)
        for _ in range(int(rng.integers(1, int(cfg["max_cracks"]) + 1))):
            draw_crack(image, mask, rng)
        name = f"{index + 1:03d}.jpg"
        cv2.imwrite(str(rgb_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(mask_dir / name), mask, [cv2.IMWRITE_JPEG_QUALITY, 92])
    write_json(output_dir / "SOURCE.json", {
        "source": "synthetic",
        "n_photos": int(cfg["n_photos"]),
        "seed": seed,
        "note": "Photos generees, pour les tests uniquement",
    })
    logger.info("%d photos synthetiques ecrites dans %s", int(cfg["n_photos"]), output_dir)
    return output_dir
