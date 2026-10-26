"""Decoupage en tuiles et recomposition, partages par l'entrainement et le service.

Utiliser exactement le meme decoupage a l'entrainement et en production evite un
ecart de pretraitement entre les deux (training/serving skew).
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def tile_positions(length: int, tile: int) -> list[int]:
    """Positions de depart couvrant [0, length) avec des tuiles de taille `tile`.

    Les tuiles sont reparties uniformement : le chevauchement eventuel est
    partage entre elles au lieu d'etre concentre sur la derniere.
    """
    if length <= tile:
        return [0]
    count = math.ceil(length / tile)
    return [int(round(v)) for v in np.linspace(0, length - tile, count)]


def tile_grid(height: int, width: int, tile: int) -> list[tuple[int, int, int, int]]:
    """Liste de (ligne, colonne, y0, x0)."""
    grid = []
    for row, y0 in enumerate(tile_positions(height, tile)):
        for col, x0 in enumerate(tile_positions(width, tile)):
            grid.append((row, col, y0, x0))
    return grid


def pad_to_min(image: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Complete l'image par reflexion si elle est plus petite qu'une tuile."""
    height, width = image.shape[:2]
    pad_h, pad_w = max(size - height, 0), max(size - width, 0)
    if pad_h == 0 and pad_w == 0:
        return image, (height, width)
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return padded, (height, width)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU entre deux ensembles de boites [x1, y1, x2, y2]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])
    inter = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(boxes_a[:, 2:4] - boxes_a[:, :2], axis=1)
    area_b = np.prod(boxes_b[:, 2:4] - boxes_b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _intersection_over_min(box_a: np.ndarray, box_b: np.ndarray) -> float:
    top_left = np.maximum(box_a[:2], box_b[:2])
    bottom_right = np.minimum(box_a[2:4], box_b[2:4])
    inter = float(np.prod(np.clip(bottom_right - top_left, 0, None)))
    smallest = min(np.prod(box_a[2:4] - box_a[:2]), np.prod(box_b[2:4] - box_b[:2]))
    return inter / max(float(smallest), 1e-9)


def nms(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Suppression des non-maxima ; detections : (N, 5) [x1, y1, x2, y2, score]."""
    if len(detections) == 0:
        return detections.reshape(0, 5)
    order = np.argsort(-detections[:, 4])
    keep = []
    while len(order):
        current = order[0]
        keep.append(current)
        if len(order) == 1:
            break
        ious = iou_matrix(detections[current:current + 1, :4], detections[order[1:], :4])[0]
        order = order[1:][ious < iou_threshold]
    return detections[keep]


def merge_tile_detections(detections: np.ndarray, iou_threshold: float = 0.5,
                          containment: float = 0.6) -> np.ndarray:
    """Fusionne les detections issues de tuiles qui se chevauchent.

    Apres une NMS classique, deux boites dont l'une est contenue a plus de
    `containment` dans l'autre (meme fissure vue par deux tuiles) sont reunies.
    """
    kept = nms(detections, iou_threshold)
    merged: list[np.ndarray] = []
    for det in kept:
        for index, other in enumerate(merged):
            if _intersection_over_min(det, other) >= containment:
                merged[index] = np.array([
                    min(det[0], other[0]), min(det[1], other[1]),
                    max(det[2], other[2]), max(det[3], other[3]),
                    max(det[4], other[4]),
                ], dtype=np.float32)
                break
        else:
            merged.append(det.astype(np.float32))
    return np.array(merged, dtype=np.float32).reshape(-1, 5)


def stitch_maps(maps: list[np.ndarray], positions: list[tuple[int, int]],
                shape: tuple[int, int]) -> np.ndarray:
    """Recompose une carte pleine image (maximum sur les zones de recouvrement)."""
    canvas = np.zeros(shape, dtype=np.float32)
    for amap, (y0, x0) in zip(maps, positions, strict=True):
        h = min(amap.shape[0], shape[0] - y0)
        w = min(amap.shape[1], shape[1] - x0)
        region = canvas[y0:y0 + h, x0:x0 + w]
        np.maximum(region, amap[:h, :w], out=region)
    return canvas
