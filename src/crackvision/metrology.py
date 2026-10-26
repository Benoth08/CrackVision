"""Metrologie des fissures detectees.

Deux niveaux de mesure :
- boite englobante (les trois modeles) : largeur, hauteur, surface, diagonale.
  C'est une estimation enveloppe : la boite surestime la fissure, surtout si elle
  est oblique ;
- pseudo-masque (modele hybride) : dans une boite confirmee, les pixels dont
  l'anomalie depasse le seuil sain forment un masque. Son squelette donne une
  longueur, et la transformee de distance une ouverture (largeur de la fissure).
  Ces deux grandeurs restent indicatives : la carte d'anomalie est lissee, elle
  elargit donc la fissure.

Toutes les mesures sont exprimees dans le repere de l'image d'origine (pixels),
et en millimetres si un facteur de calibration (pixels par mm) est fourni.
"""

from __future__ import annotations

import cv2
import numpy as np


def box_measurements(box: np.ndarray, pixel_per_mm: float | None = None) -> dict:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    width, height = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    result = {
        "x_centre_px": round((x1 + x2) / 2, 1),
        "y_centre_px": round((y1 + y2) / 2, 1),
        "largeur_px": round(width, 1),
        "hauteur_px": round(height, 1),
        "surface_px2": round(width * height, 1),
        "diagonale_px": round(float(np.hypot(width, height)), 1),
        "ratio_aspect": round(width / height, 2) if height > 0 else 0.0,
    }
    if pixel_per_mm:
        result.update({
            "largeur_mm": round(width / pixel_per_mm, 2),
            "hauteur_mm": round(height / pixel_per_mm, 2),
            "surface_mm2": round(width * height / pixel_per_mm ** 2, 2),
            "diagonale_mm": round(float(np.hypot(width, height)) / pixel_per_mm, 2),
        })
    return result


def skeleton_length(mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Longueur du squelette (pas de 1 ou racine de 2 selon la direction)."""
    skeleton = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0
    if not skeleton.any():
        return 0.0, skeleton
    horizontal = np.logical_and(skeleton[:, :-1], skeleton[:, 1:]).sum()
    vertical = np.logical_and(skeleton[:-1], skeleton[1:]).sum()
    diagonal = (np.logical_and(skeleton[:-1, :-1], skeleton[1:, 1:]).sum()
                + np.logical_and(skeleton[:-1, 1:], skeleton[1:, :-1]).sum())
    length = horizontal + vertical + np.sqrt(2) * diagonal
    return float(max(length, 1.0)), skeleton


def mask_measurements(mask: np.ndarray, scale: float = 1.0,
                      pixel_per_mm: float | None = None) -> dict:
    """Longueur et ouverture d'une fissure a partir de son masque.

    `scale` est le facteur de reduction applique avant l'inference : les mesures
    sont ramenees aux pixels de l'image d'origine.
    """
    if mask.sum() == 0:
        return {}
    length, skeleton = skeleton_length(mask)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    widths = 2.0 * distance[skeleton] if skeleton.any() else np.array([0.0])
    result = {
        "longueur_estimee_px": round(length / scale, 1),
        "ouverture_moyenne_px": round(float(widths.mean()) / scale, 2),
        "ouverture_max_px": round(float(widths.max()) / scale, 2),
    }
    if pixel_per_mm:
        result.update({
            "longueur_estimee_mm": round(result["longueur_estimee_px"] / pixel_per_mm, 2),
            "ouverture_moyenne_mm": round(result["ouverture_moyenne_px"] / pixel_per_mm, 3),
            "ouverture_max_mm": round(result["ouverture_max_px"] / pixel_per_mm, 3),
        })
    return result


def pseudo_mask(amap: np.ndarray, box_work: np.ndarray, threshold: float) -> np.ndarray:
    """Pixels anormaux dans une boite (coordonnees du repere de travail)."""
    height, width = amap.shape
    x1, y1 = int(max(np.floor(box_work[0]), 0)), int(max(np.floor(box_work[1]), 0))
    x2, y2 = int(min(np.ceil(box_work[2]), width)), int(min(np.ceil(box_work[3]), height))
    mask = np.zeros((max(y2 - y1, 1), max(x2 - x1, 1)), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask = (amap[y1:y2, x1:x2] > threshold).astype(np.uint8)
    return mask
