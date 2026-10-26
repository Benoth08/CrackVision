"""Test de resistance : defauts HORS CLASSE incrustes dans des tuiles saines.

Le jeu de donnees ne contient que des fissures. Pour mesurer ce que le non
supervise apporte face a un defaut que le detecteur n'a jamais vu (le role que
lui donne l'article de reference), on incruste dans des tuiles saines du blind
test quatre defauts synthetiques jamais annotes :

- tache d'humidite (zone assombrie aux bords flous) ;
- efflorescence (depot blanchatre) ;
- coulure de rouille (tache brun-orange) ;
- eclat de beton (zone rugueuse et sombre, au contour irregulier).

Ces defauts sont generes : les taux obtenus illustrent la complementarite des
modeles, ils ne mesurent pas une performance sur de vrais defauts.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFECTS = {
    "tache_humidite": "tache d'humidité",
    "efflorescence": "efflorescence",
    "rouille": "coulure de rouille",
    "eclat": "éclat de béton",
}


def blob_alpha(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Masque doux (0-1) forme de quelques ellipses, de 5 a 15 % de la tuile."""
    alpha = np.zeros((height, width), dtype=np.float32)
    center = rng.uniform(0.3, 0.7, 2) * [width, height]
    size = min(height, width)
    for _ in range(int(rng.integers(2, 5))):
        offset = rng.normal(0, size * 0.06, 2)
        axes = (int(rng.uniform(0.07, 0.16) * size), int(rng.uniform(0.05, 0.12) * size))
        cv2.ellipse(alpha, (int(center[0] + offset[0]), int(center[1] + offset[1])), axes,
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=size * 0.02)


def paste_defect(image: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Incruste un defaut ; renvoie l'image modifiee et le masque du defaut."""
    height, width = image.shape[:2]
    alpha = blob_alpha(height, width, rng)[..., None]
    base = image.astype(np.float32)
    noise = rng.normal(0, 1, (height, width, 1)).astype(np.float32)
    if kind == "tache_humidite":
        target = base * 0.62
    elif kind == "efflorescence":
        target = base * 0.35 + 0.65 * np.array([228, 232, 234], dtype=np.float32) + 6 * noise
    elif kind == "rouille":
        target = base * 0.45 + 0.55 * np.array([45, 95, 150], dtype=np.float32) + 5 * noise
    elif kind == "eclat":
        rough = cv2.GaussianBlur(noise, (0, 0), 1.5)[..., None] * 28
        target = base * 0.5 + rough + 10
        alpha = np.clip(alpha * 1.6, 0, 1)
    else:
        raise ValueError(f"Defaut inconnu : {kind}")
    out = np.clip(base * (1 - alpha) + target * alpha, 0, 255).astype(np.uint8)
    return out, alpha[..., 0] > 0.5


def _overlaps(regions: list[dict], mask: np.ndarray) -> bool:
    for region in regions:
        x1, y1, x2, y2 = (int(v) for v in region["box"])
        if mask[max(y1, 0):y2, max(x1, 0):x2].any():
            return True
    return False


def run(pipeline, images: list[np.ndarray], seed: int) -> tuple[dict, list[dict]]:
    """Taux de signalement par defaut et par modele, plus le taux de fausse alarme initial."""
    points = pipeline.hybrid.points
    rng = np.random.default_rng(seed)

    def decisions(predictions, masks=None) -> dict:
        yolox = [float(p.yolox[:, 4].max()) >= points["yolox"]["image_threshold"] if len(p.yolox)
                 else False for p in predictions]
        ae = [p.ae_score >= points["autoencoder"]["image_threshold"] for p in predictions]
        hybrid = [p.hybrid["decision"] for p in predictions]
        safety = [p.hybrid["decision"] or bool(p.hybrid["unknown_regions"]) for p in predictions]
        if masks is not None:
            safety = [p.hybrid["decision"] or _overlaps(p.hybrid["unknown_regions"], m)
                      for p, m in zip(predictions, masks, strict=True)]
        return {"yolox": float(np.mean(yolox)), "autoencoder": float(np.mean(ae)),
                "hybrid": float(np.mean(hybrid)), "hybrid_avec_filet": float(np.mean(safety))}

    baseline = decisions(pipeline.predict_tiles(images))
    results = {"n_tiles": len(images), "fausses_alarmes_sans_defaut": baseline, "defauts": {}}
    examples = []
    for kind in DEFECTS:
        altered, masks = zip(*(paste_defect(img, kind, rng) for img in images), strict=True)
        predictions = pipeline.predict_tiles(list(altered))
        results["defauts"][kind] = decisions(predictions, list(masks))
        examples.append({"kind": kind, "image": altered[0], "mask": masks[0],
                         "prediction": predictions[0]})
    return results, examples
