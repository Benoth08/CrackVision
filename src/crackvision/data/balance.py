"""Etape `balance` : jeu d'entrainement insensible au desequilibre.

Deux desequilibres coexistent dans les tuiles d'entrainement :

1. entre classes : les tuiles saines sont majoritaires (les fissures couvrent
   environ 1,6 % des pixels des photos) ;
2. a l'interieur des tuiles fissurees : les petites fissures, les fissures
   obliques ou multiples sont plus rares que les autres.

On corrige le second par repeat factor sampling (Gupta et al., LVIS 2019) : une
tuile dont une caracteristique a une frequence f inferieure au seuil t est
repetee sqrt(t / f) fois en moyenne. On corrige ensuite le premier en tirant
autant de tuiles saines qu'il faut pour atteindre la part de tuiles fissurees
visee (50 % par defaut).

La validation et le blind test gardent la distribution naturelle : on mesure
ainsi les performances dans les conditions reelles d'emploi.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from crackvision.data import coco
from crackvision.data.prepare import read_tiles
from crackvision.paths import Paths
from crackvision.utils import ensure_dir, get_logger, write_json

logger = get_logger(__name__)

STRATA_COLUMNS = ("size_bin", "orientation", "multiplicity")


def add_strata(positives: pd.DataFrame, size_bins: list[float]) -> pd.DataFrame:
    positives = positives.copy()
    small, large = float(size_bins[0]), float(size_bins[1])
    positives["size_bin"] = np.select(
        [positives["largest_box_ratio"] < small, positives["largest_box_ratio"] > large],
        ["petite", "grande"], default="moyenne")
    positives["multiplicity"] = np.where(positives["n_boxes"] > 1, "multiple", "unique")
    return positives


def repeat_factors(positives: pd.DataFrame, threshold: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Facteurs de repetition entiers (arrondi aleatoire sans biais)."""
    factors = np.ones(len(positives))
    for column in STRATA_COLUMNS:
        frequency = positives[column].map(positives[column].value_counts(normalize=True))
        factors = np.maximum(factors, np.sqrt(threshold / frequency.to_numpy()))
    floor = np.floor(factors)
    return (floor + (rng.random(len(factors)) < factors - floor)).astype(int)


def _distribution(frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict:
    weights = np.ones(len(frame)) if weights is None else weights
    result = {}
    for column in STRATA_COLUMNS:
        totals = pd.Series(weights, index=frame.index).groupby(frame[column]).sum()
        result[column] = {k: round(float(v / weights.sum()), 4) for k, v in totals.items()}
    return result


def build_balanced(params: dict, paths: Paths) -> dict:
    cfg = params["balance"]
    rng = np.random.default_rng(int(params["seed"]))
    tiles = read_tiles(paths, "train")
    positives = add_strata(tiles[tiles["label"] == "positive"], cfg["size_bins"])
    healthy = tiles[tiles["label"] == "healthy"]
    if positives.empty or healthy.empty:
        raise RuntimeError("Il faut des tuiles fissurees ET saines pour l'entrainement")

    repeats = repeat_factors(positives, float(cfg["rfs_threshold"]), rng)
    n_positive = int(repeats.sum())
    fraction = float(cfg["positive_fraction"])
    n_healthy = int(round(n_positive * (1 - fraction) / fraction))
    replace = n_healthy > len(healthy)
    chosen = rng.choice(len(healthy), size=n_healthy, replace=replace)
    healthy_counts = np.bincount(chosen, minlength=len(healthy))
    healthy_kept = healthy[healthy_counts > 0]
    healthy_repeats = healthy_counts[healthy_counts > 0]

    out_dir = paths.train_balanced
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    balanced = pd.concat([positives.drop(columns=["size_bin", "multiplicity"]), healthy_kept])
    all_repeats = np.concatenate([repeats, healthy_repeats]).tolist()
    coco.write_coco(balanced, out_dir / "annotations" / "instances.json", all_repeats)

    # Autoencodeur : uniquement des tuiles saines (aucun reequilibrage necessaire).
    ae_tiles = healthy
    if int(cfg["ae_max_tiles"]) > 0 and len(ae_tiles) > int(cfg["ae_max_tiles"]):
        ae_tiles = ae_tiles.sample(n=int(cfg["ae_max_tiles"]), random_state=int(params["seed"]))
    ae_tiles[["tile_id", "photo_id", "file_name"]].to_csv(out_dir / "ae_train.csv", index=False)

    report = {
        "natural": {
            "n_positive": int(len(positives)),
            "n_healthy": int(len(healthy)),
            "positive_fraction": round(len(positives) / (len(positives) + len(healthy)), 4),
            "strata": _distribution(positives),
        },
        "balanced": {
            "n_positive_samples": n_positive,
            "n_healthy_samples": n_healthy,
            "healthy_with_replacement": bool(replace),
            "positive_fraction": round(n_positive / (n_positive + n_healthy), 4),
            "strata": _distribution(positives, repeats.astype(float)),
            "max_repeat": int(repeats.max()),
        },
        "autoencoder": {"n_healthy_tiles": int(len(ae_tiles))},
    }
    write_json(out_dir / "balance_report.json", report)
    logger.info("Jeu equilibre : %d echantillons fissures, %d sains (part fissuree %.2f)",
                n_positive, n_healthy, report["balanced"]["positive_fraction"])
    return report
