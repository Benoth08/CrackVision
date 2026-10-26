"""Etape `split` : affectation des photos aux jeux train / val / blind_test.

Le decoupage se fait PAR PHOTO (et non par tuile) : les tuiles d'une meme photo
se ressemblent, les repartir entre deux jeux ferait fuir de l'information vers
la validation et le blind test.

Il est stratifie sur deux caracteristiques de chaque photo : le taux de pixels
de fissure (quantiles) et le nombre de fissures. Les strates sont parcourues
l'une apres l'autre et chaque photo va au jeu le plus en retard sur sa
proportion cible : chaque strate est ainsi repartie dans les memes proportions
que l'ensemble, a une photo pres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crackvision.paths import SPLITS, Paths
from crackvision.utils import ensure_dir, get_logger, write_json

logger = get_logger(__name__)


def assign_strata(photos: pd.DataFrame, ratio_bins: int, component_bins: list[int]) -> pd.Series:
    n_bins = max(1, min(int(ratio_bins), len(photos)))
    ranks = photos["crack_ratio"].rank(method="first")
    ratio_bin = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop").astype(int)
    edges = [edge + 0.5 for edge in component_bins]
    component_bin = np.digitize(photos["n_components"].to_numpy(), edges)
    return pd.Series([f"r{r}_c{c}" for r, c in zip(ratio_bin, component_bin, strict=True)],
                     index=photos.index)


def stratified_assignment(strata: pd.Series, fractions: dict[str, float], seed: int) -> pd.Series:
    total = sum(fractions.values())
    targets = {name: fractions[name] / total for name in SPLITS}
    rng = np.random.default_rng(seed)
    order: list = []
    for stratum in sorted(strata.unique()):
        members = strata.index[strata == stratum].to_numpy()
        order.extend(rng.permutation(members).tolist())
    counts = dict.fromkeys(SPLITS, 0)
    assignment = {}
    for position, index in enumerate(order, start=1):
        deficits = {name: targets[name] * position - counts[name] for name in SPLITS}
        chosen = max(SPLITS, key=lambda name: deficits[name])
        counts[chosen] += 1
        assignment[index] = chosen
    return pd.Series(assignment).reindex(strata.index)


def split_photos(params: dict, paths: Paths) -> pd.DataFrame:
    cfg = params["split"]
    photos = pd.read_csv(paths.photos_index, dtype={"photo_id": str})
    photos["stratum"] = assign_strata(photos, cfg["ratio_bins"], cfg["component_bins"])
    photos["split"] = stratified_assignment(photos["stratum"], cfg["fractions"],
                                            int(params["seed"]))
    ensure_dir(paths.splits_dir)
    photos[["photo_id", "split", "stratum", "crack_ratio", "n_components"]].to_csv(
        paths.photo_split, index=False)

    report = {"n_photos": int(len(photos)), "splits": {}}
    for name in SPLITS:
        subset = photos[photos["split"] == name]
        report["splits"][name] = {
            "n_photos": int(len(subset)),
            "fraction": round(len(subset) / len(photos), 4),
            "crack_ratio_mean": round(float(subset["crack_ratio"].mean()), 5),
            "n_components_mean": round(float(subset["n_components"].mean()), 3),
            "strata": {k: int(v) for k, v in subset["stratum"].value_counts().sort_index().items()},
        }
    write_json(paths.splits_dir / "split_report.json", report)
    logger.info("Photos par jeu : %s",
                {name: report["splits"][name]["n_photos"] for name in SPLITS})
    return photos
