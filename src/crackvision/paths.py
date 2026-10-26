"""Arborescence des donnees, modeles et rapports.

Tous les chemins sont relatifs a un dossier de travail : la racine du depot par
defaut (pipeline DVC), ou CRACKVISION_WORKDIR (pipeline de fumee, tests).

    data/raw/<source>/            photos et masques d'origine (immuables)
    data/interim/photos.csv       index des photos et statistiques par photo
    data/processed/splits/        affectation des photos aux jeux
    data/processed/<jeu>/         tuiles, masques, annotations COCO d'un jeu
    data/processed/train_balanced/ jeu d'entrainement reequilibre
    data/external/                poids pre-entraines telecharges
    models/<modele>/              poids, metadonnees, historique
    reports/                      metriques et figures
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from crackvision.config import project_root

SPLITS = ("train", "val", "blind_test")


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def photos_index(self) -> Path:
        return self.interim / "photos.csv"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.processed / "splits"

    @property
    def photo_split(self) -> Path:
        return self.splits_dir / "photo_split.csv"

    def split_dir(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"Jeu inconnu : {split}")
        return self.processed / split

    @property
    def train_balanced(self) -> Path:
        return self.processed / "train_balanced"

    @property
    def external(self) -> Path:
        return self.root / "data" / "external"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def model_dir(self, name: str) -> Path:
        return self.models / name

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def get_paths(root: str | Path | None = None) -> Paths:
    if root is None:
        root = os.environ.get("CRACKVISION_WORKDIR") or project_root()
    return Paths(Path(root).resolve())
