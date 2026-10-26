"""Fixtures partagees.

`smoke_workdir` execute une seule fois, pour toute la session, le pipeline complet
de fumee (donnees synthetiques, profil configs/smoke.yaml) dans un dossier
temporaire ; les tests d'API et de bout en bout reutilisent ses modeles.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crackvision.config import load_params
from crackvision.paths import get_paths


@pytest.fixture()
def smoke_params() -> dict:
    return load_params(profile="smoke")


@pytest.fixture()
def tmp_paths(tmp_path: Path):
    return get_paths(tmp_path)


@pytest.fixture(scope="session")
def smoke_workdir(tmp_path_factory) -> Path:
    from crackvision import cli

    workdir = tmp_path_factory.mktemp("smoke")
    previous = {key: os.environ.get(key) for key in
                ("CRACKVISION_WORKDIR", "CRACKVISION_PROFILE", "CRACKVISION_DEVICE",
                 "MLFLOW_TRACKING_URI")}
    os.environ.update({"CRACKVISION_WORKDIR": str(workdir), "CRACKVISION_PROFILE": "smoke",
                       "CRACKVISION_DEVICE": "cpu"})
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    try:
        cli.main(["run-all"])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return workdir
