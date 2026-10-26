"""Chargement des parametres du pipeline.

Les parametres viennent de `params.yaml` (suivi par DVC). Un profil optionnel
(`configs/<profil>.yaml`, choisi par CRACKVISION_PROFILE) surcharge ces valeurs,
par exemple pour le pipeline de fumee sur CPU.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Racine du depot (contient params.yaml, configs/, dvc.yaml)."""
    return Path(os.environ.get("CRACKVISION_PROJECT_ROOT", _PACKAGE_ROOT)).resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusion recursive : les cles de `override` remplacent celles de `base`."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_params(path: str | Path | None = None, profile: str | None = None) -> dict[str, Any]:
    """Charge params.yaml puis applique le profil demande (par defaut : aucun)."""
    root = project_root()
    params_path = Path(path or os.environ.get("CRACKVISION_PARAMS", root / "params.yaml"))
    with open(params_path, encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    profile = profile or os.environ.get("CRACKVISION_PROFILE") or "full"
    if profile != "full":
        profile_path = root / "configs" / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profil inconnu : {profile_path}")
        with open(profile_path, encoding="utf-8") as handle:
            params = deep_merge(params, yaml.safe_load(handle) or {})
    params["profile"] = profile
    return params
