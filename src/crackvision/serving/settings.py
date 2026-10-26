"""Reglages du service, lus dans les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # "mlflow" : modele hybride du registre (alias ci-dessous), "local" : dossier MODELS_DIR
    model_source: str = field(default_factory=lambda: _env("MODEL_SOURCE", "mlflow"))
    models_dir: str = field(default_factory=lambda: _env("MODELS_DIR", "/app/models"))
    registry_name: str = field(default_factory=lambda: _env("REGISTRY_MODEL", "crackvision-hybrid"))
    registry_alias: str = field(default_factory=lambda: _env("REGISTRY_ALIAS", "champion"))
    cache_dir: str = field(default_factory=lambda: _env("MODEL_CACHE_DIR", "/tmp/crackvision-models"))
    # Repli sur MODELS_DIR si le registre est injoignable au demarrage
    local_fallback: bool = field(
        default_factory=lambda: _env("MODEL_LOCAL_FALLBACK", "true").lower() == "true")
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", ""))
    max_upload_mb: float = field(default_factory=lambda: float(_env("MAX_UPLOAD_MB", "25")))
    max_megapixels: float = field(default_factory=lambda: float(_env("MAX_MEGAPIXELS", "40")))
    device: str = field(default_factory=lambda: _env("CRACKVISION_DEVICE", "cpu"))
    torch_threads: int = field(default_factory=lambda: int(_env("TORCH_THREADS", "2")))


def get_settings() -> Settings:
    return Settings()
