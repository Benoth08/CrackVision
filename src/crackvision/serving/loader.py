"""Chargement des modeles servis : registre MLflow (alias) ou dossier local."""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from crackvision.inference import CrackVisionPipeline
from crackvision.serving.settings import Settings
from crackvision.utils import get_logger, read_json

logger = get_logger(__name__)


@dataclass
class LoadedModels:
    pipeline: CrackVisionPipeline
    source: str
    version: str
    bundle_dir: str
    details: dict = field(default_factory=dict)


def download_from_registry(settings: Settings) -> tuple[Path, str]:
    """Telecharge la version du modele hybride portant l'alias demande."""
    import mlflow
    from mlflow import MlflowClient

    version = MlflowClient().get_model_version_by_alias(settings.registry_name,
                                                        settings.registry_alias)
    target = Path(settings.cache_dir) / f"{settings.registry_name}-v{version.version}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    local = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{settings.registry_name}/{version.version}", dst_path=str(target)))
    bundles = list(local.rglob("bundle"))
    bundle = next((b for b in bundles if (b / "hybrid" / "fusion.json").exists()), None)
    if bundle is None:
        raise RuntimeError(f"Paquet de modeles introuvable dans {local}")
    return bundle, str(version.version)


def load_models(settings: Settings) -> LoadedModels:
    if settings.model_source == "mlflow":
        try:
            bundle, version = download_from_registry(settings)
            source = f"mlflow:{settings.registry_name}@{settings.registry_alias}"
        except Exception as error:
            if not settings.local_fallback:
                raise
            logger.warning("Registre MLflow indisponible (%s) : repli sur %s", error,
                           settings.models_dir)
            bundle, version, source = Path(settings.models_dir), "local", "local (repli)"
    else:
        bundle, version, source = Path(settings.models_dir), "local", "local"
    pipeline = CrackVisionPipeline.from_dir(bundle, settings.device)
    details = {
        "yolox": read_json(bundle / "yolox" / "meta.json"),
        "autoencoder": read_json(bundle / "autoencoder" / "meta.json"),
        "operating_points": pipeline.hybrid.points if pipeline.hybrid else None,
    }
    logger.info("Modeles charges depuis %s (version %s)", source, version)
    return LoadedModels(pipeline, source, version, str(bundle), details)


class ModelHolder:
    """Conserve les modeles servis ; le rechargement remplace l'ensemble d'un coup."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self.models: LoadedModels | None = None
        self.last_error: str | None = None

    def load(self) -> LoadedModels:
        models = load_models(self.settings)
        with self._lock:
            self.models = models
            self.last_error = None
        return models

    def try_load(self) -> None:
        try:
            self.load()
        except Exception as error:
            self.last_error = str(error)
            logger.error("Chargement des modeles impossible : %s", error)
