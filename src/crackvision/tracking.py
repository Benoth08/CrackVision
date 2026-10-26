"""Suivi MLflow optionnel pendant l'entrainement.

Si MLFLOW_TRACKING_URI n'est pas defini (entrainement sur un GPU distant hors
ligne, tests), les fonctions ne font rien : l'historique reste ecrit dans les
fichiers du modele et sera publie plus tard par `crackvision publish`.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

from crackvision.utils import get_logger, write_json

logger = get_logger(__name__)

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "crackvision")


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _flatten(prefix: str, payload: dict, out: dict) -> dict:
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten(name, value, out)
        else:
            out[name] = value
    return out


def flatten(payload: dict, prefix: str = "") -> dict:
    return _flatten(prefix, payload, {})


class Run:
    """Enveloppe minimale autour d'un run MLflow (ou d'un run fictif)."""

    def __init__(self, run=None):
        self._run = run

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if not self._run:
            return
        import mlflow

        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, int | float) and v == v}
        try:
            mlflow.log_metrics(clean, step=step)
        except Exception as error:  # le suivi ne doit jamais interrompre un entrainement
            logger.warning("MLflow indisponible (%s) : metriques non envoyees", error)


@contextlib.contextmanager
def start_run(name: str, params: dict, model_dir: Path, tags: dict | None = None) -> Iterator[Run]:
    """Ouvre un run MLflow si le suivi est actif et note son identifiant dans model_dir."""
    if not enabled():
        yield Run()
        return
    try:
        import mlflow

        mlflow.set_experiment(EXPERIMENT)
        run = mlflow.start_run(run_name=name, tags={"stage": name, **(tags or {})})
        mlflow.log_params({k: str(v)[:500] for k, v in flatten(params).items()})
    except Exception as error:
        logger.warning("MLflow indisponible (%s) : entrainement sans suivi en direct", error)
        yield Run()
        return
    write_json(model_dir / "mlflow_run.json", {"run_id": run.info.run_id,
                                               "experiment": EXPERIMENT})
    try:
        yield Run(run)
    finally:
        mlflow.end_run()
