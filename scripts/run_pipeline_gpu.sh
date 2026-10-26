#!/usr/bin/env bash
# =============================================================================
# Pipeline complet sur une machine GPU sans Docker : tests, `dvc repro`, puis
# envoi des donnees et modeles vers le remote DVC.
#   bash scripts/gpu_setup.sh && bash scripts/run_pipeline_gpu.sh
# Sans MLFLOW_TRACKING_URI, l'entrainement est suivi hors ligne (fichiers
# history.csv) et publie ensuite dans MLflow par `crackvision publish`.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${VENV:-$ROOT/.venv}/bin/activate"

python -m pytest -q
dvc repro
dvc push
dvc metrics show
