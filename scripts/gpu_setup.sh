#!/usr/bin/env bash
# =============================================================================
# Environnement d'entrainement sur une machine GPU sans Docker (pod cloud, poste
# de calcul). Memes versions que l'image Docker "trainer", PyTorch CUDA 12.4.
#   bash scripts/gpu_setup.sh            -> cree .venv a la racine du depot
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
YOLOX_COMMIT=6ddff4824372906469a7fae2dc3206c7aa4bbaee

# Extraction des archives RAR5 du jeu de donnees
if ! command -v bsdtar >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq libarchive-tools >/dev/null
fi

command -v uv >/dev/null 2>&1 || python3 -m pip install --quiet uv
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -r "$ROOT/requirements/torch-cuda.txt" -r "$ROOT/requirements/trainer.txt"
uv pip install setuptools==75.6.0 wheel==0.45.1 ninja==1.11.1.3
uv pip install --no-deps --no-build-isolation \
    "yolox @ https://github.com/Megvii-BaseDetection/YOLOX/archive/${YOLOX_COMMIT}.zip"
uv pip install --no-deps -e "$ROOT"
python -c "import torch, yolox; print('torch', torch.__version__, '- CUDA', torch.cuda.is_available())"
