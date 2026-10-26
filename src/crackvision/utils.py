"""Fonctions utilitaires partagees : journalisation, graines, fichiers, empreintes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("crackvision")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("CRACKVISION_LOG_LEVEL", "INFO"))
        root.propagate = False
    return logging.getLogger(name if name.startswith("crackvision") else f"crackvision.{name}")


def set_seed(seed: int) -> None:
    """Fixe les graines (Python, NumPy, PyTorch) pour la reproductibilite."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # l'image d'API n'a pas toujours besoin de torch ici
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Type non serialisable : {type(value)}")


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: Iterable[Path], base: Path) -> tuple[list[tuple[str, str]], str]:
    """Empreinte de chaque fichier (chemin relatif a `base`) et empreinte globale."""
    entries = []
    for path in sorted(paths):
        entries.append((path.relative_to(base).as_posix(), sha256_file(path)))
    global_digest = hashlib.sha256()
    for relative, digest in entries:
        global_digest.update(f"{digest}  {relative}\n".encode())
    return entries, global_digest.hexdigest()


def sha256_dir(directory: str | Path, pattern: str = "**/*") -> str:
    directory = Path(directory)
    files = [p for p in directory.glob(pattern) if p.is_file()]
    return sha256_files(files, directory)[1]


def select_device(preferred: str | None = None) -> str:
    """'cuda' si disponible (ou demande), sinon 'cpu'."""
    import torch

    preferred = preferred or os.environ.get("CRACKVISION_DEVICE")
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"
