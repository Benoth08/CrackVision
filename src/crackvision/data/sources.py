"""Etape `ingest` : acquisition des photos brutes.

- `ozgenel_seg` : telechargement de l'archive Mendeley, verification de son
  empreinte SHA-256 publiee, puis extraction (RAR5, via bsdtar, unar ou 7z) ;
- `synthetic` : generation de photos de test (voir synthetic.py).

Les fichiers bruts ne sont jamais modifies ensuite : toutes les etapes suivantes
lisent data/raw/<source> et ecrivent ailleurs.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import requests

from crackvision.data import synthetic
from crackvision.paths import Paths
from crackvision.utils import ensure_dir, get_logger, sha256_file, write_json

logger = get_logger(__name__)

OZGENEL_CITATION = (
    "Ozgenel, C. F. (2019). Concrete Crack Segmentation Dataset. Mendeley Data, V1. "
    "doi:10.17632/jwsn7tfbrp.1"
)


def download(url: str, destination: Path, expected_sha256: str, retries: int = 3) -> Path:
    """Telecharge `url` vers `destination` et verifie son empreinte."""
    if destination.exists() and sha256_file(destination) == expected_sha256:
        logger.info("Archive deja presente et verifiee : %s", destination)
        return destination
    ensure_dir(destination.parent)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            logger.info("Telechargement (essai %d/%d) : %s", attempt, retries, url)
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with open(partial, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
            break
        except requests.RequestException as error:
            logger.warning("Echec du telechargement : %s", error)
            if attempt == retries:
                raise
            time.sleep(10 * attempt)
    digest = sha256_file(partial)
    if digest != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Empreinte inattendue pour {url} : {digest} (attendu {expected_sha256})")
    partial.replace(destination)
    return destination


def extract_archive(archive: Path, destination: Path) -> Path:
    """Extrait une archive RAR/ZIP avec le premier outil disponible."""
    ensure_dir(destination)
    candidates = [
        ["bsdtar", "-xf", str(archive), "-C", str(destination)],
        ["unar", "-q", "-f", "-o", str(destination), str(archive)],
        ["7z", "x", "-y", f"-o{destination}", str(archive)],
    ]
    for command in candidates:
        if shutil.which(command[0]) is None:
            continue
        logger.info("Extraction avec %s", command[0])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return destination
        logger.warning("%s a echoue : %s", command[0], result.stderr.strip()[-500:])
    raise RuntimeError("Aucun outil d'extraction RAR disponible (bsdtar, unar ou 7z)")


def ingest_ozgenel(cfg: dict, paths: Paths) -> Path:
    target = paths.raw / "ozgenel_seg"
    archive = paths.external / "downloads" / cfg["archive_name"]
    download(cfg["url"], archive, cfg["sha256"])
    if target.exists():
        shutil.rmtree(target)
    extract_archive(archive, target / "extracted")
    n_files = sum(1 for p in (target / "extracted").rglob("*") if p.is_file())
    write_json(target / "SOURCE.json", {
        "source": "ozgenel_seg",
        "url": cfg["url"],
        "sha256": cfg["sha256"],
        "doi": cfg["doi"],
        "license": cfg["license"],
        "citation": OZGENEL_CITATION,
        "n_files": n_files,
    })
    if not cfg.get("keep_archive", False):
        archive.unlink(missing_ok=True)
    logger.info("%d fichiers extraits dans %s", n_files, target)
    return target


def ingest(params: dict, paths: Paths) -> Path:
    source = params["data"]["source"]
    if source == "ozgenel_seg":
        return ingest_ozgenel(params["data"]["ozgenel_seg"], paths)
    if source == "synthetic":
        target = paths.raw / "synthetic"
        if target.exists():
            shutil.rmtree(target)
        return synthetic.generate(params["data"]["synthetic"], target, int(params["seed"]))
    raise ValueError(f"Source de donnees inconnue : {source}")


def source_dir(params: dict, paths: Paths) -> Path:
    return paths.raw / params["data"]["source"]
