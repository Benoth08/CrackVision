"""Point d'entree en ligne de commande : une sous-commande par etape du pipeline.

    python -m crackvision.cli <etape> [--split JEU]

Les etapes sont appelees par DVC (dvc.yaml) et par Airflow. `run-all` enchaine
toutes les etapes sans DVC : c'est le pipeline de fumee des tests.
"""

from __future__ import annotations

import argparse
import time

from crackvision.config import load_params
from crackvision.paths import SPLITS, get_paths
from crackvision.utils import get_logger, set_seed

logger = get_logger(__name__)


def _ingest(params, paths, args):
    from crackvision.data.sources import ingest

    ingest(params, paths)


def _index(params, paths, args):
    from crackvision.data.prepare import build_index

    build_index(params, paths)


def _split(params, paths, args):
    from crackvision.data.split import split_photos

    split_photos(params, paths)


def _tile(params, paths, args):
    from crackvision.data.prepare import tile_split

    for split in ([args.split] if args.split else SPLITS):
        tile_split(params, paths, split)


def _balance(params, paths, args):
    from crackvision.data.balance import build_balanced

    build_balanced(params, paths)


def _data_report(params, paths, args):
    from crackvision.evaluation.evaluate import data_report

    data_report(params, paths)


def _train_yolox(params, paths, args):
    from crackvision.training.train_yolox import train_yolox

    train_yolox(params, paths)


def _train_autoencoder(params, paths, args):
    from crackvision.training.train_autoencoder import train_autoencoder

    train_autoencoder(params, paths)


def _calibrate(params, paths, args):
    from crackvision.training.fit_hybrid import calibrate

    calibrate(params, paths)


def _evaluate(params, paths, args):
    from crackvision.evaluation.evaluate import evaluate_split

    for split in ([args.split] if args.split else ("val", "blind_test")):
        evaluate_split(params, paths, split)


def _publish(params, paths, args):
    from crackvision.registry import publish

    publish(params, paths)


def _gallery(params, paths, args):
    from crackvision.evaluation.gallery import run_gallery

    run_gallery(params, paths, args.split or "blind_test")


def _verify_blind(params, paths, args):
    from crackvision.data.prepare import verify_manifest

    digest = verify_manifest(paths.split_dir("blind_test"))
    logger.info("Blind test intact, empreinte %s", digest)


STAGES = {
    "ingest": _ingest,
    "index": _index,
    "split": _split,
    "tile": _tile,
    "balance": _balance,
    "data-report": _data_report,
    "train-yolox": _train_yolox,
    "train-autoencoder": _train_autoencoder,
    "calibrate": _calibrate,
    "evaluate": _evaluate,
    "publish": _publish,
    "gallery": _gallery,
    "verify-blind": _verify_blind,
}
PIPELINE = ["ingest", "index", "split", "tile", "balance", "data-report", "train-yolox",
            "train-autoencoder", "calibrate", "evaluate"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crackvision", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=[*STAGES, "run-all"])
    parser.add_argument("--split", choices=SPLITS, default=None)
    parser.add_argument("--profile", default=None, help="profil de configs/ (ex. smoke)")
    args = parser.parse_args(argv)

    params = load_params(profile=args.profile)
    paths = get_paths()
    set_seed(int(params["seed"]))
    stages = PIPELINE if args.stage == "run-all" else [args.stage]
    for stage in stages:
        started = time.time()
        logger.info("=== %s (profil %s, dossier %s)", stage, params["profile"], paths.root)
        STAGES[stage](params, paths, args)
        logger.info("=== %s termine en %.1f s", stage, time.time() - started)


if __name__ == "__main__":
    main()
