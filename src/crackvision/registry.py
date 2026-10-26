"""Commande `publish` : publication dans MLflow et promotion des modeles.

- un run parent "pipeline" porte les parametres, les donnees et les rapports ;
- un run enfant par modele porte son historique, ses metriques (validation et
  blind test) et le modele lui-meme, au format pyfunc ;
- les trois modeles sont enregistres dans le registre. Le modele hybride
  embarque YOLOX et l'autoencodeur : c'est lui que l'API charge, ce qui garantit
  un trio coherent ;
- promotion : si le blind test passe les seuils de `promotion.gates` et fait au
  moins aussi bien que le champion actuel (sur le meme blind test), les trois
  versions recoivent l'alias "champion", sinon "challenger".
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from crackvision import __version__
from crackvision.config import project_root
from crackvision.paths import Paths
from crackvision.tracking import EXPERIMENT, flatten
from crackvision.utils import get_logger, read_json, write_json

logger = get_logger(__name__)

PIP_REQUIREMENTS = [
    "torch==2.5.1",
    "torchvision==0.20.1",
    "numpy==1.26.4",
    "opencv-contrib-python-headless==4.10.0.84",
    "pyyaml==6.0.2",
    "loguru==0.7.3",
    "psutil==6.1.0",
    "yolox @ https://github.com/Megvii-BaseDetection/YOLOX/archive/"
    "6ddff4824372906469a7fae2dc3206c7aa4bbaee.zip",
]


def _python_model_class():
    import mlflow

    class CrackVisionModel(mlflow.pyfunc.PythonModel):
        """Entree : DataFrame avec une colonne `image_path` ; sortie : une ligne par image."""

        def load_context(self, context):
            from crackvision.inference import CrackVisionPipeline

            self.pipeline = CrackVisionPipeline.from_dir(context.artifacts["bundle"])

        def predict(self, context, model_input, params=None):
            import cv2

            rows = []
            for path in model_input["image_path"]:
                result = self.pipeline.predict_image(cv2.imread(str(path), cv2.IMREAD_COLOR))
                hybrid = result.get("hybrid", {})
                rows.append({"image_path": path, "ae_score": result["ae_score"],
                             "n_yolox_boxes": int(len(result["yolox"])),
                             "hybrid_proba": hybrid.get("proba"),
                             "hybrid_level": hybrid.get("level")})
            return pd.DataFrame(rows)

    return CrackVisionModel


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root(),
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"


def key_metrics(blind: dict) -> dict:
    return {
        "yolox": blind["detection"]["yolox"]["ap50"],
        "autoencoder": blind["image"]["autoencoder"]["auroc"],
        "hybrid": blind["image"]["hybrid"]["f1"],
    }


def check_gates(blind: dict, gates: dict) -> dict:
    values = key_metrics(blind)
    return {
        "yolox_ap50": values["yolox"] >= float(gates["yolox_ap50"]),
        "autoencoder_auroc": values["autoencoder"] >= float(gates["autoencoder_auroc"]),
        "hybrid_f1": values["hybrid"] >= float(gates["hybrid_f1"]),
    }


def _metrics_for(model: str, split_metrics: dict, prefix: str) -> dict:
    out = {}
    image = split_metrics["image"][model]
    for key in ("auroc", "average_precision", "f1", "precision", "recall", "specificity",
                "balanced_accuracy", "mcc"):
        out[f"{prefix}.image.{key}"] = image[key]
    if model in split_metrics["detection"]:
        det = split_metrics["detection"][model]
        for key in ("ap50", "ap50_95", "ap75", "ar100"):
            out[f"{prefix}.detection.{key}"] = det[key]
    if model == "autoencoder":
        for key, value in split_metrics["pixel_autoencoder"].items():
            out[f"{prefix}.{key}"] = value
    return {k: float(v) for k, v in out.items() if v == v}


def publish(params: dict, paths: Paths) -> dict:
    import mlflow
    from mlflow import MlflowClient

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI doit etre defini pour publier")
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    val = read_json(paths.reports / "val" / "metrics.json")
    blind = read_json(paths.reports / "blind_test" / "metrics.json")
    names = params["promotion"]["registry"]
    commit = git_commit()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    python_model = _python_model_class()()
    versions: dict[str, str] = {}

    with mlflow.start_run(run_name=f"pipeline-{stamp}") as parent:
        mlflow.set_tags({"git_commit": commit, "crackvision_version": __version__,
                         "profile": params.get("profile", "full"),
                         "blind_test_digest": blind["manifest_digest"]})
        mlflow.log_params({k: str(v)[:500] for k, v in flatten(params).items()
                           if not k.startswith("yolox.pretrained_weights")})
        if (paths.reports / "data").exists():
            mlflow.log_artifacts(str(paths.reports / "data"), "data")
        for split in ("val", "blind_test"):
            mlflow.log_artifacts(str(paths.reports / split / "figures"), f"{split}/figures")
            mlflow.log_artifact(str(paths.reports / split / "metrics.json"), split)

        for model in ("yolox", "autoencoder", "hybrid"):
            with mlflow.start_run(run_name=model, nested=True) as child:
                mlflow.set_tags({"model": model, "git_commit": commit,
                                 "blind_test_digest": blind["manifest_digest"]})
                model_dir = paths.model_dir(model)
                live = model_dir / "mlflow_run.json"
                if live.exists():
                    mlflow.set_tag("training_run_id", read_json(live)["run_id"])
                history = model_dir / "history.csv"
                if history.exists() and not live.exists():
                    # Entrainement hors ligne (GPU distant) : on rejoue l'historique.
                    for row in pd.read_csv(history).to_dict(orient="records"):
                        step = int(row.pop("epoch"))
                        mlflow.log_metrics({f"train.{k}": float(v) for k, v in row.items()
                                            if pd.notna(v)}, step=step)
                meta_file = "fusion.json" if model == "hybrid" else "meta.json"
                meta = read_json(model_dir / meta_file)
                if model != "hybrid":
                    mlflow.log_params({f"meta.{k}": str(v)[:500] for k, v in flatten(meta).items()})
                mlflow.log_metrics({**_metrics_for(model, val, "val"),
                                    **_metrics_for(model, blind, "blind_test")})
                with tempfile.TemporaryDirectory() as tmp:
                    bundle = Path(tmp) / "bundle"
                    targets = ("yolox", "autoencoder", "hybrid") if model == "hybrid" else (model,)
                    for name in targets:
                        shutil.copytree(paths.model_dir(name), bundle / name,
                                        ignore=shutil.ignore_patterns("mlflow_run.json"))
                    info = mlflow.pyfunc.log_model(
                        name="model", python_model=python_model,
                        artifacts={"bundle": str(bundle)},
                        code_paths=[str(project_root() / "src" / "crackvision")],
                        pip_requirements=PIP_REQUIREMENTS)
                registered = mlflow.register_model(info.model_uri, names[model])
                versions[model] = registered.version
                client.set_model_version_tag(names[model], registered.version,
                                             "blind_test_digest", blind["manifest_digest"])
                client.set_model_version_tag(names[model], registered.version, "key_metric",
                                             str(key_metrics(blind)[model]))
                client.set_model_version_tag(names[model], registered.version, "run_id",
                                             child.info.run_id)

        gates = check_gates(blind, params["promotion"]["gates"])
        current = None
        try:
            current = client.get_model_version_by_alias(names["hybrid"], "champion")
        except Exception:
            current = None
        better = True
        if current is not None and current.tags.get("blind_test_digest") == blind["manifest_digest"]:
            better = key_metrics(blind)["hybrid"] >= float(current.tags.get("key_metric", "0"))
        promote = all(gates.values()) and better
        alias = "champion" if promote else "challenger"
        for model, version in versions.items():
            client.set_registered_model_alias(names[model], alias, version)
        decision = {
            "date": stamp,
            "git_commit": commit,
            "parent_run_id": parent.info.run_id,
            "versions": versions,
            "alias": alias,
            "gates": gates,
            "better_than_champion": better,
            "previous_champion": current.version if current is not None else None,
            "key_metrics": key_metrics(blind),
        }
        mlflow.set_tags({"promotion": alias})
        mlflow.log_dict(decision, "promotion.json")
    write_json(paths.reports / "registry.json", decision)
    logger.info("Versions %s publiees avec l'alias %s (seuils %s)", versions, alias, gates)
    return decision
