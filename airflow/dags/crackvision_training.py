"""
===============================================================================
    CrackVision - DAG d'entrainement

    Une tache par etape du pipeline DVC, executee dans l'image `trainer` par
    DockerOperator (le depot est monte dans /workspace) :

        ingest -> index -> split -> tile x3 -> balance -> data_report
                                              balance -> train_yolox -----+
                                              balance -> train_autoencoder +-> calibrate
        calibrate -> evaluate_val -> evaluate_blind -> publish -> reload_api
                                     evaluate_blind -> dvc_push

    Parametre `mode` :
      - full  : `dvc repro --single-item <etape>` (DVC saute les etapes a jour) ;
      - smoke : memes commandes sans DVC, profil configs/smoke.yaml, donnees
                synthetiques, dossier /workspace/.smoke-run (verification rapide).

    Les taches DVC partagent un pool d'une place : DVC verrouille le depot,
    deux `dvc repro` simultanes echoueraient.
===============================================================================
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG, Param
from docker.types import DeviceRequest, Mount

HOST_PROJECT_DIR = os.environ.get("CRACKVISION_HOST_PROJECT_DIR", "/opt/crackvision")
TRAINER_IMAGE = os.environ.get("CRACKVISION_TRAINER_IMAGE", "crackvision-trainer:latest")
NETWORK = os.environ.get("CRACKVISION_DOCKER_NETWORK", "crackvision")
USE_GPU = os.environ.get("CRACKVISION_TRAINER_GPU", "false").lower() == "true"
API_URL = os.environ.get("CRACKVISION_API_URL", "http://api:8000")
POOL = "crackvision_dvc"

ENVIRONMENT = {
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "MLFLOW_EXPERIMENT_NAME": "{{ 'crackvision-smoke' if params.mode == 'smoke' else 'crackvision' }}",
    "CRACKVISION_PROFILE": "{{ 'smoke' if params.mode == 'smoke' else 'full' }}",
    "CRACKVISION_WORKDIR": "{{ '/workspace/.smoke-run' if params.mode == 'smoke' else '' }}",
    "CRACKVISION_LOG_LEVEL": "INFO",
    "AIRFLOW_RUN_ID": "{{ run_id }}",
}


def stage_command(dvc_stage: str | None, cli: str) -> str:
    """Commande shell : etape DVC en mode full, appel direct en mode smoke."""
    smoke_part = f"python -m crackvision.cli {cli}"
    full_part = f"dvc repro --single-item {dvc_stage}" if dvc_stage else smoke_part
    return ("bash -c '"
            "if [ \"{{ params.mode }}\" = \"smoke\" ]; then " + smoke_part + "; "
            "else " + full_part + "; fi'")


def trainer_task(task_id: str, dvc_stage: str | None, cli: str, gpu: bool = False,
                 pool: str | None = POOL, timeout_hours: float = 2) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=TRAINER_IMAGE,
        command=stage_command(dvc_stage, cli),
        environment=ENVIRONMENT,
        mounts=[Mount(source=HOST_PROJECT_DIR, target="/workspace", type="bind")],
        working_dir="/workspace",
        network_mode=NETWORK,
        docker_url="unix:///var/run/docker.sock",
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])] if gpu and USE_GPU
        else None,
        shm_size=2 * 1024 ** 3,
        mount_tmp_dir=False,
        auto_remove="force",
        pool=pool,
        execution_timeout=timedelta(hours=timeout_hours),
    )


def reload_api(params: dict, **_) -> dict:
    """Demande a l'API de recharger le champion du registre (sauf en mode smoke)."""
    if params.get("mode") == "smoke":
        return {"status": "ignore", "raison": "mode smoke : modeles publies sous un autre nom"}
    response = requests.post(f"{API_URL}/admin/reload", timeout=120,
                             headers={"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "")})
    response.raise_for_status()
    return response.json()


with DAG(
    dag_id="crackvision_training",
    description="Donnees -> YOLOX + autoencodeur -> hybride -> evaluation -> registre MLflow",
    start_date=datetime(2026, 10, 1),
    schedule="0 2 * * 0",  # chaque dimanche a 2 h (DVC ne refait que ce qui a change)
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "crackvision", "retries": 1, "retry_delay": timedelta(minutes=2)},
    params={"mode": Param("full", enum=["full", "smoke"],
                          description="full : DVC sur les donnees reelles ; smoke : verification")},
    tags=["crackvision", "entrainement", "dvc", "mlflow"],
) as dag:
    ingest = trainer_task("ingest", "ingest", "ingest", timeout_hours=1)
    index = trainer_task("index", "index", "index")
    split = trainer_task("split", "split", "split")
    tiles = [trainer_task(f"tile_{name}", f"tile@{name}", f"tile --split {name}")
             for name in ("train", "val", "blind_test")]
    balance = trainer_task("balance", "balance", "balance")
    data_report = trainer_task("data_report", "data_report", "data-report")
    train_yolox = trainer_task("train_yolox", "train_yolox", "train-yolox", gpu=True,
                               timeout_hours=12)
    train_autoencoder = trainer_task("train_autoencoder", "train_autoencoder",
                                     "train-autoencoder", gpu=True, timeout_hours=6)
    calibrate = trainer_task("calibrate", "calibrate", "calibrate", gpu=True)
    evaluate_val = trainer_task("evaluate_val", "evaluate@val", "evaluate --split val", gpu=True)
    evaluate_blind = trainer_task("evaluate_blind", "evaluate@blind_test",
                                  "evaluate --split blind_test", gpu=True)
    publish = trainer_task("publish", None, "publish", pool=None)
    dvc_push = DockerOperator(
        task_id="dvc_push",
        image=TRAINER_IMAGE,
        command="bash -c 'if [ \"{{ params.mode }}\" = \"smoke\" ]; then echo \"mode smoke : "
                "rien a pousser\"; else dvc push; fi'",
        environment=ENVIRONMENT,
        mounts=[Mount(source=HOST_PROJECT_DIR, target="/workspace", type="bind")],
        working_dir="/workspace",
        network_mode=NETWORK,
        docker_url="unix:///var/run/docker.sock",
        mount_tmp_dir=False,
        auto_remove="force",
        pool=POOL,
    )
    reload = PythonOperator(task_id="reload_api", python_callable=reload_api)

    ingest >> index >> split >> tiles >> balance
    balance >> data_report
    balance >> [train_yolox, train_autoencoder] >> calibrate
    calibrate >> evaluate_val >> evaluate_blind >> [publish, dvc_push]
    publish >> reload
