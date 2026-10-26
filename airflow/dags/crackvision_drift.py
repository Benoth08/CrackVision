"""
===============================================================================
    CrackVision - DAG de surveillance de derive

    Chaque heure, interroge Prometheus sur les 24 dernieres heures de trafic :
      - score d'anomalie moyen des images (centre-reduit sur le beton sain de
        validation : proche de 0 si les images ressemblent a celles d'entrainement) ;
      - part d'images declarees defectueuses par le modele hybride ;
      - luminosite moyenne des images.
    Si un indicateur sort de sa plage (Variables Airflow), le DAG le signale et,
    si la Variable `crackvision_auto_retrain` vaut "true", declenche le DAG
    d'entrainement.
===============================================================================
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import requests
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import DAG, Variable

logger = logging.getLogger("airflow.task")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090/prometheus")

QUERIES = {
    "requests": 'sum(increase(crackvision_requests_total{endpoint="/predict",status="200"}[24h]))',
    # Score moyen centre-reduit sur la reference saine publiee par l'API
    "anomaly_z_mean": "(sum(increase(crackvision_anomaly_score_sum[24h])) / "
                      "sum(increase(crackvision_anomaly_score_count[24h])) - "
                      'scalar(max(crackvision_anomaly_reference{stat="mean"}))) / '
                      'scalar(max(crackvision_anomaly_reference{stat="std"}))',
    "defect_rate": 'sum(increase(crackvision_decisions_total{model="hybrid",level!="sain"}[24h])) / '
                   'sum(increase(crackvision_decisions_total{model="hybrid"}[24h]))',
    "brightness_mean": "sum(increase(crackvision_input_brightness_sum[24h])) / "
                       "sum(increase(crackvision_input_brightness_count[24h]))",
}


def query(expression: str) -> float | None:
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": expression},
                            timeout=30)
    response.raise_for_status()
    result = response.json()["data"]["result"]
    if not result:
        return None
    value = float(result[0]["value"][1])
    return value if value == value else None


def collect_indicators(**_) -> dict:
    indicators = {name: query(expression) for name, expression in QUERIES.items()}
    logger.info("Indicateurs sur 24 h : %s", indicators)
    return indicators


def decide(ti, **_) -> str:
    indicators = ti.xcom_pull(task_ids="collect_indicators")
    min_requests = float(Variable.get("crackvision_drift_min_requests", default="50"))
    if not indicators or (indicators["requests"] or 0) < min_requests:
        logger.info("Trafic insuffisant pour conclure (%s requetes)", indicators.get("requests"))
        return "no_drift"
    alerts = []
    z_max = float(Variable.get("crackvision_drift_z_max", default="2.0"))
    if indicators["anomaly_z_mean"] is not None and indicators["anomaly_z_mean"] > z_max:
        alerts.append(f"score d'anomalie moyen {indicators['anomaly_z_mean']:.2f} > {z_max}")
    low, high = (float(v) for v in Variable.get("crackvision_defect_rate_range",
                                                 default="0.05,0.80").split(","))
    rate = indicators["defect_rate"]
    if rate is not None and not low <= rate <= high:
        alerts.append(f"part d'images defectueuses {rate:.2%} hors de [{low:.0%}, {high:.0%}]")
    if not alerts:
        return "no_drift"
    logger.warning("Derive suspectee : %s", " ; ".join(alerts))
    auto = Variable.get("crackvision_auto_retrain", default="false").lower() == "true"
    return "trigger_training" if auto else "drift_alert"


with DAG(
    dag_id="crackvision_drift_monitoring",
    description="Derive des entrees et des decisions (Prometheus) -> alerte ou reentrainement",
    start_date=datetime(2026, 10, 1),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "crackvision", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["crackvision", "monitoring", "derive"],
) as dag:
    collect = PythonOperator(task_id="collect_indicators", python_callable=collect_indicators)
    branch = BranchPythonOperator(task_id="decide", python_callable=decide)
    no_drift = EmptyOperator(task_id="no_drift")
    alert = EmptyOperator(task_id="drift_alert")
    retrain = TriggerDagRunOperator(task_id="trigger_training", trigger_dag_id="crackvision_training",
                                    conf={"mode": "full"}, wait_for_completion=False)
    collect >> branch >> [no_drift, alert, retrain]
