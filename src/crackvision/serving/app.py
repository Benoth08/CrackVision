"""API FastAPI de CrackVision.

    GET  /health          vivacite du processus
    GET  /ready           503 tant que les modeles ne sont pas charges
    GET  /models          version servie, seuils de decision, pretraitement
    POST /predict         image -> fissures, metrologie, decision (yolox | autoencoder | hybrid)
    POST /admin/reload    recharge le champion du registre (jeton X-Admin-Token)
    GET  /metrics         metriques Prometheus (non expose par nginx)

Lancement : uvicorn crackvision.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from enum import Enum

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from crackvision import __version__
from crackvision.metrology import box_measurements, mask_measurements, pseudo_mask
from crackvision.models.hybrid import unknown_regions
from crackvision.serving import monitoring as m
from crackvision.serving.loader import ModelHolder
from crackvision.serving.settings import get_settings
from crackvision.utils import get_logger

logger = get_logger(__name__)


class ModelName(str, Enum):
    yolox = "yolox"
    autoencoder = "autoencoder"
    hybrid = "hybrid"


class Crack(BaseModel):
    id: int
    confiance: float
    boite: list[float] = Field(description="[x1, y1, x2, y2] en pixels de l'image d'origine")
    mesures: dict


class Region(BaseModel):
    boite: list[float]
    score: float
    surface_px2: float


class Prediction(BaseModel):
    modele: ModelName
    version: str
    source: str
    decision: str = Field(description="'defaut' ou 'sain'")
    niveau: str = Field(description="'fissure', 'anomalie_non_classee' ou 'sain'")
    score: float
    seuil: float
    score_anomalie: float | None = None
    image: dict
    fissures: list[Crack]
    anomalies_non_classees: list[Region]
    carte_anomalie_png: str | None = None
    latence_ms: float
    avertissement: str


DISCLAIMER = ("Dimensions issues des boites englobantes (estimation par exces) ; les mesures "
              "par pseudo-masque du modele hybride restent indicatives.")

settings = get_settings()
holder = ModelHolder(settings)


def _publish_model_info() -> None:
    m.MODEL_INFO.clear()
    if holder.models:
        m.MODEL_INFO.labels(settings.registry_name, holder.models.version,
                            holder.models.source).set(1)
        calibration = holder.models.pipeline.anomaly.calibration
        m.ANOMALY_REFERENCE.labels("mean").set(calibration.get("score_mean", 0.0))
        m.ANOMALY_REFERENCE.labels("std").set(max(calibration.get("score_std", 1.0), 1e-6))
    m.MODEL_READY.set(1 if holder.models else 0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    import torch

    torch.set_num_threads(settings.torch_threads)
    holder.try_load()
    _publish_model_info()
    yield


app = FastAPI(title="CrackVision API", version=__version__, lifespan=lifespan,
              description="Detection de fissures dans le beton : YOLOX, autoencodeur, hybride")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    if holder.models is None:
        raise HTTPException(503, detail=holder.last_error or "modeles non charges")
    return {"status": "ready", "version": holder.models.version, "source": holder.models.source}


@app.get("/models")
def models_info() -> dict:
    if holder.models is None:
        raise HTTPException(503, detail=holder.last_error or "modeles non charges")
    loaded = holder.models
    return {
        "source": loaded.source,
        "version": loaded.version,
        "prepare": loaded.pipeline.prepare,
        "operating_points": loaded.details["operating_points"],
        "yolox": {k: loaded.details["yolox"].get(k) for k in
                  ("variant", "input_size", "best_epoch", "val_metrics", "yolox_commit")},
        "autoencoder": {k: loaded.details["autoencoder"].get(k) for k in
                        ("image_size", "latent_channels", "best_epoch", "best_val_loss")},
    }


def _decode(payload: bytes) -> np.ndarray:
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, detail=f"Image trop lourde (> {settings.max_upload_mb} Mo)")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, detail="Fichier illisible : envoyer une image JPEG, PNG, BMP ou TIFF")
    if image.shape[0] * image.shape[1] / 1e6 > settings.max_megapixels:
        raise HTTPException(413, detail=f"Image trop grande (> {settings.max_megapixels} Mpx)")
    return image


def _heatmap_png(amap: np.ndarray, low: float, high: float) -> str:
    scaled = np.clip((amap - low) / max(high - low, 1e-9), 0, 1)
    ok, buffer = cv2.imencode(".png", (scaled * 255).astype(np.uint8))
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else ""


def _regions(raw: list[dict]) -> list[Region]:
    return [Region(boite=[round(float(v), 1) for v in r["box"]], score=round(r["score"], 4),
                   surface_px2=float(r["area"])) for r in raw]


def build_prediction(result: dict, model: ModelName, pixel_per_mm: float | None,
                     return_heatmap: bool) -> dict:
    pipeline = holder.models.pipeline
    points = pipeline.hybrid.points
    scale = result["scale"]
    cracks, regions = [], []
    score_anomalie = result.get("ae_score")
    if model is ModelName.yolox:
        dets = result["yolox"]
        score = float(dets[:, 4].max()) if len(dets) else 0.0
        threshold = points["yolox"]["image_threshold"]
        decision = score >= threshold
        level = "fissure" if decision else "sain"
        boxes = dets[dets[:, 4] >= points["yolox"]["box_threshold"]] if len(dets) else dets
    elif model is ModelName.autoencoder:
        score = float(result["ae_score"])
        threshold = points["autoencoder"]["image_threshold"]
        decision = score >= threshold
        level = "anomalie_non_classee" if decision else "sain"
        boxes = np.zeros((0, 5))
        raw = unknown_regions(result["anomaly_map"], points["autoencoder"]["pixel_threshold"],
                              int(pipeline.hybrid.params["unknown_min_area"]), boxes, 1.0)
        regions = [{**r, "box": [v / scale for v in r["box"]]} for r in raw]
    else:
        hybrid = result["hybrid"]
        score = float(hybrid["proba"])
        threshold = points["hybrid"]["image_threshold"]
        decision, level, boxes = hybrid["decision"], hybrid["level"], hybrid["boxes"]
        regions = hybrid["unknown_regions"]

    for index, box in enumerate(boxes, start=1):
        measures = box_measurements(box, pixel_per_mm)
        if model is ModelName.hybrid:
            mask = pseudo_mask(result["anomaly_map"], np.asarray(box[:4]) * scale,
                               pipeline.hybrid.pixel_stats["threshold"])
            measures.update(mask_measurements(mask, scale, pixel_per_mm))
        cracks.append(Crack(id=index, confiance=round(float(box[4]), 4),
                            boite=[round(float(v), 1) for v in box[:4]], mesures=measures))

    heatmap = None
    if return_heatmap and "anomaly_map" in result:
        calibration = pipeline.anomaly.calibration
        heatmap = _heatmap_png(result["anomaly_map"], calibration["pixel_quantiles"]["0.99"],
                               1.6 * calibration["pixel_quantiles"]["0.999"])
    return {
        "decision": "defaut" if decision else "sain",
        "niveau": level,
        "score": round(score, 4),
        "seuil": round(float(threshold), 4),
        "score_anomalie": round(float(score_anomalie), 5) if score_anomalie is not None else None,
        "image": {"largeur": result["image_size"][0], "hauteur": result["image_size"][1],
                  "tuiles": result["n_tiles"], "echelle": scale},
        "fissures": cracks,
        "anomalies_non_classees": _regions(regions),
        "carte_anomalie_png": heatmap,
    }


def _observe(model: ModelName, image: np.ndarray, result: dict, payload: dict) -> None:
    m.BRIGHTNESS.observe(float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()))
    m.MEGAPIXELS.observe(image.shape[0] * image.shape[1] / 1e6)
    m.DECISIONS.labels(model.value, payload["niveau"]).inc()
    m.DETECTIONS.labels(model.value).observe(len(payload["fissures"]))
    for crack in payload["fissures"]:
        m.CONFIDENCE.labels(model.value).observe(crack.confiance)
    if result.get("ae_score") is not None:
        m.ANOMALY_SCORE.observe(float(result["ae_score"]))
    if model is ModelName.hybrid:
        m.HYBRID_PROBA.observe(payload["score"])
    m.UNKNOWN_REGIONS.inc(len(payload["anomalies_non_classees"]))


@app.post("/predict", response_model=Prediction)
def predict(
    file: UploadFile = File(..., description="Photo de surface beton"),
    model: ModelName = Query(ModelName.hybrid),
    pixel_per_mm: float | None = Query(None, gt=0, description="Calibration (pixels par mm)"),
    return_heatmap: bool = Query(False, description="Joindre la carte d'anomalie (PNG base64)"),
) -> dict:
    started = time.perf_counter()
    status = "500"
    try:
        if holder.models is None:
            status = "503"
            raise HTTPException(503, detail="modeles non charges")
        image = _decode(file.file.read())
        inference_start = time.perf_counter()
        result = holder.models.pipeline.predict_image(image, mode=model.value)
        m.INFERENCE_LATENCY.labels(model.value).observe(time.perf_counter() - inference_start)
        payload = build_prediction(result, model, pixel_per_mm, return_heatmap)
        _observe(model, image, result, payload)
        status = "200"
        elapsed = time.perf_counter() - started
        m.REQUEST_LATENCY.labels(model.value).observe(elapsed)
        return {"modele": model, "version": holder.models.version, "source": holder.models.source,
                "latence_ms": round(1000 * elapsed, 1), "avertissement": DISCLAIMER, **payload}
    except HTTPException as error:
        status = str(error.status_code)
        raise
    except Exception as error:
        logger.exception("Erreur de prediction")
        raise HTTPException(500, detail="erreur interne de prediction") from error
    finally:
        m.REQUESTS.labels("/predict", model.value, status).inc()


@app.post("/admin/reload")
def reload_models(x_admin_token: str = Header(default="")) -> dict:
    if not settings.admin_token or x_admin_token != settings.admin_token:
        m.REQUESTS.labels("/admin/reload", "-", "403").inc()
        raise HTTPException(403, detail="jeton d'administration invalide")
    try:
        loaded = holder.load()
    except Exception as error:
        m.RELOADS.labels("echec").inc()
        raise HTTPException(502, detail=f"rechargement impossible : {error}") from error
    m.RELOADS.labels("succes").inc()
    _publish_model_info()
    m.REQUESTS.labels("/admin/reload", "-", "200").inc()
    return {"status": "recharge", "version": loaded.version, "source": loaded.source}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
