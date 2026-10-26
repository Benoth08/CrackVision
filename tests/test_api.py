"""API FastAPI servie avec les modeles du pipeline de fumee."""

import importlib
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(smoke_workdir):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_SOURCE", "local")
    mp.setenv("MODELS_DIR", str(smoke_workdir / "models"))
    mp.setenv("ADMIN_TOKEN", "jeton-de-test")
    import crackvision.serving.app as app_module

    app_module = importlib.reload(app_module)
    with TestClient(app_module.app) as test_client:
        yield test_client
    mp.undo()


def _image_bytes(smoke_workdir) -> bytes:
    photo = next((smoke_workdir / "data" / "raw" / "synthetic" / "rgb").glob("*.jpg"))
    return photo.read_bytes()


def test_health_ready_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200
    info = client.get("/models").json()
    assert info["source"] == "local"
    assert set(info["operating_points"]) == {"yolox", "autoencoder", "hybrid"}


@pytest.mark.parametrize("model", ["yolox", "autoencoder", "hybrid"])
def test_predict_each_model(client, smoke_workdir, model):
    response = client.post(f"/predict?model={model}&pixel_per_mm=4&return_heatmap=true",
                           files={"file": ("photo.jpg", _image_bytes(smoke_workdir), "image/jpeg")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["modele"] == model
    assert body["decision"] in {"defaut", "sain"}
    assert body["image"]["largeur"] > 0
    for crack in body["fissures"]:
        assert "largeur_mm" in crack["mesures"]
    if model != "yolox":
        assert body["carte_anomalie_png"]


def test_predict_rejects_a_non_image(client):
    response = client.post("/predict", files={"file": ("x.txt", b"pas une image", "text/plain")})
    assert response.status_code == 415


def test_predict_small_png(client):
    image = np.full((90, 120, 3), 140, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    response = client.post("/predict?model=hybrid",
                           files={"file": ("petit.png", io.BytesIO(buffer.tobytes()), "image/png")})
    assert response.status_code == 200, response.text


def test_metrics_and_admin(client):
    text = client.get("/metrics").text
    assert "crackvision_requests_total" in text
    assert "crackvision_anomaly_score_bucket" in text
    assert 'crackvision_anomaly_reference{stat="std"}' in text
    assert client.post("/admin/reload").status_code == 403
    reloaded = client.post("/admin/reload", headers={"X-Admin-Token": "jeton-de-test"})
    assert reloaded.status_code == 200
