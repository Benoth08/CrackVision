"""Autoencodeur, YOLOX, fusion hybride et metrologie (sans entrainement)."""

import numpy as np
import pytest
import torch

from crackvision.metrology import box_measurements, mask_measurements, pseudo_mask
from crackvision.models.autoencoder import (
    ConvAutoencoder,
    anomaly_maps,
    image_scores,
    ssim_map,
)
from crackvision.models.hybrid import (
    HybridModel,
    LogisticFusion,
    box_features,
    unknown_regions,
)
from crackvision.models.yolox_model import YoloxDetector, build_model, preprocess


def test_autoencoder_reconstructs_the_input_shape():
    model = ConvAutoencoder(base_channels=8, depth=3, latent_channels=4)
    x = torch.rand(2, 3, 64, 64)
    assert model(x).shape == x.shape


def test_ssim_of_identical_images_is_one_and_anomaly_map_is_zero():
    x = torch.rand(1, 3, 32, 32)
    assert ssim_map(x, x).mean().item() == pytest.approx(1.0, abs=1e-4)
    amap = anomaly_maps(x, x.clone(), ssim_weight=0.5, blur_sigma=1.0)
    assert amap.shape == (1, 32, 32)
    assert amap.abs().max().item() < 1e-4


def test_anomaly_map_highlights_a_dark_line():
    x = torch.full((1, 3, 64, 64), 0.6)
    damaged = x.clone()
    damaged[:, :, 30:33, 5:60] = 0.1
    amap = anomaly_maps(damaged, x, ssim_weight=0.5, blur_sigma=1.0)[0]
    assert amap[31, 30] > 10 * amap[5, 5] + 1e-6
    scores = image_scores(amap[None], topk=0.05)
    assert scores.item() > 0


def test_yolox_detector_outputs_boxes_in_image_coordinates():
    model = build_model("nano", num_classes=1)
    detector = YoloxDetector(model, input_size=128, nms=0.5)
    images = [np.random.randint(0, 255, (100, 140, 3), dtype=np.uint8)]
    dets = detector.predict(images, conf=0.0)
    assert dets[0].ndim == 2 and dets[0].shape[1] == 5
    if len(dets[0]):
        assert dets[0][:, [0, 2]].max() <= 140 and dets[0][:, [1, 3]].max() <= 100
    array, ratio = preprocess(images[0], (128, 128))
    assert array.shape == (3, 128, 128) and ratio == pytest.approx(128 / 140)


def _fusion() -> dict:
    image_model = LogisticFusion(features=["a", "b", "c", "d"], mean=[0, 0, 0, 0],
                                 scale=[1, 1, 1, 1], coef=[4.0, 0.0, 1.0, 0.0], intercept=-2.0)
    box_model = LogisticFusion(features=["a", "b", "c", "d"], mean=[0, 0, 0, 0],
                               scale=[1, 1, 1, 1], coef=[1.0, 1.0, 0.0, 0.0], intercept=0.0)
    return {
        "operating_points": {"yolox": {"image_threshold": 0.5, "box_threshold": 0.3},
                             "autoencoder": {"image_threshold": 0.2, "pixel_threshold": 0.5},
                             "hybrid": {"image_threshold": 0.5, "box_threshold": 0.5}},
        "image_model": image_model.to_dict(),
        "box_model": box_model.to_dict(),
        "pixel_stats": {"mean": 0.1, "std": 0.1, "threshold": 0.5},
        "params": {"det_conf_floor": 0.01, "box_evidence_topk": 0.1, "unknown_threshold": 0.8,
                   "unknown_min_area": 10, "unknown_max_overlap": 0.1},
    }


def test_logistic_fusion_matches_the_formula():
    fusion = LogisticFusion(features=["x"], mean=[1.0], scale=[2.0], coef=[3.0], intercept=0.5)
    expected = 1 / (1 + np.exp(-(3.0 * (5.0 - 1.0) / 2.0 + 0.5)))
    assert fusion.predict_proba(np.array([[5.0]]))[0] == pytest.approx(expected)


def test_box_features_reflect_autoencoder_evidence():
    amap = np.zeros((64, 64), dtype=np.float32)
    amap[10:20, 10:40] = 1.0
    dets = np.array([[10, 10, 40, 20, 0.6], [45, 45, 60, 60, 0.6]], dtype=np.float32)
    features = box_features(dets, amap, {"mean": 0.1, "std": 0.1, "threshold": 0.5}, 0.1)
    assert features[0, 1] > features[1, 1]
    assert features[0, 2] == pytest.approx(1.0)
    assert features[1, 2] == 0.0


def test_unknown_regions_ignore_areas_covered_by_a_box():
    amap = np.zeros((64, 64), dtype=np.float32)
    amap[5:15, 5:15] = 1.0
    amap[40:50, 40:50] = 1.0
    boxes = np.array([[4, 4, 16, 16, 0.9]], dtype=np.float32)
    regions = unknown_regions(amap, 0.5, 10, boxes, 0.1)
    assert len(regions) == 1
    assert regions[0]["box"] == [40, 40, 50, 50]


def test_hybrid_levels():
    hybrid = HybridModel(_fusion())
    amap = np.zeros((64, 64), dtype=np.float32)
    amap[10:20, 10:40] = 1.0
    strong = hybrid.predict(np.array([[10, 10, 40, 20, 0.95]], dtype=np.float32), amap, 3.0)
    assert strong["decision"] and strong["level"] == "fissure"
    empty = hybrid.predict(np.zeros((0, 5), dtype=np.float32), np.zeros((64, 64), np.float32), -1)
    assert not empty["decision"] and empty["level"] == "sain"
    flagged = hybrid.predict(np.zeros((0, 5), dtype=np.float32), amap, 0.0)
    assert flagged["level"] == "anomalie_non_classee"
    assert flagged["unknown_regions"]


def test_box_measurements_and_calibration():
    result = box_measurements(np.array([10, 20, 40, 60]), pixel_per_mm=5.0)
    assert result["largeur_px"] == 30 and result["hauteur_px"] == 40
    assert result["diagonale_px"] == 50
    assert result["largeur_mm"] == 6.0 and result["surface_mm2"] == 48.0


def test_mask_measurements_on_a_straight_crack():
    mask = np.zeros((40, 120), dtype=np.uint8)
    mask[18:23, 10:110] = 1
    result = mask_measurements(mask, scale=0.5)
    # Longueur ~ 100 px a l'echelle de travail, donc ~ 200 px dans l'image d'origine.
    assert 180 <= result["longueur_estimee_px"] <= 210
    assert 6 <= result["ouverture_moyenne_px"] <= 14


@pytest.mark.parametrize("kind", ["tache_humidite", "efflorescence", "rouille", "eclat"])
def test_out_of_class_defects_change_the_tile_inside_their_mask(kind):
    from crackvision.evaluation.stress import paste_defect

    image = np.full((256, 256, 3), 150, dtype=np.uint8)
    altered, mask = paste_defect(image, kind, np.random.default_rng(0))
    assert 0.01 < mask.mean() < 0.5
    diff = np.abs(altered.astype(int) - image.astype(int)).sum(axis=2)
    assert diff[mask].mean() > 5 * (diff[~mask].mean() + 1e-6)


def test_pseudo_mask_is_clipped_to_the_map():
    amap = np.ones((10, 10), dtype=np.float32)
    mask = pseudo_mask(amap, np.array([-5, -5, 30, 30]), 0.5)
    assert mask.shape == (10, 10) and mask.all()
