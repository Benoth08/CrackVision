import numpy as np
import pytest

from crackvision.evaluation.metrics import (
    ap_from_matches,
    best_threshold,
    box_prf,
    cluster_bootstrap,
    coco_detection_metrics,
    image_metrics,
    match_detections,
)


def test_match_detections_is_greedy_by_score():
    gts = np.array([[0, 0, 10, 10]], dtype=np.float32)
    dets = np.array([[0, 0, 10, 10, 0.6], [0, 0, 10, 9, 0.9]], dtype=np.float32)
    flags = match_detections(dets, gts, 0.5)
    assert flags.tolist() == [False, True]


def test_ap_from_matches_extremes():
    assert ap_from_matches(np.array([0.9, 0.8]), np.array([True, True]), 2) == pytest.approx(1.0)
    assert ap_from_matches(np.array([0.9]), np.array([False]), 1) == 0.0
    assert np.isnan(ap_from_matches(np.array([]), np.array([], dtype=bool), 0))


def test_coco_metrics_for_perfect_predictions():
    gt = {"images": [{"id": 1, "width": 100, "height": 100}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 20],
                           "area": 600, "iscrowd": 0}],
          "categories": [{"id": 1, "name": "fissure"}]}
    metrics = coco_detection_metrics(gt, {1: np.array([[10, 10, 40, 30, 0.9]])})
    assert metrics["ap50"] == pytest.approx(1.0)
    assert metrics["ap50_95"] == pytest.approx(1.0)


def test_box_prf_counts():
    per_image = [(np.array([[0, 0, 10, 10, 0.9], [50, 50, 60, 60, 0.4]]),
                  np.array([[0, 0, 10, 10]])),
                 (np.zeros((0, 5)), np.array([[0, 0, 5, 5]]))]
    result = box_prf(per_image, 0.5, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)


def test_image_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    threshold = best_threshold(y, scores)
    assert threshold == pytest.approx(0.7)
    result = image_metrics(y, scores, threshold)
    assert result["f1"] == pytest.approx(1.0)
    assert result["mcc"] == pytest.approx(1.0)
    assert result["confusion"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 2}


def test_cluster_bootstrap_is_deterministic_and_resamples_groups():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    first = cluster_bootstrap(groups, lambda idx: values[idx].mean(), 200, seed=3)
    second = cluster_bootstrap(groups, lambda idx: values[idx].mean(), 200, seed=3)
    assert first == second
    assert 1.0 <= first["low"] <= 2.0 <= first["high"] <= 3.0
