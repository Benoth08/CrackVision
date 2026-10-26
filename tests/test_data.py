"""Donnees : decoupage par photo, tuiles, COCO, reequilibrage, integrite du blind test."""

import json

import numpy as np
import pandas as pd
import pytest

from crackvision.data import balance, prepare, sources, split
from crackvision.paths import SPLITS


@pytest.fixture()
def prepared(smoke_params, tmp_paths):
    """Ingestion, index, decoupage et tuiles sur des photos synthetiques."""
    sources.ingest(smoke_params, tmp_paths)
    prepare.build_index(smoke_params, tmp_paths)
    split.split_photos(smoke_params, tmp_paths)
    for name in SPLITS:
        prepare.tile_split(smoke_params, tmp_paths, name)
    return smoke_params, tmp_paths


def test_stratified_assignment_respects_fractions_and_is_deterministic():
    strata = pd.Series(["a"] * 30 + ["b"] * 50 + ["c"] * 20)
    fractions = {"train": 0.7, "val": 0.15, "blind_test": 0.15}
    first = split.stratified_assignment(strata, fractions, seed=1)
    second = split.stratified_assignment(strata, fractions, seed=1)
    assert first.equals(second)
    counts = first.value_counts()
    assert counts["train"] == 70 and counts["val"] == 15 and counts["blind_test"] == 15
    for stratum in "abc":
        share = (first[strata == stratum] == "train").mean()
        assert abs(share - 0.7) <= 2.0 / (strata == stratum).sum() + 1e-9


def test_photos_never_shared_between_splits(prepared):
    params, paths = prepared
    owners = {}
    for name in SPLITS:
        for photo in prepare.read_tiles(paths, name, include_ambiguous=True)["photo_id"].unique():
            assert photo not in owners, f"photo {photo} dans {owners.get(photo)} et {name}"
            owners[photo] = name
    assert set(owners.values()) == set(SPLITS)


def test_tiles_labels_and_coco(prepared):
    from pycocotools.coco import COCO

    params, paths = prepared
    tiles = prepare.read_tiles(paths, "train", include_ambiguous=True)
    assert {"positive", "healthy"} <= set(tiles["label"])
    positives = tiles[tiles["label"] == "positive"]
    assert (positives["n_boxes"] > 0).all()
    assert (tiles.loc[tiles["label"] == "healthy", "crack_pixels"] == 0).all()
    coco = COCO(str(paths.split_dir("train") / "annotations" / "instances.json"))
    assert len(coco.getImgIds()) == int((tiles["label"] != "ambiguous").sum())
    for ann in coco.loadAnns(coco.getAnnIds()):
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0 and x >= 0 and y >= 0
        assert x + w <= params["prepare"]["tile_size"] and y + h <= params["prepare"]["tile_size"]


def test_boxes_match_mask_components(prepared):
    params, paths = prepared
    tiles = prepare.read_tiles(paths, "train")
    row = tiles[tiles["label"] == "positive"].iloc[0]
    import cv2

    mask = cv2.imread(str(paths.split_dir("train") / "masks" / f"{row['tile_id']}.png"), 0) > 0
    boxes = prepare.tile_boxes(row)
    union = np.zeros_like(mask)
    for x1, y1, x2, y2 in boxes.astype(int):
        union[y1:y2, x1:x2] = True
    # Les pixels de fissure des grandes composantes sont dans les boites.
    assert (mask & union).sum() >= 0.8 * mask.sum()


def test_balanced_training_set(prepared):
    params, paths = prepared
    report = balance.build_balanced(params, paths)
    assert report["balanced"]["positive_fraction"] == pytest.approx(
        params["balance"]["positive_fraction"], abs=0.05)
    data = json.loads((paths.train_balanced / "annotations" / "instances.json").read_text())
    ids = [img["id"] for img in data["images"]]
    assert len(ids) == len(set(ids))
    images = paths.split_dir("train") / "images"
    assert all((images / img["file_name"]).exists() for img in data["images"])
    ae = pd.read_csv(paths.train_balanced / "ae_train.csv")
    healthy = prepare.read_tiles(paths, "train").query("label == 'healthy'")
    assert set(ae["tile_id"]) <= set(healthy["tile_id"])


def test_repeat_factors_favour_rare_strata():
    frame = pd.DataFrame({
        "largest_box_ratio": [0.01] * 2 + [0.1] * 18,
        "orientation": ["oblique"] * 2 + ["horizontale"] * 18,
        "n_boxes": [1] * 20,
    })
    frame = balance.add_strata(frame, [0.05, 0.25])
    repeats = balance.repeat_factors(frame, threshold=0.5, rng=np.random.default_rng(0))
    assert repeats[:2].min() >= 2
    assert (repeats[2:] == 1).all()


def test_manifest_detects_a_modified_blind_test(prepared):
    params, paths = prepared
    blind = paths.split_dir("blind_test")
    prepare.verify_manifest(blind)
    victim = next((blind / "images").glob("*.jpg"))
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="modifies"):
        prepare.verify_manifest(blind)
