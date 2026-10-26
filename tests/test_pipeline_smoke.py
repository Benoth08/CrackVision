"""Pipeline complet de fumee (CPU, donnees synthetiques) : chaque etape produit ses sorties."""

import pytest

from crackvision.data.prepare import verify_manifest
from crackvision.paths import get_paths
from crackvision.utils import read_json

pytestmark = pytest.mark.slow


def test_pipeline_outputs(smoke_workdir):
    paths = get_paths(smoke_workdir)
    for name in ("yolox", "autoencoder", "hybrid"):
        assert any(paths.model_dir(name).iterdir()), name
    assert (paths.model_dir("yolox") / "model.pth").exists()
    assert (paths.model_dir("autoencoder") / "calibration.json").exists()
    fusion = read_json(paths.model_dir("hybrid") / "fusion.json")
    assert set(fusion["operating_points"]) == {"yolox", "autoencoder", "hybrid"}
    for split in ("val", "blind_test"):
        metrics = read_json(paths.reports / split / "metrics.json")
        assert set(metrics["image"]) == {"yolox", "autoencoder", "hybrid"}
        assert {"ap50", "ap50_95"} <= set(metrics["detection"]["yolox"])
        assert metrics["n_tiles"] > 0
        for figure in ("image_pr_curves.png", "examples.png", "full_photo.png"):
            assert (paths.reports / split / "figures" / figure).exists(), figure
    assert (paths.reports / "data" / "balance.png").exists()
    stress = read_json(paths.reports / "blind_test" / "metrics.json")["stress_test_hors_classe"]
    assert set(stress["defauts"]) == {"tache_humidite", "efflorescence", "rouille", "eclat"}
    assert (paths.reports / "blind_test" / "figures" / "stress_rates.png").exists()


def test_gallery_runs_on_whole_photos(smoke_workdir, smoke_params):
    from crackvision.evaluation.gallery import run_gallery

    paths = get_paths(smoke_workdir)
    summary = run_gallery(smoke_params, paths)
    assert summary
    assert len({item["photo_id"] for item in summary}) == len(summary)
    assert (paths.reports / "blind_test" / "gallery" / "galerie_1.png").exists()


def test_blind_test_digest_is_recorded(smoke_workdir):
    paths = get_paths(smoke_workdir)
    metrics = read_json(paths.reports / "blind_test" / "metrics.json")
    assert metrics["manifest_digest"] == verify_manifest(paths.split_dir("blind_test"))
