"""Regles du pipeline DVC et coherence des fichiers de parametres."""

from functools import reduce

import yaml

from crackvision.config import project_root

ROOT = project_root()
ALLOWED_BLIND_READERS = {"evaluate@blind_test", "data_report"}


def _stages() -> dict:
    spec = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    expanded = {}
    for name, stage in spec["stages"].items():
        if "foreach" in stage:
            for item in stage["foreach"]:
                body = yaml.safe_dump(stage["do"]).replace("${item}", item)
                expanded[f"{name}@{item}"] = yaml.safe_load(body)
        else:
            expanded[name] = stage
    return expanded


def _keys(payload: dict, prefix: str = "") -> set[str]:
    keys = set()
    for key, value in payload.items():
        name = f"{prefix}{key}"
        keys.add(name)
        if isinstance(value, dict):
            keys |= _keys(value, f"{name}.")
    return keys


def test_only_final_evaluation_reads_the_blind_test():
    for name, stage in _stages().items():
        if name in ALLOWED_BLIND_READERS or name == "tile@blind_test":
            continue
        for dep in stage.get("deps", []):
            assert "blind_test" not in str(dep), f"{name} depend du blind test ({dep})"


def test_every_declared_param_exists():
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    known = _keys(params)
    for name, stage in _stages().items():
        for entry in stage.get("params", []):
            assert entry in known, f"{name} : parametre inconnu {entry}"


def test_smoke_profile_only_overrides_existing_keys():
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    smoke = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    unknown = _keys(smoke) - _keys(params)
    assert not unknown, f"cles du profil smoke absentes de params.yaml : {sorted(unknown)}"


def test_training_never_lists_blind_test_outputs():
    stages = _stages()
    blind_outs = set(stages["tile@blind_test"]["outs"])
    for name in ("train_yolox", "train_autoencoder", "calibrate", "evaluate@val", "balance"):
        assert not blind_outs & set(stages[name].get("deps", []))
    assert reduce(lambda acc, s: acc or "blind_test" in str(s),
                  stages["evaluate@blind_test"]["deps"], False)
