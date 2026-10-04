# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy
import re
from pathlib import Path

import pytest
import yaml

import run_orchestrator as ro

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml"))


def _load(path):
    cfg, _ = ro.load_config(path)
    return cfg


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_config_is_valid(path, config_env):
    cfg = _load(path)
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    assert warnings == []
    assert cfg["config_id"] == path.stem
    for d, stem in ro.deploy_targets(cfg):
        assert stem.endswith("_" + ro.compiler_target(d))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_config_has_no_stale_keys_or_paths(path):
    text = path.read_text()
    for stale in (
        "origami/data",
        "ml_recommender",
        "backends/hipblaslt",
        "rank_time_userargs_yaml",
        "hardening",
        "Logic/asm_full/gfx1250v0",
        "rotating_mb: 512            #",
    ):
        assert stale not in text
    assert not re.search(r"\d+(\.\d+)?\s*%", text), "configs carry no measured numbers"


def test_config_ids_are_unique():
    ids = [yaml.safe_load(p.read_text())["config_id"] for p in CONFIGS]
    assert len(ids) == len(set(ids))


def test_shipped_configs_differ_in_effect(config_env):
    seen = {}
    for p in CONFIGS:
        cfg = _load(p)
        cfg.pop("config_id")
        key = yaml.safe_dump(cfg, sort_keys=True)
        assert key not in seen, f"{p.name} duplicates {seen.get(key)}"
        seen[key] = p.name


def test_exclude_keys_match_lib_shapes():
    from lib import shapes

    assert tuple(ro.EXCLUDE_KEYS) == tuple(shapes.EXCLUDE_KEYS)


@pytest.fixture
def base_cfg(config_env):
    return _load(Path(CONFIGS[0]).parent / "gfx950_bbs_tn.yaml")


def _errors(cfg):
    return ro.validate_config(cfg)[0]


def test_unknown_key_is_an_error(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["val_frac"] = 0.5
    cfg["bogus"] = 1
    errs = _errors(cfg)
    assert "train.val_frac: unknown key" in errs
    assert "bogus: unknown key" in errs


def test_missing_required_key(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    del cfg["hipblaslt"]["library_stem"]
    assert "hipblaslt.library_stem: required key is missing" in _errors(cfg)


@pytest.mark.parametrize(
    "path,value,fragment",
    [
        (("train", "epochs"), "200", "expected an integer"),
        (("train", "epochs"), True, "expected an integer"),
        (("deploy", "apply_patch"), "yes", "expected true or false"),
        (("hipblaslt", "transA"), "X", "is not one of"),
        (("validate", "sel_eff_threshold"), 1.5, "(0, 1] or 'origami'"),
        (("deploy", "weight_dtype"), "fp16", "is not one of"),
        (("runtime", "devices"), [], "at least 1"),
        (("deploy", "targets"), ["gfx950/../x"], "invalid target"),
    ],
)
def test_type_errors(base_cfg, path, value, fragment):
    cfg = copy.deepcopy(base_cfg)
    cfg[path[0]][path[1]] = value
    assert any(fragment in e for e in _errors(cfg)), _errors(cfg)


def test_origami_threshold_is_accepted(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["validate"]["sel_eff_threshold"] = "origami"
    cfg["validate"]["split_floor"] = "origami"
    assert _errors(cfg) == []


def test_split_floor_zero_disables_the_floor(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["validate"]["split_floor"] = 0
    assert _errors(cfg) == []
    for bad in (-0.1, 1.5):
        cfg["validate"]["split_floor"] = bad
        assert any("[0, 1] (0 disables) or 'origami'" in e for e in _errors(cfg))
    cfg["validate"]["split_floor"] = 0.83
    cfg["validate"]["sel_eff_threshold"] = 0
    assert any("(0, 1] or 'origami'" in e for e in _errors(cfg))


def test_split_budget_must_be_feasible(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"]["shapes_per_cell"] = 30
    cfg["train"]["min_cell_gemms"] = 30
    cfg["validate"]["min_cell_gemms"] = 20
    assert any("split budget infeasible" in e for e in _errors(cfg))
    cfg["validate"]["max_split_depth"] = 0
    assert not any("split budget" in e for e in _errors(cfg))


def test_round_count_without_split_room_warns(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["full"]["total_rounds"] = 2
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    assert any("no split can happen" in w for w in warnings)


def test_apply_patch_needs_a_checkout(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["deploy"]["rocm_libraries_to_deploy"] = None
    assert any("requires deploy.rocm_libraries_to_deploy" in e for e in _errors(cfg))


def test_parity_without_deploy_warns(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["deploy"]["apply_patch"] = False
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    assert any("stage07" in w for w in warnings)


def test_probe_keys_required_without_adaptive_timing(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    del cfg["probe"]["duration_us"]
    assert any("probe.duration_us" in e for e in _errors(cfg))
    cfg["bench"]["adaptive"] = {"enabled": True}
    assert not any("probe" in e for e in _errors(cfg))


def test_training_gate_above_round0_budget(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"]["target_per_cell"] = 10
    assert any("round_0 would train no cell" in e for e in _errors(cfg))


def test_deploy_targets_derive_stems(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["arch"] = "gfx1250v0"
    cfg["hipblaslt"][
        "library_stem"
    ] = "TensileLibrary_X_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1250"
    cfg["deploy"]["targets"] = ["gfx1250v0", "gfx1250", "gfx1250-strict"]
    got = dict(ro.deploy_targets(cfg))
    assert got["gfx1250v0"] == got["gfx1250"] == cfg["hipblaslt"]["library_stem"]
    assert got["gfx1250-strict"].endswith("_Cijk_Dijk_gfx1250-strict")
    cfg["deploy"]["targets"] = [
        {"dir": "gfx1250-strict", "library_stem": "TensileLibrary_Y_gfx1250-strict"}
    ]
    assert ro.deploy_targets(cfg) == [
        ("gfx1250-strict", "TensileLibrary_Y_gfx1250-strict")
    ]


def test_deploy_target_stem_cannot_be_derived(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["hipblaslt"]["library_stem"] = "TensileLibrary_X_no_arch_suffix"
    cfg["deploy"]["targets"] = ["gfx1250"]
    with pytest.raises(ro.ConfigError):
        ro.deploy_targets(cfg)


def test_env_expansion(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("a: ${TW_TEST_SET}/x\nb: [~/y]\n")
    monkeypatch.setenv("TW_TEST_SET", "/root")
    cfg, raw = ro.load_config(p)
    assert cfg["a"] == "/root/x" and not cfg["b"][0].startswith("~")
    assert raw == p.read_text()
    for value in ("${TW_TEST_UNSET}", "$TW_TEST_EMPTY/z", "${TW_TEST_SET:-x}"):
        monkeypatch.setenv("TW_TEST_EMPTY", "")
        monkeypatch.delenv("TW_TEST_UNSET", raising=False)
        p.write_text(f"a: {value}\n")
        with pytest.raises(ro.ConfigError):
            ro.load_config(p)


def test_cfg_get_defaults(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    del cfg["bench"]
    del cfg["stage07"]
    assert ro.cfg_get(cfg, "bench.blocks_per_gpu") == 20
    assert ro.cfg_get(cfg, "stage07.timing_request_sizes") == [1]
    assert ro.cfg_get(cfg, "stage07.parity_min_match_rate") == 1.0
    assert ro.cfg_get(cfg, "deploy.weight_dtype") == "int8"
