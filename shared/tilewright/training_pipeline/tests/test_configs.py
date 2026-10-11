# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy
import math
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
        "probe",
        "stage01b",
    ):
        assert stale not in text
    assert not re.search(r"\d+(\.\d+)?\s*%", text), "configs carry no measured numbers"


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_configs_state_their_bench_timing(path):
    from lib.bench_yaml import ADAPTIVE_DEFAULTS

    bench = yaml.safe_load(path.read_text())["bench"]
    assert "enabled" not in bench["adaptive"]
    assert bench["adaptive"] == ADAPTIVE_DEFAULTS
    assert bench["skip_slow_solution_ratio"] == 0.45


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_configs_of_a_mode_dependent_arch_name_the_device_mode(path, config_env):
    from lib.arch import STRICT_SWITCH, mode_dependent, reported_arch
    from lib.bench_env import runtime_env

    header = "".join(ln for ln in path.read_text().splitlines(True) if ln[:1] == "#")
    cfg = _load(path)
    needs = mode_dependent(cfg["arch"])
    assert (f"{STRICT_SWITCH}=0" in header) == needs
    if needs:
        assert reported_arch("gfx1250", 0, runtime_env(cfg)) == cfg["arch"]


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
        (("validate", "sel_eff_threshold"), 1.5, "a number in (0, 1], got 1.5"),
        (("validate", "sel_eff_threshold"), "0.9", "a number in (0, 1], got '0.9'"),
        (("deploy", "weight_dtype"), "fp16", "is not one of"),
        (("runtime", "devices"), [], "at least 1"),
        (("deploy", "targets"), ["gfx950/../x"], "invalid target"),
    ],
)
def test_type_errors(base_cfg, path, value, fragment):
    cfg = copy.deepcopy(base_cfg)
    cfg[path[0]][path[1]] = value
    assert any(fragment in e for e in _errors(cfg)), _errors(cfg)


@pytest.mark.parametrize("dtype", ["f4_r", "f6_r", "bf6_r"])
def test_four_and_six_bit_operands_need_a_block_format(base_cfg, dtype):
    cfg = copy.deepcopy(base_cfg)
    cfg["hipblaslt"].update(a_type=dtype, b_type=dtype)
    cfg["hipblaslt"].pop("scaleA", None)
    cfg["hipblaslt"]["scaleB"] = 1
    errs = _errors(cfg)
    assert any(
        e.startswith(f"hipblaslt.scaleA: {dtype} operands are block-scaled")
        and e.endswith("not nothing")
        for e in errs
    ), errs
    assert any(e.startswith("hipblaslt.scaleB") and e.endswith("not 1") for e in errs)
    cfg["hipblaslt"].update(scaleA=3, scaleB=3)
    assert _errors(cfg) == []
    for mode in (4, 5, 6, 7, 8, 1001):
        cfg["hipblaslt"].update(scaleA=mode, scaleB=mode)
        assert _errors(cfg) == [], mode


def test_origami_threshold_is_rejected(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["validate"]["sel_eff_threshold"] = "origami"
    cfg["validate"]["split_floor"] = "Origami"
    errs = _errors(cfg)
    for key in ("validate.sel_eff_threshold", "validate.split_floor"):
        assert any(e.startswith(f"{key}: 'origami' is not accepted") for e in errs)
    assert ro.cfg_get({}, "validate.sel_eff_threshold") == 0.95
    assert ro.cfg_get({}, "validate.split_floor") == 0.85


def test_split_floor_zero_disables_the_floor(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["validate"]["split_floor"] = 0
    assert _errors(cfg) == []
    for bad in (-0.1, 1.5, float("nan")):
        cfg["validate"]["split_floor"] = bad
        assert any("a number in [0, 1] (0 disables)" in e for e in _errors(cfg))
    cfg["validate"]["split_floor"] = 0.83
    cfg["validate"]["sel_eff_threshold"] = 0
    assert any("a number in (0, 1]" in e for e in _errors(cfg))


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


def test_split_attempts_warning_counts_structural_splits(base_cfg):
    def attempts_warnings(cfg):
        errors, warnings = ro.validate_config(cfg)
        assert errors == []
        return [w for w in warnings if w.startswith("validate.split_after_attempts")]

    cfg = copy.deepcopy(base_cfg)
    cfg["full"]["total_rounds"] = 3
    cfg["validate"]["split_after_attempts"] = 1
    cfg["validate"]["structural_split_min_octaves"] = 0.0
    (w,) = attempts_warnings(cfg)
    assert "no round in which a split can happen" in w
    cfg["validate"]["structural_split_min_octaves"] = 3.0
    (w,) = attempts_warnings(cfg)
    assert "only the structural width trigger" in w and "a split can" not in w
    for key, value in (
        ("structural_split_min_octaves", 11.0),
        ("structural_split_max_depth", 0),
    ):
        none = copy.deepcopy(cfg)
        none["validate"][key] = value
        (w,) = attempts_warnings(none)
        assert "no round in which a split can happen" in w
    cfg["validate"]["split_after_attempts"] = 0
    assert attempts_warnings(cfg) == []


def test_apply_patch_needs_a_checkout(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["deploy"]["rocm_libraries_to_deploy"] = None
    assert any("requires deploy.rocm_libraries_to_deploy" in e for e in _errors(cfg))


def test_deploy_gate_settings(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    assert ro.cfg_get(cfg, "deploy.gate_frac") == 0.2
    assert ro.cfg_get(cfg, "validate.pick_measurement") == "warmup"
    cfg["deploy"]["gate_frac"] = 0
    errors, warnings = ro.validate_config(cfg)
    assert errors == [] and any("deploy.gate_frac is 0" in w for w in warnings)
    cfg["deploy"]["target_sel_eff"] = None
    assert not any("gate_frac" in w for w in ro.validate_config(cfg)[1])
    for bad in (0.6, -0.1):
        cfg["deploy"]["gate_frac"] = bad
        assert any(e.startswith("deploy.gate_frac") for e in _errors(cfg))
    cfg["deploy"]["gate_frac"] = 0.2
    cfg["validate"]["pick_measurement"] = "fastest"
    assert any(e.startswith("validate.pick_measurement") for e in _errors(cfg))


def test_parity_without_deploy_warns(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["deploy"]["apply_patch"] = False
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    assert any("stage07" in w for w in warnings)


PROBE_SECTION = {
    "first_line_iters": 10,
    "first_line_cold_iters": 10,
    "other_lines_iters": 1,
    "other_lines_cold_iters": 0,
    "duration_us": 2000,
    "skip_ratio_mode": "sigmoid",
}


def test_a_probe_section_is_rejected_because_timing_is_always_adaptive(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["probe"] = dict(PROBE_SECTION)
    (err,) = _errors(cfg)
    assert err.startswith(
        "probe: the probe stage was removed and adaptive timing is always on"
    )
    assert "delete this section" in err
    cfg["probe"] = {}
    assert _errors(cfg) == [err]


@pytest.mark.parametrize(
    "path, value, fragment",
    [
        (("bench", "adaptive", "enabled"), True, "adaptive timing is always on"),
        (("bench", "adaptive", "enabled"), False, "adaptive timing is always on"),
        (("bench", "rebench", "iters_factor"), 2, "keep the adaptive timing"),
        (("planner", "probe_kernels"), 3, "plans name no probe kernels"),
    ],
)
def test_removed_bench_keys_say_what_replaced_them(base_cfg, path, value, fragment):
    cfg = copy.deepcopy(base_cfg)
    node = cfg
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value
    (err,) = _errors(cfg)
    assert err.startswith(".".join(path) + ": ") and fragment in err
    assert "unknown key" not in err


def test_removed_keys_are_reported_with_the_other_errors(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["probe"] = dict(PROBE_SECTION)
    cfg["bench"]["adaptive"] = {"enabled": True, "measure_time": -1}
    errs = _errors(cfg)
    assert [e.split(":")[0] for e in errs] == [
        "probe",
        "bench.adaptive.enabled",
        "bench.adaptive.measure_time",
    ]
    assert cfg["probe"] == PROBE_SECTION and cfg["bench"]["adaptive"]["enabled"]


def test_adaptive_timing_defaults_are_the_bench_lines(base_cfg):
    from lib.bench_yaml import ADAPTIVE_DEFAULTS, ADAPTIVE_KEYS, adaptive_settings

    assert tuple(ro.SCHEMA.children["bench"].children["adaptive"].children) == (
        ADAPTIVE_KEYS
    )
    for key, default in ADAPTIVE_DEFAULTS.items():
        assert ro.cfg_get({}, f"bench.adaptive.{key}") == default
    assert adaptive_settings({}) == ADAPTIVE_DEFAULTS
    cfg = copy.deepcopy(base_cfg)
    cfg["bench"]["adaptive"] = {"measure_time": 2.5, "min_iters": 7}
    assert _errors(cfg) == []
    assert adaptive_settings(cfg) == dict(
        ADAPTIVE_DEFAULTS, measure_time=2.5, min_iters=7
    )


@pytest.mark.parametrize(
    "adaptive, fragment",
    [
        ({"min_iters": 9, "max_iters": 4}, "min_iters exceeds max_iters"),
        ({"measure_time": 9.0, "max_measure_time": 4.0}, "exceeds max_measure_time"),
        ({"max_measure_time": 0, "max_iters": 0}, "the ceiling of a measurement"),
        ({"stability_window": 1}, "needs stability_window >= 2"),
        ({"stability_interval": 0}, "stability_interval >= 1"),
    ],
)
def test_adaptive_timing_follows_the_bench_rules(base_cfg, adaptive, fragment):
    cfg = copy.deepcopy(base_cfg)
    cfg["bench"]["adaptive"] = adaptive
    assert any(fragment in e for e in _errors(cfg)), _errors(cfg)
    if "stability_window" in adaptive or "stability_interval" in adaptive:
        cfg["bench"]["adaptive"] = dict(adaptive, stability_threshold=0)
        assert _errors(cfg) == []
    if adaptive.get("max_measure_time") == 0:
        cfg["bench"]["adaptive"] = dict(adaptive, max_iters=50, min_iters=3)
        assert _errors(cfg) == []


def test_training_gate_above_round0_budget(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"]["target_per_cell"] = 10
    assert any("round_0 would train no cell" in e for e in _errors(cfg))
    cfg["seed"]["target_per_cell"] = cfg["train"]["min_cell_gemms"]
    errors, warnings = ro.validate_config(cfg)
    assert errors == [] and any("give seed.target_per_cell" in w for w in warnings)


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_configs_leave_round0_slack(path, config_env):
    cfg = _load(path)
    assert ro.cfg_get(cfg, "seed.target_per_cell") > ro.cfg_get(
        cfg, "train.min_cell_gemms"
    )


GFX1250_CONFIGS = [p for p in CONFIGS if p.name.startswith("gfx1250")]
# Data types whose gfx1250 library hipBLASLt does not build yet.
WITHOUT_LIBRARY = ("f8bs", "mxf6bs")


def test_gfx1250_has_tn_configs_of_every_datatype():
    assert sorted(p.name for p in GFX1250_CONFIGS) == [
        "gfx1250-strict_bbs_tn_v3.yaml",
        "gfx1250-strict_mxf8bs_tn_v3.yaml",
        "gfx1250_bbs_tn_v3.yaml",
        "gfx1250_f8bs_tn_v3.yaml",
        "gfx1250_mxf4bs_tn_v3.yaml",
        "gfx1250_mxf6bs_tn_v3.yaml",
        "gfx1250_mxf8bs_tn_v3.yaml",
    ]


@pytest.mark.parametrize("path", GFX1250_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_configs_read_every_device_value_at_run_time(path, config_env):
    from lib import contexts as ctxlib

    cfg = _load(path)
    assert "hardware" not in cfg
    budgets = [
        m.get("cu_budget")
        for key in ("list", "holdout")
        for m in cfg["contexts"][key]
        if m.get("cu_budget") is not None
    ]
    assert budgets == [0]
    assert ctxlib.has_unresolved(cfg) and ctxlib.trained_entries(cfg)
    text = path.read_text()
    assert not re.search(r"\b(n_cu|num_xcd|lds_bytes|l2_bytes|pci_chip_id)\b", text)
    assert not re.search(r"\b\d+\s*(CUs?|compute units)\b", text, re.I)


def _comment_above(path, key):
    """The comment lines right above the first line of `path` that sets
    `key`, joined."""
    lines = path.read_text().splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.strip().startswith(f"{key}:"))
    above = []
    for ln in reversed(lines[:at]):
        if not ln.strip().startswith("#"):
            break
        above.insert(0, ln.strip().lstrip("#").strip())
    return " ".join(above)


@pytest.mark.parametrize("path", GFX1250_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_configs_follow_the_current_recipe(path, config_env):
    from lib import bank_pool
    from lib import library_pool as lp

    cfg = _load(path)
    get = lambda dotted: ro.cfg_get(cfg, dotted)  # noqa: E731
    assert get("train.catalog") == "v3" and get("train.whitelist") == "none"
    assert get("train.cascade_k") > 0 and get("train.served_candidates")
    assert get("train.recipe") == "pooled_ft"
    assert get("train.pseudo_negatives") == "targeted"
    assert get("train.unmeasured_samples") > 0
    assert get("train.repeat_aggregation") == "median"
    assert get("train.trained_key_gate") is True
    assert get("train.non_temporal_rule") is False
    assert get("train.non_temporal_family_rule") is False
    assert get("full.final_retrain.enabled") and get("audit.build") == "before"
    assert get("bench.kernel_order") == "shuffled" and get("bench.sacrificial_kernel")
    assert get("hazards.enabled") is True
    assert get("planner.enabled") and get("planner.mode") == "planned"
    assert get("contexts.replan")
    # Origami libraries alone, named by the library section rather than by
    # the planner's legacy bank keys, so that a library file without a
    # Prediction table is refused.
    assert lp.pool_source(cfg) == lp.PoolSource(kinds=("origami",))
    assert lp.legacy_keys_set(cfg) == [] and not bank_pool.bank_source(cfg).enabled
    assert _comment_above(path, "kind").startswith(
        "The kernel pool is the Origami library alone"
    )
    assert get("audit.dir") and get("bench.rebench.enabled")
    assert "seed" in cfg and "exclude" not in cfg["seed"]
    # The correctness gate is on wherever its harness runs the problem type:
    # BF16, and the block-scaled types once the harness runs them.
    assert get("correctness.enabled") is _harness_runs(cfg)
    if cfg["hipblaslt"]["a_type"] == "bf16_r":
        assert get("correctness.enabled")
    skip = _comment_above(path, "skip_slow_solution_ratio")
    assert "Not validated on gfx1250" in skip
    assert get("bench.heavy_operand_gib") == 4
    assert _comment_above(path, "heavy_operand_gib").startswith(
        "Lines with an operand of this many GiB or more are benched last"
    )


def test_only_the_gfx1250_configs_bench_heavy_lines_last():
    """`bench.heavy_operand_gib` is off by default; a gfx950 config keeps
    the blocks its runs were benched with."""
    for path in CONFIGS:
        bench = yaml.safe_load(path.read_text()).get("bench") or {}
        assert ("heavy_operand_gib" in bench) == (path in GFX1250_CONFIGS), path
    assert ro.cfg_get({}, "bench.heavy_operand_gib") == 0.0


def _harness_runs(cfg):
    """Whether the correctness gate's exact harness runs the config's
    problem type."""
    from lib import correctness_gate as cgate

    try:
        cgate.harness_problem(cfg)
    except cgate.GateError:
        return False
    return True


# The configs of devices that report gfx1250 hold a first run's settings
# (README.md, First run on a new gfx1250 host); the gfx1250-strict ones keep
# their own.
FIRST_RUN_CONFIGS = [p for p in GFX1250_CONFIGS if p.name.startswith("gfx1250_")]
# planner.top_k by data type, sized to each library's kernel pool; FP8 and
# MXFP6, whose libraries are not built yet, take MXFP8's.
FIRST_RUN_TOP_K = {"bbs": 96, "mxf4bs": 80, "mxf8bs": 64, "f8bs": 64, "mxf6bs": 64}


def _datatype(path):
    return path.name[len("gfx1250_") :].split("_tn_")[0]


def test_first_run_configs_are_the_gfx1250_ones():
    assert sorted(_datatype(p) for p in FIRST_RUN_CONFIGS) == sorted(FIRST_RUN_TOP_K)


@pytest.mark.parametrize("path", FIRST_RUN_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_configs_budget_a_first_run(path, config_env):
    cfg = _load(path)
    get = lambda dotted: ro.cfg_get(cfg, dotted)  # noqa: E731
    assert get("seed.target_per_cell") == 150
    assert get("seed.shapes_per_cell") == 75
    assert get("full.total_rounds") == 3
    assert get("full.final_retrain.enabled")
    # A cell can still split: the last round never does, so a split needs
    # its attempts used up by the round before it.
    assert ro._split_rounds_possible(cfg)
    assert get("validate.split_after_attempts") <= get("full.total_rounds") - 3
    assert get("planner.top_k") == FIRST_RUN_TOP_K[_datatype(path)]
    assert get("bench.stall_timeout_s") == ro.cfg_get({}, "bench.stall_timeout_s")
    assert get("bench.startup_grace_s") > ro.cfg_get({}, "bench.startup_grace_s")


@pytest.mark.parametrize("path", FIRST_RUN_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_configs_reread_near_winners_in_a_wide_band(path, config_env):
    from lib.rebench import REREAD_DEFAULTS

    cfg = _load(path)
    assert ro.cfg_get(cfg, "bench.rebench.enabled")
    reread = {k: ro.cfg_get(cfg, f"bench.rebench.{k}") for k in REREAD_DEFAULTS}
    assert reread == dict(
        REREAD_DEFAULTS,
        tolerance=0.15,
        fast_below_us=100,
        fast_runs=2,
        runs=1,
        slow_runs=1,
    )
    assert reread["tolerance"] > REREAD_DEFAULTS["tolerance"]
    assert reread["fast_below_us"] > REREAD_DEFAULTS["fast_below_us"]


@pytest.mark.parametrize("path", FIRST_RUN_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_contexts_keep_every_entry_on_devices_of_three_granules_up(
    path, config_env
):
    from lib import contexts as ctxlib
    from lib.hipblaslt_launch import SM_TARGET_GRANULE as granule

    cfg = _load(path)
    fractions = [m["cu_fraction"] for m in ctxlib.trained_entries(cfg)]
    assert len(fractions) == 2
    for n_cu in range(3 * granule, 16 * granule + 1):
        resolved, lines = ctxlib.resolve_fractions(cfg, n_cu)
        assert not [ln for ln in lines if ln.endswith("left out")], lines
        assert ctxlib.context_errors(cfg, n_cu) == []
        assert ro._min_cu_fraction_errors(cfg, n_cu) == []
        budgets = [m["cu_budget"] for m in ctxlib.trained_entries(resolved)]
        assert len(set(budgets)) == len(fractions)
        assert all(b % granule == 0 for b in budgets)
        if n_cu % (3 * granule) == 0:
            assert budgets == [round(f * n_cu) for f in fractions], n_cu


@pytest.mark.parametrize("path", GFX1250_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_models_declare_no_budget_below_the_smallest_they_trained(
    path, config_env
):
    """A v3 model declares the CU fractions from train.min_cu_fraction, else
    from the smallest budget it trained (stage05 declared_contexts), up to
    1. On no device of 3 to 16 SM-target granules may that floor reach a
    budget the SM target takes below the smallest trained one, which the
    model would serve without data."""
    import stage05_train as s5
    from lib import contexts as ctxlib
    from lib.hipblaslt_launch import SM_TARGET_GRANULE as granule

    cfg = _load(path)
    floor = ro.cfg_get(cfg, "train.min_cu_fraction")
    for n_cu in range(3 * granule, 16 * granule + 1):
        resolved, _ = ctxlib.resolve_fractions(cfg, n_cu)
        trained = [c for c in ctxlib.contexts_from_config(resolved) if not c.holdout]
        gemms = [{}] + [{"cu_budget": c.cu_budget} for c in trained]
        declared = s5.declared_contexts(gemms, n_cu, floor)["min_cu_fraction"]
        lowest = min(c.cu_budget for c in trained)
        first = math.ceil(declared * n_cu / granule - 1e-9) * granule
        assert first >= lowest, (n_cu, declared, lowest)


@pytest.mark.parametrize("path", FIRST_RUN_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_configs_take_protocol_v2_until_a_power_soak(path, config_env):
    cfg = _load(path)
    assert ro.cfg_get(cfg, "audit.v2") == "rounds"
    assert ro.cfg_get(cfg, "audit.sustained") is False
    assert ro.cfg_get(cfg, "audit.official") == "v2"
    assert ro.cfg_get(cfg, "bench.sustained.enabled") is False


def test_gfx1250_configs_point_to_the_first_run_checklist():
    readme = (CONFIGS[0].parent.parent / "README.md").read_text()
    assert "\n## First run on a new gfx1250 host\n" in readme
    for path in FIRST_RUN_CONFIGS:
        header = " ".join(
            ln.lstrip("#").strip()
            for ln in path.read_text().splitlines()
            if ln.startswith("#")
        )
        assert '"First run on a new gfx1250 host"' in header, path.name


def test_gfx1250_configs_are_named_for_their_subtree():
    for path in GFX1250_CONFIGS:
        arch = yaml.safe_load(path.read_text())["arch"]
        assert path.name.startswith(arch + "_")
        assert arch in ("gfx1250", "gfx1250-strict")


@pytest.mark.parametrize("path", GFX1250_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_configs_deploy_their_revisions_subtrees_alone(path, config_env):
    cfg = _load(path)
    stem = cfg["hipblaslt"]["library_stem"]
    targets = dict(ro.deploy_targets(cfg))
    if cfg["arch"] == "gfx1250-strict":
        assert targets == {
            "gfx1250-strict": stem,
            "gfx1250v0": stem[: -len("-strict")],
        }
        assert stem.endswith("_gfx1250-strict")
    else:
        assert targets == {"gfx1250": stem} and stem.endswith("_gfx1250")


@pytest.mark.parametrize("path", GFX1250_CONFIGS, ids=lambda p: p.name)
def test_gfx1250_configs_name_their_library_and_its_scales(path, config_env):
    from lib.dat import mx_block_size_for_scale_mode, parse_scale_mode

    cfg = _load(path)
    hl = cfg["hipblaslt"]
    mx = "_MXAE8B32_MXBE8B32_" in hl["library_stem"]
    assert mx == (mx_block_size_for_scale_mode(hl["scaleA"]) == 32)
    assert hl["scaleA"] == hl["scaleB"]
    named = parse_scale_mode(hl["library_stem"] + ".dat")
    assert named in (None, hl["scaleA"])
    missing = any(f"_{t}_tn_" in path.name for t in WITHOUT_LIBRARY)
    status = "hipBLASLt has no gfx1250 library of this problem type yet"
    assert (status in path.read_text().replace("\n# ", " ")) == missing


@pytest.mark.parametrize(
    "path",
    [p for p in GFX1250_CONFIGS if any(f"_{t}_tn_" in p.name for t in WITHOUT_LIBRARY)],
    ids=lambda p: p.name,
)
def test_a_config_whose_library_is_not_built_names_it(path, config_env, tmp_path):
    import bench_fakes as bf

    cfg = _load(path)
    stem = cfg["hipblaslt"]["library_stem"]
    libdir = tmp_path / "library" / cfg["arch"]
    libdir.mkdir(parents=True)
    other = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk"
    bf.write_logic(libdir / f"{other}_Cijk_Dijk_gfx1250.dat", [bf.solution(0, 0)])
    (err,) = ro._library_errors(libdir, stem, "bench")
    assert err.startswith(f"the bench library {stem} is missing: ")
    assert "no library of its problem type at all" in err


def test_a_missing_library_names_the_built_ones_of_its_type(tmp_path):
    import bench_fakes as bf

    head = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk"
    stem, kin = f"{head}_Cijk_Dijk_gfx9990", f"{head}_Cijk_Dijk_CU7_gfx9990"
    libdir = tmp_path / "gfx9990"
    libdir.mkdir()
    bf.write_logic(libdir / f"{kin}.dat", [bf.solution(1, 1)])
    (err,) = ro._library_errors(libdir, stem, "bench")
    assert err == (
        f"the bench library {stem} has no .dat/.dat.zlib in {libdir}; the "
        f"libraries of its problem type there: {kin}"
    )
    assert ro._library_errors(libdir, kin, "bench") == []


# The gfx950 configs whose kernel pool is the Origami library alone, one per
# library with a shipped model: the recipe its model trains with, the host's
# values left to a run config outside the repository.
GFX950_ORIGAMI_CONFIGS = [
    p for p in CONFIGS if re.fullmatch(r"gfx950_\w+_origami_v3", p.stem)
]
# The environment variables the shipped configs name host paths through
# (conftest.py, config_env).
CONFIG_VARIABLES = {
    "TILEWRIGHT_BENCH_ROOT",
    "TILEWRIGHT_DEPLOY_ROOT",
    "TILEWRIGHT_DATASETS",
    "TILEWRIGHT_AUDIT_ROOT",
}


def test_gfx950_has_an_origami_config_of_every_shipped_library():
    assert sorted(p.stem for p in GFX950_ORIGAMI_CONFIGS) == [
        f"gfx950_{lib}_origami_v3"
        for lib in (
            "bbs_nn",
            "bbs_nt",
            "bbs_tn",
            "bbs_tt",
            "f8bs_tn",
            "tf32_nn",
            "tf32_nt",
            "tf32_tn",
            "tf32_tt",
        )
    ]


@pytest.mark.parametrize("mode", ["full", "audit"])
@pytest.mark.parametrize("path", GFX950_ORIGAMI_CONFIGS, ids=lambda p: p.name)
def test_gfx950_origami_configs_validate_in_full_and_audit_mode(path, mode, config_env):
    assert ro.validate_config(_load(path), mode) == ([], [])


@pytest.mark.parametrize("path", GFX950_ORIGAMI_CONFIGS, ids=lambda p: p.name)
def test_gfx950_origami_pools_are_the_prediction_table(path, config_env, tmp_path):
    """library.kind origami and none of the planner's bank keys: the pool is
    the Prediction table of the library file in table order, with no bank
    and no other file, and a library file without the table is refused."""
    import bench_fakes as bf
    from lib import bank_pool
    from lib import library_pool as lp

    cfg = _load(path)
    assert cfg["library"] == {"kind": "origami"}
    assert lp.pool_source(cfg) == lp.PoolSource(kinds=("origami",))
    assert lp.legacy_keys_set(cfg) == [] and not bank_pool.bank_source(cfg).enabled
    stem = cfg["hipblaslt"]["library_stem"]
    libdir = tmp_path / cfg["arch"]
    libdir.mkdir()
    kernels = [bf.solution(i, i) for i in (4, 1, 3, 2)]
    bf.write_logic(libdir / f"{stem}.dat", kernels, table=[3, 1, 4])
    pool = lp.config_pool(cfg, libdir, stem, apply_blocklist=False)
    assert pool.indices == [3, 1, 4] and pool.parts[0].rows == {"prediction": 3}
    assert not pool.bank.kernels and pool.files == [stem]
    assert lp.pool_files(cfg, libdir, stem) == []
    bf.write_logic(libdir / f"{stem}.dat", kernels)
    with pytest.raises(ValueError, match="no Prediction table"):
        lp.config_pool(cfg, libdir, stem, apply_blocklist=False)


def _strings(node):
    """Every string of a parsed YAML document."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str):
        yield node


@pytest.mark.parametrize("path", GFX950_ORIGAMI_CONFIGS, ids=lambda p: p.name)
def test_gfx950_origami_configs_name_no_host_path_or_device(path):
    """A host's devices, the GPUs a run must not use, its CPUs and its paths
    belong in a run config kept outside the repository: these configs list
    device 0, as the gfx1250 ones do, and name every path through the
    shipped configs' environment variables, comments included."""
    text = path.read_text()
    raw = yaml.safe_load(text)
    assert raw["runtime"] == {
        "devices": [0],
        "rocm_libraries_root": "${TILEWRIGHT_BENCH_ROOT}",
    }
    assert "hardware" not in raw and "cpus" not in raw["stage07"]
    for section in ("correctness", "hazards", "audit"):
        assert "devices" not in raw[section]
    assert raw["stage08"] == {"datasets": []}
    assert raw["audit"]["dir"] == "${TILEWRIGHT_AUDIT_ROOT}/" + path.stem
    for value in _strings(raw):
        assert not value.startswith(("/", "~")), value
        assert set(re.findall(r"\$\{?(\w+)", value)) <= CONFIG_VARIABLES, value
    assert not re.search(r"(?:^|[\s\"'(\[{,:=])~?/\w", text), "an absolute path"


@pytest.mark.parametrize("path", GFX950_ORIGAMI_CONFIGS, ids=lambda p: p.name)
def test_gfx950_origami_configs_take_the_non_temporal_rule_where_the_layout_serves_it(
    path, config_env
):
    """The non-temporal rule serves a kernel with a non-temporal A hint only
    with A transposed, and one with a B hint only with B not transposed: on,
    with its family-wise refinement, for TN, which has both, and off for NN,
    NT and TT, where it would never serve some of the hinted kernels."""
    from lib import feasibility as fz
    from lib.features import dtype_from_hipblaslt

    cfg = _load(path)
    h = cfg["hipblaslt"]
    layout = path.stem.split("_")[2]
    assert layout == (h["transA"] + h["transB"]).lower()
    assert ("_Alik_" in h["library_stem"]) == (h["transA"] == "T")
    assert ("_Bljk_" in h["library_stem"]) == (h["transB"] == "N")
    a, b = dtype_from_hipblaslt(h["a_type"]), dtype_from_hipblaslt(h["b_type"])

    def serves(hints, m, n):
        # K and DepthU aligned for the element size of every config's data.
        return fz.is_kernel_feasible(
            m,
            n,
            4096,
            1,
            a,
            b,
            h["transA"],
            h["transB"],
            256,
            256,
            128,
            16,
            16,
            32,
            *hints,
            grvw_a=1,
            grvw_b=1,
            nt_a_available=True,
            nt_b_available=True,
        )

    # Thin along M, the rule's B branch decides; thin along N, its A branch.
    both = serves((0, 4), 64, 65536) and serves((4, 0), 65536, 64)
    assert both is (layout == "tn")
    assert ro.cfg_get(cfg, "train.non_temporal_rule") is both
    assert ro.cfg_get(cfg, "train.non_temporal_family_rule") is both


@pytest.mark.parametrize("path", GFX950_ORIGAMI_CONFIGS, ids=lambda p: p.name)
def test_gfx950_origami_configs_keep_the_recipe_of_their_models(path, config_env):
    """Sizing, contexts and audit set as the runs that train these models use
    them, the training recipe, and the correctness gate of the problem
    type's exact harness."""
    from lib import contexts as ctxlib

    cfg = _load(path)
    get = lambda dotted: ro.cfg_get(cfg, dotted)  # noqa: E731
    assert (get("seed.target_per_cell"), get("seed.shapes_per_cell")) == (150, 75)
    assert get("full.total_rounds") == 3 and get("full.final_retrain.enabled")
    assert get("full.final_retrain.gate") == "not-worse"
    budgets = [(m.get("cu_budget"), m.get("fraction")) for m in get("contexts.list")]
    assert budgets == [(0, None), (192, 0.1), (128, 0.1), (64, 0.1)]
    assert all(m.get("cotenant") for m in get("contexts.list")[1:])
    assert get("contexts.holdout") == [] and get("contexts.replan")
    assert get("train.min_cu_fraction") == 0.25
    # Budgets of a whole gfx950 device, down to the fraction the models
    # declare.
    assert ctxlib.context_errors(cfg, 256) == []
    assert ro._min_cu_fraction_errors(cfg, 256) == []
    assert (get("audit.shapes_per_cell"), get("audit.shapes_per_context")) == (8, 2)
    assert get("audit.build") == "before" and get("audit.v2") == "rounds"
    assert get("audit.sustained") and get("audit.official") == "sustained"
    assert get("train.catalog") == "v3" and get("train.whitelist") == "none"
    assert get("train.recipe") == "pooled_ft" and get("train.whitelist_tier_k") == 32
    assert get("train.cascade_k") > 0 and get("train.served_candidates")
    assert get("train.pseudo_negatives") == "targeted"
    assert get("train.trained_key_gate") is True
    assert get("train.repeat_aggregation") == "median"
    assert get("planner.enabled") and get("planner.mode") == "planned"
    assert get("bench.kernel_order") == "shuffled" and get("bench.sacrificial_kernel")
    assert get("bench.rebench.enabled") and get("hazards.enabled")
    assert get("correctness.enabled") and get("correctness.lists") == []
    assert _harness_runs(cfg)
    assert not get("deploy.apply_patch")


def test_the_gfx950_fp8_config_gates_its_pool_with_the_scalar_scale_harness(
    config_env,
):
    """The exact harness runs E4M3 data with one f32 scale per tensor, so the
    FP8 config turns the gate on instead of naming a list of its findings."""
    from lib import correctness_gate as cgate

    cfg = _load(CONFIGS[0].parent / "gfx950_f8bs_tn_origami_v3.yaml")
    problem = cgate.harness_problem(cfg)
    assert problem.scalar_scales and problem.mx_bits == (0, 0)
    assert cgate.shapes_of(problem) == cgate.SCALAR_SHAPES


def test_only_the_gfx950_bbs_tn_config_distils_as_its_shipped_model():
    """The shipped BBS TN model's distillation weight and temperature and
    pooled epochs; the other libraries keep stage05's."""
    for path in GFX950_ORIGAMI_CONFIGS:
        train = yaml.safe_load(path.read_text())["train"]
        keys = [train.get(k) for k in ("distill_lam", "distill_T", "pooled_epochs")]
        if path.stem != "gfx950_bbs_tn_origami_v3":
            assert keys == [None, None, 12], path.name
            continue
        assert keys == [0.5, 4, 24]
        assert _comment_above(path, "distill_lam").startswith(
            "The shipped model of this library was trained with this recipe on "
            "the rows of a longer BBS TN run restricted to the Origami table"
        )


def test_deploy_targets_derive_stems(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["arch"] = "gfx1250-strict"
    stem = "TensileLibrary_X_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1250-strict"
    cfg["hipblaslt"]["library_stem"] = stem
    cfg["deploy"]["targets"] = ["gfx1250-strict", "gfx1250v0", "gfx1250"]
    got = dict(ro.deploy_targets(cfg))
    assert got["gfx1250-strict"] == stem
    assert got["gfx1250v0"] == got["gfx1250"] == stem[: -len("-strict")]
    assert ro.validate_config(cfg) == ([], [])
    cfg["deploy"]["targets"] = [
        {"dir": "gfx1250-strict", "library_stem": "TensileLibrary_Y_gfx1250-strict"}
    ]
    assert ro.deploy_targets(cfg) == [
        ("gfx1250-strict", "TensileLibrary_Y_gfx1250-strict")
    ]


def test_build_alias_arch_points_to_its_architecture(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["arch"] = "gfx1250v0"
    cfg["hipblaslt"]["library_stem"] = "TensileLibrary_X_gfx1250"
    cfg["deploy"]["targets"] = ["gfx1250v0"]
    errors = _errors(cfg)
    assert len(errors) == 1
    assert errors[0].startswith("arch: 'gfx1250v0' is hipBLASLt's build alias")
    assert "set arch: gfx1250-strict" in errors[0]
    assert "_gfx1250-strict" in errors[0] and "deploy.targets" in errors[0]


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
