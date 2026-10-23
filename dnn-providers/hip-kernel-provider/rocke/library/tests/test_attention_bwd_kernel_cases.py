# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``custom`` correctness configuration of ``tests/sdpa/bwd_kernel_cases.py``.

CPU only: parsing of ``ROCKE_BWD_CUSTOM_KNOBS``, rule matching per head size
and ``stage_vec``, configuration selection, and the plan a custom policy
resolves (the swept candidate's main kernel for matching requests, the arch
default for the others); the knob smoke sets of the device numerics plan on
every (arch, case) they run with every knob in the resolved spec, cover the
built-and-swept knob values that had no device run, and round-trip through
``ROCKE_BWD_CUSTOM_KNOBS``; the layout twin sets and their twins plan (legal
and within the LDS budget) on every (arch, case) the device test runs, and
differ from their twin in the LDS layout only.
"""

import ast
import dataclasses
import json

import pytest

from benchmarks.common.attention_bwd_rocke_arm import RockeConfig, config_policy
from kernels.common.attention_bwd_plan import attn_bwd_plan, attn_bwd_stage_vec

from ._attention_bwd_test_utils import find_bwd_case
from .sdpa.bwd_kernel_cases import (
    CONFIGS,
    CUSTOM_CONFIG,
    CUSTOM_KNOBS_ENV,
    KNOB_SMOKE_CASES,
    KNOB_SMOKE_REFUSED,
    KNOB_SMOKE_SETS,
    COMPOSE_REFUSED,
    COMPOSE_TWIN_SETS,
    LAYOUT_TWIN_REFUSED,
    LAYOUT_TWIN_SETS,
    SUPPLEMENTARY_CASES,
    config_knobs,
    config_policy as case_policy,
    custom_knobs_for,
    custom_rules,
    knob_smoke_params,
    knob_smoke_rules_json,
    compose_twin_params,
    layout_twin_params,
    request_from_case,
    selected_cases,
    selected_configs,
)

ARCH = "gfx950"
# A swept geometry of the d128 primary slice (not the arch default).
D128_KNOBS = {
    "waves": 8,
    "block_m": 32,
    "block_n": 128,
    "block_k4": 32,
    "warp_grid_g4": [2, 4],
}
RULES = [{"head_size": 128, "stage_vec": 8, "knobs": D128_KNOBS}]


def _cases(d, *, narrow=False):
    out = []
    for c in selected_cases(tier="smoke"):
        if c.d != d:
            continue
        req = request_from_case(c)
        if narrow:
            req = dataclasses.replace(req, tensor_alignment=8)
        if (attn_bwd_stage_vec(req) == 1) == narrow:
            out.append((c, req))
    assert out, (d, narrow)
    return out


def test_rules_parse_lists_and_matches():
    rules = custom_rules(json.dumps(RULES + [{"knobs": {"g_split": 2}}]))
    assert rules[0] == (
        {"head_size": 128, "stage_vec": 8},
        {**D128_KNOBS, "warp_grid_g4": (2, 4)},
    )
    assert rules[1] == ({}, {"g_split": 2})
    assert custom_knobs_for(rules, 128, 8)["warp_grid_g4"] == (2, 4)
    assert custom_knobs_for(rules, 128, 1) == {"g_split": 2}  # first match wins
    assert custom_knobs_for(rules[:1], 64, 8) == {}
    assert custom_knobs_for(rules[:1], 128, 1) == {}


def test_rules_unset_or_empty_is_none(monkeypatch):
    monkeypatch.delenv(CUSTOM_KNOBS_ENV, raising=False)
    assert custom_rules() is None
    assert custom_rules("  ") is None
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps(RULES))
    assert custom_rules()[0][0] == {"head_size": 128, "stage_vec": 8}


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        "[]",
        json.dumps({"knobs": {}}),
        json.dumps([{"head_size": 128}]),
        json.dumps([{"knobs": {"head_size": 64}}]),  # a request field
        json.dumps([{"knobs": {"block_q": 64}}]),  # not a knob
        json.dumps([{"dtype": "bf16", "knobs": {}}]),  # not a match key
    ],
)
def test_malformed_rules_raise(raw):
    with pytest.raises(ValueError):
        custom_rules(raw)


def test_custom_is_selected_only_by_name(monkeypatch):
    monkeypatch.delenv(CUSTOM_KNOBS_ENV, raising=False)
    assert CUSTOM_CONFIG not in CONFIGS
    assert CUSTOM_CONFIG not in selected_configs(ARCH, "both")
    with pytest.raises(ValueError):
        selected_configs(ARCH, "custom")
    with pytest.raises(ValueError):
        case_policy(CUSTOM_CONFIG, ARCH)
    with pytest.raises(ValueError):
        config_knobs(CUSTOM_CONFIG, ARCH)
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps(RULES))
    assert CUSTOM_CONFIG not in selected_configs(ARCH, "both")
    assert selected_configs(ARCH, "custom") == ["custom"]
    assert selected_configs(ARCH, "default,custom,custom") == ["default", "custom"]


def test_custom_policy_plans_the_candidate_where_it_matches(monkeypatch):
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps(RULES))
    pol = case_policy(CUSTOM_CONFIG, ARCH)
    default = case_policy("default", ARCH)
    cand = config_policy(RockeConfig("c", custom_rules()[0][1]))
    for _, req in _cases(128):
        spec = attn_bwd_plan(req, ARCH, policy=pol).specs["main"]
        assert spec == attn_bwd_plan(req, ARCH, policy=cand).specs["main"]
        assert spec != attn_bwd_plan(req, ARCH, policy=default).specs["main"]
        assert (spec.waves, spec.block_m, spec.block_n) == (8, 32, 128)
        assert spec.warp_grid_g4 == (2, 4)
    # other head sizes and the narrow-access class keep the arch default
    for _, req in _cases(64) + _cases(128, narrow=True):
        spec = attn_bwd_plan(req, ARCH, policy=pol).specs["main"]
        assert spec == attn_bwd_plan(req, ARCH, policy=default).specs["main"]


def test_custom_policy_applies_overrides_on_top(monkeypatch):
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps(RULES))
    pol = case_policy(CUSTOM_CONFIG, ARCH, {"edge_tiles": True})
    for _, req in _cases(128)[:3]:
        spec = attn_bwd_plan(req, ARCH, policy=pol).specs["main"]
        assert spec.edge_tiles and spec.block_m == 32
    for _, req in _cases(64)[:3]:
        spec = attn_bwd_plan(req, ARCH, policy=pol).specs["main"]
        assert spec.edge_tiles


def test_custom_cells_the_validator_refuses_skip(monkeypatch):
    """The ``custom`` configuration's own refused cell skips the device test
    with the validator's reason (a caller's knob stack, not a shipped one); a
    variant on top of a custom cell that plans, and a shipped configuration's
    refusal, raise (the variant tests plan first: ``skip_unless_planned``)."""
    from ._attention_bwd_harness import case_plan, plan_case

    case = _smoke_case("bwd_kern_hkv_842_tl_fp16_d64")
    refused = {"stats_vec": 4}  # an SGPR-spilling kernel on gfx950 at d64
    with pytest.raises(ValueError, match="resource gate"):
        case_plan(case, arch="gfx950", config="default", overrides=refused)
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps([{"knobs": refused}]))
    for plan in (plan_case, lambda *a, **kw: case_plan(*a, **kw)[0]):
        with pytest.raises(pytest.skip.Exception, match="custom configuration refuses"):
            plan(case, arch="gfx950", config=CUSTOM_CONFIG)
    # a variant that plans on top of the refused cell still plans
    case_plan(case, arch="gfx950", config=CUSTOM_CONFIG, overrides={"edge_tiles": True})
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps([{"knobs": {"stats_vec": 2}}]))
    case_plan(case, arch="gfx950", config=CUSTOM_CONFIG)
    with pytest.raises(ValueError, match="resource gate"):
        case_plan(case, arch="gfx950", config=CUSTOM_CONFIG, overrides=refused)


# Device test modules whose tests run under each selected configuration (a
# ``config`` parameter, the ``custom`` configuration included).
_CONFIG_DEVICE_MODULES = (
    "test_attention_bwd_fwd_chain.py",
    "test_attention_bwd_mutations.py",
    "test_attention_bwd_numeric.py",
    "test_attention_bwd_sched_numeric.py",
    "test_attention_bwd_split_numeric.py",
    "test_attention_bwd_valu_numeric.py",
)
_LAUNCHES = {"run_case", "case_plan", "plan_case"}
# calls that plan the variant first and skip a refused one
_PLAN_FIRST = {"skip_unless_planned", "_require_legal", "_split_run"}


def _call_name(node):
    f = node.func
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)


def _unplanned_variant_launches(tree):
    """``[(test, line)]`` of the launches with ``overrides`` under the test's
    ``config`` parameter that no plan-first call precedes (in the test's own
    body; a ``plan_case`` in a ``try`` whose handler skips counts as one)."""
    out = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_")):
            continue
        if "config" not in {a.arg for a in fn.args.args}:
            continue
        guards = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and _call_name(node) in _PLAN_FIRST:
                guards.append(node.lineno)
            if isinstance(node, ast.Try) and any(
                isinstance(n, ast.Call) and _call_name(n) == "skip"
                for h in node.handlers
                for n in ast.walk(h)
            ):
                guards += [n.lineno for s in node.body for n in ast.walk(s)
                           if isinstance(n, ast.Call)]  # fmt: skip
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and _call_name(node) in _LAUNCHES):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            ov, cfg = kw.get("overrides"), kw.get("config")
            if ov is None or (isinstance(ov, ast.Constant) and ov.value is None):
                continue
            if not (isinstance(cfg, ast.Name) and cfg.id == "config"):
                continue  # a named shipped configuration: a refusal raises
            if not any(g <= node.lineno for g in guards):
                out.append((fn.name, node.lineno))
    return out


def test_variant_launches_under_the_selected_configuration_plan_first():
    """Every variant a device test launches on top of its selected
    configuration is planned first (``skip_unless_planned``): a variant the
    configuration cannot take, notably on a ``custom`` cell that plans, skips
    with the validator's reason instead of failing the run. Launches under a
    named shipped configuration are left out (a refusal there raises)."""
    import pathlib

    here = pathlib.Path(__file__).parent
    bad = {}
    for name in _CONFIG_DEVICE_MODULES:
        tree = ast.parse((here / name).read_text(encoding="utf-8"))
        found = _unplanned_variant_launches(tree)
        if found:
            bad[name] = found
    assert not bad, bad
    # the check sees an unguarded launch
    probe = ast.parse(
        "def test_x(case_id, config):\n"
        "    run_case(c, arch=A, config=config, overrides={'edge_tiles': True})\n"
    )
    assert _unplanned_variant_launches(probe) == [("test_x", 2)]
    probe = ast.parse(
        "def test_x(case_id, config):\n"
        "    skip_unless_planned(c, arch=A, config=config, overrides=ov)\n"
        "    run_case(c, arch=A, config=config, overrides=ov)\n"
    )
    assert _unplanned_variant_launches(probe) == []


# --------------------------------------------------------------------------- knob smoke sets
def _smoke_case(case_id):
    from .sdpa.bwd_cases import get_bwd_case

    for c in SUPPLEMENTARY_CASES:
        if c.id == case_id:
            return c
    return get_bwd_case(case_id)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_knob_smoke_sets_plan_with_every_knob(arch):
    params = knob_smoke_params(arch)
    assert params
    for name, cid in params:
        if (name, arch, cid) in KNOB_SMOKE_REFUSED:
            continue  # test_knob_smoke_refusals_are_exactly_the_recorded_cells
        knobs = KNOB_SMOKE_SETS[name][2]
        case = _smoke_case(cid)
        assert dict(KNOB_SMOKE_CASES)[cid] == case.d
        pol = case_policy("default", arch, dict(knobs))
        spec = attn_bwd_plan(request_from_case(case), arch, policy=pol).specs["main"]
        for k, v in knobs.items():
            assert getattr(spec, k) == v, (arch, name, cid, k)
        # the custom-configuration text plans the same main kernel
        rules = custom_rules(knob_smoke_rules_json(name))
        assert custom_knobs_for(rules, case.d, 8) == dict(knobs)


def test_knob_smoke_sets_cover_the_unverified_values():
    by_arch = {a: {n for n, _ in knob_smoke_params(a)} for a in ("gfx942", "gfx950")}
    knobs = {n: s[2] for n, s in KNOB_SMOKE_SETS.items()}
    every = by_arch["gfx942"] | by_arch["gfx950"]
    assert every == set(KNOB_SMOKE_SETS)

    def has(pred):
        return [n for n in every if pred(knobs[n])]

    def half_grid(k, f):
        w = k.get("waves", 8)
        return k.get(f) == (2, w // 2) and k.get("pt_route") == "lds"

    assert has(
        lambda k: half_grid(k, "warp_grid_g02") and half_grid(k, "warp_grid_g13")
    )
    assert set(has(lambda k: k.get("pt_route") == "lds")) & by_arch["gfx942"]
    mixed = has(lambda k: "atom_g02" in k or "atom_g4" in k)
    assert {knobs[n].get("atom_g02", knobs[n].get("atom_g4")) for n in mixed} == {
        "16x16x16"
    }
    assert {k.get("waves_per_eu") for k in knobs.values()} >= {2, 3}
    assert has(lambda k: k.get("acc_in_lds") is True)
    assert {k.get("scheduler_strategy") for k in knobs.values()} >= {
        "max-ilp",
        "max-memory-clause",
        "iterative-ilp",
        "iterative-maxocc",
    }
    assert has(lambda k: k.get("agpr_alloc") == "backend")
    assert has(lambda k: k.get("block_m") == 64)
    # every G4 key-slicing value, and G0 / G2 as asm (asm_g13 with s_in_agpr)
    assert {k.get("g4_slicing") for k in knobs.values()} >= {
        "slice",
        "slice_db",
        "slice_db_g3",
    }
    assert has(lambda k: k.get("agpr_route") == "asm_g13" and k.get("s_in_agpr"))
    # the wide G1 / G3 atom (K-permuted transpose read) at d64 and d128
    wide = has(lambda k: k.get("atom_g13") == "16x16x32")
    assert {KNOB_SMOKE_SETS[n][1] for n in wide} == {(64,), (128,)}
    assert set(wide) <= by_arch["gfx950"] and not set(wide) & by_arch["gfx942"]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_knob_smoke_refusals_are_exactly_the_recorded_cells(arch):
    """The validator refuses exactly the recorded knob-set cells (a resource
    refusal or a recorded layout spill), each a real cell of the device test,
    and never every cell of a set."""
    refused = set()
    for name, cid in knob_smoke_params(arch):
        knobs = KNOB_SMOKE_SETS[name][2]
        pol = case_policy("default", arch, dict(knobs))
        try:
            attn_bwd_plan(request_from_case(_smoke_case(cid)), arch, policy=pol)
        except ValueError as exc:
            assert "RESOURCE_REFUSED" in str(exc) or "LAYOUT_SPILLING_COMBOS" in str(
                exc
            ), (arch, name, cid, exc)
            refused.add((name, arch, cid))
    assert refused == {k for k in KNOB_SMOKE_REFUSED if k[1] == arch}
    for name in {k[0] for k in refused}:
        cells = [cid for n, cid in knob_smoke_params(arch) if n == name]
        assert any((name, arch, cid) not in refused for cid in cells), (arch, name)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_compose_twin_sets_plan_on_every_case(arch):
    """Each composition twin set and its twin plan on every case the device
    test runs on ``arch`` (validator and LDS budget) and carry their knobs; the
    set differs from its twin in the composed lever."""
    from kernels.common.attention_bwd import lds_plan
    from kernels.common.attention_bwd_plan import DQ_KNOBS

    params = compose_twin_params(arch)
    if arch in ("gfx942", "gfx950"):
        assert params
    else:
        assert not params  # every composition runs on the CDNA targets
    for name, cid in params:
        if (name, arch, cid) in COMPOSE_REFUSED:
            continue  # test_compose_refusals_are_exactly_the_recorded_cells
        _archs, _sizes, knobs, twin = COMPOSE_TWIN_SETS[name]
        case = _smoke_case(cid)
        specs = []
        for kn in (knobs, twin):
            pol = case_policy("default", arch, dict(kn))
            plan = attn_bwd_plan(request_from_case(case), arch, policy=pol)
            spec = plan.specs["main"]
            for k, v in kn.items():
                # a dQ kernel knob is a field of the dQ kernel's spec
                own, f = (plan.specs["dq"], k[3:]) if k in DQ_KNOBS else (spec, k)
                assert getattr(own, f) == v, (arch, name, cid, k)
            assert lds_plan(spec, arch).fits, (arch, name, cid)
            specs.append(spec)
        a, b = specs
        diff = {f for f in a.__dataclass_fields__ if getattr(a, f) != getattr(b, f)}
        lever = set(knobs) - set(twin)
        assert diff & lever, (name, diff, lever)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_compose_refusals_are_exactly_the_recorded_cells(arch):
    """The validator refuses (a resource refusal) the set's or its twin's main
    kernel in exactly the recorded composition cells, with the recorded
    rules."""
    refused = {}
    for name, cid in compose_twin_params(arch):
        _archs, _sizes, knobs, twin = COMPOSE_TWIN_SETS[name]
        why = []
        for which, kn in (("set", knobs), ("twin", twin)):
            pol = case_policy("default", arch, dict(kn))
            try:
                attn_bwd_plan(request_from_case(_smoke_case(cid)), arch, policy=pol)
            except ValueError as exc:
                assert "RESOURCE_REFUSED" in str(exc), (arch, name, cid, exc)
                rule = str(exc).split(f"fails on {arch} (", 1)[1].split(":", 1)[0]
                why.append(f"{which}: {rule}")
        if why:
            refused[(name, arch, cid)] = ", ".join(why)
    assert refused == {k: v for k, v in COMPOSE_REFUSED.items() if k[1] == arch}
    for name in {n for n, _ in compose_twin_params(arch)}:  # a cell runs
        cells = [cid for n, cid in compose_twin_params(arch) if n == name]
        assert any((name, arch, cid) not in refused for cid in cells), (arch, name)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_layout_twin_sets_plan_on_every_case(arch):
    """Each layout twin set and its twin plan on every case the device test
    runs on ``arch`` (validator and LDS budget), carry their knobs, and differ
    only in ``transpose_source`` / ``lds_swizzle`` (the LDS layout); the
    validator refuses exactly the recorded cells (LAYOUT_TWIN_REFUSED, a
    resource refusal of the set or of its twin), and every set keeps a
    runnable cell."""
    from kernels.common.attention_bwd import lds_plan

    params = layout_twin_params(arch)
    assert params
    refused = {}
    for name, cid in params:
        _archs, _sizes, knobs, twin = LAYOUT_TWIN_SETS[name]
        why = []
        for which, kn in (("set", knobs), ("twin", twin)):
            pol = case_policy("default", arch, dict(kn))
            try:
                attn_bwd_plan(request_from_case(_smoke_case(cid)), arch, policy=pol)
            except ValueError as exc:
                assert "RESOURCE_REFUSED" in str(exc), (arch, name, cid, exc)
                rule = str(exc).split(f"fails on {arch} (", 1)[1].split(":", 1)[0]
                why.append(f"{which}: {rule}")
        if why:
            refused[(name, arch, cid)] = ", ".join(why)
    assert refused == {k: v for k, v in LAYOUT_TWIN_REFUSED.items() if k[1] == arch}
    for name in {n for n, _ in params}:  # a cell runs
        cells = [cid for n, cid in params if n == name]
        assert any((name, arch, cid) not in refused for cid in cells), (arch, name)
    for name, cid in params:
        if (name, arch, cid) in refused:
            continue
        _archs, _sizes, knobs, twin = LAYOUT_TWIN_SETS[name]
        case = _smoke_case(cid)
        specs = []
        for kn in (knobs, twin):
            pol = case_policy("default", arch, dict(kn))
            plan = attn_bwd_plan(request_from_case(case), arch, policy=pol)
            spec = plan.specs["main"]
            for k, v in kn.items():
                assert getattr(spec, k) == v, (arch, name, cid, k)
            assert lds_plan(spec, arch).fits, (arch, name, cid)
            specs.append(spec)
        a, b = specs
        diff = {f for f in a.__dataclass_fields__ if getattr(a, f) != getattr(b, f)}
        assert diff and diff <= {"transpose_source", "lds_swizzle"}, (name, diff)


# --------------------------------------------------------------------------- device variants
# Variants of the device numerics that no cell of a CDNA target can take in
# the shipped configurations, with the validator rule that refuses them (their
# cells skip with that reason; every other variant must keep a planned cell).
ALL_REFUSED_VARIANTS = {
    ("gfx942", "merge", "merge_head_outer_edge"): "ring_depth > 1 excludes the grouped head-outer form",
    ("gfx950", "merge", "merge_head_outer_edge"): "ring_depth > 1 excludes the grouped head-outer form",
    # the grouped head-outer form with the rotated q loop spills SGPRs on every
    # gfx942 cell (RESOURCE_REFUSED); gfx950 still runs it
    ("gfx942", "runtime", "q_rotate_head_outer_edge"): "RESOURCE_REFUSED (sgpr_spill)",
    # an explicit AGPR reservation on the VGPR route (agpr_route None): the
    # VALU reads its AGPR accumulators back in the hot loop on every shipped
    # tile, at every legal reservation (RESOURCE_REFUSED, agpr_copy_traffic)
    ("gfx942", "runtime", "agpr128"): "RESOURCE_REFUSED (agpr_copy_traffic)",
    ("gfx950", "runtime", "agpr128"): "RESOURCE_REFUSED (agpr_copy_traffic)",
}  # fmt: skip


def _plans(case, arch, config, overrides=None, request=None, thd=None):
    """The plan the device test launches (a THD storage form through the
    harness's ``case_plan``), or ``None`` when the validator refuses it."""
    from ._attention_bwd_harness import case_plan, plan_case

    try:
        if thd is not None:
            return case_plan(
                case, arch=arch, config=config, overrides=overrides,
                request=request, thd=thd,
            )[0]  # fmt: skip
        return plan_case(
            case, arch=arch, config=config, overrides=overrides, request=request
        )
    except ValueError:
        return None


def _variant_cells(arch):
    """``{(test, variant): [(case, config, overrides, request), ...]}`` of the
    device numerics' variant tests on ``arch`` in the shipped configurations,
    with the overrides each test plans (a reservation or chunk the test picks
    at run time is picked here the same way)."""
    from kernels.common._attention_bwd_body import DEFAULT_STAGE_ORDER, STAGE_ORDERS
    from kernels.common.attention_bwd_plan import SCALE_PLACEMENTS

    from . import test_attention_bwd_mutations as M
    from . import test_attention_bwd_numeric as N
    from . import test_attention_bwd_valu_numeric as V
    from .sdpa.bwd_kernel_cases import (
        HK_NE_HV_TABLE_CASES,
        NARROW_TWIN_CASES,
        THD_HK_NE_HV_CASES,
    )

    cfgs = [c for c in CONFIGS if c != CUSTOM_CONFIG and config_knobs(c, arch)
            is not None]  # fmt: skip
    out: dict = {}

    def add(key, case, cfg, ov=None, req=None, thd=None):
        out.setdefault(key, []).append((case, cfg, ov, req, thd))

    # the single-variant tests (each plans its variant first and skips a
    # refused cell)
    for cfg in cfgs:
        for cid in HK_NE_HV_TABLE_CASES + THD_HK_NE_HV_CASES:
            add(("select_guard", "select"), find_bwd_case(cid), cfg,
                {"dkv_atomic_guard": "select"})  # fmt: skip
        for cid, ov in N.RUN_TO_RUN:
            add(("run_to_run", cid), find_bwd_case(cid), cfg, dict(ov))
        for cid in N.VARIANT_CASES:
            for placement in SCALE_PLACEMENTS:
                for static in (placement, "runtime"):
                    ov = {"scale_placement": placement, "static_scale": static}
                    add(("static_scale", f"{placement}/{static}"), find_bwd_case(cid),
                        cfg, ov)  # fmt: skip
        for cid in N.SLAB_HOIST_CASES:
            for form, storage in N.SLAB_HOIST_FORMS.items():
                add(("slab_hoist", form), N._thd_case(cid), cfg,
                    {"dkv_slab_hoist": True}, None, storage or N.ThdStorage())  # fmt: skip
        for cid in NARROW_TWIN_CASES:
            case = find_bwd_case(cid)
            narrow = _plans(case, arch, cfg, request={"tensor_alignment": 8})
            if narrow is None:
                continue
            spec = narrow.specs["main"]
            knobs = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)
                     if f.name not in N._REQUEST_FIELDS}  # fmt: skip
            add(("narrow_twin", "wide"), case, cfg, knobs)
        for cid in M.INTERIOR_DEAD_CASES:
            add(("dead_row", "interior_edge"), M.get_bwd_case(cid), cfg,
                {"edge_tiles": True})  # fmt: skip
        for cid in V.FAINT_CASES:
            for mode in ("exact", "fast"):
                add(("fast_exp2", mode), find_bwd_case(cid), cfg, {"bwd_exp2": mode})

    for cid in N.VARIANT_CASES:
        case = find_bwd_case(cid)
        for cfg in cfgs:
            for name, (ov, req, _bit) in N.RUNTIME_VARIANTS.items():
                ov = dict(ov)
                if name == "agpr128":
                    for a in ((128, 128), (192, 192), (256, 256)):
                        add(("runtime", name), case, cfg, {"agpr_alloc": a}, req)
                    continue
                if ov.get("xcd_chunk") == "gx":
                    base = _plans(case, arch, cfg)
                    ov = {"xcd_chunk": base.launches[1].grid[0]}
                add(("runtime", name), case, cfg, ov, req)
            for sia in (False, True):
                for cluster in ("k_step", "gemm"):
                    for a in ((128, 128), (192, 192), (256, 256)):
                        ov = dict(agpr_alloc=a, agpr_route="asm_g13", s_in_agpr=sia)
                        ov.update(asm_cluster=cluster, sched="none", setprio=False)
                        add(("asm", f"s{int(sia)}_{cluster}"), case, cfg, ov)
            # the asm operand-load hoist variants (test_asm_g13_variants_agree)
            for name, nop, cluster, sia in N._asm_variants():
                if name not in N._ASM_EXTRA:
                    continue
                for a in ((128, 128), (192, 192), (256, 256)):
                    ov = dict(agpr_alloc=a, agpr_route="asm_g13", s_in_agpr=sia)
                    ov.update(asm_cluster=cluster, sched="none", setprio=False)
                    ov.update(asm_tail_nop=nop, **N._ASM_EXTRA[name])
                    add(("asm_hoist", name), case, cfg, ov)
            # the wide G1 / G3 atom (gfx950; test_wide_g13_atom_agrees)
            if arch == "gfx950":
                plain = _plans(case, arch, cfg)
                if plain is None:
                    continue
                wide = {"atom_g13": "16x16x32"}
                if plain.specs["main"].block_m < 32:
                    wide["block_m"] = 32
                for name, knobs in N.WIDE_G13_VARIANTS.items():
                    knobs = dict(knobs)
                    if not knobs.pop("asm", False):
                        add(("wide", name), case, cfg, {**wide, **knobs})
                        continue
                    for a in ((128, 128), (192, 192), (256, 256)):
                        ov = dict(agpr_alloc=a, agpr_route="asm_g13", s_in_agpr=False)
                        ov.update(asm_cluster="k_step", sched="none", setprio=False)
                        ov.update(asm_tail_nop=None, **wide, **knobs)
                        add(("wide", name), case, cfg, ov)
    for cid in N.STAGE_ORDER_CASES:
        case = N._order_case(cid)
        for cfg in cfgs:
            for name, knobs in N.MERGE_VARIANTS.items():
                add(("merge", name), case, cfg, {**knobs, **N.MERGE_ON})
            for edge in (False, True):
                base_ov = {"edge_tiles": True} if edge else {}
                for order in STAGE_ORDERS:
                    if order != DEFAULT_STAGE_ORDER:
                        ov = dict(base_ov, stage_order=order)
                        add(("stage_order", f"{order}{'_edge' if edge else ''}"),
                            case, cfg, ov)  # fmt: skip
                spec0 = _plans(case, arch, cfg, base_ov)
                if spec0 is None:
                    continue
                for value in ("slice", "slice_db", "slice_db_g3"):
                    for div in (2, 4):
                        k4 = spec0.specs["main"].block_n // div
                        ov = dict(base_ov, g4_slicing=value, block_k4=k4)
                        add(("g4", f"{value}{'_edge' if edge else ''}"), case, cfg, ov)
    for case in selected_cases(tier="smoke"):
        for cfg in cfgs:
            add(("edge_tiles", "edge"), case, cfg, {"edge_tiles": True})
    return out


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_every_device_variant_keeps_a_planned_cell(arch):
    """A refusal (a coupling or a recorded resource refusal) never takes a
    variant of the device numerics off a CDNA target unnoticed: every runtime
    variant, asm route (and its operand-load hoist variants), wide G1 / G3
    atom variant (gfx950), merge variant, stage order, G4 slicing value (with
    and without edge tiles), the edge tiles themselves and the variant of each
    single-variant test (the select guard, run to run, static scale, slab
    hoist, narrow twin, interior dead rows, fast exp2) plan on at least one
    of their cases in a shipped configuration, except the variants that
    ALL_REFUSED_VARIANTS names with their rule (every cell of which is
    refused)."""
    for key, cells in sorted(_variant_cells(arch).items()):
        planned = [c for c in cells if _plans(c[0], arch, c[1], c[2], c[3], c[4])]
        if (arch, *key) in ALL_REFUSED_VARIANTS:
            assert not planned, (arch, key, "now plans: drop it from the list")
            continue
        assert planned, (arch, key, "no cell plans in a shipped configuration")
