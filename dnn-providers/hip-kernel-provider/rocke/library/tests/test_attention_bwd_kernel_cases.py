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

import dataclasses
import json

import pytest

from benchmarks.common.attention_bwd_rocke_arm import RockeConfig, config_policy
from kernels.common.attention_bwd_plan import attn_bwd_plan, attn_bwd_stage_vec

from .sdpa.bwd_kernel_cases import (
    CONFIGS,
    CUSTOM_CONFIG,
    CUSTOM_KNOBS_ENV,
    KNOB_SMOKE_CASES,
    KNOB_SMOKE_REFUSED,
    KNOB_SMOKE_SETS,
    COMPOSE_REFUSED,
    COMPOSE_TWIN_SETS,
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
                assert getattr(spec, k) == v, (arch, name, cid, k)
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
    only in ``transpose_source`` / ``lds_swizzle`` (the LDS layout)."""
    from kernels.common.attention_bwd import lds_plan

    params = layout_twin_params(arch)
    assert params
    for name, cid in params:
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
