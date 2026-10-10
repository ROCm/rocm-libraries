# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``custom`` correctness configuration of ``tests/sdpa/bwd_kernel_cases.py``.

CPU only: parsing of ``ROCKE_BWD_CUSTOM_KNOBS``, rule matching per head size
and ``stage_vec``, configuration selection, and the plan a custom policy
resolves (the swept candidate's main kernel for matching requests, the arch
default for the others); the knob smoke sets of the device numerics plan on
every (arch, case) they run with every knob in the resolved spec, cover the
built-and-swept knob values that had no device run, and round-trip through
``ROCKE_BWD_CUSTOM_KNOBS``.
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
    KNOB_SMOKE_COMPILER_CRASHES,
    KNOB_SMOKE_SETS,
    SUPPLEMENTARY_CASES,
    config_knobs,
    config_policy as case_policy,
    custom_knobs_for,
    custom_rules,
    knob_smoke_params,
    knob_smoke_rules_json,
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


def test_knob_smoke_compiler_crashes_name_cells_the_device_test_runs():
    # every recorded toolchain crash skips real cells, and never a whole set
    sizes = dict(KNOB_SMOKE_CASES)
    for (name, arch, d), reason in KNOB_SMOKE_COMPILER_CRASHES.items():
        cells = [cid for n, cid in knob_smoke_params(arch) if n == name]
        assert any(sizes[cid] == d for cid in cells), (name, arch, d)
        assert any(sizes[cid] != d for cid in cells), (name, arch, d)
        assert "comgr" in reason
