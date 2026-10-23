# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the pruning set and of the pruning-set gate of the benchmark.

CPU: the composition of every class's pruning set, that every pruning case
plans to the very main kernel the timed cells of that class launch (so the
gate exercises the timed code object), and the gate's failure paths with a
stubbed device run. Device (gfx942 / gfx950 / gfx1151 / gfx1201): the gate on
the shipped configuration passes, and a corrupted run is rejected.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from benchmarks.common import attention_bwd_bench as bb
from benchmarks.common import attention_bwd_rocke_arm as ra
from benchmarks.common import attention_bwd_sweep as sw
from kernels.common.attention_bwd_plan import attn_bwd_plan

from . import _attention_bwd_perf_gate as pg
from ._attention_bwd_harness import ARCH, DEVICE_ARCHS
from .sdpa.bwd_kernel_cases import (
    PRUNE_HEAD_GROUPS,
    PRUNE_TAILS,
    config_policy,
    perf_prune_cases,
    request_from_case,
)
from .test_attention_bwd_rocke_arm import _tensors

CLASSES = [
    dict(mask_class=m, seq_mode=s, dkv_mode=k, g_split=g)
    for m in ("band", "none")
    for s in ("batched", "thd")
    for k, g in (("direct", 1), ("atomic", 1), ("atomic", 2))
]


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: "-".join(map(str, c.values())))
@pytest.mark.parametrize("d", [32, 128])
def test_pruning_set_composition(cls, d):
    cases = perf_prune_cases(head_size=d, dtype="fp16", **cls)
    assert 5 <= len(cases) <= 12
    lens = {n for c in cases for n in (c.lens()[0] + c.lens()[1])}
    assert set(PRUNE_TAILS) <= lens
    assert all(c.d == d and c.dtype == "fp16" for c in cases)
    assert any(c.h_q == 4 * c.h_k for c in cases)  # a GQA group of 4
    atomic = cls["dkv_mode"] == "atomic"
    assert any(c.h_k != c.h_v for c in cases) == atomic
    if atomic and cls["g_split"] == 1:
        assert all(c.h_k != c.h_v for c in cases)
    assert any(c.h_q == c.h_k for c in cases) == (cls["g_split"] == 1)
    if cls["mask_class"] == "band":
        assert any(c.fully_masked_rows for c in cases)
        assert any(c.left_bound >= 0 for c in cases)  # a window
        assert any(c.s_q != c.s_kv for c in cases)
    else:
        assert all(c.left_bound == c.right_bound == -1 for c in cases)
    thd = cls["seq_mode"] == "thd"
    assert all((c.length_mode == "ragged") == thd for c in cases)
    if thd:
        assert any(c.lens()[0][-1] == 0 and c.lens()[1][-1] == 0 for c in cases)
    with pytest.raises(ValueError):
        perf_prune_cases(head_size=d, g_split=2)  # a head split is atomic


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151"])
@pytest.mark.parametrize("cfg", ["default", "debug"])
def test_every_pruning_case_runs_the_timed_main_kernel(arch, cfg):
    """Per census class: the pruning cases plan to the main kernel that the
    census cells of the class launch under the same configuration."""
    from .sdpa.bwd_kernel_cases import config_knobs

    knobs = config_knobs(cfg, arch)
    seen = 0
    for cell in bb.census_cells(bb.load_census()):
        req = ra.request_for_cell(cell, _tensors(cell), scale=ra.scale_of(cell))
        rc = ra.RockeConfig("c", knobs)
        plan = attn_bwd_plan(req, arch, policy=ra.config_policy(rc))
        cases = pg.cases_for(plan)
        for case in cases:
            p = attn_bwd_plan(
                request_from_case(case), arch, policy=config_policy(cfg, arch)
            )
            assert p.specs["main"] == plan.specs["main"], (cell.id, case.id)
            assert {s.kernel for s in plan.launches} <= {
                s.kernel for c in cases for s in _plan(c, arch, cfg).launches
            }
        seen += 1
    assert seen == 84


def _plan(case, arch, cfg):
    return attn_bwd_plan(request_from_case(case), arch, policy=config_policy(cfg, arch))


def test_sweep_runtime_configs_keep_the_kernel():
    sl = sw.primary_slice("gfx942", "general", 128)
    c = sw.geometry_stage(sl, sw.Release(frozenset({1}))).candidates[0]
    x = c.with_runtime(g_split=2)
    plan_spec = sw.legality(x).resolved
    fake = type("P", (), {"specs": {"main": plan_spec}})()
    cases = pg.cases_for(fake, x.r)
    assert cases and all((c.h_q // c.h_k) % 2 == 0 for c in cases)
    for case in cases:
        p = attn_bwd_plan(request_from_case(case), "gfx942",
                          policy=config_policy("default", "gfx942",
                                               {**x.k, **x.r}))  # fmt: skip
        assert p.specs["main"] == plan_spec


def _census_splits() -> list[int]:
    """Every legal ``g_split`` of the census: the divisors of its head groups
    (the sweep's runtime stage enumerates exactly these)."""
    groups = {c.hq // c.hkv for c in bb.census_cells(bb.load_census())}
    return sorted({g for grp in groups for g in range(1, grp + 1) if grp % g == 0})


def test_census_head_splits_include_the_ones_beyond_the_gqa_ratio():
    # 5 and 8 do not divide the pruning set's grouped-query ratio of 4: they
    # need the rescaled cases (a split of 8 once had an empty pruning set)
    assert {5, 8} <= set(_census_splits())


@pytest.mark.parametrize("g_split", _census_splits())
@pytest.mark.parametrize("seq_mode", ["batched", "thd"])
@pytest.mark.parametrize("mask_class", ["band", "none"])
def test_every_legal_head_split_has_a_pruning_set(g_split, seq_mode, mask_class):
    """A non-empty pruning set for every legal ``g_split`` (the GQA cases of
    the set rescaled to a head group of exactly ``g_split`` where theirs does
    not take it), every case a head group the split divides, unique ids, the
    tails kept, a window where the class has one; the sets of ``g_split`` 1
    and 2 carry no rescaled case, every other split's GQA cases all run."""
    dkv = "atomic" if g_split > 1 else "direct"
    cases = perf_prune_cases(
        head_size=128, dtype="bf16", mask_class=mask_class, seq_mode=seq_mode,
        dkv_mode=dkv, g_split=g_split,
    )  # fmt: skip
    assert 5 <= len(cases) <= 12, g_split
    assert len({c.id for c in cases}) == len(cases)
    lens = {n for c in cases for n in (c.lens()[0] + c.lens()[1])}
    assert set(PRUNE_TAILS) <= lens
    if mask_class == "band":
        assert any(c.left_bound >= 0 for c in cases)
    for c in cases:
        group = math.gcd(c.h_q // c.h_k, c.h_q // c.h_v)  # G of the plan
        assert group % g_split == 0, (c.id, g_split)
    rescaled = [c for c in cases if "_hq" in c.id]
    assert bool(rescaled) == (g_split not in (1, 2))
    for c in rescaled:
        assert math.gcd(c.h_q // c.h_k, c.h_q // c.h_v) == g_split, c.id
    if g_split > 1:
        # h_k != h_v under every split (the head-split dK / dV sink with two
        # kv head counts)
        assert any(c.h_k != c.h_v for c in cases), g_split


def test_prune_head_groups_are_the_census_groups():
    """:data:`PRUNE_HEAD_GROUPS` is exactly the census's head groups (a new
    census group must get its pruning case)."""
    groups = {c.hq // c.hkv for c in bb.census_cells(bb.load_census())}
    assert set(PRUNE_HEAD_GROUPS) == groups


@pytest.mark.parametrize("seq_mode", ["batched", "thd"])
@pytest.mark.parametrize("mask_class", ["band", "none"])
@pytest.mark.parametrize("dkv", ["direct", "atomic"])
def test_every_census_head_group_runs_under_each_of_its_splits(
    seq_mode, mask_class, dkv
):
    """Every head group G of the census runs in the pruning set of every
    split of G (h40 / kv8 is G = 5: at g_split 1 and 5; G = 8 at 1, 2, 4 and
    8), as a case with ``h_q / h_k == G``."""
    for group in PRUNE_HEAD_GROUPS:
        for g in (s for s in range(1, group + 1) if group % s == 0):
            if dkv == "direct" and g > 1:
                continue  # a head split is atomic
            cases = perf_prune_cases(
                head_size=128, mask_class=mask_class, seq_mode=seq_mode,
                dkv_mode=dkv, g_split=g,
            )  # fmt: skip
            assert any(c.h_q // c.h_k == group for c in cases), (group, g)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("g_split", [s for s in _census_splits() if s > 1])
def test_head_split_pruning_cases_run_the_timed_main_kernel(arch, g_split):
    """``cases_for`` of a head-split configuration (the gate's own call) is
    non-empty for every legal split, and every case plans the timed main
    kernel (the class: atomic dK / dV of the split)."""
    sl = sw.primary_slice(arch, "general", 128)
    c = sw.geometry_stage(sl, sw.Release(frozenset({1}))).candidates[0]
    x = c.with_runtime(g_split=g_split)
    plan_spec = sw.legality(x).resolved
    fake = type("P", (), {"specs": {"main": plan_spec}})()
    cases = pg.cases_for(fake, x.r)
    assert cases, g_split
    for case in cases:
        p = attn_bwd_plan(
            request_from_case(case), arch,
            policy=config_policy("default", arch, {**x.k, **x.r}),
        )  # fmt: skip
        assert p.specs["main"] == plan_spec, case.id
        assert p.runtime["g_split"] == g_split, case.id


# --------------------------------------------------------------------------- failure paths
class _Run:
    def __init__(self, plan, case, bump=0.0, integrity=()):
        from ._attention_bwd_harness import reference

        _inp, ref = reference(case)
        self.plan, self.ref = plan, ref
        self.dq, self.dk, self.dv = ref.dq + bump, ref.dk, ref.dv
        self.integrity = list(integrity)


def _cfg_plan(arch="gfx942"):
    cells = sw.representative_cells(sw.primary_slice(arch, "general", 64))
    req = ra.request_for_cell(cells[0], _tensors(cells[0]), scale=0.125)
    return attn_bwd_plan(req, arch, policy=ra.config_policy(ra.SHIPPED))


def test_gate_failure_paths(monkeypatch, tmp_path):
    plan = _cfg_plan()
    cases = pg.cases_for(plan)
    gate = pg.PruneGate()

    def runner(bump=0.0, integrity=(), other_main=False):
        def run_case(case, *, arch, config, overrides, cache, **kw):
            p = _plan(case, arch, "default")
            if other_main:
                main = dataclasses.replace(p.specs["main"], edge_tiles=True)
                p = dataclasses.replace(p, specs={**p.specs, "main": main})
            return _Run(p, case, bump, integrity)

        return run_case

    monkeypatch.setattr(pg, "run_case", runner())
    assert gate.check("gfx942", ra.SHIPPED, plan, {}, tmp_path) == []
    monkeypatch.setattr(pg, "run_case", runner(bump=1.0))
    probs = gate.check("gfx942", ra.SHIPPED, plan, {}, tmp_path)
    assert len(probs) == len(cases) and all("dq" in p for p in probs)
    monkeypatch.setattr(pg, "run_case", runner(integrity=["dq: tail canary"]))
    assert gate.check("gfx942", ra.SHIPPED, plan, {}, tmp_path)
    monkeypatch.setattr(pg, "run_case", runner(other_main=True))
    probs = gate.check("gfx942", ra.SHIPPED, plan, {}, tmp_path)
    assert all(p.endswith("gate_kernel_mismatch") for p in probs)
    # a timed kernel no pruning case exercised fails the coverage check
    monkeypatch.setattr(pg, "run_case", runner())
    extra = dataclasses.replace(plan.launches[0], kernel=plan.launches[0].kernel + "_x")
    p2 = dataclasses.replace(plan, launches=plan.launches + (extra,))
    assert gate.check("gfx942", ra.SHIPPED, p2, {}, tmp_path)[0].startswith(
        "gate_coverage"
    )


@pytest.mark.parametrize(
    "runtime",
    [
        {"g_split": 1, "lb_order": "mirror"},
        {"g_split": 2, "lb_order": "mirror", "num_cus": ra.DEVICE_NUM_CUS},
        {"g_split": 1, "xcd_chunk": "gx", "num_xcds": 8},
        {"g_split": 1, "lb_order": "mirror", "xcd_chunk": "gx"},
    ],
    ids=["mirror", "mirror_sentinel_g2", "xcd", "mirror_xcd"],
)
def test_gate_plans_device_hinted_configs(runtime, monkeypatch, tmp_path):
    """A configuration timed with device hints (mirror pairing, the XCD remap)
    passes the gate's planning: the hints never reach the policy as knobs,
    every pruning case resolves the timed main kernel (pairing at
    ``num_cus = 1``, the remap at the timed XCD count), and a configuration
    without such levers plans its pruning cases with no hints, as before."""
    from types import SimpleNamespace

    from . import _attention_bwd_harness as h
    from .test_attention_bwd_rocke_arm import ARM_B_KNOBS, GFX950_NUM_CUS

    cells = {c.id: c for c in bb.census_cells(bb.load_census())}
    cell = cells["d128_h32_kv8_b2_q1024_k4096_causal_br_bf16_bshd"]
    req = ra.request_for_cell(cell, _tensors(cell), scale=ra.scale_of(cell))
    seen = []

    def run_case(case, *, arch, config, overrides, cache, request=None, **kw):
        seen.append((dict(overrides), request))
        p = h.plan_case(
            case, arch=arch, config=config, overrides=overrides, request=request
        )
        return SimpleNamespace(plan=p, dq=None, dk=None, dv=None, ref=None,
                               integrity=[])  # fmt: skip

    monkeypatch.setattr(pg, "run_case", run_case)
    monkeypatch.setattr(pg, "compare_bwd", lambda *a: [])
    for rt in (runtime, {"g_split": runtime["g_split"]}):
        seen.clear()
        cfg = ra.RockeConfig("c", ARM_B_KNOBS, rt)
        hints = ra.request_hints(cfg.runtime, "gfx950", 0, lambda i: GFX950_NUM_CUS)
        plan = attn_bwd_plan(
            dataclasses.replace(req, **hints), "gfx950", policy=ra.config_policy(cfg)
        )
        assert pg.gate.check("gfx950", cfg, plan, {}, tmp_path) == []
        assert seen and len(seen) == len(pg.cases_for(plan, rt))
        want = {}
        if plan.runtime["xcd_n"]:
            want["num_xcds"] = 8
        if plan.runtime["pair"]:
            want["num_cus"] = 1
        assert bool(want) == (rt is runtime)
        for overrides, request in seen:
            assert not set(overrides) & set(ra.REQUEST_HINTS)
            assert (request or {}) == want


# --------------------------------------------------------------------------- device hints x head splits
# The h40 / kv8 causal census cells (head group 5) and a head-group-8 cell,
# with the gfx950 knobs of the h40 confirm rounds (asm G1 / G3 route, DMA
# ring, edge tiles, a 4-wave 16x192 tile).
_H40 = "d128_h40_kv8_b1_q8192_k8192_causal_tl_bf16_bshd"
_H64 = "d128_h64_kv8_b4_q512_k2048_causal_br_bf16_bshd"
_H40_KNOBS = {
    "acc_in_lds": False, "atom_g02": "16x16x32", "atom_g13": "16x16x16",
    "atom_g4": "16x16x32", "block_k4": 32, "edge_tiles": True,
    "global_path": "dma", "kt_source": "reg", "kv_residency": "reg",
    "pt_route": "relabel", "ring_depth": 2, "s_in_agpr": False, "sched": "none",
    "setprio": False, "stats_fma": True, "stats_vec": 4,
    "transpose_source": "tr_read", "ws_layout": "head_major",
    "waves_per_eu": 1, "waves": 4, "block_m": 16, "block_n": 192,
    "warp_grid_g4": (1, 4), "warp_grid_g02": (1, 4), "warp_grid_g13": (4, 1),
    "agpr_route": "asm_g13", "agpr_alloc": (192, 192), "bwd_exp2": "fast",
}  # fmt: skip
# (extra knobs, runtime values): the device-hint configurations the sweep and
# the h40 rounds time. "arm_*": a hint the configuration leaves to the arm's
# default (the device's XCD / CU count).
_HINTED = {
    "none": ({}, {}),
    "xcd_gx": ({}, {"num_xcds": 8, "xcd_chunk": "gx"}),
    "xcd_gx_arm_xcds": ({}, {"xcd_chunk": "gx"}),
    "xcd_2gx": ({}, {"num_xcds": 8, "xcd_chunk": "2gx"}),
    "xcd_4gx": ({}, {"num_xcds": 8, "xcd_chunk": "4gx"}),
    "xcd_64": ({}, {"num_xcds": 8, "xcd_chunk": 64}),
    "cus_device": ({}, {"num_cus": ra.DEVICE_NUM_CUS}),
    "mirror": ({}, {"lb_order": "mirror", "num_cus": ra.DEVICE_NUM_CUS}),
    "mirror_arm_cus": ({}, {"lb_order": "mirror"}),
    "mirror_xcd": (
        {},
        {"lb_order": "mirror", "num_cus": ra.DEVICE_NUM_CUS, "num_xcds": 8,
         "xcd_chunk": "gx"},
    ),
    "mirror_xcd_arm": ({}, {"lb_order": "mirror", "xcd_chunk": "2gx"}),
    "interleave_xcd": ({}, {"lb_order": "interleave", "xcd_chunk": "4gx"}),
    "persistent": ({"persistent": True}, {}),
    "persistent_cus": ({"persistent": True}, {"num_cus": ra.DEVICE_NUM_CUS}),
    "split_dq_xcd": (
        {"dq_mode": "split"}, {"xcd_chunk": "2gx", "dq_xcd_chunk": "gx"}
    ),
    "split_dq_xcd_arm": ({"dq_mode": "split"}, {"dq_xcd_chunk": "4gx"}),
}  # fmt: skip
# Configurations every route must plan on the h40 cells (the XCD remap the
# h40 screen won with; mirror and persistent grids are route-specific).
_MUST_PLAN = ("none", "xcd_gx", "xcd_gx_arm_xcds", "xcd_2gx", "xcd_4gx", "xcd_64")


def _hinted_plan(cell_id, route, g_split, name, arch="gfx950"):
    """``(cfg, plan)`` of a hinted configuration on a census cell, planned as
    the rocKE arm plans it (its request hints, defaults included), or ``None``
    when the timed plan itself refuses it."""
    from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

    from .test_attention_bwd_rocke_arm import GFX950_NUM_CUS

    extra, rt = _HINTED[name]
    rt = dict(rt, g_split=g_split, **({"route": "dense"} if route == "dense" else {}))
    cfg = ra.RockeConfig(name, {**_H40_KNOBS, **extra}, rt)
    cell = {c.id: c for c in bb.census_cells(bb.load_census())}[cell_id]
    req = ra.request_for_cell(cell, _tensors(cell), scale=ra.scale_of(cell))
    hints = ra.request_hints(cfg.runtime, arch, 0, lambda i: GFX950_NUM_CUS)
    req = dataclasses.replace(req, **hints)
    policy = ra.config_policy(
        ra.RockeConfig(name, cfg.knobs, {k: v for k, v in rt.items() if k != "route"})
    )
    planner = attn_dense_bwd_plan if route == "dense" else attn_bwd_plan
    try:
        return cfg, planner(req, arch, policy=policy)
    except ValueError:
        return None


def _plan_only_gate(monkeypatch):
    """Stub the gate's device run: each pruning case is planned exactly as
    the gate plans it (general: the harness plan with the gate's overrides
    and request; dense: the gate's own dense plan), oracle and integrity
    checks pass. Returns the recorded ``(overrides, request)`` of the general
    runs."""
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    from . import _attention_bwd_harness as h
    from .test_attention_bwd_rocke_arm import GFX950_NUM_CUS

    seen = []

    def run_case(case, *, arch, config, cache, overrides=None, request=None,
                 mutate_plan=None, **kw):  # fmt: skip
        if mutate_plan is not None:
            p = mutate_plan(None)
        else:
            seen.append((dict(overrides or {}), dict(request or {})))
            p = h.plan_case(
                case, arch=arch, config=config, overrides=overrides, request=request
            )
        return SimpleNamespace(plan=p, dq=None, dk=None, dv=None, ref=None,
                               integrity=[])  # fmt: skip

    monkeypatch.setattr(pg, "run_case", run_case)
    monkeypatch.setattr(pg, "compare_bwd", lambda *a: [])
    monkeypatch.setattr(pg, "narrow_inputs", lambda case, view, base: (None, {}))
    monkeypatch.setattr(hip_module, "get_device_num_cus", lambda i=0: GFX950_NUM_CUS)
    return seen


@pytest.mark.parametrize("route", ["general", "dense"])
@pytest.mark.parametrize(
    "cell_id, g_split", [(_H40, 1), (_H40, 5), (_H64, 2), (_H64, 4), (_H64, 8)]
)
def test_gate_takes_every_device_hint_under_every_head_split(
    route, cell_id, g_split, monkeypatch, tmp_path
):
    """Every registered head split of the census's non-power-of-two (5) and
    largest (8) head groups crossed with the device-hint configurations (the
    XCD remap at gx / 2gx / 4gx / 64 with the XCD count named or the arm's,
    the device CU count, mirror pairing with and without the remap,
    persistent grids, the split dQ kernel's remap): wherever the timed plan
    plans, the gate has a non-empty pruning set, plans every case to the
    timed kernels, and never hands a hint to the policy as a knob; the
    general cases carry exactly the timed plan's hints (:func:`case_request`)."""
    seen = _plan_only_gate(monkeypatch)
    planned = set()
    for name in _HINTED:
        got = _hinted_plan(cell_id, route, g_split, name)
        if got is None:
            continue
        cfg, plan = got
        planned.add(name)
        assert pg.is_dense(plan) == (route == "dense")
        cases = pg.cases_for(plan, cfg.runtime)
        assert cases, name
        seen.clear()
        assert pg.gate.check("gfx950", cfg, plan, {}, tmp_path) == [], name
        want = pg.case_request(plan)
        if plan.runtime.get("xcd_n") or plan.runtime.get("dq_xcd_n"):
            assert want["num_xcds"] == 8, name
        if route == "general":
            assert len(seen) == len(cases), name
            for overrides, request in seen:
                assert not set(overrides) & set(ra.REQUEST_HINTS), name
                assert request == want, name
        else:
            values, _ = pg._split_runtime(cfg)
            assert not set(values) & set(ra.REQUEST_HINTS), name
            for case in cases:
                p = pg.dense_case_plan("gfx950", cfg, case, plan)
                assert pg.dense_identity(p.specs["dense"]) == pg.dense_identity(
                    plan.specs["dense"]
                ), (name, case.id)
    assert set(_MUST_PLAN) <= planned
    if route == "general" and g_split == 1 and cell_id == _H40:
        assert {"mirror", "mirror_xcd"} <= planned  # pairing is in the matrix


# --------------------------------------------------------------------------- h40 regressions
def test_h40_g5_config_has_a_pruning_set_and_passes(monkeypatch, tmp_path):
    """Regression (h40 / kv8, 2026-10-13): a ``g_split = 5`` configuration of
    the h40 cell had an empty pruning set (the set's GQA cases have head group
    4), so the bench reported ``prune: empty pruning set`` before any kernel
    ran. Both routes: a full set with group-5 cases, and the gate passes."""
    _plan_only_gate(monkeypatch)
    for route in ("general", "dense"):
        cfg, plan = _hinted_plan(_H40, route, 5, "xcd_gx")
        assert plan.runtime["g_split"] == 5
        cases = pg.cases_for(plan, cfg.runtime)
        assert len(cases) >= 4, route
        assert all(
            math.gcd(c.h_q // c.h_k, c.h_q // c.h_v) % 5 == 0 for c in cases
        ), route
        assert any(c.h_q // c.h_k == 5 for c in cases), route
        probs = pg.gate.check("gfx950", cfg, plan, {}, tmp_path)
        assert "prune: empty pruning set" not in probs
        assert probs == [], route


@pytest.mark.parametrize("g_split", [1, 5])
def test_h40_xcd_remap_config_is_not_dropped(g_split, monkeypatch, tmp_path):
    """Regression (h40 / kv8, 2026-10-13): the XCD-remap configurations
    (``xcd_chunk = gx``) won the h40 screen but the gate dropped them: the
    dense route re-planned the pruning cases with the configuration's own
    hints only, so a remap whose XCD count the arm supplied (the device
    default) failed every case with ``xcd_chunk needs num_xcds`` (and a
    persistent grid with ``needs num_persistent or num_cus``). The gate now
    takes the hints of the timed plan; both XCD forms pass, on both routes."""
    _plan_only_gate(monkeypatch)
    for route in ("general", "dense"):
        for name in ("xcd_gx", "xcd_gx_arm_xcds"):
            cfg, plan = _hinted_plan(_H40, route, g_split, name)
            assert plan.runtime["xcd_n"] == 8, (route, name)
            assert pg.gate.check("gfx950", cfg, plan, {}, tmp_path) == [], (
                route,
                name,
            )
    cfg, plan = _hinted_plan(_H40, "dense", g_split, "persistent")
    assert plan.runtime["num_persistent"] > 0
    assert pg.gate.check("gfx950", cfg, plan, {}, tmp_path) == []


def test_prepare_stores_the_oracles(tmp_path):
    plan = _cfg_plan()
    gate = pg.PruneGate()
    n = gate.prepare([(ra.SHIPPED, plan)], tmp_path)
    assert n == len(pg.cases_for(plan))
    assert gate.prepare([(ra.SHIPPED, plan)], tmp_path) == 0
    assert len(list(tmp_path.glob("prune-*.pkl"))) == n


# --------------------------------------------------------------------------- dense plans
# The best gfx950 general configuration on the dense route (DMA ring, stats
# levers) and the register ring on gfx942.
_DENSE_KNOBS = {
    "gfx950": {
        "atom_g02": "16x16x32", "atom_g13": "16x16x16", "atom_g4": "16x16x32",
        "block_k4": 32, "block_m": 32, "block_n": 128, "pt_route": "relabel",
        "transpose_source": "tr_read", "warp_grid_g02": (1, 8),
        "warp_grid_g13": (8, 1), "warp_grid_g4": (2, 4), "waves": 8,
        "waves_per_eu": 1, "stats_vec": 4, "stats_fma": True,
        "global_path": "dma", "ring_depth": 2, "head_pack": False,
        "edge_tiles": False,
    },
    "gfx942": {"ring_depth": 2, "head_pack": False},
}  # fmt: skip


# split dQ on the dense route (the dense dQ kernel bakes the shape too)
_SPLIT = {"dq_mode": "split", "dq_edge_tiles": True}


def _dense_cfg(arch, g_split=1, split=False):
    rt = {"route": "dense", "g_split": g_split}
    knobs = dict(_DENSE_KNOBS[arch], **(_SPLIT if split else {}))
    return ra.RockeConfig("dense", knobs, rt)


def _dense_cfg_plan(arch, g_split=1, split=False):
    """A dense plan of the d128 slice's first representative cell."""
    sl = sw.primary_slice(arch, "dense", 128)
    cell = sw.dense_cells(sl, sw.representative_cells(sl), 128)[0]
    cfg = _dense_cfg(arch, g_split, split)
    c = sw.Candidate.make(sl, dict(cfg.knobs), dict(cfg.runtime))
    return cfg, sw.dense_plan(c, cell)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("g_split", [1, 2])
@pytest.mark.parametrize("split", [False, True])
def test_dense_pruning_cases_rerun_the_timed_configuration(arch, g_split, split):
    """Dense plans: the pruning set is the dense spec's batched class, every
    case is dense-eligible and plans a dense spec (and, under split dQ, a
    dense dQ spec) equal to the timed one in every non-problem field, and the
    cases exercise every aux kernel the timed plan launches."""
    cfg, plan = _dense_cfg_plan(arch, g_split, split)
    assert pg.is_dense(plan)
    assert ("dense_dq" in plan.specs) == split
    cases = pg.cases_for(plan, cfg.runtime)
    assert len(cases) >= 4
    assert all(c.length_mode != "ragged" and c.d == 128 for c in cases)
    spec = plan.specs["dense"]
    assert (spec.dkv_mode == "atomic") == (g_split > 1)
    want = pg.dense_identity(spec)
    want_dq = pg.dense_dq_identity(plan.specs.get("dense_dq"))
    assert (want_dq is None) == (not split)
    seen = set()
    for case in cases:
        p = pg.dense_case_plan(arch, cfg, case)
        assert pg.dense_identity(p.specs["dense"]) == want, case.id
        assert pg.dense_dq_identity(p.specs.get("dense_dq")) == want_dq, case.id
        seen |= {s.kernel for s in p.launches}
    aux = {s.kernel for s in plan.launches if s.stage not in ("dense", "dense_dq")}
    assert aux <= seen
    # the general class of the same knobs is not the dense one (no KeyError)
    assert pg.cases_for(_cfg_plan(arch)) != ()


def test_dense_gate_failure_paths(monkeypatch, tmp_path):
    cfg, plan = _dense_cfg_plan("gfx950")
    cases = pg.cases_for(plan, cfg.runtime)
    gate = pg.PruneGate()

    def runner(bump=0.0, integrity=(), other=False, general=False):
        def run_case(case, *, arch, config, cache, mutate_plan=None, **kw):
            assert config == "default" and mutate_plan is not None
            p = mutate_plan(None)
            if other:
                spec = dataclasses.replace(p.specs["dense"], epilogue="direct")
                p = dataclasses.replace(p, specs={**p.specs, "dense": spec})
            if general:
                p = _plan(case, arch, "default")
            return _Run(p, case, bump, integrity)

        return run_case

    monkeypatch.setattr(pg, "run_case", runner())
    assert gate.check("gfx950", cfg, plan, {}, tmp_path) == []
    monkeypatch.setattr(pg, "run_case", runner(bump=1.0))
    probs = gate.check("gfx950", cfg, plan, {}, tmp_path)
    assert len(probs) == len(cases) and all("dq" in p for p in probs)
    monkeypatch.setattr(pg, "run_case", runner(integrity=["dk: tail canary"]))
    assert gate.check("gfx950", cfg, plan, {}, tmp_path)
    monkeypatch.setattr(pg, "run_case", runner(other=True))
    probs = gate.check("gfx950", cfg, plan, {}, tmp_path)
    assert probs and all(p.endswith("gate_kernel_mismatch") for p in probs)
    monkeypatch.setattr(pg, "run_case", runner(general=True))
    probs = gate.check("gfx950", cfg, plan, {}, tmp_path)
    assert probs and all(p.endswith("gate_kernel_mismatch") for p in probs)
    # a timed aux kernel no case exercised fails the coverage check
    monkeypatch.setattr(pg, "run_case", runner())
    extra = dataclasses.replace(plan.launches[0], kernel=plan.launches[0].kernel + "_x")
    p2 = dataclasses.replace(plan, launches=plan.launches + (extra,))
    assert gate.check("gfx950", cfg, p2, {}, tmp_path)[0].startswith("gate_coverage")
    # a configuration the dense route refuses on the pruning shapes is a
    # gate failure, not a crash
    bad = ra.RockeConfig("x", {**cfg.knobs, "head_pack": True}, cfg.runtime)
    probs = gate.check("gfx950", bad, plan, {}, tmp_path)
    assert len(probs) == len(cases) and all("head_pack" in p for p in probs)


def test_dense_split_gate_checks_the_dense_dq_spec(monkeypatch, tmp_path):
    """Split dQ: a pruning case that resolves another dense dQ spec than the
    timed one (or none) is a kernel mismatch; the timed dense dQ kernel
    (shape-baked) is not an aux kernel the coverage check asks for."""
    cfg, plan = _dense_cfg_plan("gfx950", split=True)
    gate = pg.PruneGate()

    def runner(mutate=None):
        def run_case(case, *, arch, config, cache, mutate_plan=None, **kw):
            p = mutate_plan(None)
            if mutate is not None:
                p = mutate(p)
            return _Run(p, case, 0.0, ())

        return run_case

    monkeypatch.setattr(pg, "run_case", runner())
    assert gate.check("gfx950", cfg, plan, {}, tmp_path) == []

    def other_dq(p):
        dq = p.specs["dense_dq"]
        dq = dataclasses.replace(dq, dq=dataclasses.replace(dq.dq, kv_prefetch=True))
        return dataclasses.replace(p, specs={**p.specs, "dense_dq": dq})

    def no_dq(p):
        return dataclasses.replace(
            p, specs={k: v for k, v in p.specs.items() if k != "dense_dq"}
        )

    for mutate in (other_dq, no_dq):
        monkeypatch.setattr(pg, "run_case", runner(mutate))
        probs = gate.check("gfx950", cfg, plan, {}, tmp_path)
        assert probs and all(p.endswith("gate_kernel_mismatch") for p in probs)


def test_dense_prepare_compiles_the_pruning_kernels(monkeypatch, tmp_path):
    """prepare stores the oracles and compiles every dense pruning kernel
    into the job's gate code-object cache (unless one is configured)."""
    import os

    from kernels.common import attention_bwd_run

    cfg, plan = _dense_cfg_plan("gfx942")
    seen = []

    def fake(p, *, cache=None):
        seen.append((p.specs["dense"].kernel_name(), os.environ.get(pg._HSACO_ENV)))
        return {}

    monkeypatch.setattr(attention_bwd_run, "attn_bwd_kernels", fake)
    monkeypatch.delenv(pg._HSACO_ENV, raising=False)
    n = pg.PruneGate().prepare([(cfg, plan)], tmp_path)
    cases = pg.cases_for(plan, cfg.runtime)
    assert n == len(cases)
    assert len(seen) == len(cases)
    assert {e for _, e in seen} == {str(tmp_path / "gate-hsaco")}
    assert pg._HSACO_ENV not in os.environ
    seen.clear()
    monkeypatch.setenv(pg._HSACO_ENV, str(tmp_path / "shared"))
    pg.PruneGate().prepare([(cfg, plan)], tmp_path)
    assert {e for _, e in seen} == {str(tmp_path / "shared")}


# --------------------------------------------------------------------------- device
@pytest.mark.gpu
@pytest.mark.skipif(ARCH not in DEVICE_ARCHS, reason=f"no backward device ({ARCH})")
def test_gate_on_device_passes_the_shipped_config_and_rejects_a_bad_kernel(
    tmp_path, monkeypatch
):
    from . import _attention_bwd_harness as h

    arch = ARCH if ARCH != "gfx1201" else "gfx1151"
    sl = sw.primary_slice(arch, "general", 64)
    cell = sw.representative_cells(sl)[0]
    small = dataclasses.replace(cell, b=1, sq=64, skv=64, hq=8, hkv=2)
    req = ra.request_for_cell(small, _tensors(small), scale=0.125)
    plan = attn_bwd_plan(req, arch, policy=ra.config_policy(ra.SHIPPED))
    if ARCH == "gfx1201":
        plan = dataclasses.replace(plan, arch="gfx1201")
    gate = pg.PruneGate()
    gate.prepare([(ra.SHIPPED, plan)], tmp_path)
    cache: dict = {}
    assert gate.check(ARCH, ra.SHIPPED, plan, cache, tmp_path) == []
    # negative control: the main kernel launched with a wrong scale kernarg
    real = h.run_case

    def bad(case, **kw):
        def mutate(p):
            return h._replace_launch(p, "main", scalars={"ds_mult": 0.5})

        return real(case, mutate_plan=mutate, **kw)

    monkeypatch.setattr(pg, "run_case", bad)
    assert gate.check(ARCH, ra.SHIPPED, plan, cache, tmp_path)


@pytest.mark.gpu
@pytest.mark.skipif(
    ARCH not in ("gfx942", "gfx950"), reason=f"no dense backward device ({ARCH})"
)
@pytest.mark.parametrize("g_split", [1, 2])
@pytest.mark.parametrize("split", [False, True])
def test_dense_gate_on_device_passes_the_ring_config_and_rejects_a_bad_kernel(
    tmp_path, monkeypatch, g_split, split
):
    from . import _attention_bwd_harness as h

    monkeypatch.delenv(pg._HSACO_ENV, raising=False)
    cfg, plan = _dense_cfg_plan(ARCH, g_split, split)
    gate = pg.PruneGate()
    gate.prepare([(cfg, plan)], tmp_path)
    assert list((tmp_path / "gate-hsaco").iterdir())  # the kernels were cached
    cache: dict = {}
    assert gate.check(ARCH, cfg, plan, cache, tmp_path) == []
    # negative controls: the dense kernel (or, under split, the dense dQ
    # kernel) launched with a wrong scale kernarg
    real = h.run_case
    for stage in ("dense", "dense_dq") if split else ("dense",):

        def bad(case, mutate_plan=None, stage=stage, **kw):
            def mutate(p):
                return h._replace_launch(
                    mutate_plan(p), stage, scalars={"ds_mult": 0.5}
                )

            return real(case, mutate_plan=mutate, **kw)

        monkeypatch.setattr(pg, "run_case", bad)
        assert gate.check(ARCH, cfg, plan, cache, tmp_path), stage
