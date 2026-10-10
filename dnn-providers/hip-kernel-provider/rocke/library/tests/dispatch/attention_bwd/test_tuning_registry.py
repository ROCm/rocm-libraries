# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The backward tuning registry (dispatch/attention_bwd/tuning_common.py).

* every ``AttnBwdSpec`` tuning field and every plan runtime / aux value is a
  registered axis, with ``choices[0]`` the default;
* every registered value is legal on some arch (the family validator plus the
  emission check), or marked not built and refused by the kernel or the plan,
  or gated (legal, never swept);
* the sweep driver's catalog and lever families are the registry's, and each
  registered knob is enumerated by the funnel stage that owns it.

CPU only.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import pytest

from benchmarks.common import attention_bwd_sweep as sw
from dispatch.attention_bwd import tuning_common as tc
from kernels.common.attention_bwd import (
    AGPR_ALLOCS,
    BLOCK_K4_VALUES,
    BLOCK_M_VALUES,
    BLOCK_N_VALUES,
    BWD_ARCHS,
    NOT_YET_EFFECTIVE_KNOBS,
    PT_ROUTES,
    WS_LAYOUTS,
    AttnBwdSpec,
    _not_built,
    attn_bwd_arch_facts,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_plan import (
    AUX_KNOBS,
    FEATURE_GROUPS,
    RUNTIME_KNOBS,
    AttnBwdPolicy,
    AttnBwdRequest,
    attn_bwd_plan,
)

FULL = AttnBwdPolicy(declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True)
SPEC_FIELDS = {f.name: f for f in dataclasses.fields(AttnBwdSpec)}
COMPILE = [a for a in tc.BWD_KNOB_AXES if a.kind == tc.KIND_COMPILE]
RUNTIME = [a for a in tc.BWD_KNOB_AXES if a.kind == tc.KIND_RUNTIME]


def _req(**kw):
    base = dict(b=2, h_q=8, h_k=8, h_v=8, s_q=128, s_kv=128, d_qk=64, d_v=64)
    base.update(kw)
    return AttnBwdRequest(**base)


def _plan(tuned: dict, arch="gfx942", **req):
    policy = dataclasses.replace(FULL, tuning=lambda *a: dict(tuned))
    return attn_bwd_plan(_req(**req), arch, policy=policy)


# --------------------------------------------------------------------------- coverage
def test_every_tuning_field_and_runtime_knob_is_registered():
    names = [a.name for a in tc.BWD_KNOB_AXES]
    assert len(names) == len(set(names))
    tuning_fields = set(SPEC_FIELDS) - tc.PROBLEM_FIELDS
    assert {a.name for a in COMPILE} == tuning_fields
    assert {a.name for a in RUNTIME} == set(RUNTIME_KNOBS) | set(AUX_KNOBS)
    # the problem fields are not tuning decisions; only head_size is required
    assert tc.PROBLEM_FIELDS <= set(SPEC_FIELDS)
    for a in tc.BWD_KNOB_AXES:
        assert a.family in tc.LEVER_FAMILIES and a.stage in tc.FUNNEL_STAGES
        assert a.level in tc.SWEEP_LEVELS
        assert a.numerics in tc.NUMERICS
        assert set(a.not_built) | set(a.full_only) | set(a.gated) <= set(a.choices)
        assert a.default not in set(a.not_built) | set(a.gated) | set(a.full_only)
        assert tc.backward_axis(a.name) is a
    with pytest.raises(ValueError):
        tc.backward_axis("no_such_knob")


def test_family_and_stage_agree():
    """A lever family is released into the funnel stages that own it."""
    stages_of = {
        1: {"geometry"},
        2: {"structure", "schedule", "codegen"},
        3: {"structure"},
        4: {"structure"},
        5: {"schedule"},
        6: {"codegen", "swizzle"},
        7: {"runtime"},
        8: {"structure"},
    }
    for a in tc.BWD_KNOB_AXES:
        assert a.stage in stages_of[a.family], (a.name, a.family, a.stage)
        # family 7 is the runtime values plus the compile fields the plan
        # derives from one of them (plan-owned, never enumerated on their own)
        runtime_like = a.kind == tc.KIND_RUNTIME or a.plan_owned and a.family == 7
        assert runtime_like == (a.family == 7), a.name


def test_choices0_is_the_spec_default():
    for a in COMPILE:
        assert a.default == SPEC_FIELDS[a.name].default, a.name
    # spelled-out defaults are the default spec (same name, no salt)
    explicit = AttnBwdSpec(head_size=64, **{a.name: a.default for a in COMPILE})
    assert explicit == AttnBwdSpec(head_size=64)
    assert explicit.non_default_fields() == ()


def test_choices0_is_the_plan_default():
    plain = _plan({})
    for a in RUNTIME:
        if a.name in ("g_split", "lb_order", "scale_placement") + tuple(AUX_KNOBS):
            assert plain.runtime[a.name] == a.default, a.name
    assert plain.runtime["xcd_chunk"] == tc.backward_axis("xcd_chunk").default
    assert plain.runtime["use_worklist"] == tc.backward_axis("use_worklist").default
    # passing every default explicitly plans the same instances and launches
    explicit = _plan(
        {a.name: a.default for a in RUNTIME if a.name not in ("xcd_chunk",)}
    )
    assert explicit.specs == plain.specs
    assert explicit.runtime == plain.runtime
    assert len(explicit.launches) == len(plain.launches)


def test_numerics_classes():
    by_class = {}
    for a in tc.BWD_KNOB_AXES:
        assert a.numerics in tc.NUMERICS
        by_class.setdefault(a.numerics, set()).add(a.name)
    # another fp32 order or rounding point: the step order, the head split and
    # the scale placement, LDS accumulators and packed heads
    assert by_class["reorders_fp"] == {
        "step_order",
        "q_rotate",
        "g_split",
        "scale_placement",
        "acc_in_lds",
        "head_pack",
    }
    # the schedule, epilogue, LDS layout and aux knobs are bit for bit
    added = {
        "asm_tail_nop",
        "asm_cluster",
        "stage_order",
        "loop_sync",
        "qdo_stage",
        "stats_load",
        "dkv_epilogue",
        "dkv_store_vec",
        "dkv_atomic_guard",
        "lds_order",
        "prep_rows_per_cta",
        "convert_rows_per_cta",
        "convert_vec",
        "static_scale",
        "dq_issue",
        "ptr_noalias",
        "lb_interleave",
        "xcd_remap",
        "dkv_slab_hoist",
        "interior_clamp",
    }
    assert added <= by_class["bitwise"]
    assert {"waves", "block_m", "atom_g13", "pt_route"} <= by_class["tile"]
    assert by_class["n/a"] == {"dq_mode", "use_worklist"}


# --------------------------------------------------------------------------- legality
@cache
def _pool(arch: str, d: int) -> tuple:
    """Knob sets of every legal geometry-stage candidate of a slice."""
    sl = sw.primary_slice(arch, "general", d)
    return tuple(dict(c.k) for c in sw.geometry_stage(sl, sw.Release()).candidates)


_ROUTE = {"agpr_alloc": (128, 128)}
_ASM = {"agpr_route": "asm_g13", "agpr_alloc": (128, 128)}
# Couplings a value needs (tried before the geometry pools).
_CONTEXTS = {
    "agpr_route": ({}, _ROUTE, {"agpr_alloc": (192, 192)}),
    "asm_tail_nop": (_ASM, {**_ASM, "agpr_alloc": (192, 192)}),
    "asm_cluster": (_ASM, {**_ASM, "agpr_alloc": (192, 192)}),
    "s_in_agpr": ({}, _ROUTE, {"agpr_alloc": "backend"}),
    "warp_grid_g02": ({}, {"pt_route": "lds"}),
    "warp_grid_g13": ({}, {"pt_route": "lds"}),
    "acc_in_lds": ({}, {"agpr_alloc": (0, 0)}),
    "dkv_atomic_guard": ({}, {"dkv_mode": "atomic"}),
    "ptr_noalias": ({}, {"dkv_mode": "atomic"}),
    # the K-permuted G1 / G3 operand is not built for the transpose read
    "atom_g13": ({}, {"transpose_source": "lds_plain"}, {"pt_route": "lds"}),
}


def _value_knobs(name, value) -> dict:
    if name == "lds_swizzle":
        return {} if value is None else {"lds_swizzle": f"q={value}"}
    return {name: value}


def _legal_somewhere(name, value) -> str | None:
    """An arch where ``name = value`` is legal and built, or None."""
    for ctx in _CONTEXTS.get(name, ({},)):
        for use_pool in (False, True):
            for arch in BWD_ARCHS:
                for d in (64, 128, 32):
                    bases = _pool(arch, d) if use_pool else ({},)
                    for base in bases:
                        knobs = {**base, **ctx, **_value_knobs(name, value)}
                        mode = knobs.pop("dkv_mode", "direct")
                        v = tc.backward_spec_legality(
                            knobs, arch, head_size=d, dtype="bf16", dkv_mode=mode
                        )
                        if v.ok:
                            return arch
    return None


def _refused_everywhere(name, value) -> bool:
    """``name = value`` is never legal and built, and the refusal is the
    spec's "not built" rule or the emission check (``_not_built``) on at least
    one arch where the validator accepts it."""
    ctxs = {
        "setprio": {"sched": "stage_table"},
        "global_path": {"transpose_source": "tr_read"},
    }
    knobs = {**ctxs.get(name, {}), **_value_knobs(name, value)}
    try:
        spec = AttnBwdSpec(head_size=64, dtype="bf16", **knobs)
    except ValueError as ex:
        return "not built" in str(ex)
    statuses = [tc.backward_spec_legality(spec, a).status for a in BWD_ARCHS]
    return "legal" not in statuses and "not_built" in statuses


@pytest.mark.parametrize("name", [a.name for a in COMPILE])
def test_every_registered_value_is_legal_or_not_built(name):
    axis = tc.backward_axis(name)
    for value in axis.choices:
        if value in axis.not_built:
            assert _refused_everywhere(name, value), (name, value)
            continue
        assert _legal_somewhere(name, value), (name, value)


def test_not_built_values_are_the_emission_checks():
    """Each kernel value the registry marks not built is one the emission
    check (``_not_built``) or the spec refuses, and the reverse: a default
    spec with any other registered value passes the emission check."""
    base = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dtype="bf16"), "gfx950")
    for a in COMPILE:
        for v in a.not_built:
            if a.name == "dq_mode":
                with pytest.raises(ValueError, match="not built"):
                    AttnBwdSpec(head_size=64, dq_mode=v)
                continue
            if a.name == "lds_swizzle":
                spec = dataclasses.replace(base, lds_swizzle=f"q={v}")
            elif a.name == "setprio":
                spec = dataclasses.replace(base, sched="stage_table", setprio=v)
            else:
                spec = dataclasses.replace(base, **{a.name: v})
            assert _not_built(spec), (a.name, v)


@pytest.mark.parametrize("name", [a.name for a in RUNTIME])
def test_runtime_values_are_built_or_refused_by_the_plan(name):
    axis = tc.backward_axis(name)
    for value in axis.choices:
        if value in axis.not_built:
            with pytest.raises(ValueError):
                _plan({name: value}, causal=True, num_cus=4)
            continue
        if name == "xcd_chunk":
            continue  # needs num_xcds; see test_xcd_chunk_builds_the_remap
        plan = _plan({name: value})
        assert plan.runtime[name] == value


def test_xcd_chunk_builds_the_remap():
    """Every registered xcd_chunk (symbolic multiples of the main grid x
    included) plans with the request's num_xcds; a non-zero chunk compiles
    the xcd_remap twin of the default main kernel."""
    axis = tc.backward_axis("xcd_chunk")
    assert not axis.not_built
    base = _plan({}, num_xcds=8)
    gx = base.launches[1].grid[0]
    want = {0: 0, "gx": gx, "2gx": 2 * gx, "4gx": 4 * gx, 64: 64}
    for value in axis.choices:
        plan = _plan({"xcd_chunk": value}, num_xcds=8)
        assert plan.runtime["xcd_chunk"] == want[value]
        spec = plan.specs["main"]
        assert spec.xcd_remap is bool(value)
        assert dataclasses.replace(spec, xcd_remap=False) == base.specs["main"]
        assert plan.runtime["xcd_n"] == (8 if value else 0)
    with pytest.raises(ValueError, match="num_xcds"):
        _plan({"xcd_chunk": "gx"})


def test_tail_nops_start_at_the_isa_floor():
    """asm_tail_nop: the registered counts start at the smallest per-target
    ISA floor; each target sweeps its floor up (production: the floor plus
    the probe's picks; full: every legal count), and the validator refuses a
    count below the target's floor."""
    from kernels.common._attention_bwd_caps import asm_tail_nop_min

    axis = tc.backward_axis("asm_tail_nop")
    assert dict(tc.ASM_TAIL_NOP_FLOOR) == {"gfx942": 6, "gfx950": 7}
    assert tc.ASM_TAIL_NOP_FLOOR["gfx942"] == asm_tail_nop_min("gfx942", "bf16", 16)
    assert not axis.gated and min(axis.choices[1:]) == 6
    assert axis.values("gfx942", "production") == (None, 6, 7, 8, 11)
    assert axis.values("gfx950", "production") == (None, 7, 8, 11)
    assert axis.values("gfx950", "full") == (None,) + tuple(range(7, 15))
    assert axis.values("gfx1151", "full") == (None,)
    for arch, floor in tc.ASM_TAIL_NOP_FLOOR.items():
        for n, ok in ((floor - 1, False), (floor, True)):
            v = tc.backward_spec_legality(
                {"agpr_route": "asm_g13", "agpr_alloc": (128, 128),
                 "asm_tail_nop": n},
                arch, head_size=64, dtype="bf16",
            )  # fmt: skip
            assert v.ok is ok, (arch, n, v.error)
            if not ok:
                assert "ISA" in v.error


def test_legality_helper_is_the_validator_plus_the_emission_check():
    for arch in BWD_ARCHS:
        spec = AttnBwdSpec(head_size=128, dtype="bf16")
        v = tc.backward_spec_legality(spec, arch)
        assert v.ok and v.status == "legal"
        assert v.resolved == validate_attn_bwd_spec(spec, arch)
    nb = tc.backward_spec_legality(
        {"transpose_source": "xt_lds"}, "gfx942", head_size=64
    )
    assert nb.status == "not_built" and nb.not_built
    bad = tc.backward_spec_legality({"waves": 8}, "gfx1151", head_size=64)
    assert bad.status == "illegal" and "waves" in bad.error
    refused = tc.backward_spec_legality({"loop_sync": "x"}, "gfx942", head_size=64)
    assert refused.status == "illegal" and refused.resolved is None
    with pytest.raises(ValueError):
        tc.backward_spec_legality({}, "gfx942")
    with pytest.raises(ValueError):
        tc.backward_tuning_axes("gfx90a")


def test_values_follow_the_arch_and_the_level():
    for arch in BWD_ARCHS:
        mfma = attn_bwd_arch_facts(arch).matrix_path == "mfma"
        assert tc.backward_axis("agpr_route").values(arch) == (
            tc.backward_axis("agpr_route").choices if mfma else (None,)
        )
        so = tc.backward_axis("stage_order")
        assert so.values(arch, "full") == so.choices
        assert so.values(arch, "production") == so.choices[:3]
    with pytest.raises(ValueError):
        tc.backward_axis("waves").values("gfx942", "fast")


# --------------------------------------------------------------------------- sweep
_LEGACY_SCHED_PAIRS = (
    ("none", False),
    ("iglp0", False),
    ("iglp1", False),
    ("fences_only", False),
    ("stage_table", False),
    ("stage_table", True),
    ("stage_table_atomics", False),
    ("stage_table_atomics", True),
)


def _legacy_catalog(arch):
    """The sweep catalog as it was hard-coded before the registry."""
    f = attn_bwd_arch_facts(arch)
    mfma = f.matrix_path == "mfma"
    wpe = (None, 1, 2) + ((3,) if arch == "gfx942" else ())
    return {
        "waves": f.legal_waves,
        "block_n": BLOCK_N_VALUES,
        "block_m": BLOCK_M_VALUES,
        "block_k4": BLOCK_K4_VALUES,
        "atom_g02": f.legal_atoms,
        "atom_g13": f.legal_atoms,
        "atom_g4": f.legal_atoms,
        "pt_route": PT_ROUTES if mfma else ("lds",),
        "residency": (
            (("reg", "reg"), ("reg", "lds"), ("lds", "lds"))
            if mfma
            else (("lds", "lds"),)
        ),
        "transpose_source": f.legal_transpose_sources,
        "global_path": f.legal_global_paths,
        "ring_depth": f.legal_ring_depths,
        "s_in_agpr": (False, True) if mfma else (False,),
        "sched_pairs": _LEGACY_SCHED_PAIRS,
        "waves_per_eu": wpe if mfma else (None, 1, 2),
        "agpr_alloc": AGPR_ALLOCS if mfma else (None,),
        "acc_in_lds": (False, True) if mfma else (True,),
        "scheduler_strategy": sw.CODEGEN_STRATEGIES,
        "edge_tiles": (False, True),
        "ws_layout": WS_LAYOUTS,
    }


@pytest.mark.parametrize("arch", BWD_ARCHS)
@pytest.mark.parametrize("level", tc.SWEEP_LEVELS)
def test_sweep_catalog_is_the_registry(arch, level):
    cat = sw.catalog(arch, level)
    for knob, vals in _legacy_catalog(arch).items():
        assert tuple(cat[knob]) == tuple(vals), (arch, knob)
    assert sw.SCHED_PAIRS == _LEGACY_SCHED_PAIRS
    for a in tc.backward_tuning_axes(arch, level):
        assert cat[a.name] == a.values(arch, level)


def test_sweep_families_are_the_registry():
    assert sw.KNOB_FAMILY == dict(tc.knob_family())
    assert sw.LEVER_FAMILIES is tc.LEVER_FAMILIES
    assert sw.CODEGEN_STRATEGIES is tc.CODEGEN_STRATEGIES


def test_every_knob_is_enumerated_by_its_stage():
    owners = {
        "geometry": set(sw.GEOMETRY_KNOBS),
        "structure": set(sw.STRUCTURE_KNOBS) | set(sw.STRUCTURE_EXTRA),
        "schedule": set(sw.SCHEDULE_KNOBS) | {"agpr_route"} | set(sw.SCHEDULE_EXTRA),
        "codegen": set(sw.MINOR_KNOBS) | set(sw.CODEGEN_EXTRA),
        "swizzle": {"lds_swizzle"} | set(sw.SWIZZLE_EXTRA),
        "runtime": {"g_split", "lb_order", "scale_placement", "xcd_chunk"}
        | set(sw.RUNTIME_EXTRA),
    }
    pinned = {"use_worklist"}  # every non-default value not built
    for a in tc.BWD_KNOB_AXES:
        if a.plan_owned or a.name in NOT_YET_EFFECTIVE_KNOBS:
            assert not any(a.name in s for s in owners.values()) or a.name in (
                sw.GEOMETRY_KNOBS
            )
            continue
        if a.name in pinned:
            assert set(a.choices) - set(a.not_built) == {a.default}
            continue
        assert a.name in owners[a.stage], (a.name, a.stage)
        others = [s for s, names in owners.items() if a.name in names and s != a.stage]
        assert not others, (a.name, others)


def _start(arch="gfx950", d=64, **knobs):
    sl = sw.primary_slice(arch, "general", d)
    return sw.Candidate.make(sl, {**sw.default_knobs(sl), **knobs})


ALL_PROD = sw.Release(sw.ALL_RELEASED, "production")
ALL_FULL = sw.Release(sw.ALL_RELEASED, "full")


def _values(points, name, default):
    return {p.get(name, default) for p in points}


def test_stages_enumerate_the_new_knobs():
    c = _start()
    h = sw.schedule_values(c, ALL_PROD)
    assert _values(h, "stage_order", "g1g3g4") == {"g1g3g4", "g3g1g4", "g4g1g3"}
    assert _values(h, "loop_sync", "full") == {"full", "lds_only"}
    assert _values(h, "qdo_stage", "split") == {"split", "loads_first", "joint"}
    assert _values(h, "stats_load", "at_use") == {"at_use", "step_top"}
    assert _values(h, "agpr_route", None) == set(tc.backward_axis("agpr_route").choices)
    asm = [p for p in h if p.get("agpr_route") == "asm_g13"]
    assert _values(asm, "asm_cluster", "k_step") == {"k_step", "gemm"}
    assert all("asm_cluster" not in p for p in h if p.get("agpr_route") != "asm_g13")
    # the ISA floor (7 on gfx950) and the production picks, asm route only
    assert _values(asm, "asm_tail_nop", None) == {None, 7, 8, 11}
    assert all("asm_tail_nop" not in p for p in h if p.get("agpr_route") != "asm_g13")
    # production: one registry knob moved per point
    extra = set(sw.SCHEDULE_EXTRA)
    assert all(len(extra & set(p)) <= 1 for p in h)
    full = sw.schedule_values(c, ALL_FULL)
    assert _values(full, "stage_order", "g1g3g4") == set(
        tc.backward_axis("stage_order").choices
    )
    assert len(full) > len(h)
    k = sw.minor_values(c, ALL_PROD)
    assert _values(k, "step_order", "qtile_outer") == {"qtile_outer", "head_outer"}
    assert _values(k, "dkv_epilogue", "lds_staged") == {
        "lds_staged",
        "lds_staged_pair",
        "direct",
    }
    assert _values(k, "dkv_store_vec", 8) == {8, 4, 2, 1}
    assert _values(k, "dkv_atomic_guard", "branch") == {"branch"}  # direct slice
    sl_atomic = sw.secondary_slice(c.slice, "atomic")
    ka = sw.minor_values(c.retarget(sl_atomic), ALL_PROD)
    assert _values(ka, "dkv_atomic_guard", "branch") == {"branch", "select"}
    assert _values(ka, "dkv_store_vec", 8) == {8}  # inert in atomic mode
    s = sw.swizzle_values(c, ALL_PROD)
    pads = {p["lds_swizzle"] for p in s if "lds_swizzle" in p}
    bufs = sw.pad_buffers(c)
    assert bufs and set(bufs) <= set(sw.PAD_BUFFERS)
    assert len(pads) == 3 * len(bufs)  # every present buffer, one at a time
    assert all(x.count("=") == 1 for x in pads)
    assert _values(s, "lds_order", "resident_low") == {"resident_low", "phase_low"}


def test_stages_enumerate_the_dq_and_resource_levers():
    """dq_issue (schedule), ptr_noalias / q_rotate / dkv_slab_hoist (codegen)
    are enumerated by their owning stage, each under its coupling: 'all'
    noalias in atomic mode only, the slab hoist on THD direct slices only."""
    c = _start()
    h = sw.schedule_values(c, ALL_PROD)
    assert _values(h, "dq_issue", "block") == set(tc.backward_axis("dq_issue").choices)
    k = sw.minor_values(c, ALL_PROD)
    assert _values(k, "ptr_noalias", False) == {False, "workspace"}
    assert _values(k, "q_rotate", False) == {False, True}
    assert _values(k, "dkv_slab_hoist", False) == {False}  # batched slice
    ka = sw.minor_values(c.retarget(sw.secondary_slice(c.slice, "atomic")), ALL_PROD)
    assert _values(ka, "ptr_noalias", False) == {False, "workspace", "all"}
    assert _values(ka, "dkv_slab_hoist", False) == {False}  # atomic slice
    kt = sw.minor_values(c.retarget(sw.secondary_slice(c.slice, "thd")), ALL_PROD)
    assert _values(kt, "dkv_slab_hoist", False) == {False, True}
    assert _values(kt, "ptr_noalias", False) == {False, "workspace"}
    # the interior row clamp only where the point has edge tiles
    clamped = [p for p in k if p.get("interior_clamp")]
    assert clamped and all(p.get("edge_tiles", c.k.get("edge_tiles")) for p in clamped)
    ke = sw.minor_values(c.with_knobs(edge_tiles=True), ALL_PROD)
    assert _values(ke, "interior_clamp", False) == {False, True}


def test_candidate_spec_follows_the_plan_derived_fields():
    """The plan-owned family-7 fields of a candidate's spec are derived from
    its runtime values exactly as the plan derives them, so the compile /
    resource gate sees the kernel the plan launches."""
    sl = sw.primary_slice("gfx942", "general", 64)
    c = sw.Candidate.make(sl, sw.default_knobs(sl))
    assert c.spec().static_scale == "fold_ds"
    for placement in tc.backward_axis("scale_placement").choices:
        cand = c.with_runtime(scale_placement=placement)
        plan = _plan({"scale_placement": placement})
        assert cand.spec().static_scale == placement
        assert plan.specs["main"].static_scale == placement
    # a hook that pins the one-binary kernel keeps it
    pin = {"static_scale": "runtime"}
    pinned = c.with_knobs(**pin).with_runtime(scale_placement="convert")
    assert pinned.spec().static_scale == "runtime"
    plan = _plan({"scale_placement": "convert", **pin})
    assert plan.specs["main"].static_scale == "runtime"
    inter = c.with_runtime(lb_order="interleave")
    assert inter.spec().lb_interleave
    assert _plan({"lb_order": "interleave"}).specs["main"].lb_interleave


def test_unreleased_families_add_nothing():
    c = _start()
    assert sw.schedule_values(c, sw.Release()) == [
        {k: c.k[k] for k in sw.SCHEDULE_KNOBS} | {"agpr_route": None}
    ]
    assert sw.swizzle_values(c, sw.Release()) == [{}]


def test_runtime_stage_adds_the_aux_values():
    sl = sw.primary_slice("gfx942", "general", 64)
    cells = sw.representative_cells(sl)
    vals = sw._runtime_values(sl, cells, ALL_PROD)
    for name in AUX_KNOBS:
        axis = tc.backward_axis(name)
        assert {v.get(name, axis.default) for v in vals} == set(axis.choices)
    assert not any("use_worklist" in v for v in vals)
    assert {v["lb_order"] for v in vals} == {"natural", "reverse", "interleave"}
    # the XCD remap values, each with the request's XCD count as a hint
    xcd = [v for v in vals if "xcd_chunk" in v]
    assert {v["xcd_chunk"] for v in xcd} == {"gx", "2gx", "4gx", 64}
    assert all(v["num_xcds"] == sw.SWEEP_NUM_XCDS["gfx942"] for v in xcd)
    rdna = sw.primary_slice("gfx1151", "general", 64)
    rv = sw._runtime_values(rdna, sw.representative_cells(rdna), ALL_PROD)
    assert not any("xcd_chunk" in v for v in rv)
    # the candidate spec carries the decode switches the plan sets
    c = sw.Candidate.make(sl, sw.default_knobs(sl), xcd[0])
    assert c.spec().xcd_remap and not c.spec().lb_interleave
    c = c.with_runtime(lb_order="interleave")
    assert c.spec().lb_interleave


def test_new_knobs_at_default_keep_the_candidate_key():
    c = _start()
    defaults = {n: d for n, d in sw.KEYED_IF_SET.items()}
    assert c.with_knobs(**defaults).key() == c.key()
    moved = c.with_knobs(stage_order="g4g1g3")
    assert moved.key() != c.key() and moved.k["stage_order"] == "g4g1g3"
    assert moved.with_knobs(stage_order="g1g3g4").key() == c.key()
    r = c.with_runtime(g_split=1, convert_vec=8)
    assert r.r == {"g_split": 1}


def test_effective_key_merges_inert_and_canonical_values():
    c = _start("gfx942", 64)
    # dkv_atomic_guard is inert in direct mode: one kernel
    assert sw.effective_key(c.with_knobs(dkv_atomic_guard="select")) == (
        sw.effective_key(c)
    )
    # a distinct value is a distinct kernel
    assert sw.effective_key(c.with_knobs(loop_sync="lds_only")) != (sw.effective_key(c))
