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
        8: {"structure", "runtime"},  # runtime: the dQ kernel's XCD chunk
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
    # the scale placement, LDS accumulators and packed heads, the split dQ
    # (one register sum instead of fp32 atomics) and its stats prescale, P's
    # argument as one fma and the exp2 that flushes P below 2^-126
    assert by_class["reorders_fp"] == {
        "step_order",
        "q_rotate",
        "g_split",
        "scale_placement",
        "acc_in_lds",
        "head_pack",
        "dq_mode",
        "dq_stats_prescale",
        "bwd_exp2",
        "stats_fma",
        "dkv_sink",
    }
    # the schedule, epilogue, LDS layout and aux knobs are bit for bit
    added = {
        "asm_tail_nop",
        "asm_cluster",
        "asm_load_hoist",
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
        "counted_issue",
        "lb_mirror",
        "bf16_cast",
        "stats_vec",
        "worklist",
        "use_worklist",
        "g4_slicing",
        "block_k4",
        "s_in_agpr",
        "stats_vec",
        "ws_bounds",
        "dkv_cell_vec",
        "narrow_vec",
        "slab_carry",
    }
    assert added <= by_class["bitwise"]
    assert {"waves", "block_m", "atom_g13", "pt_route"} <= by_class["tile"]
    assert {"dq_waves", "dq_block_n", "dq_atom_g02", "dq_transpose_source"} <= (
        by_class["tile"]
    )
    assert {"dq_head_loop", "dq_edge_tiles", "dq_kv_pad", "dq_xcd_chunk"} <= (
        by_class["bitwise"]
    )
    # the work list is built (dK / dV bit for bit, the dQ atomics in another
    # CTA order): no knob is left without a numerics class
    assert "n/a" not in by_class


# --------------------------------------------------------------------------- legality
@cache
def _pool(arch: str, d: int) -> tuple:
    """Knob sets of every legal geometry-stage candidate of a slice."""
    sl = sw.primary_slice(arch, "general", d)
    return tuple(dict(c.k) for c in sw.geometry_stage(sl, sw.Release()).candidates)


_ROUTE = {"agpr_alloc": (128, 128)}
_ASM = {"agpr_route": "asm_g13", "agpr_alloc": (128, 128)}
# a small tile on four waves: the transposed / replicated grids split it
_SMALL_TILE = {
    "pt_route": "lds", "waves": 4, "block_m": 32, "block_n": 16, "block_k4": 16,
    "atom_g4": "16x16x16",
}  # fmt: skip
# Couplings a value needs (tried before the geometry pools).
_CONTEXTS = {
    "agpr_route": ({}, _ROUTE, {"agpr_alloc": (192, 192)}),
    "asm_tail_nop": (_ASM, {**_ASM, "agpr_alloc": (192, 192)}),
    "asm_cluster": (_ASM, {**_ASM, "agpr_alloc": (192, 192)}),
    "asm_load_hoist": (_ASM, {**_ASM, "agpr_alloc": (192, 192)}),
    "s_in_agpr": ({}, _ROUTE, {"agpr_alloc": "backend"}),
    "warp_grid_g02": ({}, {"pt_route": "lds"}, _SMALL_TILE),
    "warp_grid_g13": ({}, {"pt_route": "lds"}, _SMALL_TILE),
    "acc_in_lds": ({}, {"agpr_alloc": (0, 0)}),
    "dkv_atomic_guard": ({}, {"dkv_mode": "atomic"}),
    "setprio": ({"sched": "stage_table"}, {"sched": "stage_table_atomics"}),
    "dkv_sink": ({}, {"dkv_mode": "atomic"}),
    "dkv_cell_vec": ({"dkv_epilogue": "direct"},),
    "narrow_vec": ({"stage_vec": 1},),
    "ptr_noalias": ({}, {"dkv_mode": "atomic"}),
    "counted_issue": ({"loop_sync": "counted"},),
    "worklist": ({"seq_mode": "thd"},),  # the THD work-list decode
    # 16x16x32 needs block_m >= 32 (the default d32 / d64 gfx950 tiles; the
    # K-permuted G1 / G3 B is read by the stacked transpose-read plan)
    "atom_g13": ({}, {"block_m": 32}),
    "ring_issue": ({"barrier_merge": "on"},),
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
        "dkv_sink": {"dkv_mode": "atomic"},  # inert (so built) in direct mode
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
            elif a.name == "dkv_sink":
                spec = dataclasses.replace(base, dkv_mode="atomic", dkv_sink=v)
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
        if (name, value) == ("lb_order", "mirror"):
            # pairing needs a band and a grid that overfills num_cus; the
            # plan compiles the lb_mirror twin of the default main kernel
            plan = _plan({name: value}, causal=True, num_cus=4)
            assert plan.runtime[name] == value and plan.runtime["pair"] == 1
            assert plan.specs["main"].lb_mirror is True
            continue
        # the work list serves THD requests (the plan refuses it on batched)
        req = {"layout": "thd"} if name == "use_worklist" else {}
        plan = _plan({name: value}, **req)
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


def test_g4_slicing_and_block_k4_axes():
    """g4_slicing: every value on MFMA, step only on WMMA; its coupling filter
    refuses LDS accumulators, the per-fragment dQ issue and slice_db_g3
    without G3 right before G4. block_k4 is worth enumerating only with a
    slicing value (inert under step), and both are legal and built (the
    asm_g13 + s_in_agpr pair too, no longer refused by the emission check)."""
    ax = tc.backward_axis("g4_slicing")
    assert ax.choices == ("step", "slice", "slice_db", "slice_db_g3")
    assert (ax.family, ax.stage, ax.numerics) == (5, "schedule", "bitwise")
    for arch in BWD_ARCHS:
        mfma = attn_bwd_arch_facts(arch).matrix_path == "mfma"
        assert ax.values(arch) == (ax.choices if mfma else ("step",))
    assert ax.applies("slice_db", {})
    assert not ax.applies("slice", {"acc_in_lds": True})
    assert not ax.applies("slice", {"dq_issue": "fragment_regs"})
    assert not ax.applies("slice_db_g3", {"stage_order": "g4g1g3"})
    assert ax.applies("slice_db_g3", {"stage_order": "g3g4g1"})
    k4 = tc.backward_axis("block_k4")
    assert k4.applies(None, {}) and not k4.applies(64, {"g4_slicing": "step"})
    assert k4.applies(64, {"g4_slicing": "slice_db"})
    for arch in ("gfx942", "gfx950"):
        for v in ax.choices[1:]:
            got = tc.backward_spec_legality({"g4_slicing": v}, arch, head_size=128)
            assert got.ok, (arch, v, got.error)
        # (d128: the gfx950 d64 tile with asm G0 / G2 is a recorded
        # resource-gate failure, RESOURCE_REFUSED)
        s = tc.backward_spec_legality(
            {"agpr_route": "asm_g13", "agpr_alloc": (128, 128), "s_in_agpr": True},
            arch, head_size=128,
        )  # fmt: skip
        assert s.ok and not s.not_built, (arch, s.error)
    assert not tc.backward_spec_legality(
        {"g4_slicing": "slice"}, "gfx1151", head_size=64
    ).ok


def test_legality_helper_is_the_validator_plus_the_emission_check(monkeypatch):
    for arch in BWD_ARCHS:
        spec = AttnBwdSpec(head_size=128, dtype="bf16")
        v = tc.backward_spec_legality(spec, arch)
        assert v.ok and v.status == "legal"
        assert v.resolved == validate_attn_bwd_spec(spec, arch)
    # tr_read with the K-permuted 16x16x32 G1 / G3 B operand (gfx950): built
    # (the stacked transpose-read plan)
    wide = {"transpose_source": "tr_read", "atom_g13": "16x16x32"}
    v = tc.backward_spec_legality(wide, "gfx950", head_size=64)
    assert v.ok and v.status == "legal" and not v.not_built, v.error
    # a finding of the emission check makes a validated spec "not_built"
    monkeypatch.setattr(tc, "_not_built", lambda spec: ["an unbuilt emission"])
    nb = tc.backward_spec_legality(wide, "gfx950", head_size=64)
    assert nb.status == "not_built" and nb.not_built == ("an unbuilt emission",)
    assert not nb.ok
    monkeypatch.undo()
    for arch in BWD_ARCHS:  # the XT transposes and the xor swizzle are built
        for knobs in ({"transpose_source": "xt_lds"}, {"lds_swizzle": "q=xor"}):
            # (d128: the one-wave gfx1151 d64 tile spills with the XT images,
            # LAYOUT_SPILLING_COMBOS)
            v = tc.backward_spec_legality(knobs, arch, head_size=128)
            assert v.status == "legal", (arch, knobs, v.error, v.not_built)
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
    # waves_per_eu 1..4 on every CDNA target since the small-tile region
    wpe = (None, 1, 2, 3, 4)
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
        "structure": set(sw.STRUCTURE_KNOBS)
        | set(sw.STRUCTURE_EXTRA)
        | set(sw.DQ_EXTRA),  # the dQ kernel's fields on split points (dq_values)
        # agpr_route crosses agpr_alloc; g4_slicing is enumerated with block_k4
        "schedule": set(sw.SCHEDULE_KNOBS)
        | {"agpr_route", "g4_slicing"}
        | set(sw.SCHEDULE_EXTRA),
        "codegen": set(sw.MINOR_KNOBS) | set(sw.CODEGEN_EXTRA),
        "swizzle": {"lds_swizzle"} | set(sw.SWIZZLE_EXTRA),
        "runtime": {"g_split", "lb_order", "scale_placement", "xcd_chunk"}
        | set(sw.RUNTIME_EXTRA)
        | {"use_worklist"}
        | {"dq_xcd_chunk"},  # stage R of split-dQ candidates (_dq_xcd_values)
    }
    for a in tc.BWD_KNOB_AXES:
        if a.plan_owned or a.name in NOT_YET_EFFECTIVE_KNOBS:
            assert not any(a.name in s for s in owners.values()) or a.name in (
                sw.GEOMETRY_KNOBS
            )
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
    assert _values(h, "loop_sync", "full") == {"full", "lds_only", "counted"}
    # the counted issue point only on top of counted waits: production moves
    # one registry knob off the start point (full waits there), so only the
    # full level reaches counted + late
    assert all(
        p.get("loop_sync") == "counted" for p in h if p.get("counted_issue") == "late"
    )
    h_full = sw.schedule_values(c, ALL_FULL)
    late = [p for p in h_full if p.get("counted_issue") == "late"]
    assert late and all(p.get("loop_sync") == "counted" for p in late)
    assert _values(h, "qdo_stage", "split") == {"split", "loads_first", "joint"}
    assert _values(h, "stats_load", "at_use") == {"at_use", "step_top"}
    assert _values(h, "agpr_route", None) == set(tc.backward_axis("agpr_route").choices)
    asm = [p for p in h if p.get("agpr_route") == "asm_g13"]
    assert _values(asm, "asm_cluster", "k_step") == {"k_step", "gemm"}
    assert all("asm_cluster" not in p for p in h if p.get("agpr_route") != "asm_g13")
    assert _values(asm, "asm_load_hoist", False) == {False, True}
    assert all("asm_load_hoist" not in p for p in h if p.get("agpr_route") != "asm_g13")
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
    # every present buffer, one at a time: three row pads and the line xor
    assert len(pads) == 4 * len(bufs)
    assert all(x.count("=") == 1 for x in pads)
    assert {x.split("=")[1] for x in pads} == {"pad8", "pad16", "pad32", "xor"}
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


def test_barrier_merge_axes_follow_the_validator_couplings():
    """barrier_merge (schedule stage) is enumerated on the CDNA targets only
    and under the couplings its hazard proof needs; ring_issue only on top of
    barrier_merge = "on"; every enumerated value the validator accepts."""
    bm, ri = tc.backward_axis("barrier_merge"), tc.backward_axis("ring_issue")
    assert (bm.family, bm.stage, bm.default) == (5, "schedule", "off")
    assert (ri.family, ri.stage, ri.default) == (5, "schedule", "early")
    assert bm.values("gfx942") == bm.values("gfx950") == ("off", "on")
    assert bm.values("gfx1151") == bm.values("gfx1201") == ("off",)
    assert bm.applies("on", {})
    for blocked in (
        {"global_path": "dma"},
        {"ring_depth": 3},
        {"pt_route": "lds"},
        {"acc_in_lds": True},
        {"stage_order": "g4g1g3"},
        {"qdo_stage": "joint"},
        {"stats_load": "step_top"},
        {"dq_mode": "split"},
        {"step_order": "head_outer", "edge_tiles": True},
    ):
        assert not bm.applies("on", blocked), blocked
    # the flat head-outer loop and edge tiles alone take the merge; the
    # grouped head-outer form takes no ring deeper than 1 (validator rule)
    assert bm.applies("on", {"step_order": "head_outer"})
    assert bm.applies("on", {"edge_tiles": True})
    rd = tc.backward_axis("ring_depth")
    grouped = {"step_order": "head_outer", "edge_tiles": True}
    for depth in (2, 3):
        assert not rd.applies(depth, grouped)
        assert rd.applies(depth, {"step_order": "head_outer"})
        assert rd.applies(depth, {"edge_tiles": True})
        with pytest.raises(ValueError, match="grouped head-outer"):
            validate_attn_bwd_spec(
                AttnBwdSpec(head_size=64, ring_depth=depth, **grouped), "gfx950"
            )
    assert not ri.applies("late", {}) and ri.applies("late", {"barrier_merge": "on"})
    # a ring_depth = 1 structure point never takes the merge (it needs the
    # 2-slot ring); a ring_depth = 2 point does
    h1 = sw.schedule_values(_start(), ALL_PROD)
    assert _values(h1, "barrier_merge", "off") == {"off"}
    c = _start(ring_depth=2)
    h = sw.schedule_values(c, ALL_PROD)
    assert _values(h, "barrier_merge", "off") == {"off", "on"}
    merged = [p for p in h if p.get("barrier_merge") == "on"]

    def status(p):
        return tc.backward_spec_legality(
            {**dict(c.k), **p}, "gfx950", head_size=64, dtype="bf16"
        ).status

    # the coupling test leaves the validator no merge coupling to refuse: a
    # merged point is refused only where its unmerged twin is refused too
    # (a coupling of the schedule point itself, e.g. asm_g13 with iglp)
    statuses = [status(p) for p in merged]
    assert "legal" in statuses
    for p, st in zip(merged, statuses):
        if st == "illegal":
            assert status({**p, "barrier_merge": "off"}) == "illegal", p
    full = sw.schedule_values(c.with_knobs(barrier_merge="on"), ALL_FULL)
    assert _values(full, "ring_issue", "early") == {"early", "late"}


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
    # the work list (THD slices): the plan and the candidate set worklist
    sl_thd = sw.secondary_slice(sl, "thd")
    t = sw.Candidate.make(sl_thd, sw.default_knobs(sl_thd))
    wl = t.with_runtime(use_worklist=True)
    assert wl.spec().worklist and not t.spec().worklist
    assert _plan({"use_worklist": True}, layout="thd").specs["main"].worklist
    assert not _plan({}, layout="thd").specs["main"].worklist


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
    assert not any("use_worklist" in v for v in vals)  # batched slice
    assert {v["lb_order"] for v in vals} == {
        "natural",
        "reverse",
        "mirror",
        "interleave",
    }
    thd = sw.secondary_slice(sl, "thd")
    tv = sw._runtime_values(thd, sw.representative_cells(thd), ALL_PROD)
    wl = [v for v in tv if v.get("use_worklist")]
    assert wl and all(v["lb_order"] == "natural" for v in wl)
    assert {v["lb_order"] for v in tv} == {v["lb_order"] for v in vals}
    full = sw._runtime_values(thd, sw.representative_cells(thd), ALL_FULL)
    assert all(v["lb_order"] == "natural" for v in full if v.get("use_worklist"))
    assert any(v.get("use_worklist") for v in full)
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


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_split_dq_schedule_couplings_agree_with_the_validator(arch, monkeypatch):
    """``sched`` / ``setprio`` x ``dq_mode`` on every body split dQ can take
    (synchronous, LDS-only barriers, register and DMA rings) and the two it
    cannot (merged barrier, counted waits): the registry's couplings, in both
    orders of the walk, enumerate exactly what the validator's rules accept
    (only ``stage_table_atomics`` is refused with split dQ on a split body; the
    couplings do not model the recorded resource refusals, which the test
    bypasses); the dQ kernel's ``dq_sched`` applies on split points only."""
    from kernels.common import attention_bwd as AB

    monkeypatch.setattr(AB, "resource_refusal", lambda spec, arch: None)
    bodies = [
        {},
        {"loop_sync": "lds_only"},
        {"ring_depth": 2},
        {"barrier_merge": "on", "ring_depth": 2},
        {"loop_sync": "counted"},
    ]
    if arch == "gfx950":
        bodies += [{"ring_depth": 3}, {"global_path": "dma", "ring_depth": 1},
                   {"global_path": "dma", "ring_depth": 2}]  # fmt: skip
    sched, prio = tc.backward_axis("sched"), tc.backward_axis("setprio")
    mode = tc.backward_axis("dq_mode")
    seen = set()
    for body in bodies:
        for s in sched.values(arch):
            for p in (False, True):
                if p and s not in ("stage_table", "stage_table_atomics"):
                    continue
                kw = dict(body, sched=s, setprio=p, dq_mode="split")
                legal = tc.backward_spec_legality(kw, arch, head_size=64).ok
                walk_a = mode.applies("split", {**body, "sched": s, "setprio": p})
                walk_b = sched.applies(s, {**body, "dq_mode": "split"}) and (
                    not p or prio.applies(True, kw)
                )
                if mode.applies("split", body):  # a body split dQ runs on
                    assert walk_a == walk_b == legal, (body, s, p, legal)
                    seen.add((s, legal))
                else:
                    assert not legal, (body, s, p)
    assert ("stage_table", True) in seen and ("stage_table_atomics", False) in seen
    assert ("stage_table_atomics", True) not in seen
    dq = tc.backward_axis("dq_sched")
    assert dq.kind == tc.KIND_DQ and dq.default == "none"
    for v in dq.choices[1:]:
        assert dq.applies(v, {"dq_mode": "split"}) and not dq.applies(v, {})
