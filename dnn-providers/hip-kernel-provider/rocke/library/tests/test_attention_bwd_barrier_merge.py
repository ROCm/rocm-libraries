# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Barrier merge of the backward q step (CPU: spec, legality, LDS plan, IR).

``barrier_merge = "on"`` (with ``ring_depth = 2``) runs a q step with one
barrier instead of three: Q / dO / lse2 / Dsum in a 2-slot LDS ring, dS in 2
slots, the next step staged into the other slot before the pre-G4 barrier.
``ring_issue`` places the next step's global loads at the top of the step
(``"early"``) or right before their LDS stores (``"late"``). Checked here:
value sets and refusals, the planner's slots and its capacity refusal, and
the IR structure (one barrier per step, every G0..G3 operand read and every
ring store before it, G4 and every dQ atomic after it, the loads ahead of the
atomics, a prologue barrier per q loop, a full barrier before the epilogue).
The device checks (dK / dV bit for bit, the mutation run) are in
``test_attention_bwd_numeric``.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common._attention_bwd_body import STAGE_ORDERS
from kernels.common.attention_bwd import (
    BARRIER_MERGE_STAGE_ORDERS,
    AttnBwdSpec,
    build_attn_bwd_main,
    lds_plan,
    validate_attn_bwd_spec,
)

from ._attention_bwd_test_utils import distinct_configs
from .sdpa.bwd_kernel_cases import config_knobs
from .test_attention_bwd_ir import _walk, q_loops

MFMA = ("gfx942", "gfx950")
MERGE = dict(barrier_merge="on", ring_depth=2)
SYNCS = ("tile.sync", "tile.sync_lds_only")
ATOMIC = "memref.global_atomic_add_f32"


def _spec(arch, cfg, d=64, **kw):
    return AttnBwdSpec(head_size=d, dtype="bf16", **{**config_knobs(cfg, arch), **kw})


def _legal(spec, arch):
    try:
        validate_attn_bwd_spec(spec, arch)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# spec values and refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(barrier_merge="yes"),
        dict(barrier_merge=True),
        dict(ring_issue="soon"),
        dict(ring_issue="late"),  # barrier_merge = "on" only
        dict(barrier_merge="on", ring_depth=1),
        dict(barrier_merge="on", ring_depth=3),
    ],
)
def test_bad_values_raise(kw):
    with pytest.raises(ValueError):
        AttnBwdSpec(head_size=64, **kw)


def test_defaults_are_off_and_unsalted():
    s = AttnBwdSpec(head_size=64)
    assert (s.barrier_merge, s.ring_issue) == ("off", "early")
    names = {
        AttnBwdSpec(head_size=64, **kw).kernel_name("main")
        for kw in (
            {},
            dict(barrier_merge="on"),
            dict(barrier_merge="on", ring_issue="late"),
        )
    }
    assert len(names) == 3
    assert AttnBwdSpec(head_size=64, barrier_merge="off").kernel_name(
        "main"
    ) == AttnBwdSpec(head_size=64).kernel_name("main")


@pytest.mark.parametrize("arch", MFMA)
def test_merge_resolves_the_two_slot_ring(arch):
    r = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, barrier_merge="on"), arch)
    assert r.ring_depth == 2
    # without the merge a 2-deep ring is the prefetch ring (its own staging,
    # two LDS-only barriers per step): a different kernel
    plain = AttnBwdSpec(head_size=64, ring_depth=2)
    k = build_attn_bwd_main(plain, arch=arch)
    merged = build_attn_bwd_main(
        AttnBwdSpec(head_size=64, barrier_merge="on"), arch=arch
    )
    assert k.name != merged.name


@pytest.mark.parametrize("arch", ("gfx1151", "gfx1201"))
def test_refused_off_cdna(arch):
    with pytest.raises(ValueError, match="barrier_merge"):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **MERGE), arch)


@pytest.mark.parametrize("arch", MFMA)
def test_refused_outside_the_proved_route(arch):
    for order in STAGE_ORDERS:
        spec = AttnBwdSpec(head_size=64, stage_order=order, **MERGE)
        if order in BARRIER_MERGE_STAGE_ORDERS:
            validate_attn_bwd_spec(spec, arch)
        else:
            with pytest.raises(ValueError, match="G4 last"):
                validate_attn_bwd_spec(spec, arch)
    for kw in (
        dict(pt_route="lds", warp_grid_g02=(2, 2), waves=4),
        dict(acc_in_lds=True, agpr_alloc=(0, 0)),
        dict(qdo_stage="loads_first"),
        dict(stats_load="step_top"),
    ):
        with pytest.raises(ValueError):
            validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **MERGE, **kw), arch)
    # the split-dQ main kernel issues no G4, so the merged barrier (before G4)
    # and the next step's slot stores would never be emitted
    with pytest.raises(ValueError, match="dq_mode='atomic'"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, dq_mode="split", **MERGE), arch
        )
    # the grouped head-outer form (its range loops spill SGPRs with the ring's
    # carried tile); the flat head-outer loop and edge tiles alone stay legal
    for d in (32, 64, 128):
        with pytest.raises(ValueError, match="grouped head-outer"):
            validate_attn_bwd_spec(
                AttnBwdSpec(
                    head_size=d, step_order="head_outer", edge_tiles=True, **MERGE
                ),
                arch,
            )
    validate_attn_bwd_spec(
        AttnBwdSpec(head_size=64, step_order="head_outer", **MERGE), arch
    )
    validate_attn_bwd_spec(AttnBwdSpec(head_size=64, edge_tiles=True, **MERGE), arch)


@pytest.mark.parametrize("arch", MFMA)
def test_ring_stats_have_lds_slots_only_under_a_ring(arch):
    """lse2 / Dsum are staged through LDS only under a ring (the merge's two
    slots); the synchronous staging reads them from the workspace and plans
    no LDS for them (the waste fix keeps the default plan byte for byte)."""
    plain = lds_plan(validate_attn_bwd_spec(AttnBwdSpec(head_size=64), arch), arch)
    assert not [b for b in plain.buffers if b.name in ("lse2", "dsum")]
    merged = lds_plan(
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **MERGE), arch), arch
    )
    for name in ("lse2", "dsum"):
        assert sorted(b.slot for b in merged.buffers if b.name == name) == [0, 1]


# ---------------------------------------------------------------------------
# LDS plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", MFMA)
@pytest.mark.parametrize("d", (32, 64, 128))
def test_plan_slots(arch, d):
    for cfg in distinct_configs(arch):
        base = validate_attn_bwd_spec(_spec(arch, cfg, d), arch)
        plan0 = lds_plan(base, arch)
        assert [b.slot for b in plan0.buffers if b.name == "ds"] == [0]
        spec = _spec(arch, cfg, d, **MERGE)
        if not _legal(spec, arch):
            continue
        plan = lds_plan(validate_attn_bwd_spec(spec, arch), arch)
        plan.check()
        for name in ("q", "do", "lse2", "dsum", "ds"):
            offs = [b.offset for b in plan.buffers if b.name == name]
            assert len(offs) == 2 and offs[0] != offs[1], name
            assert (offs[1] - offs[0]) % 16 == 0
        assert plan.total_bytes >= plan0.total_bytes


@pytest.mark.parametrize("arch", MFMA)
def test_capacity_refusal(arch):
    """Every merged spec whose plan exceeds the arch LDS is refused by the
    planner rule; on gfx942 (the smaller LDS) the search meets such a spec
    whose plain twin is legal."""
    seen = 0
    for d, bm, bn, kv in itertools.product(
        (64, 128), (16, 32, 64), (64, 128, 256), ("reg", "lds")
    ):
        kw = dict(block_m=bm, block_n=bn, kv_residency=kv)
        if kv == "lds":
            kw["kt_source"] = "lds"
        twin = AttnBwdSpec(head_size=d, dtype="bf16", **kw)
        if not _legal(twin, arch):
            continue
        spec = dataclasses.replace(twin, **MERGE)
        plan = lds_plan(spec.resolved(arch), arch)
        if plan.fits:
            continue
        with pytest.raises(ValueError, match="LDS plan"):
            validate_attn_bwd_spec(spec, arch)
        seen += 1
    assert seen or arch != "gfx942"


# ---------------------------------------------------------------------------
# IR structure
# ---------------------------------------------------------------------------

VARIANTS = (
    {},
    dict(ring_issue="late"),
    dict(loop_sync="lds_only"),
    dict(edge_tiles=True),
    dict(head_pack=True),
    dict(dkv_mode="atomic"),
    # (step_order="head_outer" with edge_tiles is refused under the merge)
    dict(step_order="head_outer", q_rotate=True),
    dict(q_rotate=True, edge_tiles=True),
    dict(dq_issue="fragment"),
    dict(stage_order="g3g1g4"),
)


def _flat(ops):
    return [op for op, _ in _walk(ops)]


def _is_load(op):
    return op.name.startswith("memref.global_load")


def _g4_mmas(plain_ops):
    """Matrix ops of G4 in a default step: between its 2nd and 3rd barrier."""
    syncs = [i for i, o in enumerate(plain_ops) if o.name in SYNCS]
    assert len(syncs) == 3
    return sum(o.name == "tile.mma" for o in plain_ops[syncs[1] + 1 : syncs[2]])


def _check_step(ops, n_g4, late):
    """One barrier; G0..G3, every global load and every LDS store before
    it; G4 and every dQ atomic after it; the loads ahead of the G0 / G2
    matrix ops ("early") or behind the G1 / G3 ones ("late")."""
    syncs = [i for i, o in enumerate(ops) if o.name in SYNCS]
    assert len(syncs) == 1, [ops[i].name for i in syncs]
    x = syncs[0]
    mmas = [i for i, o in enumerate(ops) if o.name == "tile.mma"]
    pre, g4 = mmas[:-n_g4], mmas[-n_g4:]
    assert pre and g4 and max(pre) < x < min(g4)
    atomics = [i for i, o in enumerate(ops) if o.name == ATOMIC]
    assert atomics and min(atomics) > x
    stores = [i for i, o in enumerate(ops) if o.name == "tile.smem_store"]
    loads = [i for i, o in enumerate(ops) if _is_load(o)]
    assert loads and max(loads) < x
    assert stores and max(stores) < x
    if late:
        assert min(loads) > max(pre)
    else:
        assert max(loads) < min(mmas)


def _parent_list(kernel, target):
    """``(ops, index)`` of ``target`` in the op list that holds it."""
    for ops in [kernel.body.ops] + [
        r.ops for op, _ in _walk(kernel.body.ops) for r in op.regions
    ]:
        for i, o in enumerate(ops):
            if o is target:
                return ops, i
    raise AssertionError("op not found")


@pytest.mark.parametrize("arch", MFMA)
@pytest.mark.parametrize("var", VARIANTS, ids=lambda v: "+".join(v) or "plain")
def test_one_barrier_per_step(arch, var):
    built = 0
    for cfg in distinct_configs(arch):
        for d in (32, 64, 128):
            spec = _spec(arch, cfg, d, **MERGE, **var)
            plain_kw = {k: v for k, v in var.items() if k != "ring_issue"}
            twin = _spec(arch, cfg, d, **plain_kw)
            # (a twin the validator refuses, e.g. a recorded resource-gate
            # failure, has no kernel to compare against)
            if not _legal(spec, arch) or not _legal(twin, arch):
                continue
            kernel = build_attn_bwd_main(spec, arch=arch)
            loops = q_loops(kernel)
            ploops = q_loops(build_attn_bwd_main(twin, arch=arch))
            assert len(loops) == len(ploops)
            for lp, plp in zip(loops, ploops):
                n_g4 = _g4_mmas(_flat(plp.regions[0].ops))
                _check_step(_flat(lp.regions[0].ops), n_g4, spec.ring_issue == "late")
                # the prologue: slot 0 staged under the trip test, then P
                ops, i = _parent_list(kernel, lp)
                assert ops[i - 1].name in SYNCS and ops[i - 2].name == "scf.if"
                staged = _flat(ops[i - 2].regions[0].ops)
                assert any(_is_load(s) for s in staged)
                assert any(s.name == "tile.smem_store" for s in staged)
            # a full barrier after the last loop, before the epilogue
            flat = _flat(kernel.body.ops)
            last = next(i for i, o in enumerate(flat) if o is loops[-1])
            tail = flat[last + 1 + len(_flat(loops[-1].regions[0].ops)) :]
            after = [o.name for o in tail if o.name in SYNCS]
            assert after and after[0] == "tile.sync"
            lower(kernel, arch=arch, llvm_flavor="llvm22")
            built += 1
    assert built
