# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Schedule values (``sched``, ``setprio``) on the split-dQ main kernel.

The split-dQ main kernel (``dq_mode = "split"``) has no G4, no dS tile and no
dQ atomics; its step runs three stages, A (G0), B (G2 + G1) and C (G3), on the
bodies split dQ runs on: the synchronous step (full or LDS-only barriers) and
the Q / dO prefetch rings (register path at ``ring_depth`` 2 / 3, DMA path at
1 / 2 / 3 on gfx950). ``stage_table_atomics`` has no atomics to defer there
and is refused (``SPLIT_DEFERRED``).

CPU / IR part (no comgr):

* legality: every other value is legal with every split body (the validator,
  the emission check and the tuning registry, both orders of its walk);
  ``stage_table_atomics`` is refused with its rule everywhere (validator,
  geometry, stage table, registry);
* per q step, over the shipped instance variants and the knob values that
  change the counts: the hint ops are the model's sequence (fences after A
  and B only); under a fenced value each stage's IR census (matrix ops, VMEM
  reads / writes, DS reads / writes, the reads that lead its first matrix
  op) equals the model, the register ring's prefetch is the model's, the
  DMA copy is the model's batch and the step's waits are the ring's wait
  model; the step has no G4 (its matrix ops are G0..G3) and no atomic;
* the DMA waits stay sound with no atomics (``n_a = 0``): no VMEM op between
  a step's copy and its wait, the in-order VM queue replays every depth
  without a race and the immediates stay maximal;
* LDS: the slot-aware replay of every access finds no cross-wave race;
* scheduling only: the resolved spec (the register estimate and the LDS plan
  included) is the twin's without a schedule, the matrix ops (count, atom,
  order) are the twin's, the IR differs.

Compile part (comgr; skips without it): the LLVM hint calls equal the model,
the code object keeps the twin's matrix-instruction count, and the resource
gate per (body, value) on the gate variants of gfx942 / gfx950.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
from rocke.helpers.schedule import DS_READ, DS_WRITE, MFMA, VMEM_READ, VMEM_WRITE

from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common._attention_bwd_ring import DmaBatch, dma_wait_counts
from kernels.common._attention_bwd_sched import (
    FENCED_SCHEDS,
    IGLP_SCHEDS,
    SPLIT_DEFERRED,
    SPLIT_STAGES,
    StageTable,
)
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    _not_built,
    build_attn_bwd_main,
    lds_plan,
    register_estimates,
    validate_attn_bwd_spec,
)

from ._attention_bwd_test_utils import distinct_configs
from .test_attention_bwd_ir import HEADS, VARIANTS, _walk, step_bodies
from .test_attention_bwd_ring import _Race, simulate
from .test_attention_bwd_ring_sched import (
    _dma_asm,
    _dma_per_wave,
    _split_fences,
    _vmem,
    ring_lds_hazards,
    traced_ring,
)
from .test_attention_bwd_sched_body import (
    _BARRIERS,
    BODY_VARIANTS,
    KNOBS,
    _atomics,
    _census,
    _hint_seq,
    _ids,
    _model,
)
from .sdpa.bwd_kernel_cases import config_knobs

_SPLIT = dict(dq_mode="split")
# The bodies of the split-dQ main kernel: name -> (archs, spec knobs).
BODIES = {
    "sync": (("gfx942", "gfx950"), dict(_SPLIT)),
    "ldssync": (("gfx942", "gfx950"), dict(_SPLIT, loop_sync="lds_only")),
    "vgpr2": (("gfx942", "gfx950"), dict(_SPLIT, ring_depth=2)),
    "vgpr3": (("gfx950",), dict(_SPLIT, ring_depth=3)),
    "dma1": (("gfx950",), dict(_SPLIT, global_path="dma", ring_depth=1)),
    "dma2": (("gfx950",), dict(_SPLIT, global_path="dma", ring_depth=2)),
    "dma3": (("gfx950",), dict(_SPLIT, global_path="dma", ring_depth=3)),
}
SPLIT_ARCHS = ("gfx942", "gfx950")
# Every schedule value but stage_table_atomics (refused with split dQ).
SPLIT_PAIRS = (
    ("iglp0", False),
    ("iglp1", False),
    ("fences_only", False),
    ("stage_table", False),
    ("stage_table", True),
)
PAIR_IDS = [f"{s}{'-prio' if p else ''}" for s, p in SPLIT_PAIRS]
ALL_PAIRS = (("none", False),) + SPLIT_PAIRS
ALL_PAIR_IDS = ["none"] + PAIR_IDS
# The variant knobs a split body cannot take are dropped (not legal there).
_ATOMIC_ONLY = ("dq_issue", "g4_slicing")
_HINTS = (
    "tile.sched_barrier",
    "tile.sched_group_barrier",
    "tile.s_setprio",
    "tile.iglp_opt",
)


def _spec(arch, cfg, d, var, knob, body, sched, prio):
    """The spec of a row with a split body and a schedule value (None:
    illegal)."""
    archs, body_kw = BODIES[body]
    if arch not in archs:
        return None
    kw = dict(config_knobs(cfg, arch), **VARIANTS[var])
    kw.pop("sched", None)
    kw.pop("setprio", None)
    extra = dict(KNOBS[knob])
    if any(k in extra for k in _ATOMIC_ONLY):
        return None  # a G4 knob: the split body has no G4
    if extra.get("warp_grid_g02") == "2":
        try:
            w = AttnBwdSpec(head_size=d, **kw).resolved(arch).waves
        except ValueError:
            return None
        if w < 2:
            return None
        extra["warp_grid_g02"] = extra["warp_grid_g13"] = (2, w // 2)
    kw.update(extra)
    if kw.get("loop_sync", "full") != "full" and "loop_sync" in body_kw:
        return None
    kw.update(body_kw)
    try:
        spec = AttnBwdSpec(head_size=d, sched=sched, setprio=prio, **kw)
        r = validate_attn_bwd_spec(spec, arch)
    except ValueError:
        return None
    if _not_built(r):
        return None
    return spec


def _rows():
    """Rows ``(arch, cfg, d, var, knob, body)`` where the twin (no schedule)
    or a schedule value is legal."""
    out = []
    for arch in SPLIT_ARCHS:
        for body in BODIES:
            cands = [
                (arch, cfg, d, var, "", body)
                for cfg in distinct_configs(arch)
                for d in HEADS
                for var in BODY_VARIANTS
            ]
            cands += [
                (arch, "default", d, "band", knob, body)
                for d in HEADS
                for knob in KNOBS
                if knob
            ]
            # a row carries a value where its twin is refused for its
            # resources (RESOURCE_REFUSED): the values are checked there too
            out += [
                r for r in cands if any(_spec(*r, *p) is not None for p in ALL_PAIRS)
            ]
    return out


ROWS = _rows()


@cache
def _kernel(arch, cfg, d, var, knob, body, sched, prio):
    spec = _spec(arch, cfg, d, var, knob, body, sched, prio)
    if spec is None:
        return None, None
    return spec, build_attn_bwd_main(spec, arch=arch)


def _regions(ops, t):
    """``{region name: ops}`` of a q-step body under a fenced value: the
    fence-delimited parts named by the model (:meth:`StageTable.regions`).
    The synchronous step's staging, up to its first barrier, is not part of
    A (the model does not count it)."""
    if t.pipeline == "sync":
        i = next(i for i, op in enumerate(ops) if op.name in _BARRIERS)
        ops = ops[i + 1 :]
    parts = _split_fences(ops)
    names = [name for name, _c, _l in t.regions()]
    assert len(parts) == len(names), (len(parts), names)
    return dict(zip(names, parts))


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------


def test_rows_cover_every_body_and_arch():
    """The census rows hold every body on each of its archs (the suite is
    not vacuous), and each split body's step has no G4."""
    seen = {(r[0], r[5]) for r in ROWS}
    want = {(a, b) for b, (archs, _kw) in BODIES.items() for a in archs}
    assert seen == want
    assert len(ROWS) > 300


@pytest.mark.parametrize("body", list(BODIES))
def test_every_value_is_legal_with_every_split_body(body):
    from dispatch.attention_bwd.tuning_common import (
        backward_axis,
        backward_spec_legality,
    )

    archs, body_kw = BODIES[body]
    for arch in archs:
        for sched, prio in SPLIT_PAIRS:
            kw = dict(body_kw, sched=sched, setprio=prio)
            spec = validate_attn_bwd_spec(AttnBwdSpec(head_size=128, **kw), arch)
            assert not _not_built(spec), (arch, body, sched, prio)
            assert (spec.sched, spec.setprio, spec.dq_mode) == (sched, prio, "split")
            res = backward_spec_legality(kw, arch, head_size=128, dtype="bf16")
            assert res.ok, (arch, body, sched, prio, res)
            # the registry enumerates the pair on a split point, and split on
            # a scheduled point (both orders of its walk)
            assert backward_axis("sched").applies(sched, body_kw)
            if prio:
                assert backward_axis("setprio").applies(True, kw)
            rest = {k: v for k, v in kw.items() if k != "dq_mode"}
            assert backward_axis("dq_mode").applies("split", rest)


def test_stage_table_atomics_is_refused_with_split():
    """``stage_table_atomics`` defers dQ atomics, which the split-dQ main
    kernel does not issue: refused by the validator, the geometry and the
    stage table with SPLIT_DEFERRED on every body, and by the registry in
    both orders of its walk; ``stage_table`` there is legal."""
    from dispatch.attention_bwd.tuning_common import (
        backward_axis,
        backward_spec_legality,
    )

    for body, (archs, body_kw) in BODIES.items():
        for arch in archs:
            for prio in (False, True):
                kw = dict(body_kw, sched="stage_table_atomics", setprio=prio)
                with pytest.raises(ValueError) as exc:
                    validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **kw), arch)
                assert SPLIT_DEFERRED in str(exc.value), (body, arch)
                res = backward_spec_legality(kw, arch, head_size=64, dtype="fp16")
                assert not res.ok and SPLIT_DEFERRED in res.error
            assert not backward_axis("sched").applies("stage_table_atomics", body_kw)
            sta = {k: v for k, v in body_kw.items() if k != "dq_mode"}
            sta["sched"] = "stage_table_atomics"
            assert not backward_axis("dq_mode").applies("split", sta)
            assert backward_axis("dq_mode").applies(
                "split", dict(sta, sched="stage_table")
            )
    # the geometry and the stage table refuse it on their own
    r = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dq_mode="split"), "gfx942")
    geom = BwdTileGeometry.from_spec(r, "gfx942")
    with pytest.raises(ValueError, match="no dQ atomics to defer"):
        dataclasses.replace(geom, sched="stage_table_atomics").check()
    with pytest.raises(ValueError, match="no dQ atomics to defer"):
        StageTable(arch="gfx942", head_size=64, k_m0=16, k_n0=64, k_k4=32,
                   waves=4, dq_mode="split", sched="stage_table_atomics")  # fmt: skip
    # the merged barrier and the counted waits stay atomic-only (the model
    # refuses them with split too)
    for extra in (dict(barrier_merge="on", ring_depth=2), dict(loop_sync="counted")):
        with pytest.raises(ValueError, match="dq_mode='atomic'"):
            validate_attn_bwd_spec(
                AttnBwdSpec(head_size=64, dq_mode="split", sched="stage_table",
                            **extra), "gfx942",
            )  # fmt: skip
    with pytest.raises(ValueError, match="synchronous body or a prefetch ring"):
        StageTable(arch="gfx942", head_size=64, k_m0=16, k_n0=64, k_k4=32,
                   waves=4, dq_mode="split", loop_sync="counted")  # fmt: skip


@pytest.mark.parametrize("body", list(BODIES))
def test_default_adds_no_schedule_op(body):
    """``sched = "none"``: no group, priority or iglp op; the only fence is
    the register ring's prefetch fence, as before."""
    archs, _kw = BODIES[body]
    for arch in archs:
        for d in HEADS:
            spec, k = _kernel(arch, "default", d, "band", "", body, "none", False)
            if spec is None:
                continue
            t = _model(spec, arch)
            assert t.stage_names == SPLIT_STAGES
            for ops in step_bodies(k):
                names = Counter(op.name for op, _ in _walk(ops))
                assert not names["tile.sched_group_barrier"]
                assert not names["tile.s_setprio"] and not names["tile.iglp_opt"]
                assert names["tile.sched_barrier"] == int(t.prefetch_fence)
                assert int(t.prefetch_fence) == t.ir_count_model()["tile.sched_barrier"]
                assert _hint_seq(ops) == t.hint_sequence()


# ---------------------------------------------------------------------------
# the count model against the IR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_hint_sequence_and_stage_counts_equal_the_model(row, pair):
    """Per q step: the hint ops are the model's sequence (fences after A and
    B); the ring's prefetch and waits are the model's; under a fenced value
    every stage region's census and leading reads equal the model (the
    ring's slot stores in C's region as :attr:`StageCounts.fixed`). The step
    has G0..G3 only (no G4) and no atomic."""
    arch = row[0]
    spec, kernel = _kernel(*row, *pair)
    if spec is None:
        pytest.skip("not legal for this configuration")
    t = _model(spec, arch)
    r = validate_attn_bwd_spec(spec, arch)
    staged = (r.global_path, r.ring_depth) != ("vgpr", 1)
    assert t.pipeline == ("ring" if staged else "sync")
    assert (t.sched, t.setprio, t.dq_mode) == (*pair, "split")
    assert t.stage_names == SPLIT_STAGES and t.mfma.g4 == 0
    model = t.ir_count_model()
    assert model["tile.sched_barrier"] == (
        (2 if pair[0] in FENCED_SCHEDS else 0) + int(t.prefetch_fence)
    )
    pf = t.ring_prefetch()
    for ops in step_bodies(kernel):
        assert _hint_seq(ops) == t.hint_sequence()
        flat = [op for op, _ in _walk(ops)]
        assert sum(op.name == "tile.mma" for op in flat) == t.mfma.total
        assert not _atomics(ops)
        waits = [op.attrs["vmcnt"] for op in flat if op.name == "tile.s_waitcnt"]
        assert tuple(waits) == t.step_waits()
        if pf.get("issue") == "dma":
            batch = DmaBatch(r.block_m, r.head_size, r.waves)
            per, tot = _dma_per_wave(ops)
            assert tot == Counter(x4=pf["dma_x4"], x1=pf["dma_x1"])
            counts = [len(batch.of_wave(w)) for w in range(r.waves)]
            assert pf["dma_per_wave"] == (min(counts), max(counts))
            if r.waves > 1:
                assert sorted(per) == sorted(n for n in counts if n)
        if pair[0] in IGLP_SCHEDS:
            first = next(i for i, op in enumerate(ops) if op.name == "tile.iglp_opt")
            assert not any(op.name == "tile.mma" for op, _ in _walk(ops[:first]))
        if pair[0] not in FENCED_SCHEDS:
            continue
        parts = _regions(ops, t)
        assert set(parts) <= {"prefetch", "A", "B", "C"}
        for name, want, lead_want in t.regions():
            got, lead = _census(parts[name])
            for cls, cname in ((MFMA, "mfma"), (VMEM_READ, "vmem_read"),
                               (VMEM_WRITE, "vmem_write"), (DS_READ, "ds_read"),
                               (DS_WRITE, "ds_write")):  # fmt: skip
                assert got[cls] == want.get(cls, 0), (name, cname, got[cls], want)
            if lead_want is not None:
                assert lead == lead_want.get(DS_READ, 0), (name, "lead")
        if pf.get("where") == "A":
            assert t.stage("A").vmem_read == pf["vmem_read"] > 0


def test_split_counts_drop_the_g4_terms():
    """Against the atomic-dQ model of the same tile: G4's MMAs, the dS tile
    write of the relabel route, the K^T read-ahead and the atomics leave;
    the LDS route keeps its P / dS writes; A and B are otherwise the same."""
    kw = dict(arch="gfx950", head_size=128, k_m0=32, k_n0=128, k_k4=32, waves=4,
              sched="stage_table", kt_source="lds", kv_residency="lds")  # fmt: skip
    for pt in ("relabel", "lds"):
        at = StageTable(**kw, pt_route=pt)
        sp = StageTable(**kw, pt_route=pt, dq_mode="split")
        a4, s3 = at.stages(), sp.stages()
        assert len(a4) == 4 and len(s3) == 3
        assert [s.stage for s in s3] == list(SPLIT_STAGES)
        assert s3[0] == a4[0]
        tile = at._frags("g0")[0] * at._frags("g0")[1] * at.c_frag_len
        drop = 0 if pt == "lds" else tile
        assert s3[1].ds_write == a4[1].ds_write - drop
        kt = at._frags("g4")[1] * at._frags("g4")[2] * at._transposed_reads("g4")
        assert kt > 0 and s3[2].ds_read == a4[2].ds_read - kt
        assert sp.mfma.total == at.mfma.total - at.mfma.g4
        assert sp.ir_count_model()["tile.sched_barrier"] == 2
        assert at.ir_count_model()["tile.sched_barrier"] == 3
        prio = dataclasses.replace(sp, setprio=True)
        assert prio.ir_count_model()["tile.s_setprio"] == 6
        with pytest.raises(ValueError, match="unknown stage"):
            sp.emit_stage_end(None, "D")
        with pytest.raises(ValueError):
            sp.stage("D")


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("body", ["dma2", "dma3"])
def test_dma_copy_is_followed_by_no_vmem_op(body, pair):
    """Split dQ issues no atomics: in program order nothing follows a step's
    last DMA copy before its wait (the model's ``vm_after_prefetch`` is 0,
    the ``n_a`` the immediates assume), and the immediates are the ring's
    with ``n_a = 0``."""
    arch = "gfx950"
    for cfg in distinct_configs(arch):
        for d in HEADS:
            for var in ("band", "band_edge", "thd_band_atomic"):
                spec, k = _kernel(arch, cfg, d, var, "", body, *pair)
                if spec is None:
                    continue
                t = _model(spec, arch)
                r = validate_attn_bwd_spec(spec, arch)
                assert t.ring_atomics_per_lane == 0 == t.vm_after_prefetch()
                batch = DmaBatch(r.block_m, r.head_size, r.waves)
                w = dma_wait_counts(r.ring_depth, batch.min_per_wave, 0)
                assert t.ring_waits() == (w.step_end,)
                for ops in step_bodies(k):
                    flat = [op for op, _ in _walk(ops)]
                    last = max(i for i, op in enumerate(flat) if _dma_asm(op))
                    wait = next(
                        i for i, op in enumerate(flat)
                        if i > last and op.name == "tile.s_waitcnt"
                    )  # fmt: skip
                    assert flat[wait].attrs["vmcnt"] == w.step_end
                    assert not any(_vmem(op) for op in flat)  # no VMEM op at all


@pytest.mark.parametrize("depth", (1, 2, 3))
@pytest.mark.parametrize(
    "waves,rows,cols",
    ((2, 16, 128), (4, 16, 32), (1, 32, 64), (4, 32, 128), (8, 64, 128)),
)
def test_wait_model_without_atomics_is_race_free_and_maximal(depth, waves, rows, cols):
    """The ring's in-order VM queue with no dQ atomics (``n_a = 0``): no step
    reads a tile before it lands, and one more in the step-end immediate is
    a race (R >= 2)."""
    batch = DmaBatch(rows, cols, waves)
    w = dma_wait_counts(depth, batch.min_per_wave, 0)
    for trips in (1, 2, 4):
        simulate(depth, trips, waves, rows, cols, 0, waits=w, atomics_at="top")
    if depth >= 2:
        with pytest.raises(_Race):
            simulate(depth, 4, waves, rows, cols, 0, atomics_at="top",
                     waits=dataclasses.replace(w, step_end=w.step_end + 1))  # fmt: skip


# ---------------------------------------------------------------------------
# LDS: slot-aware replay of every access
# ---------------------------------------------------------------------------

_HAZARD_ROWS = [
    (arch, body, var, knob)
    for arch in SPLIT_ARCHS
    for body in BODIES
    for var, knob in (("band_edge", ""), ("band", "ptlds"), ("band", "kvlds"),
                      ("band_pack", ""), ("thd_band_atomic", ""))  # fmt: skip
]


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("hrow", _HAZARD_ROWS, ids=_ids)
def test_split_lds_accesses_do_not_race(hrow, pair):
    """No cross-wave LDS race with any value on any split body (ring slots
    evaluated per iteration; the DMA copies pending until their wait and
    barrier). The step's reads and the staging writes are in the replay."""
    arch, body, var, knob = hrow
    n = 0
    for cfg in distinct_configs(arch):
        for d in HEADS:
            spec = _spec(arch, cfg, d, var, knob, body, *pair)
            if spec is None:
                continue
            r = validate_attn_bwd_spec(spec, arch)
            kernel, tags = traced_ring(spec, arch)
            assert not ring_lds_hazards(kernel, tags, r.ring_depth)[:4], (cfg, d)
            mine = [tags[id(op)] for ops in step_bodies(kernel)
                    for op, _ in _walk(ops) if id(op) in tags]  # fmt: skip
            assert any(x[2] for x in mine) and any(not x[2] for x in mine)
            n += 1
    if n == 0:
        pytest.skip(f"{var} / {knob} is not legal with {body} on {arch}")


# ---------------------------------------------------------------------------
# scheduling only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", SPLIT_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("body", list(BODIES))
def test_scheduling_only_on_split(body, pair):
    """Against the same split body without a schedule: the resolved spec
    apart from the two knobs (agpr_alloc, the register estimate and the LDS
    plan included) is the twin's, the kernel name is salted, the IR differs,
    and the matrix ops (count, atom, order) are the twin's."""
    archs, _kw = BODIES[body]
    for arch in archs:
        for cfg in distinct_configs(arch):
            for d in HEADS:
                base = _spec(arch, cfg, d, "band_edge", "", body, "none", False)
                spec = _spec(arch, cfg, d, "band_edge", "", body, *pair)
                if spec is None or base is None:
                    continue
                r0, r = (validate_attn_bwd_spec(s, arch) for s in (base, spec))
                assert dataclasses.replace(r, sched="none", setprio=False) == r0
                assert register_estimates(r, arch) == register_estimates(r0, arch)
                assert r.kernel_name() != r0.kernel_name()
                assert lds_plan(r, arch) == lds_plan(r0, arch)
                k0 = build_attn_bwd_main(base, arch=arch)
                k = build_attn_bwd_main(spec, arch=arch)
                ir0 = lower(k0, arch=arch, llvm_flavor="llvm22")
                ir = lower(k, arch=arch, llvm_flavor="llvm22")
                assert ir.replace(r.kernel_name(), "K") != ir0.replace(
                    r0.kernel_name(), "K"
                )
                for b0, b1 in zip(step_bodies(k0), step_bodies(k), strict=True):
                    m0 = [op.attrs["op_id"] for op, _ in _walk(b0)
                          if op.name == "tile.mma"]  # fmt: skip
                    m1 = [op.attrs["op_id"] for op, _ in _walk(b1)
                          if op.name == "tile.mma"]  # fmt: skip
                    assert m0 == m1
                    # the step's other ops are the twin's: in order under
                    # iglp, as a multiset under a fenced value (the stats
                    # loads, the read-ahead and the register ring's prefetch
                    # move between stages)
                    n0 = [op.name for op, _ in _walk(b0) if op.name not in _HINTS]
                    n1 = [op.name for op, _ in _walk(b1) if op.name not in _HINTS]
                    if pair[0] in IGLP_SCHEDS:
                        assert n1 == n0
                    else:
                        assert Counter(n1) == Counter(n0)


# ---------------------------------------------------------------------------
# compile: LLVM calls, code objects, resource gate
# ---------------------------------------------------------------------------

_LLVM = {
    "tile.sched_group_barrier": "call void @llvm.amdgcn.sched.group.barrier(",
    "tile.sched_barrier": "call void @llvm.amdgcn.sched.barrier(",
    "tile.s_setprio": "call void @llvm.amdgcn.s.setprio(",
    "tile.iglp_opt": "call void @llvm.amdgcn.iglp.opt(",
}


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("body", list(BODIES))
def test_llvm_hint_calls_equal_the_model(body, pair):
    archs, _kw = BODIES[body]
    for arch in archs:
        for cfg in distinct_configs(arch):
            for d in HEADS:
                for var in ("band", "band_edge"):
                    spec, k = _kernel(arch, cfg, d, var, "", body, *pair)
                    if spec is None:
                        continue
                    model = _model(spec, arch).ir_count_model()
                    n = len(step_bodies(k))
                    ir = lower(k, arch=arch, llvm_flavor="llvm22")
                    for key, call in _LLVM.items():
                        assert ir.count(call) == n * model[key], (key, cfg, d, var)


def _compile(kernel, arch):
    from .test_attention_bwd_sched_body import _compile as compile_

    return compile_(kernel, arch)


@pytest.mark.parametrize("d", HEADS)
@pytest.mark.parametrize("pair", SPLIT_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("body", list(BODIES))
def test_code_objects_keep_the_matrix_instructions(body, pair, d):
    """The default configuration (plain band instance) of each body arch:
    every value's code object has the twin's matrix-instruction count and
    (apart from iglp) a different text."""
    from .test_attention_bwd_sched import _MMA_RE, _disassemble

    archs, _kw = BODIES[body]
    for arch in archs:
        base, k0 = _kernel(arch, "default", d, "band", "", body, "none", False)
        spec, k = _kernel(arch, "default", d, "band", "", body, *pair)
        if spec is None or base is None:
            continue
        t0 = _disassemble(_compile(k0, arch).hsaco, arch)
        t1 = _disassemble(_compile(k, arch).hsaco, arch)
        assert len(_MMA_RE.findall(t0)) == len(_MMA_RE.findall(t1)) > 0
        body1 = [ln.split("//")[0].strip() for ln in t1.splitlines() if "\t" in ln]
        body0 = [ln.split("//")[0].strip() for ln in t0.splitlines() if "\t" in ln]
        if pair[0] not in IGLP_SCHEDS:
            assert body1 != body0, (arch, d)
        if BODIES[body][1].get("global_path") == "dma":
            assert (
                t1.count("global_load_lds_dwordx4")
                == t0.count("global_load_lds_dwordx4")
                > 0
            )


GATE_VARIANTS = ("band", "band_pack", "band_edge", "thd_band_atomic", "sv1_band")
# The points that fail the gate on the production compiler are refused
# exactly (RESOURCE_REFUSED, rows ``splitsched`` of
# test_attention_bwd_refusals.py): the iglp0 / iglp1 values listed there and
# the split twins without a schedule value that fail on their own (gfx950
# default d64: sync and ldssync thd_band_atomic, vgpr3 band_edge, dma2 band).
# No failing point is exempted.
GATE_POINTS = [
    (arch, cfg, d, body)
    for arch in SPLIT_ARCHS
    for cfg in distinct_configs(arch)
    for d in HEADS
    for body in BODIES
    if arch in BODIES[body][0]
]


@pytest.mark.parametrize("var", GATE_VARIANTS)
@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("point", GATE_POINTS, ids=_ids)
def test_resource_gate_per_split_value(point, pair, var):
    """Zero scratch, no spills, register budgets, LDS within the plan and the
    hot-loop AGPR copy ceiling for every legal (body, value) on every
    configuration, head size and gate variant of gfx942 / gfx950, and for the
    split twin without a schedule value (a point the validator refuses,
    RESOURCE_REFUSED, is not legal for it)."""
    arch, cfg, d, body = point
    pytest.importorskip("rocke.runtime.comgr")
    from kernels.common.attention_bwd import attn_bwd_arch_facts

    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    spec, k = _kernel(arch, cfg, d, var, "", body, *pair)
    if spec is None:
        pytest.skip(f"{var} with {body} {pair} is not legal on {arch} {cfg} d{d}")
    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(f"resource gate needs {GATE_FLAVOR} + production comgr")
    facts = attn_bwd_arch_facts(arch)
    r = validate_attn_bwd_spec(spec, arch)
    res = evaluate_resource_gate(
        _compile(k, arch).hsaco, arch=arch,
        lds_capacity_bytes=facts.lds_capacity_bytes, spec=r,
    )  # fmt: skip
    if not res.valid:
        pytest.skip(f"gate not valid here ({res.flavor}, {res.comgr})")
    bad = tuple(res.failures) + tuple(res.warnings)
    if res.resources.lds_bytes > lds_plan(r, arch).total_bytes:
        bad += ("LDS above the plan",)
    assert not bad, (bad, res.resources.arch_vgpr, res.resources.sgpr, r.kernel_name())
