# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Schedule values (``sched``, ``setprio``) on the staging bodies of the step.

The bodies: the Q / dO prefetch ring of ``_attention_bwd_ring`` (the DMA path
at ``ring_depth`` 1 / 2 / 3 on gfx950, the register path at 2 / 3), the
merged-barrier 2-slot ring (``barrier_merge = "on"``, both issue points) and
the counted waits (``loop_sync = "counted"``, both issue points). Their stage
table is ``StageTable.pipeline`` ``"ring"`` / ``"merged"`` / ``"sync"``.

CPU / IR part (no comgr):

* legality: every schedule value is legal with every body (the validator,
  the emission check and the tuning registry), the default emits no
  schedule op beyond the register ring's prefetch fence and the counted
  step's fences, and the refusals that remain (``stage_table_atomics`` with
  the split-dQ main kernel and with the counted late issue) keep their rule
  (the split-dQ bodies: ``test_attention_bwd_split_sched``);
* per q step, over the shipped instance variants and the knob values that
  change the counts: the hint ops are the model's sequence
  (:meth:`StageTable.hint_sequence`); under a fenced value each stage's IR
  census (matrix ops, VMEM reads / writes, DS reads / writes, the reads that
  lead the stage's first matrix op) equals the model, the register ring's
  prefetch is the model's (in stage A under a stage table, behind its fence
  otherwise), the DMA copy is the model's batch (per wave) and the step's
  ``s_waitcnt`` immediates are the ring's wait model;
* the DMA waits stay sound: in the IR every wave issues at least ``n_a``
  VMEM ops after its copy and before the step's wait (the deferred atomics
  under ``stage_table_atomics``), and the in-order VM queue of
  ``test_attention_bwd_ring`` replays every depth, wave count, tile and trip
  count with the deferred placement without a race, its immediates still
  maximal; the counted waits stay exact (the counted model's recount of
  ``test_attention_bwd_counted_sync`` from the IR);
* LDS: a slot-aware replay of every LDS access (ring slots evaluated per
  laid-out iteration, the DMA copies in flight until the wait that retires
  them) finds no cross-wave race for any value;
* ``stage_table_atomics``: the dQ atomics of each step sit in stage A (after
  the DMA copy), none in D; the partial is loop-carried with the ring's
  registers and starts at +0.0; one flush after the q loops; no branch
  encloses a dQ atomic;
* scheduling only: the resolved spec (the register estimate and the LDS plan
  included) is the twin's without a schedule, the matrix ops (count, atom,
  order) and per-step atomics are the twin's, the IR differs.

Compile part (comgr; skips without it): the LLVM hint calls equal the model,
the code object keeps the twin's matrix-instruction count, and the resource
gate (zero scratch, no spills, register budgets, the hot-loop AGPR copy
ceiling) per (ring, value) on the gate variants of gfx942 / gfx950.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
from rocke.helpers.schedule import DS_READ, DS_WRITE, MFMA, VMEM_READ, VMEM_WRITE

from kernels.common._attention_bwd_ring import (
    DMA_PIECE_BYTES,
    DmaBatch,
    QRing,
    dma_wait_counts,
)
from kernels.common._attention_bwd_sched import FENCED_SCHEDS, IGLP_SCHEDS
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    _not_built,
    build_attn_bwd_main,
    lds_plan,
    register_estimates,
    validate_attn_bwd_spec,
)

from ._attention_bwd_test_utils import distinct_configs
from .test_attention_bwd_ir import (
    HEADS,
    VARIANTS,
    _walk,
    group_loops,
    q_loops,
    step_bodies,
)
from .test_attention_bwd_ring import _SIM, _Race, simulate
from .test_attention_bwd_sched_body import (
    _BARRIERS,
    BODY_VARIANTS,
    KNOBS,
    PAIR_IDS,
    SCHED_PAIRS,
    _atomics,
    _census,
    _hint_seq,
    _ids,
    _model,
)
from .sdpa.bwd_kernel_cases import config_knobs

# The rings: name -> (archs, spec knobs).
RINGS = {
    "dma1": (("gfx950",), dict(global_path="dma", ring_depth=1)),
    "dma2": (("gfx950",), dict(global_path="dma", ring_depth=2)),
    "dma3": (("gfx950",), dict(global_path="dma", ring_depth=3)),
    "vgpr2": (("gfx942", "gfx950"), dict(global_path="vgpr", ring_depth=2)),
    "vgpr3": (("gfx950",), dict(global_path="vgpr", ring_depth=3)),
    "merge": (("gfx942", "gfx950"), dict(barrier_merge="on", ring_depth=2)),
    "merge_late": (
        ("gfx942", "gfx950"),
        dict(barrier_merge="on", ring_depth=2, ring_issue="late"),
    ),
    # (not a ring: the counted waits stage the next step in registers)
    "counted": (("gfx942", "gfx950"), dict(loop_sync="counted")),
    "counted_late": (
        ("gfx942", "gfx950"),
        dict(loop_sync="counted", counted_issue="late"),
    ),
}
RING_ARCHS = ("gfx942", "gfx950")
ALL_PAIRS = (("none", False),) + SCHED_PAIRS
ALL_PAIR_IDS = ["none"] + PAIR_IDS
_DMA_X4 = "global_load_lds_dwordx4 $0, $2"
_DMA_X1 = "global_load_lds_dword $0, $2"


def _spec(arch, cfg, d, var, knob, ring, sched, prio):
    """The spec of a row with a ring and a schedule value (None: illegal)."""
    archs, ring_kw = RINGS[ring]
    if arch not in archs:
        return None
    kw = dict(config_knobs(cfg, arch), **VARIANTS[var])
    kw.pop("sched", None)
    kw.pop("setprio", None)
    extra = dict(KNOBS[knob])
    if extra.get("warp_grid_g02") == "2":
        try:
            w = AttnBwdSpec(head_size=d, **kw).resolved(arch).waves
        except ValueError:
            return None
        if w < 2:
            return None
        extra["warp_grid_g02"] = extra["warp_grid_g13"] = (2, w // 2)
    kw.update(extra)
    kw.update(ring_kw)
    try:
        spec = AttnBwdSpec(head_size=d, sched=sched, setprio=prio, **kw)
        r = validate_attn_bwd_spec(spec, arch)
    except ValueError:
        return None
    if _not_built(r):
        return None
    return spec


def _rows():
    """Rows ``(arch, cfg, d, var, knob, ring)`` where the ring twin (no
    schedule) or a schedule value is legal."""
    out = []
    for arch in RING_ARCHS:
        for ring in RINGS:
            cands = [
                (arch, cfg, d, var, "", ring)
                for cfg in distinct_configs(arch)
                for d in HEADS
                for var in BODY_VARIANTS
            ]
            cands += [
                (arch, "default", d, "band", knob, ring)
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
def _kernel(arch, cfg, d, var, knob, ring, sched, prio):
    spec = _spec(arch, cfg, d, var, knob, ring, sched, prio)
    if spec is None:
        return None, None
    return spec, build_attn_bwd_main(spec, arch=arch)


def r_counted(spec, arch) -> bool:
    return validate_attn_bwd_spec(spec, arch).loop_sync == "counted"


def _fences(ops):
    return [i for i, op in enumerate(ops) if op.name == "tile.sched_barrier"]


def _split_fences(ops):
    """The top-level ``sched_barrier(0)``-delimited parts of a step body."""
    cuts = _fences(ops)
    assert all(ops[i].attrs["mask"] == 0 for i in cuts)
    parts, prev = [], 0
    for i in cuts:
        parts.append(ops[prev:i])
        prev = i + 1
    parts.append(ops[prev:])
    return parts


def _regions(ops, t):
    """``{region name: ops}`` of a q-step body under a fenced value: its
    fence-delimited parts named by the model (:meth:`StageTable.regions`,
    every name once)."""
    parts = _split_fences(ops)
    names = [name for name, _c, _l in t.regions()]
    assert len(parts) == len(names), (len(parts), names)
    return dict(zip(names, parts))


def _vmem(op):
    return op.name.startswith(("memref.global_load", "memref.global_atomic",
                               "memref.global_store"))  # fmt: skip


def _dma_asm(op):
    return op.name == "tile.inline_asm" and op.attrs["template"].endswith(
        (_DMA_X4, _DMA_X1)
    )


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ring", list(RINGS))
def test_every_value_is_legal_with_every_ring(ring):
    from dispatch.attention_bwd.tuning_common import backward_spec_legality

    archs, ring_kw = RINGS[ring]
    for arch in archs:
        for sched, prio in SCHED_PAIRS:
            if ring == "counted_late" and sched == "stage_table_atomics":
                continue  # test_counted_late_issue_refuses_the_deferred_atomics
            kw = dict(ring_kw, sched=sched, setprio=prio)
            spec = validate_attn_bwd_spec(AttnBwdSpec(head_size=128, **kw), arch)
            assert not _not_built(spec), (arch, ring, sched, prio)
            assert (spec.sched, spec.setprio) == (sched, prio)
            res = backward_spec_legality(kw, arch, head_size=128, dtype="bf16")
            assert res.ok, (arch, ring, sched, prio, res)


def test_the_remaining_refusals_keep_their_rule():
    """``stage_table_atomics`` with the split-dQ main kernel (no dQ atomics to
    defer) stays refused on every body (the other values run on the split
    body's stages A..C: ``test_attention_bwd_split_sched``; with the counted
    late issue: :func:`test_counted_late_issue_refuses_the_deferred_atomics`)."""
    from kernels.common._attention_bwd_sched import SPLIT_DEFERRED

    for prio in (False, True):
        for ring_kw in ({}, dict(ring_depth=2), dict(global_path="dma")):
            with pytest.raises(ValueError) as exc:
                validate_attn_bwd_spec(
                    AttnBwdSpec(head_size=64, dq_mode="split",
                                sched="stage_table_atomics", setprio=prio,
                                **ring_kw), "gfx950",
                )  # fmt: skip
            assert SPLIT_DEFERRED in str(exc.value)


@pytest.mark.parametrize("ring", list(RINGS))
def test_default_adds_no_schedule_op(ring):
    """``sched = "none"``: no group, priority or iglp op, and the only fences
    are the register ring's prefetch fence or the counted step's pair, as
    before."""
    archs, _kw = RINGS[ring]
    for arch in archs:
        for d in HEADS:
            spec, k = _kernel(arch, "default", d, "band", "", ring, "none", False)
            if spec is None:
                continue
            t = _model(spec, arch)
            for ops in step_bodies(k):
                names = Counter(op.name for op, _ in _walk(ops))
                assert not names["tile.sched_group_barrier"]
                assert not names["tile.s_setprio"] and not names["tile.iglp_opt"]
                fences = int(t.prefetch_fence) + 2 * int(r_counted(spec, arch))
                assert names["tile.sched_barrier"] == fences
                assert fences == t.ir_count_model()["tile.sched_barrier"]
                assert _hint_seq(ops) == t.hint_sequence()


# ---------------------------------------------------------------------------
# the count model against the IR
# ---------------------------------------------------------------------------


def _dma_per_wave(ops):
    """DMA asm ops per wave branch (``scf.if`` on ``wave == w``) of a step,
    and the totals per mnemonic."""
    per, tot = [], Counter()
    for op, anc in _walk(ops):
        if _dma_asm(op):
            tot["x4" if op.attrs["template"].endswith(_DMA_X4) else "x1"] += 1
    for op, _anc in _walk(ops):
        if op.name == "scf.if":
            n = sum(_dma_asm(o) for o, _ in _walk(op.regions[0].ops))
            inner = any(
                o.name == "scf.if"
                and any(_dma_asm(x) for x, _ in _walk(o.regions[0].ops))
                for o, _ in _walk(op.regions[0].ops)
            )
            if n and not inner:
                per.append(n)
    return per, tot


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_hint_sequence_and_stage_counts_equal_the_model(row, pair):
    """Per q step: the hint ops are the model's sequence; the ring's prefetch
    (register loads in stage A or behind their fence, the DMA batch per wave)
    and its waits are the model's; under a fenced value every stage region's
    census and leading reads equal the model (the ring's slot stores and
    barrier-side loads as :attr:`StageCounts.fixed`)."""
    arch = row[0]
    spec, kernel = _kernel(*row, *pair)
    if spec is None:
        pytest.skip("not legal for this configuration")
    t = _model(spec, arch)
    r = validate_attn_bwd_spec(spec, arch)
    assert t.pipeline == ("sync" if r.loop_sync == "counted" else
                          "merged" if r.barrier_merge == "on" else "ring")  # fmt: skip
    assert (t.sched, t.setprio) == pair
    pf = t.ring_prefetch()
    for ops in step_bodies(kernel):
        assert _hint_seq(ops) == t.hint_sequence()
        waits = [
            op.attrs["vmcnt"] for op, _ in _walk(ops) if op.name == "tile.s_waitcnt"
        ]
        assert tuple(waits) == t.step_waits()
        if pf.get("issue") == "dma":
            batch = DmaBatch(r.block_m, r.head_size, r.waves)
            per, tot = _dma_per_wave(ops)
            assert tot == Counter(x4=pf["dma_x4"], x1=pf["dma_x1"])
            counts = [len(batch.of_wave(w)) for w in range(r.waves)]
            assert pf["dma_per_wave"] == (min(counts), max(counts))
            if r.waves > 1:  # one branch per wave that copies
                assert sorted(per) == sorted(n for n in counts if n)
            else:
                assert counts == [sum(tot.values())]
        if pair[0] in IGLP_SCHEDS:
            # the iglp hint is the first op after the ring's prefetch issue
            first = next(i for i, op in enumerate(ops) if op.name == "tile.iglp_opt")
            assert not any(op.name == "tile.mma" for op, _ in _walk(ops[:first]))
        if pair[0] not in FENCED_SCHEDS:
            continue
        parts = _regions(ops, t)
        for name, want, lead_want in t.regions():
            got, lead = _census(parts[name])
            for cls, cname in ((MFMA, "mfma"), (VMEM_READ, "vmem_read"),
                               (VMEM_WRITE, "vmem_write"), (DS_READ, "ds_read"),
                               (DS_WRITE, "ds_write")):  # fmt: skip
                assert got[cls] == want.get(cls, 0), (name, cname, got[cls], want)
            if lead_want is not None:
                assert lead == lead_want.get(DS_READ, 0), (name, "lead")
                # the rings' stats come from LDS, the counted step's from its
                # registers: no stage leads with a VMEM read
                assert lead_want.get(VMEM_READ, 0) == 0
        if pf.get("where") == "A":
            assert t.stage("A").vmem_read == pf["vmem_read"] > 0


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("ring", ["dma2", "dma3"])
def test_dma_copy_is_followed_by_the_counted_vmem_ops(ring, pair):
    """In program order, after a step's last DMA copy op and before its wait,
    the step issues exactly the model's VMEM ops (its own atomics, or the
    deferred batch under ``stage_table_atomics``), at least the ``n_a`` the
    wait immediates assume."""
    arch = "gfx950"
    for cfg in distinct_configs(arch):
        for d in HEADS:
            for var in ("band", "band_edge", "thd_band_atomic"):
                spec, k = _kernel(arch, cfg, d, var, "", ring, *pair)
                if spec is None:
                    continue
                t = _model(spec, arch)
                for ops in step_bodies(k):
                    flat = [op for op, _ in _walk(ops)]
                    last = max(i for i, op in enumerate(flat) if _dma_asm(op))
                    wait = next(
                        i for i, op in enumerate(flat)
                        if i > last and op.name == "tile.s_waitcnt"
                    )  # fmt: skip
                    n = sum(_vmem(op) for op in flat[last + 1 : wait])
                    assert n == t.vm_after_prefetch() >= t.ring_atomics_per_lane > 0
                    assert not any(_vmem(op) for op in flat[:last])  # copy first
                    assert not any(_vmem(op) for op in flat[wait:])


_COUNTED_ROWS = [r for r in ROWS if r[5] in ("counted", "counted_late")]


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("row", _COUNTED_ROWS, ids=_ids)
def test_counted_waits_stay_exact(row, pair):
    """The counted model check of ``test_attention_bwd_counted_sync``
    (:func:`counted_ir_problems`, recounted from the IR through the back
    edge: every barrier waits for exactly the atomics since the last load,
    the next-step loads fenced and ahead of every atomic) holds with every
    value, and its waits are the stage table's (:meth:`counted_waits`)."""
    from ._attention_bwd_counted_model import counted_ir_problems

    arch = row[0]
    spec, kernel = _kernel(*row, *pair)
    if spec is None:
        pytest.skip("not legal for this configuration")
    t = _model(spec, arch)
    for lp in q_loops(kernel):
        probs, waits = counted_ir_problems(lp)
        assert not probs, probs
        assert tuple(waits) == t.counted_waits()


def test_counted_late_issue_refuses_the_deferred_atomics():
    from kernels.common._attention_bwd_sched import COUNTED_LATE_DEFERRED

    for prio in (False, True):
        spec = AttnBwdSpec(head_size=64, loop_sync="counted", counted_issue="late",
                           sched="stage_table_atomics", setprio=prio)  # fmt: skip
        with pytest.raises(ValueError) as exc:
            validate_attn_bwd_spec(spec, "gfx942")
        assert COUNTED_LATE_DEFERRED in str(exc.value)
    from dispatch.attention_bwd.tuning_common import (
        backward_axis,
        backward_spec_legality,
    )

    # the registry couples the axes the same way (both orders of the walk)
    late = {"loop_sync": "counted", "counted_issue": "late"}
    assert not backward_axis("sched").applies("stage_table_atomics", late)
    assert backward_axis("sched").applies("stage_table", late)
    sta = {"loop_sync": "counted", "sched": "stage_table_atomics"}
    assert not backward_axis("counted_issue").applies("late", sta)
    assert backward_axis("counted_issue").applies("early", sta)
    kw = dict(loop_sync="counted", counted_issue="late", sched="stage_table_atomics")
    assert not backward_spec_legality(kw, "gfx942", head_size=64, dtype="fp16").ok
    kw["counted_issue"] = "early"
    assert backward_spec_legality(kw, "gfx942", head_size=64, dtype="fp16").ok


@pytest.mark.parametrize("atomics", (True, False))
@pytest.mark.parametrize("depth,trips,waves,rows,cols", _SIM)
def test_wait_model_has_no_race_with_deferred_atomics(depth, trips, waves, rows, cols,
                                                      atomics):  # fmt: skip
    """The ring's in-order VM queue with the deferred dQ atomics in stage A
    (``stage_table_atomics``): no step reads a tile before it lands."""
    n_a = rows * cols // (waves * 64) if atomics else 0
    simulate(depth, trips, waves, rows, cols, n_a, atomics_at="top")


@pytest.mark.parametrize("depth", (2, 3))
@pytest.mark.parametrize(
    "waves,rows,cols",
    ((2, 16, 128), (4, 16, 32), (1, 32, 64), (4, 32, 128), (8, 64, 128)),
)
def test_wait_immediates_stay_maximal_with_deferred_atomics(depth, waves, rows, cols):
    n_a = rows * cols // (waves * 64)
    batch = DmaBatch(rows, cols, waves)
    w = dma_wait_counts(depth, batch.min_per_wave, n_a)
    simulate(depth, 4, waves, rows, cols, n_a, waits=w, atomics_at="top")
    with pytest.raises(_Race):
        simulate(depth, 4, waves, rows, cols, n_a, atomics_at="top",
                 waits=dataclasses.replace(w, step_end=w.step_end + 1))  # fmt: skip


def test_check_step_refuses_a_placement_the_waits_do_not_cover():
    """:meth:`QRing.check_step` rejects a step model with fewer VMEM ops
    after the copy than the immediates assume (R >= 2 DMA only)."""

    class _T:
        def vm_after_prefetch(self):
            return 3

    ring = QRing.__new__(QRing)
    ring.path, ring.depth, ring.n_atomics = "dma", 2, 4
    with pytest.raises(ValueError, match="assume at least 4"):
        ring.check_step(_T())
    ring.n_atomics = 3
    ring.check_step(_T())
    ring.depth, ring.n_atomics = 1, 99
    ring.check_step(_T())  # vmcnt(0) at the step top
    ring.path, ring.depth = "vgpr", 2
    ring.check_step(_T())  # the compiler's waits


@pytest.mark.parametrize("prio", [False, True])
@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_deferred_dq_atomics_with_ring(row, prio):
    """``stage_table_atomics`` with a ring: per step the dQ atomics sit in
    stage A (after the DMA copy and the register prefetch loads in program
    order), none in B / C / D; the partial is carried after the ring's
    registers (``...dq<i>``, +0.0 at the outermost carrier); one flush
    batch follows the q loops; no branch encloses a dQ atomic."""
    arch = row[0]
    spec, kernel = _kernel(*row, "stage_table_atomics", prio)
    if spec is None:
        pytest.skip("not legal for this configuration")
    t = _model(spec, arch)
    atoms = t.dq_atomics_per_lane
    r = validate_attn_bwd_spec(spec, arch)
    loops = q_loops(kernel)
    for ops in step_bodies(kernel):
        reg = _regions(ops, t)
        assert len(_atomics(reg["A"])) == atoms
        for st in "BCD":
            assert not _atomics(reg[st])
        flat = [op for op, _ in _walk(reg["A"])]
        first_atomic = min(i for i, op in enumerate(flat) if op.name.startswith(
            "memref.global_atomic"))  # fmt: skip
        assert not any(
            _dma_asm(op) or op.name.startswith("memref.global_load")
            for op in flat[first_atomic:]
        )
    for lp in loops:
        names = [m["name"] for m in lp.attrs["iter_args"]]
        dq = [n for n in names if "dq" in n and n[-1].isdigit() and "dqh" not in n]
        assert len(dq) == atoms
        # the deferred partial (values, row, head) is carried last, after the
        # ring's (or the counted step's) registers
        first = names.index(dq[0])
        assert all("dq" in n for n in names[first:]), names[first:]
    first = (group_loops(kernel) or loops)[0]
    metas, inits = first.attrs["iter_args"], first.operands[3:]
    for m, v in zip(metas, inits):
        if m["name"].endswith("dq0"):
            assert v.op.name == "arith.constant" and v.op.attrs["value"] == 0.0
    in_loop = {id(op) for lp in loops for op, _ in _walk(lp.regions[0].ops)}
    ws_dq = 0
    for op, anc in _walk(kernel.body.ops):
        if op.name != "memref.global_atomic_add_f32":
            continue
        if id(op) in in_loop:
            assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
            continue
        if r.dkv_mode == "atomic" and any(a.name == "scf.if" for a in anc):
            continue  # the dK / dV epilogue
        assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
        ws_dq += 1
    assert ws_dq == atoms


# ---------------------------------------------------------------------------
# LDS: slot-aware replay of every access
# ---------------------------------------------------------------------------


def _eval(v, env):
    """Integer value of an i32 IR value under the loop counters ``env``."""
    op = v.op
    if op is None:
        raise KeyError(v.name)
    if op.name == "scf.for":
        return env[v.name]  # the counter (an iter arg is not evaluable)
    if op.name == "arith.constant":
        return int(op.attrs["value"])
    a = [_eval(x, env) for x in op.operands]
    if op.name == "arith.add":
        return a[0] + a[1]
    if op.name == "arith.sub":
        return a[0] - a[1]
    if op.name == "arith.mul":
        return a[0] * a[1]
    if op.name == "arith.and":
        return a[0] & a[1]
    if op.name == "arith.mod":
        return a[0] % a[1]
    raise KeyError(op.name)


def traced_ring(spec, arch):
    """Build the kernel with every LDS access tagged ``(lo, hi, store,
    private, shift, shift_scale, site)``: the view's span in allocation
    elements, its run-time slot shift (an IR value; elements = value //
    scale) and, for a DMA copy, the issue site (one per ``issue_dma`` call).
    The DMA copies are tagged as stores of their 1 KiB piece (Q / dO) or
    stats row (lse2 / Dsum) of the slot; the register ring's joint store
    with the span of the slot's Q and dO tiles."""
    from kernels.common import _attention_bwd_body as body
    from kernels.common import _attention_bwd_ring as ringmod
    from kernels.common._attention_bwd_frag import LdsView

    from .test_attention_bwd_ir import _is_lds

    tags, state = {}, {"private": False, "views": None, "site": 0}
    orig = {n: getattr(LdsView, n) for n in ("load", "store", "tr16_load")}
    orig_acc = {n: getattr(body.BwdTileStep, n) for n in ("_load_lds_acc", "_store_lds_acc")}  # fmt: skip
    names = ("_emit_pieces", "issue_dma", "views", "store_regs")
    orig_ring = {n: getattr(ringmod.QRing, n) for n in names}

    def new_ops(b, n0):
        return b._region_stack[-1].ops[n0:]

    def tagged(name):
        def f(self, b, *args, **kw):
            n0 = len(b._region_stack[-1].ops)
            out = orig[name](self, b, *args, **kw)
            new = [op for op in new_ops(b, n0) if _is_lds(op)]
            assert new
            hi = self.base + self.rows * self.pitch * self.scale
            for op in new:
                tags[id(op)] = (self.base, hi, name == "store", state["private"],
                                self.shift, 1, None)  # fmt: skip
            return out

        return f

    def private(name):
        def f(self, *args, **kw):
            state["private"] = True
            try:
                return orig_acc[name](self, *args, **kw)
            finally:
                state["private"] = False

        return f

    def emit_pieces(self, b, pieces, tile, slabs, ws, slot_bytes):
        n0 = len(b._region_stack[-1].ops)
        orig_ring["_emit_pieces"](self, b, pieces, tile, slabs, ws, slot_bytes)
        asm = [op for op in new_ops(b, n0) if _dma_asm(op)]
        assert len(asm) == len(pieces)
        for op, (name, j) in zip(asm, pieces):
            if name in ("q", "do"):
                lo = (self.offsets[name] + j * DMA_PIECE_BYTES) // 2
                hi = lo + DMA_PIECE_BYTES // 2
            else:
                lo = self.offsets[name] // 2
                hi = lo + 64 * 4 // 2
            tags[id(op)] = (lo, hi, True, False, slot_bytes, 2, state["site"])

    def issue_dma(self, b, tile, slot):
        state["site"] += 1
        return orig_ring["issue_dma"](self, b, tile, slot)

    def views(self, b, slot):
        out = orig_ring["views"](self, b, slot)
        state["views"] = out
        return out

    def store_regs(self, b, regs, slot):
        n0 = len(b._region_stack[-1].ops)
        orig_ring["store_regs"](self, b, regs, slot)
        v_q, v_do = state["views"]
        lo = min(v_q.base, v_do.base)
        hi = max(v.base + v.rows * v.pitch * v.scale for v in (v_q, v_do))
        for op, _anc in _walk(new_ops(b, n0)):
            if _is_lds(op) and id(op) not in tags:  # the joint (Q or dO) store
                tags[id(op)] = (lo, hi, True, False, v_q.shift, 1, None)

    try:
        for n in orig:
            setattr(LdsView, n, tagged(n))
        for n in orig_acc:
            setattr(body.BwdTileStep, n, private(n))
        ringmod.QRing._emit_pieces = emit_pieces
        ringmod.QRing.issue_dma = issue_dma
        ringmod.QRing.views = views
        ringmod.QRing.store_regs = store_regs
        kernel = build_attn_bwd_main(spec, arch=arch)
    finally:
        for n, f in orig.items():
            setattr(LdsView, n, f)
        for n, f in orig_acc.items():
            setattr(body.BwdTileStep, n, f)
        for n, f in orig_ring.items():
            setattr(ringmod.QRing, n, f)
    return kernel, tags


def _ring_events(ops, tags, *, trips, env=None, in_branch=False):
    """Program-order events with the loop bodies laid out ``trips`` times
    (each with its counter bound): ``("sync",)``, ``("wait", vmcnt)``,
    ``("lds", op, lo, hi, batch)`` with the span shifted by the evaluated
    slot (``batch``: the DMA batch, its issue site and loop counters)."""
    from .test_attention_bwd_ir import _is_lds

    env = env or {}
    out = []
    for op in ops:
        if op.name in _BARRIERS + ("tile.s_barrier_bare",):  # (+ the counted one)
            if not in_branch:
                out.append(("sync",))
        elif op.name == "tile.s_waitcnt":
            # (a wait in a branch: the DMA prologue's, in the uniform branch
            # that issued the copies it waits for)
            out.append(("wait", op.attrs["vmcnt"]))
        elif id(op) in tags:
            lo, hi, _st, _pv, shift, scale, site = tags[id(op)]
            s = 0 if shift is None else _eval(shift, env) // scale
            batch = None if site is None else (site, tuple(sorted(env.items())))
            out.append(("lds", op, lo + s, hi + s, batch))
        elif _is_lds(op) and op.name != "tile.smem_addr_of":  # (the DMA's M0 base)
            raise AssertionError(f"an untagged LDS access {op.name}")
        elif op.name == "scf.for":
            iv = op.attrs["iv"]
            for k in range(trips):
                out += _ring_events(
                    op.regions[0].ops, tags, trips=trips, env={**env, iv: k},
                    in_branch=in_branch,
                )  # fmt: skip
        else:
            for r in op.regions:
                out += _ring_events(r.ops, tags, trips=trips, env=env, in_branch=True)
    return out


def ring_lds_hazards(kernel, tags, depth):
    """Pairs of LDS accesses that can race across waves (the rule of
    :func:`test_attention_bwd_ir.lds_hazards` on the evaluated slot spans).
    A DMA copy stays a pending write from its issue until a wait retires it
    (``vmcnt(0)``: every copy; otherwise all but the newest ``R - 2``
    batches, the ring's wait model) and the next barrier passes: it races
    every access to its span in between."""
    found = []
    for trips in (0, 1, 2, 4):
        ev = _ring_events(kernel.body.ops, tags, trips=trips)
        open_, inflight, order = [], [], []
        for e in ev:
            if e[0] == "sync":
                open_ = list(inflight)  # copies still landing
                continue
            if e[0] == "wait":
                keep = []
                if e[1] != 0 and depth > 2:
                    live = [b for b in order if any(x[4] == b for x in inflight)]
                    keep = live[-(depth - 2) :]
                inflight = [x for x in inflight if x[4] in keep]
                continue
            _k, op, lo, hi, batch = e
            st, priv = tags[id(op)][2], tags[id(op)][3]
            for _pk, pop, plo, phi, _pb in open_:
                pst, ppriv = tags[id(pop)][2], tags[id(pop)][3]
                if not (lo < phi and plo < hi) or (priv and ppriv):
                    continue
                if (st != pst) or (st and (lo, hi) != (plo, phi)):
                    found.append((pop.name, plo, op.name, lo, trips))
            open_.append(e)
            if batch is not None:
                if batch not in order:
                    order.append(batch)
                inflight.append(e)
    return found


_HAZARD_ROWS = [
    (arch, ring, var, knob)
    for arch in RING_ARCHS
    for ring in RINGS
    for var, knob in (("band_edge", ""), ("band", "ptlds"), ("band", "kvlds"),
                      ("band_pack", ""), ("thd_band_atomic", ""))  # fmt: skip
]


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("hrow", _HAZARD_ROWS, ids=_ids)
def test_ring_lds_accesses_do_not_race(hrow, pair):
    """No cross-wave LDS race with any value on any ring (slots evaluated per
    iteration; the DMA copies pending until their wait and barrier). The
    model is not vacuous: the step's Q / dO slot reads and the ring's slot
    writes are in it."""
    arch, ring, var, knob = hrow
    for cfg in distinct_configs(arch):
        for d in HEADS:
            spec = _spec(arch, cfg, d, var, knob, ring, *pair)
            if spec is None:
                continue
            r = validate_attn_bwd_spec(spec, arch)
            kernel, tags = traced_ring(spec, arch)
            assert not ring_lds_hazards(kernel, tags, r.ring_depth)[:4], (cfg, d)
            body = [op for lp in q_loops(kernel) for op, _ in _walk(lp.regions[0].ops)]
            mine = [tags[id(op)] for op in body if id(op) in tags]
            assert any(t[2] for t in mine)  # the slot writes (stores or copies)
            if r.ring_depth > 1:
                assert any(t[4] is not None for t in mine)  # run-time slots


def test_ring_hazard_model_catches_a_misplaced_access():
    """Mutants of the replay's inputs race: a DMA copy into the slot the
    step reads, a wait that leaves the published tile in flight, and a
    register-ring slot store into the slot the step reads (the LDS route:
    no barrier between G3's Q^T reads and the step-end stores)."""
    spec = AttnBwdSpec(head_size=128, global_path="dma", ring_depth=2,
                       sched="stage_table")  # fmt: skip
    kernel, tags = traced_ring(spec, "gfx950")
    assert not ring_lds_hazards(kernel, tags, 2)
    # copies into the read slot: drop the slot shift of every DMA tag
    bad = {
        k: (v[:4] + (None,) + v[5:] if v[6] is not None else v) for k, v in tags.items()
    }
    assert ring_lds_hazards(kernel, bad, 2)
    # a wait that retires nothing: R = 4 keeps two batches pending
    assert ring_lds_hazards(kernel, tags, 4)
    spec = AttnBwdSpec(head_size=128, ring_depth=2, pt_route="lds", sched="stage_table")
    kernel, tags = traced_ring(spec, "gfx942")
    assert not ring_lds_hazards(kernel, tags, 2)
    # the slot stores into the read slot (no shift): they race the reads
    stores = {
        k: (v[:4] + (None,) + v[5:] if v[2] and v[4] is not None else v)
        for k, v in tags.items()
    }
    assert ring_lds_hazards(kernel, stores, 2)


# ---------------------------------------------------------------------------
# scheduling only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", SCHED_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("ring", list(RINGS))
def test_scheduling_only_on_the_ring(ring, pair):
    """Against the same ring without a schedule: the resolved spec apart
    from the two knobs (agpr_alloc, the register estimate and the LDS plan
    included) is the twin's, the kernel name is salted, the IR differs, and
    the matrix ops (count, atom, order) and per-step dQ atomics are the
    twin's."""
    archs, _kw = RINGS[ring]
    for arch in archs:
        for cfg in distinct_configs(arch):
            for d in HEADS:
                base = _spec(arch, cfg, d, "band_edge", "", ring, "none", False)
                spec = _spec(arch, cfg, d, "band_edge", "", ring, *pair)
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
                    m0 = [
                        op.attrs["op_id"]
                        for op, _ in _walk(b0)
                        if op.name == "tile.mma"
                    ]
                    m1 = [
                        op.attrs["op_id"]
                        for op, _ in _walk(b1)
                        if op.name == "tile.mma"
                    ]
                    assert m0 == m1
                    assert len(_atomics(b0)) == len(_atomics(b1))


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
@pytest.mark.parametrize("ring", list(RINGS))
def test_llvm_hint_calls_equal_the_model(ring, pair):
    archs, _kw = RINGS[ring]
    for arch in archs:
        for cfg in distinct_configs(arch):
            for d in HEADS:
                for var in ("band", "band_edge"):
                    spec, k = _kernel(arch, cfg, d, var, "", ring, *pair)
                    if spec is None:
                        continue
                    model = _model(spec, arch).ir_count_model()
                    n = len(step_bodies(k))
                    ir = lower(k, arch=arch, llvm_flavor="llvm22")
                    for key, call in _LLVM.items():
                        # (the register ring's prologue has no fence: only
                        # the steps emit hint calls)
                        assert ir.count(call) == n * model[key], (key, cfg, d, var)


def _compile(kernel, arch):
    from .test_attention_bwd_sched_body import _compile as compile_

    return compile_(kernel, arch)


@pytest.mark.parametrize("d", HEADS)
@pytest.mark.parametrize("pair", SCHED_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("ring", list(RINGS))
def test_code_objects_keep_the_matrix_instructions(ring, pair, d):
    """The default configuration (plain band instance) of each ring arch:
    every value's code object has the twin's matrix-instruction count and
    (apart from iglp) a different text."""
    from .test_attention_bwd_sched import _MMA_RE, _disassemble

    archs, _kw = RINGS[ring]
    for arch in archs:
        base, k0 = _kernel(arch, "default", d, "band", "", ring, "none", False)
        spec, k = _kernel(arch, "default", d, "band", "", ring, *pair)
        if spec is None or base is None:
            continue
        t0 = _disassemble(_compile(k0, arch).hsaco, arch)
        t1 = _disassemble(_compile(k, arch).hsaco, arch)
        assert len(_MMA_RE.findall(t0)) == len(_MMA_RE.findall(t1))
        body = [ln.split("//")[0].strip() for ln in t1.splitlines() if "\t" in ln]
        body0 = [ln.split("//")[0].strip() for ln in t0.splitlines() if "\t" in ln]
        if pair[0] not in IGLP_SCHEDS:
            assert body != body0, (arch, d)
        if RINGS[ring][1].get("global_path") == "dma":
            # the DMA copy and its wait survive the scheduler
            assert (
                t1.count("global_load_lds_dwordx4")
                == t0.count("global_load_lds_dwordx4")
                > 0
            )


GATE_VARIANTS = ("band", "band_pack", "band_edge", "thd_band_atomic", "sv1_band")
GATE_POINTS = [
    (arch, cfg, d, ring)
    for arch in RING_ARCHS
    for cfg in distinct_configs(arch)
    for d in HEADS
    for ring in RINGS
    if arch in RINGS[ring][0]
]


@pytest.mark.parametrize("var", GATE_VARIANTS)
@pytest.mark.parametrize("pair", ALL_PAIRS, ids=ALL_PAIR_IDS)
@pytest.mark.parametrize("point", GATE_POINTS, ids=_ids)
def test_resource_gate_per_ring_value(point, pair, var):
    """Zero scratch, no spills, register budgets, LDS within the plan and the
    hot-loop AGPR copy ceiling for every legal (ring, value) on every
    configuration, head size and gate variant of gfx942 / gfx950, and for
    the ring twin without a schedule value. A point the validator refuses
    (RESOURCE_REFUSED: among them the ring twins that fail the gate on their
    own, gfx950 d64 DMA R = 1 / 2 with edge tiles and register R = 3 with
    head packing) is not legal and skips; no failing point is exempted."""
    arch, cfg, d, ring = point
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

    spec, k = _kernel(arch, cfg, d, var, "", ring, *pair)
    if spec is None:
        pytest.skip(f"{var} with {ring} {pair} is not legal on {arch} {cfg} d{d}")
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
