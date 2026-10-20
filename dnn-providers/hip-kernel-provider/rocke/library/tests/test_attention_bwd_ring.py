# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the attention backward Q / dO prefetch ring.

``kernels/common/_attention_bwd_ring.py`` (``ring_depth`` 2 / 3 on the register
path, ``global_path = "dma"`` with ``ring_depth`` 1 / 2 / 3 on gfx950):

* the DMA source map (:func:`dma_lane_source`, the lane-only form the emitter
  uses) equals ``DmaSlabLayout.dma_source`` for every legal tile, piece and
  lane; the slab view's index (static and emitted form) equals
  ``DmaSlabLayout.offset`` and every vector access the body issues stays in
  one 16-byte chunk; a batch writes every destination slot of Q / dO once;
* the wait model: an in-order VM queue per wave replays the prologue, every
  step (phantom steps past the last trip included) and the drain for every
  depth, wave count, tile and trip count; the tile a step reads has landed
  (in that wave) at the wait before the barrier that publishes it, no copy
  lands in a slot between its last read and the barrier after it, the slot a
  step reads holds that step's tile, and nothing is in flight after the
  drain. Mutants (an immediate one larger, the prologue wait one larger, the
  phantom copy skipped, the drain removed) fail;
* the emitted IR: per q step the batch's DMA asm ops, the model's wait
  immediates, only LDS-only barriers (the step ends with one), no global load
  in a DMA loop, the register ring's global loads (the prefetch of Q / dO and
  stats rows) all ahead of the first matrix op, the loop-carried register
  set of ``ring_depth = 3``; a prologue and a drain per q loop;
* legality: the ring and DMA values are built (not refused), DMA excludes head
  packing, a ring keeps ``loop_sync = "full"``; the plan moves a tuned DMA
  request with ``stage_vec = 1`` or head packing to the register path and
  records why.
"""

from __future__ import annotations

import dataclasses
import re
from functools import cache

import pytest

from kernels.common._attention_bwd_frag import (
    IntLaneBuilder,
    DmaSlabView,
    slab_offset,
)
from kernels.common._attention_bwd_lds import (
    DMA_FILLED,
    BwdLdsRequest,
    DmaSlabLayout,
    plan_bwd_lds,
)
from kernels.common._attention_bwd_ring import (
    RING_VMCNT_MAX,
    DmaBatch,
    dma_lane_source,
    dma_wait_counts,
    ring_slot_schedule,
)
from kernels.common._attention_bwd_refusals import resource_refusal
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    _emit_main,
    _not_built,
    validate_attn_bwd_spec,
)

_INT = IntLaneBuilder()
TILES = [(m, d) for m in (16, 32, 64) for d in (32, 64, 128)]


# ---------------------------------------------------------------------------
# source map and view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows,cols", TILES)
def test_lane_source_equals_the_layout_source(rows, cols):
    lay = DmaSlabLayout(rows, cols, 2)
    batch = DmaBatch(rows, cols, waves=1)
    for j in range(batch.pieces):
        for lane in range(64):
            got = dma_lane_source(_INT, lay, j, lane)
            assert got == lay.dma_source(64 * j + lane), (j, lane)
    # the static part of a piece plus the lane part (what the emitter adds)
    for j in range(batch.pieces):
        rj, cj = dma_lane_source(_INT, lay, j, 0)
        for lane in range(64):
            rl, cl = dma_lane_source(_INT, lay, 0, lane)
            assert (rj + rl, cj + cl) == lay.dma_source(64 * j + lane)


@pytest.mark.parametrize("rows,cols", TILES)
def test_batch_writes_every_destination_slot_once(rows, cols):
    lay = DmaSlabLayout(rows, cols, 2)
    for waves in (1, 2, 4, 8):
        batch = DmaBatch(rows, cols, waves)
        seen = {}
        for w in range(waves):
            for name, j in batch.of_wave(w):
                if name in ("lse2", "dsum"):
                    seen[(name, 0)] = seen.get((name, 0), 0) + 1
                    continue
                for lane in range(64):
                    key = (name, 64 * j + lane)
                    seen[key] = seen.get(key, 0) + 1
        slots = lay.size_bytes // 16
        for name in ("q", "do"):
            assert all(seen[(name, s)] == 1 for s in range(slots))
        assert seen[("lse2", 0)] == seen[("dsum", 0)] == 1
        assert sum(len(batch.of_wave(w)) for w in range(waves)) == 2 * batch.pieces + 2
        assert batch.min_per_wave == min(len(batch.of_wave(w)) for w in range(waves))


class _V:
    """A non-int value for the emitted (dynamic) branch of the view index."""

    def __init__(self, x: int) -> None:
        self.x = x


class _VBuilder:
    def const_i32(self, v):
        return _V(int(v))

    def _bin(f):  # noqa: N805 - small factory
        def op(self, a, b):
            return _V(f(a.x, b.x))

        return op

    add = _bin(lambda a, b: a + b)
    sub = _bin(lambda a, b: a - b)
    mul = _bin(lambda a, b: a * b)
    land = _bin(lambda a, b: a & b)
    xor = _bin(lambda a, b: a ^ b)
    shl = _bin(lambda a, b: a << b)
    lshr = _bin(lambda a, b: a >> b)


@pytest.mark.parametrize("rows,cols", TILES)
def test_slab_view_index_equals_the_layout(rows, cols):
    lay = DmaSlabLayout(rows, cols, 2)
    base = 128 * 3  # elements; any 16-byte origin
    view = DmaSlabView(None, base, cols, rows)
    vb = _VBuilder()
    for r in range(rows):
        for c in range(cols):
            want = base + lay.offset(r, c) // 2
            assert slab_offset(rows, cols, r, c) == lay.offset(r, c) // 2
            assert view.index(vb, _V(r), _V(c)).x == want
            assert view.index(vb, r, _V(c)).x == want
            assert view.index(vb, _V(r), c).x == want
    # a slot base is added to every index
    slotted = dataclasses.replace(view, shift=_V(4096))
    assert slotted.index(vb, _V(1), _V(8)).x == base + 4096 + lay.offset(1, 8) // 2
    assert view.align_elems == 8


@pytest.mark.parametrize("rows,cols", TILES)
def test_vector_accesses_stay_in_one_chunk(rows, cols):
    lay = DmaSlabLayout(rows, cols, 2)
    for n in (8, 4):  # straight A reads, transpose reads
        for r in range(rows):
            for c in range(0, cols, n):
                first = lay.offset(r, c)
                assert all(lay.offset(r, c + k) == first + 2 * k for k in range(n))
                assert first // 16 == (first + 2 * n - 1) // 16


def test_dma_fills_only_q_and_do_with_the_slab_layout():
    assert DMA_FILLED == ("q", "do")
    req = BwdLdsRequest(
        head_size=128, k_m0=16, k_n0=128, waves=8, kt_source="lds",
        transpose_source="tr_read", global_path="dma", ring_depth=3,
    )  # fmt: skip
    plan = plan_bwd_lds(req, arch="gfx950")
    for b in plan.buffers:
        if b.name in DMA_FILLED:
            assert b.layout == "dma_slab_xor"
        elif b.name not in ("lse2", "dsum"):
            assert b.layout.startswith("rowmajor:"), (b.name, b.layout)
    for name in ("q", "do", "lse2", "dsum"):
        assert plan.ring_stride(name) == plan.ring_stride("q") > 0
    one = plan_bwd_lds(dataclasses.replace(req, ring_depth=1), arch="gfx950")
    assert one.ring_stride("q") == 0


# ---------------------------------------------------------------------------
# wait model: an in-order VM queue per wave
# ---------------------------------------------------------------------------


class _Race(AssertionError):
    pass


def simulate(depth, trips, waves, rows, cols, n_a, *, waits=None, phantom=True,
             drain=True, atomics_at="end"):  # fmt: skip
    """Replay the DMA ring of one q loop wave by wave (in-order completion).

    ``atomics_at``: where a step issues its ``n_a`` dQ atomics: ``"end"``
    (its own, in stage D after the dS barrier) or ``"top"`` (the deferred
    batch of ``stage_table_atomics``: the previous step's, zeros on the
    first step, in stage A right after the step's copy and its wait).

    Raises :class:`_Race` when a step would read a tile that has not landed
    in some wave at the barrier that publishes it, a copy is issued into a
    slot that is still being read, a slot holds the wrong tile when read, or
    a copy is still in flight after the loop.
    """
    if atomics_at not in ("end", "top"):
        raise ValueError(f"atomics_at={atomics_at!r}")
    batch = DmaBatch(rows, cols, waves)
    waits = waits or dma_wait_counts(depth, batch.min_per_wave, n_a)
    sched = ring_slot_schedule(depth, "dma")
    ahead = sched["ahead"]
    for w in range(waves):
        c_w = len(batch.of_wave(w))
        queue = []  # op kinds ("dma", tile, slot) / ("atomic",), issue order
        done = 0  # ops [0, done) have completed
        slot_tile = {}
        reading = set()  # slots read since the last barrier

        batches = []  # (tile, slot) per issued batch

        def issue(tile, slot):
            if slot in reading:
                raise _Race(f"copy into slot {slot} while it is read")
            batches.append((tile, slot))
            for _ in range(c_w):
                queue.append(("dma", len(batches) - 1))
            slot_tile[slot] = (tile, len(batches) - 1)

        def wait(n):
            nonlocal done
            if n is not None:
                done = max(done, len(queue) - min(n, RING_VMCNT_MAX))

        def landed(tile):
            # the batch that last wrote the slot the tile is read from
            held, batch = slot_tile.get(tile % depth, (None, None))
            if held != tile:
                raise _Race(f"slot {tile % depth} holds {held}, tile {tile} published")
            return all(
                i < done
                for i, op in enumerate(queue)
                if op[0] == "dma" and op[1] == batch
            )

        def barrier(publish=None):
            if publish is not None and not landed(publish):
                raise _Race(f"tile {publish} not landed at its barrier (wave {w})")
            reading.clear()

        if trips and depth > 1:
            for k in range(depth - 1):
                issue(k if k < trips else 0, k % depth)
            wait(waits.prologue)
        barrier(0 if trips and depth > 1 else None)
        for i in range(trips):
            if depth == 1:
                issue(i, 0)
                wait(waits.step_top)
                barrier(i)
            else:
                t = i + ahead
                if t < trips or phantom:
                    issue(t if t < trips else i, t % depth)
            if atomics_at == "top":
                queue.extend([("atomic",)] * n_a)  # stage A
            slot = i % depth
            if slot_tile.get(slot, (None,))[0] != i:
                raise _Race(f"slot {slot} holds {slot_tile.get(slot)}, step {i}")
            reading.add(slot)
            barrier()  # mid (dS) barrier
            reading.add(slot)  # G1 / G3 read Q^T / dO^T after it
            if atomics_at == "end":
                queue.extend([("atomic",)] * n_a)
            if depth > 1:
                wait(waits.step_end)
                nxt = i + 1 if i + 1 < trips else None
                barrier(nxt)
            else:
                barrier()
        if drain:
            wait(0)
        if any(i >= done for i, op in enumerate(queue) if op[0] == "dma"):
            raise _Race(f"a copy is in flight after the loop (wave {w})")


_SIM = [
    (depth, trips, waves, rows, cols)
    for depth in (1, 2, 3)
    for trips in (0, 1, 2, 3, 4, 7)
    for waves in (1, 2, 4, 8)
    for rows, cols in ((16, 128), (32, 64), (32, 32), (64, 128), (16, 32))
]


@pytest.mark.parametrize("atomics", (True, False))
@pytest.mark.parametrize("depth,trips,waves,rows,cols", _SIM)
def test_wait_model_has_no_race(depth, trips, waves, rows, cols, atomics):
    # without atomics: the immediates must not rely on the previous step's
    n_a = rows * cols // (waves * 64) if atomics else 0
    simulate(depth, trips, waves, rows, cols, n_a)


def test_wait_immediates():
    assert dma_wait_counts(1, 2, 4) == dma_wait_counts(1, 0, 0)
    assert dma_wait_counts(1, 2, 4).step_top == 0
    w2 = dma_wait_counts(2, 2, 4)
    assert (w2.prologue, w2.step_end) == (0, 4)
    w3 = dma_wait_counts(3, 2, 4)
    assert (w3.prologue, w3.step_end) == (2, 6)
    assert dma_wait_counts(3, 40, 40).step_end == RING_VMCNT_MAX  # clamped
    with pytest.raises(ValueError):
        dma_wait_counts(4, 1, 1)


@pytest.mark.parametrize("depth", (2, 3))
@pytest.mark.parametrize(
    "waves,rows,cols",
    ((2, 16, 128), (4, 16, 32), (1, 32, 64), (4, 32, 128), (8, 64, 128)),
)
def test_wait_immediates_are_maximal(depth, waves, rows, cols):
    """One more outstanding op at the step end (or, R = 3, at the prologue)
    lets a step read a tile that has not landed (in the wave with the
    smallest share of the batch)."""
    n_a = rows * cols // (waves * 64)
    batch = DmaBatch(rows, cols, waves)
    assert batch.min_per_wave >= 1
    w = dma_wait_counts(depth, batch.min_per_wave, n_a)
    with pytest.raises(_Race):
        simulate(depth, 4, waves, rows, cols, n_a,
                 waits=dataclasses.replace(w, step_end=w.step_end + 1))  # fmt: skip
    if depth == 3:
        with pytest.raises(_Race):
            simulate(depth, 4, waves, rows, cols, n_a,
                     waits=dataclasses.replace(w, prologue=w.prologue + 1))  # fmt: skip


def test_phantom_copies_and_the_drain_are_needed():
    # R = 3 without the phantom copy: the last step's wait counts a batch
    # that was never issued
    # (the previous step's atomics would hide it: none here)
    with pytest.raises(_Race):
        simulate(3, 4, 4, 32, 128, 0, phantom=False)
    simulate(3, 4, 4, 32, 128, 0)
    # R = 3 without the drain: the last phantom copy is still in flight
    with pytest.raises(_Race):
        simulate(3, 4, 4, 32, 128, 16, drain=False)
    simulate(2, 4, 4, 32, 128, 16, drain=False)  # R = 2 waits for every copy


def test_slot_schedule():
    assert ring_slot_schedule(1, "dma") == {"read": 0, "fill": 0, "ahead": 0}
    assert ring_slot_schedule(2, "dma") == {"read": 0, "fill": 1, "ahead": 1}
    assert ring_slot_schedule(3, "dma") == {"read": 0, "fill": 2, "ahead": 2}
    assert ring_slot_schedule(2, "vgpr") == {"read": 0, "fill": 1, "ahead": 1}
    assert ring_slot_schedule(3, "vgpr") == {"read": 0, "fill": 1, "ahead": 2}
    with pytest.raises(ValueError):
        ring_slot_schedule(1, "vgpr")


def test_batch_rejects_tiles_the_dma_cannot_copy():
    with pytest.raises(ValueError):
        DmaBatch(16, 16, 4)  # 512 bytes: half a piece
    with pytest.raises(ValueError):
        DmaBatch(128, 64, 4)  # stats rows beyond one lane each
    with pytest.raises(ValueError):
        DmaBatch(16, 128, 4, wave_size=32)


# ---------------------------------------------------------------------------
# emitted IR
# ---------------------------------------------------------------------------


def _walk(ops, depth=0):
    for op in ops:
        yield op, depth
        for r in op.regions:
            yield from _walk(r.ops, depth + 1)


def _q_loops(kernel):
    loops = []
    for op, _d in _walk(kernel.body.ops):
        if op.name == "scf.for" and any(
            o.name == "tile.mma" for o, _ in _walk(op.regions[0].ops)
        ):
            inner = [o for o, _ in _walk(op.regions[0].ops) if o.name == "scf.for"]
            if not any(
                any(x.name == "tile.mma" for x, _ in _walk(i.regions[0].ops))
                for i in inner
            ):
                loops.append(op)
    return loops


def _asm(ops, mnemonic):
    return [
        op
        for op, _d in _walk(ops)
        if op.name == "tile.inline_asm"
        and op.attrs["template"].endswith(mnemonic + " $0, $2")
    ]


RING_SPECS = [
    ("gfx950", dict(head_size=128, global_path="dma", ring_depth=1)),
    ("gfx950", dict(head_size=128, global_path="dma", ring_depth=2)),
    ("gfx950", dict(head_size=128, global_path="dma", ring_depth=3)),
    ("gfx950", dict(head_size=64, global_path="dma", ring_depth=2, mask_class="band",
                    edge_tiles=True)),  # fmt: skip
    ("gfx950", dict(head_size=32, global_path="dma", ring_depth=3, seq_mode="thd")),
    ("gfx950", dict(head_size=128, global_path="dma", ring_depth=2, pt_route="lds")),
    # the flat head-outer loop (the grouped form, head_outer with edge
    # tiles, refuses every ring: test_rings_refuse_the_grouped_head_outer_form)
    ("gfx950", dict(head_size=128, global_path="dma", ring_depth=3, dkv_mode="atomic",
                    step_order="head_outer")),  # fmt: skip
    ("gfx950", dict(head_size=128, global_path="vgpr", ring_depth=2)),
    ("gfx950", dict(head_size=128, global_path="vgpr", ring_depth=3)),
    ("gfx950", dict(head_size=64, global_path="vgpr", ring_depth=3, head_pack=True)),
    ("gfx950", dict(head_size=64, global_path="vgpr", ring_depth=2, stage_vec=1)),
    ("gfx942", dict(head_size=128, global_path="vgpr", ring_depth=2)),
    ("gfx942", dict(head_size=64, global_path="vgpr", ring_depth=2, head_pack=True,
                    seq_mode="thd", dkv_mode="atomic")),  # fmt: skip
]


def _id(row):
    arch, kw = row
    return arch + "-" + "-".join(f"{k}{v}" for k, v in kw.items())


def _structural(spec, arch):
    """Resolve without the main kernel's recorded resource refusals: these
    tests check the emitted ring structure, not the code object (every static
    rule still applies; a refused row raises from the shipped validator,
    test_ring_values_are_built)."""
    return validate_attn_bwd_spec(spec, arch, main_kernel_refusals=False)


@cache
def _built(i):
    arch, kw = RING_SPECS[i]
    spec = _structural(AttnBwdSpec(**kw), arch)
    return spec, _emit_main(AttnBwdSpec(**kw), arch, main_kernel_refusals=False)


@pytest.mark.parametrize("i", range(len(RING_SPECS)), ids=[_id(r) for r in RING_SPECS])
def test_ring_q_step_ir(i):
    arch, _kw = RING_SPECS[i]
    spec, kernel = _built(i)
    assert not _not_built(spec)
    loops = _q_loops(kernel)
    assert loops
    batch = None
    if spec.global_path == "dma":
        batch = DmaBatch(spec.block_m, spec.head_size, spec.waves)
    n_a = spec.block_m * spec.head_size // (spec.waves * 64)
    for loop in loops:
        ops = loop.regions[0].ops
        names = [op.name for op, _d in _walk(ops)]
        assert "tile.sync" not in names  # every in-loop barrier is LDS-only
        top = [op for op in ops if op.name != "scf.yield"]
        assert top[-1].name == "tile.sync_lds_only"
        barriers = [op for op in top if op.name == "tile.sync_lds_only"]
        # the dS (relabel) or P / dS (LDS route) barrier, the step end, and
        # the publish barrier of the synchronous DMA step
        sync_dma = spec.global_path == "dma" and spec.ring_depth == 1
        assert len(barriers) == 2 + sync_dma
        waits = [
            op.attrs["vmcnt"] for op, _d in _walk(ops) if op.name == "tile.s_waitcnt"
        ]
        atomics = [
            op for op, _d in _walk(ops) if op.name == "memref.global_atomic_add_f32"
        ]
        assert len(atomics) == n_a
        if batch is not None:
            assert len(_asm(ops, "global_load_lds_dwordx4")) == 2 * batch.pieces
            assert len(_asm(ops, "global_load_lds_dword")) == 2
            w = dma_wait_counts(spec.ring_depth, batch.min_per_wave, n_a)
            want = [w.step_top] if spec.ring_depth == 1 else [w.step_end]
            assert waits == want
            assert not [n for n in names if n.startswith("memref.global_load")]
            # the step's own wait comes after its last atomic (end of step)
            if spec.ring_depth > 1:
                order = [op for op, _d in _walk(ops)]
                last_atomic = max(order.index(a) for a in atomics)
                wait_at = next(
                    k for k, op in enumerate(order) if op.name == "tile.s_waitcnt"
                )
                assert wait_at > last_atomic
        else:
            assert waits == []
            assert not _asm(ops, "global_load_lds_dwordx4")
            # the step's global loads are the prefetch (Q / dO chunks, stats
            # rows), all issued ahead of the first matrix op and fenced there
            order = [op.name for op, _d in _walk(ops)]
            fences = [k for k, n in enumerate(order) if n == "tile.sched_barrier"]
            loads = [
                k for k, n in enumerate(order) if n.startswith("memref.global_load")
            ]
            first_mma = order.index("tile.mma")
            assert len(fences) == 1 and loads and max(loads) < fences[0] < first_mma
            chunks = 1 if spec.block_m * spec.head_size // 8 < spec.waves * 64 else 2
            per_chunk = 1 if spec.stage_vec == 8 else 8
            if not spec.head_pack:
                assert len(loads) >= chunks * per_chunk + 2
        # the register ring of depth 3 carries one register set (8-element
        # vectors) through the back edge; nothing else carries registers
        carried = [
            m["name"] for m in loop.attrs["iter_args"]
            if re.fullmatch(r"%?bwd_(q|qi_|qe_)r\d+", m["name"])
        ]  # fmt: skip
        if spec.global_path == "vgpr" and spec.ring_depth == 3:
            assert carried
        else:
            assert not carried
    # one drain per DMA loop (wait 0 then an LDS-only barrier after the loop)
    body = kernel.body.ops
    flat = [op for op, _d in _walk(body)]
    if spec.global_path == "dma":
        drains = [
            k for k, op in enumerate(flat)
            if op.name == "tile.s_waitcnt" and op.attrs["vmcnt"] == 0
            and k + 1 < len(flat) and flat[k + 1].name == "tile.sync_lds_only"
        ]  # fmt: skip
        assert len(drains) >= len(loops)


def test_ring_names_differ_and_default_is_untouched():
    names = set()
    for arch, kw in RING_SPECS[:4]:
        names.add(_structural(AttnBwdSpec(**kw), arch).kernel_name())
    assert len(names) == 4
    d = AttnBwdSpec(head_size=128)
    assert d.ring_depth is None and d.global_path is None
    r = validate_attn_bwd_spec(d, "gfx950")
    assert (r.global_path, r.ring_depth) == ("vgpr", 1)
    assert "ring_depth" not in dict(d.non_default_fields())


# ---------------------------------------------------------------------------
# legality and the plan
# ---------------------------------------------------------------------------


def test_ring_values_are_built():
    for arch, kw in RING_SPECS:
        r = _structural(AttnBwdSpec(**kw), arch)
        assert not _not_built(r)
        if resource_refusal(r, arch) is None:
            validate_attn_bwd_spec(AttnBwdSpec(**kw), arch)
        else:
            with pytest.raises(ValueError, match="resource gate"):
                validate_attn_bwd_spec(AttnBwdSpec(**kw), arch)


@pytest.mark.parametrize("ring_depth", (2, 3))
@pytest.mark.parametrize("global_path", ("vgpr", "dma"))
def test_rings_refuse_the_grouped_head_outer_form(global_path, ring_depth):
    """Every ring (prefetch, DMA) refuses the grouped head-outer form (its
    range loops nested in the head-group loop spill SGPRs carrying the ring's
    next-step tile); the flat head-outer loop and edge tiles alone take it."""
    kw = dict(head_size=128, global_path=global_path, ring_depth=ring_depth)
    grouped = AttnBwdSpec(step_order="head_outer", edge_tiles=True, **kw)
    with pytest.raises(ValueError, match="grouped head-outer form"):
        validate_attn_bwd_spec(grouped, "gfx950")
    for alone in (dict(step_order="head_outer"), dict(edge_tiles=True)):
        _structural(AttnBwdSpec(**alone, **kw), "gfx950")


def test_couplings():
    with pytest.raises(ValueError, match="head_pack"):
        AttnBwdSpec(head_size=128, global_path="dma", head_pack=True)
    with pytest.raises(ValueError, match="loop_sync"):
        AttnBwdSpec(head_size=128, ring_depth=2, loop_sync="lds_only")
    with pytest.raises(ValueError):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=128, global_path="dma"), "gfx942")
    with pytest.raises(ValueError):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=128, ring_depth=3), "gfx942")
    with pytest.raises(ValueError):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, ring_depth=2), "gfx1151")
    # a K swizzle no longer has to be the slab XOR (K is not DMA-staged)
    validate_attn_bwd_spec(
        AttnBwdSpec(head_size=128, global_path="dma", lds_swizzle="k=pad8"), "gfx950"
    )


def test_merged_barrier_ring_is_exclusive_with_the_prefetch_ring():
    """A spec that selects the merged-barrier 2-slot ring (a ``barrier_merge``
    field set to "on", where that knob exists) is staged by that ring: no
    prefetch ring is built for it and its loop_sync is not refused here."""
    from types import SimpleNamespace

    from kernels.common.attention_bwd import _merged_ring

    assert _merged_ring(SimpleNamespace(barrier_merge="on"))
    assert not _merged_ring(SimpleNamespace(barrier_merge="off"))
    assert not _merged_ring(AttnBwdSpec(head_size=64))  # field absent here: off


def test_register_estimate_counts_the_sets_in_flight():
    from kernels.common.attention_bwd import register_estimates

    def est(**kw):
        return register_estimates(
            validate_attn_bwd_spec(AttnBwdSpec(head_size=128, **kw), "gfx950"), "gfx950"
        )["arch_vgpr"]

    base = est()
    two, three = est(ring_depth=2), est(ring_depth=3)
    per_set = 16 * 128 // (64 * 8)
    assert two - base == per_set and three - base == 2 * per_set
    assert est(global_path="dma", ring_depth=3) == base


def _plan(arch, req_kw, knobs):
    from kernels.common.attention_bwd_plan import (
        AttnBwdPolicy,
        AttnBwdRequest,
        FEATURE_GROUPS,
        attn_bwd_plan,
    )

    pol = AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS), waive_unverified=True,
        tuning=lambda *a: dict(knobs),
    )  # fmt: skip
    return attn_bwd_plan(AttnBwdRequest(**req_kw), arch, policy=pol)


def test_plan_moves_unsupported_dma_requests_to_registers():
    knobs = {"global_path": "dma", "ring_depth": 2}
    dense = dict(b=1, h_q=2, h_k=2, h_v=2, s_q=256, s_kv=256, d_qk=128, d_v=128)
    p = _plan("gfx950", dense, knobs)
    assert p.specs["main"].global_path == "dma" and "dma_downgrade" not in p.runtime
    # head packing (GQA, one query row)
    gqa = dict(dense, h_q=8, h_k=1, h_v=1, s_q=1)
    p = _plan("gfx950", gqa, knobs)
    main = p.specs["main"]
    assert main.head_pack and main.global_path == "vgpr" and main.ring_depth == 2
    assert p.runtime["dma_downgrade"] == "head_pack"
    # narrow access (tensor alignment below 16 bytes)
    p = _plan("gfx950", dict(dense, tensor_alignment=2), knobs)
    main = p.specs["main"]
    assert main.stage_vec == 1 and main.global_path == "vgpr" and main.ring_depth == 2
    assert p.runtime["dma_downgrade"] == "stage_vec=1"
