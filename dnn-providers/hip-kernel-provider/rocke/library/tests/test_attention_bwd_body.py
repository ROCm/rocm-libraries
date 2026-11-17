# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the shared backward body (geometry, LDS views, helpers).

The emitted kernel is covered by ``test_attention_bwd_ir.py`` (structure,
counts, resources) and the device tests; this file checks the pieces.
"""

from __future__ import annotations

import dataclasses

import pytest
from rocke.core.ir import F16, F32, I32, I64, IRBuilder, PtrType

from kernels.common._attention_bwd_addr import WorkspaceAddr
from kernels.common._attention_bwd_body import (
    BwdLds,
    BwdTileGeometry,
    DkDvAcc,
    StatsView,
    copy_row_sets_to_lds,
    copy_rows_to_lds,
)
from kernels.common._attention_bwd_lds import PHASE_EPILOGUE
from kernels.common.attention_bwd import AttnBwdSpec, lds_plan, validate_attn_bwd_spec

from .sdpa.bwd_kernel_cases import config_knobs

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")


def _resolved(arch, cfg="default", d=64, **kw):
    knobs = dict(config_knobs(cfg, arch))
    knobs.update(kw)
    spec = AttnBwdSpec(head_size=d, mask_class="none", **knobs)
    return validate_attn_bwd_spec(spec, arch)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("d", (32, 64, 128))
def test_geometry_gemm_shapes(arch, d):
    r = _resolved(arch, d=d)
    g = BwdTileGeometry.from_spec(r, arch)
    assert (g.g0.m, g.g0.n, g.g0.k) == (r.block_m, r.block_n, d)
    assert (g.g1.m, g.g1.n, g.g1.k) == (r.block_n, d, r.block_m)
    assert (g.g4.m, g.g4.n, g.g4.k) == (r.block_m, d, r.block_n)
    assert g.g0.grid == (1, r.waves) and g.g1.grid == (r.waves, 1)
    assert g.g4.grid == tuple(r.warp_grid_g4)
    assert g.threads == r.waves * g.wave_size
    st = g.stage_table()
    assert st.mfma.g0 == g.g0.mma_per_wave and st.mfma.g1 == g.g1.mma_per_wave
    assert st.mfma.g4 == g.g4.mma_per_wave
    # the frozen geometry is hashable and equal for equal specs
    assert hash(g) == hash(BwdTileGeometry.from_spec(r, arch))


def test_geometry_needs_a_resolved_spec():
    with pytest.raises(ValueError, match="resolved"):
        BwdTileGeometry.from_spec(AttnBwdSpec(head_size=64), "gfx942")


def test_relabel_needs_the_natural_grids():
    r = _resolved("gfx942", d=64)
    bad = dataclasses.replace(r, warp_grid_g02=(2, 2))
    with pytest.raises(ValueError):
        BwdTileGeometry.from_spec(bad, "gfx942")
    ok = dataclasses.replace(bad, pt_route="lds")
    assert BwdTileGeometry.from_spec(ok, "gfx942").relabel_plan is None


@pytest.mark.parametrize("arch", ARCHS)
def test_lds_views_follow_the_plan(arch):
    for cfg in ("debug", "default"):
        r = _resolved(arch, cfg, d=128)
        plan = lds_plan(r, arch)
        b = IRBuilder("view_probe")
        lds = BwdLds(b, plan, r.dtype)
        assert lds.smem.type.shape == (plan.total_bytes // 2,)
        for buf in plan.buffers:
            if buf.layout.startswith("vector"):
                continue
            cols = r.block_n if buf.name in ("ds", "pt") else r.head_size
            v = lds.view(buf.name, cols, buf.slot)
            assert v.base * 2 == buf.offset
            if buf.layout == "rowmajor:f32":
                assert v.scale == 2 and v.rows * 4 * cols == buf.nbytes
            else:
                assert v.scale == 1 and v.rows * 2 * v.pitch == buf.nbytes
        # the epilogue transpose buffer exists unless the accumulators are in LDS
        assert plan.has("dkv_out") is (not r.acc_in_lds)
        if plan.has("dkv_out"):
            assert plan.buffer("dkv_out").phases == (PHASE_EPILOGUE,)


_VIEW_ROUTES = (
    {},
    {"pt_route": "lds", "warp_grid_g02": (2, 2)},
    {"kv_residency": "lds", "kt_source": "lds"},
)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("route", range(len(_VIEW_ROUTES)))
def test_every_planned_lds_buffer_is_read_by_the_kernel(arch, route, monkeypatch):
    """No dead reservation: each planner buffer gets a view in the emitted
    kernel (a buffer nobody views only inflates the group segment)."""
    from kernels.common.attention_bwd import build_attn_bwd_main

    seen = set()
    real_view = BwdLds.view

    def view(self, name, cols, slot=0):
        seen.add((name, slot))
        return real_view(self, name, cols, slot)

    monkeypatch.setattr(BwdLds, "view", view)
    for cfg in ("debug", "default"):
        for dkv_mode in ("direct", "atomic"):
            try:
                r = _resolved(arch, cfg, d=64, dkv_mode=dkv_mode, **_VIEW_ROUTES[route])
            except ValueError:
                continue  # the route is not legal for this configuration
            seen.clear()
            build_attn_bwd_main(r, arch=arch)
            planned = {(buf.name, buf.slot) for buf in lds_plan(r, arch).buffers}
            if dkv_mode == "atomic":
                # The atomic epilogue reads the accumulators directly; its
                # transpose buffer stays planned (epilogue phase, aliased
                # with the loop) and is not the stats reservation checked here.
                planned.discard(("dkv_out", 0))
            assert planned == seen, (cfg, dkv_mode, sorted(planned ^ seen))


def test_padded_rows_and_unknown_layouts():
    r = _resolved("gfx942", "debug", d=64, lds_swizzle="q=pad16")
    plan = lds_plan(r, "gfx942")
    lds = BwdLds(IRBuilder("pad_probe"), plan, r.dtype)
    assert lds.view("q", 64).pitch == 64 + 8
    with pytest.raises(KeyError):
        lds.view("kt_res", 16)


def test_dkdv_acc_round_trip():
    b = IRBuilder("acc_probe")
    vals = [b.zero_vec_f32(4) for _ in range(12)]
    acc = DkDvAcc.from_flat(vals, 2, 3)
    assert len(acc.dk) == 2 and len(acc.dk[0]) == 3
    assert acc.flat() == vals
    with pytest.raises(ValueError):
        DkDvAcc.from_flat(vals[:10], 2, 3)


def test_stats_view_takes_one_value_per_row():
    b = IRBuilder("stats_probe")
    p = b.param("WS", PtrType(F32, "global"))
    row0 = b.zext(b.const_i32(0), I64)
    one = WorkspaceAddr(b, p, ws_rows=b.const_i32(8), width=1, row0=row0)
    wide = WorkspaceAddr(b, p, ws_rows=b.const_i32(8), width=4, row0=row0)
    StatsView(one, one)
    with pytest.raises(ValueError):
        StatsView(one, wide)


@pytest.mark.parametrize(
    "rows,threads,guards", [(16, 64, 0), (16, 256, 1), (32, 96, 1)]
)
def test_copy_rows_guards_only_a_partial_last_pass(rows, threads, guards):
    from kernels.common._attention_bwd_addr import StridedTensorAddr, decode_batched
    from kernels.common._attention_bwd_frag import LdsView

    b = IRBuilder("copy_probe")
    p = b.param("X", PtrType(F16, "global"))
    n = b.param("n", I32)
    seq = decode_batched(b, b.const_i32(0), s_max=n)
    slab = StridedTensorAddr(b, p, dtype=F16, strides=(0, 0, 64), seq=seq).slab(0, 0)
    smem = b.smem_alloc(F16, (rows * 64,), "flat")
    copy_rows_to_lds(
        b, slab, LdsView(smem, 0, 64, rows), row0=b.const_i32(0), rows=rows, cols=64,
        tid=b.thread_id_x(), threads=threads,
    )  # fmt: skip
    ops = b.kernel.body.ops
    assert sum(op.name == "scf.if" for op in ops) == guards
    stores = sum(
        1
        for op in ops
        for r in ([op] + [x for reg in op.regions for x in reg.ops])
        if r.name == "tile.smem_store_vN"
    )
    assert stores == -(-rows * 8 // threads)


class _ThreadEval:
    """Evaluates the copy helpers' builder calls for one thread (Python ints);
    ``guards`` holds the conditions of the enclosing branches (a load or store
    inside a false branch is not performed)."""

    def __init__(self):
        self.guards = []

    const_i32 = staticmethod(lambda x: x)
    add = staticmethod(lambda a, b: a + b)
    sub = staticmethod(lambda a, b: a - b)
    mul = staticmethod(lambda a, b: a * b)
    div = staticmethod(lambda a, b: a // b)
    mod = staticmethod(lambda a, b: a % b)
    smin = staticmethod(min)
    cmp_lt = staticmethod(lambda a, b: a < b)
    cmp_ge = staticmethod(lambda a, b: a >= b)
    land = staticmethod(lambda a, b: a and b)

    def scf_if(self, cond):
        ev = self

        class _If:
            def __enter__(self):
                ev.guards.append(bool(cond))

            def __exit__(self, *exc):
                ev.guards.pop()

        return _If()


def _simulate(order, *, rows, cols, threads, wave_size):
    """``({(tile, r, c): writer threads}, loads per thread)`` of a two-tile
    copy, each store checked to carry its own cell's source."""
    written, loads = {}, []
    for tid in range(threads):
        ev = _ThreadEval()
        issued = []

        def loader(tile):
            def load(r, c):
                if all(ev.guards):  # a load in a false branch is not issued
                    issued.append((tile, r, c))
                    assert 0 <= r < rows and 0 <= c < cols and c % 8 == 0
                return (tile, r, c)

            return load

        class View:
            def __init__(self, tile):
                self.tile = tile

            def store(self, b, r, c, val, vec):
                if all(ev.guards):
                    assert val == (self.tile, r, c) and vec == 8
                    written.setdefault((self.tile, r, c), []).append(tid)

        copy_row_sets_to_lds(
            ev, ((loader(0), View(0)), (loader(1), View(1))), order=order,
            rows=rows, cols=cols, tid=tid, threads=threads, wave=tid // wave_size,
            wave_size=wave_size,
        )  # fmt: skip
        loads.append(len(issued))
    return written, loads


@pytest.mark.parametrize(
    "rows,cols,threads,wave_size",
    [(16, 128, 512, 64), (16, 128, 256, 64), (32, 64, 512, 64), (32, 32, 256, 64),
     (64, 128, 256, 64), (16, 64, 64, 64), (16, 32, 32, 32), (16, 128, 128, 32)],
)  # fmt: skip
def test_copy_orders_write_every_cell_once(rows, cols, threads, wave_size):
    """Every order writes each cell of both tiles exactly once, from its own
    source cell; ``joint`` (legal when both tiles fit one pass) issues one
    load per busy thread and leaves no thread with two."""
    cells = {(t, r, c) for t in (0, 1) for r in range(rows) for c in range(0, cols, 8)}
    chunks = rows * cols // 8
    orders = ["split", "loads_first"]
    if 2 * chunks <= threads and chunks % wave_size == 0:
        orders.append("joint")
    for order in orders:
        written, loads = _simulate(
            order, rows=rows, cols=cols, threads=threads, wave_size=wave_size
        )
        assert set(written) == cells, order
        assert all(len(w) == 1 for w in written.values()), order
        if order == "joint":
            assert sorted(loads) == [0] * (threads - 2 * chunks) + [1] * (2 * chunks)
        elif order == "loads_first":
            # tails load a clamped chunk: every thread issues every pass
            assert set(loads) == {2 * -(-chunks // threads)}
    if "joint" not in orders:
        with pytest.raises(ValueError, match="joint copy needs whole waves"):
            _simulate(
                "joint", rows=rows, cols=cols, threads=threads, wave_size=wave_size
            )
    with pytest.raises(ValueError, match="copy order"):
        _simulate(
            "interleaved", rows=rows, cols=cols, threads=threads, wave_size=wave_size
        )
