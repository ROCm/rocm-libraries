# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke.helpers.loads.RegTransposeTileLoader``: free-axis global vectors
transposed in registers into a K-contiguous LDS tile.

* ``plan`` / ``choose_vec``: the work split (every thread, whole waves only,
  or rejected) and the widest usable width.
* The constructor refuses non-16-bit elements and splits ``plan`` rejects.
* A small kernel lowered for gfx1201: one 16-byte LDS store per free-axis row
  (no 16-bit LDS stores) and one ``v_perm_b32`` per output dword; partial
  participation guards the idle waves; ``guard`` puts the reads in their own
  branch; ``col_swizzle`` is applied to every store.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir import F16, F32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.loads import REGT_K_RUN, REGT_VECS, RegTransposeTileLoader as L
from rocke.helpers.tensor_view import make_buffer_resource


def test_plan_full_partial_and_rejected_splits():
    # 128x32 at 4 wide: 32 row groups x 4 K chunks over 128 threads.
    assert L.plan(tile_rows=128, tile_cols=32, block_size=128, wave_size=32, vec=4) == (
        128,
        128,
        "",
    )
    # 256x16 at 8 wide over 1024 threads: 64 units, the first two waves.
    assert L.plan(
        tile_rows=256, tile_cols=16, block_size=1024, wave_size=32, vec=8
    ) == (64, 64, "")
    # 192x16 at 8 wide: 48 units neither divide 1024 nor fill whole waves.
    assert L.plan(tile_rows=192, tile_cols=16, block_size=1024, wave_size=32, vec=8)[2]
    assert (
        L.choose_vec(
            tile_rows=192, tile_cols=16, block_size=1024, wave_size=32, max_vec=8
        )
        == 4
    )
    # K extent not a multiple of the 8-deep run, width outside REGT_VECS.
    assert L.plan(tile_rows=64, tile_cols=12, block_size=64, wave_size=32, vec=2)[2]
    assert 3 not in REGT_VECS
    assert L.plan(tile_rows=64, tile_cols=16, block_size=64, wave_size=32, vec=3)[2]


def test_constructor_rejects():
    with pytest.raises(ValueError):
        L(
            tile_rows=64,
            tile_cols=32,
            block_size=128,
            wave_size=32,
            vec=4,
            elem_dtype=F32,
        )
    with pytest.raises(ValueError):
        L(tile_rows=192, tile_cols=16, block_size=1024, wave_size=32, vec=8)


def _kernel(loader: L, *, split: bool = False, guard: bool = False):
    b = IRBuilder("regt_loader_check")
    b.kernel.attrs["max_workgroup_size"] = loader.block_size
    x = b.param("X", PtrType(F16, "global"))
    tid = b.thread_id_x()
    tile = b.smem_alloc(F16, [loader.tile_rows, loader.tile_cols], name_hint="tile")
    rsrc = make_buffer_resource(b, x, num_bytes=b.const_i32(1 << 20)).rsrc

    def desc(b_, row, col):
        # Free axis (rows) contiguous, K (cols) strided.
        return b_.add(b_.mul(col, b_.const_i32(4096)), row), None

    if split:
        staged = loader.load_global(b, tid=tid, descriptor=desc, rsrc=rsrc, guard=guard)
        loader.store_lds(b, smem_dst=tile, staged=staged)
    else:
        loader.load(b, tid=tid, smem_dst=tile, descriptor=desc, rsrc=rsrc)
    b.ret()
    return _lower_kernel_to_llvm_python(b.kernel, arch="gfx1201")


def test_wide_stores_and_one_perm_per_dword():
    loader = L(tile_rows=128, tile_cols=32, block_size=128, wave_size=32, vec=4)
    ll = _kernel(loader)
    assert not re.search(r"store half [^\n]*addrspace\(3\)", ll)
    assert re.search(r"store <8 x half> [^\n]*addrspace\(3\)", ll)
    # 1 unit x 4 rows x 4 dwords.
    assert len(re.findall(r"call i32 @llvm\.amdgcn\.perm\(", ll)) == 4 * (
        REGT_K_RUN // 2
    )


def test_partial_participation_and_guarded_reads():
    loader = L(tile_rows=256, tile_cols=16, block_size=1024, wave_size=32, vec=8)
    assert loader.partial and loader.active_threads == 64
    plain = _kernel(loader, split=True)
    guarded = _kernel(loader, split=True, guard=True)
    loads = r"call <\d+ x i32> @llvm\.amdgcn\.raw\.ptr\.buffer\.load"
    assert len(re.findall(loads, plain)) == len(re.findall(loads, guarded))
    # guard: the reads sit in a branch of their own, so one more branch.
    assert guarded.count("br i1") > plain.count("br i1")


def test_guard_is_refused_where_it_cannot_apply():
    # Not partial: every wave loads, there is nothing to guard.
    full = L(tile_rows=128, tile_cols=32, block_size=128, wave_size=32, vec=4)
    assert not full.partial and not full.guard_applies
    with pytest.raises(ValueError, match="guard needs partial participation"):
        _kernel(full, split=True, guard=True)
    # Partial but scalar (vec == 1): the guarded form needs vec > 1.
    scalar = L(tile_rows=256, tile_cols=16, block_size=1024, wave_size=32, vec=1)
    assert scalar.partial and not scalar.guard_applies
    with pytest.raises(ValueError, match="guard needs partial participation"):
        _kernel(scalar, split=True, guard=True)
    wide = L(tile_rows=256, tile_cols=16, block_size=1024, wave_size=32, vec=8)
    assert wide.guard_applies


def test_col_swizzle_applies_to_every_store():
    calls = []

    def swz(b, row, col):
        calls.append(1)
        return b.xor(col, b.const_i32(8))

    loader = L(
        tile_rows=64, tile_cols=32, block_size=64, wave_size=32, vec=2, col_swizzle=swz
    )
    _kernel(loader)
    assert len(calls) == loader.units_per_thread * loader.vec


def test_plan_refuses_above_regt_max_units():
    """plan rejects configurations where any thread would hold > REGT_MAX_UNITS
    micro-tiles; the C++ staged heap has exactly that many slots."""
    from rocke.helpers.loads import REGT_MAX_UNITS

    # At boundary: (256 * 1) units over 1 thread -- exactly at the cap.
    # tile_rows=256, tile_cols=8, vec=1 -> units = (256//1) * (8//8) = 256.
    # block_size=1 -> per_thread = 256 == REGT_MAX_UNITS -> accepted.
    units, active, why = L.plan(
        tile_rows=256, tile_cols=8, block_size=1, wave_size=32, vec=1
    )
    assert not why, f"expected accepted at boundary, got: {why}"
    assert units == REGT_MAX_UNITS

    # Above boundary: 257 micro-tiles per thread.
    # tile_rows=257 is not divisible by vec=1 … use tile_rows=512, block_size=1.
    # units = (512 // 1) * (8 // 8) = 512 > REGT_MAX_UNITS -> rejected.
    _, _, why_over = L.plan(
        tile_rows=512, tile_cols=8, block_size=1, wave_size=32, vec=1
    )
    assert why_over, "expected rejection above REGT_MAX_UNITS"
    assert str(REGT_MAX_UNITS) in why_over
