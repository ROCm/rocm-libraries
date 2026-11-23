# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""XOR-swizzled LDS columns: ``lds_col_xor_params``, ``swizzle_lds_col`` and
``CoalescedTileLoader.col_xor``.

* The swizzle is evaluated on plain integers (a builder whose ops are Python
  arithmetic), so the bank claims of ``lds_col_xor_params`` are checked on the
  helper's own code: every row is permuted, and the rows one ``ds_load_b128``
  phase reads at one chunk -- and the chunks one row-major store phase writes
  -- hit distinct 16-byte bank groups at 32 and at 64 banks.
* The loader's stores are replayed the same way for every thread: each logical
  element lands exactly once, at the swizzled column of its own row, in the
  ``"col"``, ``"row"`` (transpose-on-store) and scalar paths.
* The C++ engine's mirror is byte-compared by the parity pair
  ``tests/instances/parity/coalesced_loader_col_xor_emit.{py,c}``.
"""

from __future__ import annotations

import dataclasses

import pytest
from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.loads import (
    CoalescedTileLoader,
    lds_col_xor_params,
    swizzle_lds_col,
)


class _IntBuilder:
    """The IRBuilder ops the swizzle and the sync loader use, on Python ints.

    ``buffer_load*`` return the logical tile coordinates the loaded vector
    holds (decoded from the byte offset the loader computed), and every LDS
    store is recorded.
    """

    def __init__(self, tile_cols: int, vector_axis: str):
        self.tile_cols = tile_cols
        self.vector_axis = vector_axis
        self.stores = []

    def const_i32(self, v):
        return int(v)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a // b

    def mod(self, a, b):
        return a % b

    def lshr(self, a, b):
        return a >> b

    def shl(self, a, b):
        return a << b

    def land(self, a, b):
        return a & b

    def xor(self, a, b):
        return a ^ b

    def select(self, c, a, b):
        return a if c else b

    def _logical(self, off_bytes, n):
        row, col = divmod(off_bytes // 2, self.tile_cols)
        if self.vector_axis == "row":
            return [(row + i, col) for i in range(n)]
        return [(row, col + i) for i in range(n)]

    def buffer_load(self, rsrc, off, soff, dtype):
        return self._logical(off, 1)

    def buffer_load_vN(self, rsrc, off, soff, dtype, n):
        return self._logical(off, n)

    def vec_extract(self, v, i):
        return [v[i]]

    def smem_store_vN(self, smem, idx, v, n):
        row, col = idx
        for i, logical in enumerate(v):
            self.stores.append(((row, col + i), logical))


def _swz(row, col, p):
    return swizzle_lds_col(_IntBuilder(0, "col"), row, col, p)


@pytest.mark.parametrize(
    "cols, want",
    [
        (16, (2, 1, 3, 3)),
        (32, (1, 3, 3, 3)),
        (64, (0, 7, 3, 3)),
        (128, (0, 15, 3, 3)),
    ],
)
def test_params(cols, want):
    assert lds_col_xor_params(cols) == want


@pytest.mark.parametrize(
    "cols, chunk",
    [(8, 8), (24, 8), (48, 8), (256, 8), (60, 8), (64, 6), (64, 0)],
)
def test_params_refuse_what_the_swizzle_cannot_cover(cols, chunk):
    with pytest.raises(ValueError):
        lds_col_xor_params(cols, chunk)


def _slot(row, phys_col, cols, banks_16b):
    """The 16-byte bank group (of ``banks_16b``) element (row, col) sits in."""
    return (row * cols + phys_col) // 8 % banks_16b


@pytest.mark.parametrize("cols", [16, 32, 64, 128])
def test_every_row_is_a_permutation(cols):
    p = lds_col_xor_params(cols)
    for row in range(64):
        assert sorted(_swz(row, c, p) for c in range(cols)) == list(range(cols))


def _accepted_params():
    """Every ``(cols, chunk)`` :func:`lds_col_xor_params` accepts, chunks 1-64."""
    out = []
    for chunk in (1 << k for k in range(7)):
        for c in (2, 4, 8, 16):
            out.append((c * chunk, chunk))
    return out


@pytest.mark.parametrize("cols, chunk", _accepted_params())
def test_every_accepted_chunk_permutes_each_row(cols, chunk):
    """What the API promises for every chunk it accepts: each row's columns are
    permuted within the row, whole chunks moving together. The bank claim is
    for 16-byte chunks only (the test below)."""
    p = lds_col_xor_params(cols, chunk)
    for row in range(64):
        phys = [_swz(row, c, p) for c in range(cols)]
        assert sorted(phys) == list(range(cols))
        for c in range(cols):
            assert phys[c] % chunk == c % chunk


@pytest.mark.parametrize("cols", [16, 32, 64, 128])
@pytest.mark.parametrize("lanes", [8, 16])
def test_read_and_store_phases_are_conflict_free(cols, lanes):
    """``lanes`` lanes x 16 B per LDS cycle: 32 banks at 8, 64 banks at 16.
    The default ``chunk = 8`` of 16-bit elements, the only chunk the bank
    claim of :func:`lds_col_xor_params` covers."""
    p = lds_col_xor_params(cols)
    chunks = cols // 8
    for r0 in range(0, 64, lanes):
        for ch in range(chunks):
            got = {
                _slot(r, _swz(r, ch * 8, p), cols, lanes) for r in range(r0, r0 + lanes)
            }
            assert len(got) == lanes, ("read", cols, r0, ch)
    for i0 in range(0, 64 * chunks, lanes):
        got = set()
        for i in range(i0, i0 + lanes):
            r, ch = divmod(i, chunks)
            got.add(_slot(r, _swz(r, ch * 8, p), cols, lanes))
        assert len(got) == lanes, ("store", cols, i0)


def test_the_unswizzled_tile_conflicts():
    """The control: the same reads of an unpadded 64-column tile all hit one
    bank group, which is what the swizzle (or a K pad) is for."""
    got = {_slot(r, 0, 64, 8) for r in range(8)}
    assert len(got) == 1


def _replay(ldr: CoalescedTileLoader, *, path: str = "load"):
    """Run ``ldr``'s stores for every thread; returns {physical: logical}."""
    seen = {}
    for tid in range(ldr.block_size):
        b = _IntBuilder(ldr.tile_cols, ldr.vector_axis)

        def descriptor(b_, row, col):
            return b_.add(b_.mul(row, ldr.tile_cols), col), None

        if path == "load":
            ldr.load(b, tid=tid, smem_dst=None, descriptor=descriptor, rsrc=object())
        else:
            staged = ldr.load_global(b, tid=tid, descriptor=descriptor, rsrc=object())
            ldr.store_lds(b, smem_dst=None, staged=staged)
        for phys, logical in b.stores:
            assert phys not in seen, (tid, phys)
            seen[phys] = logical
    return seen


@pytest.mark.parametrize(
    "rows, cols, block, vec, axis, path",
    [
        (64, 64, 256, 8, "col", "load"),
        (32, 16, 64, 8, "col", "load"),
        (64, 32, 256, 4, "col", "split"),
        (64, 32, 256, 8, "row", "load"),
        (32, 64, 128, 2, "row", "split"),
        (16, 16, 256, 1, "col", "load"),
    ],
)
def test_every_element_lands_once_at_its_rows_swizzled_column(
    rows, cols, block, vec, axis, path
):
    p = lds_col_xor_params(cols)
    ldr = CoalescedTileLoader(
        tile_rows=rows,
        tile_cols=cols,
        block_size=block,
        load_vec=vec,
        vector_axis=axis,
        col_xor=p,
    )
    seen = _replay(ldr, path=path)
    assert len(seen) == rows * cols
    for (prow, pcol), (lrow, lcol) in seen.items():
        assert prow == lrow
        assert pcol == _swz(lrow, lcol, p), ((prow, pcol), (lrow, lcol))


def _ir(ldr: CoalescedTileLoader) -> str:
    b = IRBuilder("col_xor_ir")
    b.kernel.attrs["max_workgroup_size"] = ldr.block_size
    x = b.param("X", PtrType(F16, "global"))
    n = b.param("N", I32)
    rsrc = b.buffer_rsrc(x, n)
    tid = b.thread_id_x()
    tile = b.smem_alloc(F16, [ldr.tile_rows, ldr.tile_cols], name_hint="tile")
    c_cols = b.const_i32(ldr.tile_cols)
    ldr.load(
        b,
        tid=tid,
        smem_dst=tile,
        descriptor=lambda b_, r, c: (b_.add(b_.mul(r, c_cols), c), None),
        rsrc=rsrc,
    )
    b.ret()
    return _lower_kernel_to_llvm_python(b.kernel, arch="gfx1201", llvm_flavor="llvm22")


def test_unset_col_xor_emits_no_swizzle():
    ldr = CoalescedTileLoader.from_tile(tile_rows=64, tile_cols=64, block_size=256)
    assert ldr.col_xor is None
    assert " xor " not in _ir(ldr)
    swz = dataclasses.replace(ldr, col_xor=lds_col_xor_params(64))
    assert " xor " in _ir(swz)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"col_xor": (0, 7, 3)}, "four non-negative ints"),
        ({"col_xor": (0, 7, 3, -1)}, "four non-negative ints"),
        ({"col_xor": (0, 7, 3, True)}, "four non-negative ints"),
        ({"col_xor": (0, 15, 3, 3)}, "row does not hold whole groups"),
        # a single 64-column chunk: the flip bit alone would move it out of the row
        ({"col_xor": (0, 0, 6, 3)}, "row does not hold whole groups"),
        ({"col_xor": (32, 7, 3, 3)}, "shifts must fit an i32"),
        ({"col_xor": (0, 7, 3, 32)}, "shifts must fit an i32"),
        ({"col_xor": (0, 7, 31, 3)}, "shifts must fit an i32"),
        ({"col_xor": (0, 15, 2, 3)}, "divide its 4-column chunk"),
    ],
)
def test_loader_refuses_a_swizzle_it_cannot_store(kw, match):
    with pytest.raises(ValueError, match=match):
        CoalescedTileLoader(
            tile_rows=64, tile_cols=64, block_size=256, load_vec=8, **kw
        )


def test_loader_refuses_a_row_of_partial_xor_groups():
    # 48 columns = 6 chunks; the 32-column swizzle permutes groups of 4, so
    # chunks 4 and 5 would be XORed onto columns 48-63, outside the row.
    with pytest.raises(ValueError, match="row does not hold whole groups"):
        CoalescedTileLoader(
            tile_rows=16,
            tile_cols=48,
            block_size=64,
            load_vec=8,
            col_xor=lds_col_xor_params(32),
        )


def test_loader_refuses_a_row_of_partial_chunks():
    # 36 columns = 4 whole 8-column chunks (one whole XOR group) plus 4
    # columns; only the ragged-row clause catches it. Without it the tail's
    # stores land up to column 59, outside the row.
    with pytest.raises(ValueError, match="row does not hold whole groups"):
        CoalescedTileLoader(
            tile_rows=64,
            tile_cols=36,
            block_size=64,
            load_vec=4,
            col_xor=lds_col_xor_params(32),
        )


@pytest.mark.parametrize(
    "cols, chunk",
    [(16, 8), (32, 8), (64, 8), (128, 8), (8, 4), (16, 4), (32, 4), (64, 4)],
)
def test_loader_takes_every_lds_col_xor_params_tuple(cols, chunk):
    CoalescedTileLoader(
        tile_rows=16,
        tile_cols=cols,
        block_size=64,
        load_vec=4,
        col_xor=lds_col_xor_params(cols, chunk),
    )


def test_row_axis_takes_any_vector_width():
    # The scatter writes one element per row, so the chunk does not bound vec.
    CoalescedTileLoader(
        tile_rows=64,
        tile_cols=64,
        block_size=256,
        load_vec=8,
        vector_axis="row",
        col_xor=lds_col_xor_params(64, chunk=4),
    )


def test_async_copy_refuses_col_xor():
    ldr = CoalescedTileLoader(
        tile_rows=16,
        tile_cols=32,
        block_size=64,
        load_vec=8,
        col_xor=lds_col_xor_params(32),
    )
    b = IRBuilder("k")
    with pytest.raises(ValueError, match="does not support col_xor"):
        ldr.load_async_to_lds(
            b,
            tid=b.thread_id_x(),
            smem_dst=b.smem_alloc(F16, [16, 32]),
            descriptor=lambda b_, r, c: (c, None),
            ptr=b.param("X", PtrType(F16, "global")),
            num_bytes=b.const_i32(1024),
        )
