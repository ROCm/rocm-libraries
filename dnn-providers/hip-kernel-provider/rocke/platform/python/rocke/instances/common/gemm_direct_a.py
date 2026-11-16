# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1201 WMMA GEMM that reads A straight into registers (``pipeline="direct_a"``).

``C[M, N] = A[M, K] @ B[N, K].T`` with the standard ``(A, B, C, M, N, K)`` ABI,
followed by ``(lda, ldb)`` when ``trait.row_strides`` is set.

* Every wave owns its own M rows (``warp_n == 1``), so each wave loads its A
  fragments from global memory in the matrix operand layout. A never enters
  LDS and no A traffic is duplicated between waves.
* B, which all waves share, goes through one LDS buffer. The next B tile is
  stored after the last K atom's LDS reads and before that atom's matrix
  operations, so the overwrite rendezvous runs under matrix work.
* A and B are read and C is written through bounded buffer resources. A load
  past the end of A or B reads zero and a store outside C is dropped, which is
  what makes partial M and N tiles correct with no extra code in the loop.
* K is whole 16-element atoms, not whole tiles. The loop runs
  ``ceil(K / tile_k)`` tiles and the K atoms past K load zero from both
  operands, through an out-of-range offset.
* ``trait.row_strides`` adds the row strides of A and B, in elements, as
  kernel arguments. One A fragment load touches 16 rows, and when those rows
  are 16 KiB apart (16-bit elements at a stride that is a multiple of 8192)
  they alias in the caches. A caller that owns the allocation avoids that with
  a padded stride. B's stride matters as much as A's.

A and B are FP16 or BF16 (the same type); C is that type or FP32, which stores
the accumulator unrounded.

Launch contract: ``grid = (ceil(N / tile_n), ceil(M / tile_m))``,
``block = spec.block_size``, ``K % 16 == 0``, A and B row strides ``lda`` and
``ldb`` of at least K (exactly K without ``row_strides``), packed C (row
stride N), and every byte span below 2 GiB with M and N rounded up to whole
tiles: ``2 * ceil_tile(M) * lda``, ``2 * ceil_tile(N) * ldb`` and
``c_bytes * M * N``. :func:`direct_a_launch_supported` is that check.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ...core.arch import ArchTarget
from ...core.ir import BF16, F16, F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.grid import super_tile_swizzle_dynamic

__all__ = [
    "build_direct_a_gemm",
    "direct_a_launch_supported",
    "direct_a_local_prefetch",
    "is_valid_direct_a_spec",
]

_MAX_SPAN_BYTES = (1 << 31) - 1
# A byte offset no span of the launch contract reaches, and that the largest
# in-contract offset can be added to without leaving 32 bits.
_OUT_OF_RANGE = 0x7FFFFFF0
# Accumulators, two A tiles (current and next) and the B fragments held at once
# must leave room for addresses and temporaries in the 256-VGPR file.
_LOCAL_PREFETCH_VGPR_BUDGET = 176
# The same, with the next B tile's vectors a thread holds between its loads
# and its LDS stores and the loop-invariant offsets of every load and store.
# Built kernels use about 28 registers more; past that the allocator spills
# in the K loop.
_VGPR_BUDGET = 228
_AB_DTYPES = {"fp16": F16, "f16": F16, "bf16": BF16}
_F32_NAMES = ("fp32", "f32")


def _c_is_f32(spec) -> bool:
    return spec.data.dtype_c in _F32_NAMES


def _fragment_vgprs(spec, local_prefetch: bool) -> int:
    t = spec.tile
    mm, nn, nk = t.mfmas_per_warp_m, t.mfmas_per_warp_n, t.k_atoms_per_tile_k
    return 8 * mm * nn + 2 * 4 * mm * nk + 4 * nn * (2 if local_prefetch else 1)


def _held_vgprs(spec, local_prefetch: bool) -> int:
    """Registers a thread holds across the K loop: the fragments, the next B
    tile's vectors it copies to LDS, and one offset per A load, B load and B
    store."""
    t = spec.tile
    b_vectors = t.tile_n * t.tile_k // (8 * spec.block_size)
    offsets = t.mfmas_per_warp_m * t.k_atoms_per_tile_k + 2 * b_vectors
    return _fragment_vgprs(spec, local_prefetch) + 4 * b_vectors + offsets


def direct_a_local_prefetch(spec) -> bool:
    """Read K atom ``kk + 1``'s B fragments before atom ``kk``'s matrix ops.

    Hides the LDS read latency, at the cost of a second set of B fragments.
    Chosen from the tile alone, so a spec always builds the same kernel.
    """
    return (
        _fragment_vgprs(spec, True) <= _LOCAL_PREFETCH_VGPR_BUDGET
        and _held_vgprs(spec, True) <= _VGPR_BUDGET
    )


def direct_a_launch_supported(
    spec,
    M: int,
    N: int,
    K: int,
    lda: Optional[int] = None,
    ldb: Optional[int] = None,
) -> Tuple[bool, str]:
    """Host check of the launch contract for runtime sizes ``(M, N, K)``.

    ``lda`` and ``ldb`` are the row strides of A and B in elements; ``None``
    is the packed stride K.
    """
    t = spec.tile
    if min(M, N, K) <= 0:
        return False, "M, N and K must be positive"
    if K % t.warp_tile_k:
        return False, f"K must be a multiple of {t.warp_tile_k}"
    lda = K if lda is None else lda
    ldb = K if ldb is None else ldb
    if min(lda, ldb) < K:
        return False, "the A and B row strides must be at least K"
    if not spec.trait.row_strides and (lda, ldb) != (K, K):
        return False, "A and B must be packed (row stride K) without row_strides"
    pad_m = -(-M // t.tile_m) * t.tile_m
    pad_n = -(-N // t.tile_n) * t.tile_n
    c_bytes = 4 if _c_is_f32(spec) else 2
    if max(2 * pad_m * lda, 2 * pad_n * ldb, c_bytes * M * N) > _MAX_SPAN_BYTES:
        return False, "A, B and C byte spans (whole tiles) must each be below 2 GiB"
    return True, "ok"


def is_valid_direct_a_spec(spec, arch: str = "gfx1201") -> Tuple[bool, str]:
    from .gemm_universal import _atom_frag_lengths, _resolve_mma_op

    t, tr, d = spec.tile, spec.trait, spec.data
    if arch != "gfx1201" or spec.wave_size != 32:
        return False, "direct_a is a gfx1201 wave32 pipeline"
    if d.dtype_a not in _AB_DTYPES or d.dtype_b != d.dtype_a:
        return False, "direct_a requires FP16 or BF16 A and B of the same type"
    if d.dtype_c != d.dtype_a and d.dtype_c not in _F32_NAMES:
        return False, "direct_a requires C in the A/B type or FP32"
    if d.dtype_acc not in ("fp32", "f32") or d.layout != "RCR":
        return False, "direct_a requires an fp32 accumulator and the RCR layout"
    if tr.scheduler != "intrawave" or tr.epilogue != "default":
        return False, "direct_a requires the intrawave scheduler and default epilogue"
    if (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k, t.warp_k) != (16, 16, 16, 1):
        return False, "direct_a requires the 16x16x16 WMMA atom and warp_k=1"
    if t.warp_n != 1:
        return False, "direct_a requires warp_n=1 (each wave owns its M rows)"
    if min(t.tile_m, t.tile_n, t.tile_k, t.warp_m) <= 0:
        return False, "tile dimensions must be positive"
    if t.tile_m % (t.warp_m * 16) or t.tile_n % 16 or t.tile_k % 16:
        return False, "tile must be whole 16x16x16 atoms per wave"
    if spec.block_size != t.warp_m * spec.wave_size:
        return False, "block_size must be warp_m waves"
    if (t.tile_n * t.tile_k) % (8 * spec.block_size):
        return False, "the B tile must be whole 8-element vectors per thread"
    if spec.batched or tr.persistent or tr.split_k != 1:
        return False, "direct_a does not support batching, persistence or split-K"
    if getattr(spec, "_fused_epilogue", None) is not None:
        return False, "direct_a does not support a fused epilogue"
    unsupported = (
        tr.pad_m,
        tr.pad_n,
        tr.pad_k,
        tr.chiplet_swizzle,
        tr.preshuffle_b,
        tr.direct_to_lds,
        tr.dtl_prefetch,
        tr.tdm,
        tr.active_tile_skip,
        tr.lds_store_after_mma,
        tr.lds_db_allow_1wg,
        tr.cshuffle_no_alias,
        tr.split_barrier,
        tr.lds_k_pad,
        tr.ab_load_elem_bytes is not None,
        tr.global_addressing != "legacy",
    )
    if any(unsupported):
        return False, (
            "direct_a takes only group_m, k_stagger, lds_swizzle, row_strides and "
            "waves_per_eu; partial M/N tiles, the K tail and bounded addressing "
            "are built in"
        )
    if tr.group_m < 0 or tr.k_stagger < 0 or tr.k_stagger_step < 1:
        return (
            False,
            "group_m and k_stagger must be nonnegative, k_stagger_step positive",
        )
    if tr.lds_swizzle and (t.tile_k // 8) & (t.tile_k // 8 - 1):
        return (
            False,
            "lds_swizzle needs a power-of-two count of 8-element slots per row",
        )
    op = _resolve_mma_op(spec, arch)
    if op is None or _atom_frag_lengths(op) != (8, 8, 8):
        return False, "direct_a requires the gfx12 eight-element WMMA fragments"
    if not ArchTarget.from_gfx(arch).fits_lds(2 * t.tile_n * t.tile_k):
        return False, "the B tile exceeds the LDS capacity"
    if _held_vgprs(spec, False) > _VGPR_BUDGET:
        return False, "the tile's accumulators and fragments exceed the VGPR file"
    return True, "ok"


def build_direct_a_gemm(spec, arch: str = "gfx1201") -> KernelDef:
    from .gemm_universal import (
        _atom_frag_lengths,
        _emit_mma,
        _emit_zero_acc_op,
        _resolve_mma_op,
        wmma_lds_swizzle_row_shift,
    )

    ok, why = is_valid_direct_a_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid GEMM spec: {why}")
    t, tr = spec.tile, spec.trait
    op = _resolve_mma_op(spec, arch)
    _, _, c_len = _atom_frag_lengths(op)
    mm, nn, nk = t.mfmas_per_warp_m, t.mfmas_per_warp_n, t.k_atoms_per_tile_k
    local_prefetch = direct_a_local_prefetch(spec)

    b = IRBuilder(spec.kernel_name())
    b.fence_barriers = "compiler"
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if tr.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = tr.waves_per_eu
    ab_type = _AB_DTYPES[spec.data.dtype_a]
    c_f32 = _c_is_f32(spec)
    c_type = F32 if c_f32 else ab_type
    A = b.param("A", PtrType(ab_type, "global"), noalias=True, readonly=True, align=16)
    B = b.param("B", PtrType(ab_type, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(c_type, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)
    N = b.param("N", I32)
    K = b.param("K", I32)
    # Row strides of A and B in elements; K when both are packed.
    lda = b.param("lda", I32) if tr.row_strides else K
    ldb = b.param("ldb", I32) if tr.row_strides else K

    zero = b.const_i32(0)
    two = b.const_i32(2)
    c_bytes = b.const_i32(4 if c_f32 else 2)
    bk = b.const_i32(t.tile_k)
    far = b.const_i32(_OUT_OF_RANGE)
    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(spec.wave_size))
    lane = b.mod(tid, b.const_i32(spec.wave_size))
    warp_m = b.mul(wave, b.const_i32(mm * t.warp_tile_m))

    # Whole and partial tiles: the grid is ceil(extent / tile).
    n_tiles = b.div(b.add(N, b.const_i32(t.tile_n - 1)), b.const_i32(t.tile_n))
    flat_wg = b.add(b.mul(b.block_id_y(), n_tiles), b.block_id_x())
    if tr.group_m:
        m_tiles = b.div(b.add(M, b.const_i32(t.tile_m - 1)), b.const_i32(t.tile_m))
        raster = super_tile_swizzle_dynamic(
            b, flat_wg, num_pid_m=m_tiles, num_pid_n=n_tiles, wgm=tr.group_m
        )
        pid_m, pid_n = raster.row, raster.col
    else:
        pid_m, pid_n = b.block_id_y(), b.block_id_x()
    block_m = b.to_sgpr_u32(b.mul(pid_m, b.const_i32(t.tile_m)))
    block_n = b.to_sgpr_u32(b.mul(pid_n, b.const_i32(t.tile_n)))

    # Bounded resources. The launch contract keeps every byte count in i32.
    # A strided operand ends with its last row's K elements.
    def span_bytes(rows: Value, stride: Value) -> Value:
        if not tr.row_strides:
            return b.mul(b.mul(rows, K), two)
        last_row = b.mul(b.sub(rows, b.const_i32(1)), stride)
        return b.mul(b.add(last_row, K), two)

    a_rsrc = b.buffer_rsrc(A, span_bytes(M, lda))
    b_rsrc = b.buffer_rsrc(B, span_bytes(N, ldb))
    c_rsrc = b.buffer_rsrc(C, b.mul(b.mul(M, N), c_bytes))

    a_row, a_k = op.a_layout().coord(b, lane, 0)
    b_k, b_col = op.b_layout().coord(b, lane, 0)

    # Loop-invariant byte offsets of this lane's A fragments (K offset excluded).
    a_base: List[Value] = []
    for kk in range(nk):
        for mi in range(mm):
            row = b.add(
                b.add(b.add(block_m, warp_m), b.const_i32(mi * t.warp_tile_m)), a_row
            )
            col = b.add(a_k, b.const_i32(kk * t.warp_tile_k))
            a_base.append(b.mul(b.add(b.mul(row, lda), col), two))

    # This thread's share of the B tile copy: 8-element vectors, row-major.
    slots = t.tile_k // 8
    b_smem = b.smem_alloc(ab_type, [t.tile_n, t.tile_k], name_hint="B_smem")
    swizzle_shift = wmma_lds_swizzle_row_shift(2 * t.tile_k)

    def swizzle(col: Value, row: Value) -> Value:
        if not tr.lds_swizzle:
            return col
        key = b.lshr(row, b.const_i32(swizzle_shift)) if swizzle_shift else row
        return b.xor(col, b.shl(b.mod(key, b.const_i32(slots)), b.const_i32(3)))

    # When the block holds whole rows of vectors, a thread's vectors all sit
    # at one K column, so one comparison per tile masks all of them.
    one_b_col = spec.block_size % slots == 0
    b_base: List[Value] = []
    b_cols: List[Value] = []
    b_dest: List[Tuple[Value, Value]] = []
    for e in range(t.tile_n * t.tile_k // (8 * spec.block_size)):
        vi = b.add(b.const_i32(e * spec.block_size), tid)
        row = b.div(vi, b.const_i32(slots))
        col = b.mul(b.mod(vi, b.const_i32(slots)), b.const_i32(8))
        b_base.append(b.mul(b.add(b.mul(b.add(block_n, row), ldb), col), two))
        b_cols.append(col)
        b_dest.append((row, swizzle(col, row)))

    # K as whole tiles. The whole tiles run first, in the stagger's order, and
    # a partial last tile after them, so only the prologue and one loop trip
    # pay for the masks. ``k_wrap`` is the whole-tile range the stagger walks,
    # one tile when K has none.
    k_ceil = b.mul(b.div(b.add(K, b.const_i32(t.tile_k - 1)), bk), bk)
    k_floor = b.mul(b.div(K, bk), bk)
    k_wrap = b.select(b.cmp_lt(k_floor, bk), bk, k_floor)

    stagger: Optional[Value] = None
    if tr.k_stagger:
        stagger = b.to_sgpr_u32(
            b.mod(
                b.mul(
                    b.mod(flat_wg, b.const_i32(tr.k_stagger)),
                    b.const_i32(tr.k_stagger_step * t.tile_k),
                ),
                k_wrap,
            )
        )

    def k_tile(k: Value) -> Value:
        if stagger is None:
            return k
        shifted = b.add(k, stagger)
        return b.select(b.cmp_ge(shifted, k_wrap), b.sub(shifted, k_wrap), shifted)

    def load_a(k: Value, partial: bool = False) -> List[Value]:
        k_bytes = b.mul(k, two)
        offsets = [k_bytes] * nk
        if partial:
            # One scalar offset per K atom: the tile's, or out of range past K.
            for kk in range(1, nk):
                inside = b.cmp_lt(b.add(k, b.const_i32(kk * t.warp_tile_k)), K)
                offsets[kk] = b.select(inside, k_bytes, far)
        return [
            b.buffer_load_vN(a_rsrc, base, offsets[index // mm], ab_type, 8)
            for index, base in enumerate(a_base)
        ]

    def load_b(k: Value, partial: bool = False) -> List[Value]:
        k_bytes = b.mul(k, two)
        if not partial:
            return [
                b.buffer_load_vN(b_rsrc, base, k_bytes, ab_type, 8) for base in b_base
            ]
        left = b.sub(K, k)
        inside = b.cmp_lt(b_cols[0], left) if one_b_col else None
        out = []
        for base, col in zip(b_base, b_cols):
            keep = inside if one_b_col else b.cmp_lt(col, left)
            offset = b.select(keep, base, far)
            out.append(b.buffer_load_vN(b_rsrc, offset, k_bytes, ab_type, 8))
        return out

    def store_b(values: Sequence[Value]) -> None:
        for (row, col), value in zip(b_dest, values):
            b.smem_store_vN(b_smem, [row, col], value, 8)

    def read_b(kk: int) -> List[Value]:
        fragments = []
        for ni in range(nn):
            row = b.add(b.const_i32(ni * t.warp_tile_n), b_col)
            col = swizzle(b.add(b_k, b.const_i32(kk * t.warp_tile_k)), row)
            fragments.append(b.smem_load_vN(b_smem, row, col, dtype=ab_type, n=8))
        return fragments

    def compute(
        a_fragments: Sequence[Value], accs: Sequence[Value], overwrite=None
    ) -> List[Value]:
        out = list(accs)
        current = read_b(0) if local_prefetch else None
        for kk in range(nk):
            bb = current if local_prefetch else read_b(kk)
            following = None
            if local_prefetch and kk + 1 < nk:
                following = read_b(kk + 1)
                b.sched_barrier(0)
            if overwrite is not None and kk == nk - 1:
                b.sched_barrier(0)
                overwrite()
                b.sched_barrier(0)
            for mi in range(mm):
                for ni in range(nn):
                    index = mi * nn + ni
                    out[index] = _emit_mma(
                        b, op, a_fragments[kk * mm + mi], bb[ni], out[index]
                    )
            if local_prefetch:
                b.sched_barrier(0)
                current = following
        return out

    # The first tile is the partial one when K is shorter than a tile.
    first_k = k_tile(zero)
    first_a = load_a(first_k, partial=True)
    store_b(load_b(first_k, partial=True))
    acc_init = _emit_zero_acc_op(b, op)
    a_count = len(first_a)
    args = [(f"a_{i}", value) for i, value in enumerate(first_a)]
    args += [(f"acc_{i}", acc_init) for i in range(mm * nn)]

    def step(
        progress: Value,
        a_current: Sequence[Value],
        accs: Sequence[Value],
        partial: bool = False,
    ):
        """One K tile: fetch the next tile, then compute the current one."""
        following = b.add(progress, bk)
        next_k = following if partial else k_tile(following)
        # Publish this tile's B before any wave reads it.
        b.sync()
        next_a = load_a(next_k, partial)
        pending = load_b(next_k, partial)
        b.sched_barrier(0)

        def overwrite() -> None:
            # Every wave has drained its last LDS reads of this tile.
            b.sync()
            store_b(pending)

        updated = compute(a_current, accs, overwrite)
        b.sched_barrier(0)
        return next_a, updated

    def one_tile_loop(start: Value, stop: Value, carried_in, tag: str, partial: bool):
        loop_args = [
            (f"a_{tag}_{i}", value) for i, value in enumerate(carried_in[:a_count])
        ]
        loop_args += [
            (f"acc_{tag}_{i}", value) for i, value in enumerate(carried_in[a_count:])
        ]
        loop = b.scf_for_iter(
            start, stop, bk, loop_args, iv_name=f"k_{tag}", elide_trailing_barrier=False
        )
        with loop as (progress, carried):
            a_1, acc_1 = step(progress, carried[:a_count], carried[a_count:], partial)
            b.scf_yield(*a_1, *acc_1)
        return loop.results

    # Two K tiles per iteration. The A fragments fetched in the second half
    # are the loop-carried ones, and the first half's registers are free by
    # then, so the back edge needs no register copies. A one-tile loop covers
    # an odd remainder, another fetches a partial last tile (at most one trip
    # each), and the last tile is computed after them.
    two_bk = b.const_i32(2 * t.tile_k)
    span = b.sub(k_wrap, bk)
    paired = b.mul(b.div(span, two_bk), two_bk)
    loop = b.scf_for_iter(
        zero, paired, two_bk, args, iv_name="k_progress", elide_trailing_barrier=False
    )
    with loop as (progress, carried):
        a_1, acc_1 = step(progress, carried[:a_count], carried[a_count:])
        a_2, acc_2 = step(b.add(progress, bk), a_1, acc_1)
        b.scf_yield(*a_2, *acc_2)
    odd = one_tile_loop(paired, span, loop.results, "odd", False)
    final = one_tile_loop(span, b.sub(k_ceil, bk), odd, "tail", True)
    b.sync()
    result = compute(final[:a_count], final[a_count:])

    # Accumulator scatter. Each slot's (row, col) comes from the op's C layout.
    # A cell outside C gets an out-of-range byte offset, which the buffer
    # store drops: that is the whole partial-tile epilogue.
    oob = b.const_i32(-1)
    c_map = op.c_layout()
    m_base = b.add(block_m, warp_m)
    store = {
        F32: b.buffer_store_f32,
        F16: b.buffer_store_f16,
        BF16: b.buffer_store_bf16,
    }[c_type]
    flat = 0
    for mi in range(mm):
        atom_m = b.add(m_base, b.const_i32(mi * t.warp_tile_m))
        for ni in range(nn):
            acc = result[flat]
            flat += 1
            atom_n = b.add(block_n, b.const_i32(ni * t.warp_tile_n))
            values = acc if c_f32 else b.vec_cast_f32_to(acc, c_type)
            for slot in range(c_len):
                row_in_atom, col_in_atom = c_map.coord(b, lane, slot)
                row = b.add(atom_m, row_in_atom)
                col = b.add(atom_n, col_in_atom)
                inside = b.land(b.cmp_lt(row, M), b.cmp_lt(col, N))
                offset = b.mul(b.add(b.mul(row, N), col), c_bytes)
                store(
                    c_rsrc,
                    b.select(inside, offset, oob),
                    zero,
                    b.vec_extract(values, slot),
                )
    return b.kernel
