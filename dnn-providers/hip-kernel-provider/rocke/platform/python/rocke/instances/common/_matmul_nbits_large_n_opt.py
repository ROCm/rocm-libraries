# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Combined-optimization large-N WMMA body for ``MatMulNBits`` (gfx1201).

This is an experimental variant of :mod:`._matmul_nbits_large_n` that folds in
three optimizations at once so their *combination* can be measured against the
baseline body on real hardware:

  (1)  **LDS-staged A** — the ``tile_m x tile_k`` activation tile is loaded once
      per K-group cooperatively into LDS, then every wave reads its A fragments
      from LDS. The baseline reloads A from global once per wave (warp_n-fold
      redundant); staging removes that redundancy at the cost of two workgroup
      barriers per group.

  (2)  **tile_k = group_size (scale on accumulator)** — the K tile spans exactly
      one int4 group (``tile_k == group_size == 32``), so the per-group scale is
      constant across the whole tile. The B fragments are dequantised to fp16
      **without** the scale (raw signed int4 -> f16, exact in ``[-8, 7]``); the
      group is contracted into a group-local f32 accumulator, then scaled once
      per output column (a single ``v_pk_fma`` over the ``<8 x f32>`` lane
      accumulator) and added into the main accumulator. This hoists the scale
      multiply out of the per-fragment dequant.

  (3)  **LDS epilogue transpose (vectorized stores)** — the WMMA accumulator is
      column-distributed on gfx12 (each lane owns one column x 8 rows), so the
      baseline epilogue emits 64 scalar ``global_store_b16``. Here the whole
      ``tile_m x tile_n`` output tile is written to LDS, then stored to global
      cooperatively as coalesced ``global_store_b128`` (8 halves/transaction).

Preconditions (validated by the caller / asserted here): gfx1201, ``tile_k ==
group_size``, and ``M`` a multiple of ``tile_m`` (the cooperative A load and the
LDS epilogue assume full tiles; the per-element store M-guard is dropped).
"""

from __future__ import annotations

from ...core.ir import F16, F32, I8, I32, IRBuilder, PtrType
from ...helpers.i4_dequant import unpack_i4_byte_to_pair_f16
from ._matmul_nbits_common import MatMulNBitsSpec, _scale_wire_dtype
from ._matmul_nbits_large_n import _wmma_params

__all__ = ["build_large_n_opt_matmul_nbits"]


def build_large_n_opt_matmul_nbits(
    spec: MatMulNBitsSpec, arch: str = "gfx1201"
) -> "object":
    """Build the combined-optimization large-N ``KernelDef`` for ``spec``."""
    wp = _wmma_params(arch)
    t = spec.tile
    N, K, group = spec.N, spec.K, spec.group_size

    if t.tile_k != group:
        raise ValueError(
            f"opt body requires tile_k == group_size; got tile_k={t.tile_k}, "
            f"group_size={group}"
        )
    wtm, wtn, wtk = t.warp_tile_m, t.warp_tile_n, t.warp_tile_k
    wave = spec.wave_size

    rows_per_wave = t.tile_m // t.warp_m
    cols_per_wave = t.tile_n // t.warp_n
    n_sub_m = rows_per_wave // wtm
    n_sub_n = cols_per_wave // wtn
    n_acc = n_sub_m * n_sub_n
    n_ksub = group // wtk  # WMMA K-substeps inside one group tile
    k_packed_stride = K // 2
    k_group_stride = K // group

    tile_m, tile_n, tile_k = t.tile_m, t.tile_n, t.tile_k
    bs = spec.block_size

    # Cooperative-load geometry: every thread moves `elems_per_thread` halves as
    # `frag_k`-wide (8) chunks. Both the A tile (tile_m*tile_k) and the C tile
    # (tile_m*tile_n) must tile evenly across the block in 8-wide vectors.
    a_elems = tile_m * tile_k
    c_elems = tile_m * tile_n
    if a_elems % (bs * 8) or c_elems % (bs * 8):
        raise ValueError(
            "opt body needs tile_m*tile_k and tile_m*tile_n divisible by "
            f"block_size*8 (got A={a_elems}, C={c_elems}, bs={bs})"
        )
    a_chunks = a_elems // (bs * 8)
    c_chunks = c_elems // (bs * 8)

    scale_t = F16 if _scale_wire_dtype(spec.scale_dtype) == "f16" else F32

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = bs

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    Sp = b.param(
        "Scales", PtrType(scale_t, "global"), noalias=True, readonly=True, align=8
    )
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841 — full-tile precondition; kept for ABI parity

    a_smem = b.smem_alloc(F16, [tile_m, tile_k], name_hint="A_smem")
    c_smem = b.smem_alloc(F16, [tile_m, tile_n], name_hint="C_smem")

    c0 = b.const_i32(0)
    cK = b.const_i32(K)
    cN = b.const_i32(N)
    c16 = b.const_i32(16)
    c8 = b.const_i32(
        8
    )  # noqa: F841 -- side-effecting: emits a kernel constant; keep for byte-identity
    c2 = b.const_i32(2)
    c_tile_k = b.const_i32(tile_k)
    c_group = b.const_i32(group)
    cwave = b.const_i32(wave)

    tid = b.thread_id_x()
    wave_id = b.div(tid, cwave)
    lane = b.mod(tid, cwave)
    frag = b.mod(lane, c16)  # lane%16: frag row/col selector
    half = b.div(lane, c16)  # lane/16: gfx12 K-half / column-distribution half

    half_k_elem = b.mul(half, b.const_i32(wp.frag_k)) if wp.split_k_by_half else c0
    half_k_byte = b.mul(half, b.const_i32(wp.frag_k // 2)) if wp.split_k_by_half else c0

    wave_m = b.div(wave_id, b.const_i32(t.warp_n))
    wave_n = b.mod(wave_id, b.const_i32(t.warp_n))

    m0 = b.mul(b.block_id_y(), b.const_i32(tile_m))
    n0 = b.mul(b.block_id_x(), b.const_i32(tile_n))
    wm_local = b.mul(wave_m, b.const_i32(rows_per_wave))  # LDS row base of wave
    wn_base = b.add(n0, b.mul(wave_n, b.const_i32(cols_per_wave)))  # global col base

    # Loop-invariant per-lane B row bases (packed-byte + scale offsets).
    b_byte_off = []
    b_scale_off = []
    for sn in range(n_sub_n):
        b_row = b.add(wn_base, b.add(b.const_i32(sn * wtn), frag))
        b_byte_off.append(b.mul(b_row, b.const_i32(k_packed_stride)))
        b_scale_off.append(b.mul(b_row, b.const_i32(k_group_stride)))

    acc0 = [b.zero_vec_f32(8) for _ in range(n_acc)]
    loop = b.scf_for_iter(
        c0, cK, c_tile_k, [(f"acc{i}", acc0[i]) for i in range(n_acc)], iv_name="k0"
    )
    with loop as (k0, accs):
        k_grp = b.div(k0, c_group)  # scale group index
        k_half_base = b.div(k0, c2)  # packed-byte base for this group

        # --- step 1: cooperatively stage the A tile [tile_m x tile_k] into LDS ---
        for ch in range(a_chunks):
            lin = b.add(b.mul(tid, b.const_i32(a_chunks * 8)), b.const_i32(ch * 8))
            r = b.div(lin, c_tile_k)
            c = b.mod(lin, c_tile_k)
            g_idx = b.add(b.add(b.mul(b.add(m0, r), cK), k0), c)
            a_vec = b.global_load_vN_f16(A, g_idx, 8)
            b.smem_store_vN_f16(a_smem, [r, c], a_vec, 8)
        b.sync()  # A tile visible to all waves

        # --- step 2: contract the whole group into a group-local f32 accumulator,
        #          feeding UNSCALED int4->f16 fragments to the WMMA atom. ---
        gacc = [b.zero_vec_f32(8) for _ in range(n_acc)]
        for ks in range(n_ksub):
            a_col = b.add(b.const_i32(ks * wtk), half_k_elem)
            a_frag = []
            for sm in range(n_sub_m):
                a_row = b.add(wm_local, b.add(b.const_i32(sm * wtm), frag))
                a_frag.append(b.smem_load_vN_f16(a_smem, a_row, a_col, n=wp.frag_k))

            n_bytes = wp.frag_k // 2
            b_byte_k = b.add(
                b.add(k_half_base, b.const_i32(ks * (wtk // 2))), half_k_byte
            )
            b_frag = []
            for sn in range(n_sub_n):
                packed = b.global_load_vN(
                    Bp, b.add(b_byte_off[sn], b_byte_k), I8, n_bytes
                )
                comps = []
                for j in range(n_bytes):
                    byte = b.vec_extract(packed, j)
                    lo, hi = unpack_i4_byte_to_pair_f16(b, byte)  # no scale here
                    comps.append(lo)
                    comps.append(hi)
                b_frag.append(b.vec_pack(comps, F16))

            wmma = getattr(b, wp.wmma_op)
            ng = []
            for sm in range(n_sub_m):
                for sn in range(n_sub_n):
                    idx = sm * n_sub_n + sn
                    ng.append(wmma(a_frag[sm], b_frag[sn], gacc[idx]))
            gacc = ng

        # --- scale the group accumulator by the per-column group scale and add
        #     into the main accumulator (one v_pk_fma over <8 x f32> per col). ---
        scale_vec = []
        for sn in range(n_sub_n):
            s = b.global_load(Sp, b.add(b_scale_off[sn], k_grp), scale_t)
            scale_vec.append(b.vector_splat(b.cast_to_f32(s), 8))
        nacc = []
        for sm in range(n_sub_m):
            for sn in range(n_sub_n):
                idx = sm * n_sub_n + sn
                nacc.append(b.vector_fma(gacc[idx], scale_vec[sn], accs[idx]))

        b.sync()  # all A-tile reads done before next group overwrites a_smem
        b.scf_yield(*nacc)

    results = loop.results

    # --- step 3: write the column-distributed accumulator to LDS, then store the
    #         whole tile to global as coalesced b128 (8 halves/transaction). ---
    wn_local = b.mul(wave_n, b.const_i32(cols_per_wave))  # LDS col base of wave
    for sm in range(n_sub_m):
        r0 = b.add(wm_local, b.const_i32(sm * wtm))
        for sn in range(n_sub_n):
            acc = results[sm * n_sub_n + sn]
            col = b.add(wn_local, b.add(b.const_i32(sn * wtn), frag))
            for i in range(8):
                h = b.trunc_f32_to_f16(b.vec_extract(acc, i))
                row = b.add(r0, b.add(half_k_elem, b.const_i32(i)))
                b.smem_store_vN_f16(c_smem, [row, col], h, 1)
    b.sync()

    c_tile_n = b.const_i32(tile_n)
    for ch in range(c_chunks):
        lin = b.add(b.mul(tid, b.const_i32(c_chunks * 8)), b.const_i32(ch * 8))
        r = b.div(lin, c_tile_n)
        c = b.mod(lin, c_tile_n)
        v = b.smem_load_vN_f16(c_smem, r, c, n=8)
        g_idx = b.add(b.add(b.mul(b.add(m0, r), cN), n0), c)
        b.global_store_vN_f16(C, g_idx, v, 8)

    return b.kernel
