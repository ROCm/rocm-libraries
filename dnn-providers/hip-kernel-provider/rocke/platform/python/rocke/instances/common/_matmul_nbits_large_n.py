# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Large-N WMMA tiled body for the ``MatMulNBits`` gfx1151 instance (Milestone 3).

Correctness-first ``C[M, N] = A[M, K] @ dequant(B, scales)^T`` for the large-N
family (``N`` is a multiple of ``tile_n``; ``K`` a multiple of ``tile_k``). The
kernel feeds the gfx1151 ``wmma_f32_16x16x16_f16`` atom: ``A`` is loaded as fp16
``<16 x half>`` fragments directly from global, and each packed-int4 ``B``
fragment is dequantised on the fly to an fp16 ``<16 x half>`` via
:func:`rocke.helpers.i4_dequant.dequant_i4_byte_to_f16_pair`.

Block / wave mapping (one block owns a ``tile_m × tile_n`` output tile):

  * grid ``(ceil(N / tile_n), ceil(M / tile_m), 1)`` — ``block_id.x`` selects the
    N tile, ``block_id.y`` the M tile (matches :func:`matmul_nbits_grid` and the
    manifest ``grid_order="NM"``);
  * ``warp_m × warp_n`` waves per block, ``wave_size`` (32) lanes each; wave
    ``w`` owns rows ``[wave_m*rpw, +rpw)`` × cols ``[wave_n*cpw, +cpw)`` of the
    block tile, where ``rpw = tile_m/warp_m`` and ``cpw = tile_n/warp_n``;
  * each wave walks ``rpw/16 × cpw/16`` WMMA 16×16 output sub-tiles, carrying one
    ``<8 x float>`` accumulator per sub-tile through the K loop.

Scale lookup: a 16-wide K fragment starting at a 16-aligned ``k0`` lies wholly
inside one ``group_size=32`` group, so every fragment uses the single scale
``Scales[b_row, k0 / group]``.

This is the no-shared-memory reference body; LDS staging / wider M tiling is a
follow-on. It assumes ``M`` is a multiple of ``tile_m`` for the in-bounds A
loads (the store is row-guarded); true partial-M tail handling lands later.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ir import F16, F32, I8, I32, IRBuilder, PtrType
from ...helpers.i4_dequant import dequant_i4_byte_to_f16_pair
from ._matmul_nbits_common import MatMulNBitsSpec, V1_ARCH, _scale_wire_dtype

__all__ = ["build_large_n_matmul_nbits"]


@dataclass(frozen=True)
class _WmmaParams:
    """Per-arch WMMA 16x16x16 f16 fragment ABI for the matmul_nbits body.

    The RDNA3/3.5 (gfx1151) and RDNA4 (gfx1201) WMMA atoms differ in operand
    width and accumulator distribution:

      * gfx1151: A/B are ``<16 x half>`` per lane (cross-half duplication); the
        full K=16 lives in every lane. Accumulator is row-distributed:
        ``out_row = r0 + 2*i + (lane//16)``.
      * gfx1201: A/B are ``<8 x half>`` per lane (no duplication); the 16
        K-elements split across lane-halves (lanes 0-15 hold K 0..7, lanes
        16-31 hold K 8..15). Accumulator is column-distributed:
        ``out_row = r0 + (lane//16)*8 + i``.
    """

    wmma_op: str  # IRBuilder method name for the WMMA atom
    frag_k: int  # fp16 elements per lane in one A/B fragment (16 | 8)
    split_k_by_half: bool  # gfx12: K offset within a step = (lane//16)*frag_k


def _wmma_params(arch: str) -> _WmmaParams:
    if arch == "gfx1201":
        return _WmmaParams(
            wmma_op="wmma_gfx12_f32_16x16x16_f16",
            frag_k=8,
            split_k_by_half=True,
        )
    return _WmmaParams(
        wmma_op="wmma_f32_16x16x16_f16",
        frag_k=16,
        split_k_by_half=False,
    )


def build_large_n_matmul_nbits(spec: MatMulNBitsSpec, arch: str = V1_ARCH) -> "object":
    """Build the large-N WMMA ``KernelDef`` for ``spec`` (validated by caller)."""
    wp = _wmma_params(arch)
    t = spec.tile
    N, K, group = spec.N, spec.K, spec.group_size
    wtm, wtn, wtk = t.warp_tile_m, t.warp_tile_n, t.warp_tile_k
    wave = spec.wave_size

    rows_per_wave = t.tile_m // t.warp_m
    cols_per_wave = t.tile_n // t.warp_n
    n_sub_m = rows_per_wave // wtm
    n_sub_n = cols_per_wave // wtn
    k_packed_stride = K // 2  # packed-byte row stride for B (two int4 per byte)
    k_group_stride = K // group  # scale row stride

    scale_t = F16 if _scale_wire_dtype(spec.scale_dtype) == "f16" else F32

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    Sp = b.param(
        "Scales", PtrType(scale_t, "global"), noalias=True, readonly=True, align=8
    )
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)

    c0 = b.const_i32(0)
    cK = b.const_i32(K)
    cN = b.const_i32(N)
    c16 = b.const_i32(16)
    c2 = b.const_i32(2)
    cwtk = b.const_i32(wtk)
    cwave = b.const_i32(wave)
    cgroup = b.const_i32(group)

    tid = b.thread_id_x()
    wave_id = b.div(tid, cwave)
    lane = b.mod(tid, cwave)
    frag = b.mod(lane, c16)  # lane%16: A-frag row / B-frag col / output col
    half = b.div(lane, c16)  # lane/16: output-row selector / gfx12 K-half

    # gfx12 splits the WMMA's K=16 across lane-halves (lanes 0-15 -> K 0..7,
    # 16-31 -> K 8..15); per-lane K offset within a step is half*frag_k. gfx11
    # carries the full K in every lane (no per-half offset).
    half_k_elem = b.mul(half, b.const_i32(wp.frag_k)) if wp.split_k_by_half else c0
    half_k_byte = b.mul(half, b.const_i32(wp.frag_k // 2)) if wp.split_k_by_half else c0

    wave_m = b.div(wave_id, b.const_i32(t.warp_n))
    wave_n = b.mod(wave_id, b.const_i32(t.warp_n))

    m0 = b.mul(b.block_id_y(), b.const_i32(t.tile_m))
    n0 = b.mul(b.block_id_x(), b.const_i32(t.tile_n))
    wm_base = b.add(m0, b.mul(wave_m, b.const_i32(rows_per_wave)))
    wn_base = b.add(n0, b.mul(wave_n, b.const_i32(cols_per_wave)))

    # Loop-invariant per-lane row bases (element / byte / group offsets).
    a_row_off = []  # A element offset of (sub-tile row + frag), pre-* K
    for sm in range(n_sub_m):
        a_row = b.add(wm_base, b.add(b.const_i32(sm * wtm), frag))
        a_row_off.append(b.mul(a_row, cK))
    b_byte_off = []  # B packed-byte offset of (col + frag) row
    b_scale_off = []  # B scale offset of (col + frag) row
    for sn in range(n_sub_n):
        b_row = b.add(wn_base, b.add(b.const_i32(sn * wtn), frag))
        b_byte_off.append(b.mul(b_row, b.const_i32(k_packed_stride)))
        b_scale_off.append(b.mul(b_row, b.const_i32(k_group_stride)))

    n_acc = n_sub_m * n_sub_n
    acc0 = [b.zero_vec_f32(8) for _ in range(n_acc)]
    loop = b.scf_for_iter(
        c0, cK, cwtk, [(f"acc{i}", acc0[i]) for i in range(n_acc)], iv_name="k0"
    )
    with loop as (k0, accs):
        k_half = b.div(k0, c2)  # packed-byte K offset
        k_grp = b.div(k0, cgroup)  # scale group index

        a_frag = [
            b.global_load_vN_f16(
                A, b.add(b.add(a_row_off[sm], k0), half_k_elem), wp.frag_k
            )
            for sm in range(n_sub_m)
        ]

        n_bytes = wp.frag_k // 2  # packed B bytes feeding one fragment
        b_frag = []
        for sn in range(n_sub_n):
            packed = b.global_load_vN(
                Bp, b.add(b.add(b_byte_off[sn], k_half), half_k_byte), I8, n_bytes
            )
            scale = b.global_load(Sp, b.add(b_scale_off[sn], k_grp), scale_t)
            scale_f32 = b.cast_to_f32(scale)
            comps = []
            for j in range(n_bytes):
                byte = b.vec_extract(packed, j)
                lo, hi = dequant_i4_byte_to_f16_pair(b, byte, scale=scale_f32)
                comps.append(lo)
                comps.append(hi)
            b_frag.append(b.vec_pack(comps, F16))

        wmma = getattr(b, wp.wmma_op)
        nacc = []
        for sm in range(n_sub_m):
            for sn in range(n_sub_n):
                idx = sm * n_sub_n + sn
                nacc.append(wmma(a_frag[sm], b_frag[sn], accs[idx]))
        b.scf_yield(*nacc)

    results = loop.results

    # Epilogue: scatter each <8 x float> sub-tile accumulator (row-guarded on M).
    for sm in range(n_sub_m):
        r0 = b.add(wm_base, b.const_i32(sm * wtm))
        for sn in range(n_sub_n):
            acc = results[sm * n_sub_n + sn]
            out_col = b.add(b.add(wn_base, b.const_i32(sn * wtn)), frag)
            for i in range(8):
                h = b.trunc_f32_to_f16(b.vec_extract(acc, i))
                if wp.split_k_by_half:
                    # gfx12 column-distributed: out_row = r0 + (lane//16)*8 + i.
                    out_row = b.add(r0, b.add(half_k_elem, b.const_i32(i)))
                else:
                    # gfx11 row-distributed: out_row = r0 + 2*i + (lane//16).
                    out_row = b.add(r0, b.add(b.const_i32(2 * i), half))
                with b.scf_if(b.cmp_lt(out_row, M)):
                    b.global_store(C, b.add(b.mul(out_row, cN), out_col), h)

    return b.kernel
