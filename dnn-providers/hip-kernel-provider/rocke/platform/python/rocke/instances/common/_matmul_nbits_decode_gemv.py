# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Decode-GEMV body for the ``MatMulNBits`` instance (Milestone 6).

Specialization for the decode-phase ``lm_head`` projection: very large ``N``
(e.g. Qwen3.5-9B vocab N=248320) with a tiny ``M`` (the single decode token,
M=1). The large-N WMMA body wastes a 16-wide WMMA M tile on one row and would
read tile_m rows of ``A`` out of bounds, so decode-GEMV is a dedicated scalar
body: **one thread per output column**, no WMMA, arch-agnostic.

  * grid ``(ceil(N / block_size), ceil(M / block_m=1), 1)`` — ``block_id.x``
    selects a contiguous block of ``block_size`` output columns; thread ``t``
    owns column ``block_id.x * block_size + t`` (guarded ``n < N`` for the
    ragged final block);
  * each thread walks the full ``K`` as ``K/2`` packed bytes, dequantising two
    signed int4 per byte to f32 and accumulating ``A[m, k] * w[n, k]`` in f32;
  * an outer loop over ``m < M`` keeps the body correct for ``M > 1`` even
    though the decode contract is ``M = 1``.

Scale lookup mirrors the large-N body: the two ``k`` values in packed byte ``j``
are ``2j`` and ``2j+1``; with ``group_size`` even they share the group index
``j // (group_size / 2)``.
"""

from __future__ import annotations

from ...core.ir import F16, F32, I8, I32, IRBuilder, PtrType
from ...helpers.i4_dequant import unpack_i4_byte_to_pair_f32
from ._matmul_nbits_common import MatMulNBitsSpec, V1_ARCH, _scale_wire_dtype

__all__ = ["build_decode_gemv_matmul_nbits"]


def build_decode_gemv_matmul_nbits(
    spec: MatMulNBitsSpec, arch: str = V1_ARCH
) -> "object":
    """Build the decode-GEMV ``KernelDef`` for ``spec`` (validated by caller)."""
    N, K, group = spec.N, spec.K, spec.group_size
    bs = spec.block_size
    k_packed_stride = K // 2  # packed-byte row stride for B (two int4 per byte)
    k_group_stride = K // group  # scale row stride

    scale_t = F16 if _scale_wire_dtype(spec.scale_dtype) == "f16" else F32

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = bs

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    Sp = b.param(
        "Scales", PtrType(scale_t, "global"), noalias=True, readonly=True, align=8
    )
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    cN = b.const_i32(N)
    cK = b.const_i32(K)
    c_half_k = b.const_i32(k_packed_stride)
    c_half_group = b.const_i32(group // 2)

    tid = b.thread_id_x()
    n = b.add(b.mul(b.block_id_x(), b.const_i32(bs)), tid)

    with b.scf_if(b.cmp_lt(n, cN)):
        b_byte_base = b.mul(n, c_half_k)
        b_scale_base = b.mul(n, b.const_i32(k_group_stride))

        mloop = b.scf_for(c0, M, c1, iv_name="m")
        with mloop as m:
            a_row_base = b.mul(m, cK)
            kloop = b.scf_for_iter(
                c0, c_half_k, c1, [("acc", b.const_f32(0.0))], iv_name="j"
            )
            with kloop as (j, accs):
                acc = accs[0]
                byte = b.global_load(Bp, b.add(b_byte_base, j), I8)
                lo, hi = unpack_i4_byte_to_pair_f32(b, byte)
                kgrp = b.div(j, c_half_group)
                scale = b.global_load(Sp, b.add(b_scale_base, kgrp), scale_t)
                scale_f32 = b.cast_to_f32(scale)
                k_even = b.mul(j, c2)
                a_lo = b.cast_to_f32(b.global_load(A, b.add(a_row_base, k_even), F16))
                a_hi = b.cast_to_f32(
                    b.global_load(A, b.add(a_row_base, b.add(k_even, c1)), F16)
                )
                prod = b.fadd(
                    b.fmul(a_lo, b.fmul(lo, scale_f32)),
                    b.fmul(a_hi, b.fmul(hi, scale_f32)),
                )
                b.scf_yield(b.fadd(acc, prod))
            out_h = b.trunc_f32_to_f16(kloop.results[0])
            b.global_store(C, b.add(b.mul(m, cN), n), out_h)

    return b.kernel
