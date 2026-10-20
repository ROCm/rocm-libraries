# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared building blocks for the gfx1151 (RDNA3.5) WMMA GEMM instances.

The standalone WMMA GEMM kernels in this package (f16, int8-storage, native iu8,
and iu8-dequant) all target the same fixed ``16x16x16`` WMMA tile on a single
wave32, so they share the tile geometry constants, the per-tile launch grid, and
the wave-size precondition. Centralizing them keeps the four kernels in lockstep
without changing any emitted value: the constants carry the same magnitudes, the
grid returns the same ceil-divided extents, and the guard returns the same
``(ok, reason)`` pair the hand-rolled checks produced.
"""

from __future__ import annotations

from typing import Any, Tuple

from ...core.ir import I32, IRBuilder, Value

# Fixed WMMA tile geometry: one wave (32 lanes) computes one 16x16 output tile.
_WMMA_M = 16
_WMMA_N = 16
_WMMA_K = 16
_WAVE = 32
_K_PER_I32 = 4  # int8 K-values packed per i32 fragment slot


def _iu8_gemm_lane_and_acc(
    b: IRBuilder, A: Value, Bp: Value, op: Any, K: Value
) -> tuple[Value, Value, Value, Value, Value]:
    """Shared iu8 lane/tile setup and K-loop MFMA accumulation."""
    c0 = b.const_i32(0)
    c4 = b.const_i32(_K_PER_I32)
    c16 = b.const_i32(_WMMA_K)
    c32 = b.const_i32(_WAVE)
    lane = b.mod(b.thread_id_x(), c32)
    frag = b.mod(lane, c16)
    half = b.div(lane, c16)
    m0 = b.mul(b.block_id_x(), c16)
    n0 = b.mul(b.block_id_y(), c16)
    k4 = b.div(K, c4)
    a_base = b.mul(b.add(m0, frag), k4)
    b_base = b.mul(b.add(n0, frag), k4)
    acc0 = b.zero_vec(I32, 8)
    loop = b.scf_for_iter(c0, k4, c4, [("acc", acc0)], iv_name="k0")
    with loop as (k0, (acc,)):
        a_frag = b.global_load_vN(A, b.add(a_base, k0), I32, _K_PER_I32)
        b_frag = b.global_load_vN(Bp, b.add(b_base, k0), I32, _K_PER_I32)
        nacc = b.mma(op, a_frag, b_frag, acc)
        b.scf_yield(nacc)
    acc = loop.results[0]
    return frag, half, m0, n0, acc


def wmma16_grid(M: int, N: int) -> Tuple[int, int, int]:
    """Launch grid ``(gx, gy, 1)`` for problem ``(M, N)``: one wave per 16x16 tile."""
    return ((M + _WMMA_M - 1) // _WMMA_M, (N + _WMMA_N - 1) // _WMMA_N, 1)


def wmma32_wave_guard(target: object, arch: str):
    """Return ``(ok, reason)`` enforcing the wave32 precondition for ``arch``.

    Returns ``(True, "")`` when ``target`` is wave32, otherwise the same
    rejection message the standalone kernels report. Callers fold the ``ok`` flag
    into their own validity result so the iu8 / f16 / int8 paths stay aligned.
    """
    if target.wave_size != _WAVE:
        return False, f"this WMMA kernel is wave32; {arch} is wave{target.wave_size}"
    return True, "ok"
