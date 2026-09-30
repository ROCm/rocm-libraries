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

from typing import Tuple

# Fixed WMMA tile geometry: one wave (32 lanes) computes one 16x16 output tile.
_WMMA_M = 16
_WMMA_N = 16
_WMMA_K = 16
_WAVE = 32


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
