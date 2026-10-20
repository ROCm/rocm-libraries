# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native integer WMMA GEMM for gfx1151 (RDNA3.5) — iu8 lane-map probe.

Standalone, no-LDS proof of the ``wmma_i32_16x16x16_iu8`` fragment ABI, the
integer twin of :mod:`rocke.instances.gfx1151.wmma_gemm`. It pins the iu8 A/B/
accumulator lane maps with a *bit-exact* integer reference before they are used
in the deep-fusion kernel.

Layout (RCR, int8 in / i32 acc / i32 out; matches ``C = A @ B.T`` with A
row-major ``M×K`` int8 and B row-major ``N×K`` int8):

  * One wave (32 lanes) computes one ``16×16`` output tile; grid is
    ``(ceil(M/16), ceil(N/16))``, ``threads_per_block=32``.
  * Each lane's WMMA operand fragment is ``<4 x i32>`` = 16 int8 K-values packed
    4-per-i32 (slot ``j`` holds K=[4j..4j+3]). The host packs int8 rows into i32
    so a plain ``<4 x i32>`` load reproduces the fragment; A/B pointers are i32.
  * Accumulator is a loop-carried ``<8 x i32>``; the per-lane slot→output map is
    identical to the f16 WMMA (``row = m0 + 2*i + l/16, col = n0 + l%16``), only
    the element type is i32. Output is stored as i32.

The integer WMMA does no rounding — verify expects ``max_abs_diff == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.arch import ArchTarget
from ...core.ir import I32, IRBuilder, KernelDef, PtrType
from ._wmma_common import (
    _K_PER_I32,
    _WAVE,
    _WMMA_K,
    wmma16_grid,
    wmma32_wave_guard,
    _iu8_gemm_lane_and_acc,
)

_K_PER_I32 = 4  # int8 K-values packed per i32 fragment slot
_OP_ID = "wmma_i32_16x16x16_iu8"


@dataclass(frozen=True)
class WmmaGemmIu8Spec:
    """A gfx1151 native-int WMMA GEMM instance (int8 in / i32 out)."""

    name: str = "rocke_wmma_gemm_iu8"

    @property
    def block_size(self) -> int:
        return _WAVE

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(self.name, "wmma16x16x16", "iu8", "rcr")


def is_valid_spec(spec: WmmaGemmIu8Spec, arch: str = "gfx1151"):
    """Return ``(ok, reason)``. The iu8 WMMA atom must exist on ``arch``."""
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if target.mma.by_op_id(_OP_ID) is None:
        return False, f"{_OP_ID} atom absent on {arch}"
    return wmma32_wave_guard(target, arch)


def build_wmma_gemm_iu8(spec: WmmaGemmIu8Spec, arch: str = "gfx1151") -> KernelDef:
    """Build the gfx1151 native-int WMMA GEMM ``KernelDef`` (RCR, iu8)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid iu8 WMMA GEMM spec: {why}")
    target = ArchTarget.from_gfx(arch)
    op = target.mma.by_op_id(_OP_ID)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _WAVE

    # A/B are int8 logically but passed packed as i32 (4 int8/i32). C is i32.
    A = b.param("A", PtrType(I32, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(I32, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(I32, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841 (implied by grid; kept for ABI parity)
    N = b.param("N", I32)
    K = b.param("K", I32)  # logical int8 K (multiple of 16)

    frag, half, m0, n0, acc = _iu8_gemm_lane_and_acc(b, A, Bp, op, K)

    # Epilogue: slot i of lane l -> (row = m0 + 2*i + l/16, col = n0 + l%16).
    out_col = b.add(n0, frag)
    for i in range(8):
        elem = b.vec_extract(acc, i)
        out_row = b.add(m0, b.add(b.const_i32(2 * i), half))
        idx = b.add(b.mul(out_row, N), out_col)
        b.global_store(C, idx, elem)

    return b.kernel


def wmma_gemm_iu8_grid(M: int, N: int):
    """Launch grid (gx, gy, 1) for problem (M, N): one wave per 16x16 tile."""
    return wmma16_grid(M, N)
