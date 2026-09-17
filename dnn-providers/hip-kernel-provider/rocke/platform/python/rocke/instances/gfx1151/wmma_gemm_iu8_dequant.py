# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""True-INT8 WMMA GEMM with f16 dequant output for gfx1151 (RDNA3.5).

The quantized-GEMM sibling of the native-int :mod:`rocke.instances.gfx1151.
wmma_gemm_iu8` (which outputs raw i32). This kernel runs the *same* hardware
``wmma_i32_16x16x16_iu8`` instruction (int8×int8 → int32 accumulate) but applies
a per-tensor symmetric dequant in the epilogue and stores **f16** — i.e. a usable
quantized GEMM result, matching the C++ ``14_gemm_quantization`` ``Mul_Clamp``
intent. It is the true-int8-compute counterpart to ``wmma_gemm_int8.py`` (which is
int8 *storage* but f16 *compute*), enabling an apples-to-apples (both f16-out) A/B
throughput comparison — see ``examples/gfx1151/gemm/scripts/05_int8_perf_a_vs_b.py``.

Reuses upstream's iu8 atom and operand ABI verbatim:

  * A/B are int8 logically but **passed packed as i32** (4 int8 per i32, slot ``j``
    holds K=[4j..4j+3]); A/B pointers are i32. C is **f16**.
  * Each lane's WMMA operand fragment is ``<4 x i32>`` (16 int8); the accumulator
    is a loop-carried ``<8 x i32>`` (slot ``i`` -> row ``m0 + 2*i + l/16``, col
    ``n0 + l%16``, same map as the f16 WMMA).
  * Epilogue: ``f16 = trunc(sitofp(acc_i32) * (scale_a*scale_b))``.

RCR layout (A row-major ``M×K`` int8, B row-major ``N×K`` int8, ``C = A @ B.T``),
one wave (32 lanes) per 16×16 tile, no LDS.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.arch import ArchTarget
from ...core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType
from ._wmma_common import _WAVE, _WMMA_K, wmma16_grid, wmma32_wave_guard

_K_PER_I32 = 4  # int8 K-values packed per i32 fragment slot
_OP_ID = "wmma_i32_16x16x16_iu8"


@dataclass(frozen=True)
class WmmaGemmIu8DequantSpec:
    """A gfx1151 true-int8 WMMA GEMM with f16 dequant output (int8 in / i32 acc /
    f16 out). Per-tensor symmetric scales are runtime args."""

    name: str = "rocke_wmma_gemm_iu8_dequant"

    @property
    def block_size(self) -> int:
        return _WAVE

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(self.name, "wmma16x16x16", "iu8_f16", "rcr")


def is_valid_spec(spec: WmmaGemmIu8DequantSpec, arch: str = "gfx1151"):
    """Return ``(ok, reason)``. The iu8 WMMA atom must exist on ``arch``."""
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if target.mma.by_op_id(_OP_ID) is None:
        return False, f"{_OP_ID} atom absent on {arch}"
    return wmma32_wave_guard(target, arch)


def build_wmma_gemm_iu8_dequant(
    spec: WmmaGemmIu8DequantSpec, arch: str = "gfx1151"
) -> KernelDef:
    """Build the gfx1151 true-int8 WMMA GEMM ``KernelDef`` (RCR, iu8 in / f16 out)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid iu8 dequant WMMA GEMM spec: {why}")
    target = ArchTarget.from_gfx(arch)
    op = target.mma.by_op_id(_OP_ID)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _WAVE

    # A/B are int8 logically but passed packed as i32 (4 int8/i32). C is f16.
    A = b.param("A", PtrType(I32, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(I32, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841 (implied by grid; kept for ABI parity)
    N = b.param("N", I32)
    K = b.param("K", I32)  # logical int8 K (multiple of 16)
    scale_a = b.param("scale_a", F32)
    scale_b = b.param("scale_b", F32)

    # Combined per-tensor dequant scale, applied once per output element.
    scale = b.fmul(scale_a, scale_b)

    c0 = b.const_i32(0)
    c4 = b.const_i32(_K_PER_I32)
    c16 = b.const_i32(_WMMA_K)
    c32 = b.const_i32(_WAVE)

    lane = b.mod(b.thread_id_x(), c32)
    frag = b.mod(lane, c16)  # lane%16: A-frag row, B-frag col, output col
    half = b.div(lane, c16)  # lane/16: even/odd output row selector

    m0 = b.mul(b.block_id_x(), c16)
    n0 = b.mul(b.block_id_y(), c16)

    # i32-element row bases: row r starts at int8 offset r*K = i32 offset r*(K/4).
    k4 = b.div(K, c4)  # i32 columns per row
    a_base = b.mul(b.add(m0, frag), k4)
    b_base = b.mul(b.add(n0, frag), k4)

    # K-loop: stride 16 int8 == 4 i32 columns per WMMA step.
    acc0 = b.zero_vec(I32, 8)
    loop = b.scf_for_iter(c0, k4, c4, [("acc", acc0)], iv_name="k0")
    with loop as (k0, (acc,)):
        a_frag = b.global_load_vN(A, b.add(a_base, k0), I32, _K_PER_I32)
        b_frag = b.global_load_vN(Bp, b.add(b_base, k0), I32, _K_PER_I32)
        nacc = b.mma(op, a_frag, b_frag, acc)
        b.scf_yield(nacc)
    acc = loop.results[0]

    # Epilogue: slot i of lane l -> (row = m0 + 2*i + l/16, col = n0 + l%16).
    # Dequantize the i32 accumulator (-> f32 -> * scale) before the f16 store.
    out_col = b.add(n0, frag)
    for i in range(8):
        deq = b.fmul(b.sitofp_f32(b.vec_extract(acc, i)), scale)
        h = b.trunc_f32_to_f16(deq)
        out_row = b.add(m0, b.add(b.const_i32(2 * i), half))
        idx = b.add(b.mul(out_row, N), out_col)
        b.global_store(C, idx, h)

    return b.kernel


def wmma_gemm_iu8_dequant_grid(M: int, N: int):
    """Launch grid (gx, gy, 1) for problem (M, N): one wave per 16x16 tile."""
    return wmma16_grid(M, N)
