# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""INT8-storage / f16-compute WMMA GEMM for gfx1151 (RDNA3.5 / Strix Halo).

A quantized-GEMM sibling of ``instances/gfx1151/wmma_gemm.py``. The operands are
stored as symmetric per-tensor int8 (A and B); each fragment is converted to f16
on-chip and fed to the *existing*, hardware-verified ``wmma_f32_16x16x16_f16``
path (f32 accumulate). The combined dequant scale ``scale_a * scale_b`` is folded
into the epilogue rather than applied per element:

  * int8 values (|x| <= 127) are exact in f16, so the in-loop conversion is
    lossless and the f32 accumulator holds an exact integer dot product;
  * one ``acc * (scale_a*scale_b)`` multiply per output element reconstructs the
    dequantized result before the f16 store.

This is the "Path B" example: it demonstrates quantized GEMM in rocke with **no
DSL core changes** (no int8 WMMA atom). The compute throughput equals the f16
kernel; the win is int8 storage / memory bandwidth. A faithful ``i8*i8 -> i32``
WMMA port (true INT8 tensor-core throughput) needs a new ``wmma_i32_16x16x16_iu8``
atom and is a separate follow-on.

Layout matches ``wmma_gemm.py`` exactly (RCR: A row-major ``M×K``, B row-major
``N×K``, ``C = A @ B.T``), grid ``(ceil(M/16), ceil(N/16))`` with
``block_id.x -> M-tile, block_id.y -> N-tile``, one wave (32 lanes) per 16×16 tile.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ir import F16, F32, I8, I32, IRBuilder, KernelDef, PtrType
from ._wmma_common import (
    _WAVE,
    _WMMA_K,
    _WMMA_M,
    _WMMA_N,
    wmma16_grid,
    wmma32_wave_guard,
)


@dataclass(frozen=True)
class WmmaGemmInt8Spec:
    """A gfx1151 int8-storage / f16-compute WMMA GEMM instance.

    M/N/K and the per-tensor dequant scales are runtime kernel args; the spec
    only carries the storage dtype and a name (the tile is the fixed 16x16x16
    WMMA, compute is f16)."""

    name: str = "rocke_wmma_gemm_int8"
    dtype: str = "i8"

    def __post_init__(self) -> None:
        if self.dtype != "i8":
            raise ValueError(
                f"WmmaGemmInt8Spec stores operands as int8 only, got {self.dtype!r}"
            )

    @property
    def block_size(self) -> int:
        return _WAVE

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(self.name, "wmma16x16x16", "i8_f16", "rcr")


def is_valid_spec(spec: WmmaGemmInt8Spec, arch: str = "gfx1151"):
    """Return ``(ok, reason)``. The f16 WMMA 16x16x16 *compute* atom must exist on
    ``arch`` (operands are int8 in memory but dequantized to f16 before the MMA)."""
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if not target.mma.has_shape(
        family="wmma",
        a_dtype="fp16",
        b_dtype="fp16",
        c_dtype="fp32",
        m=_WMMA_M,
        n=_WMMA_N,
        k=_WMMA_K,
    ):
        return False, (
            f"WMMA {_WMMA_M}x{_WMMA_N}x{_WMMA_K} f16 compute atom absent on {arch} "
            f"(WMMA is an RDNA/gfx11 instruction)"
        )
    return wmma32_wave_guard(target, arch)


def build_wmma_gemm_int8(spec: WmmaGemmInt8Spec, arch: str = "gfx1151") -> KernelDef:
    """Build the gfx1151 int8-storage WMMA GEMM ``KernelDef`` (RCR, i8 in / f16 out)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid WMMA int8 GEMM spec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _WAVE

    A = b.param("A", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841 (M is implied by the grid; kept for ABI parity)
    N = b.param("N", I32)
    K = b.param("K", I32)
    scale_a = b.param("scale_a", F32)
    scale_b = b.param("scale_b", F32)

    # Combined per-tensor dequant scale, applied once per output element.
    scale = b.fmul(scale_a, scale_b)

    c0 = b.const_i32(0)
    c16 = b.const_i32(_WMMA_K)
    c32 = b.const_i32(_WAVE)

    # Wave-relative lane and its fragment coordinates (identical to wmma_gemm).
    lane = b.mod(b.thread_id_x(), c32)
    frag = b.mod(lane, c16)  # lane%16: A-frag row, B-frag row, output col
    half = b.div(lane, c16)  # lane/16: 0 or 1, selects even/odd output rows

    m0 = b.mul(b.block_id_x(), c16)  # output-tile row base
    n0 = b.mul(b.block_id_y(), c16)  # output-tile col base

    a_base = b.mul(b.add(m0, frag), K)
    b_base = b.mul(b.add(n0, frag), K)

    # K-loop accumulating the <8 x float> WMMA fragment. Each int8 operand
    # fragment is sign-extended -> f32 -> f16 (lossless for |x|<=127) and packed
    # into the <16 x half> the WMMA atom expects.
    acc0 = b.zero_vec_f32(8)
    loop = b.scf_for_iter(c0, K, c16, [("acc", acc0)], iv_name="k0")
    with loop as (k0, (acc,)):
        a_i8 = b.global_load_vN(A, b.add(a_base, k0), I8, 16, align=16)
        b_i8 = b.global_load_vN(Bp, b.add(b_base, k0), I8, 16, align=16)
        a_frag = b.vec_pack([_i8_to_f16(b, a_i8, i) for i in range(16)], F16)
        b_frag = b.vec_pack([_i8_to_f16(b, b_i8, i) for i in range(16)], F16)
        nacc = b.wmma_f32_16x16x16_f16(a_frag, b_frag, acc)
        b.scf_yield(nacc)
    acc = loop.results[0]

    # Epilogue: slot i of lane l -> (row = m0 + 2*i + l/16, col = n0 + l%16).
    # Fold the dequant scale into the f32 accumulator before truncating to f16.
    out_col = b.add(n0, frag)
    for i in range(8):
        elem = b.fmul(b.vec_extract(acc, i), scale)
        h = b.trunc_f32_to_f16(elem)
        out_row = b.add(m0, b.add(b.const_i32(2 * i), half))
        idx = b.add(b.mul(out_row, N), out_col)
        b.global_store(C, idx, h)

    return b.kernel


def _i8_to_f16(b: IRBuilder, vec, i: int):
    """Extract lane-fragment slot ``i`` (int8), sign-extend and convert to f16."""
    return b.cast_f32_to(b.sitofp_f32(b.sext(b.vec_extract(vec, i), I32)), F16)


def wmma_gemm_int8_grid(M: int, N: int):
    """Launch grid (gx, gy, 1) for problem (M, N): one wave per 16x16 tile."""
    return wmma16_grid(M, N)
