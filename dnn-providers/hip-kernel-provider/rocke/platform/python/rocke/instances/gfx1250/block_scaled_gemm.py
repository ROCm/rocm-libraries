# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 block-scaled low-bit dense GEMM.

The production target is a WMMA-backed FP8/BF8 expert GEMM. ROCm 7.13 currently
does not select gfx1250 FP8 WMMA builtins, so this day-0 instance ships a real
scalar correctness kernel behind the same gfx1250-owned ABI. It validates
Qwen expert shapes, scale indexing, graph/dispatch plumbing, and GPU numerics
while the low-bit WMMA atom plumbing is brought up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ...core.ir import (
    BF16,
    F16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    VectorType,
)
from ...helpers.quant import quant_ir_type
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join

_LOWBIT_DTYPES = {"fp8", "fp8e4m3", "bf8", "bf8e5m2"}
_OUTPUT_DTYPES = {"fp16", "f16", "bf16"}
_SCALE_DTYPES = {"fp16", "f16", "fp32", "f32"}
_SUPPORTED_MATRIX_PATHS = {"auto", "wmma", "wmma_scaffold", "mfma"}
_BLOCK_M = 16
_BLOCK_N = 16
_WMMA_K = 64
_WAVE = 32
_HALF_K = 32  # K-elements per lane-half for the K=64 atom
_ACC = 8  # accumulator slots per lane (<8 x f32>)


def _wmma_op_id(dtype_a: str, dtype_b: str) -> str:
    return (
        f"wmma_gfx1250_f32_16x16x64_{_canon_lowbit(dtype_a)}_{_canon_lowbit(dtype_b)}"
    )


def _canon_lowbit(dtype: str) -> str:
    if dtype in ("fp8", "fp8e4m3"):
        return "fp8"
    if dtype in ("bf8", "bf8e5m2"):
        return "bf8"
    raise ValueError(f"expected fp8/bf8 low-bit dtype, got {dtype!r}")


def _wire_scale_dtype(dtype: str) -> str:
    if dtype in ("fp16", "f16"):
        return "f16"
    if dtype in ("fp32", "f32"):
        return "f32"
    raise ValueError(f"scale_dtype must be fp16/fp32, got {dtype!r}")


@dataclass(frozen=True)
class BlockScaledGemmSpec:
    """One RCR block-scaled low-bit dense GEMM contract for gfx1250."""

    name: str
    M: int
    N: int
    K: int
    dtype_a: str = "fp8"
    dtype_b: str = "fp8"
    dtype_c: str = "bf16"
    dtype_acc: str = "fp32"
    scale_dtype: str = "fp32"
    block_k: int = 128
    layout: str = "RCR"
    matrix_path: str = "auto"
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 128

    @property
    def block_size(self) -> int:
        return _WAVE

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            "block_scaled",
            f"{_canon_lowbit(self.dtype_a)}_{_canon_lowbit(self.dtype_b)}",
            f"M{self.M}N{self.N}K{self.K}",
            f"bk{self.block_k}",
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            flags={"wmma": self.resolved_matrix_path() == "wmma"},
        )

    def resolved_matrix_path(self) -> str:
        # "auto" and the legacy "wmma_scaffold" token both resolve to the real
        # gfx1250 K=64 WMMA path now that the kernel body is implemented.
        if self.matrix_path in ("auto", "wmma_scaffold"):
            return "wmma"
        return self.matrix_path


def is_valid_spec(spec: BlockScaledGemmSpec, arch: str = "gfx1250") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for the build-only gfx1250 scaffold."""
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if arch != "gfx1250":
        return False, f"block_scaled_gemm scaffold is gfx1250-only (got {arch!r})"
    if spec.matrix_path not in _SUPPORTED_MATRIX_PATHS:
        return False, (
            f"matrix_path must be one of {sorted(_SUPPORTED_MATRIX_PATHS)} "
            f"(got {spec.matrix_path!r})"
        )
    if spec.matrix_path == "mfma" or spec.resolved_matrix_path() == "mfma":
        return False, (
            "gfx1250 has no MFMA block_scale path; use matrix_path='wmma' "
            "(the K=64 FP8/BF8 WMMA atom)"
        )
    if not target.has_wmma:
        return False, f"{arch} does not expose WMMA for block-scaled GEMM"
    if target.has_mfma:
        return False, f"{arch} unexpectedly exposes MFMA; expected WMMA-only"

    if spec.M <= 0 or spec.N <= 0 or spec.K <= 0:
        return False, f"M/N/K must be positive (got M={spec.M}, N={spec.N}, K={spec.K})"
    if spec.dtype_a not in _LOWBIT_DTYPES or spec.dtype_b not in _LOWBIT_DTYPES:
        return False, (
            f"A/B must be fp8 or bf8 (got A={spec.dtype_a!r}, B={spec.dtype_b!r})"
        )
    if not target.mma.has_shape(
        family="wmma",
        a_dtype=_canon_lowbit(spec.dtype_a),
        b_dtype=_canon_lowbit(spec.dtype_b),
        c_dtype="fp32",
        m=_BLOCK_M,
        n=_BLOCK_N,
        k=_WMMA_K,
    ):
        return False, (
            f"no gfx1250 16x16x64 WMMA atom for "
            f"{_canon_lowbit(spec.dtype_a)}/{_canon_lowbit(spec.dtype_b)}"
        )
    if spec.dtype_c not in ("bf16", "fp16", "f16"):
        return False, f"day-0 gfx1250 output must be bf16/fp16 (got {spec.dtype_c!r})"
    if spec.dtype_acc not in ("fp32", "f32"):
        return False, f"accumulator dtype must be fp32 (got {spec.dtype_acc!r})"
    if spec.layout != "RCR":
        return False, f"block_scaled_gemm supports RCR only (got {spec.layout!r})"
    try:
        _wire_scale_dtype(spec.scale_dtype)
    except ValueError as e:
        return False, str(e)
    if spec.block_k <= 0 or spec.K % spec.block_k:
        return False, f"K ({spec.K}) must be divisible by block_k ({spec.block_k})"
    if spec.block_k % _WMMA_K:
        return False, f"block_k ({spec.block_k}) must be a multiple of {_WMMA_K}"
    if spec.K % _WMMA_K:
        return False, f"K ({spec.K}) must be a multiple of the WMMA K={_WMMA_K}"
    if (spec.tile_m, spec.tile_n) != (_BLOCK_M, _BLOCK_N):
        return False, "gfx1250 block_scaled_gemm uses fixed 16x16 output tiles"
    if spec.M % _BLOCK_M or spec.N % _BLOCK_N:
        return False, "M and N must be multiples of 16"

    return True, "ok: gfx1250 K=64 FP8/BF8 WMMA block-scaled GEMM"


def block_scaled_gemm_signature(spec: BlockScaledGemmSpec) -> List[dict]:
    """Manifest-style signature for the scaffold contract."""
    scale = _wire_scale_dtype(spec.scale_dtype)
    return (
        SignatureBuilder()
        .ptr("A", _canon_lowbit(spec.dtype_a))
        .ptr("B", _canon_lowbit(spec.dtype_b))
        .ptr("A_scale", scale)
        .ptr("B_scale", scale)
        .ptr("C", spec.dtype_c)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .build()
    )


def block_scaled_gemm_grid(spec: BlockScaledGemmSpec) -> Tuple[int, int, int]:
    return ceil_div_grid((spec.N, spec.tile_n), (spec.M, spec.tile_m))


def _storage_type(dtype: str) -> Type:
    if dtype in ("fp16", "f16"):
        return F16
    if dtype == "bf16":
        return BF16
    return quant_ir_type(dtype)


def _scale_type(dtype: str) -> Type:
    return F16 if _wire_scale_dtype(dtype) == "f16" else F32


def _as_f32(b: IRBuilder, v):
    return v if v.type.name == "f32" else b.cast_to_f32(v)


def build_block_scaled_gemm(
    spec: BlockScaledGemmSpec, arch: str = "gfx1250"
) -> KernelDef:
    """Build the gfx1250 K=64 FP8/BF8 block-scaled GEMM (RCR, ``C = A @ B^T``).

    One wave (32 lanes) computes one 16x16 output tile, no LDS. The K loop runs
    in ``block_k``-sized groups; each group accumulates ``block_k / 64`` WMMA
    K=64 steps into a fresh ``<8 x f32>`` accumulator, then applies the per-block
    scales ``a_scale[m, kg] * b_scale[kg, n]`` to each of the lane's 8 output
    slots (group scales are constant within a group, so post-scaling the group
    accumulator is exact). The fragment layout mirrors the verified gfx1250 f16
    K=32 atom, doubled to K=64: lane ``l`` carries 32 low-bit bytes
    (``<8 x i32>``) starting at K = ``k0 + (l // 16) * 32``; the accumulator is
    the column-distributed gfx12 ``<8 x f32>`` (slot ``i`` of lane ``l`` ->
    output ``(row = m0 + (l // 16) * 8 + i, col = n0 + l % 16)``).
    """
    ok, reason = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid block_scaled_gemm spec for {arch}: {reason}")

    a_ty = _storage_type(spec.dtype_a)
    b_ty = _storage_type(spec.dtype_b)
    c_ty = _storage_type(spec.dtype_c)
    scale_ty = _scale_type(spec.scale_dtype)
    op_id = _wmma_op_id(spec.dtype_a, spec.dtype_b)
    a_frag_ty = VectorType(I32, _ACC)

    groups = spec.K // spec.block_k
    steps_per_group = spec.block_k // _WMMA_K

    ir = IRBuilder(spec.kernel_name())
    ir.kernel.attrs["max_workgroup_size"] = spec.block_size

    A = ir.param("A", PtrType(a_ty, "global"), noalias=True, readonly=True, align=16)
    B = ir.param("B", PtrType(b_ty, "global"), noalias=True, readonly=True, align=16)
    AScale = ir.param(
        "A_scale", PtrType(scale_ty, "global"), noalias=True, readonly=True, align=4
    )
    BScale = ir.param(
        "B_scale", PtrType(scale_ty, "global"), noalias=True, readonly=True, align=4
    )
    C = ir.param("C", PtrType(c_ty, "global"), noalias=True, writeonly=True, align=16)
    M = ir.param("M", I32)  # noqa: F841 - ABI mirror; grid defines bounds
    N = ir.param("N", I32)  # noqa: F841
    K = ir.param("K", I32)  # noqa: F841

    cK = ir.const_i32(spec.K)
    cN = ir.const_i32(spec.N)
    c16 = ir.const_i32(_BLOCK_M)
    c32 = ir.const_i32(_WAVE)

    lane = ir.mod(ir.thread_id_x(), c32)
    frag = ir.mod(lane, c16)  # lane%16: A row / B col / output col
    half = ir.div(lane, c16)  # lane//16: K-half (operands) + row-block (acc)
    half_k = ir.mul(half, ir.const_i32(_HALF_K))  # (l//16)*32 K offset within step

    m0 = ir.mul(ir.block_id_y(), c16)
    n0 = ir.mul(ir.block_id_x(), c16)
    a_row = ir.add(m0, frag)  # this lane's A row
    b_row = ir.add(n0, frag)  # this lane's B row (= output col n)
    a_base = ir.mul(a_row, cK)
    b_base = ir.mul(b_row, cK)

    def _load_frag(ptr, base, storage_ty, k0):
        # 32 low-bit bytes per lane. fp8/bf8 vector loads cap at 16 elements,
        # so issue two 16-wide loads and concat into the <8 x i32> fragment.
        off0 = ir.add(ir.add(base, ir.const_i32(k0)), half_k)
        off1 = ir.add(off0, ir.const_i32(16))
        lo = ir.global_load_vN(ptr, off0, storage_ty, 16, align=16)
        hi = ir.global_load_vN(ptr, off1, storage_ty, 16, align=16)
        return ir.bitcast(ir.vec_concat(lo, hi), a_frag_ty)

    # Per-lane f32 output accumulators (8 column-distributed slots).
    outer = [ir.const_f32(0.0) for _ in range(_ACC)]

    for kg in range(groups):
        acc = ir.zero_vec_f32(_ACC)
        for step in range(steps_per_group):
            k0 = kg * spec.block_k + step * _WMMA_K
            a_frag = _load_frag(A, a_base, a_ty, k0)
            b_frag = _load_frag(B, b_base, b_ty, k0)
            acc = ir.mma(op_id, a_frag, b_frag, acc)

        # b_scale[kg, n] (col = n0 + frag), shared across this lane's 8 slots.
        b_scale_off = ir.add(ir.mul(ir.const_i32(kg), cN), b_row)
        b_scale = _as_f32(ir, ir.global_load(BScale, b_scale_off, scale_ty, align=4))

        for i in range(_ACC):
            out_row = ir.add(
                m0, ir.add(ir.mul(half, ir.const_i32(_ACC)), ir.const_i32(i))
            )
            a_scale_off = ir.add(
                ir.mul(out_row, ir.const_i32(groups)), ir.const_i32(kg)
            )
            a_scale = _as_f32(
                ir, ir.global_load(AScale, a_scale_off, scale_ty, align=4)
            )
            ab = ir.fmul(a_scale, b_scale)
            outer[i] = ir.fadd(outer[i], ir.fmul(ir.vec_extract(acc, i), ab))

    out_col = ir.add(n0, frag)
    row_base = ir.add(m0, ir.mul(half, ir.const_i32(_ACC)))
    for i in range(_ACC):
        out_row = ir.add(row_base, ir.const_i32(i))
        idx = ir.add(ir.mul(out_row, cN), out_col)
        ir.global_store(C, idx, ir.cast_f32_to(outer[i], c_ty), align=2)
    return ir.kernel
