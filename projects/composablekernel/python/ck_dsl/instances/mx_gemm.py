# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MX (microscaling) GEMM kernel (CK Tile ``42_mx_gemm`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/42_mx_gemm``. The MX
GEMM uses the OCP MX-spec shared-exponent format: each 32-element
mantissa block carries one 8-bit unbiased E8M0 scale, so the per-
element math is::

 A_mantissa[i] = fp8_or_fp6_or_fp4 value
 A_scale[i // 32] = 8-bit unbiased exponent (E8M0)
 A_real[i] = A_mantissa[i] * 2^(A_scale[i // 32] - 127)

Same for B. The kernel does an A * B^T contraction in f32 (the MX
intrinsic family ``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4``
applies the scale in-instruction; v1 uses the explicit
:func:`ck_dsl.helpers.mx_scale.decode_mx_scale_e8m0` +
:func:`ck_dsl.helpers.mx_scale.apply_mx_scale` chain instead so the
scalar inner GEMM stays interpretable).

What v1 ships:

* **fp8 mantissa** for both A and B (``mantissa_dtype="fp8e4m3"``);
 the fp6 and fp4 mantissa families are reachable through the same
 spec surface once the matching unpack helpers land in
 :mod:`ck_dsl.helpers.i4_dequant` (the fp4 / fp6 unpackers are a
 natural extension of the i4 nibble-extract pattern).
* **Block size 32 for the shared exponent** (the MX-spec default);
 ``group_k`` is the only knob (default 32).
* **f32 output**; the caller can post-convert to bf16/f16 via
 :func:`ck_dsl.helpers.io.store_scalar_from_f32` or a follow-on
 finalize kernel.

Scalar inner v1 (per output ``(m, n)``):

 acc = 0
 for k in 0..K:
 a_mantissa = load A[m, k]; b_mantissa = load B[k, n]
 a_scale = decode(A_scale[m, k // 32])
 b_scale = decode(B_scale[k // 32, n]) # transposed for column-major B
 a = cvt_<mantissa_dtype>_to_f32(a_mantissa) * a_scale
 b = cvt_<mantissa_dtype>_to_f32(b_mantissa) * b_scale
 acc += a * b
 C[m, n] = acc

The MFMA-based body ( follow-on) replaces the per-element loop
with one ``mfma_f32_16x16x32_fp8(...)`` call per MX block plus the
block-scale multiply post-MFMA, against the same spec + helper
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F32, I8, I32, IRBuilder, KernelDef, PtrType
from ..helpers.atoms import MfmaAtom
from ..helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    mfma_k_loop,
)
from ..helpers.mx_scale import decode_mx_scale_e8m0
from ..helpers.quant import quant_ir_type
from ..helpers.spec import SignatureBuilder, kernel_name_join


MxMantissaDType = Literal["fp8e4m3", "bf8e5m2"]


@dataclass(frozen=True)
class MxGemmSpec:
    """One concrete MX GEMM kernel configuration."""

    M: int
    N: int
    K: int
    mantissa_dtype: MxMantissaDType = "fp8e4m3"
    group_k: int = 32  # MX spec: shared-exponent block size
    block_tile_m: int = 16
    block_tile_n: int = 16
    name: str = "ck_dsl_mx_gemm"

    @property
    def atom(self) -> MfmaAtom:
        """fp8 / bf8 MFMA atom selected from ``mantissa_dtype``."""
        if (self.block_tile_m, self.block_tile_n) != (16, 16):
            raise ValueError(
                f"mx_gemm MFMA path supports 16x16 tiles only "
                f"(got {self.block_tile_m}x{self.block_tile_n})"
            )
        return (
            MfmaAtom.fp8_16x16x32()
            if self.mantissa_dtype == "fp8e4m3"
            else MfmaAtom.bf8_16x16x32()
        )

    @property
    def block_size(self) -> int:
        # MFMA path: one wave64 warp per CTA.
        return 64

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            self.mantissa_dtype,
            f"gk{self.group_k}",
            f"t{self.block_tile_m}x{self.block_tile_n}",
        )


def is_valid_spec(spec: MxGemmSpec) -> Tuple[bool, str]:
    if spec.mantissa_dtype not in ("fp8e4m3", "bf8e5m2"):
        return False, (
            f"unsupported mantissa_dtype {spec.mantissa_dtype!r}; "
            "v1 ships fp8e4m3 / bf8e5m2 only (fp4 / fp6 are v2)"
        )
    if spec.group_k != 32:
        return False, (
            f"MX spec requires group_k = 32 (shared exponent per 32 "
            f"mantissa elements); got group_k={spec.group_k}"
        )
    if spec.K % spec.group_k:
        return False, f"K ({spec.K}) must be divisible by group_k ({spec.group_k})"
    if spec.M % spec.block_tile_m or spec.N % spec.block_tile_n:
        return False, "M / N must divide their tile sizes (v1 no partial tiles)"
    if spec.block_size > 1024:
        return False, f"block_size {spec.block_size} > 1024"
    return True, "ok"


def build_mx_gemm(spec: MxGemmSpec) -> KernelDef:
    """Build the IR for one MX GEMM instance (v1 scalar inner).

    Kernel signature::

    (A: ptr<fp8 | bf8>, # (M, K) mantissa
    AScale: ptr<i8>, # (M, K / 32) E8M0 shared exponent
    B: ptr<fp8 | bf8>, # (K, N) mantissa
    BScale: ptr<i8>, # (K / 32, N) E8M0 shared exponent
    C: ptr<f32>, # (M, N) f32 output
    M: i32, N: i32, K: i32)

    Grid: ``(ceil(N / block_tile_n), ceil(M / block_tile_m), 1)``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid mx_gemm spec: {why}")

    mantissa_ty = quant_ir_type(spec.mantissa_dtype)
    BS = spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(mantissa_ty, "global"), readonly=True, align=16)
    AScale = b.param("AScale", PtrType(I8, "global"), readonly=True, align=1)
    Bp = b.param("B", PtrType(mantissa_ty, "global"), readonly=True, align=16)
    BScale = b.param("BScale", PtrType(I8, "global"), readonly=True, align=1)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=4)
    _M = b.param("M", I32)  # noqa: F841 -- ABI
    _N = b.param("N", I32)  # noqa: F841 -- ABI
    _K = b.param("K", I32)  # noqa: F841 -- ABI

    lane = b.thread_id_x()
    bid_n = b.block_id_x()
    bid_m = b.block_id_y()
    atom = spec.atom
    m_tile_base = b.mul(bid_m, b.const_i32(spec.block_tile_m))
    n_tile_base = b.mul(bid_n, b.const_i32(spec.block_tile_n))

    lane_decode = decode_mfma_lanes(b, atom, lane)
    if spec.group_k % atom.k != 0:
        raise ValueError(
            f"MX group_k ({spec.group_k}) must be a multiple of atom.k "
            f"({atom.k}) so the per-group scale apply aligns with whole "
            f"MFMA invocations"
        )
    k_scale_count = spec.K // spec.group_k

    # Two-level K iteration: outer over MX shared-exponent groups (one
    # E8M0 byte per group, decoded once per outer iter), inner emits
    # ``atoms_per_group`` MFMA invocations on fresh per-group acc, then
    # scales by ``a_scale * b_scale`` and folds into the outer
    # accumulator. The MX ``mfma_f32_16x16x128_f8f6f4`` intrinsic does
    # this in-instruction; the v1 chain replicates it explicitly so
    # parity isn't blocked on the in-instruction intrinsic.
    outer = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(k_scale_count),
        b.const_i32(1),
        [("oacc", atom.zero_acc(b))],
        iv_name="kg",
    )
    with outer as (kg, (outer_acc,)):
        # Per-group E8M0 scale loads + decode.
        m_global_row = b.add(m_tile_base, lane_decode.m_in_atom)
        n_global_col = b.add(n_tile_base, lane_decode.n_in_atom)
        a_scale_off = b.add(
            b.mul(m_global_row, b.const_i32(k_scale_count)),
            kg,
        )
        b_scale_off = b.add(
            b.mul(kg, b.const_i32(spec.N)),
            n_global_col,
        )
        a_scale = decode_mx_scale_e8m0(
            b,
            b.global_load(AScale, a_scale_off, I8, align=1),
        )
        b_scale = decode_mx_scale_e8m0(
            b,
            b.global_load(BScale, b_scale_off, I8, align=1),
        )
        # The MX-spec apply is ``elem * 2^(scale_unbiased - 127)``;
        # mathematically equivalent to ``elem * 2^scale_unbiased / 2^127``
        # and since a_scale / b_scale are already the post-decode f32
        # multipliers, the combined scale is just their product.
        ab_scale = b.fmul(a_scale, b_scale)

        k_group_base = b.mul(kg, b.const_i32(spec.group_k))

        def _load_a_in_group(b, kt_local, base=k_group_base):
            return load_a_row_major_contiguous(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=b.add(
                    base,
                    b.mul(kt_local, b.const_i32(atom.k)),
                ),
                K=spec.K,
            )

        def _load_b_in_group(b, kt_local, base=k_group_base):
            return load_b_col_strided_scalars(
                b,
                B=Bp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=b.add(
                    base,
                    b.mul(kt_local, b.const_i32(atom.k)),
                ),
                N=spec.N,
            )

        group_acc = mfma_k_loop(
            b,
            K=spec.group_k,
            atom=atom,
            load_a=_load_a_in_group,
            load_b=_load_b_in_group,
            iv_name="kk",
            acc_name="gacc",
        )

        # Scale-and-accumulate the group's MFMA result.
        scaled = b.zero_vec_f32(atom.c_per_lane)
        for i in range(atom.c_per_lane):
            v = b.vec_extract(group_acc, i)
            old = b.vec_extract(outer_acc, i)
            new = b.fadd(old, b.fmul(v, ab_scale))
            scaled = b.vec_insert(scaled, new, i)
        b.scf_yield(scaled)

    acc_final = outer.results[0]

    # Output store: each lane writes c_per_lane cells via the atom's
    # lane→output mapping.
    for i in range(atom.c_per_lane):
        row_in, col_in = atom.lane_to_output(b, lane, i)
        row = b.add(m_tile_base, row_in)
        col = b.add(n_tile_base, col_in)
        addr = b.add(b.mul(row, b.const_i32(spec.N)), col)
        b.global_store(C, addr, b.vec_extract(acc_final, i), align=4)
    b.ret()
    return b.kernel


def mx_gemm_grid(spec: MxGemmSpec) -> Tuple[int, int, int]:
    n_tiles = (spec.N + spec.block_tile_n - 1) // spec.block_tile_n
    m_tiles = (spec.M + spec.block_tile_m - 1) // spec.block_tile_m
    return (n_tiles, m_tiles, 1)


def mx_gemm_signature(spec: MxGemmSpec):
    return (
        SignatureBuilder()
        .ptr("A", spec.mantissa_dtype)
        .ptr("AScale", "i8")
        .ptr("B", spec.mantissa_dtype)
        .ptr("BScale", "i8")
        .ptr("C", "f32")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .build()
    )


__all__ = [
    "MxGemmSpec",
    "MxMantissaDType",
    "build_mx_gemm",
    "is_valid_spec",
    "mx_gemm_grid",
    "mx_gemm_signature",
]
