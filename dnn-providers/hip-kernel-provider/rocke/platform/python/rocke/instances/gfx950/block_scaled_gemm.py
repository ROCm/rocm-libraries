# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native gfx950 RCR GEMM with independent packed matrices and K32 E8M0 scales."""

from dataclasses import dataclass

from ...core.arch import ArchTarget
from ...core.arch.mfma_scale import MATRIX_FORMATS, matrix_layout
from ...core.dtypes import normalize_dtype
from ...core.ir import F32, I8, I32, IRBuilder, PtrType, dtype_to_ir_type
from ...core.ir_storage import storage_ir_type
from ...helpers.mma_operand import prepare_scaled_mma_operand
from ...helpers.tiling import (
    Fragment,
    load_fragment,
    make_matrix_fragment_desc,
    make_mma_accumulator_desc,
    make_tensor_desc,
    make_mma_scale_desc,
    make_window,
    store_fragment,
)
from ...helpers.spec import SignatureBuilder

LOWBIT_DTYPES = frozenset(MATRIX_FORMATS)


@dataclass(frozen=True)
class BlockScaledGemmSpec:
    name: str
    M: int
    N: int
    K: int
    dtype_a: str = "fp8e4m3"
    dtype_b: str = "fp8e4m3"
    dtype_c: str = "bf16"
    block_k: int = 32

    def __post_init__(self):
        for field in ("dtype_a", "dtype_b", "dtype_c"):
            object.__setattr__(self, field, normalize_dtype(getattr(self, field)))

    @property
    def block_size(self):
        return 64

    def kernel_name(self):
        return f"{self.name}_mfma_scaled_{self.dtype_a}_{self.dtype_b}_{self.dtype_c}_M{self.M}N{self.N}K{self.K}"

    def resolved_matrix_path(self):
        return "mfma_scale"


def is_valid_spec(spec, arch="gfx950"):
    if arch != "gfx950":
        return False, "scaled MFMA GEMM requires gfx950"
    if spec.dtype_a not in MATRIX_FORMATS or spec.dtype_b not in MATRIX_FORMATS:
        return False, "A/B must be FP8, BF8, FP6, BF6 or FP4"
    if spec.dtype_c not in ("bf16", "fp16"):
        return False, "output must be bf16 or fp16"
    if spec.block_k != 32:
        return False, "E8M0 scale groups must have block_k=32"
    if (
        spec.M <= 0
        or spec.N <= 0
        or spec.K <= 0
        or spec.M % 32
        or spec.N % 32
        or spec.K % 64
    ):
        return False, "positive M/N multiples of 32 and K multiples of 64 required"
    # All tensor offsets use signed i32, including packed row strides.
    if max(spec.M * spec.K, spec.N * spec.K, spec.M * spec.N) > 0x7FFFFFFF:
        return False, "tensor indexing exceeds i32 range"
    return True, "ok"


def block_scaled_gemm_grid(spec):
    return spec.N // 32, spec.M // 32, 1


def block_scaled_gemm_signature(spec):
    signature = SignatureBuilder()
    for name, dtype in (
        ("A", spec.dtype_a),
        ("B", spec.dtype_b),
        ("A_scale", "e8m0"),
        ("B_scale", "e8m0"),
        ("C", spec.dtype_c),
    ):
        signature.ptr(name, storage_ir_type(dtype).name)
    for name in ("M", "N", "K"):
        signature.scalar(name, "i32")
    return signature.build()


def build_block_scaled_gemm(spec, *, arch="gfx950"):
    """One wave computes a 32x32 tile. A[M,K], B[N,K] are densely packed.

    A_scale[M,K/32] and B_scale[K/32,N] contain raw E8M0 bytes. Each
    lane's selected scale is zero-extended into the low byte of an i32 VGPR.
    Matrix payloads use eight i32 ABI words, with zero padding for FP6/FP4.
    """
    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(reason)
    atom = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="mfma_scaled",
        a_dtype=spec.dtype_a,
        b_dtype=spec.dtype_b,
        c_dtype="fp32",
        scales=("e8m0", "e8m0", 32),
        m=32,
        n=32,
        k=64,
    )
    ir = IRBuilder(spec.kernel_name())
    ir.kernel.attrs["max_workgroup_size"] = spec.block_size
    A = ir.param(
        "A",
        PtrType(storage_ir_type(spec.dtype_a), "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    B = ir.param(
        "B",
        PtrType(storage_ir_type(spec.dtype_b), "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    AS = ir.param(
        "A_scale", PtrType(I8, "global"), noalias=True, readonly=True, align=1
    )
    BS = ir.param(
        "B_scale", PtrType(I8, "global"), noalias=True, readonly=True, align=1
    )
    C = ir.param(
        "C",
        PtrType(storage_ir_type(spec.dtype_c), "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    for name in ("M", "N", "K"):
        ir.param(name, I32)
    c32 = ir.const_i32(32)
    lane = ir.mod(ir.thread_id_x(), ir.const_i32(64))
    m0 = ir.mul(ir.block_id_y(), c32)
    n0 = ir.mul(ir.block_id_x(), c32)
    a_desc = make_matrix_fragment_desc(matrix_layout(spec.dtype_a))
    b_desc = make_matrix_fragment_desc(matrix_layout(spec.dtype_b))
    result_desc = make_mma_accumulator_desc(atom, target=arch, result=True)
    a_tensor = make_tensor_desc(
        (spec.M, spec.K), (spec.K, 1), dtype_to_ir_type(spec.dtype_a)
    )
    b_tensor = make_tensor_desc(
        (spec.N, spec.K), (spec.K, 1), dtype_to_ir_type(spec.dtype_b)
    )
    groups = spec.K // 32
    a_scales = make_tensor_desc((spec.M, groups), (groups, 1), dtype_to_ir_type("e8m0"))
    b_scales = make_tensor_desc((spec.N, groups), (1, spec.N), dtype_to_ir_type("e8m0"))
    scale_desc = make_mma_scale_desc(atom, target=arch)
    c_tensor = make_tensor_desc(
        (spec.M, spec.N), (spec.N, 1), dtype_to_ir_type(spec.dtype_c)
    )
    acc = ir.zero_vec_f32(16)
    for step in range(spec.K // 64):
        af = load_fragment(
            ir,
            A,
            make_window(a_tensor, (m0, step * 64)),
            a_desc,
            lane,
            alignment_bytes=16,
        )
        bf = load_fragment(
            ir,
            B,
            make_window(b_tensor, (n0, step * 64)),
            b_desc,
            lane,
            alignment_bytes=16,
        )
        a = prepare_scaled_mma_operand(ir, atom, "a", af)
        b = prepare_scaled_mma_operand(ir, atom, "b", bf)
        sa_fragment = load_fragment(
            ir,
            AS,
            make_window(a_scales, (m0, step * 2)),
            scale_desc,
            lane,
            alignment_bytes=1,
        )
        sb_fragment = load_fragment(
            ir,
            BS,
            make_window(b_scales, (n0, step * 2)),
            scale_desc,
            lane,
            alignment_bytes=1,
        )
        sa = ir.zext(ir.vec_extract(sa_fragment.value, 0), I32)
        sb = ir.zext(ir.vec_extract(sb_fragment.value, 0), I32)
        acc = ir.mma(atom, a, b, acc, sa, sb)
    store_fragment(
        ir,
        C,
        make_window(c_tensor, (m0, n0)),
        Fragment(result_desc, F32, acc),
        lane,
        alignment_bytes=16,
    )
    return ir.kernel
