# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Actual SSA operand checks shared by LLVM and HIP MFMA emitters."""

from __future__ import annotations

from .ir import BF8E5M2, BF16, F16, F32, FP8E4M3, I8, I32, U8, Op, Type, VectorType
from .tf32 import TF32_MMA, mma_instruction_id, tf32_mma_error

# These are the implemented instructions' fixed emission signatures, independent
# of the authoring catalog's logical src2/dst metadata. Packed instructions use
# a bitcast and therefore accept any supported numeric carrier of equal width.
_SOURCES = {
    "mfma_f32_16x16x4_f32": F32,
    "mfma_f32_32x32x2_f32": F32,
    "mfma_f32_16x16x16_f16": VectorType(F16, 4),
    "mfma_f32_16x16x32_f16": VectorType(F16, 8),
    "mfma_f32_32x32x8_f16": VectorType(F16, 4),
    "mfma_f32_32x32x16_f16": VectorType(F16, 8),
    "mfma_f32_4x4x4_f16": VectorType(F16, 4),
    "mfma_f32_16x16x16_bf16": VectorType(BF16, 4),
    "mfma_f32_16x16x32_bf16": VectorType(BF16, 8),
    "mfma_f32_32x32x8_bf16": VectorType(BF16, 4),
    "mfma_f32_32x32x16_bf16": VectorType(BF16, 8),
    "mfma_f32_16x16x32_fp8": VectorType(I8, 8),
    "mfma_f32_16x16x32_bf8": VectorType(I8, 8),
    "mfma_f32_32x32x16_fp8": VectorType(I8, 8),
    "mfma_f32_32x32x16_bf8": VectorType(I8, 8),
}
_PACKED_BITS = {
    "mfma_f32_16x16x128_fp4": 64,
    "mfma_f32_16x16x96_fp6": 96,
    "mfma_f32_16x16x128_fp8": 256,
    "mfma_scale_f32_16x16x128_f8f6f4": 256,
}
# Only scalar types already mapped by the emitters qualify as packed carriers.
# Logical fp4/fp6 formats do not have a standalone LLVM scalar representation.
_CARRIER_BITS = {
    "i1": 1,
    "i8": 8,
    "u8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "f64": 64,
    "fp8e4m3": 8,
    "bf8e5m2": 8,
}


def _carrier_bits(ty: Type) -> int:
    if isinstance(ty, VectorType):
        if type(ty.elem) is not Type or type(ty.count) is not int or ty.count <= 0:
            return 0
        return _CARRIER_BITS.get(ty.elem.name, 0) * ty.count
    return _CARRIER_BITS.get(ty.name, 0) if type(ty) is Type else 0


def validate_mfma(op: Op) -> None:
    """Reject actual values that cannot use the selected fixed instruction ABI.

    Keep src2 and dst independent in generic MMA authoring. Each implemented
    MFMA emitter imposes its own physical signature before emitting any text.
    XF32 retains its existing logical TF32 diagnostics.
    """
    op_id = mma_instruction_id(op)
    if op_id in TF32_MMA:
        error = tf32_mma_error(op_id, op.operands, op.results)
        if error:
            raise ValueError(error)
        return
    scaled = op_id == "mfma_scale_f32_16x16x128_f8f6f4"
    arity = 5 if scaled else 3
    if len(op.operands) != arity or len(op.results) != 1:
        raise ValueError(f"MFMA expects {arity} operands and 1 result")
    bits = _PACKED_BITS.get(op_id)
    if bits is not None:
        if any(_carrier_bits(v.type) != bits for v in op.operands[:2]):
            raise ValueError(f"MFMA requires src0 and src1 to have {bits} bits")
    else:
        source = _SOURCES[op_id]
        if source == VectorType(I8, 8):
            valid = all(
                isinstance(v.type, VectorType)
                and type(v.type.count) is int
                and v.type.count == 8
                and v.type.elem in (I8, U8, FP8E4M3, BF8E5M2)
                for v in op.operands[:2]
            )
        else:
            valid = all(
                v.type == source or (source == F32 and v.type == VectorType(F32, 1))
                for v in op.operands[:2]
            )
        if not valid:
            raise ValueError(f"MFMA requires src0 and src1 to be {source.name}")
    count = 16 if op_id.startswith("mfma_f32_32x32") else 4
    expected = VectorType(F32, count)
    if op.operands[2].type != expected or op.result.type != expected:
        raise ValueError(f"MFMA requires src2 and dst to be {expected.name}")
    if scaled and any(v.type != I32 for v in op.operands[3:]):
        raise ValueError("MFMA requires scale operands to be i32")
