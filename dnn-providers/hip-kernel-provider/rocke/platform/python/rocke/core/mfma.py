# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Actual SSA operand checks shared by LLVM and HIP MFMA emitters."""

from __future__ import annotations

from .ir import BF8E5M2, BF16, F16, F32, F64, FP8E4M3, I8, I32, U8, Op, Type, VectorType
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
    "mfma_f32_4x4x4_bf16": VectorType(BF16, 4),
    "mfma_f64_16x16x4_f64": F64,
    "mfma_f64_4x4x4_f64": F64,
    "mfma_f32_16x16x16_bf16": VectorType(BF16, 4),
    "mfma_f32_16x16x32_bf16": VectorType(BF16, 8),
    "mfma_f32_32x32x8_bf16": VectorType(BF16, 4),
    "mfma_f32_32x32x16_bf16": VectorType(BF16, 8),
    "mfma_f32_16x16x32_fp8": VectorType(FP8E4M3, 8),
    "mfma_f32_16x16x32_bf8": VectorType(BF8E5M2, 8),
    "mfma_f32_32x32x16_fp8": VectorType(FP8E4M3, 8),
    "mfma_f32_32x32x16_bf8": VectorType(BF8E5M2, 8),
}
_PACKED_BITS = {
    "mfma_f32_16x16x128_fp4fp6fp8": 256,
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


def _carrier_element(ty: Type) -> tuple[Type | None, int]:
    """The scalar element/count of a well-formed register carrier, or no shape."""
    if type(ty) is Type:
        return ty, 1
    if (
        isinstance(ty, VectorType)
        and type(ty.elem) is Type
        and type(ty.count) is int
        and ty.count > 0
    ):
        return ty.elem, ty.count
    return None, 0


def _carrier_bits(ty: Type) -> int:
    element, count = _carrier_element(ty)
    return _CARRIER_BITS.get(element.name, 0) * count if element is not None else 0


def _fixed_type_matches(actual: Type, expected: Type) -> bool:
    if isinstance(expected, VectorType):
        return (
            isinstance(actual, VectorType)
            and type(actual.count) is int
            and type(actual.elem) is Type
            and actual == expected
        )
    return type(actual) is Type and actual == expected


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
        for role, value in enumerate(op.operands[:2]):
            actual = value.type
            element, _ = _carrier_element(actual)
            if element is not None and element.name not in _CARRIER_BITS:
                raise ValueError(
                    f"MFMA unsupported packed carrier - src{role}={actual.name}, "
                    f"expected a supported {bits}-bit register carrier"
                )
            if _carrier_bits(actual) != bits:
                raise ValueError(f"MFMA requires src0 and src1 to have {bits} bits")
    else:
        source = _SOURCES[op_id]
        byte_source = isinstance(source, VectorType) and source.elem in (
            FP8E4M3,
            BF8E5M2,
        )
        if byte_source:
            # A byte bitcast preserves storage, not the logical number format.
            valid = True
            for role, value in enumerate(op.operands[:2]):
                actual = value.type
                valid = (
                    isinstance(actual, VectorType)
                    and type(actual.count) is int
                    and actual.count == 8
                    and actual.elem in (I8, U8, source.elem)
                )
                if valid:
                    continue
                if (
                    isinstance(actual, VectorType)
                    and type(actual.count) is int
                    and actual.count == 8
                    and actual.elem in (FP8E4M3, BF8E5M2)
                    and actual.elem != source.elem
                ):
                    raise ValueError(
                        f"MFMA source format mismatch - src{role}={actual.name}, "
                        f"expected {source.name} or vec<i8x8> or vec<u8x8>"
                    )
                break
        else:
            valid = all(
                _fixed_type_matches(v.type, source)
                or (
                    source in (F32, F64)
                    and _fixed_type_matches(v.type, VectorType(source, 1))
                )
                for v in op.operands[:2]
            )
        if not valid:
            required = VectorType(I8, 8) if byte_source else source
            raise ValueError(f"MFMA requires src0 and src1 to be {required.name}")
    count = 16 if op_id.startswith("mfma_f32_32x32") else 4
    if op_id == "mfma_f64_4x4x4_f64":
        count = 1
    expected = VectorType(F64 if op_id.startswith("mfma_f64_") else F32, count)
    if not _fixed_type_matches(
        op.operands[2].type, expected
    ) or not _fixed_type_matches(op.result.type, expected):
        raise ValueError(f"MFMA requires src2 and dst to be {expected.name}")
    if scaled and any(v.type != I32 for v in op.operands[3:]):
        raise ValueError("MFMA requires scale operands to be i32")
