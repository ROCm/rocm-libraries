# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Actual SSA operand checks shared by LLVM and HIP WMMA emitters."""

from __future__ import annotations

from .ir import I32, Op, Type, VectorType


def validate_unscaled_wmma(
    op: Op, elem: Type, count: int = 8, *, source_elem: Type, source_count: int
) -> None:
    """Reject values that cannot use the selected instruction's fixed ABI.

    Generic MMA metadata allows independent src2/dst contracts. Only the
    concrete emitter can impose the physical accumulator/result signature.
    Mirrors ``_validate_unscaled_wmma`` in the native LLVM lowerer.
    """
    if len(op.operands) != 3 or len(op.results) != 1:
        raise ValueError("unscaled WMMA expects 3 operands and 1 result")
    expected = VectorType(elem, count)
    if op.operands[2].type != expected or op.result.type != expected:
        raise ValueError(f"unscaled WMMA requires src2 and dst to be {expected.name}")
    source = VectorType(source_elem, source_count)
    for role in (0, 1):
        if op.operands[role].type != source:
            raise ValueError(f"unscaled WMMA requires src{role} to be {source.name}")


def validate_scaled_wmma_sources(op: Op, words: tuple[int, int], scale: Type) -> None:
    """Check each selected matrix carrier and the shared scale carrier."""
    for role, count in enumerate(words):
        expected = VectorType(I32, count)
        if op.operands[role].type != expected:
            raise ValueError(f"scaled WMMA requires src{role} to be {expected.name}")
    if op.operands[3].type != scale or op.operands[4].type != scale:
        raise ValueError(
            f"{op.name} expects {scale.name} scale operands, got "
            f"{op.operands[3].type.name}/{op.operands[4].type.name}"
        )
