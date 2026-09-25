# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Logical TF32 operand contracts, independent of the physical i32 carrier."""

TF32_MMA = {"mfma_f32_16x16x8_xf32": 4, "mfma_f32_32x32x4_xf32": 16}


def tf32_mma_error(op_id, operands, results=None):
    count = TF32_MMA.get(op_id)
    if count is None:
        if any(
            v.type.name == "tf32" or v.type.name.startswith("vec<tf32x")
            for v in operands
        ):
            return "TF32 operands require an XF32 MMA"
        return None
    expected = ["vec<tf32x2>", "vec<tf32x2>", f"vec<f32x{count}>"]
    if [v.type.name for v in operands] != expected:
        return "XF32 MMA requires two vec<tf32x2> operands and its FP32 accumulator"
    if results is not None and [v.type.name for v in results] != [expected[2]]:
        return "XF32 MMA result must match its FP32 accumulator"
    return None


def tf32_op_error(op):
    op_id = op.attrs.get("op_id", op.name.removeprefix("tile."))
    error = (
        tf32_mma_error(op_id, op.operands, op.results)
        if op.name == "tile.mma" or op.name.startswith(("tile.mfma", "tile.wmma"))
        else None
    )
    if error:
        return error
    if op.name.startswith(("arith.", "math.")) and op.name not in (
        "arith.bitcast",
        "arith.select",
    ):
        if any(
            v.type.name == "tf32" or v.type.name.startswith("vec<tf32x")
            for v in [*op.operands, *op.results]
        ):
            return "TF32 arithmetic requires an explicit conversion to f32"
    return None
