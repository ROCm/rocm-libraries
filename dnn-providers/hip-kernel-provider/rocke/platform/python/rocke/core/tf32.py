# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Logical TF32 operand contracts, independent of the physical i32 carrier."""

from __future__ import annotations

from .arch import ArchTarget, known_arches

TF32_MMA = {"mfma_f32_16x16x8_xf32": 4, "mfma_f32_32x32x4_xf32": 16}


def tf32_mma_target_error(op_id: str | None, arch: ArchTarget) -> str | None:
    """The error for a TF32 MMA ``op_id`` that ``arch``'s MMA catalog does not list.

    Returns None for any other op_id, and for a TF32 op_id the catalog lists.
    """
    if op_id not in TF32_MMA or arch.mma.by_op_id(op_id) is not None:
        return None
    listing = [
        gfx for gfx in known_arches() if ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    ]
    where = (
        f"it is listed for {', '.join(listing)}" if listing else "no target lists it"
    )
    return (
        f"no MMA instruction found for {op_id!r} on {arch.gfx!r} -- the target's "
        f"MMA catalog does not list it; {where}"
    )


def _emulated_tf32_mma_error(op_id, lens, operands, results):
    """The operand contract of a TF32 atom the hardware does not encode.

    Same ABI as the native atoms -- logical TF32 A/B carried as I32, FP32
    accumulator -- so one kernel body reaches either through a catalog query.
    Only the fragment lengths differ, and they come from the catalog.
    """
    a_len, b_len, c_len = lens
    expected = [f"vec<tf32x{a_len}>", f"vec<tf32x{b_len}>", f"vec<f32x{c_len}>"]
    if [v.type.name for v in operands] != expected:
        return f"{op_id} requires {expected[0]}, {expected[1]} and {expected[2]}"
    if results is not None and [v.type.name for v in results] != [expected[2]]:
        return f"{op_id} result must match its FP32 accumulator"
    return None


def tf32_mma_error(op_id, operands, results=None):
    count = TF32_MMA.get(op_id)
    if count is None:
        from .arch.target import emulated_tf32_operand_lens

        lens = emulated_tf32_operand_lens(op_id)
        if lens is not None:
            return _emulated_tf32_mma_error(op_id, lens, operands, results)
        if any(
            v.type.name == "tf32" or v.type.name.startswith("vec<tf32x")
            for v in operands
        ):
            return "TF32 operands require an XF32 MMA"
        if results is not None and any(
            v.type.name == "tf32" or v.type.name.startswith("vec<tf32x")
            for v in results
        ):
            return "MMA results must not use TF32"
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
    # Vector arithmetic must not operate on the integer carrier bits either.
    # Keep the operations that only move, reinterpret, or select payloads.
    if op.name.startswith(("arith.", "math.", "vector.")) and op.name not in (
        "arith.bitcast",
        "arith.select",
        "vector.bitcast",
        "vector.extract",
        "vector.pack",
        "vector.concat",
        "vector.insert",
        "vector.splat",
        "vector.select",
    ):
        if any(
            v.type.name == "tf32" or v.type.name.startswith("vec<tf32x")
            for v in [*op.operands, *op.results]
        ):
            return "TF32 arithmetic requires an explicit conversion to f32"
    return None
