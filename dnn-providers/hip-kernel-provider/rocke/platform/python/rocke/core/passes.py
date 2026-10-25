# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Conservative IR canonicalization passes.

These passes operate on the Python IR before LLVM lowering. They are small on
purpose: only pure scalar/vector bookkeeping ops are folded, CSE'd, or removed.
Loads, stores, barriers, async copies, and MFMA ops are never moved or removed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from .ir import KernelDef, Op, Region, Type, Value


@dataclass(frozen=True)
class PassStats:
    constants_folded: int = 0
    common_subexpressions: int = 0
    dead_ops_removed: int = 0

    def __add__(self, other: "PassStats") -> "PassStats":
        return PassStats(
            constants_folded=self.constants_folded + other.constants_folded,
            common_subexpressions=self.common_subexpressions
            + other.common_subexpressions,
            dead_ops_removed=self.dead_ops_removed + other.dead_ops_removed,
        )


def optimize_kernel(kernel: KernelDef, *, max_iter: int = 3) -> PassStats:
    """Run the default conservative pass pipeline in-place."""

    total = PassStats()
    for _ in range(max_iter):
        stats = canonicalize_region(kernel.body)
        total += stats
        if stats == PassStats():
            break
    return total


def canonicalize_region(region: Region) -> PassStats:
    """Fold constants, CSE pure ops, and remove dead pure ops in a region."""

    stats = PassStats()
    for op in region.ops:
        for nested in op.regions:
            stats += canonicalize_region(nested)

    replacements: Dict[str, Value] = {}
    cse_table: Dict[Tuple, Value] = {}
    new_ops: List[Op] = []
    folded = cse = 0

    for op in region.ops:
        _rewrite_operands(op, replacements)

        folded_value = _try_fold(op)
        if folded_value is not None and len(op.results) == 1:
            op.name = "arith.constant"
            op.operands = []
            op.attrs = {
                "value": _wrap_to_width(folded_value, op.result.type),
                "ity": _constant_ity(op.result.type),
            }
            op.regions = []
            folded += 1

        if op.is_pure and len(op.results) == 1:
            key = _cse_key(op)
            if key in cse_table:
                replacements[op.result.name] = cse_table[key]
                cse += 1
                continue
            cse_table[key] = op.result

        new_ops.append(op)

    region.ops = new_ops
    _rewrite_region_operands(region, replacements)
    dead = eliminate_dead_pure_ops(region)
    return PassStats(
        constants_folded=stats.constants_folded + folded,
        common_subexpressions=stats.common_subexpressions + cse,
        dead_ops_removed=stats.dead_ops_removed + dead,
    )


def eliminate_dead_pure_ops(region: Region) -> int:
    uses = _count_uses(region)
    kept: List[Op] = []
    removed = 0
    for op in region.ops:
        for nested in op.regions:
            removed += eliminate_dead_pure_ops(nested)
        if (
            op.is_pure
            and op.results
            and all(uses.get(r.name, 0) == 0 for r in op.results)
        ):
            removed += 1
            continue
        kept.append(op)
    region.ops = kept
    return removed


def _rewrite_region_operands(region: Region, replacements: Mapping[str, Value]) -> None:
    for op in region.ops:
        _rewrite_operands(op, replacements)
        for nested in op.regions:
            _rewrite_region_operands(nested, replacements)


def _rewrite_operands(op: Op, replacements: Mapping[str, Value]) -> None:
    op.operands = [replacements.get(v.name, v) for v in op.operands]
    for nested in op.regions:
        _rewrite_region_operands(nested, replacements)


def _count_uses(region: Region) -> Dict[str, int]:
    uses: Dict[str, int] = {}
    for op in region.ops:
        for operand in op.operands:
            uses[operand.name] = uses.get(operand.name, 0) + 1
        for nested in op.regions:
            nested_uses = _count_uses(nested)
            for k, v in nested_uses.items():
                uses[k] = uses.get(k, 0) + v
    return uses


def _cse_key(op: Op) -> Tuple:
    attrs = tuple(
        sorted((k, _freeze_attr(v)) for k, v in op.attrs.items() if k != "loc")
    )
    operands = tuple(v.name for v in op.operands)
    result_types = tuple(v.type.name for v in op.results)
    return (op.name, operands, attrs, result_types)


def _freeze_attr(v):
    """``v`` as a hashable key that is equal only for attrs that lower the same.

    Python's ``==`` would merge values the C++ engine keeps apart (it compares
    an attr's kind, and a float's bits): ``-0.0 == 0.0``, ``1 == 1.0 == True``,
    and a NaN never equals itself. So a float is keyed by its bit pattern and a
    bool by its kind; two constants ``0.0`` and ``-0.0`` stay two values.
    """
    if isinstance(v, (tuple, list)):
        return tuple(_freeze_attr(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze_attr(x)) for k, x in v.items()))
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, float):
        return ("float", struct.pack("<d", v))
    return v


def _constant_ity(t: Type) -> str:
    if t.name == "i1":
        return "i1"
    if t.name == "i64":
        return "i64"
    return "i32"


def _const_int(v: Value) -> Optional[int]:
    op = v.op
    if op is None or op.name != "arith.constant":
        return None
    ity = op.attrs.get("ity", "i32")
    if ity not in ("i1", "i32", "i64"):
        return None
    # Read at the constant's width, as folded results are written: an i32
    # literal 0x80000000 and a folded i32 sum of the same bits are one value.
    return _wrap_to_width(int(op.attrs["value"]), v.type)


def _int_width(t: Type) -> int:
    """Bit width of an integer constant's type (the folder's three widths)."""
    return {"i1": 1, "i64": 64}.get(t.name, 32)


def _wrap_to_width(value: int, t: Type) -> int:
    """``value`` as the hardware holds it in an integer of type ``t``.

    The fold computes with unbounded ints, but the folded constant stands for
    a fixed-width register, so it wraps as two's complement: an i32 sum of
    2**31 - 1 and 1 (defined for ``add_nuw``) reads back as -2**31, which is
    what later signed folds must compare. An i1 keeps 0 / 1, the form the
    compare folds and the printers use.
    """
    width = _int_width(t)
    if width == 1:
        return value & 1
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def _signed_quotient(a: int, b: int, t: Type) -> Optional[int]:
    """``a / b`` as LLVM ``sdiv`` computes it, or None when it cannot fold.

    ``arith.div`` and ``arith.mod`` lower to ``sdiv`` and ``srem``, which round
    the quotient toward zero (so ``-7 / 2`` is -3 and ``-7 % 2`` is -1), unlike
    Python's floored ``//`` and ``%``. Integer arithmetic keeps i64 values
    exact. A zero divisor, or a quotient past the signed range (the minimum
    value divided by -1), is undefined in LLVM and is left to run.
    """
    if b == 0:
        return None
    quotient = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        quotient = -quotient
    if quotient != _wrap_to_width(quotient, t):
        return None
    return quotient


def _try_fold(op: Op) -> Optional[int]:
    if len(op.results) != 1:
        return None
    if op.name == "arith.assume_range":
        # The marker is its first operand; a runtime end (operand 1) does not
        # stop a constant marked value from folding.
        return _const_int(op.operands[0])
    ints = [_const_int(v) for v in op.operands]
    if any(v is None for v in ints):
        return None

    if op.name == "arith.add":
        return ints[0] + ints[1]
    if op.name == "arith.sub":
        return ints[0] - ints[1]
    if op.name == "arith.mul":
        return ints[0] * ints[1]
    if op.name in ("arith.div", "arith.mod"):
        quotient = _signed_quotient(ints[0], ints[1], op.results[0].type)
        if quotient is None:
            return None
        return quotient if op.name == "arith.div" else ints[0] - ints[1] * quotient
    if op.name == "arith.and":
        return ints[0] & ints[1]
    if op.name == "arith.or":
        return ints[0] | ints[1]
    if op.name == "arith.zext":
        # Zero-extend: the operand read as unsigned at its own width.
        return ints[0] & ((1 << _int_width(op.operands[0].type)) - 1)
    if op.name == "arith.sext":
        return ints[0]
    if op.name == "arith.cmp":
        pred = op.attrs.get("pred", "lt")
        a, b = ints
        if pred == "ult":
            # Unsigned at the operand width: -1 reads as the largest value.
            mask = (1 << _int_width(op.operands[0].type)) - 1
            return int((a & mask) < (b & mask))
        return int(
            {
                "lt": a < b,
                "le": a <= b,
                "gt": a > b,
                "ge": a >= b,
                "eq": a == b,
                "ne": a != b,
            }[pred]
        )
    if op.name == "arith.select":
        cond, lhs, rhs = ints
        return lhs if cond else rhs
    return None
