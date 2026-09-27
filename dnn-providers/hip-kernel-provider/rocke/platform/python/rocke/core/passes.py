# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Conservative IR canonicalization passes.

These passes operate on the Python IR before LLVM lowering. They are small on
purpose: only pure scalar/vector bookkeeping ops are folded, CSE'd, or removed.
Loads, stores, barriers, async copies, and MFMA ops are never moved or removed.
"""

from __future__ import annotations

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
            op.attrs = {"value": folded_value, "ity": _constant_ity(op.result.type)}
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
    if isinstance(v, (tuple, list)):
        return tuple(_freeze_attr(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze_attr(x)) for k, x in v.items()))
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
    return int(op.attrs["value"])


def _try_fold(op: Op) -> Optional[int]:
    if len(op.results) != 1:
        return None
    ints = [_const_int(v) for v in op.operands]
    if any(v is None for v in ints):
        return None

    if op.name == "arith.add":
        return ints[0] + ints[1]
    if op.name == "arith.sub":
        return ints[0] - ints[1]
    if op.name == "arith.mul":
        return ints[0] * ints[1]
    if op.name == "arith.div" and ints[1] != 0:
        return int(ints[0] / ints[1])
    if op.name == "arith.mod" and ints[1] != 0:
        return ints[0] % ints[1]
    if op.name == "arith.and":
        return ints[0] & ints[1]
    if op.name == "arith.or":
        return ints[0] | ints[1]
    if op.name == "arith.zext" or op.name == "arith.sext":
        return ints[0]
    if op.name == "arith.cmp":
        pred = op.attrs.get("pred", "lt")
        a, b = ints
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
