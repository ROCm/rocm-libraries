# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where an integer Value comes from: who can see it, and whether an unchecked
load reaches it.

``IRBuilder.gpu_checked_value`` reads both before it emits a check:

- **Who can see a value.** A value *depends on* a leaf when the leaf reaches it
  through any operand, a select or branch condition and a loop-carried value
  included. :func:`sources_of` lists the leaves: kernel parameters, constants
  and thread ids, which the host checks or bounds before every launch, and
  ``gpu_only`` when any other leaf is reached -- a memory load, a block id, any
  value the host cannot compute. The host never checks the grid, so a value
  computed from block ids is GPU-only.
- **Load taint.** A value is *tainted* when an unchecked load reaches it
  through data operands: every operand except a select's condition, an
  ``scf.if_else`` condition and a loop's upper bound. A compare's result is
  never tainted (a 0 or 1 cannot wrap), and neither is a range marker's (its
  writer checked it). :func:`first_unchecked_arithmetic` finds an ``add``,
  ``sub``, ``mul``, ``div`` or ``mod`` with a tainted operand on the data path:
  the first two can wrap and lower with ``nsw`` (poison on overflow), the others
  are undefined for a zero or overflowing divisor, so the value could be wrong
  before any check sees it.

Both walks read the IR only (no builder) and know nothing of the tiling layer.
``cpp/core/ir/gpu_check.cpp`` mirrors them, in the same order, so both engines
name the same Value in the same message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .ir import Op, Region, Value

__all__ = [
    "THREAD_ID_OP",
    "WRAPPING_ARITHMETIC",
    "Sources",
    "first_unchecked_arithmetic",
    "loaded_directly",
    "reads_memory",
    "sources_of",
    "value_text",
]

THREAD_ID_OP = "gpu.thread_id"
# The arithmetic a tainted operand must not reach before its check.
WRAPPING_ARITHMETIC = ("arith.add", "arith.sub", "arith.mul", "arith.div", "arith.mod")
_CONSTANT = "arith.constant"
_MARKER = "arith.assume_range"
_LOOP = "scf.for"
_BRANCH = "scf.if_else"
_SELECT = "arith.select"
_COMPARES = ("arith.cmp", "arith.fcmp")
_READFIRSTLANE = "tile.readfirstlane"
# An op that reads memory has one of these in its name (global, buffer, LDS and
# cluster loads, LDS transposed reads, atomics that return the old value).
_MEMORY_READ_PARTS = ("load", "atomic", "ds_read")
_SYMBOLS = {
    "arith.add": "+",
    "arith.sub": "-",
    "arith.mul": "*",
    "arith.div": "/",
    "arith.mod": "%",
}
_IDS = {"gpu.thread_id": "thread_id", "gpu.block_id": "block_id"}


def reads_memory(op: Op) -> bool:
    """Whether ``op``'s result is read from memory, where no check has seen it."""
    return not op.is_pure and any(part in op.name for part in _MEMORY_READ_PARTS)


# --- message text ---------------------------------------------------------------------------


def value_text(value: Value) -> str:
    """``value`` as an error message names it: a parameter's name, an int, a
    ``thread_id_x``, or one ``add`` / ``sub`` / ``mul`` / ``div`` / ``mod`` of
    those (``T - 1``); anything else by its SSA name."""
    op = value.op
    if value.param is None and op is not None and op.name in _SYMBOLS:
        if len(op.operands) == 2:
            left, right = op.operands
            return f"{_atom_text(left)} {_SYMBOLS[op.name]} {_atom_text(right)}"
    return _atom_text(value)


def _atom_text(value: Value) -> str:
    if value.param is not None:
        return value.param.name
    op = value.op
    if op is None:
        return value.name
    if op.name == _CONSTANT:
        constant = op.attrs.get("value")
        if isinstance(constant, int) and not isinstance(constant, bool):
            return str(constant)
    if op.name in _IDS and isinstance(op.attrs.get("axis"), str):
        return f"{_IDS[op.name]}_{op.attrs['axis']}"
    return value.name


# --- structured inputs ------------------------------------------------------------------------


def _carried_index(value: Value, loop: Op) -> int | None:
    """Which loop-carried value ``value`` is (an iter arg or a loop result), or
    ``None`` for the induction variable."""
    for index, result in enumerate(loop.results):
        if result is value:
            return index
    if value.name == loop.attrs.get("iv"):
        return None
    names = [arg.get("name") for arg in loop.attrs.get("iter_args", [])]
    return names.index(value.name) if value.name in names else None


def _yielded(region: Region, index: int) -> list[Value]:
    """The value ``region``'s closing ``scf.yield`` gives at ``index``, if any."""
    if region.ops and region.ops[-1].name == "scf.yield":
        values = region.ops[-1].operands
        if index < len(values):
            return [values[index]]
    return []


def _loop_inputs(value: Value, loop: Op, *, data: bool) -> list[Value]:
    """A loop value's inputs. The induction variable takes the lower bound and
    the step (data) and the upper bound (the condition); a carried value takes
    its init and what the body yields for it (data), and the three bounds."""
    lower, upper, step = loop.operands[:3]
    index = _carried_index(value, loop)
    if index is None:
        return [lower, step] if data else [lower, upper, step]
    carried = [loop.operands[3 + index], *_yielded(loop.regions[0], index)]
    return carried if data else [lower, upper, step, *carried]


def _branch_inputs(value: Value, branch: Op, *, data: bool) -> list[Value]:
    """An ``scf.if_else`` result's inputs: what each branch yields (data), and
    the condition."""
    index = next(i for i, result in enumerate(branch.results) if result is value)
    yielded = [v for region in branch.regions for v in _yielded(region, index)]
    return yielded if data else [branch.operands[0], *yielded]


# --- who can see a value ----------------------------------------------------------------------


class Sources(NamedTuple):
    """The leaves a Value depends on. ``gpu_only`` is True when one of them is
    a value only the GPU can see; ``params`` and ``thread_ids`` are then
    incomplete (the walk stops there). Otherwise every leaf is a kernel
    parameter, a constant or a thread id, listed in the order first reached."""

    gpu_only: bool
    params: tuple[Value, ...]
    thread_ids: tuple[Value, ...]


def sources_of(value: Value) -> Sources:
    """The leaves ``value`` depends on, depth first in operand order."""
    seen: set[int] = set()
    params: list[Value] = []
    thread_ids: list[Value] = []
    stack = [value]
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        op = v.op
        if v.param is not None:
            params.append(v)
            continue
        if op is None:
            return Sources(True, tuple(params), tuple(thread_ids))
        if op.name == _CONSTANT:
            continue
        if op.name == THREAD_ID_OP:
            thread_ids.append(v)
            continue
        if op.name == _LOOP:
            inputs = _loop_inputs(v, op, data=False)
        elif op.name == _BRANCH:
            inputs = _branch_inputs(v, op, data=False)
        elif op.is_pure and op.operands:
            inputs = list(op.operands)
        else:
            # A load, a block id, or any other value the host cannot compute.
            return Sources(True, tuple(params), tuple(thread_ids))
        stack.extend(reversed(inputs))
    return Sources(False, tuple(params), tuple(thread_ids))


# --- load taint -------------------------------------------------------------------------------


def _data_inputs(value: Value) -> list[Value]:
    op = value.op
    if value.param is not None or op is None:
        return []
    if op.name in (_CONSTANT, _MARKER, *_COMPARES):
        return []
    if op.name == _LOOP:
        return _loop_inputs(value, op, data=True)
    if op.name == _BRANCH:
        return _branch_inputs(value, op, data=True)
    if op.name == _SELECT:
        return list(op.operands[1:])
    return list(op.operands)


def _post_order(value: Value) -> list[Value]:
    """Every Value on ``value``'s data path, each after its data inputs (a
    loop-carried value is reached before the body that yields it)."""
    seen: set[int] = set()
    order: list[Value] = []
    stack: list[tuple[Value, bool]] = [(value, False)]
    while stack:
        v, expanded = stack.pop()
        if expanded:
            order.append(v)
            continue
        if id(v) in seen:
            continue
        seen.add(id(v))
        stack.append((v, True))
        stack.extend((d, False) for d in reversed(_data_inputs(v)) if id(d) not in seen)
    return order


def loaded_directly(value: Value) -> bool:
    """Whether ``value`` is a memory load's result, possibly made wave-uniform
    by ``readfirstlane``."""
    while value.op is not None and value.op.name == _READFIRSTLANE:
        value = value.op.operands[0]
    return value.op is not None and value.param is None and reads_memory(value.op)


def first_unchecked_arithmetic(value: Value) -> tuple[Op, Value] | None:
    """The first ``add`` / ``sub`` / ``mul`` / ``div`` / ``mod`` on ``value``'s
    data path with a tainted operand, and that operand (a loaded value itself
    when the op has one); ``None`` when there is none. "First" is depth first
    in operand order, inputs before users, so the op reported is the one
    nearest the load."""
    order = _post_order(value)
    tainted = {
        id(v): v.op is not None and v.param is None and reads_memory(v.op)
        for v in order
    }
    changed = True
    while changed:  # a loop-carried value can be tainted by what its body yields
        changed = False
        for v in order:
            if not tainted[id(v)] and any(tainted[id(d)] for d in _data_inputs(v)):
                tainted[id(v)] = True
                changed = True
    for v in order:
        op = v.op
        if op is not None and v.param is None and op.name in WRAPPING_ARITHMETIC:
            operands = [d for d in op.operands if tainted[id(d)]]
            if operands:
                # Name the loaded value itself when the op has one: that is what to check.
                direct = [d for d in operands if loaded_directly(d)]
                return op, (direct or operands)[0]
    return None
