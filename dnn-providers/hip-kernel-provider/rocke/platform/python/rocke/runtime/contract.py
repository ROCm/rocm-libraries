# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A kernel's parameter contract: what the host checks before every launch.

A kernel states facts on its parameter declarations (``b.param("M", I32, ge=1,
multiple_of=64)``), alignments on its pointer parameters, and may state its exact
block size (``b.kernel.attrs["workgroup_size"]``). The GPU trusts all of them, so
the host must check them before it launches. :func:`parameter_contract` turns a
``KernelDef`` into those checks -- one row per stated fact -- and
:meth:`ParameterContract.check` evaluates them against one launch's arguments.

The rows are:

- an integer fact: ``ge`` / ``gt`` / ``le`` / ``lt`` / ``eq`` / ``multiple_of``
  on an integer parameter. A bound is an int or a relation over earlier
  parameters (``ge=product_of(M, ldc)``), evaluated by the one intexpr
  evaluator, ``rocke.core.intexpr.eval_intexpr``. ``eq`` states a baked length;
- a pointer alignment (``align``), checked on the pointer's address;
- the exact workgroup size, checked on the launch block.

The contract also lists, without checking them, the pointer promises the GPU
trusts (``noalias``, ``readonly`` and the other LLVM argument attributes) and
the facts the kernel checks itself with ``gpu_checked_value`` (a false one traps
the kernel, after which the process can no longer use the GPU).

The contract is read from the ``KernelDef`` a compile artifact holds; nothing
about it is emitted into the code object. So only a launcher that holds the
kernel can check it: ``KernelLauncher.from_artifact``. This module lives in
``rocke.runtime``, below ``rocke.helpers``, and reads only ``rocke.core``.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.intexpr import IntExprError, eval_intexpr, relation_error, relation_text
from ..core.ir import (
    _INT_TYPE_LIMITS,
    FACT_WORDS,
    KernelDef,
    Op,
    PtrType,
    assume_range_bound_operands,
    param_fact_errors,
)
from ..core.value_sources import value_text
from .packing import _as_ptr

__all__ = [
    "AlignmentRow",
    "FactRow",
    "LaunchRefused",
    "LaunchSupport",
    "ParameterContract",
    "parameter_contract",
]

# The pointer attributes the GPU trusts and nothing checks. ``align`` is a row;
# ``addr_space`` only picks how the pointer lowers.
_PROMISES = (
    "noalias",
    "readonly",
    "writeonly",
    "nocapture",
    "nonnull",
    "dereferenceable",
)
# How a refusal shows a relation with the arguments put in.
_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "//"}
# A refusal row prints the declaration that states it in a column of its own.
_HINT_COLUMN = 60


class LaunchRefused(ValueError):
    """A launch the parameter contract does not accept.

    ``reasons`` holds one line per failed row; the message lists them all.
    """

    def __init__(self, message: str, reasons: Sequence[str]) -> None:
        super().__init__(message)
        self.reasons: tuple[str, ...] = tuple(reasons)

    @classmethod
    def of(cls, kernel_name: str, reasons: Sequence[str]) -> LaunchRefused:
        """The refusal of one kernel's launch."""
        lines = "".join(f"\n  {r}" for r in reasons)
        return cls(f"{kernel_name} refused this launch:{lines}", reasons)


@dataclass(frozen=True)
class LaunchSupport:
    """Whether a contract accepts a launch's arguments.

    ``reasons`` names every failed row. ``checked_at_launch`` names the rows
    this check could not evaluate because the arguments left out what they
    need -- an alignment row without its pointer -- and that the launch checks;
    they count neither as passed nor as failed.
    """

    ok: bool
    reasons: tuple[str, ...] = ()
    checked_at_launch: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactRow:
    """``param word bound``: an integer fact the host checks.

    ``bound`` is an int (also ``multiple_of``'s divisor) or a relation's intexpr
    tree, whose ``{"param": NAME}`` leaves name earlier integer parameters.
    """

    param: str
    word: str
    bound: int | dict[str, Any]

    @property
    def bound_text(self) -> str:
        """The bound as the author wrote it: ``64``, ``N`` or ``product_of(M, ldc)``."""
        return relation_text(self.bound)

    def __str__(self) -> str:
        return f"{self.param} {self.word} {self.bound_text}"


@dataclass(frozen=True)
class AlignmentRow:
    """``param``'s address is a multiple of ``align`` bytes."""

    param: str
    align: int

    def __str__(self) -> str:
        return f"{self.param}: align {self.align}"


@dataclass(frozen=True)
class ParameterContract:
    """Everything one kernel states about its launch arguments.

    ``facts``, ``alignments`` and ``workgroup_size`` are the rows the host
    checks; ``promises`` (``(param, attr text)``) and ``gpu_checks`` (one text
    per ``gpu_checked_value``) are listed and not checked. ``params`` is the
    parameter list, ``(name, type name)`` in argument order.
    """

    kernel_name: str
    params: tuple[tuple[str, str], ...]
    facts: tuple[FactRow, ...]
    alignments: tuple[AlignmentRow, ...]
    workgroup_size: int | None
    promises: tuple[tuple[str, str], ...]
    gpu_checks: tuple[str, ...]

    @property
    def states_facts(self) -> bool:
        """Whether the kernel states an integer fact or an exact workgroup size:
        rows a launch from the code-object bytes alone would leave unchecked.

        Pointer alignment is left out: it is an LLVM argument attribute kernels
        stated long before the rest of the contract, and their written code
        objects were always launched without checking it.
        """
        return bool(self.facts) or self.workgroup_size is not None

    def check(
        self,
        values: Mapping[str, Any],
        *,
        block: Sequence[int] | None = None,
        at_launch: bool = False,
    ) -> LaunchSupport:
        """Evaluate every row against ``values`` (argument name -> value).

        ``block`` is the launch block, checked against ``workgroup_size`` when
        both are given (``None`` takes the kernel's own size). With
        ``at_launch=False`` -- an applicability check, whose arguments may hold
        no pointers -- an alignment row whose pointer is left out is reported in
        ``checked_at_launch``; at launch every argument a row reads must be given.
        """
        reasons: list[str] = []
        ints: dict[str, int] = {}
        for name, type_name in self.params:
            if type_name in _INT_TYPE_LIMITS and name in values:
                ints[name] = operator.index(values[name])
                lo, hi = _INT_TYPE_LIMITS[type_name]
                if not lo <= ints[name] <= hi:
                    reasons.append(
                        f"{name} = {ints[name]}: outside {type_name.upper()} "
                        f"({lo} .. {hi})"
                    )
        missing = set()
        for row in self.facts:
            needed = [row.param, *_leaf_names(row.bound)]
            absent = [n for n in needed if n not in ints and n not in missing]
            for n in absent:
                missing.add(n)
                reasons.append(f"{n}: no value given")
            if any(n not in ints for n in needed):
                continue
            reason = _fact_reason(row, ints)
            if reason is not None:
                reasons.append(reason)
        unchecked: list[str] = []
        for row in self.alignments:
            if row.param not in values:
                if at_launch:
                    reasons.append(f"{row.param}: no value given")
                else:
                    unchecked.append(f"{row} -- checked at launch")
                continue
            address = _as_ptr(values[row.param])
            if address % row.align:
                reasons.append(f"{row.param} = 0x{address:x}: needs align {row.align}")
        if block is not None and self.workgroup_size is not None:
            given = tuple(int(x) for x in block)
            exact = (self.workgroup_size, 1, 1)
            if given != exact:
                reasons.append(
                    f"block {given} differs from the kernel's workgroup_size "
                    f"{self.workgroup_size} -- pass {exact} or leave block unset"
                )
        return LaunchSupport(not reasons, tuple(reasons), tuple(unchecked))

    def __str__(self) -> str:
        host: list[str] = []
        stated: dict[str, list[str]] = {}
        for row in self.facts:
            stated.setdefault(row.param, []).append(f"{row.word} {row.bound_text}")
        for name, type_name in self.params:
            if name in stated:
                host.append(
                    f"{name}: {', '.join(_with_type_limits(stated[name], type_name))}"
                )
        host.extend(str(row) for row in self.alignments)
        if self.workgroup_size is not None:
            host.append(f"block: workgroup_size {self.workgroup_size}")
        promised: dict[str, list[str]] = {}
        for name, text in self.promises:
            promised.setdefault(name, []).append(text)
        groups = (
            ("checked on the host before every launch", host),
            (
                "trusted by the GPU, not checked",
                [f"{n}: {', '.join(t)}" for n, t in promised.items()],
            ),
            (
                "checked on the GPU (a false fact traps the kernel; the process "
                "can no longer use the GPU)",
                list(self.gpu_checks),
            ),
        )
        lines = [f"parameter contract of {self.kernel_name}:"]
        for title, rows in groups:
            lines.append(f"  {title}:")
            lines.extend(f"    {r}" for r in rows or ["(none)"])
        return "\n".join(lines)


def parameter_contract(kernel: KernelDef) -> ParameterContract:
    """The contract ``kernel`` states, read from ``Param.attrs``, the
    ``workgroup_size`` attr and its ``gpu_checked_value`` markers.

    Raises ``ValueError`` with the verifier's message when a declaration states
    something it may not, since such a contract cannot be checked.
    """
    facts: list[FactRow] = []
    alignments: list[AlignmentRow] = []
    promises: list[tuple[str, str]] = []
    for index, param in enumerate(kernel.params):
        errors = param_fact_errors(kernel.params, index)
        if errors:
            raise ValueError(f"{kernel.name}: {errors[0]}")
        for word in FACT_WORDS:
            if word in param.attrs:
                facts.append(FactRow(param.name, word, _bound(param.attrs[word])))
        if isinstance(param.type, PtrType):
            if "align" in param.attrs:
                alignments.append(AlignmentRow(param.name, param.attrs["align"]))
            for attr in _PROMISES:
                value = param.attrs.get(attr, False)
                if value is True:
                    promises.append((param.name, attr))
                elif value is not False:
                    promises.append((param.name, f"{attr} {value}"))
    size = kernel.attrs.get("workgroup_size")
    return ParameterContract(
        kernel_name=kernel.name,
        params=tuple((p.name, p.type.name) for p in kernel.params),
        facts=tuple(facts),
        alignments=tuple(alignments),
        workgroup_size=None if size is None else int(size),
        promises=tuple(promises),
        gpu_checks=tuple(_gpu_check_text(op) for op in _gpu_checks(kernel.body.ops)),
    )


# --- rows ----------------------------------------------------------------------------------------


def _bound(stored: Any) -> int | dict[str, Any]:
    """A stored bound: an int, or the tree a relation string holds."""
    if isinstance(stored, str):
        tree = json.loads(stored)
        if relation_error(tree) is not None:  # param_fact_errors refused it already
            raise ValueError(f"not a relation: {stored}")
        return tree
    return stored


def _leaf_names(bound: Any) -> list[str]:
    if isinstance(bound, dict):
        ((key, value),) = bound.items()
        if key == "param":
            return [value]
        return _leaf_names(value[0]) + _leaf_names(value[1])
    return []


def _holds(word: str, value: int, bound: int) -> bool:
    if word == "multiple_of":
        return value % bound == 0
    return {
        "ge": value >= bound,
        "gt": value > bound,
        "le": value <= bound,
        "lt": value < bound,
        "eq": value == bound,
    }[word]


def _fact_reason(row: FactRow, ints: Mapping[str, int]) -> str | None:
    """Why ``row`` fails for the arguments ``ints``, or ``None`` when it holds."""
    value = ints[row.param]
    head = f"{row.param} = {value}: needs {row}"
    try:
        bound = eval_intexpr(row.bound, {}, {}, {}, params=ints)
    except IntExprError as exc:
        return f"{head}, which cannot be evaluated: {exc}"
    if _holds(row.word, value, bound):
        return None
    if not isinstance(row.bound, dict):
        return _with_hint(head, f"{row.word}={row.bound_text}", row.param)
    if "param" in row.bound:
        return _with_hint(
            f"{head} ({row.bound_text} = {bound})",
            f"{row.word}={row.bound_text}",
            row.param,
        )
    return f"{head} (= {_infix(row.bound, ints)} = {bound})"


def _with_hint(text: str, words: str, param: str) -> str:
    """``text`` with the declaration that states the row in its own column."""
    hint = f'(b.param("{param}", ..., {words}))'
    return f"{text:<{_HINT_COLUMN - 1}} {hint}"


def _infix(tree: Any, ints: Mapping[str, int]) -> str:
    """A relation with the arguments put in: ``1000 * 100``."""

    def node(t: Any, nested: bool) -> str:
        if not isinstance(t, dict):
            return str(t)
        ((key, value),) = t.items()
        if key == "param":
            return str(ints[value])
        text = f"{node(value[0], True)} {_INFIX[key]} {node(value[1], True)}"
        return f"({text})" if nested else text

    return node(tree, False)


def _with_type_limits(words: list[str], type_name: str) -> list[str]:
    """A parameter's stated words (``FACT_WORDS`` order), with each side no word
    states shown at its type's limit."""
    if any(w.startswith("eq ") for w in words):
        return words
    lo, hi = _INT_TYPE_LIMITS[type_name]
    limit = f"({type_name.upper()} limit)"
    lower = [w for w in words if w.startswith(("ge ", "gt "))] or [f"ge {lo} {limit}"]
    upper = [w for w in words if w.startswith(("le ", "lt "))] or [f"le {hi} {limit}"]
    rest = [w for w in words if w.startswith("multiple_of ")]
    return lower + upper + rest


# --- gpu_checked_value markers -------------------------------------------------------------------


def _gpu_checks(ops: Sequence[Op]) -> list[Op]:
    """Every ``gpu_checked_value`` marker in ``ops`` and their regions, in order."""
    found: list[Op] = []
    for op in ops:
        if (
            op.name == "arith.assume_range"
            and op.attrs.get("by") == "gpu_checked_value"
        ):
            found.append(op)
        for region in op.regions:
            found.extend(_gpu_checks(region.ops))
    return found


def _gpu_check_text(op: Op) -> str:
    """``value: ge 0, le T`` for one ``gpu_checked_value`` marker."""
    attrs = op.attrs
    ge_operand, lt_operand = assume_range_bound_operands(op)
    words: list[str] = []
    if ge_operand is not None:
        words.append(
            _end_text("ge", "gt", value_text(ge_operand), attrs.get("ge_plus", 0))
        )
    elif "ge" in attrs:
        words.append(f"ge {attrs['ge']}")
    if lt_operand is not None:
        words.append(
            _end_text("lt", "le", value_text(lt_operand), attrs.get("lt_plus", 0))
        )
    elif "lt" in attrs:
        words.append(f"lt {attrs['lt']}")
    if attrs.get("multiple_of", 1) > 1:
        words.append(f"multiple_of {attrs['multiple_of']}")
    return f"{value_text(op.operands[0])}: {', '.join(words)}"


def _end_text(word: str, off_by_one: str, bound: str, plus: int) -> str:
    """One Value end of a marker: ``lt T`` with ``lt_plus`` 1 reads ``le T``."""
    if plus == 0:
        return f"{word} {bound}"
    if plus == 1:
        return f"{off_by_one} {bound}"
    sign = "+" if plus > 0 else "-"
    return f"{word} {bound} {sign} {abs(plus)}"
