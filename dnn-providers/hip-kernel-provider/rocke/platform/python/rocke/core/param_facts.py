# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The facts a kernel parameter states on its declaration, and their rules.

A kernel parameter states its facts on its declaration (``IRBuilder.param``)
and they live in ``Param.attrs``: ``KernelLauncher.from_artifact`` checks them
on the host before each launch, and the GPU trusts them.
The six :data:`FACT_WORDS` are the only spelling of a comparison fact. A bound
is an int, or a *relation*: an intexpr tree over earlier parameters
(``rocke.core.intexpr``), stored as canonical JSON in one string attr.

This module is the one home of what a declaration may state
(:func:`declaration_errors`: the builder raises the first message, the
verifier reports all of them) and of how a stated bound resolves to numbers
(:func:`param_interval_resolver`, :func:`stated_param_interval`).
``cpp/core/ir/param_facts.cpp`` gives the same texts. It reads a declaration
through its ``name``, ``type`` and ``attrs`` only, so it sits below
``rocke.core.ir``, which imports it.
"""

from __future__ import annotations

import json
import numbers
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any

from .dtypes import UNSIGNED_INTEGER_DTYPES, dtype_info
from .intexpr import (
    EXACT_INT_MAX,
    INT64_MAX,
    INT64_MIN,
    relation_error,
    relation_names,
    relation_text,
)

if TYPE_CHECKING:
    from .ir import Param, Type

__all__ = [
    "FACT_INT_TYPES",
    "FACT_WORDS",
    "LOWER_WORDS",
    "PARAM_ATTR_KEYS",
    "POINTER_ATTRS",
    "UNSTATED",
    "UPPER_WORDS",
    "Interval",
    "bound_text",
    "checked_end",
    "declaration_errors",
    "int_type_limits",
    "is_plain_int",
    "param_fact_texts",
    "param_interval_resolver",
    "stated_param_interval",
    "stored_int",
    "stored_relation",
    "two_bounds_refusals",
]

# The inclusive ``(lower, upper)`` a value can take; ``None`` for an end that
# is not known.
Interval = tuple[int | None, int | None]

# The words that take a bound, in the order they are checked and printed, with
# the end of the value each states (eq states both): ROCKE_INT_BOUND_WORDS in C++.
_BOUND_WORD_ENDS = {
    "ge": "lower",
    "gt": "lower",
    "le": "upper",
    "lt": "upper",
    "eq": "both",
}
_BOUND_WORDS = tuple(_BOUND_WORD_ENDS)
# The words that state an integer fact: the words that take a bound, then multiple_of.
FACT_WORDS = (*_BOUND_WORDS, "multiple_of")
# The words that state each end; a fact states each end once (two_bounds_refusals).
LOWER_WORDS = tuple(w for w, end in _BOUND_WORD_ENDS.items() if end == "lower")
UPPER_WORDS = tuple(w for w, end in _BOUND_WORD_ENDS.items() if end == "upper")
# LLVM argument attributes, emitted for a pointer parameter (``align`` is also
# host-checked).
POINTER_ATTRS = (
    "noalias",
    "readonly",
    "writeonly",
    "nocapture",
    "nonnull",
    "align",
    "dereferenceable",
    "addr_space",
)
# Every key a parameter may carry, as the portable-IR importers list it
# (ROCKE_PARAM_ATTR_KEYS in rocke/ir.h).
PARAM_ATTR_KEYS = POINTER_ATTRS + FACT_WORDS

# The integer types a fact may be stated on (the integer scalars a value can
# have). ``rocke_int_type_limits`` in ``cpp/core/ir/param_facts.cpp`` lists
# the same.
FACT_INT_TYPES = ("i8", "i16", "i32", "i64", "u8")


def int_type_limits(type_name: str) -> tuple[int, int]:
    """The inclusive ``(lo, hi)`` range of ``type_name``, one of
    :data:`FACT_INT_TYPES`: its width from ``dtype_info``, unsigned for
    ``UNSIGNED_INTEGER_DTYPES`` and two's complement otherwise. The C++ engine's
    ``rocke_int_type_limits`` is its twin.

    Raises ``ValueError`` for any other type.
    """
    if type_name not in FACT_INT_TYPES:
        raise ValueError(
            f"int_type_limits failed - type_name={type_name!r}, expected one of "
            f"{FACT_INT_TYPES}"
        )
    bits = dtype_info(type_name).encoded_bits
    if type_name in UNSIGNED_INTEGER_DTYPES:
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _type_name(t: Type | None) -> str:
    """``t.name``; a placeholder declaration may carry no type yet (``None``)."""
    return t.name if t is not None else "None"


def is_plain_int(value: Any) -> bool:
    """Whether ``value`` is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def stored_relation(value: str) -> Any:
    """The tree a stored relation string holds, or ``None`` when it holds none
    (the verifier reports that)."""
    try:
        tree = json.loads(value)
    except ValueError:
        return None
    return None if relation_error(tree) else tree


def bound_text(value: Any, leaf_text: Callable[[str], str] = str) -> str:
    """A stored bound as the author writes it: an int, or a relation's text
    (a string that holds no relation prints as stored)."""
    if isinstance(value, str):
        tree = stored_relation(value)
        return value if tree is None else relation_text(tree, leaf_text)
    # Neither an int nor a relation: the verifier reports it on its own.
    return str(value) if is_plain_int(value) else "?"


def param_fact_texts(param: Param) -> list[tuple[str, str]]:
    """``(word, text)`` for each fact ``param`` states, in ``FACT_WORDS`` order,
    with relation leaves printed as ``%NAME`` (``print_ir``'s form)."""
    return [
        (word, bound_text(param.attrs[word], lambda name: f"%{name}"))
        for word in FACT_WORDS
        if word in param.attrs
    ]


def checked_end(value: int | None) -> int | None:
    """An interval end, or ``None`` (unknown) when it leaves int64."""
    if value is None or value < INT64_MIN or value > INT64_MAX:
        return None
    return value


def _relation_interval(
    tree: Any, param_interval: Callable[[str], Interval]
) -> Interval:
    """``[lo, hi]`` a relation can take, ``None`` for an end it cannot bound.

    Interval arithmetic over the leaves' intervals. An end outside int64 is
    unknown, and a ``div`` is bounded only when its dividend is known
    non-negative and its divisor at least 1 -- the cases its evaluation accepts.
    """
    if not isinstance(tree, dict):
        return int(tree), int(tree)
    ((op, operands),) = tree.items()
    if op == "param":
        return param_interval(operands)
    (alo, ahi), (blo, bhi) = (
        _relation_interval(operands[0], param_interval),
        _relation_interval(operands[1], param_interval),
    )
    if op == "add":
        return (
            checked_end(None if alo is None or blo is None else alo + blo),
            checked_end(None if ahi is None or bhi is None else ahi + bhi),
        )
    if op == "sub":
        return (
            checked_end(None if alo is None or bhi is None else alo - bhi),
            checked_end(None if ahi is None or blo is None else ahi - blo),
        )
    if op == "mul":
        if None in (alo, ahi, blo, bhi):
            return None, None
        products = [alo * blo, alo * bhi, ahi * blo, ahi * bhi]
        return checked_end(min(products)), checked_end(max(products))
    # div: floor division, defined for a non-negative dividend and a divisor >= 1.
    if alo is None or alo < 0 or blo is None or blo < 1:
        return None, None
    return (
        checked_end(None if bhi is None else alo // bhi),
        checked_end(None if ahi is None else ahi // blo),
    )


def param_interval_resolver(
    params: Sequence[Param],
) -> Callable[[str], Interval]:
    """The interval of each named parameter in ``params``, from its stated
    facts; an unstated side takes its type's limit. A name that is not an
    integer parameter of ``params``, or a fact that does not resolve, is
    unknown (``None``)."""
    by_name = {p.name: p for p in params}
    memo: dict[str, Interval] = {}

    def interval(name: str) -> Interval:
        if name in memo:
            return memo[name]
        memo[name] = (None, None)  # a cycle (refused elsewhere) stays unknown
        param = by_name.get(name)
        if param is None or _type_name(param.type) not in FACT_INT_TYPES:
            return memo[name]
        tmin, tmax = int_type_limits(param.type.name)
        lo, hi = stated_param_interval(param, interval)
        memo[name] = (tmin if lo is UNSTATED else lo, tmax if hi is UNSTATED else hi)
        return memo[name]

    return interval


# The end of a stated interval that no word states (``stated_param_interval``).
UNSTATED: Any = object()


def stated_param_interval(
    param: Param, interval: Callable[[str], Interval]
) -> tuple[Any, Any]:
    """``param``'s stated ``(lower, upper)``, inclusive, resolved to numbers.

    Each end is ``UNSTATED`` when no word states it and ``None`` when the
    stated bound does not resolve to a number.
    """

    def bound(word: str) -> Interval:
        value = param.attrs[word]
        if isinstance(value, str):
            tree = stored_relation(value)
            return (None, None) if tree is None else _relation_interval(tree, interval)
        return (value, value) if is_plain_int(value) else (None, None)

    attrs = param.attrs
    if "eq" in attrs:
        return bound("eq")
    lower: Any = UNSTATED
    upper: Any = UNSTATED
    if "ge" in attrs:
        lower = bound("ge")[0]
    elif "gt" in attrs:
        lo = bound("gt")[0]
        lower = None if lo is None else checked_end(lo + 1)
    if "le" in attrs:
        upper = bound("le")[1]
    elif "lt" in attrs:
        hi = bound("lt")[1]
        upper = None if hi is None else checked_end(hi - 1)
    return lower, upper


def _quotient_errors(
    name: str,
    tree: Any,
    params: Sequence[Param],
    interval: Callable[[str], Interval],
) -> list[str]:
    """The ``quotient_of`` rule over every ``div`` in ``tree``: its divisor's
    stated lower bound is at least 1 and its dividend's at least 0.

    A compound dividend has no stated bound; evaluation refuses a negative one
    at launch, so both engines never disagree.
    """
    if not isinstance(tree, dict) or "param" in tree:
        return []
    ((op, (left, right)),) = tree.items()
    errors: list[str] = []
    if op == "div":
        text = relation_text(tree)
        by_name = {p.name: p for p in params}
        for operand, least, role in ((right, 1, "divisor"), (left, 0, "dividend")):
            if isinstance(operand, dict) and "param" in operand:
                errors.extend(
                    _stated_lower_errors(
                        name, text, by_name[operand["param"]], least, interval
                    )
                )
            elif not isinstance(operand, dict):
                if operand < least:
                    errors.append(
                        f"param {name}: {text} needs a {role} of at least {least}, "
                        f"got {operand}"
                    )
            elif role == "divisor":
                errors.append(
                    f"param {name}: {text} needs a divisor that is an int or one "
                    "kernel parameter, so that its lower bound is stated"
                )
    return (
        errors
        + _quotient_errors(name, left, params, interval)
        + _quotient_errors(name, right, params, interval)
    )


def _stated_lower_errors(
    name: str,
    text: str,
    operand: Param,
    least: int,
    interval: Callable[[str], Interval],
) -> list[str]:
    lower, _ = stated_param_interval(operand, interval)
    if lower is not UNSTATED and lower is not None and lower >= least:
        return []
    q = operand.name
    if lower is UNSTATED:
        declared = "is declared with no lower bound"
    else:
        # The word stating the lower end: eq, which states both ends, ahead of a lower
        # word; among lower words, the first in _BOUND_WORDS order (ge, gt), as in C++.
        # The order matters only for a declaration the rules already refuse.
        word = next(
            w
            for w in ("eq", *_BOUND_WORDS)
            if w in operand.attrs and w not in UPPER_WORDS
        )
        declared = f"is declared with {word}={bound_text(operand.attrs[word])}"
    fix = f'b.param("{q}", {operand.type.name.upper()}, ge={least}, ...)'
    return [f"param {name}: {text} needs {q} ge {least} -- {q} {declared}.\nFix: {fix}"]


def two_bounds_refusals(stated: Collection[str]) -> list[str]:
    """``"states two lower bounds (ge, gt) -- state one"`` for each end whose two
    words ``stated`` both names, the lower end first: a fact states each end once.
    The caller prefixes its subject. ``rocke_int_two_bounds_refusals`` in
    ``cpp/include/rocke/ir_internal.h`` gives the same texts."""
    return [
        f"states two {end} bounds ({', '.join(words)}) -- state one"
        for end, words in (("lower", LOWER_WORDS), ("upper", UPPER_WORDS))
        if all(word in stated for word in words)
    ]


def stored_int(name: str, word: str, value: Any, *, relations: bool = False) -> int:
    """``value`` as a plain int, or a ``TypeError`` naming what ``word`` takes."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        takes = (
            "an int, a parameter of this kernel, or sum_of / product_of / "
            "quotient_of / difference_of over those"
            if relations
            else "an int"
        )
        raise TypeError(f"param {name}: {word} is {takes}, got {value!r}")
    return int(value)


def declaration_errors(
    params: Sequence[Param], index: int, *, is_pointer: bool
) -> list[str]:
    """Every rule the declaration ``params[index]`` breaks, in order.

    ``params`` is the kernel's parameter list in declaration order; a relation
    may name only parameters declared before ``params[index]``. ``is_pointer``
    says whether ``params[index]`` has a pointer type, which only the IR's type
    classes can tell (``rocke.core.ir.param_fact_errors`` passes it). The one
    home of these rules in Python: ``IRBuilder.param`` raises the first message
    and the verifier reports all of them. ``cpp/core/ir/param_facts.cpp``
    mirrors them.
    """
    param = params[index]
    name, attrs = param.name, param.attrs
    type_name = _type_name(param.type)
    errors: list[str] = []
    for key in sorted(attrs):
        if key not in PARAM_ATTR_KEYS:
            errors.append(
                f"param '{name}' has unknown attr '{key}' -- a param attr is one of "
                + ", ".join(PARAM_ATTR_KEYS)
            )
    is_int = type_name in FACT_INT_TYPES
    for word in FACT_WORDS:
        if word in attrs and not is_int:
            errors.append(
                f"param {name}: {word} states an integer fact, and {name} is "
                f"{type_name} -- state it on an integer parameter"
            )
    for key in POINTER_ATTRS:
        if key in attrs and not is_pointer:
            errors.append(
                f"param {name}: {key} applies to a pointer parameter, and {name} is {type_name}"
            )
    if errors:
        return errors
    for word in _BOUND_WORDS:
        value = attrs.get(word)
        if word in attrs and not (is_plain_int(value) or isinstance(value, str)):
            errors.append(f"param {name}: {word} must be an integer or a relation")
    for key in ("multiple_of", "align", "dereferenceable"):
        if key in attrs and not is_plain_int(attrs[key]):
            errors.append(f"param {name}: {key} must be an integer")
    if errors:
        return errors
    errors.extend(f"param {name}: {text}" for text in two_bounds_refusals(attrs))
    if "eq" in attrs:
        others = [w for w in FACT_WORDS if w != "eq" and w in attrs]
        if others:
            errors.append(
                f"param {name}: eq excludes every other word -- "
                f"eq={bound_text(attrs['eq'])} already decides {', '.join(others)}"
            )
    if attrs.get("multiple_of", 1) < 1:
        errors.append(
            f"param {name}: multiple_of must be >= 1, got {attrs['multiple_of']}"
        )
    align = attrs.get("align", 1)
    if align < 1 or align & (align - 1):
        errors.append(f"param {name}: align must be a power of two >= 1, got {align}")
    if attrs.get("dereferenceable", 1) < 1:
        errors.append(
            f"param {name}: dereferenceable must be >= 1, got {attrs['dereferenceable']}"
        )
    if is_int:
        tmin, tmax = int_type_limits(type_name)
        for word in FACT_WORDS:
            value = attrs.get(word)
            if not is_plain_int(value):
                continue
            if value < tmin or value > tmax:
                errors.append(
                    f"param {name}: {word}={value} is outside the range of {type_name}"
                )
            elif abs(value) > EXACT_INT_MAX:
                errors.append(
                    f"param {name}: {word}={value} is outside +/-(2^53 - 1), which a "
                    "stored fact cannot hold exactly"
                )
    interval = param_interval_resolver(params[:index])
    position = {p.name: i for i, p in enumerate(params)}
    for word in _BOUND_WORDS:
        value = attrs.get(word)
        if not isinstance(value, str):
            continue
        try:
            tree = json.loads(value)
        except ValueError:
            tree, reason = None, "it is not JSON"
        else:
            reason = relation_error(tree)
        if reason is not None:
            errors.append(f"param {name}: {word} is not a relation: {reason}")
            continue
        name_errors = _relation_name_errors(name, index, tree, params, position)
        errors.extend(name_errors)
        if not name_errors:
            errors.extend(_quotient_errors(name, tree, params[:index], interval))
    return errors


def _relation_name_errors(
    name: str,
    index: int,
    tree: Any,
    params: Sequence[Param],
    position: dict[str, int],
) -> list[str]:
    """Each leaf of a relation names an integer parameter declared before ``name``."""
    errors: list[str] = []
    leaves = list(dict.fromkeys(relation_names(tree)))
    for q in leaves:
        if q == name:
            errors.append(
                f"{name}: the relation names {name} itself -- a parameter cannot bound itself"
            )
        elif q not in position:
            errors.append(
                f"{name}: the relation names {q}, which is not a parameter of this kernel"
            )
        elif position[q] > index:
            last = max((p for p in leaves if p in position), key=lambda p: position[p])
            errors.append(
                f"{name}: the relation names {q}, which is declared later -- state it on "
                f"the last-declared term, {last} (difference_of and quotient_of move a sum "
                "or a product to its side)"
            )
        elif _type_name(params[position[q]].type) not in FACT_INT_TYPES:
            errors.append(
                f"{name}: the relation names {q}, which is "
                f"{_type_name(params[position[q]].type)}, not an integer parameter"
            )
    return errors
