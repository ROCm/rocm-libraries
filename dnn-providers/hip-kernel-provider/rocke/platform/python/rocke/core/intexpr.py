# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integer expressions: the one tree form, its evaluator and its constructors.

An *intexpr* is a small JSON tree of integer arithmetic. Recipes use it for
sizes, loop bounds and admission guards; kernel parameters will use it for
bounds stated in terms of other parameters. It is one form with two
evaluators -- this module and ``rocke_intexpr_eval`` in
``cpp/support/intexpr.cpp`` -- and both must give the same value or the same
error text for every tree, because a caller may ask in one engine and the
other may enforce. ``tests/core/test_intexpr.py`` pins both against one
table of cases.

The grammar::

    <intexpr> := integer
               | {"spec": NAME} | {"var": NAME} | {"param": NAME}
               | {"spec_str_eq": [NAME, literal]}
               | {"<OP>": [e, e]}  for OP in add sub mul div mod eq ne lt le gt ge
               | {"<FN>": e}       for FN in magic_multiplier magic_shift

Evaluation computes in int64, the range the C++ mirror works in, and refuses
rather than guesses wherever the two languages could disagree:

- a number that is not an exact integer within +/-(2^53 - 1) (the JSON tree
  stores numbers as doubles, so a larger one may already have been rounded);
- a ``div`` or ``mod`` by zero;
- a ``div`` or ``mod`` with a negative operand (Python floors, C truncates);
- any ``add``, ``sub`` or ``mul`` result outside int64 (undefined in C).

``sum_of``, ``product_of``, ``quotient_of`` and ``difference_of`` build a
bound from integers and leaves (for example kernel parameters). They are free
functions, not builder methods, so nobody reads them as GPU instructions: they
describe arithmetic the host evaluates, and they emit nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "IntExpr",
    "IntExprError",
    "difference_of",
    "eval_intexpr",
    "magic_division_constants",
    "product_of",
    "quotient_of",
    "relation_error",
    "relation_json",
    "relation_names",
    "relation_text",
    "renamed_relation",
    "sum_of",
]

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
# The largest integer a double holds exactly; both evaluators refuse a number
# leaf beyond it rather than trust its spelling.
_EXACT_DOUBLE_INT = (1 << 53) - 1


class IntExprError(ValueError):
    """An intexpr that cannot be evaluated: malformed, unbound or refused."""


def magic_division_constants(divisor: int) -> tuple[int, int]:
    """`(multiplier_i32, shift)` for a strength-reduced unsigned `n // divisor`.

    The canonical mirror of `helpers/magic_division.py::calculate_magic_numbers` plus
    `do_magic_division`'s two's-complement wrap (itself a port of CK Tile's
    `magic_division32_bit_range`). Kernels bake these two integers in as
    `const_i32` operands of `(umul_hi(n, M) + n) >> s`, so a recipe that wants to
    stay parametric in the divisor has to regenerate them rather than fit them --
    the shift is logarithmic and the multiplier depends on the divisor's odd part.

    `cpp/support/intexpr.cpp` implements the same two lines; the portable-IR
    roller imports this for recognition. A test pins all three against the DSL
    helper.
    """
    # The upper bound keeps this inside the int64 range the C++ mirror computes
    # in, so both mirrors agree everywhere they are defined.
    if divisor < 1 or divisor > 0x7FFFFFFF:
        raise IntExprError(f"magic division needs 1 <= divisor < 2^31, got {divisor}")
    shift = 0
    while (1 << shift) < divisor:
        shift += 1
    mult = (((1 << shift) - divisor) << 32) // divisor + 1
    return (mult - (1 << 32) if mult >= (1 << 31) else mult), shift


_UNARY: dict[str, Callable[[int], int]] = {
    "magic_multiplier": lambda d: magic_division_constants(d)[0],
    "magic_shift": lambda d: magic_division_constants(d)[1],
}

# Key order is the evaluation order of both mirrors: a node is read as the first
# of these keys it carries.
_BINARY = ("add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "gt", "ge")

# Leaf kinds that name an integer, with the noun each one's "unknown" error uses.
_LEAVES = (("spec", "spec int"), ("var", "loop var"), ("param", "param"))


def _checked_int64(op: str, a: int, b: int, result: int) -> int:
    if result < INT64_MIN or result > INT64_MAX:
        raise IntExprError(f"intexpr {op} overflows int64 ({a}, {b})")
    return result


def _apply_binary(op: str, a: int, b: int) -> int:
    if op in ("div", "mod"):
        if b == 0:
            raise IntExprError(f"intexpr {op} by zero")
        if a < 0 or b < 0:
            # Python floors and C truncates; for non-negative operands they agree.
            raise IntExprError(
                f"intexpr {op} needs non-negative operands, got ({a}, {b})"
            )
        return a // b if op == "div" else a % b
    if op == "add":
        return _checked_int64(op, a, b, a + b)
    if op == "sub":
        return _checked_int64(op, a, b, a - b)
    if op == "mul":
        return _checked_int64(op, a, b, a * b)
    return int(
        {
            "eq": a == b,
            "ne": a != b,
            "lt": a < b,
            "le": a <= b,
            "gt": a > b,
            "ge": a >= b,
        }[op]
    )


def _number(node: float) -> int:
    if isinstance(node, float):
        if not node.is_integer() or abs(node) > _EXACT_DOUBLE_INT:
            raise IntExprError("intexpr number is not an integer")
        return int(node)
    if abs(node) > _EXACT_DOUBLE_INT:
        raise IntExprError("intexpr number is not an integer")
    return node


def eval_intexpr(
    node: Any,
    ivars: Mapping[str, int],
    spec_int: Mapping[str, int],
    spec_str: Mapping[str, str],
    *,
    params: Mapping[str, int] | None = None,
) -> int:
    """Evaluate one intexpr tree to an int, or raise `IntExprError`.

    `ivars` binds `{"var": ...}` leaves (a recipe's compile-time loop indices),
    `spec_int` / `spec_str` bind `{"spec": ...}` / `{"spec_str_eq": ...}`, and
    `params` binds `{"param": ...}` (kernel parameter values). A leaf whose
    name is not bound is an error, never a default.
    """
    # A JSON true/false is not a number to the C++ DOM either: refused below.
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return _number(node)
    if isinstance(node, dict):
        bindings = {"spec": spec_int, "var": ivars, "param": params or {}}
        for key, noun in _LEAVES:
            if key in node:
                name = node[key]
                if not isinstance(name, str) or name not in bindings[key]:
                    shown = name if isinstance(name, str) else "?"
                    raise IntExprError(f"unknown {noun} '{shown}'")
                return bindings[key][name]
        sse = node.get("spec_str_eq")
        if isinstance(sse, (list, tuple)) and len(sse) == 2:
            value = spec_str.get(sse[0]) if isinstance(sse[0], str) else None
            return int(
                value is not None and isinstance(sse[1], str) and value == sse[1]
            )
        # Unary functions take the operand directly, not a 2-element array.
        for key, fn in _UNARY.items():
            if key in node:
                return fn(
                    eval_intexpr(node[key], ivars, spec_int, spec_str, params=params)
                )
        for op in _BINARY:
            if op in node:
                pair = node[op]
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise IntExprError(f"bad intexpr '{op}'")
                a = eval_intexpr(pair[0], ivars, spec_int, spec_str, params=params)
                b = eval_intexpr(pair[1], ivars, spec_int, spec_str, params=params)
                return _apply_binary(op, a, b)
    raise IntExprError("bad intexpr")


# --------------------------------------------------------------------------
# constructors
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IntExpr:
    """A host-evaluated bound built by `sum_of` / `product_of` / `quotient_of` /
    `difference_of`.

    `op` is the intexpr operator (`add`, `mul`, `div` or `sub`) and `terms` its
    operands: ints, nested `IntExpr`s, or leaves the caller resolves to a name
    (a kernel parameter's Value). A sum or product keeps all its terms, so its
    text reads the way the author wrote it; its tree nests them left to right
    (`sum_of(a, b, c)` is `{"add": [{"add": [a, b]}, c]}`).
    """

    op: str
    terms: tuple[Any, ...]

    def tree(self, leaf_name: Callable[[Any], str]) -> dict[str, Any]:
        """The intexpr tree, with each leaf written as `{"param": leaf_name(leaf)}`."""

        def node(term: Any) -> Any:
            if isinstance(term, IntExpr):
                return term.tree(leaf_name)
            if isinstance(term, int):
                return term
            return {"param": leaf_name(term)}

        out = node(self.terms[0])
        for term in self.terms[1:]:
            out = {self.op: [out, node(term)]}
        return out

    def text(self, leaf_text: Callable[[Any], str]) -> str:
        """The bound as the author wrote it, e.g. `product_of(M, ldc)`."""

        def node(term: Any) -> str:
            if isinstance(term, IntExpr):
                return term.text(leaf_text)
            if isinstance(term, int):
                return str(term)
            return leaf_text(term)

        return f"{_FUNCTION_OF[self.op]}({', '.join(node(t) for t in self.terms)})"


_FUNCTION_OF = {
    "add": "sum_of",
    "mul": "product_of",
    "div": "quotient_of",
    "sub": "difference_of",
}


def _term(function: str, term: Any) -> Any:
    """Refuse a term no evaluator can read exactly; pass the rest through.

    Leaves are checked where they are resolved to names (the kernel parameter
    declaration), because only that caller knows which leaves it accepts.
    """
    if isinstance(term, bool) or term is None or isinstance(term, (float, str)):
        raise TypeError(
            f"{function}: a term is an int, a kernel parameter or a nested "
            f"sum_of / product_of / quotient_of / difference_of, got {term!r}"
        )
    if isinstance(term, int) and abs(term) > _EXACT_DOUBLE_INT:
        raise ValueError(
            f"{function}: {term} is outside +/-(2^53 - 1), which the stored "
            "expression cannot hold exactly"
        )
    return term


def sum_of(*terms: Any) -> IntExpr:
    """`terms[0] + terms[1] + ...`, evaluated on the host (at least two terms)."""
    if len(terms) < 2:
        raise TypeError(f"sum_of needs at least two terms, got {len(terms)}")
    return IntExpr("add", tuple(_term("sum_of", t) for t in terms))


def product_of(*terms: Any) -> IntExpr:
    """`terms[0] * terms[1] * ...`, evaluated on the host (at least two terms)."""
    if len(terms) < 2:
        raise TypeError(f"product_of needs at least two terms, got {len(terms)}")
    return IntExpr("mul", tuple(_term("product_of", t) for t in terms))


def quotient_of(dividend: Any, divisor: Any) -> IntExpr:
    """`dividend // divisor`, evaluated on the host.

    Floor division. Both operands must evaluate non-negative and the divisor
    non-zero; evaluation refuses otherwise, so the two engines never disagree.
    """
    return IntExpr(
        "div", (_term("quotient_of", dividend), _term("quotient_of", divisor))
    )


def difference_of(minuend: Any, subtrahend: Any) -> IntExpr:
    """`minuend - subtrahend`, evaluated on the host."""
    return IntExpr(
        "sub", (_term("difference_of", minuend), _term("difference_of", subtrahend))
    )


# --------------------------------------------------------------------------
# relations: the intexpr subset a kernel parameter's bound is stored in
# --------------------------------------------------------------------------
# A relation is an int, a ``{"param": NAME}`` leaf, or an ``add`` / ``sub`` /
# ``mul`` / ``div`` node over two relations: exactly what ``sum_of`` /
# ``product_of`` / ``difference_of`` / ``quotient_of`` build. It is stored as
# canonical JSON (sorted keys, no spaces) in one string attr, so it round-trips
# through every serializer with no new attr kind. ``cpp/core/ir/param_facts.cpp``
# mirrors these readers, with the same texts.

_RELATION_OPS = ("add", "sub", "mul", "div")


def relation_json(tree: Any) -> str:
    """The canonical JSON text of a relation tree: sorted keys, no spaces."""
    return json.dumps(tree, sort_keys=True, separators=(",", ":"))


def relation_error(tree: Any) -> str | None:
    """Why ``tree`` is not a relation, or ``None`` when it is one."""
    if isinstance(tree, bool):
        return "a relation is an int, a param leaf, or add / sub / mul / div of two relations"
    if isinstance(tree, (int, float)):
        if isinstance(tree, float) and not tree.is_integer():
            return "intexpr number is not an integer"
        if abs(tree) > _EXACT_DOUBLE_INT:
            return "intexpr number is not an integer"
        return None
    if isinstance(tree, dict) and len(tree) == 1:
        ((key, value),) = tree.items()
        if key == "param":
            return None if isinstance(value, str) else "a param leaf names a parameter"
        if key in _RELATION_OPS:
            if not isinstance(value, list) or len(value) != 2:
                return f"bad intexpr '{key}'"
            return relation_error(value[0]) or relation_error(value[1])
    return (
        "a relation is an int, a param leaf, or add / sub / mul / div of two relations"
    )


def relation_names(tree: Any) -> list[str]:
    """The parameter names a relation's leaves name, left to right."""
    if isinstance(tree, dict):
        ((key, value),) = tree.items()
        if key == "param":
            return [value]
        return relation_names(value[0]) + relation_names(value[1])
    return []


def renamed_relation(tree: Any, rename: Callable[[str], str]) -> Any:
    """``tree`` with every ``param`` leaf renamed through ``rename``."""
    if isinstance(tree, dict):
        ((key, value),) = tree.items()
        if key == "param":
            return {"param": rename(value)}
        return {
            key: [
                renamed_relation(value[0], rename),
                renamed_relation(value[1], rename),
            ]
        }
    return tree


def relation_text(tree: Any, leaf_text: Callable[[str], str] = str) -> str:
    """A relation as the author writes it, e.g. ``product_of(M, ldc)``.

    A left-nested chain of one ``add`` or ``mul`` prints as one call, so
    ``sum_of(a, b, c)`` prints as written. ``leaf_text`` renders a parameter
    name (``print_ir`` passes one that adds ``%``).
    """
    if isinstance(tree, dict):
        ((key, value),) = tree.items()
        if key == "param":
            return leaf_text(value)
        terms = [value[1]]
        head = value[0]
        while key in ("add", "mul") and isinstance(head, dict) and key in head:
            terms.append(head[key][1])
            head = head[key][0]
        terms.append(head)
        inner = ", ".join(relation_text(t, leaf_text) for t in reversed(terms))
        return f"{_FUNCTION_OF[key]}({inner})"
    return str(int(tree))
