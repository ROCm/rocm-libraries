# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Relations between kernel parameters, read for the tiling layer's build-time proofs.

A kernel parameter may state a relation to earlier parameters on its declaration
(``ldc = b.param("ldc", I32, ge=N)``, ``b.param("sb", I32, ge=product_of(M, ldc))``).
``KernelLauncher.from_artifact`` checks it before each launch, so the build may rely on it. A
relation is stated on whichever of its parameters is declared last, so one fact can take several
shapes: ``sb ge product_of(M, ldc)`` on ``sb``, or ``ldc le quotient_of(sb, M)`` on ``ldc`` when
``sb`` and ``M`` come first. This module reads every relation once and puts it in one shape,
``big >= small``, both sides integer polynomials over parameter names:

- ``x ge t`` is ``x >= t`` and ``x gt t`` is ``x >= t + 1``;
- ``x le t`` is ``t >= x`` and ``x lt t`` is ``t >= x + 1``; ``x eq t`` gives both directions;
- ``quotient_of(a, y) >= s`` becomes ``a >= s * y`` (``a // y >= s`` with ``y >= 1`` gives
  ``a >= y * (a // y) >= y * s``; the declaration rules make every divisor at least 1);
- ``difference_of(a, y) >= s`` becomes ``a >= s + y``.

So ``ldc le quotient_of(sb, M)`` reads ``sb >= ldc * M``, the same fact as ``sb ge product_of(M,
ldc)``. A relation that has a quotient left on its smaller side (``x ge quotient_of(a, y)``) is left
out: it gives no exact product. Only relations are read here; a fact with an int bound is a number,
which :func:`rocke.core.facts.stated_facts_of` already gives.

:func:`proving_relation` answers the one question the proofs ask: which stated relation shows
``big >= small`` -- its bigger side is exactly ``big`` (or, with ``big=None``, any constant), and its
smaller side minus ``small`` is a sum of terms that cannot be negative. Each relation keeps the text
the author stated it with, so a report names exactly the facts a proof used.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rocke.core.facts import stated_facts_of
from rocke.core.intexpr import relation_error, relation_text
from rocke.core.ir import Param, Value

__all__ = [
    "Poly",
    "StatedRelation",
    "poly_of",
    "poly_product",
    "proving_relation",
    "relations_among",
]

# An integer polynomial: each term a sorted tuple of parameter names (``()`` for the constant) to
# its coefficient. Terms with a zero coefficient are dropped.
Poly = dict[tuple[str, ...], int]

_COMPARISON_WORDS = ("ge", "gt", "le", "lt", "eq")


@dataclass(frozen=True)
class StatedRelation:
    """``big >= small``, read from one stated relation; ``text`` is the fact as the author stated
    it (``ldc ge N``, ``ldc le quotient_of(sb, M)``)."""

    big: tuple[tuple[tuple[str, ...], int], ...]
    small: tuple[tuple[tuple[str, ...], int], ...]
    text: str

    def big_poly(self) -> Poly:
        return dict(self.big)

    def small_poly(self) -> Poly:
        return dict(self.small)


def poly_of(term: int | Value) -> Poly | None:
    """``term`` as a polynomial: an int is a constant, a kernel parameter's Value is its name;
    any other Value (computed in the kernel) is ``None``, since no relation can name it.
    """
    if isinstance(term, Value):
        if term.param is None:
            return None
        return {(term.param.name,): 1}
    return _clean({(): int(term)})


def _clean(p: Poly) -> Poly:
    return {k: v for k, v in p.items() if v != 0}


def _add(a: Poly, b: Poly, sign: int = 1) -> Poly:
    out = dict(a)
    for key, coeff in b.items():
        out[key] = out.get(key, 0) + sign * coeff
    return _clean(out)


def poly_product(a: Poly, b: Poly) -> Poly:
    """``a * b``."""
    out: Poly = {}
    for ka, ca in a.items():
        for kb, cb in b.items():
            key = tuple(sorted(ka + kb))
            out[key] = out.get(key, 0) + ca * cb
    return _clean(out)


def _tree_poly(tree: object) -> Poly | None:
    """An intexpr tree with no quotient as a polynomial, else ``None``."""
    if not isinstance(tree, dict):
        return _clean({(): int(tree)})
    ((op, operands),) = tree.items()
    if op == "param":
        return {(operands,): 1}
    if op == "div":
        return None
    left, right = (_tree_poly(t) for t in operands)
    if left is None or right is None:
        return None
    if op == "add":
        return _add(left, right)
    if op == "sub":
        return _add(left, right, sign=-1)
    if op == "mul":
        return poly_product(left, right)
    return None


def _at_least(big: object, small: Poly) -> tuple[Poly, Poly] | None:
    """``big >= small`` with the quotients and differences of ``big`` moved to the smaller side,
    as polynomials, or ``None`` when that leaves a quotient behind."""
    while isinstance(big, dict) and next(iter(big)) in ("div", "sub"):
        ((op, (left, right)),) = big.items()
        right_poly = _tree_poly(right)
        if right_poly is None:
            return None
        small = (
            poly_product(small, right_poly) if op == "div" else _add(small, right_poly)
        )
        big = left
    big_poly = _tree_poly(big)
    return None if big_poly is None else (big_poly, small)


def _frozen(p: Poly) -> tuple[tuple[tuple[str, ...], int], ...]:
    return tuple(sorted(p.items()))


def _stated(param: Param) -> list[StatedRelation]:
    """The relations ``param`` states, each in the ``big >= small`` shape."""
    found = []
    name = {(param.name,): 1}
    for word in _COMPARISON_WORDS:
        stored = param.attrs.get(word)
        if not isinstance(stored, str):
            continue  # an int bound is a number, not a relation
        try:
            tree = json.loads(stored)
        except ValueError:
            continue
        if relation_error(tree) is not None:
            continue
        text = f"{param.name} {word} {relation_text(tree)}"
        shapes: list[tuple[Poly, Poly] | None] = []
        if word in ("ge", "gt", "eq"):
            bound = _tree_poly(tree)
            if bound is not None:
                plus = {(): 1} if word == "gt" else {}
                shapes.append((name, _add(bound, plus)))
        if word in ("le", "lt", "eq"):
            plus = {(): 1} if word == "lt" else {}
            shapes.append(_at_least(tree, _add(name, plus)))
        found.extend(
            StatedRelation(_frozen(shape[0]), _frozen(shape[1]), text)
            for shape in shapes
            if shape is not None
        )
    return found


def relations_among(values: Iterable[int | Value]) -> tuple[StatedRelation, ...]:
    """Every relation stated on the kernel parameters among ``values`` and the parameters declared
    before them -- the parameters a relation over ``values`` can be stated on."""
    params: dict[str, Param] = {}
    for value in values:
        if isinstance(value, Value) and value.param is not None:
            for param in (*value.param.declared_before, value.param):
                params.setdefault(param.name, param)
    return tuple(r for param in params.values() for r in _stated(param))


def _never_negative(
    term: tuple[str, ...], coeff: int, lower: Callable[[str], int | None]
) -> bool:
    """Whether ``coeff * prod(term)`` cannot be negative: a non-negative coefficient over
    parameters each stated at least 0."""
    if coeff < 0:
        return False
    return all((least := lower(name)) is not None and least >= 0 for name in term)


def proving_relation(
    relations: Iterable[StatedRelation],
    small: Poly,
    *,
    big: Poly | None,
    values: Iterable[int | Value],
) -> StatedRelation | None:
    """The first stated relation that shows ``big >= small``: its bigger side is ``big`` (with
    ``big=None``, any constant -- read the bound from the relation), and its smaller side exceeds
    ``small`` only by terms that cannot be negative. ``values`` are the Values whose parameters'
    stated lower bounds decide that; a parameter not among them proves nothing."""
    lowers: dict[str, int | None] = {}
    for value in values:
        if isinstance(value, Value) and value.param is not None:
            for param in (*value.param.declared_before, value.param):
                if param.name not in lowers:
                    probe = Value(f"%{param.name}", param.type, param=param)
                    lowers[param.name] = stated_facts_of(probe).lower
    for relation in relations:
        stated_big = relation.big_poly()
        if big is None:
            if any(term != () for term in stated_big):
                continue
        elif stated_big != big:
            continue
        rest = _add(relation.small_poly(), small, sign=-1)
        if all(_never_negative(t, c, lowers.get) for t, c in rest.items()):
            return relation
    return None
