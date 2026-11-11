# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Immutable integer expressions and predicates for LoopIR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from .values import IRTypeError


class _ExpressionOperand:
    """Construction operators shared by expressions, parameters, and axes."""

    __hash__ = object.__hash__

    def _as_expr(self) -> "Expr":
        raise NotImplementedError

    def __add__(self, other) -> "Expr":
        return _add(self._as_expr(), as_expr(other))

    def __radd__(self, other) -> "Expr":
        return _add(as_expr(other), self._as_expr())

    def __sub__(self, other) -> "Expr":
        return _add(self._as_expr(), -as_expr(other))

    def __rsub__(self, other) -> "Expr":
        return _add(as_expr(other), -self._as_expr())

    def __neg__(self) -> "Expr":
        return _mul(-1, self._as_expr())

    def __mul__(self, coefficient: int) -> "Expr":
        if not isinstance(coefficient, int) or isinstance(coefficient, bool):
            return NotImplemented
        return _mul(coefficient, self._as_expr())

    def __rmul__(self, coefficient: int) -> "Expr":
        return self * coefficient

    def __floordiv__(self, divisor: int) -> "Expr":
        return _divide(self._as_expr(), divisor)

    def __mod__(self, modulus: int) -> "Expr":
        return _mod(self._as_expr(), modulus)

    def __eq__(self, other) -> "Compare":  # type: ignore[override]
        return Compare("==", self._as_expr(), as_expr(other))

    def __ne__(self, other) -> "Compare":  # type: ignore[override]
        return Compare("!=", self._as_expr(), as_expr(other))

    def __lt__(self, other) -> "Compare":
        return Compare("<", self._as_expr(), as_expr(other))

    def __le__(self, other) -> "Compare":
        return Compare("<=", self._as_expr(), as_expr(other))

    def __gt__(self, other) -> "Compare":
        return Compare(">", self._as_expr(), as_expr(other))

    def __ge__(self, other) -> "Compare":
        return Compare(">=", self._as_expr(), as_expr(other))


@dataclass(frozen=True, eq=False)
class Param(_ExpressionOperand):
    """A runtime integer scalar."""

    name: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise IRTypeError("a Param needs a non-empty string name")

    def _as_expr(self) -> "Expr":
        return Expr("param", (self,))

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False, init=False)
class Axis(_ExpressionOperand):
    """One integer coordinate with a required finite extent expression."""

    name: str
    extent: "Expr"

    def __init__(self, name: str, extent):
        if not isinstance(name, str) or not name:
            raise IRTypeError("an Axis needs a non-empty string name")
        normalized_extent = as_expr(extent)
        if normalized_extent.constant_value is not None \
                and normalized_extent.constant_value < 0:
            raise IRTypeError("an Axis extent must be non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "extent", normalized_extent)

    def _as_expr(self) -> "Expr":
        return Expr("axis", (self,))

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class Expr(_ExpressionOperand):
    """The complete LoopIR integer-expression tree."""

    op: str
    args: tuple

    def __post_init__(self):
        if self.op == "const":
            valid = len(self.args) == 1 and isinstance(self.args[0], int) \
                and not isinstance(self.args[0], bool)
        elif self.op == "param":
            valid = len(self.args) == 1 and isinstance(self.args[0], Param)
        elif self.op == "axis":
            valid = len(self.args) == 1 and isinstance(self.args[0], Axis)
        elif self.op == "add":
            valid = len(self.args) >= 2 and all(isinstance(arg, Expr) for arg in self.args)
        elif self.op == "mul":
            valid = len(self.args) == 2 and isinstance(self.args[0], int) \
                and not isinstance(self.args[0], bool) and isinstance(self.args[1], Expr)
        elif self.op in ("floordiv", "mod"):
            valid = len(self.args) == 2 and isinstance(self.args[0], Expr) \
                and isinstance(self.args[1], int) and self.args[1] > 0
        else:
            valid = False
        if not valid:
            raise IRTypeError(f"malformed {self.op!r} expression")

        object.__setattr__(self, "_free_axes", None)
        object.__setattr__(self, "_free_params", None)
        object.__setattr__(self, "_linear_terms", None)
        object.__setattr__(self, "_linear_constant", None)
        object.__setattr__(self, "_equivalence_key", None)
        object.__setattr__(self, "_affine", None)
        object.__setattr__(self, "_affine_computed", False)

    @staticmethod
    def constant(value: int) -> "Expr":
        if not isinstance(value, int) or isinstance(value, bool):
            raise IRTypeError("an expression constant must be an integer")
        return Expr("const", (value,))

    @staticmethod
    def parameter(value: Param) -> "Expr":
        if not isinstance(value, Param):
            raise IRTypeError("parameter expressions require a Param")
        return value._as_expr()

    @staticmethod
    def axis_value(value: Axis) -> "Expr":
        if not isinstance(value, Axis):
            raise IRTypeError("axis expressions require an Axis")
        return value._as_expr()

    def _as_expr(self) -> "Expr":
        return self

    def __deepcopy__(self, memo):
        result = Expr(self.op, deepcopy(self.args, memo))
        memo[id(self)] = result
        return result

    def __reduce__(self):
        return Expr, (self.op, self.args)

    def _equivalence(self):
        if self._equivalence_key is not None:
            return self._equivalence_key

        terms = {}
        constant = 0
        if self.op == "const":
            constant = self.args[0]
        elif self.op in ("param", "axis"):
            terms[(self.op, id(self.args[0]))] = 1
        elif self.op == "add":
            for argument in self.args:
                argument._equivalence()
                constant += argument._linear_constant
                for key, coefficient in argument._linear_terms:
                    terms[key] = terms.get(key, 0) + coefficient
        elif self.op == "mul":
            coefficient, argument = self.args
            argument._equivalence()
            constant = coefficient * argument._linear_constant
            terms = {
                key: coefficient * value
                for key, value in argument._linear_terms
            }
        else:
            argument, divisor = self.args
            terms[(self.op, argument._equivalence(), divisor)] = 1
        normalized = tuple(sorted(
            (
                (key, coefficient)
                for key, coefficient in terms.items()
                if coefficient
            ),
            key=repr,
        ))
        key = ("linear", normalized, constant)
        object.__setattr__(self, "_linear_terms", normalized)
        object.__setattr__(self, "_linear_constant", constant)
        object.__setattr__(self, "_equivalence_key", key)
        return key

    def _affine_form(self):
        if self._affine_computed:
            return self._affine

        if self.op == "const":
            affine = ((), self.args[0])
        elif self.op == "axis":
            affine = (((self.args[0], 1),), 0)
        elif self.op == "add":
            arguments = tuple(
                argument._affine_form()
                for argument in self.args
            )
            if any(argument is None for argument in arguments):
                affine = None
            else:
                coefficients = {}
                constant = 0
                for terms, add in arguments:
                    constant += add
                    for axis, coefficient in terms:
                        coefficients[axis] = \
                            coefficients.get(axis, 0) + coefficient
                affine = (
                    tuple(sorted(
                        coefficients.items(),
                        key=lambda item: id(item[0]),
                    )),
                    constant,
                )
        elif self.op == "mul":
            coefficient, argument = self.args
            nested = argument._affine_form()
            if nested is None:
                affine = None
            else:
                terms, constant = nested
                affine = (
                    tuple(
                        (axis, coefficient * value)
                        for axis, value in terms
                    ),
                    coefficient * constant,
                )
        else:
            affine = None

        object.__setattr__(self, "_affine", affine)
        object.__setattr__(self, "_affine_computed", True)
        return affine

    @property
    def constant_value(self) -> int | None:
        return self.args[0] if self.op == "const" else None

    def free_axes(self) -> frozenset[Axis]:
        if self._free_axes is None:
            if self.op == "axis":
                value = frozenset((self.args[0],))
            elif self.op in ("const", "param"):
                value = frozenset()
            else:
                value = frozenset(
                    axis
                    for argument in self.args
                    if isinstance(argument, Expr)
                    for axis in argument.free_axes()
                )
            object.__setattr__(self, "_free_axes", value)
        return self._free_axes

    def free_params(self) -> frozenset[Param]:
        if self._free_params is None:
            if self.op == "param":
                value = frozenset((self.args[0],))
            elif self.op in ("const", "axis"):
                value = frozenset()
            else:
                value = frozenset(
                    param
                    for argument in self.args
                    if isinstance(argument, Expr)
                    for param in argument.free_params()
                )
            object.__setattr__(self, "_free_params", value)
        return self._free_params

    def evaluate(self, bindings: Mapping[object, int]) -> int:
        if self.op == "const":
            return self.args[0]
        if self.op in ("param", "axis"):
            value = self.args[0]
            if value in bindings:
                return int(bindings[value])
            if value.name in bindings:
                return int(bindings[value.name])
            raise KeyError(value.name)
        if self.op == "add":
            return sum(arg.evaluate(bindings) for arg in self.args)
        if self.op == "mul":
            return self.args[0] * self.args[1].evaluate(bindings)
        if self.op == "floordiv":
            return self.args[0].evaluate(bindings) // self.args[1]
        if self.op == "mod":
            return self.args[0].evaluate(bindings) % self.args[1]
        raise AssertionError(self.op)

    def substitute(self, replacements: Mapping[object, object]) -> "Expr":
        if self.op in ("param", "axis"):
            value = self.args[0]
            if value in replacements:
                return as_expr(replacements[value])
            if value.name in replacements:
                return as_expr(replacements[value.name])
            return self
        if self.op == "const":
            return self
        if self.op == "add":
            result = Expr.constant(0)
            for arg in self.args:
                result = _add(result, arg.substitute(replacements))
            return result
        if self.op == "mul":
            return _mul(self.args[0], self.args[1].substitute(replacements))
        if self.op == "floordiv":
            return _divide(self.args[0].substitute(replacements), self.args[1])
        if self.op == "mod":
            return _mod(self.args[0].substitute(replacements), self.args[1])
        raise AssertionError(self.op)

    def render(self) -> str:
        if self.op == "const":
            return str(self.args[0])
        if self.op in ("param", "axis"):
            return self.args[0].name
        if self.op == "add":
            text = self.args[0].render()
            for arg in self.args[1:]:
                if arg.op == "const" and arg.args[0] < 0:
                    text += f"-{abs(arg.args[0])}"
                elif arg.op == "mul" and arg.args[0] < 0:
                    coefficient, value = arg.args
                    rendered = value.render()
                    if value.op == "add":
                        rendered = f"({rendered})"
                    factor = abs(coefficient)
                    text += "-" + (
                        rendered if factor == 1
                        else f"{factor}*{rendered}"
                    )
                else:
                    text += f"+{arg.render()}"
            return text
        if self.op == "mul":
            coefficient, value = self.args
            rendered = value.render()
            if value.op == "add":
                rendered = f"({rendered})"
            return f"{coefficient}*{rendered}"
        if self.op == "floordiv":
            return f"({self.args[0].render()})//{self.args[1]}"
        if self.op == "mod":
            return f"({self.args[0].render()})%{self.args[1]}"
        raise AssertionError(self.op)

    def __str__(self) -> str:
        return self.render()


def _add(left: Expr, right: Expr) -> Expr:
    values = []
    constant = 0
    for item in (left, right):
        terms = item.args if item.op == "add" else (item,)
        for term in terms:
            if term.op == "const":
                constant += term.args[0]
            else:
                values.append(term)
    if constant or not values:
        values.append(Expr.constant(constant))
    if len(values) == 1:
        return values[0]
    return Expr("add", tuple(values))


def _mul(coefficient: int, value: Expr) -> Expr:
    if coefficient == 0:
        return Expr.constant(0)
    if coefficient == 1:
        return value
    if value.op == "const":
        return Expr.constant(coefficient * value.args[0])
    if value.op == "mul":
        return _mul(coefficient * value.args[0], value.args[1])
    return Expr("mul", (coefficient, value))


def _positive_divisor(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise IRTypeError(f"{name} must be a positive integer")
    return value


def _divide(value: Expr, divisor: int) -> Expr:
    divisor = _positive_divisor(divisor, "expression divisor")
    if value.op == "const":
        return Expr.constant(value.args[0] // divisor)
    if divisor == 1:
        return value
    return Expr("floordiv", (value, divisor))


def _mod(value: Expr, modulus: int) -> Expr:
    modulus = _positive_divisor(modulus, "expression modulus")
    if value.op == "const":
        return Expr.constant(value.args[0] % modulus)
    if modulus == 1:
        return Expr.constant(0)
    return Expr("mod", (value, modulus))


def as_expr(value) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, (Param, Axis)):
        return value._as_expr()
    if isinstance(value, int) and not isinstance(value, bool):
        return Expr.constant(value)
    raise IRTypeError(f"cannot convert {type(value).__name__} to Expr")


class Pred:
    """Base class for the complete LoopIR boolean grammar."""

    __slots__ = ()

    def __bool__(self):
        raise IRTypeError("a LoopIR predicate cannot be converted to a Python bool")

    def __and__(self, other) -> "And":
        return And(self, as_pred(other))

    def __or__(self, other) -> "Or":
        return Or(self, as_pred(other))

    def __invert__(self) -> "Not":
        return Not(self)

    def free_axes(self) -> frozenset[Axis]:
        raise NotImplementedError

    def free_params(self) -> frozenset[Param]:
        raise NotImplementedError

    def evaluate(self, bindings: Mapping[object, int]) -> bool | None:
        raise NotImplementedError

    def substitute(self, replacements: Mapping[object, object]) -> "Pred":
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, eq=False, init=False)
class Compare(Pred):
    op: str
    lhs: Expr
    rhs: Expr

    def __init__(self, op: str, lhs, rhs):
        if op not in ("==", "!=", "<", "<=", ">", ">="):
            raise IRTypeError(f"unsupported comparison operator {op!r}")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "lhs", as_expr(lhs))
        object.__setattr__(self, "rhs", as_expr(rhs))
        object.__setattr__(
            self,
            "_free_axes",
            self.lhs.free_axes() | self.rhs.free_axes(),
        )
        object.__setattr__(
            self,
            "_free_params",
            self.lhs.free_params() | self.rhs.free_params(),
        )

    def __deepcopy__(self, memo):
        result = Compare(
            self.op,
            deepcopy(self.lhs, memo),
            deepcopy(self.rhs, memo),
        )
        memo[id(self)] = result
        return result

    def __reduce__(self):
        return Compare, (self.op, self.lhs, self.rhs)

    def free_axes(self) -> frozenset[Axis]:
        return self._free_axes

    def free_params(self) -> frozenset[Param]:
        return self._free_params

    def evaluate(self, bindings: Mapping[object, int]) -> bool | None:
        try:
            left = self.lhs.evaluate(bindings)
            right = self.rhs.evaluate(bindings)
        except KeyError:
            return None
        return {
            "==": left == right,
            "!=": left != right,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }[self.op]

    def substitute(self, replacements: Mapping[object, object]) -> "Compare":
        return Compare(
            self.op,
            self.lhs.substitute(replacements),
            self.rhs.substitute(replacements),
        )

    def render(self) -> str:
        return f"{self.lhs.render()} {self.op} {self.rhs.render()}"


@dataclass(frozen=True, eq=False, init=False)
class And(Pred):
    operands: tuple[Pred, ...]

    def __init__(self, *operands):
        values = tuple(as_pred(item) for item in operands)
        if len(values) < 2:
            raise IRTypeError("And requires at least two predicates")
        object.__setattr__(self, "operands", values)
        object.__setattr__(
            self,
            "_free_axes",
            frozenset(
                axis for item in values for axis in item.free_axes()
            ),
        )
        object.__setattr__(
            self,
            "_free_params",
            frozenset(
                param for item in values for param in item.free_params()
            ),
        )

    def __deepcopy__(self, memo):
        result = And(*(deepcopy(item, memo) for item in self.operands))
        memo[id(self)] = result
        return result

    def __reduce__(self):
        return And, self.operands

    def free_axes(self) -> frozenset[Axis]:
        return self._free_axes

    def free_params(self) -> frozenset[Param]:
        return self._free_params

    def evaluate(self, bindings: Mapping[object, int]) -> bool | None:
        values = tuple(item.evaluate(bindings) for item in self.operands)
        if False in values:
            return False
        return None if None in values else True

    def substitute(self, replacements: Mapping[object, object]) -> "And":
        return And(*(item.substitute(replacements) for item in self.operands))

    def render(self) -> str:
        return " and ".join(f"({item.render()})" for item in self.operands)


@dataclass(frozen=True, eq=False, init=False)
class Or(Pred):
    operands: tuple[Pred, ...]

    def __init__(self, *operands):
        values = tuple(as_pred(item) for item in operands)
        if len(values) < 2:
            raise IRTypeError("Or requires at least two predicates")
        object.__setattr__(self, "operands", values)
        object.__setattr__(
            self,
            "_free_axes",
            frozenset(
                axis for item in values for axis in item.free_axes()
            ),
        )
        object.__setattr__(
            self,
            "_free_params",
            frozenset(
                param for item in values for param in item.free_params()
            ),
        )

    def __deepcopy__(self, memo):
        result = Or(*(deepcopy(item, memo) for item in self.operands))
        memo[id(self)] = result
        return result

    def __reduce__(self):
        return Or, self.operands

    def free_axes(self) -> frozenset[Axis]:
        return self._free_axes

    def free_params(self) -> frozenset[Param]:
        return self._free_params

    def evaluate(self, bindings: Mapping[object, int]) -> bool | None:
        values = tuple(item.evaluate(bindings) for item in self.operands)
        if True in values:
            return True
        return None if None in values else False

    def substitute(self, replacements: Mapping[object, object]) -> "Or":
        return Or(*(item.substitute(replacements) for item in self.operands))

    def render(self) -> str:
        return " or ".join(f"({item.render()})" for item in self.operands)


@dataclass(frozen=True, eq=False, init=False)
class Not(Pred):
    operand: Pred

    def __init__(self, operand):
        object.__setattr__(self, "operand", as_pred(operand))
        object.__setattr__(self, "_free_axes", self.operand.free_axes())
        object.__setattr__(self, "_free_params", self.operand.free_params())

    def __deepcopy__(self, memo):
        result = Not(deepcopy(self.operand, memo))
        memo[id(self)] = result
        return result

    def __reduce__(self):
        return Not, (self.operand,)

    def free_axes(self) -> frozenset[Axis]:
        return self._free_axes

    def free_params(self) -> frozenset[Param]:
        return self._free_params

    def evaluate(self, bindings: Mapping[object, int]) -> bool | None:
        value = self.operand.evaluate(bindings)
        return None if value is None else not value

    def substitute(self, replacements: Mapping[object, object]) -> "Not":
        return Not(self.operand.substitute(replacements))

    def render(self) -> str:
        return f"not ({self.operand.render()})"


def as_pred(value) -> Pred:
    if isinstance(value, Pred):
        return value
    raise IRTypeError(f"cannot convert {type(value).__name__} to Pred")


# Explicit predicates used where an unrestricted or empty domain is part of the
# IR contract.  Semantic absence is never represented by ``None``.
TRUE = Compare("==", 0, 0)
FALSE = Compare("!=", 0, 0)


def expr_equivalent(left, right) -> bool:
    """Return structural expression equality without constructing a predicate."""

    return _expr_key(as_expr(left)) == _expr_key(as_expr(right))


def _expr_key(value: Expr):
    return value._equivalence()
