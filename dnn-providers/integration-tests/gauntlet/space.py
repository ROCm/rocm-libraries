#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Space declarations for the integration gauntlet (RFC 0023, section 5).

A space file ``spaces/<Op>[.<Name>].space.json`` names an existing sweep template and
declares the axes a sampler draws from, the values derived from them, the legality
constraints, and the sweep case to emit. This module loads and validates a space and
evaluates its expressions; ``sampler.py`` draws from it.

Expressions are a small, safe subset of Python, evaluated by a tree walker (never by
``eval``): literals, lists, arithmetic, comparisons, ``and``/``or``/``not``,
``x if c else y``, indexing, and the functions ``min``, ``max``, ``len``, ``abs``.
Anything else (calls to other functions, attribute access, comprehensions, lambdas) is a
load error that names the space.

Space file layout::

    {
      "version": 1,
      "operation": "SdpaFwd",
      "description": "...",                      # optional, informational
      "engine": "hipkernel:Gfx950AttentionDense", # optional, informational
      "template": "../../integration-test-bundles/quick/SdpaFwd/Default/graph.template.json",
      "axes":        {"b": {"values": [1, 2]}, "sq": {"range": [1, 64], "boundaries": [1, 64]}},
      "derive":      {"hq": "b * 2"},
      "constraints": [{"rule": "sq <= 64", "reason": "why"}],
      "case":        {"values": {...}, "metadata": {"purpose": "probe"}}
    }

An axis may carry ``"late": true``: it was added after seeds had been recorded against the
space, and the sampler draws it from a stream of its own so the earlier axes keep their draws.
Late axes come last.

In ``case``, a string that starts with ``${`` and ends with ``}`` is replaced by the value
of the expression between them, keeping its JSON type; every other string is copied.
"""

import ast
import copy
import hashlib
import json
import operator
import re
from pathlib import Path

SPACE_FILE_SUFFIX = ".space.json"
DEFAULT_SPACE_NAME = "Default"
SPACE_VERSION = 1
DEFAULT_SPACES_DIR = Path(__file__).resolve().parent / "spaces"

# Case-metadata keys the driver owns; a space that sets one is a load error.
DRIVER_METADATA_KEYS = ("format_version", "generator", "seed", "notes")

_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
_FUNCTIONS = {"min": min, "max": max, "len": len, "abs": abs}
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_CONSTANT_TYPES = (bool, int, float, str, type(None))
# An integer exponent beyond this would only be a typo or an attempt to hang the loader.
_MAX_EXPONENT = 4096


class SpaceError(Exception):
    """A space cannot be loaded, or one of its expressions cannot be evaluated."""


class _Expr:
    """One validated expression of a space, ready to evaluate against an environment."""

    def __init__(self, source, space_name, where, known_names):
        self.source = source
        self._space_name = space_name
        self._where = where
        if not isinstance(source, str) or not source.strip():
            raise self._error("expression must be a non-empty string")
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError as e:
            raise self._error(f"syntax error: {e.msg}") from None
        self._body = tree.body
        self._validate(self._body, known_names)

    def _error(self, message):
        return SpaceError(
            f"space '{self._space_name}': {self._where}: {message}"
            f" (expression {self.source!r})"
        )

    def _validate(self, node, known):
        t = type(node)
        if t is ast.Constant:
            if not isinstance(node.value, _CONSTANT_TYPES):
                raise self._error(
                    f"constant of type {type(node.value).__name__} is not allowed"
                )
        elif t is ast.Name:
            if node.id not in known:
                raise self._error(f"unknown name '{node.id}'")
        elif t is ast.List:
            for e in node.elts:
                self._validate(e, known)
        elif t is ast.BinOp:
            if type(node.op) not in _BINOPS:
                raise self._error(f"operator {type(node.op).__name__} is not allowed")
            self._validate(node.left, known)
            self._validate(node.right, known)
        elif t is ast.UnaryOp:
            if type(node.op) not in _UNARYOPS and not isinstance(node.op, ast.Not):
                raise self._error(f"operator {type(node.op).__name__} is not allowed")
            self._validate(node.operand, known)
        elif t is ast.BoolOp:
            for v in node.values:
                self._validate(v, known)
        elif t is ast.Compare:
            for op in node.ops:
                if type(op) not in _CMPOPS:
                    raise self._error(f"comparison {type(op).__name__} is not allowed")
            self._validate(node.left, known)
            for c in node.comparators:
                self._validate(c, known)
        elif t is ast.IfExp:
            self._validate(node.test, known)
            self._validate(node.body, known)
            self._validate(node.orelse, known)
        elif t is ast.Subscript:
            if isinstance(node.slice, ast.Slice):
                raise self._error("slices are not allowed, only indexing")
            self._validate(node.value, known)
            self._validate(node.slice, known)
        elif t is ast.Call:
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise self._error(
                    "calls are limited to " + ", ".join(sorted(_FUNCTIONS))
                )
            if node.keywords:
                raise self._error("keyword arguments are not allowed")
            for a in node.args:
                if isinstance(a, ast.Starred):
                    raise self._error("starred arguments are not allowed")
                self._validate(a, known)
        elif t is ast.Attribute:
            raise self._error("attribute access is not allowed")
        elif t in (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp):
            raise self._error("comprehensions are not allowed")
        elif t is ast.Lambda:
            raise self._error("lambdas are not allowed")
        else:
            raise self._error(f"{t.__name__} is not allowed")

    def evaluate(self, env):
        try:
            return self._eval(self._body, env)
        except SpaceError:
            raise
        except (
            TypeError,
            ValueError,
            ZeroDivisionError,
            IndexError,
            OverflowError,
        ) as e:
            raise self._error(f"evaluation failed: {type(e).__name__}: {e}") from None

    def _eval(self, node, env):
        t = type(node)
        if t is ast.Constant:
            return node.value
        if t is ast.Name:
            return env[node.id]
        if t is ast.List:
            return [self._eval(e, env) for e in node.elts]
        if t is ast.BinOp:
            left = self._eval(node.left, env)
            right = self._eval(node.right, env)
            if (
                isinstance(node.op, ast.Pow)
                and isinstance(right, int)
                and abs(right) > _MAX_EXPONENT
            ):
                raise self._error(f"exponent {right} exceeds {_MAX_EXPONENT}")
            return _BINOPS[type(node.op)](left, right)
        if t is ast.UnaryOp:
            value = self._eval(node.operand, env)
            if isinstance(node.op, ast.Not):
                return not value
            return _UNARYOPS[type(node.op)](value)
        if t is ast.BoolOp:
            is_and = isinstance(node.op, ast.And)
            result = None
            for v in node.values:
                result = self._eval(v, env)
                if bool(result) != is_and:
                    return result
            return result
        if t is ast.Compare:
            left = self._eval(node.left, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, env)
                if not _CMPOPS[type(op)](left, right):
                    return False
                left = right
            return True
        if t is ast.IfExp:
            branch = node.body if self._eval(node.test, env) else node.orelse
            return self._eval(branch, env)
        if t is ast.Subscript:
            return self._eval(node.value, env)[self._eval(node.slice, env)]
        # ast.Call: validated to a whitelisted function.
        return _FUNCTIONS[node.func.id](*(self._eval(a, env) for a in node.args))


class _Subst:
    """A ``${expr}`` string of the case block."""

    def __init__(self, expr):
        self.expr = expr


def _is_subst(text):
    return isinstance(text, str) and text.startswith("${") and text.endswith("}")


def _compile_case(node, space_name, where, known):
    """Copy of a case-block JSON tree with every ``${expr}`` string precompiled."""
    if isinstance(node, str):
        if _is_subst(node):
            return _Subst(_Expr(node[2:-1], space_name, where, known))
        return node
    if isinstance(node, list):
        return [
            _compile_case(v, space_name, f"{where}[{i}]", known)
            for i, v in enumerate(node)
        ]
    if isinstance(node, dict):
        return {
            k: _compile_case(v, space_name, f"{where}.{k}", known)
            for k, v in node.items()
        }
    return node


def _render_case(node, env):
    if isinstance(node, _Subst):
        return copy.deepcopy(node.expr.evaluate(env))
    if isinstance(node, list):
        return [_render_case(v, env) for v in node]
    if isinstance(node, dict):
        return {k: _render_case(v, env) for k, v in node.items()}
    return node


class Axis:
    """One sampled axis: a ``values`` list, or an integer ``range`` with ``boundaries``.

    A ``late`` axis was added to a space after seeds had been recorded against it. The sampler
    draws it from a stream of its own, so the draws of every earlier axis, and so the shape a
    shape seed fixes, stay exactly what they were (see ``sampler.sample``).
    """

    def __init__(self, name, values=None, lo=None, hi=None, boundaries=(), late=False):
        self.name = name
        self.values = values
        self.lo = lo
        self.hi = hi
        self.boundaries = tuple(boundaries)
        self.late = late

    @property
    def is_range(self):
        return self.values is None

    def draw(self, rng):
        """One value: a boundary with probability 1/2, else uniform. Always consumes
        the same random draws per axis kind so a seed fixes the whole sample."""
        if not self.is_range:
            return copy.deepcopy(self.values[rng.randrange(len(self.values))])
        use_boundary = rng.random() < 0.5
        if use_boundary and self.boundaries:
            return self.boundaries[rng.randrange(len(self.boundaries))]
        return rng.randint(self.lo, self.hi)


class Constraint:
    def __init__(self, expr, reason):
        self.expr = expr
        self.reason = reason

    @property
    def rule(self):
        return self.expr.source


class Space:
    """A loaded, validated space declaration."""

    def __init__(self):
        self.path = None
        self.operation = None
        self.name = None
        self.description = None
        self.engine = None
        self.template_path = None
        self.axes = {}
        self.derive = {}
        self.constraints = []
        self.case_values = None
        self.case_metadata = {}
        self.space_sha256 = None
        self.template_sha256 = None

    @property
    def space_id(self):
        """``<Op>/<Space>``, the name ``--filter`` matches."""
        return f"{self.operation}/{self.name}"

    @property
    def suite_name(self):
        """The gtest suite the harness registers for this space's sweep directory."""
        return f"gauntlet_{self.operation}_{self.name}"

    def derive_env(self, axis_values):
        """Axis values plus every derived value, in declaration order."""
        env = dict(axis_values)
        for name, expr in self.derive.items():
            env[name] = expr.evaluate(env)
        return env

    def first_violation(self, env):
        """The first violated constraint, or None when the sample is legal."""
        for c in self.constraints:
            if not c.expr.evaluate(env):
                return c
        return None

    def render(self, env):
        """The sweep case ``{"values": ..., "metadata": ...}`` for a legal sample."""
        return {
            "values": _render_case(self.case_values, env),
            "metadata": _render_case(self.case_metadata, env),
        }


def _split_file_name(path):
    name = path.name
    if not name.endswith(SPACE_FILE_SUFFIX):
        raise SpaceError(
            f"space file '{path}': name must end with '{SPACE_FILE_SUFFIX}'"
        )
    stem = name[: -len(SPACE_FILE_SUFFIX)]
    parts = stem.split(".")
    if len(parts) > 2 or not all(_NAME_RE.fullmatch(p) for p in parts):
        raise SpaceError(
            f"space file '{path}': expected <Op>[.<Name>]{SPACE_FILE_SUFFIX} with"
            " [A-Za-z0-9_] names"
        )
    return parts[0], parts[1] if len(parts) == 2 else DEFAULT_SPACE_NAME


def _read_json_object(path, label, space_name):
    try:
        data = path.read_bytes()
    except OSError as e:
        raise SpaceError(
            f"space '{space_name}': cannot read {label} '{path}': {e}"
        ) from None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as e:
        raise SpaceError(
            f"space '{space_name}': {label} '{path}' is not valid JSON: {e}"
        ) from None
    if not isinstance(obj, dict):
        raise SpaceError(
            f"space '{space_name}': {label} '{path}' must be a JSON object"
        )
    return obj, hashlib.sha256(data).hexdigest()


def _load_axis(space_name, name, decl, reserved):
    where = f"axis '{name}'"
    if not _NAME_RE.fullmatch(name) or name[0].isdigit():
        raise SpaceError(f"space '{space_name}': {where}: not a valid name")
    if name in reserved:
        raise SpaceError(f"space '{space_name}': {where}: name is reserved")
    if not isinstance(decl, dict):
        raise SpaceError(f"space '{space_name}': {where}: must be an object")
    unknown = set(decl) - {"values", "range", "boundaries", "late"}
    if unknown:
        raise SpaceError(
            f"space '{space_name}': {where}: unknown keys {sorted(unknown)}"
        )
    late = decl.get("late", False)
    if not isinstance(late, bool):
        raise SpaceError(f"space '{space_name}': {where}: 'late' must be true or false")
    if ("values" in decl) == ("range" in decl):
        raise SpaceError(
            f"space '{space_name}': {where}: needs exactly one of 'values' or 'range'"
        )
    if "values" in decl:
        if "boundaries" in decl:
            raise SpaceError(
                f"space '{space_name}': {where}: 'boundaries' needs a 'range'"
            )
        values = decl["values"]
        if not isinstance(values, list) or not values:
            raise SpaceError(
                f"space '{space_name}': {where}: 'values' must be a non-empty list"
            )
        return Axis(name, values=values, late=late)
    rng = decl["range"]
    ints = (
        isinstance(rng, list)
        and len(rng) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in rng)
    )
    if not ints or rng[0] > rng[1]:
        raise SpaceError(
            f"space '{space_name}': {where}: 'range' must be [lo, hi] integers, lo <= hi"
        )
    boundaries = decl.get("boundaries", [])
    if not isinstance(boundaries, list) or not all(
        isinstance(b, int) and not isinstance(b, bool) and rng[0] <= b <= rng[1]
        for b in boundaries
    ):
        raise SpaceError(
            f"space '{space_name}': {where}: 'boundaries' must be integers inside the range {rng}"
        )
    return Axis(name, lo=rng[0], hi=rng[1], boundaries=boundaries, late=late)


def load_space(path):
    """Load and validate one space file. Raises ``SpaceError`` naming the space."""
    path = Path(path)
    op, name = _split_file_name(path)
    space_name = f"{op}/{name}"
    decl, space_sha = _read_json_object(path, "space file", space_name)

    unknown = set(decl) - {
        "version", "operation", "description", "engine", "template",
        "axes", "derive", "constraints", "case",
    }  # fmt: skip
    if unknown:
        raise SpaceError(
            f"space '{space_name}': unknown top-level keys {sorted(unknown)}"
        )
    if decl.get("version") != SPACE_VERSION or isinstance(decl.get("version"), bool):
        raise SpaceError(f"space '{space_name}': 'version' must be {SPACE_VERSION}")
    if decl.get("operation") != op:
        raise SpaceError(
            f"space '{space_name}': 'operation' is {decl.get('operation')!r},"
            f" but the file name says '{op}'"
        )
    for key in ("description", "engine"):
        if key in decl and not isinstance(decl[key], str):
            raise SpaceError(f"space '{space_name}': '{key}' must be a string")

    space = Space()
    space.path = path.resolve()
    space.operation = op
    space.name = name
    space.description = decl.get("description")
    space.engine = decl.get("engine")
    space.space_sha256 = space_sha

    template = decl.get("template")
    if not isinstance(template, str) or not template:
        raise SpaceError(
            f"space '{space_name}': 'template' must be a relative path string"
        )
    space.template_path = (path.parent / template).resolve()
    if not space.template_path.is_file():
        raise SpaceError(
            f"space '{space_name}': template '{template}' not found"
            f" (resolved to {space.template_path})"
        )
    _template, space.template_sha256 = _read_json_object(
        space.template_path, "template", space_name
    )

    reserved = set(_FUNCTIONS) | {"True", "False", "None"}
    axes = decl.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise SpaceError(f"space '{space_name}': 'axes' must be a non-empty object")
    for axis_name, axis_decl in axes.items():
        axis = _load_axis(space_name, axis_name, axis_decl, reserved)
        if space.axes and not axis.late and next(reversed(space.axes.values())).late:
            raise SpaceError(
                f"space '{space_name}': axis '{axis_name}' follows a 'late' axis but is not"
                " late itself; late axes come last"
            )
        space.axes[axis_name] = axis

    known = set(space.axes)
    derive = decl.get("derive", {})
    if not isinstance(derive, dict):
        raise SpaceError(f"space '{space_name}': 'derive' must be an object")
    for dname, source in derive.items():
        where = f"derive '{dname}'"
        if not _NAME_RE.fullmatch(dname) or dname[0].isdigit() or dname in reserved:
            raise SpaceError(f"space '{space_name}': {where}: not a valid name")
        if dname in known:
            raise SpaceError(
                f"space '{space_name}': {where}: name is already an axis or derived value"
            )
        # A derived value may use axes and earlier derived values, not itself or later ones.
        space.derive[dname] = _Expr(source, space_name, where, known)
        known.add(dname)

    constraints = decl.get("constraints", [])
    if not isinstance(constraints, list):
        raise SpaceError(f"space '{space_name}': 'constraints' must be a list")
    for i, c in enumerate(constraints):
        where = f"constraint #{i}"
        if (
            not isinstance(c, dict)
            or set(c) != {"rule", "reason"}
            or not isinstance(c["reason"], str)
            or not c["reason"]
        ):
            raise SpaceError(
                f"space '{space_name}': {where}: must be an object with 'rule' and a"
                " non-empty 'reason'"
            )
        space.constraints.append(
            Constraint(_Expr(c["rule"], space_name, where, known), c["reason"])
        )

    case = decl.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("values"), dict):
        raise SpaceError(
            f"space '{space_name}': 'case' must be an object with a 'values' object"
        )
    if set(case) - {"values", "metadata"}:
        raise SpaceError(f"space '{space_name}': 'case' has unknown keys")
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        raise SpaceError(f"space '{space_name}': 'case.metadata' must be an object")
    clash = sorted(set(metadata) & set(DRIVER_METADATA_KEYS))
    if clash:
        raise SpaceError(
            f"space '{space_name}': 'case.metadata' sets driver-owned keys {clash}"
        )
    space.case_values = _compile_case(case["values"], space_name, "case.values", known)
    space.case_metadata = _compile_case(metadata, space_name, "case.metadata", known)
    return space


def space_files(spaces_dir=None):
    """Every ``*.space.json`` file in a spaces directory, sorted by name."""
    spaces_dir = Path(spaces_dir) if spaces_dir is not None else DEFAULT_SPACES_DIR
    return sorted(spaces_dir.glob(f"*{SPACE_FILE_SUFFIX}"))


def load_spaces(spaces_dir=None):
    """Load every space of a directory, sorted by ``<Op>/<Space>``."""
    spaces = [load_space(p) for p in space_files(spaces_dir)]
    seen = {}
    for s in spaces:
        if s.suite_name in seen:
            raise SpaceError(
                f"spaces '{seen[s.suite_name]}' and '{s.space_id}' map to the same gtest suite"
                f" '{s.suite_name}'"
            )
        seen[s.suite_name] = s.space_id
    return sorted(spaces, key=lambda s: s.space_id)
