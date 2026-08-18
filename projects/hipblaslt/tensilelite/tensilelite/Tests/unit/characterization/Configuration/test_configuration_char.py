################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Configuration``: the ``Parameter``
operator surface, ``ReadWriteTransformDict``, and ``ProjectConfig`` (sections,
dotted keys, defaults, constraints via the AST ``ExpressionEvaluator``)."""

import ast
import operator

import pytest

from Tensile.Configuration import (
    Parameter, ProjectConfig, ReadWriteTransformDict,
    CallableParameter, ExpressionEvaluator,
)

pytestmark = pytest.mark.unit


def _p(name, val):
    return Parameter(name, val)


# --- Parameter comparison operators (param-vs-param + param-vs-scalar) -------

@pytest.mark.parametrize("op", [operator.lt, operator.le, operator.eq, operator.ne,
                                operator.gt, operator.ge],
                         ids=["lt", "le", "eq", "ne", "gt", "ge"])
def test_parameter_comparisons(op, snapshot):
    a, b = _p("a", 8), _p("b", 2)
    assert {"pp": op(a, b), "ps": op(a, 5), "sp": op(5, a)} == snapshot


# --- Parameter arithmetic/bitwise (param-param, param-scalar, scalar-param) --

@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv,
                                operator.floordiv, operator.mod, operator.pow,
                                operator.rshift, operator.lshift,
                                operator.and_, operator.or_, operator.xor],
                         ids=["add", "sub", "mul", "div", "floordiv", "mod", "pow",
                              "rshift", "lshift", "and", "or", "xor"])
def test_parameter_binary_ops(op, snapshot):
    a, b = _p("a", 12), _p("b", 4)
    assert {"pp": op(a, b), "ps": op(a, 3), "sp": op(48, a)} == snapshot


def test_parameter_unary_and_bool():
    a = _p("a", 5)
    z = _p("z", 0)
    assert -a == -5 and +a == 5 and ~a == ~5
    assert bool(a) is True and bool(z) is False


def test_parameter_value_accessors_and_reset():
    p = Parameter("p", 5, defaultValue=99, description="desc")
    assert p.getValue() == 5
    assert p.getDefault() == 99
    assert p.getDescription() == "desc"
    p.value = 7
    assert p.getValue() == 7
    p.resetToDefault()
    assert p.getValue() == 99


def test_parameter_type_preservation_raises():
    p = Parameter("p", 5)
    with pytest.raises(AttributeError):
        p.value = "a string"  # type mismatch


# --- ReadWriteTransformDict -------------------------------------------------

def test_rwt_dict_basic():
    d = ReadWriteTransformDict()
    d.set("k", 1)
    assert d.get("k") == 1
    assert d.get("missing", "def") == "def"
    assert "k" in d.toDict()


def test_rwt_dict_transforms():
    d = ReadWriteTransformDict(readTransformFunc=lambda o, k: o.readNoTransform(k) * 10,
                               writeTransformFunc=lambda o, k, v: o.writeNoTransform(k, v + 1))
    assert d.hasReadTransform() and d.hasWriteTransform()
    d["x"] = 4            # write transform -> stores 5
    assert d.readNoTransform("x") == 5
    assert d["x"] == 50   # read transform -> 5*10


# --- ProjectConfig ----------------------------------------------------------

def test_project_config_values_and_sections():
    cfg = ProjectConfig()
    cfg.createValue("a", 1, defaultValue=10, description="A")
    sec = cfg.createSection("sub")
    sec.createValue("b", 2, defaultValue=20)
    assert cfg["a"] == 1
    assert cfg["sub.b"] == 2          # dotted access
    assert "a" in cfg and "sub.b" in cfg
    assert cfg.getDefaultValue("a") == 10
    assert cfg.getDescription("a") == "A"


def test_project_config_setitem_and_reset():
    cfg = ProjectConfig()
    cfg.createValue("a", 1, defaultValue=10)
    sec = cfg.createSection("sub")
    sec.createValue("b", 2, defaultValue=20)
    cfg["a"] = 5             # top-level set
    assert cfg["a"] == 5
    cfg.resetToDefaults()    # recurses into the section too
    assert cfg["a"] == 10


@pytest.mark.parametrize("meth", ["__rlt__", "__rle__", "__req__", "__rne__", "__rgt__", "__rge__"],
                         ids=["rlt", "rle", "req", "rne", "rgt", "rge"])
def test_parameter_reflected_comparisons_explicit(meth, snapshot):
    # Python's comparison protocol uses the opposite operator (a.__gt__) for
    # `scalar < param`, so these __r*__ comparison methods are never auto-called
    # (see DECISIONS D9). Call them explicitly to pin their logic.
    a = _p("a", 8)
    fn = getattr(a, meth)
    assert {"vs_param": fn(_p("b", 2)), "vs_scalar": fn(5)} == snapshot


def test_project_config_constraints_pass():
    cfg = ProjectConfig()
    cfg.createValue("x", 5, defaultValue=5)
    cfg.addConstraint("x < 10")
    assert cfg.checkConstraints()


def test_project_config_constraints_fail():
    cfg = ProjectConfig()
    cfg.createValue("x", 50, defaultValue=50)
    cfg.addConstraint("x < 10")
    with pytest.raises(AssertionError):
        cfg.checkConstraints()


# --- ExpressionEvaluator + CallableParameter factories ----------------------

def _ev(expr, ctx):
    r = ExpressionEvaluator().evaluate(ast.parse(expr, mode="exec"), ctx)
    return r() if callable(r) else r


@pytest.mark.parametrize(
    "expr,ctx,expected",
    [
        ("x > 5", {"x": 10}, True),          # Compare / Gt
        ("x < 5", {"x": 10}, False),         # Compare / Lt
        ("x + 3", {"x": 4}, 7),              # BinOp / Add
        ("x * y", {"x": 4, "y": 5}, 20),     # BinOp / Mult
        ("x and y", {"x": 1, "y": 2}, 2),    # BoolOp / And
        ("x or y", {"x": 0, "y": 9}, 9),     # BoolOp / Or
        ("max(x, y)", {"x": 3, "y": 7}, 7),  # Call (2-arg) / max
        ("min(x, y)", {"x": 3, "y": 7}, 3),  # Call (2-arg) / min
        ("5", {}, 5),                        # Constant/Num leaf
        ("'hi'", {}, "hi"),                  # Constant/Str leaf
        ("x if y > 0 else z", {"x": 1, "y": 1, "z": 2}, 1),   # IfExp (true; test is callable)
        ("x if y > 0 else z", {"x": 1, "y": -1, "z": 2}, 2),  # IfExp (false)
    ],
    ids=["gt", "lt", "add", "mult", "and", "or", "max", "min",
         "num", "str", "ifexp_t", "ifexp_f"],
)
def test_expression_evaluator(expr, ctx, expected):
    assert _ev(expr, ctx) == expected


@pytest.mark.parametrize("expr,ctx,expected",
                         [("not x", {"x": 0}, True), ("-x", {"x": 5}, -5)],
                         ids=["not", "usub"])
def test_expression_evaluator_unary(expr, ctx, expected):
    assert _ev(expr, ctx) == expected


def test_expression_evaluator_name_no_context(capsys):
    # A name not in context -> prints + returns the name string.
    r = ExpressionEvaluator().evaluate(ast.parse("undefined_var", mode="exec"), {})
    assert r == "undefined_var"
    assert "No context" in capsys.readouterr().out


def test_expression_evaluator_assign():
    ctx = {}
    ExpressionEvaluator().evaluate(ast.parse("x = 7", mode="exec"), ctx)
    assert ctx["x"] == 7


def test_create_binary_op_custom_function():
    # op passed as a callable -> the CustomBinaryOp branch.
    op = CallableParameter.createBinaryOp(3, 4, lambda a, b: a * b + 1)
    assert op() == 13


def test_create_binary_op_bad_func_raises():
    with pytest.raises(CallableParameter.BadFunc):
        CallableParameter.createBinaryOp("a", 1, "Add")  # str + int -> TypeError


def test_create_unary_op_custom_and_none():
    assert CallableParameter.createUnaryOp(5, "None")() == 5
    assert CallableParameter.createUnaryOp(5, lambda v: v + 100)() == 105
