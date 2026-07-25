################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Behavioral kill tests for Configuration.py mutation-testing survivors (slice-4).

Each test pins an observable behavior of Tensile.Configuration that distinguishes
the original implementation from a specific mutant. Grouped by the class/method
under test. Companion equivalence proofs for the residual (unkillable) survivors
live in the slice-4 certificate.
"""

import ast

import pytest

from Tensile.Configuration import (
    CallableParameter,
    ExpressionEvaluator,
    Parameter,
    ProjectConfig,
    ReadWriteTransformDict,
)


class _EqOnNthCall:
    """Callable whose ``__eq__`` returns True on the Nth comparison.

    ``createBinaryOp``/``createUnaryOp`` resolve ``opKey`` with
    ``next(k for k in FuncMap if FuncMap[k] == op)``. A plain function/string op
    never equals a FuncMap lambda, so ``opKey`` stays None and the bool-wrap
    branch (and the guard list it consults) is unreachable. An op whose reflected
    ``__eq__`` returns True on the Nth comparison forces ``opKey`` to the Nth
    FuncMap key, exercising that branch. ``echo`` makes the call value depend on
    its argument so operand-dropping mutants are observable.
    """

    def __init__(self, n, echo=False):
        self._n = n
        self._calls = 0
        self._echo = echo

    def __call__(self, *args):
        return args[-1] if self._echo else 2

    def __eq__(self, other):
        self._calls += 1
        return self._calls == self._n

    def __hash__(self):
        return id(self)


class _Recording2ArgEq:
    """Two-arg callable that records its last call and forces ``opKey`` on the Nth.

    Unlike ``_EqOnNthCall`` this requires exactly two positional arguments and
    records them, so bool-wrap mutants that replace an operand with ``None`` or
    drop an operand (turning the call into a one-arg call) are observable: the
    recorded tuple changes, or the dropped-operand call raises ``TypeError``.
    """

    def __init__(self, n):
        self._n = n
        self._calls = 0
        self.last = None

    def __call__(self, a, b):
        self.last = (a, b)
        return True

    def __eq__(self, other):
        self._calls += 1
        return self._calls == self._n

    def __hash__(self):
        return id(self)


class TestCreateBinaryOpStringSemantics:
    """String-op path: key lookup + operator semantics for every FuncMap entry."""

    CASES = [
        ("And", 2, 3, 3),
        ("And", 0, 5, 0),
        ("Or", 0, 3, 3),
        ("Or", 4, 9, 4),
        ("Lt", 5, 5, False),
        ("Lt", 2, 5, True),
        ("LtE", 5, 5, True),
        ("LtE", 6, 5, False),
        ("Eq", 5, 5, True),
        ("Eq", 5, 6, False),
        ("NotEq", 5, 5, False),
        ("NotEq", 5, 6, True),
        ("Gt", 5, 5, False),
        ("Gt", 6, 5, True),
        ("GtE", 5, 5, True),
        ("GtE", 4, 5, False),
        ("Mult", 3, 4, 12),
        ("Pow", 2, 3, 8),
        ("Div", 6, 4, 1.5),
        ("FloorDiv", 7, 2, 3),
        ("Mod", 7, 3, 1),
        ("Add", 3, 4, 7),
        ("Sub", 10, 3, 7),
        ("BitAnd", 6, 3, 2),
        ("BitOr", 4, 1, 5),
        ("BitXor", 5, 3, 6),
        ("LShift", 1, 3, 8),
        ("RShift", 16, 2, 4),
        ("min", 3, 7, 3),
        ("max", 3, 7, 7),
    ]

    @pytest.mark.parametrize("op,lhs,rhs,expected", CASES)
    def test_binary_op_semantics(self, op, lhs, rhs, expected):
        binOp = CallableParameter.createBinaryOp(lhs, rhs, op)
        assert binOp() == expected
        assert binOp.readNoTransform("name") == op
        assert binOp.readNoTransform("description") == "Binary operaton with two operands"

    def test_unknown_string_op_raises(self):
        with pytest.raises(AssertionError) as e:
            CallableParameter.createBinaryOp(1, 2, "NoSuchOp")
        assert str(e.value) == "Missing operation in funcMap: NoSuchOp"


class TestCreateBinaryOpBoolWrap:
    """opKey/guard/list bool-wrap branch, reached via an ``__eq__``-True op.

    The guard list is ["And","Or","Lt","Le","Eq","NotEq","Gt","Ge"]. The keys
    that are ALSO in FuncMap (And=1,Or=2,Lt=3,Eq=5,NotEq=6,Gt=7 by insertion
    order) reach the bool wrap; "Le"/"Ge" are not FuncMap keys, so those guard
    slots are dead (proved equivalent, not tested here).
    """

    IN_GUARD_POSITIONS = [1, 2, 3, 5, 6, 7]

    @pytest.mark.parametrize("n", IN_GUARD_POSITIONS)
    def test_in_guard_key_is_bool_wrapped(self, n):
        op = _EqOnNthCall(n)
        binOp = CallableParameter.createBinaryOp(1, 1, op)
        assert binOp() is True

    def test_and_not_or_in_guard_condition(self):
        # opKey='Mult' (position 9): not in the guard list, so NOT bool-wrapped.
        # Original: (True and False) -> else -> raw 2. `or` mutant: (True or
        # False) -> if -> bool(2) is True.
        op = _EqOnNthCall(9)
        binOp = CallableParameter.createBinaryOp(1, 1, op)
        assert binOp() == 2

    def test_bool_wrap_passes_both_operands_in_order(self):
        # opKey='And' (position 1) reaches the bool-wrap lambda
        # `bool(func(obj.lhs, obj.rhs))`. A recording two-arg op detects operand
        # replacement (None) and operand drop (one-arg call -> TypeError).
        op = _Recording2ArgEq(1)
        binOp = CallableParameter.createBinaryOp(5, 7, op)
        assert binOp() is True
        assert op.last == (5, 7)


class TestCreateUnaryOpStringSemantics:
    """String-op path for the unary FuncMap."""

    CASES = [
        ("Not", 0, True),
        ("Not", 5, False),
        ("Invert", 5, ~5),
        ("USub", 5, -5),
        ("UAdd", 5, 5),
        ("None", 7, 7),
    ]

    @pytest.mark.parametrize("op,val,expected", CASES)
    def test_unary_op_semantics(self, op, val, expected):
        unOp = CallableParameter.createUnaryOp(val, op)
        assert unOp() == expected
        assert unOp.readNoTransform("name") == op
        assert unOp.readNoTransform("description") == "Unary operaton with one operand"

    def test_custom_func_op_name(self):
        unOp = CallableParameter.createUnaryOp(5, lambda v: v)
        assert unOp.readNoTransform("name") == "CustomUnaryOp"

    def test_unknown_string_op_raises(self):
        with pytest.raises(AssertionError) as e:
            CallableParameter.createUnaryOp(1, "NoSuchOp")
        assert str(e.value) == "Missing operation in funcMap: NoSuchOp"


class TestCreateUnaryOpBoolWrap:
    """'Not' bool-wrap branch, reached via an ``__eq__``-True op (position 1)."""

    def test_not_branch_is_bool_wrapped(self):
        # opKey='Not' -> callFunc = bool(func(rhs)); rhs truthy -> True.
        # Any mutant that skips the wrap (opKey=None, is-None, not-in, mutated
        # guard list) falls to else -> raw rhs (1), and None/bool(None) mutants
        # of the wrap body yield None/False. All differ from True.
        op = _EqOnNthCall(1, echo=True)
        unOp = CallableParameter.createUnaryOp(1, op)
        assert unOp() is True

    def test_and_not_or_in_not_guard(self):
        # opKey='Invert' (position 2): not in ['Not'], so NOT wrapped.
        # Original else -> raw 3; `or` mutant -> bool(3) is True.
        op = _EqOnNthCall(2, echo=True)
        unOp = CallableParameter.createUnaryOp(3, op)
        assert unOp() == 3


class TestCallableParameterInit:
    """CallableParameter.__init__ container setup and value/write protection."""

    def test_default_description_empty(self):
        p = CallableParameter("x", lambda s: 1)
        assert p.readNoTransform("description") == ""

    def test_initial_value_is_zero_int(self):
        p = CallableParameter("x", lambda s: 1)
        assert p.readNoTransform("type") is int
        assert p.readNoTransform("value") == 0

    def test_default_value_is_zero(self):
        p = CallableParameter("x", lambda s: 1)
        assert p.getDefault() == 0

    def test_value_writes_are_suppressed(self):
        p = CallableParameter("x", lambda s: 1)
        p["value"] = 99
        assert p.readNoTransform("value") == 0

    def test_non_value_writes_are_rejected(self):
        p = CallableParameter("x", lambda s: 1)
        with pytest.raises(AttributeError):
            p.bogusAttr = 5

    def test_call_returns_callfunc_result(self):
        p = CallableParameter("x", lambda s: 42)
        assert p() == 42


class TestParameterComparisons:
    """Parameter comparison dunders, both Parameter and scalar branches."""

    def test_lt(self):
        p = Parameter("a", 10)
        q = Parameter("b", 10)
        assert (p < q) is False
        assert (p < 10) is False
        assert (p < 11) is True

    def test_gt(self):
        p = Parameter("a", 10)
        q = Parameter("b", 10)
        assert (p > q) is False
        assert (p > 10) is False
        assert (p > 9) is True

    def test_reflected_lt(self):
        a = Parameter("a", 10)
        assert a.__rlt__(Parameter("b", 10)) is False
        assert a.__rlt__(10) is False
        assert a.__rlt__(9) is True

    def test_reflected_le(self):
        a = Parameter("a", 10)
        assert a.__rle__(Parameter("b", 10)) is True
        assert a.__rle__(10) is True
        assert a.__rle__(11) is False

    def test_reflected_gt(self):
        a = Parameter("a", 10)
        assert a.__rgt__(Parameter("b", 10)) is False
        assert a.__rgt__(10) is False
        assert a.__rgt__(11) is True

    def test_reflected_ge(self):
        a = Parameter("a", 10)
        assert a.__rge__(Parameter("b", 10)) is True
        assert a.__rge__(10) is True
        assert a.__rge__(9) is False


class TestParameterReflectedArithmetic:
    """Parameter reflected arithmetic dunders, Parameter-lhs branch."""

    def test_radd(self):
        assert Parameter("b", 3).__radd__(Parameter("a", 5)) == 8

    def test_rsub(self):
        assert Parameter("b", 3).__rsub__(Parameter("a", 5)) == 2

    def test_rmul(self):
        assert Parameter("b", 3).__rmul__(Parameter("a", 5)) == 15

    def test_rtruediv(self):
        assert Parameter("b", 3).__rtruediv__(Parameter("a", 6)) == 2.0

    def test_rfloordiv(self):
        assert Parameter("b", 3).__rfloordiv__(Parameter("a", 7)) == 2

    def test_rmod(self):
        assert Parameter("b", 3).__rmod__(Parameter("a", 7)) == 1

    def test_rpow(self):
        assert Parameter("b", 3).__rpow__(Parameter("a", 2)) == 8

    def test_rrshift(self):
        assert Parameter("b", 1).__rrshift__(Parameter("a", 8)) == 4

    def test_rlshift(self):
        assert Parameter("b", 1).__rlshift__(Parameter("a", 8)) == 16

    def test_rand(self):
        assert Parameter("b", 3).__rand__(Parameter("a", 6)) == 2

    def test_ror(self):
        assert Parameter("b", 3).__ror__(Parameter("a", 6)) == 7

    def test_rxor(self):
        assert Parameter("b", 3).__rxor__(Parameter("a", 6)) == 5


class TestParameterInit:
    """Parameter.__init__ defaults and write-protection diagnostics."""

    def test_default_description_empty(self):
        p = Parameter("test", 100)
        assert p.readNoTransform("description") == ""

    def test_cannot_write_unknown_attribute_message(self):
        p = Parameter("test", 42)
        with pytest.raises(AttributeError) as e:
            p["type"] = str
        assert str(e.value) == "Cannot write attribute: type"

    def test_type_preservation_message_exact(self):
        p = Parameter("test", 42)
        with pytest.raises(AttributeError) as e:
            p["value"] = "string"
        assert str(e.value) == (
            "Type preservation: stored <class 'int'> != incoming <class 'str'>"
        )


class TestExpressionEvaluatorBehaviors:
    """ExpressionEvaluator.evaluate observable behaviors (non-message)."""

    def test_boolop_uses_both_operands(self):
        result = ExpressionEvaluator().evaluate(
            ast.parse("a and b", mode="exec"), {"a": False, "b": True}
        )
        assert result == False

    def test_call_one_arg_unary(self):
        result = ExpressionEvaluator().evaluate(
            ast.parse("USub(a)", mode="exec"), {"a": 10}
        )
        assert result == -10

    def test_assign_name_uses_context(self):
        ctx = {"a": 42}
        result = ExpressionEvaluator().evaluate(ast.parse("x = a", mode="exec"), ctx)
        assert result == 42
        assert ctx["x"] == 42

    def test_assign_expression_value_unwrapped(self):
        ctx = {"a": 10, "b": 20}
        ExpressionEvaluator().evaluate(ast.parse("x = a + b", mode="exec"), ctx)
        assert ctx["x"] == 30

    def test_assign_to_attribute_target(self):
        ctx = {"obj": {"x": 0}}
        ExpressionEvaluator().evaluate(ast.parse("obj.x = 5", mode="exec"), ctx)
        assert ctx["obj"]["x"] == 5

    def test_assign_unwraps_value_attribute(self):
        # `valueToAssign = value.value if hasattr(value, "value") else value`.
        # A boxed RHS whose .value differs from itself proves the object is
        # unwrapped; the hasattr(None, ...) mutant assigns the box instead.
        class Boxed:
            def __init__(self, v):
                self.value = v

        ctx = {"a": Boxed(42)}
        ExpressionEvaluator().evaluate(ast.parse("x = a", mode="exec"), ctx)
        assert ctx["x"] == 42

    def test_name_not_found_returns_string_and_prints(self, capsys):
        result = ExpressionEvaluator().evaluate(
            ast.parse("undefined_var", mode="exec"), {}
        )
        out = capsys.readouterr().out
        assert result == "undefined_var"
        assert out == "No context for named variable: undefined_var\n"

    def test_num_node_returns_n(self):
        class Num:
            pass

        n = Num()
        n.n = 42
        assert ExpressionEvaluator().evaluate(n, {}) == 42

    def test_str_node_returns_s(self):
        class Str:
            pass

        s = Str()
        s.s = "hi"
        assert ExpressionEvaluator().evaluate(s, {}) == "hi"


class TestExpressionEvaluatorAssertMessages:
    """ExpressionEvaluator.evaluate diagnostic-message contracts (exact text)."""

    def test_module_single_expression_message(self):
        with pytest.raises(AssertionError) as e:
            ExpressionEvaluator().evaluate(ast.parse("a\nb", mode="exec"), {})
        assert str(e.value) == "Expecting only one expression"

    def test_bad_arity_message(self):
        with pytest.raises(AssertionError) as e:
            ExpressionEvaluator().evaluate(ast.parse("foo(a, b, c)", mode="exec"), {})
        assert str(e.value) == "Unknown function call with 3 parameters"

    def test_unhandled_target_message(self):
        with pytest.raises(AssertionError) as e:
            ExpressionEvaluator().evaluate(ast.parse("d[0] = 5", mode="exec"), {"d": {}})
        assert str(e.value) == "Don't know how to handle target node type: Subscript"

    def test_missing_attribute_message(self):
        with pytest.raises(AssertionError) as e:
            ExpressionEvaluator().evaluate(
                ast.parse("obj.missing", mode="exec"), {"obj": {"x": 1}}
            )
        assert str(e.value) == "No attribute for named variable: missing"

    def test_unknown_node_type_message(self):
        with pytest.raises(AssertionError) as e:
            ExpressionEvaluator().evaluate(ast.parse("[1, 2]", mode="exec"), {})
        assert str(e.value) == "Unknown node type: List"


class TestProjectConfigDottedAccess:
    """ProjectConfig.__getContainer / __setitem__ dotted-key recursion."""

    def _two_level(self, desc="port desc", default=None):
        cfg = ProjectConfig()
        net = cfg.createSection("Network")
        if default is None:
            net.createValue("PORT", 8080, description=desc)
        else:
            net.createValue("PORT", 8080, defaultValue=default, description=desc)
        return cfg

    def test_get_description_two_level(self):
        cfg = self._two_level()
        assert cfg.getDescription("Network.PORT") == "port desc"

    def test_get_description_three_level(self):
        cfg = ProjectConfig()
        a = cfg.createSection("A")
        b = a.createSection("B")
        b.createValue("C", 1, defaultValue=7, description="cdesc")
        assert cfg.getDescription("A.B.C") == "cdesc"
        assert cfg.getDefaultValue("A.B.C") == 7

    def test_setitem_three_level(self):
        cfg = ProjectConfig()
        a = cfg.createSection("A")
        b = a.createSection("B")
        b.createValue("C", 1)
        cfg["A.B.C"] = 999
        assert cfg["A.B.C"] == 999


class TestProjectConfigCreateValue:
    """ProjectConfig.createValue name/description handling."""

    def test_default_description_empty(self):
        cfg = ProjectConfig()
        cfg.createValue("p", 1)
        assert cfg.getDescription("p") == ""

    def test_name_is_preserved(self):
        cfg = ProjectConfig()
        cfg.createValue("p", 1)
        assert cfg.readNoTransform("p").name == "p"


class TestProjectConfigConstraints:
    """ProjectConfig.checkConstraints diagnostic-message contract."""

    def test_failed_constraint_message_exact(self):
        cfg = ProjectConfig()
        cfg.addConstraint("False")
        with pytest.raises(AssertionError) as e:
            cfg.checkConstraints()
        assert str(e.value) == "Constraint evaluation failed: False"

    def test_satisfied_constraint_returns_truthy(self):
        cfg = ProjectConfig()
        cfg.addConstraint("True")
        assert cfg.checkConstraints()


class TestReadWriteTransformDictInit:
    """ReadWriteTransformDict.__init__ transform guards."""

    def test_non_callable_read_transform_is_ignored(self):
        d = ReadWriteTransformDict(5)
        assert d.hasReadTransform() is False

    def test_non_callable_write_transform_is_ignored(self):
        d = ReadWriteTransformDict(writeTransformFunc=5)
        assert d.hasWriteTransform() is False


class TestReadWriteTransformDictRepr:
    """ReadWriteTransformDict.__repr__ / __toPrettyLines formatting."""

    def test_repr_header_and_no_marker(self):
        d = ReadWriteTransformDict()
        d.writeNoTransform("a", 1)
        text = repr(d)
        assert "XX" not in text
        assert text.startswith("<ReadWriteTransformDict(")
        assert "\ta: 1" in text

    def test_repr_nested_header_and_pop(self):
        inner = ReadWriteTransformDict()
        inner.writeNoTransform("c", 1)
        top = ReadWriteTransformDict()
        top.writeNoTransform("inner", inner)
        text = repr(top)
        assert "XX" not in text
        assert text.startswith("<ReadWriteTransformDict(")
        assert "\tinner: <ReadWriteTransformDict(" in text
        assert "\t\tc: 1" in text


class TestReadWriteTransformDictFlatten:
    """ReadWriteTransformDict.flattenDict / toFlattenedDict."""

    def _nested(self):
        inner = ReadWriteTransformDict()
        inner.writeNoTransform("c", 1)
        param = ReadWriteTransformDict()
        param.writeNoTransform("value", inner)
        top = ReadWriteTransformDict()
        top.writeNoTransform("a", param)
        return top

    def test_default_separator_joins_with_dot(self):
        flat = ReadWriteTransformDict.flattenDict(self._nested())
        assert "a.c" in flat
        assert flat["a.c"] == 1

    def test_custom_separator_threaded(self):
        flat = self._nested().toFlattenedDict(separator="_")
        assert "a_c" in flat

    def test_three_level_flatten(self):
        inner = ReadWriteTransformDict()
        inner.writeNoTransform("c", 1)
        p2 = ReadWriteTransformDict()
        p2.writeNoTransform("value", inner)
        mid = ReadWriteTransformDict()
        mid.writeNoTransform("b", p2)
        pmid = ReadWriteTransformDict()
        pmid.writeNoTransform("value", mid)
        top = ReadWriteTransformDict()
        top.writeNoTransform("a", pmid)
        assert "a.b.c" in top.toFlattenedDict()

    def test_deepcopy_of_container_keeps_none(self):
        from copy import deepcopy

        d = ReadWriteTransformDict()
        d.writeNoTransform("x", 1)
        c = deepcopy({"a": d, "b": None})
        assert c["b"] is None
        assert c["a"].readNoTransform("x") == 1


class TestReadWriteTransformDictDeepCopyMemo:
    """__deepcopy__ memo-protocol contract.

    Kills the ``memo[id(self)] = result -> memo[id(self)] = None`` mutant. That
    mutation is masked under ``copy.deepcopy`` (the outer machinery re-records
    ``memo[id(self)]`` after ``__deepcopy__`` returns, even under aliasing), so a
    direct ``__deepcopy__(memo)`` call is the only observation that distinguishes
    it. Adversarial audit of the slice-4 residuals surfaced this; the original
    certificate wrongly labelled the mutant unkillable by any test.
    """

    def test_deepcopy_records_result_in_memo(self):
        d = ReadWriteTransformDict()
        d.writeNoTransform("a", 1)
        memo = {}
        result = d.__deepcopy__(memo)
        assert memo[id(d)] is result
