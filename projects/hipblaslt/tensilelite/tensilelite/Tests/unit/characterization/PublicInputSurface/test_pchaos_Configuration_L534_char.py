################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Characterization tests for Configuration.py line 534.

Branch ID: cab4f49fe2f435431e9ffccc9c89367a371c595b
Predicate:  isinstance(op, str)   (Configuration.py:534 in CallableParameter.createBinaryOp)

The predicate gates whether createBinaryOp treats its `op` argument as a
FuncMap key (str) or as a raw callable (non-str).  Both branches are pinned
here using actual CallableParameter.createBinaryOp calls.

TRUE branch  (isinstance True):  op='Add'    -> name='Add',     result=lhs+rhs
FALSE branch (isinstance False): op=<lambda> -> name='CustomBinaryOp', result=callable(lhs,rhs)

CPU-only.  No GPU hardware required.
"""

import pytest
from Tensile.Configuration import CallableParameter

pytestmark = pytest.mark.unit


def _predicate_is_str(op) -> bool:
    """Mirror of the predicate at Configuration.py:534."""
    return isinstance(op, str)


class TestPredicateIsStr:
    """Pure helper tests for the isinstance(op, str) predicate (L534)."""

    def test_str_op_yields_true(self):
        """isinstance('Add', str) is True -- TRUE branch of L534."""
        assert _predicate_is_str("Add") is True

    def test_lambda_op_yields_false(self):
        """isinstance(lambda, str) is False -- FALSE branch of L534."""
        assert _predicate_is_str(lambda lhs, rhs: lhs + rhs) is False

    def test_funcmap_key_variations_all_true(self):
        """All canonical FuncMap key strings satisfy the predicate."""
        keys = [
            "And", "Or", "Lt", "LtE", "Eq", "NotEq", "Gt", "GtE",
            "Mult", "Pow", "Div", "FloorDiv", "Mod", "Add", "Sub",
            "BitAnd", "BitOr", "BitXor", "LShift", "RShift", "min", "max",
        ]
        for k in keys:
            assert _predicate_is_str(k) is True, f"Expected True for key {k!r}"

    def test_builtin_function_yields_false(self):
        """A built-in callable (not a str) yields False."""
        assert _predicate_is_str(int) is False

    def test_none_yields_false(self):
        """None is not a str."""
        assert _predicate_is_str(None) is False


class TestCreateBinaryOpStrBranch:
    """TRUE branch: op is a str key into FuncMap."""

    def test_add_str_name_is_add(self):
        """L534 TRUE: op='Add' -> name attribute is 'Add'."""
        result = CallableParameter.createBinaryOp(1, 2, "Add")
        assert result.name == "Add"

    def test_add_str_evaluates_correctly(self):
        """L534 TRUE: op='Add' -> binop(1, 2) == 3."""
        result = CallableParameter.createBinaryOp(1, 2, "Add")
        assert result() == 3

    def test_sub_str_name_is_sub(self):
        """L534 TRUE: op='Sub' -> name attribute is 'Sub'."""
        result = CallableParameter.createBinaryOp(10, 3, "Sub")
        assert result.name == "Sub"

    def test_sub_str_evaluates_correctly(self):
        """L534 TRUE: op='Sub' -> binop(10, 3) == 7."""
        result = CallableParameter.createBinaryOp(10, 3, "Sub")
        assert result() == 7

    def test_mult_str_evaluates_correctly(self):
        """L534 TRUE: op='Mult' -> binop(4, 5) == 20."""
        result = CallableParameter.createBinaryOp(4, 5, "Mult")
        assert result() == 20

    def test_invalid_str_raises_assertion(self):
        """L535 assertion: unknown str raises AssertionError."""
        with pytest.raises(AssertionError, match="Missing operation in funcMap"):
            CallableParameter.createBinaryOp(1, 2, "NotAKey")


class TestCreateBinaryOpCallableBranch:
    """FALSE branch: op is a callable (not a str)."""

    def test_lambda_name_is_custom(self):
        """L534 FALSE: op=lambda -> name attribute is 'CustomBinaryOp'."""
        custom = lambda lhs, rhs: lhs * rhs + 1
        result = CallableParameter.createBinaryOp(3, 4, custom)
        assert result.name == "CustomBinaryOp"

    def test_lambda_evaluates_correctly(self):
        """L534 FALSE: op=lambda lhs,rhs: lhs*rhs+1 -> binop(3, 4) == 13."""
        custom = lambda lhs, rhs: lhs * rhs + 1
        result = CallableParameter.createBinaryOp(3, 4, custom)
        assert result() == 13

    def test_builtin_min_as_callable(self):
        """L534 FALSE: passing min() directly -> name='CustomBinaryOp'."""
        result = CallableParameter.createBinaryOp(7, 3, min)
        assert result.name == "CustomBinaryOp"
        assert result() == 3

    def test_builtin_max_as_callable(self):
        """L534 FALSE: passing max() directly -> name='CustomBinaryOp'."""
        result = CallableParameter.createBinaryOp(7, 3, max)
        assert result.name == "CustomBinaryOp"
        assert result() == 7

    def test_bad_callable_raises_badfunc(self):
        """L547-549: callable that raises on (lhs,rhs) -> BadFunc propagated."""
        def bad_op(lhs, rhs):
            raise ValueError("bad")
        with pytest.raises(CallableParameter.BadFunc):
            CallableParameter.createBinaryOp(1, 2, bad_op)
