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

"""Characterization tests for Configuration.py line 673.

Branch ID: 8226b3bb80ef12780d069187df1f3cf8965ca735
Predicate:  nodeType == "Call"   (Configuration.py:673 in ExpressionEvaluator.evaluate)
            where nodeType = type(node).__name__  (set at line 627)

ExpressionEvaluator.evaluate is a recursive AST visitor.  The elif at line 673
dispatches to a "Call" handler when the current ast node is a function call
(e.g. abs(x)).

TRUE branch  (nodeType == "Call"):   expression "abs(x)" -> inner node is Call
             -> evaluate enters the Call arm, tries to look up "abs" in funcMap,
             raises AssertionError "Missing operation in funcMap: abs".

FALSE branch (nodeType != "Call"):   expression "BenchmarkTaskSize > 0" -> inner
             node is Compare, never reaches L673, evaluates to True via Compare arm.

Dead-in-production note: all production addConstraint() calls use Compare/BoolOp
expressions, never bare function calls, so L673 is unreachable in normal usage.
Both branches are pinned here via actual ExpressionEvaluator.evaluate calls.

CPU-only.  No GPU hardware required.
"""

import ast

import pytest

from Tensile.Configuration import ExpressionEvaluator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: pure predicate mirroring L673
# ---------------------------------------------------------------------------

def _predicate_is_call(node) -> bool:
    """Mirror of the predicate at Configuration.py:673."""
    return type(node).__name__ == "Call"


class TestPredicateIsCall:
    """Pure predicate tests: nodeType == 'Call' (L673)."""

    def test_call_node_yields_true(self):
        """ast.Call node -> predicate True (TRUE branch of L673)."""
        tree = ast.parse("abs(x)", mode="exec")
        inner = tree.body[0].value  # Call node
        assert _predicate_is_call(inner) is True

    def test_compare_node_yields_false(self):
        """ast.Compare node -> predicate False (FALSE branch of L673)."""
        tree = ast.parse("BenchmarkTaskSize > 0", mode="exec")
        inner = tree.body[0].value  # Compare node
        assert _predicate_is_call(inner) is False

    def test_binop_node_yields_false(self):
        """ast.BinOp node -> predicate False."""
        tree = ast.parse("a + b", mode="exec")
        inner = tree.body[0].value  # BinOp node
        assert _predicate_is_call(inner) is False

    def test_boolop_node_yields_false(self):
        """ast.BoolOp node -> predicate False."""
        tree = ast.parse("a and b", mode="exec")
        inner = tree.body[0].value  # BoolOp node
        assert _predicate_is_call(inner) is False

    def test_name_node_yields_false(self):
        """ast.Name node -> predicate False."""
        tree = ast.parse("some_var", mode="exec")
        inner = tree.body[0].value  # Name node
        assert _predicate_is_call(inner) is False


# ---------------------------------------------------------------------------
# End-to-end pin: TRUE branch (nodeType == "Call")
# ---------------------------------------------------------------------------

class TestEvaluateCallBranchTrue:
    """TRUE branch: ExpressionEvaluator hits L673 when node is ast.Call."""

    def test_abs_call_enters_call_branch_raises_missing_func(self):
        """L673 TRUE: abs(x) -> Call node -> AssertionError Missing operation in funcMap: abs.

        abs is not in CallableParameter funcMap; the Call branch is entered,
        the func name abs is extracted, then createUnaryOp raises the assertion.
        This is the canonical witness that L673 is reachable.
        """
        ev = ExpressionEvaluator()
        tree = ast.parse("abs(x)", mode="exec")
        with pytest.raises(AssertionError, match="Missing operation in funcMap: abs"):
            ev.evaluate(tree, {"x": 2})

    def test_unknown_func_call_enters_call_branch(self):
        """L673 TRUE: foo(x) -> Call node -> AssertionError Missing operation in funcMap: foo.

        Any unary function call enters the Call branch at L673.
        """
        ev = ExpressionEvaluator()
        tree = ast.parse("foo(x)", mode="exec")
        with pytest.raises(AssertionError, match="Missing operation in funcMap: foo"):
            ev.evaluate(tree, {"x": 0})

    def test_inner_call_node_is_ast_call(self):
        """Confirm abs(x) inner node type is Call (underpins TRUE branch entry)."""
        tree = ast.parse("abs(x)", mode="exec")
        inner = tree.body[0].value
        assert type(inner).__name__ == "Call"


# ---------------------------------------------------------------------------
# End-to-end pin: FALSE branch (nodeType != "Call")
# ---------------------------------------------------------------------------

class TestEvaluateCallBranchFalse:
    """FALSE branch: ExpressionEvaluator does NOT hit L673 when node is not ast.Call."""

    def test_compare_never_enters_call_branch(self):
        """L673 FALSE: BenchmarkTaskSize > 0 -> Compare branch -> evaluates True.

        This is a real production addConstraint string from TensileBenchmarkCluster.py:283.
        It never reaches L673.
        """
        ev = ExpressionEvaluator()
        tree = ast.parse("BenchmarkTaskSize > 0", mode="exec")
        result = ev.evaluate(tree, {"BenchmarkTaskSize": 256})
        assert result is True

    def test_compare_false_case(self):
        """L673 FALSE: BenchmarkTaskSize > 0 with 0 -> evaluates False."""
        ev = ExpressionEvaluator()
        tree = ast.parse("BenchmarkTaskSize > 0", mode="exec")
        result = ev.evaluate(tree, {"BenchmarkTaskSize": 0})
        assert result is False

    def test_boolop_never_enters_call_branch(self):
        """L673 FALSE: a and b -> BoolOp branch -> evaluates correctly."""
        ev = ExpressionEvaluator()
        tree = ast.parse("a and b", mode="exec")
        result = ev.evaluate(tree, {"a": True, "b": True})
        assert result is True

    def test_inner_compare_node_is_not_call(self):
        """Confirm BenchmarkTaskSize > 0 inner node type is Compare, not Call."""
        tree = ast.parse("BenchmarkTaskSize > 0", mode="exec")
        inner = tree.body[0].value
        assert type(inner).__name__ == "Compare"
        assert type(inner).__name__ != "Call"
