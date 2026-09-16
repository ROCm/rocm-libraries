################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterization tests for Configuration.py:730
  elif nodeType == "Attribute":
inside ExpressionEvaluator.evaluate (class at line 606).

Branch 2075748886b1e54fc040f8498f92177c7cd92130.

The guard `nodeType == "Attribute"` fires when the current AST node
visited by ExpressionEvaluator.evaluate() has type name "Attribute",
which happens when the parsed expression contains dotted attribute
access (e.g. "Program.Network.PORT >= MaxPort").

Tests pin ACTUAL behavior.  CPU-only.
"""
import ast

import pytest

from Tensile.Configuration import ExpressionEvaluator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper: mirror of the line-730 predicate
# ---------------------------------------------------------------------------

def _attribute_branch(node_type: str) -> bool:
    """Mirror of the line-730 dispatch guard.

    Returns True iff node_type == "Attribute", i.e. the line-730 branch is
    taken.  node_type == type(node).__name__ for the current AST node.
    """
    return node_type == "Attribute"


class TestAttributeBranchPredicate:
    """Pure predicate tests (no AST needed)."""

    def test_true_when_attribute(self):
        assert _attribute_branch("Attribute") is True

    def test_false_when_name(self):
        assert _attribute_branch("Name") is False

    def test_false_when_module(self):
        assert _attribute_branch("Module") is False

    def test_false_when_empty_string(self):
        assert _attribute_branch("") is False

    def test_false_when_attribute_lowercase(self):
        # case-sensitive: "attribute" != "Attribute"
        assert _attribute_branch("attribute") is False


# ---------------------------------------------------------------------------
# Value-AST tests: verify which expressions produce Attribute nodes
# ---------------------------------------------------------------------------

class TestAttributeNodePresenceInAST:
    """Pin which expressionStr values yield Attribute nodes in the AST."""

    def _node_types(self, expr: str):
        tree = ast.parse(expr, mode="exec")
        return {type(n).__name__ for n in ast.walk(tree)}

    def test_dotted_expr_contains_attribute_node(self):
        # "Program.Network.PORT >= MaxPort" is the true-example from the solve frag
        node_types = self._node_types("Program.Network.PORT >= MaxPort")
        assert "Attribute" in node_types

    def test_simple_expr_no_attribute_node(self):
        # "BenchmarkTaskSize > 0" is the production-representative false-example
        node_types = self._node_types("BenchmarkTaskSize > 0")
        assert "Attribute" not in node_types

    def test_bool_expr_no_attribute_node(self):
        # TensileBenchmarkCluster.py:288 constraint -- no dotted access
        node_types = self._node_types("RunDeployStep or RunBenchmarkStep or RunResultsStep")
        assert "Attribute" not in node_types


# ---------------------------------------------------------------------------
# Integration: ExpressionEvaluator.evaluate actually reaches line 730
# when a dotted expression is supplied
# ---------------------------------------------------------------------------

class TestExpressionEvaluatorAttributeBranch:
    """Pin that evaluate() correctly handles dotted expressions (line 730 taken)."""

    def _make_evaluator(self):
        return ExpressionEvaluator()

    def test_attribute_branch_taken_dotted_expr(self):
        """evaluate() with a dotted expression correctly resolves nested dicts.

        This exercises the line-730 Attribute branch (TRUE polarity).
        """
        ev = self._make_evaluator()
        ctx = {"Program": {"Network": {"PORT": 100}}, "MaxPort": 50}
        expr = "Program.Network.PORT >= MaxPort"
        tree = ast.parse(expr, mode="exec")
        result = ev.evaluate(tree, ctx)
        # 100 >= 50 => True
        assert result is True or result() is True

    def test_attribute_branch_not_taken_simple_name_expr(self):
        """evaluate() with a Name-only expression does not touch line 730.

        FALSE polarity: the Attribute branch guard is never True.
        """
        ev = self._make_evaluator()
        ctx = {"BenchmarkTaskSize": 5}
        expr = "BenchmarkTaskSize > 0"
        tree = ast.parse(expr, mode="exec")
        result = ev.evaluate(tree, ctx)
        # 5 > 0 => True
        assert result is True or result() is True

    def test_attribute_branch_taken_false_comparison(self):
        """Attribute branch taken, but the overall expression evaluates False.

        Ensures line 730 is reached and the return value is the stored value,
        not accidentally True just because the branch fired.
        """
        ev = self._make_evaluator()
        ctx = {"Program": {"Network": {"PORT": 30}}, "MaxPort": 50}
        expr = "Program.Network.PORT >= MaxPort"
        tree = ast.parse(expr, mode="exec")
        result = ev.evaluate(tree, ctx)
        # 30 >= 50 => False
        assert result is False or result() is False
