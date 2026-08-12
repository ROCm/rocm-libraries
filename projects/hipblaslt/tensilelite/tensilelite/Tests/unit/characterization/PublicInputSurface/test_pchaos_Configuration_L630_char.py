################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: the nodeType == "Module" dispatch
in ExpressionEvaluator.evaluate (Tensile/Configuration.py:630).

Branch 05506103d2678e4391d5a18d052fcc09b0d5981f. The predicate is a
runtime nodeType dispatch inside the recursive AST walker:

    nodeType = type(node).__name__
    if nodeType == "Module":          # L630 -- this branch
        return self.evaluate(node.body[0], namesContext)

TRUE branch  -- nodeType is "Module": the root ast.Module node
                encountered on the first (top-level) call to evaluate().
                Every valid expression string produces an ast.Module root,
                so the top-level call is a structural tautology.
FALSE branch -- nodeType is anything else ("Expr", "Compare", "BinOp", ...):
                encountered in all recursive calls into the module body.

These tests pin ACTUAL observed behavior; classification: runtime-dependent.
"""

import ast

import pytest

from Tensile.Configuration import ExpressionEvaluator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper: mirror of the L630 dispatch
# ---------------------------------------------------------------------------

def is_module_node(nodeType: str) -> bool:
    """Mirror of Configuration.py:630 dispatch. True iff nodeType == "Module"."""
    return nodeType == "Module"


class TestIsModuleNodePureHelper:
    """Seeded-enum witnesses (Z3-proven, runtime-confirmed)."""

    def test_module_returns_true(self):
        assert is_module_node("Module") is True

    def test_expr_returns_false(self):
        assert is_module_node("Expr") is False

    def test_other_node_types_return_false(self):
        for nt in ("Compare", "BinOp", "BoolOp", "UnaryOp", "Call", "Name", "Constant"):
            assert is_module_node(nt) is False


class TestAstNodeTypeAtTraversalPositions:
    """Pin actual node-type values at each AST tree level."""

    def test_root_of_comparison_is_module(self):
        tree = ast.parse("a == 1", mode="exec")
        assert type(tree).__name__ == "Module"

    def test_body0_of_comparison_is_expr(self):
        tree = ast.parse("a == 1", mode="exec")
        assert type(tree.body[0]).__name__ == "Expr"

    def test_body0_value_of_comparison_is_compare(self):
        tree = ast.parse("a == 1", mode="exec")
        assert type(tree.body[0].value).__name__ == "Compare"

    def test_root_of_binop_is_module(self):
        tree = ast.parse("a + b", mode="exec")
        assert type(tree).__name__ == "Module"

    def test_body0_value_of_binop_is_binop(self):
        tree = ast.parse("a + b", mode="exec")
        assert type(tree.body[0].value).__name__ == "BinOp"

    def test_structural_tautology_all_valid_exprs_root_is_module(self):
        expressions = ["a == 1", "a + b", "x > 5 and y < 3", "-a"]
        for expr in expressions:
            tree = ast.parse(expr, mode="exec")
            assert type(tree).__name__ == "Module"


class TestExpressionEvaluatorModuleBranch:
    """Pin ExpressionEvaluator.evaluate() behavior at the Module entry (CPU-only)."""

    def _eval(self, expr_str, ctx):
        evaluator = ExpressionEvaluator()
        tree = ast.parse(expr_str, mode="exec")
        return evaluator.evaluate(tree, ctx)

    def test_module_branch_taken_comparison_true(self):
        result = self._eval("a == 1", {"a": 1})
        actual = result() if callable(result) else result
        assert actual is True

    def test_module_branch_taken_comparison_false(self):
        result = self._eval("a == 1", {"a": 99})
        actual = result() if callable(result) else result
        assert actual is False

    def test_module_branch_taken_arithmetic(self):
        result = self._eval("a + b", {"a": 3, "b": 4})
        actual = result() if callable(result) else result
        assert actual == 7

    def test_module_branch_taken_inequality(self):
        result = self._eval("x > 5", {"x": 10})
        actual = result() if callable(result) else result
        assert actual is True

    def test_module_branch_raises_on_multiple_statements(self):
        # L632: assert len(node.body) == 1 -> AssertionError on multi-statement
        import ast as _ast
        tree = _ast.parse('a = 1' + chr(10) + 'b = 2', mode='exec')
        evaluator = ExpressionEvaluator()
        with pytest.raises(AssertionError):
            evaluator.evaluate(tree, {})
