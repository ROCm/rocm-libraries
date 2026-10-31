# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The integer-expression evaluator and its constructors (rocke.core.intexpr).

One case table, two evaluators: every case is evaluated by the Python mirror
and checked against the expected outcome, and, when ``ROCKE_INTEXPR_CLI``
names the built ``rocke_intexpr`` program, the same table goes through the C++
evaluator and must produce the same lines. A caller may ask in one engine and
the other may enforce, so a value or an error text that differs between them
is a bug in one of them.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from rocke.core import intexpr
from rocke.core.intexpr import (
    IntExprError,
    difference_of,
    eval_intexpr,
    product_of,
    quotient_of,
    sum_of,
)

TWO_31 = 1 << 31
# 2^62, built by arithmetic because a number leaf may not exceed 2^53 - 1.
TWO_62 = {"mul": [TWO_31, TWO_31]}
INT64_MAX = {"add": [{"sub": [TWO_62, 1]}, TWO_62]}
INT64_MIN = {"sub": [{"sub": [0, TWO_62]}, TWO_62]}

ENV = {
    "spec": {"D": 6, "neg": -7},
    "var": {"i": 2},
    "param": {"M": 64, "ldc": 72},
    "spec_str": {"dtype": "f16"},
}

# (intexpr, expected outcome line). "value <n>" or "error <text>", the format
# the C++ program prints.
CASES = [
    # leaves
    (5, "value 5"),
    (-3, "value -3"),
    (4.0, "value 4"),
    ({"spec": "D"}, "value 6"),
    ({"var": "i"}, "value 2"),
    ({"param": "M"}, "value 64"),
    ({"spec": "x"}, "error unknown spec int 'x'"),
    ({"var": "x"}, "error unknown loop var 'x'"),
    ({"param": "N"}, "error unknown param 'N'"),
    ({"param": 3}, "error unknown param '?'"),
    ({"spec_str_eq": ["dtype", "f16"]}, "value 1"),
    ({"spec_str_eq": ["dtype", "f32"]}, "value 0"),
    ({"spec_str_eq": ["layout", "f16"]}, "value 0"),
    # arithmetic and comparisons
    ({"add": [{"spec": "D"}, {"param": "M"}]}, "value 70"),
    ({"sub": [{"spec": "D"}, {"param": "M"}]}, "value -58"),
    ({"mul": [{"param": "M"}, {"param": "ldc"}]}, "value 4608"),
    ({"div": [{"param": "ldc"}, {"spec": "D"}]}, "value 12"),
    ({"mod": [{"param": "M"}, {"spec": "D"}]}, "value 4"),
    ({"eq": [1, 1]}, "value 1"),
    ({"ne": [1, 1]}, "value 0"),
    ({"lt": [1, 2]}, "value 1"),
    ({"le": [2, 2]}, "value 1"),
    ({"gt": [1, 2]}, "value 0"),
    ({"ge": [{"spec": "neg"}, 0]}, "value 0"),
    ({"magic_shift": 7}, "value 3"),
    ({"magic_multiplier": 7}, "value 613566757"),
    ({"magic_shift": 0}, "error magic division needs 1 <= divisor < 2^31, got 0"),
    # the int64 range: its ends are reachable, one step past them is refused
    (INT64_MAX, "value 9223372036854775807"),
    (INT64_MIN, "value -9223372036854775808"),
    (
        {"add": [INT64_MAX, 1]},
        "error intexpr add overflows int64 (9223372036854775807, 1)",
    ),
    (
        {"sub": [INT64_MIN, 1]},
        "error intexpr sub overflows int64 (-9223372036854775808, 1)",
    ),
    (
        {"mul": [TWO_62, 2]},
        "error intexpr mul overflows int64 (4611686018427387904, 2)",
    ),
    ({"mul": [TWO_62, -2]}, "value -9223372036854775808"),
    # floor and truncating division agree only on non-negative operands
    ({"div": [{"spec": "D"}, 0]}, "error intexpr div by zero"),
    ({"mod": [{"spec": "D"}, 0]}, "error intexpr mod by zero"),
    (
        {"div": [{"spec": "neg"}, 2]},
        "error intexpr div needs non-negative operands, got (-7, 2)",
    ),
    (
        {"mod": [7, -2]},
        "error intexpr mod needs non-negative operands, got (7, -2)",
    ),
    # number leaves the JSON tree cannot hold exactly
    (2.5, "error intexpr number is not an integer"),
    ((1 << 53), "error intexpr number is not an integer"),
    (1e16, "error intexpr number is not an integer"),
    # malformed trees
    (True, "error bad intexpr"),
    ("D", "error bad intexpr"),
    ({"nope": 1}, "error bad intexpr"),
    ({"add": [1]}, "error bad intexpr 'add'"),
    # a failure deep in a tree is the tree's failure
    ({"add": [1, {"div": [1, 0]}]}, "error intexpr div by zero"),
]


def _python_outcome(expr) -> str:
    try:
        value = eval_intexpr(
            expr, ENV["var"], ENV["spec"], ENV["spec_str"], params=ENV["param"]
        )
    except IntExprError as e:
        return f"error {e}"
    return f"value {value}"


@pytest.mark.parametrize("expr,want", CASES, ids=[str(i) for i in range(len(CASES))])
def test_python_evaluator(expr, want):
    assert _python_outcome(expr) == want


def test_cpp_evaluator_prints_the_same_lines():
    """The C++ evaluator gives every case the same value or the same error text.

    ``ROCKE_INTEXPR_CLI`` names the built ``rocke_intexpr``: ``tests/run_all.py``
    resolves it from a configured build and the installed ``rocke_pytest`` ctest
    sets it. A value that names no file is a broken setup, so it fails instead
    of skipping.
    """
    executable = os.environ.get("ROCKE_INTEXPR_CLI")
    if not executable:
        pytest.skip("Set ROCKE_INTEXPR_CLI to the built rocke_intexpr")
    path = Path(executable).resolve()
    assert path.is_file(), f"ROCKE_INTEXPR_CLI names {path}, not a file"
    printed = subprocess.run(
        [str(path), "--eval"],
        input=json.dumps({"env": ENV, "cases": [c for c, _ in CASES]}),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert printed.stdout.splitlines() == [want for _, want in CASES]


def test_recipe_expander_uses_this_evaluator():
    """Moved, not copied: the recipe expander and launch planner call the one
    evaluator, so a fix here reaches every portable-IR caller."""
    from rocke.portable_ir.src import launch
    from rocke.portable_ir.utils import recipe_expand

    assert recipe_expand.eval_intexpr is intexpr.eval_intexpr
    assert launch.eval_intexpr is intexpr.eval_intexpr


class _Param:
    """A stand-in for a kernel parameter leaf: the constructors only carry it."""

    def __init__(self, name: str):
        self.name = name


M, N, LDC = _Param("M"), _Param("N"), _Param("ldc")


def _name(leaf) -> str:
    return leaf.name


def test_constructors_build_the_evaluated_tree():
    bound = sum_of(product_of(M, LDC), 64)
    tree = bound.tree(_name)
    assert tree == {"add": [{"mul": [{"param": "M"}, {"param": "ldc"}]}, 64]}
    assert bound.text(_name) == "sum_of(product_of(M, ldc), 64)"
    assert eval_intexpr(tree, {}, {}, {}, params={"M": 64, "ldc": 72}) == 4672


def test_variadic_terms_nest_left_to_right():
    assert product_of(M, N, LDC).tree(_name) == {
        "mul": [{"mul": [{"param": "M"}, {"param": "N"}]}, {"param": "ldc"}]
    }


def test_quotient_and_difference_flip_a_relation():
    """`sb >= M * ldc` stated on ldc reads `ldc <= sb // M`; `total >= a + b`
    stated on b reads `b <= total - a`."""
    sb, total, a = _Param("sb"), _Param("total"), _Param("a")
    assert quotient_of(sb, M).tree(_name) == {"div": [{"param": "sb"}, {"param": "M"}]}
    assert difference_of(total, a).tree(_name) == {
        "sub": [{"param": "total"}, {"param": "a"}]
    }
    assert difference_of(quotient_of(total, 4), a).text(_name) == (
        "difference_of(quotient_of(total, 4), a)"
    )


@pytest.mark.parametrize("term", [True, 1.5, "M", None])
def test_a_term_must_be_an_int_a_leaf_or_a_bound(term):
    with pytest.raises(TypeError, match=r"^sum_of: a term is an int"):
        sum_of(M, term)


def test_an_int_term_must_fit_the_stored_tree():
    with pytest.raises(ValueError, match=r"outside \+/-\(2\^53 - 1\)"):
        product_of(M, 1 << 53)


def test_sums_and_products_need_two_terms():
    with pytest.raises(
        TypeError, match=r"^product_of needs at least two terms, got 1$"
    ):
        product_of(M)
