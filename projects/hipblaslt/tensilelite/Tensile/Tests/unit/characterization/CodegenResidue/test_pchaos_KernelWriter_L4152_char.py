################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id 6c1a009494a8fc7621b802253026a798f5e94f76

Predicate : kernel["HalfPLRB"]   (bare dict lookup, truthy boolean)
Site      : Tensile/KernelWriter.py:4152
Solver    : exhaustive enumeration — SAT (both polarities reachable)

Derivation
----------
HalfPLRB is derived from HalfPLR via:

    state["HalfPLRB"] = bool(halfPLR & 0x02)   # Solution.py:2280

Domain of HalfPLR: [0, 1, 2, 3]  (ValidParameters.py:1048)

Truth table (exhaustive):
  HalfPLR=0 -> HalfPLRB=False  (0x00 & 0x02 = 0)
  HalfPLR=1 -> HalfPLRB=False  (0x01 & 0x02 = 0)
  HalfPLR=2 -> HalfPLRB=True   (0x02 & 0x02 = 2)
  HalfPLR=3 -> HalfPLRB=True   (0x03 & 0x02 = 2)

Tests here:
  1. Predicate semantics — pure derivation helper, pinned witnesses.
  2. Domain exhaustion — all four valid HalfPLR values match the truth table.
  3. Derivation pin — confirm Solution.py:2280 still uses `bool(halfPLR & 0x02)`.
"""

import ast
import inspect
import textwrap

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: mirrors the derivation and predicate
# ---------------------------------------------------------------------------

def _derive_halfplrb(half_plr: int) -> bool:
    """Mirror of Solution.py:2280: state['HalfPLRB'] = bool(halfPLR & 0x02)"""
    return bool(half_plr & 0x02)


def _pred(kernel_halfplrb: bool) -> bool:
    """Mirror of KernelWriter.py:4152: if kernel['HalfPLRB']:"""
    return bool(kernel_halfplrb)


# ---------------------------------------------------------------------------
# Witness tests — pin both polarities (true/false branch)
# ---------------------------------------------------------------------------

def test_halfplrb_true_witness_halfplr2():
    """True witness: HalfPLR=2 -> HalfPLRB=True -> predicate enters branch."""
    halfplrb = _derive_halfplrb(2)
    assert halfplrb is True
    assert _pred(halfplrb) is True


def test_halfplrb_true_witness_halfplr3():
    """True witness: HalfPLR=3 -> HalfPLRB=True -> predicate enters branch."""
    halfplrb = _derive_halfplrb(3)
    assert halfplrb is True
    assert _pred(halfplrb) is True


def test_halfplrb_false_witness_halfplr0():
    """False witness: HalfPLR=0 -> HalfPLRB=False -> predicate skips branch."""
    halfplrb = _derive_halfplrb(0)
    assert halfplrb is False
    assert _pred(halfplrb) is False


def test_halfplrb_false_witness_halfplr1():
    """False witness: HalfPLR=1 -> HalfPLRB=False -> predicate skips branch."""
    halfplrb = _derive_halfplrb(1)
    assert halfplrb is False
    assert _pred(halfplrb) is False


# ---------------------------------------------------------------------------
# Domain exhaustion — full valid parameter domain [0, 1, 2, 3]
# ---------------------------------------------------------------------------

_TRUTH_TABLE = {0: False, 1: False, 2: True, 3: True}


@pytest.mark.parametrize("half_plr,expected", list(_TRUTH_TABLE.items()))
def test_halfplrb_truth_table_exhaustive(half_plr, expected):
    """Exhaustive enumeration over HalfPLR valid domain confirms truth table."""
    assert _derive_halfplrb(half_plr) is expected


# ---------------------------------------------------------------------------
# Derivation pin — Solution.py still uses bool(halfPLR & 0x02) for HalfPLRB
# ---------------------------------------------------------------------------

def test_halfplrb_derivation_pin_in_solution_py():
    """
    Solution.py:2280 must assign state['HalfPLRB'] = bool(halfPLR & 0x02).
    Pin the actual source so any future change is caught here.
    """
    from Tensile.SolutionStructs.Solution import Solution  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(Solution.assignDerivedParameters))
    tree = ast.parse(source)

    found_halfplrb_assignment = False
    for node in ast.walk(tree):
        # Match: state["HalfPLRB"] = bool(halfPLR & 0x02)
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "state"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "HalfPLRB"
        ):
            continue
        # Value must be bool(halfPLR & 0x02) — check it's a bool() call with a BitAnd
        val = node.value
        if not (isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "bool"):
            continue
        if len(val.args) != 1:
            continue
        arg = val.args[0]
        if not (
            isinstance(arg, ast.BinOp)
            and isinstance(arg.op, ast.BitAnd)
            and isinstance(arg.right, ast.Constant)
            and arg.right.value == 0x02
        ):
            continue
        found_halfplrb_assignment = True
        break

    assert found_halfplrb_assignment, (
        "Solution.assignDerivedParameters no longer assigns "
        "state['HalfPLRB'] = bool(halfPLR & 0x02); "
        "re-verify the derivation for KernelWriter.py:4152 "
        "(branch_id 6c1a009494a8fc7621b802253026a798f5e94f76)."
    )
