################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test -- branch_id 3a433f9e15d67650bc2250f006f451f96af7633f

Predicate : kernel["HalfPLRA"]   (bare truthiness on dict value)
Site      : Tensile/KernelWriter.py:4072  (inside _loopBody)
Solver    : z3 -- SAT  (sat-bounded; HalfPLR seeded domain {0,1,2,3})
Classification: fully-static

Derivation chain
----------------
  Solution.py:2279: state["HalfPLRA"] = bool(HalfPLR & 0x01)
  where HalfPLR = state["HalfPLR"]  (int solution parameter, line 2278).
  The predicate at KernelWriter.py:4072 is a bare truthiness test on this
  precomputed bool dict entry. True iff HalfPLR is odd (low bit set).

Tests here
----------
  1. Pure-helper test  -- pin that half_plr_a(half_plr) == bool(half_plr & 0x01)
     for all four witnesses in the seeded domain {0,1,2,3}.
  2. Derivation-site pin -- verify Solution.py:2279 still performs the
     expected bitwise-AND derivation via AST inspection (no GPU, CPU-only).
"""

import pytest

from char_paths import resolve_tensile_path

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper -- mirrors Solution.py:2279 derivation
# ---------------------------------------------------------------------------

def half_plr_a(half_plr: int) -> bool:
    """Mirror of Solution.py:2279: state['HalfPLRA'] = bool(HalfPLR & 0x01).
    True iff HalfPLR has its low bit set (i.e. HalfPLR is odd).
    Seeded domain: int {0, 1, 2, 3}.
    """
    return bool(half_plr & 0x01)


# --- TRUE witness: HalfPLR=1 (z3-proven) ---

def test_half_plra_true_witness_halfplr_1():
    """z3 true witness: HalfPLR=1 -> HalfPLRA=True -> predicate enters branch."""
    assert half_plr_a(1) is True


# --- TRUE witness: HalfPLR=3 (python-confirmed) ---

def test_half_plra_true_witness_halfplr_3():
    """Additional true witness: HalfPLR=3 -> HalfPLRA=True (low bit set)."""
    assert half_plr_a(3) is True


# --- FALSE witness: HalfPLR=0 (z3-proven) ---

def test_half_plra_false_witness_halfplr_0():
    """z3 false witness: HalfPLR=0 -> HalfPLRA=False -> predicate skips branch."""
    assert half_plr_a(0) is False


# --- FALSE witness: HalfPLR=2 (python-confirmed) ---

def test_half_plra_false_witness_halfplr_2():
    """Additional false witness: HalfPLR=2 -> HalfPLRA=False (low bit clear)."""
    assert half_plr_a(2) is False


# --- Full truth table over seeded domain ---

def test_half_plra_full_truth_table():
    """Pin the complete truth table for HalfPLR in {0,1,2,3}.
    Odd values -> True; even values -> False.
    """
    expected = {0: False, 1: True, 2: False, 3: True}
    actual = {v: half_plr_a(v) for v in range(4)}
    assert actual == expected


# ---------------------------------------------------------------------------
# Derivation-site pin -- Solution.py:2279 AST check
# ---------------------------------------------------------------------------

def test_halfplra_derivation_site_is_bitwise_and():
    """
    Pin that Tensile/SolutionStructs/Solution.py:2279 still derives HalfPLRA
    as bool(HalfPLR & 0x01) via AST inspection.

    If the derivation changes, this test will fail and the char classification
    for KernelWriter.py:4072 must be re-evaluated.
    CPU-only: reads source, no GPU access.
    """
    import ast
    import textwrap

    target_file = resolve_tensile_path("Tensile/SolutionStructs/Solution.py")
    with open(target_file) as fh:
        source = fh.read()

    tree = ast.parse(source)

    found_derivation = False
    for node in ast.walk(tree):
        # Match: state["HalfPLRA"] = bool(HalfPLR & 0x01)
        # i.e. an Assign where:
        #   target is a Subscript: state["HalfPLRA"]
        #   value is a Call to bool() wrapping a BinOp (BitAnd) of
        #   Name("HalfPLR") & Constant(1)
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
            and target.slice.value == "HalfPLRA"
        ):
            continue
        val = node.value
        if not (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Name)
            and val.func.id == "bool"
            and len(val.args) == 1
        ):
            continue
        binop = val.args[0]
        if not (
            isinstance(binop, ast.BinOp)
            and isinstance(binop.op, ast.BitAnd)
            and isinstance(binop.left, ast.Name)
            and binop.left.id == "halfPLR"
            and isinstance(binop.right, ast.Constant)
            and binop.right.value == 1
        ):
            continue
        found_derivation = True
        break

    assert found_derivation, (
        "Solution.py no longer derives state['HalfPLRA'] = bool(halfPLR & 0x01); "
        "re-evaluate the fully-static classification for KernelWriter.py:4072 "
        "(branch_id 3a433f9e15d67650bc2250f006f451f96af7633f)."
    )
