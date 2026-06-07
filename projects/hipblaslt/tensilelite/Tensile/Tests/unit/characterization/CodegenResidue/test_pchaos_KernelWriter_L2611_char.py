################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test -- branch_id 4944b8f577187e45cae6f9454baa75d19009c23d

Predicate : not kernel["UseSubtileImpl"]
Site      : Tensile/KernelWriter.py:2611  (inside setupNewTile)
Solver    : z3 -- SAT  (solver-backed-under-assumptions)
Classification: solver-backed-under-assumptions

Derivation chain
----------------
  Solution.py:812: isgfx950 = state["ISA"] == IsaVersion(9,5,0)
  Solution.py:813: state["UseSubtileImpl"] = state["UseSubtileImpl"] and isgfx950

  So kernel["UseSubtileImpl"] == (raw_use_subtile_impl AND (ISA == IsaVersion(9,5,0))).
  The branch predicate at KernelWriter.py:2611 is the negation: NOT of that derived value.

  Truth table (raw x ISA):
    raw=False, ISA=(9,5,0)  -> predicate = True
    raw=False, ISA=(9,4,2)  -> predicate = True
    raw=True,  ISA=(9,5,0)  -> predicate = False  [sole False witness]
    raw=True,  ISA=(9,4,2)  -> predicate = True

Tests here
----------
  1. Pure-helper test  -- pin that subtile_branch_not_taken(use_subtile_impl, isa_is_gfx950)
     matches the NOT(raw AND isgfx950) logic for all four witness combinations from the
     truth table.
  2. Derivation-site pin -- verify Solution.py still derives state["UseSubtileImpl"]
     as the AND of the raw value and isgfx950 via AST inspection (CPU-only, no GPU).
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper -- mirrors Solution.py:812-813 derivation + KernelWriter.py:2611 predicate
# ---------------------------------------------------------------------------

def subtile_branch_not_taken(use_subtile_impl: bool, isa_is_gfx950: bool) -> bool:
    """Mirror of KernelWriter.py:2611 predicate `not kernel["UseSubtileImpl"]`.

    kernel["UseSubtileImpl"] is derived (Solution.py:813) as the raw yaml value ANDed
    with isgfx950 (= ISA == IsaVersion(9,5,0)). The branch at line 2611 is the negation.
    post: __return__ == (not (use_subtile_impl and isa_is_gfx950))
    """
    kernel_use_subtile_impl = use_subtile_impl and isa_is_gfx950
    return not kernel_use_subtile_impl


# ---------------------------------------------------------------------------
# TRUE witnesses (branch is taken: predicate == True)
# ---------------------------------------------------------------------------

def test_subtile_branch_not_taken_true_raw_false_gfx950():
    """z3 true witness: raw=False, ISA=(9,5,0) -> kernel val=False -> predicate=True.

    User did not request subtile; AND is False regardless of ISA -> not False = True.
    """
    assert subtile_branch_not_taken(use_subtile_impl=False, isa_is_gfx950=True) is True


def test_subtile_branch_not_taken_true_raw_false_non_gfx950():
    """z3 true witness: raw=False, ISA=(9,4,2) -> kernel val=False -> predicate=True.

    Not requested and not gfx950 -> both conditions fail -> not False = True.
    """
    assert subtile_branch_not_taken(use_subtile_impl=False, isa_is_gfx950=False) is True


def test_subtile_branch_not_taken_true_raw_true_non_gfx950():
    """z3 true witness: raw=True, ISA=(9,4,2) -> kernel val=False -> predicate=True.

    Requested but ISA != gfx950 -> derived value forced to False -> not False = True.
    """
    assert subtile_branch_not_taken(use_subtile_impl=True, isa_is_gfx950=False) is True


# ---------------------------------------------------------------------------
# FALSE witness (branch is NOT taken: predicate == False)
# ---------------------------------------------------------------------------

def test_subtile_branch_not_taken_false_raw_true_gfx950():
    """z3 false witness: raw=True, ISA=(9,5,0) -> kernel val=True -> predicate=False.

    Sole model making the predicate False: subtile requested AND target is gfx950.
    This is the ONLY input combination under which the if-block at KernelWriter.py:2611
    is skipped.
    """
    assert subtile_branch_not_taken(use_subtile_impl=True, isa_is_gfx950=True) is False


# ---------------------------------------------------------------------------
# Full truth table parametric sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,gfx950,expected", [
    (False, True,  True),   # raw=False, ISA=(9,5,0)
    (False, False, True),   # raw=False, ISA=(9,4,2)
    (True,  True,  False),  # raw=True,  ISA=(9,5,0)  <-- sole False
    (True,  False, True),   # raw=True,  ISA=(9,4,2)
])
def test_subtile_branch_not_taken_truth_table(raw, gfx950, expected):
    """Exhaustive truth table over raw{T,F} x isgfx950{T,F}; both polarities reachable."""
    assert subtile_branch_not_taken(use_subtile_impl=raw, isa_is_gfx950=gfx950) is expected


# ---------------------------------------------------------------------------
# Derivation-site pin -- AST inspection of Solution.py:812-813
# ---------------------------------------------------------------------------

def test_solution_usesubtileimpl_derivation_ast():
    """Pin that Solution.py still derives state['UseSubtileImpl'] as:
        isgfx950 = state['ISA'] == IsaVersion(9,5,0)
        state['UseSubtileImpl'] = state['UseSubtileImpl'] and isgfx950

    If the derivation changes, this test will fail and the char classification
    for KernelWriter.py:2611 must be re-evaluated.
    CPU-only: reads source, no GPU access.
    """
    import ast

    target_file = "Tensile/SolutionStructs/Solution.py"
    with open(target_file) as fh:
        source = fh.read()

    tree = ast.parse(source)

    # Look for: state["UseSubtileImpl"] = state["UseSubtileImpl"] and isgfx950
    # i.e. an Assign where:
    #   target is a Subscript: state["UseSubtileImpl"]
    #   value is a BoolOp (And) with:
    #     left: Subscript state["UseSubtileImpl"]
    #     right: Name("isgfx950")
    found_derivation = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        # Target must be state["UseSubtileImpl"]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "state"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "UseSubtileImpl"
        ):
            continue
        val = node.value
        # Value must be a BoolOp with And
        if not (isinstance(val, ast.BoolOp) and isinstance(val.op, ast.And)):
            continue
        # Must have exactly 2 values: state["UseSubtileImpl"] and isgfx950
        if len(val.values) != 2:
            continue
        lhs, rhs = val.values
        lhs_ok = (
            isinstance(lhs, ast.Subscript)
            and isinstance(lhs.value, ast.Name)
            and lhs.value.id == "state"
            and isinstance(lhs.slice, ast.Constant)
            and lhs.slice.value == "UseSubtileImpl"
        )
        rhs_ok = isinstance(rhs, ast.Name) and rhs.id == "isgfx950"
        if lhs_ok and rhs_ok:
            found_derivation = True
            break

    assert found_derivation, (
        "Solution.py no longer derives state['UseSubtileImpl'] = state['UseSubtileImpl'] and isgfx950; "
        "re-evaluate the solver-backed-under-assumptions classification for KernelWriter.py:2611 "
        "(branch_id 4944b8f577187e45cae6f9454baa75d19009c23d)."
    )
