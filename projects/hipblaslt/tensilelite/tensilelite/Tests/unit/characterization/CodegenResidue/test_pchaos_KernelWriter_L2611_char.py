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
  Solution.py: isgfx950   = isa[:2] == (9, 5)
  Solution.py: isgfx1250  = isa[:2] == (12, 5)
  Solution.py: state["UseSubtileImpl"] = state["UseSubtileImpl"] and (isgfx950 or isgfx1250)

  So kernel["UseSubtileImpl"] == (raw_use_subtile_impl AND (isgfx950 OR isgfx1250)).
  The branch predicate at KernelWriter.py:2611 is the negation: NOT of that derived value.

  Truth table (raw x (isgfx950 or isgfx1250)):
    raw=False, subtile-arch=True   -> predicate = True
    raw=False, subtile-arch=False  -> predicate = True
    raw=True,  subtile-arch=True   -> predicate = False  [sole False witness]
    raw=True,  subtile-arch=False  -> predicate = True

Tests here
----------
  1. Pure-helper test  -- pin that subtile_branch_not_taken(use_subtile_impl, isa_is_subtile_arch)
     matches the NOT(raw AND subtile-arch) logic for all four witness combinations from the
     truth table.
  2. Derivation-site pin -- verify Solution.py still derives state["UseSubtileImpl"]
     as the AND of the raw value and (isgfx950 or isgfx1250) via AST inspection
     (CPU-only, no GPU).
"""

import pytest

from char_paths import resolve_tensile_path

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper -- mirrors Solution.py UseSubtileImpl derivation + KernelWriter.py:2611 predicate
# ---------------------------------------------------------------------------

def subtile_branch_not_taken(use_subtile_impl: bool, isa_is_subtile_arch: bool) -> bool:
    """Mirror of KernelWriter.py:2611 predicate `not kernel["UseSubtileImpl"]`.

    kernel["UseSubtileImpl"] is derived in Solution.py as the raw yaml value ANDed
    with (isgfx950 or isgfx1250) -- i.e. the ISA is a subtile-capable arch. The branch
    at line 2611 is the negation.
    post: __return__ == (not (use_subtile_impl and isa_is_subtile_arch))
    """
    kernel_use_subtile_impl = use_subtile_impl and isa_is_subtile_arch
    return not kernel_use_subtile_impl


# ---------------------------------------------------------------------------
# TRUE witnesses (branch is taken: predicate == True)
# ---------------------------------------------------------------------------

def test_subtile_branch_not_taken_true_raw_false_subtile_arch():
    """z3 true witness: raw=False, subtile-arch=True -> kernel val=False -> predicate=True.

    User did not request subtile; AND is False regardless of ISA -> not False = True.
    """
    assert subtile_branch_not_taken(use_subtile_impl=False, isa_is_subtile_arch=True) is True


def test_subtile_branch_not_taken_true_raw_false_non_subtile_arch():
    """z3 true witness: raw=False, subtile-arch=False -> kernel val=False -> predicate=True.

    Not requested and not a subtile arch -> both conditions fail -> not False = True.
    """
    assert subtile_branch_not_taken(use_subtile_impl=False, isa_is_subtile_arch=False) is True


def test_subtile_branch_not_taken_true_raw_true_non_subtile_arch():
    """z3 true witness: raw=True, subtile-arch=False -> kernel val=False -> predicate=True.

    Requested but ISA is not gfx950/gfx1250 -> derived value forced to False -> not False = True.
    """
    assert subtile_branch_not_taken(use_subtile_impl=True, isa_is_subtile_arch=False) is True


# ---------------------------------------------------------------------------
# FALSE witness (branch is NOT taken: predicate == False)
# ---------------------------------------------------------------------------

def test_subtile_branch_not_taken_false_raw_true_subtile_arch():
    """z3 false witness: raw=True, subtile-arch=True -> kernel val=True -> predicate=False.

    Sole model making the predicate False: subtile requested AND target is a subtile
    arch (gfx950 or gfx1250). This is the ONLY input combination under which the if-block
    at KernelWriter.py:2611 is skipped.
    """
    assert subtile_branch_not_taken(use_subtile_impl=True, isa_is_subtile_arch=True) is False


# ---------------------------------------------------------------------------
# Full truth table parametric sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,subtile_arch,expected", [
    (False, True,  True),   # raw=False, subtile-arch (gfx950/gfx1250)
    (False, False, True),   # raw=False, non-subtile arch
    (True,  True,  False),  # raw=True,  subtile-arch  <-- sole False
    (True,  False, True),   # raw=True,  non-subtile arch
])
def test_subtile_branch_not_taken_truth_table(raw, subtile_arch, expected):
    """Exhaustive truth table over raw{T,F} x subtile-arch{T,F}; both polarities reachable."""
    assert subtile_branch_not_taken(use_subtile_impl=raw, isa_is_subtile_arch=subtile_arch) is expected


# ---------------------------------------------------------------------------
# Derivation-site pin -- AST inspection of the UseSubtileImpl derivation
# ---------------------------------------------------------------------------

def test_solution_usesubtileimpl_derivation_ast():
    """Pin that Solution.py still derives state['UseSubtileImpl'] as:
        state['UseSubtileImpl'] = state['UseSubtileImpl'] and (isgfx950 or isgfx1250)

    The gate was widened from `isgfx950` alone to `(isgfx950 or isgfx1250)`; this
    matcher pins the widened ANDed-with-(OR-of-arch-flags) form. If the derivation
    changes again, this test will fail and the char classification for
    KernelWriter.py:2611 must be re-evaluated.
    CPU-only: reads source, no GPU access.
    """
    import ast

    target_file = resolve_tensile_path("Tensile/SolutionStructs/Solution.py")
    with open(target_file) as fh:
        source = fh.read()

    tree = ast.parse(source)

    # Look for:
    #   state["UseSubtileImpl"] = state["UseSubtileImpl"] and (isgfx950 or isgfx1250)
    # i.e. an Assign where:
    #   target is a Subscript: state["UseSubtileImpl"]
    #   value is a BoolOp (And) with exactly 2 values:
    #     left:  Subscript state["UseSubtileImpl"]
    #     right: BoolOp (Or) of Name("isgfx950"), Name("isgfx1250")
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
        # Must have exactly 2 values: state["UseSubtileImpl"] and (isgfx950 or isgfx1250)
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
        # RHS must be an OR over the subtile-arch flags isgfx950, isgfx1250.
        rhs_ok = (
            isinstance(rhs, ast.BoolOp)
            and isinstance(rhs.op, ast.Or)
            and {n.id for n in rhs.values if isinstance(n, ast.Name)}
            == {"isgfx950", "isgfx1250"}
        )
        if lhs_ok and rhs_ok:
            found_derivation = True
            break

    assert found_derivation, (
        "Solution.py no longer derives state['UseSubtileImpl'] = "
        "state['UseSubtileImpl'] and (isgfx950 or isgfx1250); "
        "re-evaluate the solver-backed-under-assumptions classification for KernelWriter.py:2611 "
        "(branch_id 4944b8f577187e45cae6f9454baa75d19009c23d)."
    )
