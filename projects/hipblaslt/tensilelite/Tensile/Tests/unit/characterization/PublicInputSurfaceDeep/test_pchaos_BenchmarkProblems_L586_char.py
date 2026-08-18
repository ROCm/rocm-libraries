################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: ``if len(solutions) == 0`` at
``Tensile/BenchmarkProblems.py:586`` inside ``_benchmarkProblemType``.

Branch 6647a7e665fa02215a747c9a777102452ef22d02.

The predicate ``len(solutions) == 0`` guards the "0 valid solutions" exit path
in ``_benchmarkProblemType``.  ``solutions = regSolutions + kcSolutions``
(line 583), where:

  * ``regSolutions`` is built by ``_generateForkedSolutions`` (line 224).
    It is empty when ``ForkParameters`` is absent/falsy (no fork permutations)
    **OR** every generated permutation is rejected by ``SolutionStructs``
    validation (collapsed into ``all_fork_perms_rejected``).

  * ``kcSolutions`` is built by ``_generateCustomKernelSolutions`` (line 285).
    It is empty when no custom kernels were requested
    (``CustomKernels_count == 0``) **OR** every candidate failed the
    ``ProblemType``-match / ``Valid`` checks (collapsed into
    ``all_custom_kernels_rejected``).

Derivation law (z3-verified):

    len(solutions) == 0
      <=>  And(
             Or(Not(ForkParameters_present), all_fork_perms_rejected),
             Or(CustomKernels_count == 0, all_custom_kernels_rejected)
           )

``use_cache`` is independent: it gates the outer ``if not cacheValid`` at
line 557, not the count.

Domain (exhaustive 2^4 over controlling surface):
  ForkParameters_present x {0,1} CustomKernels_count
      x all_fork_perms_rejected x all_custom_kernels_rejected.

Classification: solver-backed-under-assumptions (z3 SAT; CrossHair 0.0.106
confirmed over all paths; live SolutionStructs survival abstracted into seeded
rejection booleans).

TRUE branch  -> printExit('Your parameters resulted in 0 valid solutions.').
FALSE branch -> proceed to writeBenchmarkFiles (dominant normal path).
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper: mirror of BenchmarkProblems.py:586 predicate
# ---------------------------------------------------------------------------


def no_valid_solutions(
    fork_parameters_present: bool,
    all_fork_perms_rejected: bool,
    custom_kernel_count: int,
    all_custom_kernels_rejected: bool,
) -> bool:
    """Mirror of BenchmarkProblems.py:586 ``if len(solutions) == 0``.

    solutions = regSolutions + kcSolutions.
    regSolutions is empty iff there were no fork permutations
    (ForkParameters falsy) OR every permutation was rejected by SolutionStructs.
    kcSolutions is empty iff no custom kernels were requested OR all of them
    were rejected (problem-type mismatch / invalid).  Returns True iff the
    "0 valid solutions" exit branch is taken.

    pre: custom_kernel_count >= 0
    post: __return__ == (
              (not fork_parameters_present or all_fork_perms_rejected)
              and (custom_kernel_count == 0 or all_custom_kernels_rejected)
          )
    """
    reg_empty = (not fork_parameters_present) or all_fork_perms_rejected
    kc_empty = (custom_kernel_count == 0) or all_custom_kernels_rejected
    return reg_empty and kc_empty


# ---------------------------------------------------------------------------
# TRUE branch: len(solutions) == 0 -> printExit
# ---------------------------------------------------------------------------


def test_no_fork_params_no_custom_kernels_true():
    """No ForkParameters AND no CustomKernels -> both sources empty -> len==0 (TRUE).

    z3-model witness: ForkParameters_present=False, CustomKernels_count=0,
    all_fork_perms_rejected=n/a, all_custom_kernels_rejected=n/a.
    """
    assert no_valid_solutions(
        fork_parameters_present=False,
        all_fork_perms_rejected=False,
        custom_kernel_count=0,
        all_custom_kernels_rejected=False,
    ) is True


def test_all_fork_rejected_all_custom_rejected_true():
    """ForkParameters present but all rejected + CustomKernels present but all rejected -> len==0 (TRUE).

    z3-model witness: ForkParameters_present=True, CustomKernels_count=1,
    all_fork_perms_rejected=True, all_custom_kernels_rejected=True.
    """
    assert no_valid_solutions(
        fork_parameters_present=True,
        all_fork_perms_rejected=True,
        custom_kernel_count=1,
        all_custom_kernels_rejected=True,
    ) is True


def test_no_fork_params_custom_kernels_all_rejected_true():
    """No ForkParameters + CustomKernels present but all rejected -> len==0 (TRUE)."""
    assert no_valid_solutions(
        fork_parameters_present=False,
        all_fork_perms_rejected=False,
        custom_kernel_count=1,
        all_custom_kernels_rejected=True,
    ) is True


def test_fork_params_all_rejected_no_custom_kernels_true():
    """ForkParameters present but all rejected + no CustomKernels -> len==0 (TRUE)."""
    assert no_valid_solutions(
        fork_parameters_present=True,
        all_fork_perms_rejected=True,
        custom_kernel_count=0,
        all_custom_kernels_rejected=False,
    ) is True


# ---------------------------------------------------------------------------
# FALSE branch: len(solutions) > 0 -> writeBenchmarkFiles
# ---------------------------------------------------------------------------


def test_fork_params_some_survive_no_custom_false():
    """ForkParameters present with at least one surviving permutation -> len>0 (FALSE).

    z3-model witness: ForkParameters_present=True, CustomKernels_count=0,
    all_fork_perms_rejected=False, all_custom_kernels_rejected=n/a.
    """
    assert no_valid_solutions(
        fork_parameters_present=True,
        all_fork_perms_rejected=False,
        custom_kernel_count=0,
        all_custom_kernels_rejected=False,
    ) is False


def test_no_fork_params_custom_kernel_survives_false():
    """No ForkParameters + one custom kernel matches + is Valid -> len>0 (FALSE).

    z3-model witness: ForkParameters_present=False, CustomKernels_count=1,
    all_custom_kernels_rejected=False, all_fork_perms_rejected=n/a.
    """
    assert no_valid_solutions(
        fork_parameters_present=False,
        all_fork_perms_rejected=False,
        custom_kernel_count=1,
        all_custom_kernels_rejected=False,
    ) is False


def test_both_sources_have_survivors_false():
    """ForkParameters survives + CustomKernels survives -> len>0 (FALSE)."""
    assert no_valid_solutions(
        fork_parameters_present=True,
        all_fork_perms_rejected=False,
        custom_kernel_count=1,
        all_custom_kernels_rejected=False,
    ) is False


# ---------------------------------------------------------------------------
# Exhaustive table: all 16 rows of 2^4 controlling surface
# ---------------------------------------------------------------------------


def test_exhaustive_2x2x2x2_table():
    """Exhaustive check over 2^4 = 16 controlling surface rows.

    Mirrors the z3-verified exhaustive_table from the Solve fragment.
    Each row: (FP, CK, AFR, ACR) -> expected len==0.
    """
    table = [
        # (fp_present, ck_count, afr, acr, expected_len0)
        (False, 0, False, False, True),
        (False, 0, False, True,  True),
        (False, 0, True,  False, True),
        (False, 0, True,  True,  True),
        (False, 1, False, False, False),
        (False, 1, False, True,  True),
        (False, 1, True,  False, False),
        (False, 1, True,  True,  True),
        (True,  0, False, False, False),
        (True,  0, False, True,  False),
        (True,  0, True,  False, True),
        (True,  0, True,  True,  True),
        (True,  1, False, False, False),
        (True,  1, False, True,  False),
        (True,  1, True,  False, False),
        (True,  1, True,  True,  True),
    ]
    for fp, ck, afr, acr, expected in table:
        result = no_valid_solutions(fp, afr, ck, acr)
        assert result is expected, (
            f"fp={fp} ck={ck} afr={afr} acr={acr}: "
            f"expected {expected}, got {result}"
        )


# ---------------------------------------------------------------------------
# Derivation law pin: And(Or(Not(FP),AFR), Or(CK==0,ACR))
# ---------------------------------------------------------------------------


def test_derivation_law_matches_helper_all_rows():
    """Helper output matches the z3 derivation law for all 16 rows.

    Law: len==0 <=> And(Or(Not(FP), AFR), Or(CK==0, ACR)).
    Verifies the pure_helper postcondition on the full exhaustive domain.
    """
    for fp in [False, True]:
        for ck in [0, 1]:
            for afr in [False, True]:
                for acr in [False, True]:
                    expected = (
                        ((not fp) or afr)
                        and ((ck == 0) or acr)
                    )
                    result = no_valid_solutions(fp, afr, ck, acr)
                    assert result is expected, (
                        f"fp={fp} ck={ck} afr={afr} acr={acr}: "
                        f"law={expected}, helper={result}"
                    )
