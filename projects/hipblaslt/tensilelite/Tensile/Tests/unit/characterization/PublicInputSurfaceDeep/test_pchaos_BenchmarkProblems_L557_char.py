################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

# Characterization of Tensile/BenchmarkProblems.py:557 [if]
#   branch_id: 099093bf09abd54f42b409be3afe7454cfaf8eaf
#   function: _benchmarkProblemType
#   predicate: not cacheValid
#     true_branch  -> recompile / generate new solutions (dominant path)
#     false_branch -> skip recompile, use cached code objects
#
# Classification: solver-backed-under-assumptions.
# cacheValid is a derived local initialized False (L541) and set True only when:
#   (a) useCache is True  (--use-cache CLI flag, Tensile.py:503 argparse store_true)
#   AND
#   (b) _loadCacheIfMatches/_loadLegacyCacheIfMatches returns a non-None matchCO,
#       which happens only when cache.yaml exists on disk AND _cacheDataMatches(params).
#
# Derivation law (z3-exhaustive over 2x3 domain):
#   cacheValid == (useCache AND filesystem_cache_state == present_matching)
#   not cacheValid == NOT(useCache AND filesystem_cache_state == present_matching)
#
# Exhaustive table (6 domain points):
#   useCache=False, missing        -> cacheValid=False -> predicate=True  (recompile)
#   useCache=False, present_match  -> cacheValid=False -> predicate=True  (recompile)
#   useCache=False, present_mismat -> cacheValid=False -> predicate=True  (recompile)
#   useCache=True,  missing        -> cacheValid=False -> predicate=True  (recompile)
#   useCache=True,  present_match  -> cacheValid=True  -> predicate=False (use cache)
#   useCache=True,  present_mismat -> cacheValid=False -> predicate=True  (recompile)

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper: faithful model of BenchmarkProblems.py:541-557 derivation
#
# post: __return__ == (not (use_cache and cache_present_and_matching))
# ---------------------------------------------------------------------------
def solution_generation_runs(use_cache: bool, cache_present_and_matching: bool) -> bool:
    """Mirror of BenchmarkProblems.py:557 `if not cacheValid`.

    cacheValid becomes True only when --use-cache was passed AND a cache.yaml
    on disk both exists and matches the benchmarkStep params.  The recompile
    branch is taken (returns True) otherwise.

    post: __return__ == (not (use_cache and cache_present_and_matching))
    """
    cache_valid = use_cache and cache_present_and_matching
    return not cache_valid


# ---------------------------------------------------------------------------
# Pure-helper tests: exhaustive over the 2x3 seeded domain collapsed to 2x2
# (missing and present_mismatching both map to cache_present_and_matching=False)
# ---------------------------------------------------------------------------

def test_no_cache_flag_missing_fs_recompiles():
    # useCache=False, filesystem=missing -> cacheValid=False -> recompile
    assert solution_generation_runs(use_cache=False, cache_present_and_matching=False) is True


def test_no_cache_flag_present_match_recompiles():
    # useCache=False, filesystem=present_matching -> cache never consulted -> recompile
    assert solution_generation_runs(use_cache=False, cache_present_and_matching=True) is True


def test_cache_flag_missing_fs_recompiles():
    # useCache=True, filesystem=missing -> matchCO=None -> cacheValid=False -> recompile
    assert solution_generation_runs(use_cache=True, cache_present_and_matching=False) is True


def test_cache_flag_present_mismatching_recompiles():
    # useCache=True, filesystem=present_mismatching -> matchCO=None -> cacheValid=False -> recompile
    # (present_mismatching collapses to cache_present_and_matching=False)
    assert solution_generation_runs(use_cache=True, cache_present_and_matching=False) is True


def test_cache_flag_present_matching_skips_recompile():
    # useCache=True, filesystem=present_matching -> matchCO not None -> cacheValid=True -> no recompile
    assert solution_generation_runs(use_cache=True, cache_present_and_matching=True) is False


# ---------------------------------------------------------------------------
# Derivation law equivalence: exhaustive 2x3 -> 2x2 collapse is lossless
# (missing and present_mismatching both yield cache_present_and_matching=False)
# ---------------------------------------------------------------------------

def test_exhaustive_table_all_six_domain_points():
    """Pin the full 6-point exhaustive table from the Solve fragment."""
    table = [
        # (use_cache, cache_pam, expected_predicate)
        (False, False, True),   # missing
        (False, True,  True),   # present_matching but flag absent
        (False, False, True),   # present_mismatching (collapses to False)
        (True,  False, True),   # missing
        (True,  True,  False),  # present_matching AND flag -> cache hit
        (True,  False, True),   # present_mismatching
    ]
    for use_cache, cache_pam, expected in table:
        result = solution_generation_runs(use_cache, cache_pam)
        assert result is expected, (
            f"solution_generation_runs(use_cache={use_cache}, "
            f"cache_present_and_matching={cache_pam}) "
            f"-> {result}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Derivation law postcondition cross-check
# ---------------------------------------------------------------------------

def test_postcondition_holds_for_all_2x2_inputs():
    """For all (use_cache, cache_pam) combos, result == not(use_cache and cache_pam)."""
    for use_cache in (False, True):
        for cache_pam in (False, True):
            expected = not (use_cache and cache_pam)
            result = solution_generation_runs(use_cache, cache_pam)
            assert result is expected, (
                f"postcondition violated: use_cache={use_cache}, "
                f"cache_pam={cache_pam}: got {result}, expected {expected}"
            )
