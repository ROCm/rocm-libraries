################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################
"""Unit tests for fork-parameter permutation generation in BenchmarkStructs.

``constructForkPermutations`` (the length) and ``constructLazyForkPermutations``
(the generator) were changed so that when a parameter *group* sets a value for a
parameter that is also a fork parameter, the group value wins and the fork
parameter is no longer expanded over its own values. Previously that produced
duplicate permutations and an inflated ``len()``.

The pre-change generator/length are copied below verbatim and used as oracles to
prove the new code is backward compatible whenever no group overwrites a fork
parameter.
"""

import itertools

import pytest

pytestmark = pytest.mark.unit

from Tensile.BenchmarkStructs import constructForkPermutations, constructLazyForkPermutations


# --------------------------------------------------------------------------- #
# Pre-change implementation, copied verbatim (origin/develop) as an oracle.
# --------------------------------------------------------------------------- #
def _old_constructLazyForkPermutations(forkParams, paramGroups):
    params_list = []
    for name, values in forkParams.items():
        params_list.append((name, values, False))

    for i, group in enumerate(paramGroups):
        params_list.append((f"_group{i}", group, True))

    params_list_reversed = list(reversed(params_list))
    values_reversed = [values for _, values, _ in params_list_reversed]
    for combination in itertools.product(*values_reversed):
        permutation = {}
        for (name, _, isgroup), value in zip(params_list, reversed(combination)):
            if isgroup:
                permutation.update(value)
            else:
                permutation[name] = value
        yield permutation


def _old_total_permutations(forkParams, paramGroups):
    totalPermutations = 1
    for values in forkParams.values():
        totalPermutations *= len(values)
    for group in paramGroups:
        totalPermutations *= len(group)
    return totalPermutations


def _canon(perm):
    """Hashable canonical form of a permutation dict (for set/duplicate checks)."""
    return tuple(sorted(perm.items()))


FORK = {"A": [1, 2], "B": ["x", "y", "z"]}


@pytest.mark.parametrize(
    "forkParams, paramGroups",
    [
        pytest.param(FORK, [], id="no-group"),
        pytest.param(FORK, [[{"C": 7}, {"C": 8}]], id="non-overlapping-group"),
        pytest.param(
            {"A": [1, 2], "B": ["x", "y"]},
            [[{"C": 0}, {"C": 1}], [{"D": 9}]],
            id="two-non-overlapping-groups",
        ),
    ],
)
def test_matches_previous_behavior_without_overwrite(forkParams, paramGroups):
    """No group overrides a fork param -> identical output (same set AND order)."""
    new = list(constructLazyForkPermutations(forkParams, paramGroups))
    old = list(_old_constructLazyForkPermutations(forkParams, paramGroups))

    assert new == old  # same permutations, same order

    # Length is unchanged from the old formula and equals the yielded count.
    length = len(constructForkPermutations(forkParams, paramGroups))
    assert length == _old_total_permutations(forkParams, paramGroups)
    assert length == len(new)


def test_group_overwrite_pins_param_with_no_duplicates():
    """A group that sets a fork param pins it to the group value (no dup expansion)."""
    forkParams = {"A": [1, 2], "B": ["x", "y", "z"]}
    paramGroups = [[{"A": 99}, {"A": 88}]]  # group overrides fork param "A"

    perms = list(constructLazyForkPermutations(forkParams, paramGroups))

    # "A" only takes the group values, never its own fork values 1/2.
    assert {p["A"] for p in perms} == {99, 88}
    assert all(p["A"] not in (1, 2) for p in perms)
    # "B" (not overridden) still varies fully.
    assert {p["B"] for p in perms} == {"x", "y", "z"}
    # No duplicate permutations are produced.
    assert len({_canon(p) for p in perms}) == len(perms)
    # 2 group settings x 3 values of B = 6 (NOT 2*3*2 = 12).
    assert len(perms) == 6
    assert len(constructForkPermutations(forkParams, paramGroups)) == len(perms)

    # The old implementation over-expanded "A": same count (12) with duplicates,
    # whose de-duplicated set is exactly what the new code yields.
    old = list(_old_constructLazyForkPermutations(forkParams, paramGroups))
    assert _old_total_permutations(forkParams, paramGroups) == 12
    assert len({_canon(p) for p in old}) < len(old)
    assert {_canon(p) for p in perms} == {_canon(p) for p in old}


@pytest.mark.parametrize(
    "forkParams, paramGroups",
    [
        pytest.param(FORK, [], id="no-group"),
        pytest.param(FORK, [[{"C": 7}, {"C": 8}]], id="non-overlapping-group"),
        pytest.param(FORK, [[{"A": 99}, {"A": 88}]], id="overwrite-group"),
        pytest.param(
            {"A": [1, 2], "B": ["x", "y"]},
            [[{"A": 9}], [{"D": 0}, {"D": 1}]],
            id="overwrite-plus-extra-group",
        ),
    ],
)
def test_len_matches_number_yielded(forkParams, paramGroups):
    """``len(constructForkPermutations(...))`` equals the number actually yielded."""
    perms = constructForkPermutations(forkParams, paramGroups)
    yielded = list(constructLazyForkPermutations(forkParams, paramGroups))

    assert len(perms) == len(yielded)
