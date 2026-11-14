#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# LibraryLogic unit tests for pruneSolutions data/exactWinners remap logic.
#
# Covers LibraryLogic.py:1425-1480 (pruneSolutions method) — the index remapping
# when keeping a subset of solutions or an empty set.
#
# The existing test (test_p4_librarylogic_char.py::test_prune_solutions_keeps_specified)
# only asserts numSolutions after pruning. It does NOT validate:
#   - data array remap: data[p*new_n+new_idx] = oldData[p*old_n+old_idx]
#   - exactWinners remap: winner indices updated to new dense indices
#   - exactWinners[size][0] == -1 when winner was pruned
#   - empty keepSolutions case: numSolutions == 0, data is empty
#
# An index-mapping regression would leave stale/misaligned data and exactWinners,
# producing wrong library logic with no error.
################################################################################

import array
import pytest
from copy import deepcopy
from unittest.mock import patch, MagicMock


def _mock_name(solution, splitGSU=False):
    """Mock getSolutionNameMin to avoid dependency on solution dict structure."""
    if isinstance(solution, dict):
        return f"sol_{solution.get('idx', 0)}"
    return f"sol_{solution}"


class _MockSolution(dict):
    """Minimal solution-like dict, hashable by identity."""
    def __init__(self, idx, tile0=64, tile1=64):
        super().__init__({
            "idx": idx,
            "MacroTile0": tile0,
            "MacroTile1": tile1,
            "SolutionIndex": idx,
        })
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return self is other


class _MockDataType:
    """Minimal DataType mock satisfying LibraryLogic.LogicAnalyzer interface."""
    def flopsPerMac(self):
        return 2
    def __repr__(self):
        return "s"
    def __str__(self):
        return "s"


class _MockProblemType(dict):
    """Dict-backed ProblemType mock with required LogicAnalyzer fields."""
    def __init__(self):
        super().__init__()
        self.update({
            "Index0": 0,
            "Index1": 1,
            "IndexUnroll": 3,
            "TotalIndices": 4,
            "NumIndicesLD": 0,
            "NumIndicesC": 3,
            "NumIndicesFree": 2,
            "NumIndicesBatch": 1,
            "NumIndicesSummation": 1,
            "IndicesFree": [0, 1],
            "IndicesBatch": [2],
            "IndicesSummation": [3],
            "IndexAssignmentsA": [3, 0, 2],
            "IndexAssignmentsB": [3, 1, 2],
            "IndexAssignmentsLD": [],
            "DataType": _MockDataType(),
            "OperationType": "GEMM",
            "TileAwareSelection": False,
            "GroupedGemm": False,
        })
    def __hash__(self):
        return hash(id(self))
    def __eq__(self, other):
        return self is other


class _MockProblemSizes:
    """Minimal ProblemSizes-like object."""
    def __init__(self):
        self.problems = []
        self.exacts = []
        self.ranges = []


def _build_minimal_analyzer(n_solutions=3, n_problems=4):
    """Build a minimal LogicAnalyzer with injected data for pruneSolutions testing.

    Returns:
        la: LogicAnalyzer instance
        initial_data: copy of data before pruning
        initial_exact_winners: copy of exactWinners before pruning
    """
    from tensilelite.LibraryLogic import LogicAnalyzer

    # Create mock objects
    problem_type = _MockProblemType()
    mock_ps = _MockProblemSizes()

    solutions = [_MockSolution(i, tile0=64*(i+1), tile1=64) for i in range(n_solutions)]
    solutions_list = [solutions]

    analysis_params = {"NumBestSolutions": 1, "NumBestSolutionsMinimum": 1}

    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getSolutionNameFull", _mock_name):
        la = LogicAnalyzer(
            problem_type, [mock_ps], solutions_list,
            [], analysis_params, splitGSU=False,
        )

    # Inject minimal data structure
    la.numSolutions = n_solutions
    la.solutions = solutions
    la.totalProblems = n_problems
    la.totalSize = la.totalProblems * la.numSolutions

    # Create data array with distinct values per (problem, solution) cell
    # data[p*n_solutions+s] = p*100 + s*10 (e.g., problem 0 solution 2 = 20)
    la.data = array.array('f', [0] * la.totalSize)
    for p in range(n_problems):
        for s in range(n_solutions):
            la.data[p * n_solutions + s] = float(p * 100 + s * 10)

    # Create exactWinners: map problem sizes to [winner_idx, gflops]
    # Set different winners for different problem sizes
    la.exactWinners = {}
    for p in range(n_problems):
        winner_idx = p % n_solutions  # Rotate: 0,1,2,0,1,2,...
        size_tuple = (64 + p * 64, 64, 1, 512)
        la.exactWinners[size_tuple] = [winner_idx, 10.0 + p]

    # Initialize solutionNames and solutionTiles (needed by pruneSolutions)
    la.solutionNames = [f"sol_{i}" for i in range(n_solutions)]
    la.solutionTiles = [f"{s['MacroTile0']}x{s['MacroTile1']}" for s in solutions]

    initial_data = deepcopy(la.data)
    initial_exact_winners = deepcopy(la.exactWinners)

    return la, initial_data, initial_exact_winners


def test_prune_solutions_remaps_data_correctly():
    """pruneSolutions correctly remaps data array when keeping subset of solutions.

    Covers LibraryLogic.py:1453-1460 — the data remap loop:
        for problemIndex in range(0, self.totalProblems):
            for newSolutionIdx in range(0, self.numSolutions):
                oldSolutionIdx = solutionMapNewToOld[newSolutionIdx]
                self.data[problemIndex*self.numSolutions+newSolutionIdx] =
                    oldData[problemIndex*oldNumSolutions+oldSolutionIdx]

    Critical: when keeping {0,2} of 3 solutions, the new data layout must be:
        data[p*2+0] = oldData[p*3+0]  (solution 0 stays at new index 0)
        data[p*2+1] = oldData[p*3+2]  (solution 2 moves to new index 1)

    Without this test, a wrong remap would pass silently (existing test only checks numSolutions).
    """
    la, initial_data, _ = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Keep solutions {0, 2}, pruning solution 1
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.pruneSolutions({0, 2})

    assert la.numSolutions == 2, "Should have 2 solutions after keeping {0,2}"
    assert len(la.data) == la.totalProblems * 2, "Data size should be totalProblems * 2"

    # Verify data remap for each problem
    for p in range(la.totalProblems):
        # Solution 0 (old idx 0) -> new idx 0
        expected_val_0 = float(p * 100 + 0 * 10)  # p*100 + 0
        actual_val_0 = la.data[p * 2 + 0]
        assert actual_val_0 == expected_val_0, (
            f"Problem {p}, new solution 0: expected {expected_val_0}, got {actual_val_0}"
        )

        # Solution 2 (old idx 2) -> new idx 1
        expected_val_2 = float(p * 100 + 2 * 10)  # p*100 + 20
        actual_val_1 = la.data[p * 2 + 1]
        assert actual_val_1 == expected_val_2, (
            f"Problem {p}, new solution 1 (was 2): expected {expected_val_2}, got {actual_val_1}"
        )


def test_prune_solutions_remaps_exact_winners():
    """pruneSolutions remaps exactWinners indices to new dense solution indices.

    Covers LibraryLogic.py:1462-1472 — the exactWinners remap loop:
        for problemSize in self.exactWinners:
            self.exactWinners[problemSize][0] = solutionMapOldToNew[self.exactWinners[problemSize][0]]

    Critical: when keeping {0,2} of 3 solutions:
        - Winner was solution 0 -> remains 0 (solutionMapOldToNew[0] = 0)
        - Winner was solution 1 -> becomes -1 (solutionMapOldToNew[1] = -1, pruned)
        - Winner was solution 2 -> becomes 1 (solutionMapOldToNew[2] = 1, remapped)

    Without this test, stale winner indices would point to wrong solutions after pruning.
    """
    la, _, initial_exact_winners = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Verify initial winners (rotates 0,1,2,0 for problems 0,1,2,3)
    assert initial_exact_winners[(64, 64, 1, 512)][0] == 0
    assert initial_exact_winners[(128, 64, 1, 512)][0] == 1
    assert initial_exact_winners[(192, 64, 1, 512)][0] == 2
    assert initial_exact_winners[(256, 64, 1, 512)][0] == 0

    # Keep solutions {0, 2}, pruning solution 1
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.pruneSolutions({0, 2})

    # Verify remapped winners
    # Problem 0: winner was 0 -> still 0
    assert la.exactWinners[(64, 64, 1, 512)][0] == 0, "Winner 0 should remain 0"

    # Problem 1: winner was 1 (pruned) -> becomes -1
    assert la.exactWinners[(128, 64, 1, 512)][0] == -1, "Pruned winner 1 should become -1"

    # Problem 2: winner was 2 -> becomes 1 (new dense index)
    assert la.exactWinners[(192, 64, 1, 512)][0] == 1, "Winner 2 should remap to 1"

    # Problem 3: winner was 0 -> still 0
    assert la.exactWinners[(256, 64, 1, 512)][0] == 0, "Winner 0 should remain 0"


def test_prune_solutions_empty_keep_set():
    """pruneSolutions handles empty keepSolutions set (prune all solutions).

    Covers LibraryLogic.py:1437-1444 — when keepSolutions is empty:
        - No solutions are kept: solutionMapNewToOld is []
        - All old solutions map to -1: solutionMapOldToNew = [-1, -1, -1]
        - numSolutions becomes 0
        - data array becomes empty

    Critical edge case: empty keepSolutions should produce a valid empty analyzer
    (numSolutions=0, len(data)=0), not crash or leave stale state.

    Without this test, an assertion or index error in the data remap loop could
    go unnoticed (existing test never tries empty keepSolutions).
    """
    la, _, _ = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    initial_num_solutions = la.numSolutions
    assert initial_num_solutions == 3

    # Prune all solutions (empty keep set)
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.pruneSolutions(set())

    assert la.numSolutions == 0, "Empty keepSolutions should result in numSolutions=0"
    assert len(la.solutions) == 0, "solutions list should be empty"
    assert len(la.data) == 0, "data array should be empty"
    assert la.totalSize == 0, "totalSize should be 0 (totalProblems * 0)"
    assert len(la.solutionNames) == 0, "solutionNames should be empty"
    assert len(la.solutionTiles) == 0, "solutionTiles should be empty"


def test_prune_solutions_single_keep():
    """pruneSolutions handles keeping a single solution (not necessarily index 0).

    Covers the case where only one solution is kept, but it's not solution 0.
    This validates that the remap correctly handles a single-element keepSolutions
    where the kept solution has a non-zero old index.
    """
    la, initial_data, initial_exact_winners = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Keep only solution 2 (prune 0 and 1)
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.pruneSolutions({2})

    assert la.numSolutions == 1, "Should have 1 solution after keeping {2}"
    assert len(la.data) == la.totalProblems * 1, "Data size should be totalProblems * 1"

    # Verify data remap: old solution 2 becomes new solution 0
    for p in range(la.totalProblems):
        expected_val = float(p * 100 + 2 * 10)  # Was solution 2 in old data
        actual_val = la.data[p * 1 + 0]
        assert actual_val == expected_val, (
            f"Problem {p}: expected {expected_val} (old sol 2), got {actual_val}"
        )

    # Verify exactWinners remap
    # Problem 0: winner was 0 (pruned) -> -1
    assert la.exactWinners[(64, 64, 1, 512)][0] == -1, "Pruned winner 0 should become -1"
    # Problem 1: winner was 1 (pruned) -> -1
    assert la.exactWinners[(128, 64, 1, 512)][0] == -1, "Pruned winner 1 should become -1"
    # Problem 2: winner was 2 (kept) -> 0 (only remaining solution)
    assert la.exactWinners[(192, 64, 1, 512)][0] == 0, "Winner 2 should remap to 0"
    # Problem 3: winner was 0 (pruned) -> -1
    assert la.exactWinners[(256, 64, 1, 512)][0] == -1, "Pruned winner 0 should become -1"


def test_prune_solutions_updates_solution_metadata():
    """pruneSolutions updates solutionNames and solutionTiles for kept solutions.

    Covers LibraryLogic.py:1445-1451 — rebuilding solutionNames and solutionTiles:
        self.solutionNames = []
        self.solutionTiles = []
        for solution in self.solutions:
            self.solutionNames.append(getSolutionNameMin(solution, self.splitGSU))
            self.solutionTiles.append("%ux%u"%(solution["MacroTile0"], solution["MacroTile1"]))

    Critical: the metadata arrays must stay in sync with the pruned solutions list.
    """
    la, _, _ = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Verify initial state
    assert len(la.solutionNames) == 3
    assert len(la.solutionTiles) == 3
    assert la.solutionNames == ["sol_0", "sol_1", "sol_2"]
    assert la.solutionTiles == ["64x64", "128x64", "192x64"]

    # Keep solutions {0, 2}
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.pruneSolutions({0, 2})

    # Verify metadata updated
    assert len(la.solutionNames) == 2, "solutionNames should have 2 entries"
    assert len(la.solutionTiles) == 2, "solutionTiles should have 2 entries"
    assert la.solutionNames == ["sol_0", "sol_2"], "solutionNames should be updated"
    assert la.solutionTiles == ["64x64", "192x64"], "solutionTiles should be updated"
    assert len(la.solutions) == 2, "solutions list should have 2 entries"


# ===========================================================================
# Tests for removeSolution method (LibraryLogic.py:1377-1418)
# ===========================================================================


def test_remove_solution_remaps_data_correctly():
    """removeSolution correctly compacts data array when removing a solution.

    Covers LibraryLogic.py:1397-1406 — the data compaction loop:
        for problemIndex in range(0, self.totalProblems):
            newSolutionIdx = 0
            for oldSolutionIdx in range(0, oldNumSolutions):
                if oldSolutionIdx != removeSolutionIdx:
                    self.data[problemIndex*self.numSolutions+newSolutionIdx] =
                        oldData[problemIndex*oldNumSolutions+oldSolutionIdx]
                    newSolutionIdx += 1

    Critical: when removing solution 1 of 3 solutions, the new data layout must be:
        data[p*2+0] = oldData[p*3+0]  (solution 0 stays at index 0)
        data[p*2+1] = oldData[p*3+2]  (solution 2 shifts to index 1)

    Without this test, a wrong column compaction would pass silently (existing test
    only checks numSolutions and totalSize).
    """
    la, initial_data, _ = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Remove solution 1 (middle solution)
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.removeSolution(1)

    assert la.numSolutions == 2, "Should have 2 solutions after removing 1"
    assert len(la.data) == la.totalProblems * 2, "Data size should be totalProblems * 2"

    # Verify data compaction for each problem
    for p in range(la.totalProblems):
        # Solution 0 (old idx 0) -> new idx 0
        expected_val_0 = float(p * 100 + 0 * 10)
        actual_val_0 = la.data[p * 2 + 0]
        assert actual_val_0 == expected_val_0, (
            f"Problem {p}, new solution 0: expected {expected_val_0}, got {actual_val_0}"
        )

        # Solution 2 (old idx 2) -> new idx 1
        expected_val_2 = float(p * 100 + 2 * 10)
        actual_val_1 = la.data[p * 2 + 1]
        assert actual_val_1 == expected_val_2, (
            f"Problem {p}, new solution 1 (was 2): expected {expected_val_2}, got {actual_val_1}"
        )


def test_remove_solution_decrements_winner_indices():
    """removeSolution decrements exactWinners indices for solutions after removed index.

    Covers LibraryLogic.py:1408-1411 — the exactWinners adjustment:
        for problemSize in self.exactWinners:
            if self.exactWinners[problemSize][0] >= removeSolutionIdx:
                self.exactWinners[problemSize][0] -= 1

    Critical: when removing solution 1 of 3 solutions:
        - Winner was solution 0 (< 1) -> stays 0 (not decremented)
        - Winner was solution 1 (== 1) -> becomes 0 (decremented by >= check)
        - Winner was solution 2 (> 1) -> becomes 1 (decremented)

    The `>=` comparison means a winner equal to removeSolutionIdx is decremented,
    effectively shifting it to the previous solution. This is the critical behavior
    this test documents.
    """
    la, _, initial_exact_winners = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Verify initial winners (rotates 0,1,2,0 for problems 0,1,2,3)
    assert initial_exact_winners[(64, 64, 1, 512)][0] == 0
    assert initial_exact_winners[(128, 64, 1, 512)][0] == 1
    assert initial_exact_winners[(192, 64, 1, 512)][0] == 2
    assert initial_exact_winners[(256, 64, 1, 512)][0] == 0

    # Remove solution 1
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.removeSolution(1)

    # Verify winner adjustments
    # Problem 0: winner was 0 (< 1) -> stays 0
    assert la.exactWinners[(64, 64, 1, 512)][0] == 0, "Winner 0 should stay 0"

    # Problem 1: winner was 1 (== 1, removed) -> becomes 0 (decremented by >= check)
    assert la.exactWinners[(128, 64, 1, 512)][0] == 0, (
        "Winner 1 (removed) should become 0 (shifted to previous solution)"
    )

    # Problem 2: winner was 2 (> 1) -> becomes 1
    assert la.exactWinners[(192, 64, 1, 512)][0] == 1, "Winner 2 should become 1"

    # Problem 3: winner was 0 (< 1) -> stays 0
    assert la.exactWinners[(256, 64, 1, 512)][0] == 0, "Winner 0 should stay 0"


def test_remove_solution_at_index_zero():
    """removeSolution handles removing solution at index 0 (first solution).

    Edge case: removing the first solution. Tests that:
        - Data compaction works correctly (all solutions shift left)
        - Winner indices are decremented correctly
        - Winner at index 0 becomes -1 (decremented from 0)
    """
    la, initial_data, _ = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Remove solution 0 (first solution)
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.removeSolution(0)

    assert la.numSolutions == 2, "Should have 2 solutions after removing 0"

    # Verify data compaction: old solutions 1,2 become new solutions 0,1
    for p in range(la.totalProblems):
        # Old solution 1 -> new solution 0
        expected_val_1 = float(p * 100 + 1 * 10)
        actual_val_0 = la.data[p * 2 + 0]
        assert actual_val_0 == expected_val_1, (
            f"Problem {p}, new solution 0 (was 1): expected {expected_val_1}, got {actual_val_0}"
        )

        # Old solution 2 -> new solution 1
        expected_val_2 = float(p * 100 + 2 * 10)
        actual_val_1 = la.data[p * 2 + 1]
        assert actual_val_1 == expected_val_2, (
            f"Problem {p}, new solution 1 (was 2): expected {expected_val_2}, got {actual_val_1}"
        )

    # Verify winner adjustments: all winners >= 0 are decremented
    # Problem 0: winner was 0 (== 0, removed) -> becomes -1
    assert la.exactWinners[(64, 64, 1, 512)][0] == -1, "Winner 0 (removed) should become -1"

    # Problem 1: winner was 1 (> 0) -> becomes 0
    assert la.exactWinners[(128, 64, 1, 512)][0] == 0, "Winner 1 should become 0"

    # Problem 2: winner was 2 (> 0) -> becomes 1
    assert la.exactWinners[(192, 64, 1, 512)][0] == 1, "Winner 2 should become 1"

    # Problem 3: winner was 0 (== 0, removed) -> becomes -1
    assert la.exactWinners[(256, 64, 1, 512)][0] == -1, "Winner 0 (removed) should become -1"


def test_remove_solution_at_last_index():
    """removeSolution handles removing the last solution.

    Edge case: removing solution at index numSolutions-1. Tests that:
        - Data compaction works (last column removed)
        - Winners < last index are unchanged
        - Winner at last index is decremented to last-1
    """
    la, initial_data, _ = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Remove solution 2 (last solution)
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.removeSolution(2)

    assert la.numSolutions == 2, "Should have 2 solutions after removing last"

    # Verify data compaction: solutions 0,1 remain
    for p in range(la.totalProblems):
        # Solution 0 stays at index 0
        expected_val_0 = float(p * 100 + 0 * 10)
        assert la.data[p * 2 + 0] == expected_val_0

        # Solution 1 stays at index 1
        expected_val_1 = float(p * 100 + 1 * 10)
        assert la.data[p * 2 + 1] == expected_val_1

    # Verify winner adjustments
    # Problem 0: winner was 0 (< 2) -> stays 0
    assert la.exactWinners[(64, 64, 1, 512)][0] == 0

    # Problem 1: winner was 1 (< 2) -> stays 1
    assert la.exactWinners[(128, 64, 1, 512)][0] == 1

    # Problem 2: winner was 2 (== 2, removed) -> becomes 1
    assert la.exactWinners[(192, 64, 1, 512)][0] == 1, (
        "Winner 2 (removed, last solution) should become 1"
    )

    # Problem 3: winner was 0 (< 2) -> stays 0
    assert la.exactWinners[(256, 64, 1, 512)][0] == 0


def test_remove_solution_single_remaining():
    """removeSolution handles removing the only remaining solution.

    Critical edge case: when numSolutions==1, removing it leaves numSolutions==0
    with an empty data array and solutions list.
    """
    la, _, _ = _build_minimal_analyzer(n_solutions=1, n_problems=4)

    assert la.numSolutions == 1
    assert len(la.data) == 4  # 4 problems * 1 solution

    # Remove the only solution
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.removeSolution(0)

    assert la.numSolutions == 0, "Should have 0 solutions after removing the only one"
    assert len(la.solutions) == 0, "solutions list should be empty"
    assert len(la.data) == 0, "data array should be empty"
    assert la.totalSize == 0, "totalSize should be 0"
    assert len(la.solutionNames) == 0, "solutionNames should be empty"
    assert len(la.solutionTiles) == 0, "solutionTiles should be empty"


def test_remove_solution_updates_metadata():
    """removeSolution updates solutionNames and solutionTiles for remaining solutions.

    Covers LibraryLogic.py:1389-1395 — rebuilding solution metadata:
        self.solutionNames = []
        self.solutionTiles = []
        for solution in self.solutions:
            self.solutionNames.append(getSolutionNameMin(solution, self.splitGSU))
            self.solutionTiles.append("%ux%u"%(solution["MacroTile0"], solution["MacroTile1"]))
    """
    la, _, _ = _build_minimal_analyzer(n_solutions=3, n_problems=4)

    # Verify initial state
    assert la.solutionNames == ["sol_0", "sol_1", "sol_2"]
    assert la.solutionTiles == ["64x64", "128x64", "192x64"]

    # Remove solution 1
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", _mock_name):
        la.removeSolution(1)

    # Verify metadata updated
    assert len(la.solutionNames) == 2, "solutionNames should have 2 entries"
    assert len(la.solutionTiles) == 2, "solutionTiles should have 2 entries"
    assert la.solutionNames == ["sol_0", "sol_2"], "solutionNames should be updated"
    assert la.solutionTiles == ["64x64", "192x64"], "solutionTiles should be updated"
    assert len(la.solutions) == 2, "solutions list should have 2 entries"
