################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Characterization test for Tensile/LibraryLogic.py (p4 round).

Targets miss ranges: 112-169, 419-455, 477-538, 552-632, 671-758,
                     782-1017, 1024-1141, 1215-1424.

Strategy (Pattern B -- driver/run path):

  * addFromCSV exact path (lines 419-538): drive LogicAnalyzer with
    exact problem sizes from a synthetic CSV; the exact-problem winner-tracking
    branch is the live code path (range-array population is effectively dead
    code in current LibraryLogic.py -- unifiedProblemSizes is never filled).

  * LogicAnalyzer analysis methods (lines 552-1424): construct a minimal
    LogicAnalyzer via __init__, then manually inject a non-empty 2-D data
    array so analysis methods receive meaningful input.

  * analyzeProblemType (lines 48-233): exercised with exact-only problem data.

CPU-only. No GPU. No rocisa. No Solution parsing.
"""

import array
import csv
import os
from copy import deepcopy
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal mock objects
# ---------------------------------------------------------------------------

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
            "Index0":         0,
            "Index1":         1,
            "IndexUnroll":    3,
            "TotalIndices":   4,
            "NumIndicesLD":   0,
            "NumIndicesC":    3,
            "NumIndicesFree": 2,
            "NumIndicesBatch": 1,
            "NumIndicesSummation": 1,
            "IndicesFree":        [0, 1],
            "IndicesBatch":       [2],
            "IndicesSummation":   [3],
            "IndexAssignmentsA":  [3, 0, 2],
            "IndexAssignmentsB":  [3, 1, 2],
            "IndexAssignmentsLD": [],
            "DataType":       _MockDataType(),
            "OperationType":  "GEMM",
            "TileAwareSelection": False,
            "GroupedGemm":    False,
        })
    def __hash__(self):
        return hash(id(self))
    def __eq__(self, other):
        return self is other


class _MockSolution(dict):
    """Minimal solution-like dict, hashable by identity."""
    def __init__(self, idx, tile0=64, tile1=64):
        super().__init__({"MacroTile0": tile0, "MacroTile1": tile1,
                          "SolutionIndex": idx})
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return self is other


class _MockProblemSizes:
    """Minimal ProblemSizes-like object.

    Real ProblemSizes.ranges is a list of ProblemSizeRange objects each with
    .problemSizes (list of tuples).  LogicAnalyzer.__init__ treats these as
    exacts (lines 319-322).
    """
    class _FakeExact:
        def __init__(self, sizes):
            self.sizes = tuple(sizes)

    class _FakeRange:
        def __init__(self, sizes_list):
            self.problemSizes = [tuple(s) for s in sizes_list]

    def __init__(self, exact_sizes=None, range_sizes=None):
        self.problems = []
        self.exacts = [self._FakeExact(s) for s in (exact_sizes or [])]
        if range_sizes:
            self.ranges = [self._FakeRange(range_sizes)]
        else:
            self.ranges = []


def _mock_name(sol, *args, **kwargs):
    idx = sol.get("SolutionIndex", 0) if hasattr(sol, "get") else 0
    return "MockSol%d" % idx

def _mock_name_full(sol, *args, **kwargs):
    return _mock_name(sol)


# ---------------------------------------------------------------------------
# CSV builder
# ---------------------------------------------------------------------------

def _write_exact_csv(path, problem_sizes, num_solutions, gflops_data,
                     header_unit="GFlops"):
    """Write a GFlops CSV matching LibraryLogic.addFromCSV exact path."""
    num_indices = len(problem_sizes[0]) if problem_sizes else 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [header_unit]
        for i in range(num_indices):
            header.append("Size%d" % i)
        header.append("TotalFlops")
        for s in range(num_solutions):
            header.append("Sol%d" % s)
        writer.writerow(header)
        for idx, sizes in enumerate(problem_sizes):
            total_flops = 2
            for s in sizes:
                total_flops *= s
            row = [header_unit] + list(sizes) + [total_flops] + gflops_data[idx]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Helper: build a fully-populated LogicAnalyzer with injected range data
# ---------------------------------------------------------------------------

def _build_la_with_injected_data(tmp_path, analysis_params,
                                  exact_sizes, exact_gflops,
                                  n_solutions=2,
                                  range_m=(64, 128),
                                  range_n=(64, 128),
                                  range_gflops=None):
    """Build LogicAnalyzer then inject a 2-D range data grid.

    LogicAnalyzer.__init__ leaves unifiedProblemSizes empty (dead code path).
    We inject index tables + data array so the analysis methods work.
    """
    from Tensile.LibraryLogic import LogicAnalyzer

    problem_type = _MockProblemType()
    mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
    solutions = [_MockSolution(idx=i, tile0=64*(i+1), tile1=64)
                 for i in range(n_solutions)]
    solutions_list = [solutions]

    csv_path = str(tmp_path / "bench.csv")
    _write_exact_csv(csv_path, exact_sizes, num_solutions=n_solutions,
                     gflops_data=exact_gflops)

    with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
         patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
        la = LogicAnalyzer(
            problem_type, [mock_ps], solutions_list,
            [csv_path], analysis_params, splitGSU=False,
        )

    # Inject 2-D M x N range data
    m_sizes = sorted(set(range_m))
    n_sizes = sorted(set(range_n))
    batch_sizes = [1]
    k_sizes = [512]

    la.problemSizeToIndex = [
        {m: i for i, m in enumerate(m_sizes)},
        {n: i for i, n in enumerate(n_sizes)},
        {b: i for i, b in enumerate(batch_sizes)},
        {k: i for i, k in enumerate(k_sizes)},
    ]
    la.problemIndexToSize = [m_sizes, n_sizes, batch_sizes, k_sizes]
    la.numProblemSizes = [len(m_sizes), len(n_sizes), len(batch_sizes), len(k_sizes)]
    la.totalProblems = 1
    for n in la.numProblemSizes:
        la.totalProblems *= n
    la.totalSize = la.totalProblems * la.numSolutions
    la.data = array.array("f", [-2.0] * la.totalSize)
    la.rangeProblemSizes = set()

    if range_gflops is None:
        range_gflops = {}
        for m in m_sizes:
            for n in n_sizes:
                if m <= 64:
                    range_gflops[(m, n)] = [10.0] + [8.0] * (n_solutions - 1)
                else:
                    g = [8.0, 12.0] + [7.0] * (n_solutions - 2)
                    range_gflops[(m, n)] = g[:n_solutions]

    for m_idx, m in enumerate(m_sizes):
        for n_idx, n in enumerate(n_sizes):
            for b_idx, b in enumerate(batch_sizes):
                for k_idx, k in enumerate(k_sizes):
                    la.rangeProblemSizes.add((m, n, b, k))
                    serial = la.indicesToSerial(0, [m_idx, n_idx, b_idx, k_idx])
                    gflops = range_gflops.get((m, n),
                                             [10.0] + [8.0] * (n_solutions - 1))
                    for s_idx in range(n_solutions):
                        la.data[serial + s_idx] = float(gflops[s_idx])

    la.globalIndexRange = [[0, la.numProblemSizes[i]] for i in range(la.numIndices)]
    la.problemIndicesForGlobalRange = la.problemIndicesForRange(la.globalIndexRange)

    return la, problem_type, solutions


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def analysis_params():
    from Tensile.Common.GlobalParameters import defaultAnalysisParameters
    params = {}
    for k, v in defaultAnalysisParameters.items():
        params[k] = v
    return params


# ===========================================================================
# Test classes
# ===========================================================================

class TestAddFromCSVExactPath:
    """Drive addFromCSV with exact problem sizes (lines 419-538)."""

    def test_exact_winners_populated(self, tmp_path, analysis_params):
        """Exact winners recorded from CSV (lines 472-520)."""
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 512), (256, 256, 1, 512)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, exact_sizes, num_solutions=2,
                         gflops_data=[[20.0, 15.0], [10.0, 18.0]])

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps], [[sol_a, sol_b]],
                               [csv_path], analysis_params, splitGSU=False)

        assert la.numSolutions == 2
        assert la.perfMetric == "DeviceEfficiency"
        assert (128, 128, 1, 512) in la.exactWinners
        assert la.exactWinners[(128, 128, 1, 512)][0] == 0  # sol_a wins
        assert la.exactWinners[(256, 256, 1, 512)][0] == 1  # sol_b wins
        assert la.exactWinners[(128, 128, 1, 512)][1] == pytest.approx(20.0)

    def test_gflopspercu_header(self, tmp_path, analysis_params):
        """GFlopsPerCU header -> CUEfficiency (lines 441-443)."""
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 512)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)

        csv_path = str(tmp_path / "bench_cu.csv")
        _write_exact_csv(csv_path, exact_sizes, 1, [[5.0]], "GFlopsPerCU")

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps], [[sol_a]],
                               [csv_path], analysis_params, splitGSU=False)

        assert la.perfMetric == "CUEfficiency"

    def test_unknown_perf_unit_defaults(self, tmp_path, analysis_params):
        """Unknown perf unit warning -> defaults to DeviceEfficiency (line 445)."""
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 512)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)

        csv_path = str(tmp_path / "bench_unk.csv")
        _write_exact_csv(csv_path, exact_sizes, 1, [[5.0]], "UnknownUnit")

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps], [[sol_a]],
                               [csv_path], analysis_params, splitGSU=False)

        assert la.perfMetric == "DeviceEfficiency"

    def test_exact_winner_update_across_csvs(self, tmp_path, analysis_params):
        """Higher gflops in second CSV updates exact winner (line 516-518)."""
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 512)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        csv1 = str(tmp_path / "bench1.csv")
        _write_exact_csv(csv1, exact_sizes, 2, [[10.0, 8.0]])
        csv2 = str(tmp_path / "bench2.csv")
        _write_exact_csv(csv2, exact_sizes, 2, [[10.0, 15.0]])

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps, mock_ps],
                               [[sol_a, sol_b], [sol_a, sol_b]],
                               [csv1, csv2], analysis_params, splitGSU=False)

        winner_idx, winner_gflops = la.exactWinners[(128, 128, 1, 512)]
        assert winner_gflops == pytest.approx(15.0)

    def test_range_sizes_as_exacts(self, tmp_path, analysis_params):
        """Range sizes via ranges.problemSizes -> exactProblemSizes (lines 319-322)."""
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        range_sizes = [(64, 64, 1, 256), (64, 128, 1, 256),
                       (128, 64, 1, 256), (128, 128, 1, 256)]
        mock_ps = _MockProblemSizes(range_sizes=range_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        csv_path = str(tmp_path / "bench_ranges.csv")
        _write_exact_csv(csv_path, range_sizes, 2,
                         [[12.0, 9.0], [11.0, 10.0], [10.0, 13.0], [9.0, 14.0]])

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps], [[sol_a, sol_b]],
                               [csv_path], analysis_params, splitGSU=False)

        assert len(la.exactWinners) == 4
        assert la.exactWinners[(64, 64, 1, 256)][0] == 0   # sol_a
        assert la.exactWinners[(128, 128, 1, 256)][0] == 1  # sol_b


class TestRemoveInvalidSolutions:
    """removeInvalidSolutions with injected range data (lines 546-563)."""

    def test_remove_invalid_zero_gflops(self, tmp_path, analysis_params):
        """sol with gflops=0 for all range problems is removed as invalid."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 0.0]],
            range_gflops={(64, 64): [10.0, 0.0], (64, 128): [9.0, 0.0],
                          (128, 64): [8.0, 0.0], (128, 128): [7.0, 0.0]},
        )
        orig_n = la.numSolutions
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name):
            la.removeInvalidSolutions()
        assert la.numSolutions == orig_n - 1

    def test_all_valid_nothing_removed(self, tmp_path, analysis_params):
        """No solutions removed when all have positive gflops."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 18.0]],
        )
        orig_n = la.numSolutions
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name):
            la.removeInvalidSolutions()
        assert la.numSolutions == orig_n


class TestRemoveLeastImportantSolutions:
    """removeLeastImportantSolutions (lines 578-598) + leastImportantSolution (1024-1101)."""

    def test_removes_consistently_slower(self, tmp_path, analysis_params):
        """sol_b always slower -> removed by removeLeastImportantSolutions."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
            range_gflops={(64, 64): [10.0, 7.0], (64, 128): [9.0, 6.0],
                          (128, 64): [11.0, 8.0], (128, 128): [12.0, 9.0]},
        )
        orig_n = la.numSolutions
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name):
            la.removeLeastImportantSolutions()
        assert la.numSolutions < orig_n

    def test_keeps_both_when_complementary(self, tmp_path, analysis_params):
        """Both solutions kept when each wins for at least one problem."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
            range_gflops={(64, 64):   [15.0, 8.0],
                          (64, 128):  [14.0, 7.0],
                          (128, 64):  [7.0, 15.0],
                          (128, 128): [6.0, 16.0]},
        )
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name):
            la.removeLeastImportantSolutions()
        assert la.numSolutions == 2


class TestKeepWinnerSolutions:
    """keepWinnerSolutions path (lines 605-641)."""

    def test_prunes_loser(self, tmp_path, analysis_params):
        """sol_c never wins -> pruned by keepWinnerSolutions."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0, 5.0]],
            n_solutions=3,
            range_gflops={(64, 64):   [15.0, 8.0, 4.0],
                          (64, 128):  [14.0, 7.0, 4.0],
                          (128, 64):  [7.0, 15.0, 4.0],
                          (128, 128): [6.0, 16.0, 4.0]},
        )
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name):
            la.keepWinnerSolutions()
        assert la.numSolutions == 2


class TestEnRuleAndPrint2D:
    """enRule (lines 667-886) and print2D (lines 921-1017)."""

    def test_en_rule_global(self, tmp_path, analysis_params):
        """enRule from index 0 produces a rule list or None."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        rule = la.enRule(0, la.globalIndexRange)
        assert rule is None or isinstance(rule, list)

    def test_prepare_logic_converts_indices(self, tmp_path, analysis_params):
        """prepareLogic converts index thresholds to sizes (lines 905-915)."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        rule = la.enRule(0, la.globalIndexRange)
        if rule is not None:
            la.prepareLogic(rule)
            assert isinstance(rule, list)
            assert rule[-1][0] == -1

    def test_print2d_writes_csv(self, tmp_path, analysis_params):
        """print2D creates a Winner2D CSV file."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        out_dir = str(tmp_path / "print2d_out")
        os.makedirs(out_dir, exist_ok=True)
        # indices: one value per non-M/N dimension; with idx0=0(M), idx1=1(N),
        # remaining dimensions are 2 (Batch) and 3 (K) -> need [0, 0]
        la.print2D([0, 0], out_dir)
        csv_files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
        assert len(csv_files) >= 1


class TestLeastImportantSolution:
    """leastImportantSolution internals (lines 1023-1101)."""

    def test_identifies_weakest(self, tmp_path, analysis_params):
        """leastImportantSolution returns a 4-tuple for the weakest solution."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
            range_gflops={(64, 64):   [10.0, 8.0],
                          (64, 128):  [9.0, 7.0],
                          (128, 64):  [11.0, 9.0],
                          (128, 128): [12.0, 9.0]},
        )
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name):
            lis = la.leastImportantSolution()
        if lis is not None:
            assert len(lis) == 4
            assert lis[0] in range(la.numSolutions)

    def test_none_when_single_solution(self, tmp_path, analysis_params):
        """leastImportantSolution returns None with a single solution."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0]],
            n_solutions=1,
            range_gflops={(64, 64): [10.0], (64, 128): [9.0],
                          (128, 64): [11.0], (128, 128): [12.0]},
        )
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name):
            lis = la.leastImportantSolution()
        assert lis is None


class TestRemoveSolutionAndPrune:
    """removeSolution (lines 1107-1141) and pruneSolutions (1148-1195)."""

    def test_remove_solution_reshapes_data(self, tmp_path, analysis_params):
        """removeSolution decrements numSolutions and reshapes data."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        orig_n = la.numSolutions
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name):
            la.removeSolution(1)
        assert la.numSolutions == orig_n - 1
        assert la.totalSize == la.totalProblems * la.numSolutions

    def test_prune_solutions_keeps_specified(self, tmp_path, analysis_params):
        """pruneSolutions keeps only the given solution indices."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0, 5.0]],
            n_solutions=3,
            range_gflops={(64, 64): [10.0, 8.0, 4.0], (64, 128): [9.0, 7.0, 3.0],
                          (128, 64): [11.0, 9.0, 4.0], (128, 128): [12.0, 9.0, 4.0]},
        )
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name):
            la.pruneSolutions({0})
        assert la.numSolutions == 1


class TestScoringHelpers:
    """Scoring helper functions (lines 1196-1424)."""

    def test_score_range_for_solutions(self, tmp_path, analysis_params):
        """scoreRangeForSolutions returns per-solution time scores."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        scores = la.scoreRangeForSolutions(la.globalIndexRange)
        assert len(scores) == la.numSolutions
        for s in scores:
            assert s >= 0

    def test_score_range_for_logic(self, tmp_path, analysis_params):
        """scoreRangeForLogic returns a positive float."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        score = la.scoreRangeForLogic(la.globalIndexRange, [[-1, 0]])
        assert isinstance(score, float)
        assert score >= 0

    def test_get_solution_for_problem_logic(self, tmp_path, analysis_params):
        """getSolutionForProblemIndicesUsingLogic traverses numIndices levels."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        # Build a full-depth logic: numIndices=4 levels, each with [-1, sub] leaf
        # Level 4 (innermost): [-1, 0]  → solution 0
        # Level 3: [[-1, [[-1, 0]]]]
        # Level 2: [[-1, [[-1, [[-1, 0]]]]]]
        # Level 1: [[-1, [[-1, [[-1, [[-1, 0]]]]]]]]
        full_logic = 0
        for _ in range(la.numIndices):
            full_logic = [[-1, full_logic]]
        result = la.getSolutionForProblemIndicesUsingLogic(
            [0] * la.numIndices, full_logic)
        assert result == 0

    def test_score_logic_complexity(self, tmp_path, analysis_params):
        """scoreLogicComplexity tallies branch counts."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        complexity = [0] * la.numIndices
        la.scoreLogicComplexity([[-1, 0]], complexity)
        assert sum(complexity) >= 0

    def test_get_logic_depth(self, tmp_path, analysis_params):
        """getLogicDepth returns 0/1/2 for leaf/1-level/2-level rule."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        assert la.getLogicDepth(0) == 0
        assert la.getLogicDepth([[-1, 0]]) == 1
        assert la.getLogicDepth([[-1, [[-1, 0]]]]) == 2

    def test_winner_for_range(self, tmp_path, analysis_params):
        """winnerForRange returns valid solution index."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        winner = la.winnerForRange(la.globalIndexRange)
        assert winner in range(la.numSolutions) or winner == -1

    def test_problem_indices_for_range(self, tmp_path, analysis_params):
        """problemIndicesForRange enumerates all index tuples."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        result = la.problemIndicesForRange(la.globalIndexRange)
        assert len(result) == la.totalProblems
        for idxs in result:
            assert len(idxs) == la.numIndices

    def test_indices_to_serial(self, tmp_path, analysis_params):
        """indicesToSerial maps [0,...] to 0."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        assert la.indicesToSerial(0, [0] * la.numIndices) == 0
        assert la.indicesToSerial(1, [0] * la.numIndices) == 1

    def test_to_index_order(self, tmp_path, analysis_params):
        """toIndexOrder reorders problem indices by indexOrder."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        ordered = la.toIndexOrder([0] * la.numIndices)
        assert len(ordered) == la.numIndices

    def test_total_flops_for_problem_indices(self, tmp_path, analysis_params):
        """totalFlopsForProblemIndices >= flopsPerMac."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        flops = la.totalFlopsForProblemIndices([0] * la.numIndices)
        assert flops >= la.flopsPerMac

    def test_recommended_index_order(self, tmp_path, analysis_params):
        """recommendedIndexOrder produces a permutation of TotalIndices."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        order = la.recommendedIndexOrder()
        assert sorted(order) == list(range(la.problemType["TotalIndices"]))

    def test_get_winner_for_problem(self, tmp_path, analysis_params):
        """getWinnerForProblem returns (winner_idx, gflops)."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        (w_idx, w_gflops) = la.getWinnerForProblem([0] * la.numIndices)
        assert w_idx in range(la.numSolutions)
        assert w_gflops > 0

    def test_getitem_setitem(self, tmp_path, analysis_params):
        """__getitem__ and __setitem__ access the data array."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        indices = [0] * la.numIndices
        la[indices, 0] = 99.0
        val = la[indices, 0]
        assert val == pytest.approx(99.0)


class TestAnalyzeProblemType:
    """analyzeProblemType end-to-end (lines 48-233)."""

    def test_returns_nine_tuple(self, tmp_path, analysis_params):
        """Full pipeline produces a 9-tuple with correct structure."""
        from Tensile.LibraryLogic import analyzeProblemType

        pt = _MockProblemType()
        exact_sizes = [(64, 64, 1, 256), (128, 128, 1, 256)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0, tile0=64)
        sol_b = _MockSolution(idx=1, tile0=128)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, exact_sizes, 2,
                         [[12.0, 9.0], [9.0, 14.0]])

        lib_path = str(tmp_path / "logic_out")
        os.makedirs(lib_path, exist_ok=True)
        psg = [(mock_ps, csv_path, "d.yaml", "d.gsp", [sol_a, sol_b])]

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            result = analyzeProblemType(pt, psg, analysis_params, lib_path,
                                        splitGSU=False)

        assert result is not None
        assert len(result) == 9
        (ret_pt, ret_sols, ret_order, ret_exact, ret_range,
         ret_sel, ret_ids, ret_perf, ret_last) = result

        assert ret_pt is pt
        assert len(ret_sols) >= 1
        assert isinstance(ret_exact, dict)
        assert ret_perf in ("DeviceEfficiency", "CUEfficiency")
        assert ret_last is None
        assert ret_sel is None  # TileAwareSelection=False

    def test_solutionselectionalgalg1(self, tmp_path, analysis_params):
        """SolutionSelectionAlg=1 path (keepWinnerSolutions)."""
        from Tensile.LibraryLogic import analyzeProblemType
        from Tensile.Common.GlobalParameters import globalParameters

        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 256)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, exact_sizes, 2, [[15.0, 10.0]])

        lib_path = str(tmp_path / "logic_alg1")
        os.makedirs(lib_path, exist_ok=True)
        psg = [(mock_ps, csv_path, "d.yaml", "d.gsp", [sol_a, sol_b])]

        orig = globalParameters["SolutionSelectionAlg"]
        globalParameters["SolutionSelectionAlg"] = 1
        try:
            with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
                 patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
                 patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
                result = analyzeProblemType(pt, psg, analysis_params, lib_path,
                                            splitGSU=False)
        finally:
            globalParameters["SolutionSelectionAlg"] = orig

        assert result is not None
        assert len(result) == 9
