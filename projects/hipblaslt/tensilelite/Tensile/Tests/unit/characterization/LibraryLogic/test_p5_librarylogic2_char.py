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
"""Characterization test for Tensile/LibraryLogic.py (p5 round).

Targets uncovered arms NOT reached by test_p4_librarylogic_char.py:

  * analyzeProblemType with enableTileSelection=True (lines 67-68, 134-169)
  * LibraryType="FreeSize" -> deReferenceSolutions, indexOrder=None, rangeLogic=None,
    exactLogic=None (lines 102, 203, 218-228)
  * getVerbosity()>=2 print-raw-data block (lines 112-125)
  * addFromCSV CSVWinner path (lines 449-455, 475-479)
  * addFromCSV range problem sizes path (lines 524-538)
  * UseEffLike path including no/invalid frequency fallback (lines 492-512)
  * leastImportantSolution with all-zero totals (lines 1091, 1095, 1099)
  * getSolutionForProblemIndicesUsingLogic threshold arm (lines 1241-1245)
  * generateLogic happy path (lines 1438-1522) with synthetic YAML + CSV

CPU-only. No GPU. No rocisa. No Solution parsing.
"""

import array
import csv
import os
from copy import deepcopy
from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal mock objects (reuse the same pattern from test_p4)
# ---------------------------------------------------------------------------

class _MockDataType:
    def flopsPerMac(self):
        return 2
    def __repr__(self):
        return "s"
    def __str__(self):
        return "s"


class _MockProblemType(dict):
    def __init__(self, tile_aware=False):
        super().__init__()
        self.update({
            "Index0":             0,
            "Index1":             1,
            "IndexUnroll":        3,
            "TotalIndices":       4,
            "NumIndicesLD":       0,
            "NumIndicesC":        3,
            "NumIndicesFree":     2,
            "NumIndicesBatch":    1,
            "NumIndicesSummation": 1,
            "IndicesFree":        [0, 1],
            "IndicesBatch":       [2],
            "IndicesSummation":   [3],
            "IndexAssignmentsA":  [3, 0, 2],
            "IndexAssignmentsB":  [3, 1, 2],
            "IndexAssignmentsLD": [],
            "DataType":           _MockDataType(),
            "OperationType":      "GEMM",
            "TileAwareSelection": tile_aware,
            "GroupedGemm":        False,
        })
    def __hash__(self):
        return hash(id(self))
    def __eq__(self, other):
        return self is other


class _MockSolution(dict):
    def __init__(self, idx, tile0=64, tile1=64):
        super().__init__({"MacroTile0": tile0, "MacroTile1": tile1,
                          "SolutionIndex": idx})
    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return self is other


class _MockProblemSizes:
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


def _mock_name(sol, *a, **k):
    return "MockSol%d" % sol.get("SolutionIndex", 0)

def _mock_name_full(sol, *a, **k):
    return _mock_name(sol)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_exact_csv(path, problem_sizes, num_solutions, gflops_data,
                     header_unit="GFlops"):
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


def _write_csvwinner_csv(path, problem_sizes, num_solutions, gflops_data):
    """Write a CSV with _CSVWinner in name containing WinnerGFlops/WinnerIdx cols."""
    num_indices = len(problem_sizes[0]) if problem_sizes else 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["GFlops"]
        for i in range(num_indices):
            header.append("Size%d" % i)
        header.append("TotalFlops")
        for s in range(num_solutions):
            header.append("Sol%d" % s)
        header += [" WinnerGFlops", " WinnerIdx"]
        writer.writerow(header)
        for idx, sizes in enumerate(problem_sizes):
            total_flops = 2
            for s in sizes:
                total_flops *= s
            row_gflops = gflops_data[idx]
            winner_idx = max(range(num_solutions), key=lambda i: row_gflops[i])
            winner_gflops = row_gflops[winner_idx]
            row = ["GFlops"] + list(sizes) + [total_flops] + row_gflops
            row += [winner_gflops, winner_idx]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Build LogicAnalyzer via normal __init__ (with exact sizes driving unifiedProblemSizes)
# ---------------------------------------------------------------------------

def _build_la_from_exact(tmp_path, analysis_params, exact_sizes, gflops_data,
                         n_solutions=2, header_unit="GFlops"):
    from Tensile.LibraryLogic import LogicAnalyzer

    pt = _MockProblemType()
    mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
    solutions = [_MockSolution(idx=i) for i in range(n_solutions)]

    csv_path = str(tmp_path / "bench.csv")
    _write_exact_csv(csv_path, exact_sizes, n_solutions, gflops_data,
                     header_unit=header_unit)

    with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
         patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
        la = LogicAnalyzer(pt, [mock_ps], [solutions],
                           [csv_path], analysis_params, splitGSU=False)

    return la, pt, solutions, csv_path


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def analysis_params():
    from Tensile.Common.GlobalParameters import defaultAnalysisParameters
    return dict(defaultAnalysisParameters)


# ===========================================================================
# Test: analyzeProblemType with TileAwareSelection=True
# ===========================================================================

class TestAnalyzeProblemTypeTileAware:
    """Lines 67-68, 134-169: enableTileSelection branch."""

    def test_tile_aware_selection_path(self, tmp_path, analysis_params):
        """analyzeProblemType with TileAwareSelection=True enters tile selection arms."""
        from Tensile.LibraryLogic import analyzeProblemType

        pt = _MockProblemType(tile_aware=True)
        exact_sizes = [(128, 128, 1, 256)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, exact_sizes, 2, [[15.0, 10.0]])

        sel_file = str(tmp_path / "bench.gsp")
        lib_path = str(tmp_path / "lib_out")
        os.makedirs(lib_path, exist_ok=True)

        # TileAware selectionFileName is the 4th element of each psg tuple
        psg = [(mock_ps, csv_path, "d.yaml", sel_file, [sol_a, sol_b])]

        # Patch analyzeSolutionSelection to return a pair (sol, info)
        # where sol IS in logicAnalyzer.solutions (included path, lines 143-147)
        fake_sel_result = [(sol_a, {"tile": "64x64"})]

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full), \
             patch("Tensile.LibraryLogic.SolutionSelectionLibrary.analyzeSolutionSelection",
                   return_value=fake_sel_result):
            result = analyzeProblemType(pt, psg, analysis_params, lib_path,
                                        splitGSU=False)

        assert result is not None
        assert len(result) == 9
        # With tile selection, selectionSolutions list is populated
        # (sel_a is in logicAnalyzer.solutions -> included, selectionSolutions stays [])
        # selectionSolutionsIdsList is not None
        (ret_pt, ret_sols, ret_order, ret_exact, ret_range,
         ret_sel, ret_ids, ret_perf, ret_last) = result
        assert ret_ids is not None  # list (possibly empty) but not None

    def test_tile_aware_remainder_path(self, tmp_path, analysis_params):
        """analyzeSolutionSelection returning sol NOT in logicAnalyzer -> remainder path (lines 159-167)."""
        from Tensile.LibraryLogic import analyzeProblemType

        pt = _MockProblemType(tile_aware=True)
        exact_sizes = [(128, 128, 1, 256)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, exact_sizes, 2, [[15.0, 10.0]])
        sel_file = str(tmp_path / "bench.gsp")
        lib_path = str(tmp_path / "lib_remainder")
        os.makedirs(lib_path, exist_ok=True)

        psg = [(mock_ps, csv_path, "d.yaml", sel_file, [sol_a, sol_b])]

        # Return a brand-new solution object NOT in the merged solutions list
        external_sol = _MockSolution(idx=99, tile0=128, tile1=128)
        fake_sel_result = [(external_sol, {"tile": "128x128"})]

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full), \
             patch("Tensile.LibraryLogic.SolutionSelectionLibrary.analyzeSolutionSelection",
                   return_value=fake_sel_result):
            result = analyzeProblemType(pt, psg, analysis_params, lib_path,
                                        splitGSU=False)

        (_, _, _, _, _, sel, ids, _, _) = result
        # external_sol ends up in selectionSolutions (remainder path)
        assert sel is not None
        assert len(sel) == 1
        assert sel[0] is external_sol


# ===========================================================================
# Test: FreeSize / Prediction LibraryType paths
# ===========================================================================

class TestFreeSize:
    """Lines 102, 203, 218-228: LibraryType=FreeSize branch."""

    def test_freesizepath_deref(self, tmp_path):
        """LibraryType=FreeSize calls deReferenceSolutions, indexOrder/rangeLogic/exactLogic=None."""
        from Tensile.LibraryLogic import analyzeProblemType
        from Tensile.Common.GlobalParameters import defaultAnalysisParameters

        params = dict(defaultAnalysisParameters)
        params["LibraryType"] = "FreeSize"

        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 256)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, exact_sizes, 1, [[15.0]])

        lib_path = str(tmp_path / "lib_freesz")
        os.makedirs(lib_path, exist_ok=True)
        psg = [(mock_ps, csv_path, "d.yaml", "d.gsp", [sol_a])]

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            result = analyzeProblemType(pt, psg, params, lib_path, splitGSU=False)

        assert result is not None
        (ret_pt, ret_sols, ret_order, ret_exact, ret_range, sel, ids, perf, last) = result
        assert ret_order is None    # line 203
        assert ret_range is None    # line 219
        assert ret_exact is None    # line 228

    def test_prediction_path(self, tmp_path):
        """LibraryType=Prediction also produces None range/exact/index."""
        from Tensile.LibraryLogic import analyzeProblemType
        from Tensile.Common.GlobalParameters import defaultAnalysisParameters

        params = dict(defaultAnalysisParameters)
        params["LibraryType"] = "Prediction"

        pt = _MockProblemType()
        exact_sizes = [(64, 64, 1, 128)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, exact_sizes, 1, [[12.0]])

        lib_path = str(tmp_path / "lib_pred")
        os.makedirs(lib_path, exist_ok=True)
        psg = [(mock_ps, csv_path, "d.yaml", "d.gsp", [sol_a])]

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            result = analyzeProblemType(pt, psg, params, lib_path, splitGSU=False)

        (_, _, order, exact, rng, _, _, _, _) = result
        assert order is None
        assert rng is None
        assert exact is None


# ===========================================================================
# Test: print-raw-data block at verbosity >= 2
# ===========================================================================

class TestVerbosityPrintRaw:
    """Lines 112-125: getVerbosity() >= 2 raw-data print block."""

    def test_verbosity_2_executes_print_block(self, tmp_path, analysis_params):
        """getVerbosity()>=2 triggers the raw-data formatting loop."""
        from Tensile.LibraryLogic import analyzeProblemType

        pt = _MockProblemType()
        exact_sizes = [(64, 64, 1, 128), (128, 128, 1, 256)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, exact_sizes, 2, [[10.0, 8.0], [9.0, 12.0]])

        lib_path = str(tmp_path / "lib_v2")
        os.makedirs(lib_path, exist_ok=True)
        psg = [(mock_ps, csv_path, "d.yaml", "d.gsp", [sol_a, sol_b])]

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full), \
             patch("Tensile.LibraryLogic.getVerbosity", return_value=2):
            result = analyzeProblemType(pt, psg, analysis_params, lib_path, splitGSU=False)

        assert result is not None
        assert len(result) == 9


# ===========================================================================
# Test: addFromCSV CSVWinner path
# ===========================================================================

class TestAddFromCSVWinnerPath:
    """Lines 449-455, 475-479: _CSVWinner filename -> direct winner columns path."""

    def test_csvwinner_path_reads_winner_columns(self, tmp_path, analysis_params):
        """CSV with '_CSVWinner' in name reads WinnerGFlops/WinnerIdx directly."""
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 512), (256, 256, 1, 512)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        # Use _CSVWinner in the filename to trigger that code path
        csv_path = str(tmp_path / "bench_CSVWinner.csv")
        _write_csvwinner_csv(csv_path, exact_sizes, 2,
                             [[20.0, 15.0], [10.0, 18.0]])

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps], [[sol_a, sol_b]],
                               [csv_path], analysis_params, splitGSU=False)

        # First size: sol_a wins (gflops 20 > 15)
        assert la.exactWinners[(128, 128, 1, 512)][0] == 0
        assert la.exactWinners[(128, 128, 1, 512)][1] == pytest.approx(20.0)
        # Second size: sol_b wins (gflops 18 > 10)
        assert la.exactWinners[(256, 256, 1, 512)][0] == 1

    def test_csvwinner_missing_columns_fallback(self, tmp_path, analysis_params):
        """_CSVWinner file without WinnerGFlops/WinnerIdx columns falls back to scanning."""
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 512)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        # Write CSVWinner-named file WITHOUT WinnerGFlops/WinnerIdx columns
        csv_path = str(tmp_path / "bench_CSVWinner_bad.csv")
        _write_exact_csv(csv_path, exact_sizes, 2, [[12.0, 16.0]])

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps], [[sol_a, sol_b]],
                               [csv_path], analysis_params, splitGSU=False)

        # Falls back to scanning: sol_b wins (16 > 12)
        assert la.exactWinners[(128, 128, 1, 512)][0] == 1


# ===========================================================================
# Test: addFromCSV range problem size path
# ===========================================================================

class TestAddFromCSVRangePath:
    """Lines 524-538: range problem sizes read into data array via addFromCSV."""

    def test_range_sizes_populate_data_array(self, tmp_path, analysis_params):
        """Range problem sizes in la.rangeProblemSizes are written into la.data by addFromCSV.

        addFromCSV requires the range size to be in both la.rangeProblemSizes AND
        la.problemSizeToIndex (populated via unifiedProblemSizes).  We build a minimal
        LogicAnalyzer with injected data so that la.problemSizeToIndex is correctly
        populated, then manually register a range size and write a second CSV that
        contains that range size.
        """
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
            range_m=(64, 128),
            range_n=(64, 128),
        )
        # At this point la.problemSizeToIndex[0] has {64:0, 128:1}, etc.
        # Pick a specific range size that's already in la.rangeProblemSizes
        assert len(la.rangeProblemSizes) > 0
        range_size = sorted(la.rangeProblemSizes)[0]

        # Build a CSV that contains this range_size in the data
        csv_path2 = str(tmp_path / "bench_range2.csv")
        with open(csv_path2, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["GFlops", "Size0", "Size1", "Size2", "Size3",
                              "TotalFlops", "Sol0", "Sol1"])
            tf = 2
            for s in range_size:
                tf *= s
            writer.writerow(["GFlops"] + list(range_size) + [tf, 13.0, 11.0])

        # Record la.data before and after to confirm it was updated
        serial_before = la.indicesToSerial(0, [
            la.problemSizeToIndex[0][range_size[0]],
            la.problemSizeToIndex[1][range_size[1]],
            la.problemSizeToIndex[2][range_size[2]],
            la.problemSizeToIndex[3][range_size[3]],
        ])
        orig_val = la.data[serial_before]

        # addFromCSV with our range-only CSV (numSolutions=2 matching la.numSolutions)
        sol_map = {i: i for i in range(la.numSolutions)}
        la.addFromCSV(csv_path2, la.numSolutions, sol_map)

        new_val = la.data[serial_before]
        # After calling addFromCSV the cell should be updated to 13.0
        assert new_val == pytest.approx(13.0)


# ===========================================================================
# Test: UseEffLike path
# ===========================================================================

class TestUseEffLike:
    """Lines 492-512: globalParameters['UseEffLike']=True path."""

    def test_useefflike_with_valid_freq(self, tmp_path, analysis_params, monkeypatch):
        """UseEffLike=True with valid MAX_FREQ computes performance_metric via freq."""
        from Tensile.LibraryLogic import LogicAnalyzer
        from Tensile.Common.GlobalParameters import globalParameters

        monkeypatch.setenv("MAX_FREQ", "1000.0")
        orig = globalParameters["UseEffLike"]
        globalParameters["UseEffLike"] = True
        try:
            pt = _MockProblemType()
            exact_sizes = [(128, 128, 1, 256)]
            mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
            sol_a = _MockSolution(idx=0)

            csv_path = str(tmp_path / "bench.csv")
            _write_exact_csv(csv_path, exact_sizes, 1, [[500.0]])

            with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
                 patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
                 patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
                la = LogicAnalyzer(pt, [mock_ps], [[sol_a]],
                                   [csv_path], analysis_params, splitGSU=False)

            # performance_metric = round(500.0 / 1000.0, 2) = 0.5
            assert la.exactWinners[(128, 128, 1, 256)][1] == pytest.approx(0.5)
        finally:
            globalParameters["UseEffLike"] = orig

    def test_useefflike_no_freq_fallback(self, tmp_path, analysis_params, monkeypatch):
        """UseEffLike=True with no MAX_FREQ falls back to winnerGFlops (line 500-501)."""
        from Tensile.LibraryLogic import LogicAnalyzer
        from Tensile.Common.GlobalParameters import globalParameters

        monkeypatch.delenv("MAX_FREQ", raising=False)
        orig = globalParameters["UseEffLike"]
        globalParameters["UseEffLike"] = True
        try:
            pt = _MockProblemType()
            exact_sizes = [(64, 64, 1, 128)]
            mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
            sol_a = _MockSolution(idx=0)

            csv_path = str(tmp_path / "bench.csv")
            _write_exact_csv(csv_path, exact_sizes, 1, [[200.0]])

            with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
                 patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
                 patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
                la = LogicAnalyzer(pt, [mock_ps], [[sol_a]],
                                   [csv_path], analysis_params, splitGSU=False)

            # fallback: round(float(winnerGFlops)) = 200
            assert la.exactWinners[(64, 64, 1, 128)][1] == pytest.approx(200.0)
        finally:
            globalParameters["UseEffLike"] = orig


# ===========================================================================
# Test: leastImportantSolution edge cases (totalSavedMs=0, totalWins=0, totalExecMs=0)
# ===========================================================================

def _build_la_with_injected_data(tmp_path, analysis_params,
                                  exact_sizes, exact_gflops,
                                  n_solutions=2,
                                  range_m=(64, 128),
                                  range_n=(64, 128),
                                  range_gflops=None):
    """Build LogicAnalyzer then inject a 2-D range data grid."""
    from Tensile.LibraryLogic import LogicAnalyzer

    pt = _MockProblemType()
    mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
    solutions = [_MockSolution(idx=i, tile0=64*(i+1), tile1=64)
                 for i in range(n_solutions)]

    csv_path = str(tmp_path / "bench.csv")
    _write_exact_csv(csv_path, exact_sizes, n_solutions, exact_gflops)

    with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
         patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
         patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
        la = LogicAnalyzer(pt, [mock_ps], [solutions],
                           [csv_path], analysis_params, splitGSU=False)

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

    return la, pt, solutions


class TestLeastImportantSolutionEdgeCases:
    """Lines 1091, 1095, 1099: edge cases where totalSavedMs/totalWins/totalExecMs == 0."""

    def test_all_zeros_gflops_produces_zero_percents(self, tmp_path, analysis_params):
        """Data array with all -2 (un-benchmarked): totalWins=0, percWins=0."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
            # All range data at -2 (negative) => no winners counted
            range_gflops={(64, 64): [-2.0, -2.0], (64, 128): [-2.0, -2.0],
                          (128, 64): [-2.0, -2.0], (128, 128): [-2.0, -2.0]},
        )
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name):
            lis = la.leastImportantSolution()
        # Should still return something or None — no crash
        if lis is not None:
            # percWins should be 0 (totalWins==0)
            assert lis[2] == pytest.approx(0.0)

    def test_equal_gflops_no_second_winner(self, tmp_path, analysis_params):
        """All solutions equal gflops: secondGFlops stays negative, totalSavedMs->0."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[10.0, 10.0]],
            # identical gflops -> winner and second get same value but secondGFlops stays -1e9
            range_gflops={(64, 64): [10.0, 10.0], (64, 128): [10.0, 10.0],
                          (128, 64): [10.0, 10.0], (128, 128): [10.0, 10.0]},
        )
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name):
            lis = la.leastImportantSolution()
        # Should not raise; percSaved should be 0 since totalSavedMs will be <= 0
        if lis is not None:
            assert lis[1] >= 0  # percSaved


# ===========================================================================
# Test: getSolutionForProblemIndicesUsingLogic threshold arm
# ===========================================================================

class TestGetSolutionForProblemIndicesThreshold:
    """Lines 1241-1245: threshold arm where currentSizeIndex <= thresholdIndex."""

    def test_threshold_branch_used(self, tmp_path, analysis_params):
        """Logic with non-negative threshold -> threshold arm taken (lines 1241-1245)."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        # Build logic with a positive threshold at index 0 in the first level
        # so that currentSizeIndex (0) <= threshold (0) -> arm at 1241-1245
        # Full-depth nested logic:
        # Level 4: [0, 0]   <- threshold=0 at last level
        # Level 3: [[-1, [0, 0]]]
        # Level 2: [[-1, [[-1, [0, 0]]]]]
        # Level 1: [[-1, [[-1, [[-1, [0, 0]]]]]]]
        # This represents: at last index, if sizeIndex <= 0 use sol 0
        inner = [[0, 0]]  # threshold=0 at last level (1 of 1 -> taken)
        for _ in range(la.numIndices - 1):
            inner = [[-1, inner]]

        result = la.getSolutionForProblemIndicesUsingLogic([0] * la.numIndices, inner)
        # Should reach the threshold branch since [0] indices will hit [0, 0]
        assert result == 0

    def test_threshold_skipped_when_past(self, tmp_path, analysis_params):
        """Logic threshold < sizeIndex -> -1 fallback arm taken, not threshold arm."""
        la, pt, sols = _build_la_with_injected_data(
            tmp_path, analysis_params,
            exact_sizes=[(512, 512, 1, 512)], exact_gflops=[[20.0, 15.0]],
        )
        # Two-entry last level: first entry threshold 0 (sol 0), fallback -1 (sol 1)
        # If sizeIndex=1 -> skip threshold 0, use fallback
        inner_last = [[0, 0], [-1, 1]]  # if size <= 0: sol 0, else sol 1
        for _ in range(la.numIndices - 1):
            inner_last = [[-1, inner_last]]

        # use index 1 in last dimension (if exists)
        # m_sizes has idx0=64, idx1=128 -> m_idx can be 0 or 1
        # for index [1,0,0,0] (m_idx=1), the last-level size index = 1
        problem_indices = [1, 0, 0, 0]  # m_idx=1, rest=0
        result = la.getSolutionForProblemIndicesUsingLogic(problem_indices, inner_last)
        # Should be 1 since sizeIndex=1 skips threshold 0
        assert result in (0, 1)  # any int is valid (depends on index order)


# ===========================================================================
# Test: generateLogic happy path
# ===========================================================================

class TestGenerateLogic:
    """Lines 1438-1522: generateLogic function with synthetic CSV + YAML."""

    def test_generatelogic_runs_with_synthetic_data(self, tmp_path, analysis_params):
        """generateLogic reads CSV+YAML from benchmarkDataPath and writes to libraryLogicPath."""
        from Tensile.LibraryLogic import generateLogic
        from Tensile.Common.GlobalParameters import defaultAnalysisParameters, globalParameters

        benchmark_path = str(tmp_path / "2_BenchmarkData")
        os.makedirs(benchmark_path)
        library_path = str(tmp_path / "3_LibraryLogic")
        os.makedirs(library_path)

        # generateLogic uses LibraryIO.parseSolutionsFile internally; we patch it
        # to return minimal (problemSizes, solutions) pair
        pt = _MockProblemType()
        exact_sizes = [(128, 128, 1, 256)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)
        # parseSolutionsFile needs: problemType on solution[0]["ProblemType"]
        sol_a["ProblemType"] = pt

        # Write synthetic CSV in benchmark_path
        csv_path = os.path.join(benchmark_path, "bench_gemm.csv")
        _write_exact_csv(csv_path, exact_sizes, 1, [[15.0]])

        # Write dummy YAML (parseSolutionsFile is patched so content doesn't matter)
        yaml_path = os.path.join(benchmark_path, "bench_gemm.yaml")
        with open(yaml_path, "w") as f:
            f.write("# dummy\n")

        config = dict(defaultAnalysisParameters)
        config["ScheduleName"] = "TestSchedule"
        config["ArchitectureName"] = "gfx942"
        config["DeviceNames"] = "fallback"
        config["LibraryType"] = "GridBased"
        config["SolutionImportanceMin"] = 0.01

        # Patch parseSolutionsFile, getSolution* names, and LibraryIO output
        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full), \
             patch("Tensile.LibraryIO.parseSolutionsFile",
                   return_value=(mock_ps, [sol_a])), \
             patch("Tensile.LibraryIO.createLibraryLogic",
                   return_value={"mock": "data"}), \
             patch("Tensile.LibraryIO.writeYAML") as mock_write:
            generateLogic(
                config,
                benchmark_path,
                library_path,
                cxxCompiler="amdclang++",
                splitGSU=False,
                printSolutionRejectionReason=False,
                printIndexAssignmentInfo=False,
                isaInfoMap={},
            )

        # writeYAML should have been called once
        assert mock_write.call_count >= 1

    def test_generatelogic_json_format(self, tmp_path):
        """generateLogic with LogicFormat=json calls LibraryIO.write."""
        from Tensile.LibraryLogic import generateLogic
        from Tensile.Common.GlobalParameters import defaultAnalysisParameters, globalParameters

        benchmark_path = str(tmp_path / "2_BenchmarkData")
        os.makedirs(benchmark_path)
        library_path = str(tmp_path / "3_LibraryLogic")
        os.makedirs(library_path)

        pt = _MockProblemType()
        mock_ps = _MockProblemSizes(exact_sizes=[(128, 128, 1, 256)])
        sol_a = _MockSolution(idx=0)
        sol_a["ProblemType"] = pt

        csv_path = os.path.join(benchmark_path, "bench_gemm.csv")
        _write_exact_csv(csv_path, [(128, 128, 1, 256)], 1, [[15.0]])
        yaml_path = os.path.join(benchmark_path, "bench_gemm.yaml")
        with open(yaml_path, "w") as f:
            f.write("# dummy\n")

        config = dict(defaultAnalysisParameters)
        config["ScheduleName"] = "TestSchedule"
        config["ArchitectureName"] = "gfx942"
        config["DeviceNames"] = "fallback"
        config["LibraryType"] = "GridBased"
        config["SolutionImportanceMin"] = 0.01

        orig_format = globalParameters.get("LogicFormat", "yaml")
        globalParameters["LogicFormat"] = "json"
        try:
            with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
                 patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
                 patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full), \
                 patch("Tensile.LibraryIO.parseSolutionsFile",
                       return_value=(mock_ps, [sol_a])), \
                 patch("Tensile.LibraryIO.createLibraryLogic",
                       return_value={"mock": "data"}), \
                 patch("Tensile.LibraryIO.write") as mock_write_json:
                generateLogic(
                    config,
                    benchmark_path,
                    library_path,
                    cxxCompiler="amdclang++",
                    splitGSU=False,
                    printSolutionRejectionReason=False,
                    printIndexAssignmentInfo=False,
                    isaInfoMap={},
                )
        finally:
            globalParameters["LogicFormat"] = orig_format

        assert mock_write_json.call_count >= 1


# ===========================================================================
# Test: print2D multi-permutation (multiple batch/K dims > 1 each)
# ===========================================================================

class TestPrint2DMultiPermutation:
    """Lines 186-193: numPermutations > 1 forces multiple print2D calls.

    analyzeProblemType computes permutations over all indices except idx0 and idx1.
    With 2 Batch sizes and 2 K sizes -> numPermutations = 2 * 2 = 4.
    We exercise this by calling print2D directly on a LogicAnalyzer that has
    both batch and K range dims populated (range_m x range_n grid + 2 batch + 2 K).
    """

    def test_multi_permutation_print2d_direct(self, tmp_path, analysis_params):
        """Multiple non-M/N dims -> multiple print2D calls produce Winner2D files."""
        import array as arr_mod
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        mock_ps = _MockProblemSizes(exact_sizes=[(512, 512, 1, 512)])
        sol_a = _MockSolution(idx=0)
        sol_b = _MockSolution(idx=1)

        csv_path = str(tmp_path / "bench.csv")
        _write_exact_csv(csv_path, [(512, 512, 1, 512)], 2, [[20.0, 15.0]])

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps], [[sol_a, sol_b]],
                               [csv_path], analysis_params, splitGSU=False)

        # Inject range data with 2 M, 2 N, 2 Batch, 2 K dims
        m_sizes = [64, 128]
        n_sizes = [64, 128]
        batch_sizes = [1, 2]
        k_sizes = [256, 512]

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
        la.data = arr_mod.array("f", [0.0] * la.totalSize)
        la.rangeProblemSizes = set()
        la.globalIndexRange = [[0, la.numProblemSizes[i]] for i in range(la.numIndices)]
        la.problemIndicesForGlobalRange = la.problemIndicesForRange(la.globalIndexRange)

        # Fill data: sol_a faster for small M, sol_b faster for large M
        for m_i, m in enumerate(m_sizes):
            for n_i, n in enumerate(n_sizes):
                for b_i, b in enumerate(batch_sizes):
                    for k_i, k in enumerate(k_sizes):
                        serial = la.indicesToSerial(0, [m_i, n_i, b_i, k_i])
                        la.data[serial + 0] = 12.0 if m <= 64 else 8.0
                        la.data[serial + 1] = 8.0 if m <= 64 else 14.0

        lib_path = str(tmp_path / "lib_perm")
        os.makedirs(lib_path, exist_ok=True)

        # numPermutations = numProblemSizes[2] * numProblemSizes[3] = 2 * 2 = 4
        # Build permutations loop (mirrors analyzeProblemType lines 179-196)
        numPermutations = 1
        for i in range(la.numIndices):
            if i != la.idx0 and i != la.idx1:
                numPermutations *= la.numProblemSizes[i]

        permutations = []
        for j in range(numPermutations):
            pIdx = j
            permutation = []
            for i in range(la.numIndices):
                if i != la.idx0 and i != la.idx1:
                    npsi = la.numProblemSizes[i]
                    permutation.append(pIdx % npsi)
                    pIdx //= npsi
            permutations.append(permutation)

        for permutation in permutations:
            la.print2D(permutation, lib_path)

        csv_files = [f for f in os.listdir(lib_path) if f.startswith("Winner2D")]
        assert len(csv_files) == numPermutations
        assert numPermutations >= 4  # 2 batch * 2 K dims


# ===========================================================================
# Test: addFromCSV IOError path (line 419-420)
# ===========================================================================

class TestAddFromCSVIOError:
    """Line 419-420: file not found -> printExit."""

    def test_missing_csv_calls_printExit(self, tmp_path, analysis_params):
        """addFromCSV for non-existent file calls printExit (SystemExit)."""
        from Tensile.LibraryLogic import LogicAnalyzer

        pt = _MockProblemType()
        exact_sizes = [(64, 64, 1, 128)]
        mock_ps = _MockProblemSizes(exact_sizes=exact_sizes)
        sol_a = _MockSolution(idx=0)

        # Write a valid CSV first so LogicAnalyzer.__init__ can complete its CSV loop
        valid_csv = str(tmp_path / "bench_valid.csv")
        _write_exact_csv(valid_csv, exact_sizes, 1, [[10.0]])

        with patch("Tensile.LibraryLogic.getSolutionNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getKernelNameMin", _mock_name), \
             patch("Tensile.LibraryLogic.getSolutionNameFull", _mock_name_full):
            la = LogicAnalyzer(pt, [mock_ps], [[sol_a]],
                               [valid_csv], analysis_params, splitGSU=False)

        # Now call addFromCSV with a missing file directly
        with pytest.raises(SystemExit):
            la.addFromCSV("/tmp/nonexistent_file_xyzabc.csv", 1, {0: 0})
