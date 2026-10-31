# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""BenchmarkProblems.main copies the client's results CSV only if it exists.

After the client runs, main copies the results CSV, the solutions YAML and
the granularity CSV from 1_BenchmarkProblems to 2_BenchmarkData. A client run
that wrote no results CSV must not fail the copy, as for the granularity CSV.
"""

import os

import pytest

import Tensile.BenchmarkProblems as bp
from Tensile.Common import BENCHMARK_DATA_DIR, DebugConfig
from Tensile.Common.GlobalParameters import globalParameters

pytestmark = pytest.mark.unit

PROBLEM_TYPE_NAME = "Cijk_Ailk_Bljk_HHS_BH"


def _run_main(tmp_path, monkeypatch, clientOutputs):
    """Run main with a stubbed benchmark step that writes ``clientOutputs``.

    clientOutputs maps a file suffix (".csv", ".yaml", "_Granularity.csv") to
    the content the stubbed client run leaves next to the results file base.
    Returns the 2_BenchmarkData directory and the list of benchmarked
    problem size group indices (one entry per stubbed benchmark step run).
    """
    resultsFileBase = str(tmp_path / "1_BenchmarkProblems" / "00_Final" / "data" / "benchmark")
    benchmarked = []

    def fakeBenchmarkProblemType(**kwargs):
        benchmarked.append(kwargs["problemSizeGroupIdx"])
        os.makedirs(os.path.dirname(resultsFileBase), exist_ok=True)
        for suffix, content in clientOutputs.items():
            with open(resultsFileBase + suffix, "w") as f:
                f.write(content)
        return resultsFileBase, 0

    monkeypatch.setattr(bp, "_benchmarkProblemType", fakeBenchmarkProblemType)
    monkeypatch.setattr(bp, "ProblemType", lambda config, printInfo: PROBLEM_TYPE_NAME)
    monkeypatch.setitem(globalParameters, "CSVExportWinner", False)
    monkeypatch.setitem(globalParameters, "ForceRedoBenchmarkProblems", False)
    monkeypatch.setitem(globalParameters, "ExitOnFails", True)
    monkeypatch.setitem(globalParameters, "ConfigPath", str(tmp_path / "config.yaml"))

    outputPath = tmp_path / "out"
    bp.main(
        backend=None,
        config=[[{}, {}]],
        useCache=False,
        asmToolchain=None,
        srcToolchain=None,
        cCompiler="",
        outputPath=outputPath,
        buildTmpPath=tmp_path / "build_tmp",
        debugConfig=DebugConfig(),
        deviceId=0,
        gfxName="gfx950",
        isaInfoMap={},
        probSolMap={},
    )
    return outputPath / BENCHMARK_DATA_DIR, benchmarked


def test_missing_results_csv_is_not_copied(tmp_path, monkeypatch):
    """No results CSV: main succeeds and still copies the solutions YAML."""
    dataPath, _ = _run_main(tmp_path, monkeypatch, {".yaml": "solutions\n"})

    assert sorted(os.listdir(dataPath)) == [PROBLEM_TYPE_NAME + "_00.yaml"]
    assert (dataPath / (PROBLEM_TYPE_NAME + "_00.yaml")).read_text() == "solutions\n"


def test_existing_results_csv_is_copied(tmp_path, monkeypatch):
    """Results and granularity CSVs present: both are copied with the YAML."""
    dataPath, _ = _run_main(
        tmp_path,
        monkeypatch,
        {".csv": "results\n", ".yaml": "solutions\n", "_Granularity.csv": "granularity\n"},
    )

    assert sorted(os.listdir(dataPath)) == sorted(
        PROBLEM_TYPE_NAME + suffix for suffix in ("_00.csv", "_00.yaml", "_00.gsp")
    )
    assert (dataPath / (PROBLEM_TYPE_NAME + "_00.csv")).read_text() == "results\n"
    assert (dataPath / (PROBLEM_TYPE_NAME + "_00.gsp")).read_text() == "granularity\n"


def test_empty_results_csv_is_copied(tmp_path, monkeypatch):
    """The guard tests existence only: an empty results CSV is copied as before."""
    dataPath, _ = _run_main(tmp_path, monkeypatch, {".csv": "", ".yaml": "solutions\n"})

    assert (dataPath / (PROBLEM_TYPE_NAME + "_00.csv")).read_text() == ""


def test_results_csv_marks_problem_type_as_benchmarked(tmp_path, monkeypatch):
    """The copied results CSV is the "already benchmarked, skip" marker of a rerun."""
    _run_main(tmp_path, monkeypatch, {".csv": "results\n", ".yaml": "solutions\n"})
    _, benchmarked = _run_main(tmp_path, monkeypatch, {".csv": "results\n", ".yaml": "solutions\n"})

    assert benchmarked == []


def test_missing_results_csv_reruns_next_time(tmp_path, monkeypatch):
    """Without a copied results CSV a rerun benchmarks the problem type again."""
    _run_main(tmp_path, monkeypatch, {".yaml": "solutions\n"})
    _, benchmarked = _run_main(tmp_path, monkeypatch, {".yaml": "solutions\n"})

    assert benchmarked == [0]
