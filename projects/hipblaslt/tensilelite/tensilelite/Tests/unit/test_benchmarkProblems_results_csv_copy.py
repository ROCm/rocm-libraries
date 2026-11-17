# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Result collection distinguishes missing required CSVs from deliberate omission."""

import os

import pytest

import tensilelite.BenchmarkProblems as bp
from tensilelite.Common import BENCHMARK_DATA_DIR, DebugConfig
from tensilelite.Common.GlobalParameters import globalParameters

pytestmark = pytest.mark.unit

PROBLEM_TYPE_NAME = "Cijk_Ailk_Bljk_HHS_BH"


def _run_main(
    tmp_path, monkeypatch, clientOutputs, *, benchmarkErrors=0, exitOnFails=True,
    forceRedo=False, **mainKwargs,
):
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
        return resultsFileBase, benchmarkErrors

    monkeypatch.setattr(bp, "_benchmarkProblemType", fakeBenchmarkProblemType)
    monkeypatch.setattr(bp, "ProblemType", lambda config, printInfo: PROBLEM_TYPE_NAME)
    monkeypatch.setitem(globalParameters, "CSVExportWinner", False)
    monkeypatch.setitem(globalParameters, "ForceRedoBenchmarkProblems", forceRedo)
    monkeypatch.setitem(globalParameters, "ExitOnFails", exitOnFails)
    monkeypatch.setitem(globalParameters, "ConfigPath", str(tmp_path / "config.yaml"))

    outputPath = tmp_path / "out"
    bp.main(
        backend=mainKwargs.pop("backend", None),
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
        **mainKwargs,
    )
    return outputPath / BENCHMARK_DATA_DIR, benchmarked


def test_disabled_results_csv_is_not_copied(tmp_path, monkeypatch):
    """Deliberately omitted results remain successful and retain the solutions YAML."""
    dataPath, _ = _run_main(
        tmp_path, monkeypatch, {".yaml": "solutions\n"}, hasLibraryLogic=False
    )

    assert sorted(os.listdir(dataPath)) == [PROBLEM_TYPE_NAME + "_00_00.yaml"]
    assert (dataPath / (PROBLEM_TYPE_NAME + "_00_00.yaml")).read_text() == "solutions\n"


@pytest.mark.parametrize(
    "mainKwargs",
    [{}, {"hasLibraryLogic": False, "backend": {"Name": "Ductile", "Config": {}}}],
    ids=["library-logic", "ductile"],
)
def test_missing_required_results_csv_fails(tmp_path, monkeypatch, capsys, mainKwargs):
    """A zero client exit cannot hide missing results needed by either consumer."""
    with pytest.raises(SystemExit) as exc:
        _run_main(tmp_path, monkeypatch, {".yaml": "solutions\n"}, **mainKwargs)

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Expected results CSV was not produced" in output
    assert "clientExit=1 (ERROR)" in output
    dataPath = tmp_path / "out" / BENCHMARK_DATA_DIR
    assert sorted(os.listdir(dataPath)) == [PROBLEM_TYPE_NAME + "_00_00.yaml"]


@pytest.mark.parametrize("benchmarkErrors, expectedErrors", [(0, 1), (3, 3)])
def test_missing_required_csv_respects_exit_policy(
    tmp_path, monkeypatch, capsys, benchmarkErrors, expectedErrors
):
    """ExitOnFails=False permits continuation, but the run still reports failure."""
    _run_main(
        tmp_path, monkeypatch, {".yaml": "solutions\n"},
        benchmarkErrors=benchmarkErrors, exitOnFails=False,
    )

    assert f"clientExit={expectedErrors} (ERROR)" in capsys.readouterr().out


def test_missing_required_csv_removes_stale_rerun_marker(tmp_path, monkeypatch):
    dataPath = tmp_path / "out" / BENCHMARK_DATA_DIR
    dataPath.mkdir(parents=True)
    staleCsv = dataPath / (PROBLEM_TYPE_NAME + "_00_00.csv")
    staleCsv.write_text("previous run\n")

    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, {".yaml": "solutions\n"}, forceRedo=True)

    assert not staleCsv.exists()
    _, benchmarked = _run_main(
        tmp_path, monkeypatch, {".yaml": "solutions\n"}, exitOnFails=False
    )
    assert benchmarked == [0]


def test_existing_results_csv_is_copied(tmp_path, monkeypatch):
    """Results and granularity CSVs present: both are copied with the YAML."""
    dataPath, _ = _run_main(
        tmp_path,
        monkeypatch,
        {".csv": "results\n", ".yaml": "solutions\n", "_Granularity.csv": "granularity\n"},
    )

    assert sorted(os.listdir(dataPath)) == sorted(
        PROBLEM_TYPE_NAME + suffix for suffix in ("_00_00.csv", "_00_00.yaml", "_00_00.gsp")
    )
    assert (dataPath / (PROBLEM_TYPE_NAME + "_00_00.csv")).read_text() == "results\n"
    assert (dataPath / (PROBLEM_TYPE_NAME + "_00_00.gsp")).read_text() == "granularity\n"


def test_empty_results_csv_is_copied(tmp_path, monkeypatch):
    """The guard tests existence only: an empty results CSV is copied as before."""
    dataPath, _ = _run_main(tmp_path, monkeypatch, {".csv": "", ".yaml": "solutions\n"})

    assert (dataPath / (PROBLEM_TYPE_NAME + "_00_00.csv")).read_text() == ""


def test_results_csv_marks_problem_type_as_benchmarked(tmp_path, monkeypatch):
    """The copied results CSV is the "already benchmarked, skip" marker of a rerun."""
    _run_main(tmp_path, monkeypatch, {".csv": "results\n", ".yaml": "solutions\n"})
    _, benchmarked = _run_main(tmp_path, monkeypatch, {".csv": "results\n", ".yaml": "solutions\n"})

    assert benchmarked == []


def test_missing_results_csv_reruns_next_time(tmp_path, monkeypatch):
    """Without requested results a rerun benchmarks the problem type again."""
    _run_main(tmp_path, monkeypatch, {".yaml": "solutions\n"}, hasLibraryLogic=False)
    _, benchmarked = _run_main(
        tmp_path, monkeypatch, {".yaml": "solutions\n"}, hasLibraryLogic=False
    )

    assert benchmarked == [0]
