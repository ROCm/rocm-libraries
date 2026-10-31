# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The benchmark client writes its results CSV only when something reads it.

BenchmarkProblems asks the client for a results CSV only if the config has a
LibraryLogic section or uses the Ductile backend; otherwise ClientParameters.ini
sets no-results-file. Without the CSV, a rerun into the same output directory
benchmarks again (the CSV is the "already benchmarked" marker).
"""

import copy
import os
import types
from pathlib import Path

import pytest
import yaml

import Tensile.BenchmarkProblems as bp
import Tensile.ClientWriter as ClientWriter
import Tensile.ParallelExecution as ParallelExecution
from Tensile import Tensile
from Tensile.Common import BENCHMARK_DATA_DIR, DebugConfig
from Tensile.Common.GlobalParameters import globalParameters, restoreDefaultGlobalParameters

pytestmark = pytest.mark.unit

PROBLEM_TYPE_NAME = "Cijk_Ailk_Bljk_HHS_BH"


@pytest.fixture
def _restore_gp():
    """Snapshot/restore globalParameters around tests that run Tensile or reset defaults."""
    saved = copy.deepcopy(globalParameters)
    try:
        yield
    finally:
        globalParameters.clear()
        globalParameters.update(saved)


@pytest.mark.parametrize(
    "hasLibraryLogic, backendName, expected",
    [
        (True, "tensile", True),
        (False, "ductile", True),
        (True, "ductile", True),
        (False, "tensile", False),
    ],
)
def test_results_file_consumed(hasLibraryLogic, backendName, expected):
    assert bp._resultsFileConsumed(hasLibraryLogic, backendName) is expected


# ---------------------------------------------------------------------------
# BenchmarkProblems.main with a stubbed benchmark step
# ---------------------------------------------------------------------------
def _run_main(tmp_path, monkeypatch, backend=None, forceRedo=False, **mainKwargs):
    """Run main with a stubbed benchmark step that writes only the solutions YAML.

    Returns the 2_BenchmarkData directory and the writeResultsFile value of each
    stubbed benchmark step run.
    """
    resultsFileBase = str(tmp_path / "1_BenchmarkProblems" / "00_Final" / "data" / "benchmark")
    writeResultsFiles = []

    def fakeBenchmarkProblemType(**kwargs):
        writeResultsFiles.append(kwargs["writeResultsFile"])
        os.makedirs(os.path.dirname(resultsFileBase), exist_ok=True)
        with open(resultsFileBase + ".yaml", "w") as f:
            f.write("solutions\n")
        return resultsFileBase, 0

    monkeypatch.setattr(bp, "_benchmarkProblemType", fakeBenchmarkProblemType)
    monkeypatch.setattr(bp, "ProblemType", lambda config, printInfo: PROBLEM_TYPE_NAME)
    monkeypatch.setitem(globalParameters, "CSVExportWinner", False)
    monkeypatch.setitem(globalParameters, "ForceRedoBenchmarkProblems", forceRedo)
    monkeypatch.setitem(globalParameters, "ExitOnFails", True)
    monkeypatch.setitem(globalParameters, "ConfigPath", str(tmp_path / "config.yaml"))

    outputPath = tmp_path / "out"
    bp.main(
        backend=backend,
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
    return outputPath / BENCHMARK_DATA_DIR, writeResultsFiles


@pytest.mark.parametrize(
    "hasLibraryLogic, backend, expected",
    [
        (True, None, True),
        (False, {"Name": "Ductile", "Config": {}}, True),
        (False, None, False),
        (False, {"Name": "tensile"}, False),
    ],
)
def test_main_asks_for_results_only_when_consumed(
    tmp_path, monkeypatch, hasLibraryLogic, backend, expected
):
    _, writeResultsFiles = _run_main(
        tmp_path, monkeypatch, backend=backend, hasLibraryLogic=hasLibraryLogic
    )

    assert writeResultsFiles == [expected]


def test_main_default_writes_results(tmp_path, monkeypatch):
    """A caller that does not say whether LibraryLogic runs keeps the results CSV."""
    _, writeResultsFiles = _run_main(tmp_path, monkeypatch)

    assert writeResultsFiles == [True]


def test_main_removes_stale_results_csv_when_none_written(tmp_path, monkeypatch):
    """ForceRedoBenchmarkProblems into an output directory with an earlier run's CSV:
    the CSV is not this run's result, so it does not stay next to the new YAML."""
    dataPath = tmp_path / "out" / BENCHMARK_DATA_DIR
    dataPath.mkdir(parents=True)
    (dataPath / (PROBLEM_TYPE_NAME + "_00.csv")).write_text("earlier run\n")

    _run_main(tmp_path, monkeypatch, forceRedo=True, hasLibraryLogic=False)

    assert sorted(os.listdir(dataPath)) == [PROBLEM_TYPE_NAME + "_00.yaml"]


# ---------------------------------------------------------------------------
# ClientWriter: the ini asks for no results file
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "writeResultsFile, tileAwareSelection, expected",
    [
        (True, False, os.path.join("STEP", "../Data", "name.csv")),
        (True, True, os.path.join("STEP", "../Data", "name_Granularity.csv")),
        (False, False, None),
        (False, True, None),
    ],
)
def test_write_client_config_results_file(
    monkeypatch, writeResultsFile, tileAwareSelection, expected
):
    passed = []
    monkeypatch.setattr(
        ClientWriter, "writeClientConfigIni", lambda *args: passed.append(args[9])
    )
    newLibrary = types.SimpleNamespace(
        solutions={0: types.SimpleNamespace(problemType=None)}
    )

    ClientWriter.writeClientConfig(
        True, [], None, None, None, None, None, "name", "STEP", newLibrary, [],
        tileAwareSelection, 0, "gfx950", libraryFile="/LIB/TensileLibrary.dat",
        sourceDir="/SRC", writeResultsFile=writeResultsFile,
    )

    assert passed == [expected]


def _make_problem_type():
    from Tensile.Contractions import ProblemType

    return ProblemType.FromOriginalState(
        {
            "OperationType": "GEMM",
            "DataType": "S",
            "DestDataType": "S",
            "ComputeDataType": "S",
            "TransposeA": 0,
            "TransposeB": 0,
            "Batched": True,
            "UseBeta": True,
            "TotalIndices": 4,
            "NumIndicesC": 3,
            "IndicesFree": [0, 1],
            "IndicesBatch": [2],
            "IndicesSummation": [3],
            "IndexAssignmentsA": [0, 3, 2],
            "IndexAssignmentsB": [3, 1, 2],
            "ComplexConjugateA": False,
            "ComplexConjugateB": False,
            "ActivationComputeDataType": "S",
            "UseBias": 1,
            "BiasSrc": "D",
            "HighPrecisionAccumulate": False,
        }
    )


def test_write_client_config_ini_no_results_file(tmp_path, _restore_gp):
    """resultsFileName None replaces the results-file line by no-results-file=True;
    every other line is unchanged."""
    from Tensile.SolutionStructs.Problem import ProblemSizesMockDummy

    restoreDefaultGlobalParameters()

    def writeIni(resultsFileName, name):
        path = tmp_path / name
        ClientWriter.writeClientConfigIni(
            True, ProblemSizesMockDummy(), "", types.SimpleNamespace(factorDims=[0]), "", "",
            _make_problem_type(), str(tmp_path), ["TensileLibrary_gfx950.co"],
            resultsFileName, str(path), 0, "gfx950",
            libraryFile="/LIB/gfx950/TensileLibrary.dat",
        )
        return path.read_text().splitlines()

    withFile = writeIni("/DATA/benchmark.csv", "with.ini")
    withoutFile = writeIni(None, "without.ini")

    assert "results-file=/DATA/benchmark.csv" in withFile
    assert not any(line.startswith("no-results-file=") for line in withFile)
    assert withoutFile == [
        "no-results-file=True" if line == "results-file=/DATA/benchmark.csv" else line
        for line in withFile
    ]


# ---------------------------------------------------------------------------
# ParallelExecution: no merge (and no empty results CSV) without a results file
# ---------------------------------------------------------------------------
class _Proc:
    returncode = 0

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


class _DummyLock:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_parallel_execution_no_results_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ParallelExecution, "ClientExecutionLock", _DummyLock)
    monkeypatch.setitem(globalParameters, "ClientExecutionLockPath", str(tmp_path / "lock"))
    perGpuConfigs = []

    def popen(args, cwd):
        perGpuConfigs.append(Path(args[2]).read_text())
        return _Proc()

    monkeypatch.setattr(ParallelExecution.subprocess, "Popen", popen)
    cfg = tmp_path / "ClientParameters.ini"
    cfg.write_text(
        "device-idx=0\nno-results-file=True\n" + "".join(f"problem-size={i}\n" for i in range(4))
    )

    rc = ParallelExecution.runClientParallel(
        str(tmp_path), [str(cfg)], numGpus=2, timingEnabled=False,
        getClientExecutablePath=lambda: "/bin/client",
    )

    assert rc == 0
    assert len(perGpuConfigs) == 2
    for text in perGpuConfigs:
        lines = text.splitlines()
        assert "no-results-file=True" in lines
        assert not any(line.startswith("results-file=") for line in lines)
    assert sorted(os.listdir(tmp_path)) == ["ClientParameters.ini"]  # no CSV, per-GPU dir removed
    assert "No results-file found" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# End to end, GPU-less (--cpu-only): Tensile with and without LibraryLogic
# ---------------------------------------------------------------------------
_CPU_ONLY_CONFIG = Path(__file__).parent / "test_data" / "cpu_only.yaml"
# ForceRedoBenchmarkProblems defaults to True; only False consults the marker.
_NO_FORCE_REDO = ("--global-parameters", "ForceRedoBenchmarkProblems=False")


def _config(tmp_path, withLibraryLogic):
    config = yaml.safe_load(_CPU_ONLY_CONFIG.read_text())
    assert "LibraryLogic" in config
    if not withLibraryLogic:
        del config["LibraryLogic"]
    path = tmp_path / ("ll.yaml" if withLibraryLogic else "no_ll.yaml")
    path.write_text(yaml.safe_dump(config))
    return path


def _run_tensile(tensile_args, monkeypatch, config, outputDir, *extra):
    """Run Tensile GPU-less; returns the number of benchmark client launches."""
    launches = []

    def countingRunClient(*args, **kwargs):
        launches.append(args)
        return ClientWriter.runClient(*args, **kwargs)

    monkeypatch.setattr(bp, "runClient", countingRunClient)
    # --gpu-targets after tensile_args: last wins over a forwarded default.
    Tensile.Tensile(
        [str(config), str(outputDir), *tensile_args, *extra,
         "--cpu-only", "--gpu-targets", "gfx950"]
    )
    return len(launches)


def _client_inis(outputDir):
    return list((outputDir / "1_BenchmarkProblems").rglob("ClientParameters.ini"))


@pytest.fixture
def _gpu_less(monkeypatch, _restore_gp):
    def _boom(*args, **kwargs):
        raise AssertionError("builtins.input() called on the --cpu-only path")

    monkeypatch.setattr("builtins.input", _boom)
    monkeypatch.setenv("CU", "304")  # LibraryLogic's CU-count probe (see test_cpu_only_switch)


def test_end_to_end_with_library_logic(tensile_args, tmp_path, monkeypatch, _gpu_less):
    """LibraryLogic config: the ini names the results CSV, the CSV reaches
    2_BenchmarkData and the logic file is written; a rerun with
    ForceRedoBenchmarkProblems=False skips the benchmark."""
    outputDir = tmp_path / "out"
    config = _config(tmp_path, True)

    assert _run_tensile(tensile_args, monkeypatch, config, outputDir) == 1

    (ini,) = _client_inis(outputDir)
    lines = ini.read_text().splitlines()
    assert any(line.startswith("results-file=") for line in lines)
    assert "no-results-file=True" not in lines
    assert list((outputDir / BENCHMARK_DATA_DIR).glob("*.csv"))
    assert list((outputDir / "3_LibraryLogic").glob("*.yaml"))

    assert _run_tensile(tensile_args, monkeypatch, config, outputDir, *_NO_FORCE_REDO) == 0


def test_end_to_end_without_library_logic(tensile_args, tmp_path, monkeypatch, _gpu_less):
    """No LibraryLogic: the ini sets no-results-file, no CSV is written anywhere
    (not even the --cpu-only placeholder), and a rerun with
    ForceRedoBenchmarkProblems=False benchmarks again."""
    outputDir = tmp_path / "out"
    config = _config(tmp_path, False)

    assert _run_tensile(tensile_args, monkeypatch, config, outputDir) == 1

    (ini,) = _client_inis(outputDir)
    lines = ini.read_text().splitlines()
    assert "no-results-file=True" in lines
    assert not any(line.startswith("results-file=") for line in lines)
    assert not list(outputDir.rglob("*.csv"))
    assert [p.suffix for p in (outputDir / BENCHMARK_DATA_DIR).iterdir()] == [".yaml"]
    assert not (outputDir / "3_LibraryLogic").exists()

    assert _run_tensile(tensile_args, monkeypatch, config, outputDir, *_NO_FORCE_REDO) == 1


def test_end_to_end_force_redo_drops_stale_results(tensile_args, tmp_path, monkeypatch, _gpu_less):
    """A LibraryLogic run, then ForceRedoBenchmarkProblems without LibraryLogic into the
    same output directory: the first run's CSVs are not left as the second run's results."""
    outputDir = tmp_path / "out"
    _run_tensile(tensile_args, monkeypatch, _config(tmp_path, True), outputDir)
    assert list((outputDir / BENCHMARK_DATA_DIR).glob("*.csv"))

    launches = _run_tensile(
        tensile_args, monkeypatch, _config(tmp_path, False), outputDir,
        "--global-parameters", "ForceRedoBenchmarkProblems=True",
    )

    assert launches == 1
    assert not list(outputDir.rglob("*.csv"))
    assert [p.suffix for p in (outputDir / BENCHMARK_DATA_DIR).iterdir()] == [".yaml"]


def test_end_to_end_cached_rerun_without_library_logic(
    tensile_args, tmp_path, monkeypatch, capsys, _gpu_less
):
    """--use-cache rewrites ClientParameters.ini for the run at hand: a cache built by a
    LibraryLogic run, reused without LibraryLogic, asks for no results file."""
    outputDir = tmp_path / "out"
    _run_tensile(tensile_args, monkeypatch, _config(tmp_path, True), outputDir)
    capsys.readouterr()

    _run_tensile(tensile_args, monkeypatch, _config(tmp_path, False), outputDir, "--use-cache")

    assert "# Using cached solution data" in capsys.readouterr().out
    (ini,) = _client_inis(outputDir)
    lines = ini.read_text().splitlines()
    assert "no-results-file=True" in lines
    assert not any(line.startswith("results-file=") for line in lines)
    assert not list(outputDir.rglob("*.csv"))
