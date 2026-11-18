# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The benchmark client writes its results CSV only when something reads it.

BenchmarkProblems asks the client for a results CSV only if the config has a
LibraryLogic section, uses a backend that reads benchmark results, or enables ForceResultsFile;
otherwise ClientParameters.ini sets no-results-file. Without the CSV, a rerun into the same output directory
benchmarks again (the CSV is the "already benchmarked" marker).
"""

import copy
import os
import types
from pathlib import Path

import pytest
import yaml

import tensilelite.BenchmarkProblems as bp
import tensilelite.ClientWriter as ClientWriter
import tensilelite.ParallelExecution as ParallelExecution
from tensilelite import run
from tensilelite.Common import BENCHMARK_DATA_DIR, DebugConfig
from tensilelite.Common.GlobalParameters import globalParameters, restoreDefaultGlobalParameters

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


# ---------------------------------------------------------------------------
# BenchmarkProblems.main with a stubbed benchmark step
# ---------------------------------------------------------------------------
def _run_main(tmp_path, monkeypatch, backend=None, forceRedo=False, **mainKwargs):
    """Run main with a stubbed benchmark step that writes requested client outputs.

    Returns the 2_BenchmarkData directory and the arguments passed to each benchmark step.
    """
    resultsFileBase = str(tmp_path / "1_BenchmarkProblems" / "00_Final" / "data" / "benchmark")
    steps = []

    def fakeBenchmarkProblemType(**kwargs):
        steps.append(kwargs)
        os.makedirs(os.path.dirname(resultsFileBase), exist_ok=True)
        with open(resultsFileBase + ".yaml", "w") as f:
            f.write("solutions\n")
        if kwargs["writeResultsFile"]:
            with open(resultsFileBase + ".csv", "w") as f:
                f.write("results\n")
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
    return outputPath / BENCHMARK_DATA_DIR, steps


@pytest.mark.parametrize(
    "hasLibraryLogic, backend, readsResults",
    [
        (True, None, False),
        (False, {"Name": "Ductile", "Config": {}}, True),
        (False, {"Name": "custom"}, True),
        (False, None, False),
        (False, {"Name": "tensile"}, False),
    ],
)
@pytest.mark.parametrize("forceResultsFile", [False, True])
def test_main_asks_for_results_only_when_consumed(
    tmp_path, monkeypatch, hasLibraryLogic, backend, readsResults, forceResultsFile
):
    monkeypatch.setitem(globalParameters, "ForceResultsFile", forceResultsFile)
    from tensilelite.backends.ductile_backend import DuctileBackend

    # Exercise the default backend capability under a name unknown to the CSV policy.
    monkeypatch.setitem(bp.BackendFactory._backends, "custom", DuctileBackend)
    dataPath, (step,) = _run_main(
        tmp_path, monkeypatch, backend=backend, hasLibraryLogic=hasLibraryLogic
    )

    expectCSV = hasLibraryLogic or readsResults or forceResultsFile
    assert step["writeResultsFile"] is expectCSV
    assert (step["stepBatch"] is None) is readsResults
    assert (dataPath / (PROBLEM_TYPE_NAME + "_00_00.csv")).is_file() is expectCSV


def test_main_default_writes_results(tmp_path, monkeypatch):
    """A caller that does not say whether LibraryLogic runs keeps the results CSV."""
    _, (step,) = _run_main(tmp_path, monkeypatch)

    assert step["writeResultsFile"] is True


def test_main_removes_stale_results_csv_when_none_written(tmp_path, monkeypatch):
    """ForceRedoBenchmarkProblems into an output directory with an earlier run's CSV:
    the CSV is not this run's result, so it does not stay next to the new YAML."""
    dataPath = tmp_path / "out" / BENCHMARK_DATA_DIR
    dataPath.mkdir(parents=True)
    (dataPath / (PROBLEM_TYPE_NAME + "_00_00.csv")).write_text("earlier run\n")

    _run_main(tmp_path, monkeypatch, forceRedo=True, hasLibraryLogic=False)

    assert sorted(os.listdir(dataPath)) == [PROBLEM_TYPE_NAME + "_00_00.yaml"]


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


@pytest.mark.parametrize("writeResultsFile", [True, False])
@pytest.mark.parametrize("tileAwareSelection", [True, False])
def test_write_benchmark_files_results_policy(
    tmp_path, monkeypatch, writeResultsFile, tileAwareSelection
):
    """Exercise the benchmark caller and writer together, with codegen stubbed out."""
    class Solution(dict):
        def getKernels(self):
            return []

    solution = Solution(
        ProblemType={"TileAwareSelection": tileAwareSelection, "Batched": True},
        MacroTile0=16, MacroTile1=32,
    )
    library = types.SimpleNamespace(
        solutions={0: types.SimpleNamespace(problemType=None)},
        applyNaming=lambda *args: None,
    )
    monkeypatch.setattr(bp, "copyStaticFiles", lambda *args: None)
    monkeypatch.setattr(bp, "HelperKernelObjectsMemo", lambda *args: lambda solution: [])
    monkeypatch.setattr(bp, "KernelWriterAssembly", lambda *args: None)
    monkeypatch.setattr(bp, "cmdLineArchsFor", lambda *args: ["gfx950"])
    monkeypatch.setattr(bp, "prepareSolutionsAndKernels", lambda *args, **kwargs: object())
    monkeypatch.setattr(bp, "generateBenchmarkKernels", lambda *args: None)
    monkeypatch.setattr(bp, "finishSolutionsAndKernels", lambda *args: ([], None))
    monkeypatch.setattr(bp, "getSolutionNameMin", lambda *args: "solution")
    monkeypatch.setattr(bp, "getKernelNameMin", lambda *args: "kernel")
    monkeypatch.setattr(
        bp.SolutionLibrary.MasterSolutionLibrary, "BenchmarkingLibrary",
        lambda *args: library,
    )
    monkeypatch.setattr(bp, "state", lambda *args: {})
    monkeypatch.setattr(bp.LibraryIO, "write", lambda *args: None)
    idealSizes = []

    def problemSizes(problemType, sizes):
        idealSizes.extend(sizes)
        return sizes

    monkeypatch.setattr(bp, "ProblemSizes", problemSizes)
    passed = []
    monkeypatch.setattr(
        ClientWriter, "writeClientConfigIni", lambda *args: passed.append(args[9])
    )
    stepDir = tmp_path / "STEP"
    bp.writeBenchmarkFiles(
        stepBaseDir=str(stepDir), solutions=[solution], problemSizes=None,
        biasTypeArgs=None, factorDimArgs=None, activationArgs=None, icacheFlushArgs=None,
        stepName="name", solutionSummationSizes=[64],
        asmToolchain=types.SimpleNamespace(
            assembler=types.SimpleNamespace(path="/bin/assembler")
        ),
        srcToolchain=types.SimpleNamespace(compiler=types.SimpleNamespace(default_args=[])),
        sourcePath=stepDir / "source", debugConfig=DebugConfig(),
        gateTypeArgs=None, deviceId=0, gfxName="gfx950", isaInfoMap={}, probSolMap={},
        writeResultsFile=writeResultsFile,
    )

    suffix = "_Granularity.csv" if tileAwareSelection else ".csv"
    expected = os.path.join(str(stepDir), "../Data", "name" + suffix)
    assert passed == [expected if writeResultsFile else None]
    assert idealSizes == ([{"Exact": [576, 1152, 1, 64]}] if tileAwareSelection else [])


def _make_problem_type():
    from tensilelite.Contractions import ProblemType

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
    from tensilelite.SolutionStructs.Problem import ProblemSizesMockDummy

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


@pytest.mark.parametrize("writeResultsFile", [False, True], ids=["disable", "enable"])
def test_cached_tile_reporting_policy(tmp_path, monkeypatch, _restore_gp, writeResultsFile):
    """A cached policy change reaches every launched INI and preserves tile settings."""
    from tensilelite.SolutionStructs.Problem import ProblemSizesMockDummy

    restoreDefaultGlobalParameters()
    monkeypatch.setitem(globalParameters, "CpuOnly", False)
    monkeypatch.setitem(globalParameters, "ForceRedoBenchmarkProblems", True)

    class ProblemType(dict):
        def __str__(self):
            return "TYPE"

    class Step(types.SimpleNamespace):
        def __str__(self):
            return "00_Final"

        def isFinal(self):
            return True

    problemType = ProblemType(TileAwareSelection=True)
    problemSizes = ProblemSizesMockDummy()
    problemSizes.totalProblemSizes = 1
    step = Step(
        problemSizes=problemSizes,
        biasTypeArgs=types.SimpleNamespace(totalProblemSizes=1, biasTypes=[]),
        factorDimArgs=types.SimpleNamespace(totalProblemSizes=1, factorDims=[0]),
        activationArgs=types.SimpleNamespace(totalProblemSizes=1, settingList=[]),
        icacheFlushArgs=[], forkParams={}, internalSupportParams={}, deviceScalarAlphaModes=[],
    )

    class Process:
        def __init__(self, *args, **kwargs):
            self.problemType = problemType

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return step

    groupPath = tmp_path / "TYPE_00_00"
    sourcePath = groupPath / "00_Final" / "caches" / "KEY" / "source"
    sourcePath.mkdir(parents=True)
    library = sourcePath / "TensileLibrary.dat"
    library.write_text("cached library\n")
    dataPath = groupPath / "Data"
    dataPath.mkdir()
    resultsBase = dataPath / "00_Final"
    resultsCsv = Path(str(resultsBase) + ".csv")
    granularityCsv = Path(str(resultsBase) + "_Granularity.csv")
    tileConfig = sourcePath / "ClientParameters_Granularity.ini"
    tileSettings = (
        b"# cached tile configuration\r\nproblem-size=576,1152,1,64\r\n"
        b"problem-size=576,1152,1,128\r\na-strides=1,576,73728\r\n"
    )
    oldPolicy = (
        b"no-results-file=True\n" if writeResultsFile else
        f"results-file={granularityCsv}\nno-results-file=False\n".encode()
    )
    tileConfig.write_bytes(oldPolicy + tileSettings)
    if not writeResultsFile:
        resultsCsv.write_text("previous run\n")
        granularityCsv.write_text("previous tile run\n")

    monkeypatch.setattr(bp, "BenchmarkProcess", Process)
    monkeypatch.setattr(bp, "_computeCacheKey", lambda *args: "KEY")
    monkeypatch.setattr(
        bp, "_loadCacheIfMatches",
        lambda *args: {"CodeObjectFiles": [], "LibraryFile": str(library)},
    )
    monkeypatch.setattr(bp, "tensileLibraryFile", lambda *args: library)
    monkeypatch.setattr(bp, "ProblemType", lambda *args: {})
    conProblemType = _make_problem_type()
    monkeypatch.setattr(
        bp.ContractionsProblemType, "FromOriginalState", lambda *args: conProblemType
    )
    monkeypatch.setattr(bp.LibraryIO, "writeSolutions", lambda *args, **kwargs: None)
    monkeypatch.setattr(bp, "cmdLineArchsFor", lambda *args: ["gfx950"])
    launched = {}

    def launch(*args, **kwargs):
        for config in kwargs["configPaths"]:
            path = Path(config)
            launched[path.name] = path.read_text().splitlines()
            for line in launched[path.name]:
                if line.startswith("results-file="):
                    Path(line.split("=", 1)[1]).write_text("current run\n")
        return 0

    monkeypatch.setattr(bp, "runClient", launch)

    def runBackend(config, benchmarkConfig, runner, **kwargs):
        assert kwargs["cacheValid"] is True
        return runner([], isCached=True)

    monkeypatch.setattr(
        bp.BackendFactory, "create", lambda *args: types.SimpleNamespace(run=runBackend)
    )
    finalBase, failures = bp._benchmarkProblemType(
        backendConfig={"Name": "tensile"}, problemTypeConfig={},
        problemSizeGroupConfig={"ForkParameters": []}, problemSizeGroupIdx=0,
        outerBenchmarkIdx=0, configPath="config.yaml", useCache=True,
        asmToolchain=types.SimpleNamespace(assembler=None),
        srcToolchain=types.SimpleNamespace(compiler="compiler"), cCompiler="compiler",
        buildTmpPath=tmp_path / "build", benchmarkProblemsPath=tmp_path,
        debugConfig=DebugConfig(), deviceId=0, gfxName="gfx950", isaInfoMap={}, probSolMap={},
        writeResultsFile=writeResultsFile,
    )

    assert finalBase == str(resultsBase)
    assert failures == 0
    assert set(launched) == {"ClientParameters.ini", "ClientParameters_Granularity.ini"}
    for name, lines in launched.items():
        directives = [
            line for line in lines
            if line.startswith(("results-file=", "no-results-file="))
        ]
        path = granularityCsv if name == tileConfig.name else resultsCsv
        assert directives == (
            [f"results-file={path}"] if writeResultsFile else ["no-results-file=True"]
        )
    expectedPolicy = (
        f"results-file={granularityCsv}\n".encode() if writeResultsFile else
        b"no-results-file=True\n"
    )
    assert tileConfig.read_bytes() == expectedPolicy + tileSettings
    assert resultsCsv.exists() is writeResultsFile
    assert granularityCsv.exists() is writeResultsFile


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


@pytest.mark.parametrize(
    "resultsSetting",
    [
        "results-file={results}",
        "results-file = {results}",
        "results-file= {results}",
        "results-file={results} # benchmark output",
        " results-file\t=\t{results}\t # benchmark output",
    ],
)
@pytest.mark.parametrize(
    "setting, writeResultsFile",
    [
        ("no-results-file=True", False),
        ("no-results-file = True", False),
        ("no-results-file= True", False),
        ("no-results-file=True # omit CSV", False),
        (" no-results-file\t=\tyEs\t # omit CSV", False),
        ("no-results-file=1", False),
        ("no-results-file=False", True),
        ("no-results-file = 0 # keep CSV", True),
        ("no-results-file= no", True),
    ],
)
def test_parallel_execution_results_file_policy(
    tmp_path, monkeypatch, capsys, setting, writeResultsFile, resultsSetting
):
    """Valid client INI formatting preserves the requested output policy during merging."""
    monkeypatch.setattr(ParallelExecution, "ClientExecutionLock", _DummyLock)
    monkeypatch.setitem(globalParameters, "ClientExecutionLockPath", str(tmp_path / "lock"))
    perGpuConfigs = []

    def popen(args, cwd):
        text = Path(args[2]).read_text()
        gpu = len(perGpuConfigs)
        perGpuConfigs.append(text)
        if writeResultsFile:
            # Model the client's valid INI syntax independently of the Python reader.
            options = dict(
                tuple(part.strip() for part in line.split("#", 1)[0].split("=", 1))
                for line in text.splitlines() if "=" in line.split("#", 1)[0]
            )
            result = options["results-file"]
            Path(result).write_text(f"Problem,GFLOPS\n{gpu},1\n")
        return _Proc()

    monkeypatch.setattr(ParallelExecution.subprocess, "Popen", popen)
    cfg = tmp_path / "ClientParameters.ini"
    results = tmp_path / "results.csv"
    cfg.write_text(
        f"device-idx=0\n{resultsSetting.format(results=results)}\n{setting}\n"
        + "".join(f"problem-size={i}\n" for i in range(4))
    )

    rc = ParallelExecution.runClientParallel(
        str(tmp_path), [str(cfg)], numGpus=2, timingEnabled=False,
        getClientExecutablePath=lambda: "/bin/client",
    )

    assert rc == 0
    assert len(perGpuConfigs) == 2
    for text in perGpuConfigs:
        lines = text.splitlines()
        assert setting in lines
    assert results.exists() is writeResultsFile
    if writeResultsFile:
        assert results.read_text() == "Problem,GFLOPS\n0,1\n1,1\n"
    assert sorted(os.listdir(tmp_path)) == (
        ["ClientParameters.ini", "results.csv"] if writeResultsFile else ["ClientParameters.ini"]
    )
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
    run.main(
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


@pytest.mark.parametrize("useCache", [False, True], ids=["fresh", "cached"])
def test_end_to_end_force_results_file(
    tensile_args, tmp_path, monkeypatch, _gpu_less, useCache
):
    """The YAML opt-in restores externally consumed CSVs, including on cached reruns."""
    outputDir = tmp_path / "out"
    configPath = _config(tmp_path, False)
    _run_tensile(tensile_args, monkeypatch, configPath, outputDir)
    assert not list(outputDir.rglob("*.csv"))

    config = yaml.safe_load(configPath.read_text())
    config["GlobalParameters"]["ForceResultsFile"] = True
    configPath.write_text(yaml.safe_dump(config))
    cacheArgs = ("--use-cache",) if useCache else ()
    assert _run_tensile(tensile_args, monkeypatch, configPath, outputDir, *cacheArgs) == 1

    (ini,) = _client_inis(outputDir)
    assert "no-results-file=True" not in ini.read_text().splitlines()
    assert any(line.startswith("results-file=") for line in ini.read_text().splitlines())
    assert list((outputDir / BENCHMARK_DATA_DIR).glob("*.csv"))
    assert not (outputDir / "3_LibraryLogic").exists()
    assert _run_tensile(tensile_args, monkeypatch, configPath, outputDir, *_NO_FORCE_REDO) == 0

    # Turning the opt-in off must refresh the INI and remove the earlier run's CSVs.
    config["GlobalParameters"]["ForceResultsFile"] = False
    configPath.write_text(yaml.safe_dump(config))
    assert _run_tensile(tensile_args, monkeypatch, configPath, outputDir, *cacheArgs) == 1
    assert "no-results-file=True" in ini.read_text().splitlines()
    assert not list(outputDir.rglob("*.csv"))


@pytest.mark.parametrize("useCache", [False, True], ids=["fresh", "cached"])
def test_end_to_end_force_redo_drops_stale_results(
    tensile_args, tmp_path, monkeypatch, _gpu_less, useCache
):
    """A LibraryLogic run, then ForceRedoBenchmarkProblems without LibraryLogic into the
    same output directory: the first run's CSVs are not left as the second run's results."""
    outputDir = tmp_path / "out"
    _run_tensile(tensile_args, monkeypatch, _config(tmp_path, True), outputDir)
    assert list((outputDir / BENCHMARK_DATA_DIR).glob("*.csv"))
    (sourceCsv,) = (outputDir / "1_BenchmarkProblems").rglob("00_Final.csv")
    sourceCsv.with_name("00_Final_Granularity.csv").write_text("previous tile run\n")
    (collectedCsv,) = (outputDir / BENCHMARK_DATA_DIR).glob("*.csv")
    collectedCsv.with_suffix(".gsp").write_text("previous tile run\n")

    launches = _run_tensile(
        tensile_args, monkeypatch, _config(tmp_path, False), outputDir,
        "--global-parameters", "ForceRedoBenchmarkProblems=True",
        *(("--use-cache",) if useCache else ()),
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
