# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The benchmark steps of a config share one codegen pool (timing_runs/DECISIONS.md topic 7, C).

With a backend that does not use the benchmark results (TensileBackend), BenchmarkProblems.main
prepares every step first (solutions, helper kernels written and their compile queued), then
generates the kernels of all of them in one pool, then finishes each step in order (assembly,
library, client run, results copy). A step waits for the earlier ones when it would read what
they write: a problem size group that comes again without ForceRedoBenchmarkProblems (the
"already benchmarked" check), or a step that builds into the same directory. Ductile reads the
results, so its steps still run one after the other. A --use-cache hit has no kernels to
generate."""

import threading
import types
from unittest.mock import MagicMock

import pytest

import Tensile.BenchmarkProblems as BP
from Tensile.Common import DebugConfig, IsaVersion
from Tensile.TensileCreateLibrary import Run

pytestmark = pytest.mark.unit

PROBLEM_TYPE_NAME = "Cijk_Ailk_Bljk_HHS_BH"


# ---------------------------------------------------------------------------
# TensileCreateLibrary.Run: one pool for the kernels of several builds
# ---------------------------------------------------------------------------
def _kernelBuild(name, kernels):
    return Run.BenchmarkKernelBuild(
        outputPath=None,
        asmToolchain=None,
        solutions=[],
        kernelWriterAssembly=f"writer-{name}",
        splitGSU=False,
        archNames={},
        errorTolerant=True,
        generateSourcesAndExit=False,
        compress=False,
        removeTemporaries=True,
        destRoot=None,
        buildTmpPath=None,
        assemblyTmpPath=None,
        asmGroups=[],
        asmKernels=kernels,
        rocisaData=f"data-{name}",
        outOptions=f"options-{name}",
        poolSize=7,
        helperBuild=None,
    )


def test_kernels_of_all_builds_are_generated_in_one_pool(monkeypatch):
    calls = []

    def parallelMap2(function, objects, message="", procs=None, **kw):
        objects = list(objects)
        calls.append((function, message, procs, objects))
        return [f"result-{args[-1]}" for args in objects]

    monkeypatch.setattr(Run, "ParallelMap2", parallelMap2)
    builds = [_kernelBuild("a", ["a0", "a1"]), _kernelBuild("b", []), _kernelBuild("c", ["c0"])]
    Run.generateBenchmarkKernels(builds)

    assert len(calls) == 1
    function, message, procs, objects = calls[0]
    assert message == "Generating assembly kernels"
    assert procs == 7
    # The sources of all builds are held until each step is finished: compressed.
    assert function.keywords["compress"] is True
    # Each kernel is generated with its own build's writer and output options.
    assert objects == [
        ("writer-a", "data-a", "options-a", False, "a0"),
        ("writer-a", "data-a", "options-a", False, "a1"),
        ("writer-c", "data-c", "options-c", False, "c0"),
    ]
    assert [b.asmResults for b in builds] == [["result-a0", "result-a1"], [], ["result-c0"]]


def test_one_build_keeps_its_sources_uncompressed(monkeypatch):
    calls = []

    def parallelMap2(function, objects, message="", procs=None, **kw):
        calls.append(function)
        return [None for _ in objects]

    monkeypatch.setattr(Run, "ParallelMap2", parallelMap2)
    Run.generateBenchmarkKernels([_kernelBuild("a", ["a0", "a1"])])

    assert calls[0].keywords["compress"] is False


def test_no_builds_start_no_pool(monkeypatch):
    monkeypatch.setattr(Run, "ParallelMap2", MagicMock(side_effect=AssertionError("no pool")))
    Run.generateBenchmarkKernels([])


def test_queued_helper_compiles_are_cancelled_on_error():
    running = threading.Event()
    release = threading.Event()
    ran = []

    def compileHelpers(name):
        running.set()
        release.wait(timeout=10)
        ran.append(name)

    with pytest.raises(RuntimeError, match="codegen failed"):
        with Run.helperCompileExecutor() as executor:
            futures = [executor.submit(compileHelpers, n) for n in ("step1", "step2", "step3")]
            assert running.wait(timeout=10)
            release.set()
            raise RuntimeError("codegen failed")
    # The running compile finished before the error left the block; the queued ones never ran.
    assert ran == ["step1"]
    assert futures[0].done() and futures[1].cancelled() and futures[2].cancelled()


# ---------------------------------------------------------------------------
# BenchmarkProblems.main with stubbed problem size groups
# ---------------------------------------------------------------------------
def _runMain(
    tmp_path,
    monkeypatch,
    config,
    backend=None,
    returncodes=None,
    exitOnFails=False,
    forceRedo=True,
    events=None,
    failPreparing=None,
    failWithPrintExit=False,
    failGenerating=False,
):
    """Run main with a stubbed _benchmarkProblemType. A group prepares one step into stepBatch
    (when main passes one) whose finish function records an event, writes the group's results
    and returns its client return code (returncodes, by group index); without stepBatch, the
    group runs at once. Preparing group failPreparing raises, so does generating the kernels
    with failGenerating. Returns the events in order (also recorded into events)."""
    events = [] if events is None else events
    returncodes = returncodes or {}
    groups = []

    def fakeBenchmarkProblemType(**kwargs):
        group = len(groups)
        groups.append(group)
        resultsFileBase = str(tmp_path / f"group{group}" / "benchmark")
        (tmp_path / f"group{group}").mkdir()

        def finish():
            events.append(f"finish {group}")
            for suffix in (".csv", ".yaml"):
                with open(resultsFileBase + suffix, "w") as f:
                    f.write(f"group {group}\n")
            return resultsFileBase + ".csv", returncodes.get(group, 0)

        events.append(f"prepare {group}")
        if group == failPreparing:
            if failWithPrintExit:
                BP.printExit("preparing failed")
            raise RuntimeError("preparing failed")
        stepBatch = kwargs["stepBatch"]
        if stepBatch is None:
            _, returncode = finish()
            return resultsFileBase, 1 if returncode else 0
        stepBatch.addStep(str(tmp_path / f"group{group}" / "cache"), f"kernels {group}", finish)
        return resultsFileBase, 0

    def generateBenchmarkKernels(builds):
        events.append(f"generate {', '.join(builds)}")
        if failGenerating:
            raise RuntimeError("codegen failed")

    realCopy = BP.shutil.copy

    def copy(src, dst):
        if dst.endswith(".yaml"):
            events.append(f"collect {open(src).read().split()[1]}")
        return realCopy(src, dst)

    monkeypatch.setattr(BP, "_benchmarkProblemType", fakeBenchmarkProblemType)
    monkeypatch.setattr(BP, "generateBenchmarkKernels", generateBenchmarkKernels)
    monkeypatch.setattr(BP.shutil, "copy", copy)
    monkeypatch.setattr(
        BP, "ProblemType", lambda config, printInfo: config.get("Name", PROBLEM_TYPE_NAME)
    )
    monkeypatch.setitem(BP.globalParameters, "CSVExportWinner", False)
    monkeypatch.setitem(BP.globalParameters, "ForceRedoBenchmarkProblems", forceRedo)
    monkeypatch.setitem(BP.globalParameters, "ExitOnFails", exitOnFails)
    monkeypatch.setitem(BP.globalParameters, "ConfigPath", str(tmp_path / "config.yaml"))

    BP.main(
        backend=backend,
        config=config,
        useCache=False,
        asmToolchain=None,
        srcToolchain=None,
        cCompiler="",
        outputPath=tmp_path / "out",
        buildTmpPath=tmp_path / "build_tmp",
        debugConfig=DebugConfig(),
        deviceId=0,
        gfxName="gfx950",
        isaInfoMap={},
        probSolMap={},
    )
    return events


def test_main_generates_the_kernels_of_all_groups_in_one_pool(tmp_path, monkeypatch):
    events = _runMain(tmp_path, monkeypatch, [[{}, {}, {}], [{"Name": "Other"}, {}]])

    assert events == [
        "prepare 0",
        "prepare 1",
        "prepare 2",
        "generate kernels 0, kernels 1, kernels 2",
        "finish 0",
        "collect 0",
        "finish 1",
        "collect 1",
        "finish 2",
        "collect 2",
    ]


def test_main_repeated_group_shares_the_pool(tmp_path, monkeypatch):
    # The third group is <PROBLEM_TYPE_NAME>_00 again: its 2_BenchmarkData files are written
    # after group 0's, as when the groups ran one after the other.
    events = _runMain(tmp_path, monkeypatch, [[{}, {}, {}], [{}, {}]])

    assert events == [
        "prepare 0",
        "prepare 1",
        "prepare 2",
        "generate kernels 0, kernels 1, kernels 2",
        "finish 0",
        "collect 0",
        "finish 1",
        "collect 1",
        "finish 2",
        "collect 2",
    ]


def test_main_repeated_group_without_force_redo_waits_for_the_earlier_one(tmp_path, monkeypatch):
    # Without ForceRedoBenchmarkProblems, main skips a group whose results exist: group 0's
    # results, collected before the check, make it skip the third group.
    events = _runMain(tmp_path, monkeypatch, [[{}, {}, {}], [{}, {}]], forceRedo=False)

    assert events == [
        "prepare 0",
        "prepare 1",
        "generate kernels 0, kernels 1",
        "finish 0",
        "collect 0",
        "finish 1",
        "collect 1",
    ]


def test_main_counts_the_failures_of_deferred_steps(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _runMain(tmp_path, monkeypatch, [[{}, {}, {}]], returncodes={0: 3}, exitOnFails=True)

    out = capsys.readouterr().out
    assert out.count("clientExit=1 (ERROR)") == 2


def test_main_runs_ductile_groups_one_after_the_other(tmp_path, monkeypatch):
    events = _runMain(
        tmp_path, monkeypatch, [[{}, {}, {}]], backend={"Name": "Ductile", "Config": {}}
    )

    assert events == ["prepare 0", "finish 0", "collect 0", "prepare 1", "finish 1", "collect 1"]


def test_main_finishes_the_prepared_groups_when_preparing_fails(tmp_path, monkeypatch):
    """As when each group was built before the next was prepared: the groups before the
    failing one are built, run and collected, then the error goes on."""
    events = []
    with pytest.raises(RuntimeError, match="preparing failed"):
        _runMain(tmp_path, monkeypatch, [[{}, {}, {}, {}]], events=events, failPreparing=2)

    assert events == [
        "prepare 0", "prepare 1", "prepare 2",
        "generate kernels 0, kernels 1",
        "finish 0", "collect 0", "finish 1", "collect 1",
    ]  # fmt: skip


def test_main_stops_when_generating_the_kernels_fails(tmp_path, monkeypatch):
    """The prepared steps' kernels are missing: none is finished, the pool runs once."""
    events = []
    with pytest.raises(RuntimeError, match="codegen failed"):
        _runMain(tmp_path, monkeypatch, [[{}, {}, {}]], events=events, failGenerating=True)

    assert events == ["prepare 0", "prepare 1", "generate kernels 0, kernels 1"]


def test_batch_finished_within_a_group_carries_its_failures():
    """A step that has to wait for the earlier steps finishes them, also those of its own
    group: their failures are reported with the group."""
    collected = []
    stepBatch = BP._StepBatch("executor")
    stepBatch.addStep("dir0", None, lambda: ("results0", 0))
    stepBatch.endGroup("group0.csv", 0, collected.append)
    stepBatch.addStep("dir1", None, lambda: ("results1", 5))
    assert stepBatch.buildsInto("dir1") and stepBatch.collectsInto("group0.csv")
    stepBatch.finish()
    assert collected == [0]
    assert not stepBatch.buildsInto("dir1") and not stepBatch.collectsInto("group0.csv")
    stepBatch.addStep("dir1", None, lambda: ("results1", 0))
    stepBatch.endGroup("group1.csv", 1, collected.append)
    stepBatch.finish()
    assert collected == [0, 2]


# ---------------------------------------------------------------------------
# _benchmarkProblemType with a stepBatch
# ---------------------------------------------------------------------------
class _Step:
    problemSizes = types.SimpleNamespace(totalProblemSizes=1)
    factorDimArgs = types.SimpleNamespace(totalProblemSizes=1)
    biasTypeArgs = types.SimpleNamespace(totalProblemSizes=1)
    activationArgs = types.SimpleNamespace(totalProblemSizes=1)
    icacheFlushArgs = []
    forkParams = {}
    paramGroups = []
    constantParams = {}
    customKernels = []
    internalSupportParams = {}
    customKernelWildcard = False

    def isFinal(self):
        return True

    def __str__(self):
        return "00_Final"


class _FakeBenchmarkProcess:
    def __init__(self, *_args, **_kwargs):
        self.problemType = {"TileAwareSelection": False}

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return _Step()


@pytest.fixture
def stubbedStep(monkeypatch, tmp_path):
    """_benchmarkProblemType with TensileBackend's solution generation and the file writers
    stubbed; records the calls."""
    calls = []
    monkeypatch.setattr(BP, "BenchmarkProcess", _FakeBenchmarkProcess)
    monkeypatch.setattr(BP, "_computeCacheKey", lambda *_a: "abc123")
    monkeypatch.setattr(BP, "_generateForkedSolutions", lambda *_a: [{}])
    monkeypatch.setattr(BP, "_generateCustomKernelSolutions", lambda *_a: [])
    monkeypatch.setattr(BP, "getSolutionNameMin", lambda *_a: "s")
    monkeypatch.setattr(BP, "getKernelNameMin", lambda *_a: "k")
    monkeypatch.setattr(BP, "startTime", 0.0)

    def prepareBenchmarkFiles(*args, helperExecutor, **kwargs):
        calls.append(("prepareBenchmarkFiles", helperExecutor))
        return BP.BenchmarkFilesBuild(
            "kernels", lambda: calls.append("finish files") or (["k.co"], "lib.dat")
        )

    monkeypatch.setattr(BP, "prepareBenchmarkFiles", prepareBenchmarkFiles)
    monkeypatch.setattr(
        BP, "writeBenchmarkFiles", MagicMock(side_effect=AssertionError("deferred"))
    )
    monkeypatch.setattr(BP, "writeClientConfigIni", lambda *_a, **_kw: calls.append("client ini"))
    monkeypatch.setattr(BP, "tensileLibraryFile", lambda *_a: str(tmp_path / "lib.dat"))
    monkeypatch.setattr(BP.LibraryIO, "writeYAML", lambda *_a: calls.append("cache.yaml"))
    monkeypatch.setattr(
        BP.LibraryIO, "writeSolutions", lambda *_a, **_kw: calls.append("solutions")
    )
    monkeypatch.setitem(BP.globalParameters, "ConfigPath", ["/tmp/cfg.yaml"])
    monkeypatch.setitem(BP.globalParameters, "LibraryFormat", "msgpack")
    monkeypatch.calls = calls
    return monkeypatch


def _benchmarkProblemType(tmp_path, stepBatch, useCache=False, problemSizeGroupIdx=0):
    return BP._benchmarkProblemType(
        backendConfig={"Name": "tensile"},
        problemTypeConfig={"OperationType": "GEMM", "DataType": "s"},
        problemSizeGroupConfig={"ForkParameters": True},
        problemSizeGroupIdx=problemSizeGroupIdx,
        outerBenchmarkIdx=0,
        configPath="/tmp/cfg.yaml",
        useCache=useCache,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        cCompiler="cc",
        buildTmpPath=tmp_path / "build_tmp",
        benchmarkProblemsPath=tmp_path / "bench",
        debugConfig=DebugConfig(),
        deviceId=0,
        gfxName="gfx950",
        isaInfoMap={IsaVersion(9, 5, 0): {}},
        probSolMap={},
        buildOnly=True,
        stepBatch=stepBatch,
    )


def test_step_is_only_prepared_into_the_batch(tmp_path, stubbedStep):
    generated = []
    stubbedStep.setattr(BP, "generateBenchmarkKernels", generated.append)
    stepBatch = BP._StepBatch("executor")

    resultsFileBase, fails = _benchmarkProblemType(tmp_path, stepBatch)

    assert resultsFileBase.endswith("00_Final") and fails == 0
    assert stubbedStep.calls == [("prepareBenchmarkFiles", "executor")]
    stepBatch.finish()
    assert generated == [["kernels"]]
    assert stubbedStep.calls[1:] == ["finish files", "cache.yaml", "solutions"]


def test_step_without_solutions_exits_after_the_earlier_steps(tmp_path, stubbedStep, capsys):
    """Without a batch, the steps before one with 0 valid solutions were built, run and
    collected before the exit; with a batch they are finished before it too, and the FATAL
    line is still the last output."""
    generated, collected = [], []
    stubbedStep.setattr(BP, "generateBenchmarkKernels", generated.append)
    stubbedStep.setattr(BP, "_generateForkedSolutions", lambda *_a: [])

    def collect(errors):
        print("collected group 0")
        collected.append(errors)

    with pytest.raises(SystemExit):
        with BP._stepBatchFor(True) as stepBatch:
            with stubbedStep.context() as firstStep:
                firstStep.setattr(BP, "_generateForkedSolutions", lambda *_a: [{}])
                _benchmarkProblemType(tmp_path, stepBatch, problemSizeGroupIdx=0)
            stepBatch.endGroup("group0.csv", 0, collect)
            _benchmarkProblemType(tmp_path, stepBatch, problemSizeGroupIdx=1)

    assert generated == [["kernels"]]
    assert stubbedStep.calls[0] == ("prepareBenchmarkFiles", stepBatch.helperExecutor)
    assert stubbedStep.calls[1:] == ["finish files", "cache.yaml", "solutions"]
    assert collected == [0]
    lines = capsys.readouterr().out.splitlines()
    fatal = [i for i, line in enumerate(lines) if line.startswith("Tensile::FATAL:")]
    # The reason, after the earlier step's output; nothing to finish once it is printed.
    assert fatal and lines.index("collected group 0") < fatal[0]
    assert lines[fatal[0]].startswith("Tensile::FATAL: Your parameters resulted in 0 valid")
    assert "stopped by the error above" not in "\n".join(lines)


def test_main_ends_with_a_fatal_line_when_preparing_exits(tmp_path, monkeypatch, capsys):
    """A printExit while preparing a step prints its reason before the earlier steps are
    finished: a closing FATAL line after them points back to it."""
    events = []
    with pytest.raises(SystemExit):
        _runMain(
            tmp_path,
            monkeypatch,
            [[{}, {}, {}]],
            events=events,
            failPreparing=1,
            failWithPrintExit=True,
        )

    assert events == ["prepare 0", "prepare 1", "generate kernels 0", "finish 0", "collect 0"]
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert "Tensile::FATAL: preparing failed" in lines
    assert lines[-1].startswith("Tensile::FATAL: stopped by the error above")


def test_step_building_into_a_pending_directory_waits(tmp_path, stubbedStep):
    """The cache key leaves out the problem sizes: two groups of the same name that differ only
    in them build into the same directory, which the second one resets."""
    generated = []
    stubbedStep.setattr(BP, "generateBenchmarkKernels", generated.append)
    stepBatch = BP._StepBatch("executor")

    _benchmarkProblemType(tmp_path, stepBatch)
    _benchmarkProblemType(tmp_path, stepBatch)

    # The first step was generated and finished before the second one was prepared.
    assert generated == [["kernels"]]
    assert stubbedStep.calls == [
        ("prepareBenchmarkFiles", "executor"),
        "finish files",
        "cache.yaml",
        "solutions",
        ("prepareBenchmarkFiles", "executor"),
    ]
    stepBatch.finish()
    assert generated == [["kernels"], ["kernels"]]


def test_cache_hit_has_no_kernels_to_generate(tmp_path, stubbedStep):
    stubbedStep.setattr(
        BP,
        "_loadCacheIfMatches",
        lambda *_a: {"CodeObjectFiles": ["k.co"], "LibraryFile": "lib.dat"},
    )
    stubbedStep.setattr(BP.os.path, "isfile", lambda _p: True)
    stubbedStep.setattr(BP, "ProblemType", lambda *_a: None)
    stubbedStep.setattr(
        BP, "ContractionsProblemType", types.SimpleNamespace(FromOriginalState=lambda *_a: None)
    )
    stubbedStep.setattr(
        BP,
        "generateBenchmarkKernels",
        MagicMock(side_effect=AssertionError("no codegen for a cache hit")),
    )
    stepBatch = BP._StepBatch("executor")

    _benchmarkProblemType(tmp_path, stepBatch, useCache=True)
    stepBatch.finish()

    assert stubbedStep.calls == ["client ini", "solutions"]
