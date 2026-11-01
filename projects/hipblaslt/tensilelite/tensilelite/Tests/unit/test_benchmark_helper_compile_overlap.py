# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The benchmark path compiles the helper kernels (Kernels.cpp) of a step in a background thread
(prepareSolutionsAndKernels) while the assembly kernels are generated (generateBenchmarkKernels),
assembled and linked (finishSolutionsAndKernels). The compile takes helperCompileJobs slots of the
pool budget (one per chunk it is split into), and its failure still fails the step. Every pool of
the benchmark run has the same size (benchmarkPoolSize), so joblib never resizes its worker pool."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from Tensile import BenchmarkProblems
from Tensile.TensileCreateLibrary import Run

pytestmark = pytest.mark.unit


def _write(tmp_path, generateSourcesAndExit=False, srcToolchain=None, kernelHelperObjs=()):
    """One step's kernel build, as BenchmarkProblems.writeBenchmarkFiles runs it. Returns the
    build."""
    with Run.helperCompileExecutor() as helperExecutor:
        build = Run.prepareSolutionsAndKernels(
            str(tmp_path),  # outputPath
            MagicMock(),  # asmToolchain
            srcToolchain or MagicMock(),
            [],  # solutions
            [],  # kernels
            list(kernelHelperObjs),
            MagicMock(),  # kernelWriterAssembly
            False,  # splitGSU
            ["gfx950"],  # cmdlineArchs
            helperExecutor,
            generateSourcesAndExit=generateSourcesAndExit,
        )
        Run.generateBenchmarkKernels([build])
        Run.finishSolutionsAndKernels(build)
    return build


@pytest.fixture
def builders(monkeypatch):
    """Stub the toolchain work; record the procs of every pool and the order of events."""
    events = []
    poolProcs = []

    def parallelMap2(function, objects, message="", procs=None, **kw):
        events.append(message)
        poolProcs.append(procs)
        return []

    monkeypatch.setattr(Run, "rocisa", MagicMock())
    monkeypatch.setattr(Run, "buildAssemblyCodeObjectFiles", lambda *a, **kw: [])
    monkeypatch.setattr(Run, "ParallelMap2", parallelMap2)
    monkeypatch.setattr(BenchmarkProblems, "ParallelMap2", parallelMap2)
    monkeypatch.setattr(BenchmarkProblems, "rocisa", MagicMock())
    monkeypatch.setattr(Run, "CPUThreadCount", lambda: 8)
    monkeypatch.setattr(
        Run, "buildSourceCodeObjectFiles", lambda *a, **kw: events.append("helper compile")
    )
    monkeypatch.events = events
    monkeypatch.poolProcs = poolProcs
    return monkeypatch


def test_helper_compile_runs_while_assembly_kernels_are_generated(tmp_path, builders):
    compileStarted = threading.Event()
    codegenDone = threading.Event()
    seen = {}

    def compileHelpers(*a, **kw):
        seen["kernelsCppWritten"] = (tmp_path / "Kernels.cpp").is_file()
        compileStarted.set()
        # True only if codegen ran while this compile was still running.
        seen["codegenRanDuringCompile"] = codegenDone.wait(timeout=10)

    def parallelMap2(function, objects, message="", **kw):
        if message == "Generating assembly kernels":
            seen["compileStartedDuringCodegen"] = compileStarted.wait(timeout=10)
            codegenDone.set()
        return []

    builders.setattr(Run, "buildSourceCodeObjectFiles", compileHelpers)
    builders.setattr(Run, "ParallelMap2", parallelMap2)
    _write(tmp_path)

    assert seen == {
        "kernelsCppWritten": True,
        "compileStartedDuringCodegen": True,
        "codegenRanDuringCompile": True,
    }


def test_helper_compile_takes_one_slot_of_the_pool_budget(tmp_path, builders):
    _write(tmp_path)
    # Codegen, the invalid-kernel check and assembly.
    assert builders.poolProcs == [7, 7, 7]


@pytest.mark.parametrize(
    "budget, generateSourcesAndExit, expected",
    [
        (64, False, 60),
        (24, False, 20),
        (24, True, 24),
        (12, False, 10),
        (8, False, 7),
        (2, False, 1),
        (8, True, 8),
        (1, False, 1),
        (0, False, 0),
    ],
)
def test_benchmark_pool_size(monkeypatch, budget, generateSourcesAndExit, expected):
    monkeypatch.setattr(Run, "CPUThreadCount", lambda: budget)
    assert Run.benchmarkPoolSize(generateSourcesAndExit) == expected


@pytest.mark.parametrize(
    "budget, expected",
    [(256, 4), (64, 4), (24, 4), (23, 3), (18, 3), (12, 2), (11, 1), (6, 1), (2, 1), (1, 1)],
)
def test_helper_compile_jobs(monkeypatch, budget, expected):
    """One chunk process per _BUDGET_PER_HELPER_CHUNK of the budget, 1 to MAX_HELPER_CHUNKS."""
    monkeypatch.setattr(Run, "CPUThreadCount", lambda: budget)
    assert Run.helperCompileJobs() == expected


class _Helper:
    sharedHeader = False

    def __init__(self, name):
        self.name = name

    def getKernelName(self):
        return self.name

    def getSourceFileString(self):
        return (0, f"\n// {self.name}\n")

    def getHeaderFileString(self):
        return ""


HELPERS = [_Helper(f"Cijk_Kernel{i}") for i in range(12)]


def _chunkingToolchain():
    compiler = SimpleNamespace(launcher=[], default_args=[])
    return SimpleNamespace(compiler=compiler, bundler=None)


def test_helper_compile_gets_one_chunk_per_compile_job(tmp_path, builders):
    builders.setattr(Run, "CPUThreadCount", lambda: 24)
    seen = []
    builders.setattr(
        Run,
        "buildSourceCodeObjectFiles",
        lambda *a, **kw: seen.append({Path(p).name: Path(p).is_file() for p in a[7]}),
    )
    _write(tmp_path, srcToolchain=_chunkingToolchain(), kernelHelperObjs=HELPERS)

    assert seen == [{f"Kernels_chunk{i}.cpp": True for i in range(4)}]
    assert builders.poolProcs == [20, 20, 20]


@pytest.mark.parametrize(
    "budget, srcToolchain, generateSourcesAndExit",
    [
        (1, _chunkingToolchain(), False),  # no overlap: the compile runs alone, in one piece
        (24, MagicMock(), False),  # a compile that cannot be chunked (here: a launcher)
        (24, _chunkingToolchain(), True),  # no compile
    ],
)
def test_no_chunks(tmp_path, builders, budget, srcToolchain, generateSourcesAndExit):
    builders.setattr(Run, "CPUThreadCount", lambda: budget)
    seen = []
    builders.setattr(Run, "buildSourceCodeObjectFiles", lambda *a, **kw: seen.append(a[7]))
    _write(tmp_path, generateSourcesAndExit, srcToolchain, HELPERS)
    assert seen == ([] if generateSourcesAndExit else [[]])
    assert not list(tmp_path.rglob("Kernels_chunk*"))


def test_solution_generation_pools_have_the_benchmark_pool_size(builders):
    builders.setitem(BenchmarkProblems.globalParameters, "GenerateSourcesAndExit", False)
    BenchmarkProblems._generateForkedSolutions(None, None, [], None, None, None)
    BenchmarkProblems._loadSolutionPool([])
    builders.setattr(BenchmarkProblems, "ProblemType", lambda *a: None)
    BenchmarkProblems._constructAllPoolSolutions(
        [("pool.yaml", {"ProblemType": {}, "Solutions": []})], None, MagicMock(), None
    )
    assert builders.poolProcs == [7, 7, 7]


def test_budget_of_one_compiles_before_codegen(tmp_path, builders):
    builders.setattr(Run, "CPUThreadCount", lambda: 1)
    _write(tmp_path)
    assert builders.events == [
        "helper compile",
        "Generating assembly kernels",
        "check invalid kernels and results",
        "Writing assembly kernels",
    ]
    assert builders.poolProcs == [1, 1, 1]


def test_generated_sources_are_dropped_once_written(tmp_path, builders):
    """The generated sources of every step of a pool are held until the step is finished;
    finishing it drops them."""
    generated = []
    generate = Run.generateBenchmarkKernels

    def generateAndRecord(builds):
        generate(builds)
        generated.extend(build.asmResults for build in builds)

    builders.setattr(Run, "generateBenchmarkKernels", generateAndRecord)
    build = _write(tmp_path)
    assert generated == [[]]
    assert build.asmResults is None


def test_generate_sources_and_exit_does_not_compile(tmp_path, builders):
    _write(tmp_path, generateSourcesAndExit=True)
    assert "helper compile" not in builders.events
    assert builders.poolProcs == [8, 8, 8]
    assert (tmp_path / "Kernels.cpp").is_file()


def test_helper_compile_failure_is_raised(tmp_path, builders):
    def compileHelpers(*a, **kw):
        raise RuntimeError("Kernels.cpp failed to compile")

    builders.setattr(Run, "buildSourceCodeObjectFiles", compileHelpers)
    with pytest.raises(RuntimeError, match="Kernels.cpp failed to compile"):
        _write(tmp_path)


def test_codegen_failure_waits_for_the_helper_compile(tmp_path, builders):
    compileStarted = threading.Event()
    compileFinished = threading.Event()

    def compileHelpers(*a, **kw):
        compileStarted.set()
        time.sleep(0.2)
        compileFinished.set()

    def parallelMap2(*a, **kw):
        assert compileStarted.wait(timeout=10)
        raise RuntimeError("codegen failed")

    builders.setattr(Run, "buildSourceCodeObjectFiles", compileHelpers)
    builders.setattr(Run, "ParallelMap2", parallelMap2)
    with pytest.raises(RuntimeError, match="codegen failed"):
        _write(tmp_path)
    # No compile is left running in the background once the error reaches the caller.
    assert compileFinished.is_set()
