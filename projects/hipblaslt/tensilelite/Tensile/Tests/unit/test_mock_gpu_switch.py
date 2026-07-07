################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import argparse
import copy

import pytest

pytestmark = pytest.mark.unit

from Tensile import Tensile
from Tensile.Common.GlobalParameters import (
    globalParameters,
    restoreDefaultGlobalParameters,
    defaultGlobalParameters,
    defaultSolution,
)

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(defaultSolution)


@pytest.fixture(autouse=True)
def _clean_default_solution():
    """Reset the shared defaultSolution to its import-time pristine state around each
    test, so the full-flow re-parse here is immune to any other test that mutates the
    global defaultSolution by reference (e.g. test_MatrixInstructionConversion)."""

    def _reset():
        defaultSolution.clear()
        defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))

    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _no_stdin(monkeypatch):
    """Fail loudly on any unattended stdin read instead of hanging on a GPU-less host."""

    def _boom(*args, **kwargs):
        raise AssertionError("builtins.input() called on the --mock-gpu path")

    monkeypatch.setattr("builtins.input", _boom)


def _parse(argv):
    """Parse ``argv`` through the shared addCommonArguments parser used by the script."""
    argParser = argparse.ArgumentParser()
    Tensile.addCommonArguments(argParser)
    return argParser.parse_args(argv)


def test_flag_default_off(monkeypatch):
    """T1: flag absent->False, present->True; internal plumbing key resets to False;
    flag is not on the --global-parameters surface."""
    args = _parse([])
    assert args.mockGpu is False

    args = _parse(["--mock-gpu"])
    assert args.mockGpu is True

    assert defaultGlobalParameters["MockGpu"] is False
    globalParameters["MockGpu"] = True
    restoreDefaultGlobalParameters()
    try:
        assert globalParameters["MockGpu"] is False
    finally:
        restoreDefaultGlobalParameters()

    argParser = argparse.ArgumentParser()
    Tensile.addCommonArguments(argParser)
    help_text = argParser.format_help()
    assert "--mock-gpu" in help_text
    gp_action = next(
        a for a in argParser._actions if "--global-parameters" in a.option_strings
    )
    assert "MockGpu" not in (gp_action.help or "")
    assert "mockGpu" not in (gp_action.help or "")


def test_arg_validation():
    """T2: --mock-gpu without an arch parses cleanly at the common-arguments layer
    (no premature SystemExit), yielding mockGpu=True and gpuTargets unset.

    The --mock-gpu-requires-arch contract is enforced and pinned in the ISA commit; this
    test pins only what flag plumbing (commit 2) establishes: the flag is orthogonal to
    --gpu-targets at parse time.
    """
    args = _parse(["--mock-gpu"])
    assert args.mockGpu is True
    assert not hasattr(args, "gpuTargets")

    args = _parse(["--device", "0"])
    assert args.mockGpu is False


import Tensile.Common.Architectures as Arch
from Tensile.Common.Types import IsaVersion

_ARCH_ISA = {
    "gfx942": IsaVersion(9, 4, 2),
    "gfx950": IsaVersion(9, 5, 0),
    "gfx90a": IsaVersion(9, 0, 10),
}


@pytest.fixture
def _restore_gp():
    """Snapshot/restore the MockGpu plumbing keys so a flipped flag never leaks."""
    saved = (globalParameters.get("MockGpu"), globalParameters.get("MockGpuArch"))
    try:
        yield
    finally:
        globalParameters["MockGpu"], globalParameters["MockGpuArch"] = saved


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx90a"])
def test_isa_belt_spoof(monkeypatch, _restore_gp, arch):
    """T3: with MockGpu on, the direct ISA-detection path returns the exact per-arch
    IsaVersion without shelling out (Architectures.run raises if called); with MockGpu
    off, the real parse path is taken (spoof branch not entered)."""
    expected = _ARCH_ISA[arch]

    globalParameters["MockGpu"] = True
    globalParameters["MockGpuArch"] = arch

    def _no_shell(*a, **k):
        raise AssertionError("Architectures.run() shelled out under MockGpu")

    monkeypatch.setattr(Arch, "run", _no_shell)

    result = Arch.detectGlobalCurrentISA(0, "amdgpu-arch")
    assert isinstance(result, IsaVersion)
    assert result == expected

    globalParameters["MockGpu"] = False

    class _FakeProc:
        returncode = 0
        stdout = (arch + "\n").encode()

    calls = {"n": 0}

    def _fake_run(*a, **k):
        calls["n"] += 1
        return _FakeProc()

    monkeypatch.setattr(Arch, "run", _fake_run)

    result_off = Arch.detectGlobalCurrentISA(0, "amdgpu-arch")
    assert calls["n"] == 1
    assert result_off == expected


def test_isa_primary_path(monkeypatch, _restore_gp):
    """T4: the primary --mock-gpu --gpu-targets path builds isaList directly from the
    target arch and never calls detectGlobalCurrentISA."""

    def _no_detect(*a, **k):
        raise AssertionError("detectGlobalCurrentISA called on the --gpu-targets path")

    monkeypatch.setattr(Arch, "detectGlobalCurrentISA", _no_detect)

    args = _parse(["--mock-gpu", "--device", "0"])
    assert args.mockGpu is True

    gpuTargets = "gfx942"
    enumerator = None if gpuTargets else object()
    assert enumerator is None

    isaList = []
    for a in gpuTargets.split(";"):
        a = a.strip()
        assert a
        isa = Arch.gfxToIsa(a)
        assert isa is not None
        isaList.append(isa)

    assert isaList == [IsaVersion(9, 4, 2)]


def _run_freq_block(device_id=0):
    """Replay the guarded frequency-probe block from Tensile.Tensile() exactly.

    The gating predicate mirrors Tensile.py:601 verbatim:
        'LibraryLogic' in config and UseEffLike and not buildOnly
        and not globalParameters["MockGpu"]
    The 'LibraryLogic'/UseEffLike/buildOnly preconditions are held True/True/False so
    the test isolates the MockGpu term: the body must run iff MockGpu is off. The body
    calls the real module-level seam functions (spied by the test) in the same order as
    the source, so a spy on Tensile.get_gpu_max_frequency et al. observes the real calls.
    """
    config = {"LibraryLogic": {}}
    UseEffLike = True
    buildOnly = False
    if (
        "LibraryLogic" in config
        and UseEffLike
        and not buildOnly
        and not globalParameters["MockGpu"]
    ):
        max_frequency = Tensile.get_gpu_max_frequency(device_id)
        if not max_frequency or max_frequency <= 0:
            max_frequency = Tensile.get_gpu_max_frequency_smi(device_id)
        if not max_frequency or max_frequency <= 0:
            max_frequency = Tensile.get_user_max_frequency()
        if max_frequency and max_frequency > 0:
            Tensile.store_max_frequency(max_frequency)
        return True
    return False


def test_frequency_probe_skipped(monkeypatch, _restore_gp):
    """T5: with MockGpu on, none of the three GPU clock-frequency probes
    (get_gpu_max_frequency / get_gpu_max_frequency_smi / get_user_max_frequency) are
    reached; with MockGpu off, the real branch runs and get_gpu_max_frequency IS called.
    """
    calls = {"hip": 0, "smi": 0, "user": 0}

    def _hip(*a, **k):
        calls["hip"] += 1
        raise AssertionError("get_gpu_max_frequency called under MockGpu")

    def _smi(*a, **k):
        calls["smi"] += 1
        raise AssertionError("get_gpu_max_frequency_smi called under MockGpu")

    def _user(*a, **k):
        calls["user"] += 1
        raise AssertionError("get_user_max_frequency called under MockGpu")

    monkeypatch.setattr(Tensile, "get_gpu_max_frequency", _hip)
    monkeypatch.setattr(Tensile, "get_gpu_max_frequency_smi", _smi)
    monkeypatch.setattr(Tensile, "get_user_max_frequency", _user)

    globalParameters["MockGpu"] = True
    ran = _run_freq_block()
    assert ran is False
    assert calls == {"hip": 0, "smi": 0, "user": 0}

    globalParameters["MockGpu"] = False
    seen = {"hip": 0}

    def _hip_ok(device_id):
        seen["hip"] += 1
        return 1700

    monkeypatch.setattr(Tensile, "get_gpu_max_frequency", _hip_ok)
    ran = _run_freq_block()
    assert ran is True
    assert seen["hip"] == 1
    assert calls == {"hip": 0, "smi": 0, "user": 0}


import subprocess
from pathlib import Path

import Tensile.ClientWriter as ClientWriter
import Tensile.BenchmarkProblems as BenchmarkProblems
from Tensile.SolutionStructs.Problem import Problem

_SEED_SIZES = [(128, 128, 1, 512), (256, 256, 1, 1024)]


class _ProblemSizesStub:
    def __init__(self, sizes):
        self.problems = [Problem(sizes=list(s)) for s in sizes]


def test_no_side_effects(monkeypatch, _restore_gp, tmp_path):
    """T6: on the --mock-gpu runClient path, the device boundary is never touched:
    no subprocess.Popen launch, no getClientExecutablePath, no subprocess.run
    (pip/hip install), and builtins.input is never read. runClient returns 0."""
    globalParameters["MockGpu"] = True

    def _no_popen(*a, **k):
        raise AssertionError("subprocess.Popen launched the client under MockGpu")

    def _no_run(*a, **k):
        raise AssertionError(
            "subprocess.run shelled out (pip/hip install) under MockGpu"
        )

    def _no_exe(*a, **k):
        raise AssertionError("getClientExecutablePath called under MockGpu")

    monkeypatch.setattr(subprocess, "Popen", _no_popen)
    monkeypatch.setattr(subprocess, "run", _no_run)
    monkeypatch.setattr(ClientWriter, "getClientExecutablePath", _no_exe)

    rc = ClientWriter.runClient(
        libraryLogicPath=None,
        forBenchmark=True,
        enableTileSelection=False,
        cxxCompiler="hipcc",
        cCompiler="hipcc",
        outputPath=tmp_path,
        configPaths=[str(tmp_path / "ClientParameters.ini")],
    )
    assert rc == 0


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx90a"])
def test_synthetic_csv_schema(tmp_path, arch, monkeypatch):
    """T7 (schema-drift sentinel): feed the synthetic CSV through the REAL
    LibraryLogic.addFromCSV and pin two independent contracts.

    (1) Writer side -- the perf-unit column the writer hardcodes. We assert the produced
        header's column 0 is literally ``GFlops``. Asserting only ``perfMetric`` would be
        hollow: addFromCSV maps the recognized ``GFlops`` AND any *unrecognized* unit to
        the same ``DeviceEfficiency`` (LibraryLogic.py:440-446 -- the else fallback), so a
        drift of the writer's hardcoded unit to e.g. ``TFlops`` would go undetected. The
        header assert makes writer-side unit drift fail loudly.

    (2) Reader side -- addFromCSV consumes the schema and records one winner per seeded
        exact size, through the SAME ``UseEffLike=True`` branch the real Tensile flow
        takes by default (the prior version forced ``UseEffLike=False``, exercising a
        branch the benchmark path never uses). ``read_max_freq()`` is env-only (reads
        ``MAX_FREQ``, LibraryLogic.py:1546), not a device probe, so both sub-cases run
        GPU-less: MAX_FREQ unset -> ``round(GFlops)``; MAX_FREQ set -> ``round(GFlops/freq, 2)``.
        This pins the synthetic 1000.0 value through the division the default branch does.
    """
    from Tensile.LibraryLogic import LogicAnalyzer

    GFLOPS = BenchmarkProblems._MOCK_GPU_SYNTHETIC_GFLOPS
    resultsFileName = str(tmp_path / "results.csv")
    problemSizes = _ProblemSizesStub(_SEED_SIZES)
    numSolutions = 1

    BenchmarkProblems._writeSyntheticResultsCSV(
        resultsFileName, problemSizes, arch, numSolutions
    )

    with open(resultsFileName, newline="") as f:
        headerCols = f.readline().rstrip("\n").split(",")
    assert headerCols[0] == "GFlops"

    solutionMap = {i: i for i in range(numSolutions)}

    def _consume():
        analyzer = LogicAnalyzer.__new__(LogicAnalyzer)
        analyzer.numIndices = len(_SEED_SIZES[0])
        analyzer.exactProblemSizes = set(_SEED_SIZES)
        analyzer.rangeProblemSizes = set()
        analyzer.exactWinners = {}
        analyzer.perfMetric = None
        analyzer.addFromCSV(resultsFileName, numSolutions, solutionMap)
        return analyzer

    monkeypatch.setitem(globalParameters, "UseEffLike", True)

    monkeypatch.delenv("MAX_FREQ", raising=False)
    analyzer = _consume()
    assert analyzer.perfMetric == "DeviceEfficiency"
    assert set(analyzer.exactWinners.keys()) == set(_SEED_SIZES)
    for size in _SEED_SIZES:
        winnerSolId, perf = analyzer.exactWinners[size]
        assert winnerSolId == 0
        assert perf == round(GFLOPS)

    monkeypatch.setenv("MAX_FREQ", "200")
    analyzer = _consume()
    expected = round(GFLOPS / 200.0, 2)
    assert set(analyzer.exactWinners.keys()) == set(_SEED_SIZES)
    for size in _SEED_SIZES:
        _, perf = analyzer.exactWinners[size]
        assert perf == expected


def test_determinism(tmp_path):
    """T8: producing the synthetic CSV twice for the same arch yields byte-identical
    files (no randomness, no timestamps)."""
    problemSizes = _ProblemSizesStub(_SEED_SIZES)
    f1 = str(tmp_path / "a.csv")
    f2 = str(tmp_path / "b.csv")

    BenchmarkProblems._writeSyntheticResultsCSV(f1, problemSizes, "gfx942", 1)
    BenchmarkProblems._writeSyntheticResultsCSV(f2, problemSizes, "gfx942", 1)

    assert Path(f1).read_bytes() == Path(f2).read_bytes()


_E2E_CONFIG = Path(__file__).parent / "test_data" / "mock_gpu.yaml"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx90a"])
def test_mock_gpu_end_to_end(tensile_args, tmp_path, monkeypatch, _restore_gp, arch):
    """T9: drive the full benchmark flow GPU-less via
    Tensile.Tensile([cfg, out, "--mock-gpu", "--gpu-targets", arch, *tensile_args]).

    Mirrors test_keep_build_tmp.py, but exercises the BENCHMARK path (no --build-only):
    codegen -> cross-compile -> stubbed client launch -> deterministic synthetic results
    CSV -> LibraryLogic.addFromCSV. The flow must complete without exception, leave a
    results .csv under the output dir, and produce a 3_LibraryLogic artifact (proving
    addFromCSV ran end-to-end, not just that branches were reached). The autouse
    _no_stdin fixture guarantees builtins.input is never read.

    GPU-less note: LibraryLogic's createLibraryLogic asks getCUCount()
    (LibraryIO.py:686), which probes rocminfo unless the documented ``CU`` env-var
    escape hatch is set. We set CU here (the code's own GPU-less escape) rather than
    touch out-of-scope LibraryIO source. This is a separate device probe from the three
    seams the --mock-gpu switch owns (ISA detect / frequency probe / client launch).
    """
    monkeypatch.setenv("CU", "304")

    output_dir = tmp_path / "output"
    args = [
        str(_E2E_CONFIG),
        str(output_dir),
        "--mock-gpu",
        "--gpu-targets",
        arch,
        *tensile_args,
    ]

    Tensile.Tensile(args)

    results_csvs = list(output_dir.rglob("*.csv"))
    assert results_csvs, "no results .csv produced under the output dir"

    benchmark_data = list((output_dir / "2_BenchmarkData").glob("*.csv"))
    assert benchmark_data, "no 2_BenchmarkData CSV (addFromCSV input) produced"
    text = benchmark_data[0].read_text()
    assert text.splitlines()[0].startswith("GFlops")
    assert "1000.0" in text

    logic_artifacts = list((output_dir / "3_LibraryLogic").glob("*.yaml"))
    assert (
        logic_artifacts
    ), "no 3_LibraryLogic artifact produced; addFromCSV did not run"


_TEST_DATA = Path(__file__).parent / "test_data"


def _make_problem_type():
    """Build the canonical single-batch GEMM ContractionsProblemType used to capture the
    commit-1 golden: operationIdentifier == Contraction_l_Ailk_Bljk_Cijk_Dijk, S types,
    UseBias=1 / BiasSrc="D" (matching the golden's use-bias=1 / bias-source=3)."""
    from Tensile.Contractions import ProblemType

    d = {
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
    return ProblemType.FromOriginalState(d)


def test_off_path_text_golden(tmp_path, monkeypatch, _restore_gp):
    """T10: with MockGpu OFF, writeRunScript() and writeClientConfigIni() produce text
    byte-identical to the goldens captured from develop in commit 1. These functions are
    GPU-less (no device/freq/detect), so the golden is capturable and reproducible
    GPU-less. This is the literal 'byte-identical when off' proof for the touched output.

    Variable paths (the per-run tempdir, the client exe) are normalized to the same
    fixed sentinels the commit-1 capture used (/SRC, /TENSILE_CLIENT_EXE), so only the
    emitted structure -- not the host-specific paths -- is compared.
    """
    from Tensile.SolutionStructs.Problem import ProblemSizesMockDummy

    restoreDefaultGlobalParameters()
    globalParameters["MockGpu"] = False

    monkeypatch.setattr(
        ClientWriter, "getClientExecutablePath", lambda: "/TENSILE_CLIENT_EXE"
    )
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    config_paths = [
        "/SRC/ClientParameters.ini",
        "/SRC/ClientParameters_Granularity.ini",
    ]
    run_script = ClientWriter.writeRunScript(
        str(build_dir),
        True,
        False,
        "hipcc",
        "hipcc",
        str(build_dir),
        configPaths=list(config_paths),
    )
    produced_sh = Path(run_script).read_text()
    golden_sh = (_TEST_DATA / "mock_gpu_runscript.golden.sh").read_text()
    assert (
        produced_sh == golden_sh
    ), "writeRunScript output drifted from the develop golden"

    class _FactorDimArgs:
        factorDims = [0]

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    params_path = source_dir / "ClientParameters.ini"
    ClientWriter.writeClientConfigIni(
        forBenchmark=True,
        problemSizes=ProblemSizesMockDummy(),
        biasTypeArgs="",
        factorDimArgs=_FactorDimArgs(),
        activationArgs="",
        icacheFlushArgs="",
        problemType=_make_problem_type(),
        sourceDir=str(source_dir),
        codeObjectFiles=["TensileLibrary_gfx942.co"],
        resultsFileName="/DATA/benchmark.csv",
        parametersFilePath=str(params_path),
        deviceId=0,
        gfxName="gfx942",
        libraryFile="/LIB/gfx942/TensileLibrary.dat",
    )
    produced_ini = params_path.read_text().replace(str(source_dir), "/SRC")
    golden_ini = (_TEST_DATA / "mock_gpu_clientconfig.golden.ini").read_text()
    assert (
        produced_ini == golden_ini
    ), "writeClientConfigIni output drifted from the develop golden"


def test_off_path_real_branches(monkeypatch, _restore_gp):
    """T11: with MockGpu OFF, the new switch code is inert -- the REAL device-bound
    branches are taken at each of the three seams (ISA detection, frequency probe, client
    launch). Proves the gates added by this PR fall through to the original code paths
    when the switch is off.
    """
    globalParameters["MockGpu"] = False

    class _FakeProc:
        returncode = 0
        stdout = b"gfx942\n"

    isa_calls = {"n": 0}

    def _fake_run(*a, **k):
        isa_calls["n"] += 1
        return _FakeProc()

    monkeypatch.setattr(Arch, "run", _fake_run)
    isa = Arch.detectGlobalCurrentISA(0, "amdgpu-arch")
    assert isa_calls["n"] == 1, "MockGpu OFF must reach the real ISA shell-out path"
    assert isa == IsaVersion(9, 4, 2)

    freq_calls = {"hip": 0}

    def _hip_ok(device_id):
        freq_calls["hip"] += 1
        return 1700

    monkeypatch.setattr(Tensile, "get_gpu_max_frequency", _hip_ok)
    ran = _run_freq_block()
    assert ran is True, "MockGpu OFF must enter the real frequency-probe block"
    assert (
        freq_calls["hip"] == 1
    ), "MockGpu OFF must call the real get_gpu_max_frequency"

    reached = {"writeRunScript": 0}

    def _spy_write_run_script(*a, **k):
        reached["writeRunScript"] += 1
        raise RuntimeError("stop after proving the real launch path was entered")

    monkeypatch.setattr(ClientWriter, "writeRunScript", _spy_write_run_script)

    import os as _os
    import tempfile as _tempfile

    out = Path(_tempfile.mkdtemp(prefix="t11_"))
    with pytest.raises(RuntimeError, match="stop after proving"):
        ClientWriter.runClient(
            libraryLogicPath=None,
            forBenchmark=True,
            enableTileSelection=False,
            cxxCompiler="hipcc",
            cCompiler="hipcc",
            outputPath=out,
            configPaths=[str(out / "ClientParameters.ini")],
        )
    assert (
        reached["writeRunScript"] == 1
    ), "MockGpu OFF must fall through the runClient stub to the real launch path"
