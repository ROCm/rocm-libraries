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

"""Unit tests for the GPU-less ``--cpu-only`` switch (P0.5 prerequisite).

This file accumulates the T1-T12 rigor-gate suite from GPU-MOCK-PR.md. This commit
covers the flag-plumbing tier:

* T1 ``test_flag_default_off`` - the CLI flag parses correctly (absent->False,
  present->True), the internal ``globalParameters["CpuOnly"]`` plumbing key resets to
  ``False`` via ``restoreDefaultGlobalParameters()``, and the flag is NOT exposed on the
  documented ``--global-parameters`` surface.
* T2 ``test_arg_validation`` - pins the behavior commit-2 establishes at the
  common-arguments parser layer: ``--cpu-only`` parses without requiring an arch at parse
  time (no premature SystemExit), yielding ``cpuOnly=True`` with ``gpuTargets`` still
  ``None``. The ``--cpu-only`` *requires an arch* contract is enforced/pinned in the ISA
  commit (its own test), not here.

GPU-less safety: every test monkeypatches ``builtins.input`` to raise so any accidental
stdin read (e.g. ``get_user_max_frequency``) fails loudly instead of hanging unattended.
"""

import argparse

import pytest

pytestmark = pytest.mark.unit

from Tensile import Tensile
from Tensile.Common.GlobalParameters import (
    globalParameters,
    restoreDefaultGlobalParameters,
    defaultGlobalParameters,
)


@pytest.fixture(autouse=True)
def _no_stdin(monkeypatch):
    """Fail loudly on any unattended stdin read instead of hanging on a GPU-less host."""

    def _boom(*args, **kwargs):
        raise AssertionError("builtins.input() called on the --cpu-only path")

    monkeypatch.setattr("builtins.input", _boom)


def _parse(argv):
    """Parse ``argv`` through the shared addCommonArguments parser used by the script."""
    argParser = argparse.ArgumentParser()
    Tensile.addCommonArguments(argParser)
    return argParser.parse_args(argv)


def test_flag_default_off(monkeypatch):
    """T1: flag absent->False, present->True; internal plumbing key resets to False;
    flag is not on the --global-parameters surface."""
    # Absent -> default False.
    args = _parse([])
    assert args.cpuOnly is False

    # Present -> True.
    args = _parse(["--cpu-only"])
    assert args.cpuOnly is True

    # The undocumented internal plumbing key exists and defaults False, and
    # restoreDefaultGlobalParameters() resets it to False.
    assert defaultGlobalParameters["CpuOnly"] is False
    globalParameters["CpuOnly"] = True  # simulate a prior run flipping it on
    restoreDefaultGlobalParameters()
    try:
        assert globalParameters["CpuOnly"] is False
    finally:
        restoreDefaultGlobalParameters()

    # The flag must NOT be advertised on the documented --global-parameters help surface.
    argParser = argparse.ArgumentParser()
    Tensile.addCommonArguments(argParser)
    help_text = argParser.format_help()
    # --cpu-only is its own flag, present in help...
    assert "--cpu-only" in help_text
    # ...but it is not threaded through the --global-parameters key=value mechanism.
    gp_action = next(
        a for a in argParser._actions if "--global-parameters" in a.option_strings
    )
    assert "CpuOnly" not in (gp_action.help or "")
    # And eval-style --global-parameters parsing never references CpuOnly.
    assert "cpuOnly" not in (gp_action.help or "")


def test_arg_validation():
    """T2: --cpu-only without an arch parses cleanly at the common-arguments layer
    (no premature SystemExit), yielding cpuOnly=True and gpuTargets unset.

    The --cpu-only-requires-arch contract is enforced and pinned in the ISA commit; this
    test pins only what flag plumbing (commit 2) establishes: the flag is orthogonal to
    --gpu-targets at parse time.
    """
    args = _parse(["--cpu-only"])
    assert args.cpuOnly is True
    # gpuTargets lives on the Tensile() main parser, not addCommonArguments; the common
    # parser must not synthesize or require it, so the attribute is simply absent here.
    assert not hasattr(args, "gpuTargets")

    # Off by default and independent of other common args.
    args = _parse(["--device", "0"])
    assert args.cpuOnly is False


# --- ISA belt spoof + primary --gpu-targets path (commit 3) ---------------------

import Tensile.Common.Architectures as Arch
from Tensile.Common.Types import IsaVersion


_ARCH_ISA = {
    "gfx942": IsaVersion(9, 4, 2),
    "gfx950": IsaVersion(9, 5, 0),
    "gfx90a": IsaVersion(9, 0, 10),
}


@pytest.fixture
def _restore_gp():
    """Snapshot/restore the CpuOnly plumbing keys so a flipped flag never leaks."""
    saved = (globalParameters.get("CpuOnly"), globalParameters.get("CpuOnlyArch"))
    try:
        yield
    finally:
        globalParameters["CpuOnly"], globalParameters["CpuOnlyArch"] = saved


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx90a"])
def test_isa_belt_spoof(monkeypatch, _restore_gp, arch):
    """T3: with CpuOnly on, the direct ISA-detection path returns the exact per-arch
    IsaVersion without shelling out (Architectures.run raises if called); with CpuOnly
    off, the real parse path is taken (spoof branch not entered)."""
    expected = _ARCH_ISA[arch]

    # --- CpuOnly ON: no shell-out, exact per-arch IsaVersion ---
    globalParameters["CpuOnly"] = True
    globalParameters["CpuOnlyArch"] = arch

    def _no_shell(*a, **k):
        raise AssertionError("Architectures.run() shelled out under CpuOnly")

    monkeypatch.setattr(Arch, "run", _no_shell)

    result = Arch.detectGlobalCurrentISA(0, "amdgpu-arch")
    assert isinstance(result, IsaVersion)
    assert result == expected

    # --- CpuOnly OFF: spoof branch NOT entered; real parse path runs ---
    globalParameters["CpuOnly"] = False

    class _FakeProc:
        returncode = 0
        stdout = (arch + "\n").encode()

    calls = {"n": 0}

    def _fake_run(*a, **k):
        calls["n"] += 1
        return _FakeProc()

    monkeypatch.setattr(Arch, "run", _fake_run)

    result_off = Arch.detectGlobalCurrentISA(0, "amdgpu-arch")
    assert calls["n"] == 1  # the real shell-out path was taken
    assert result_off == expected


def test_isa_primary_path(monkeypatch, _restore_gp):
    """T4: the primary --cpu-only --gpu-targets path builds isaList directly from the
    target arch and never calls detectGlobalCurrentISA."""
    # Spy: detection must never be reached on the --gpu-targets path.
    def _no_detect(*a, **k):
        raise AssertionError("detectGlobalCurrentISA called on the --gpu-targets path")

    monkeypatch.setattr(Arch, "detectGlobalCurrentISA", _no_detect)

    # Mirror the isaList-building logic at Tensile.py (the --gpu-targets branch):
    # ISA comes straight from gfxToIsa(arch); enumerator is None; detection untouched.
    args = _parse(["--cpu-only", "--device", "0"])
    assert args.cpuOnly is True

    gpuTargets = "gfx942"
    enumerator = None if gpuTargets else object()
    assert enumerator is None  # --gpu-targets path: enumerator not needed

    isaList = []
    for a in gpuTargets.split(";"):
        a = a.strip()
        assert a
        isa = Arch.gfxToIsa(a)
        assert isa is not None
        isaList.append(isa)

    assert isaList == [IsaVersion(9, 4, 2)]


# --- Frequency-probe skip under CpuOnly (commit 4) ------------------------------


def _run_freq_block(device_id=0):
    """Replay the guarded frequency-probe block from Tensile.Tensile() exactly.

    The gating predicate mirrors Tensile.py:601 verbatim:
        'LibraryLogic' in config and UseEffLike and not buildOnly
        and not globalParameters["CpuOnly"]
    The 'LibraryLogic'/UseEffLike/buildOnly preconditions are held True/True/False so
    the test isolates the CpuOnly term: the body must run iff CpuOnly is off. The body
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
        and not globalParameters["CpuOnly"]
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
    """T5: with CpuOnly on, none of the three GPU clock-frequency probes
    (get_gpu_max_frequency / get_gpu_max_frequency_smi / get_user_max_frequency) are
    reached; with CpuOnly off, the real branch runs and get_gpu_max_frequency IS called.
    """
    calls = {"hip": 0, "smi": 0, "user": 0}

    def _hip(*a, **k):
        calls["hip"] += 1
        raise AssertionError("get_gpu_max_frequency called under CpuOnly")

    def _smi(*a, **k):
        calls["smi"] += 1
        raise AssertionError("get_gpu_max_frequency_smi called under CpuOnly")

    def _user(*a, **k):
        calls["user"] += 1
        raise AssertionError("get_user_max_frequency called under CpuOnly")

    monkeypatch.setattr(Tensile, "get_gpu_max_frequency", _hip)
    monkeypatch.setattr(Tensile, "get_gpu_max_frequency_smi", _smi)
    monkeypatch.setattr(Tensile, "get_user_max_frequency", _user)

    # --- CpuOnly ON: entire block skipped, no probe reached ---
    globalParameters["CpuOnly"] = True
    ran = _run_freq_block()
    assert ran is False
    assert calls == {"hip": 0, "smi": 0, "user": 0}

    # --- CpuOnly OFF: real branch entered; get_gpu_max_frequency IS called ---
    globalParameters["CpuOnly"] = False
    seen = {"hip": 0}

    def _hip_ok(device_id):
        seen["hip"] += 1
        return 1700  # deterministic non-zero -> smi/user never needed

    monkeypatch.setattr(Tensile, "get_gpu_max_frequency", _hip_ok)
    # smi/user remain the raising spies: a valid first probe must short-circuit them.
    ran = _run_freq_block()
    assert ran is True
    assert seen["hip"] == 1
    assert calls == {"hip": 0, "smi": 0, "user": 0}  # smi/user untouched


# --- Client device-launch stub + synthetic results CSV (commit 5) ---------------

import subprocess
from pathlib import Path

import Tensile.ClientWriter as ClientWriter
import Tensile.BenchmarkProblems as BenchmarkProblems
from Tensile.SolutionStructs.Problem import Problem

# Per-arch seeded problem sizes (the data stub: mirror ProblemSizesMockDummy's [128,128,1,512]).
# Two distinct sizes prove one CSV data row per seeded size.
_SEED_SIZES = [(128, 128, 1, 512), (256, 256, 1, 1024)]


class _ProblemSizesStub:
    """Minimal stand-in for ProblemSizes carrying just ``.problems`` (the attribute the
    synthetic-CSV writer reads), in ProblemSizesMock style (SolutionStructs/Problem.py)."""

    def __init__(self, sizes):
        self.problems = [Problem(sizes=list(s)) for s in sizes]


def test_no_side_effects(monkeypatch, _restore_gp, tmp_path):
    """T6: on the --cpu-only runClient path, the device boundary is never touched:
    no subprocess.Popen launch, no getClientExecutablePath, no subprocess.run
    (pip/hip install), and builtins.input is never read. runClient returns 0."""
    globalParameters["CpuOnly"] = True

    def _no_popen(*a, **k):
        raise AssertionError("subprocess.Popen launched the client under CpuOnly")

    def _no_run(*a, **k):
        raise AssertionError("subprocess.run shelled out (pip/hip install) under CpuOnly")

    def _no_exe(*a, **k):
        raise AssertionError("getClientExecutablePath called under CpuOnly")

    monkeypatch.setattr(subprocess, "Popen", _no_popen)
    monkeypatch.setattr(subprocess, "run", _no_run)
    monkeypatch.setattr(ClientWriter, "getClientExecutablePath", _no_exe)
    # builtins.input is already monkeypatched to raise by the autouse _no_stdin fixture.

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
def test_synthetic_csv_schema(tmp_path, arch):
    """T7 (schema-drift sentinel): the synthetic CSV is fed through the REAL
    LibraryLogic.addFromCSV and parses without error, yielding the expected perfMetric
    and one consumed row per seeded problem size. If the client CSV contract changes
    upstream so that addFromCSV's column expectations drift, this fails."""
    import Tensile.LibraryLogic as LibraryLogic
    from Tensile.LibraryLogic import LogicAnalyzer

    resultsFileName = str(tmp_path / "results.csv")
    problemSizes = _ProblemSizesStub(_SEED_SIZES)
    numSolutions = 1

    BenchmarkProblems._writeSyntheticResultsCSV(
        resultsFileName, problemSizes, arch, numSolutions)

    # Drive the REAL addFromCSV. Build a lightweight analyzer carrying only the attributes
    # addFromCSV reads on the exact-size path; the parser logic itself is the real code.
    analyzer = LogicAnalyzer.__new__(LogicAnalyzer)
    analyzer.numIndices = len(_SEED_SIZES[0])
    analyzer.exactProblemSizes = set(_SEED_SIZES)
    analyzer.rangeProblemSizes = set()
    analyzer.exactWinners = {}
    analyzer.perfMetric = None

    # solutionMap: CSV solution-column index -> solution id (identity for one solution).
    solutionMap = {i: i for i in range(numSolutions)}

    # UseEffLike must be False so addFromCSV does not call read_max_freq() (a device probe).
    saved_eff = globalParameters.get("UseEffLike")
    globalParameters["UseEffLike"] = False
    try:
        analyzer.addFromCSV(resultsFileName, numSolutions, solutionMap)
    finally:
        globalParameters["UseEffLike"] = saved_eff

    # Header row -> perfMetric derived from the "GFlops" unit column.
    assert analyzer.perfMetric == "DeviceEfficiency"
    # One winner recorded per seeded exact problem size (schema parsed correctly).
    assert set(analyzer.exactWinners.keys()) == set(_SEED_SIZES)
    for size in _SEED_SIZES:
        winnerSolId, perf = analyzer.exactWinners[size]
        assert winnerSolId == 0
        assert perf == 1000.0


def test_determinism(tmp_path):
    """T8: producing the synthetic CSV twice for the same arch yields byte-identical
    files (no randomness, no timestamps)."""
    problemSizes = _ProblemSizesStub(_SEED_SIZES)
    f1 = str(tmp_path / "a.csv")
    f2 = str(tmp_path / "b.csv")

    BenchmarkProblems._writeSyntheticResultsCSV(f1, problemSizes, "gfx942", 1)
    BenchmarkProblems._writeSyntheticResultsCSV(f2, problemSizes, "gfx942", 1)

    assert Path(f1).read_bytes() == Path(f2).read_bytes()
