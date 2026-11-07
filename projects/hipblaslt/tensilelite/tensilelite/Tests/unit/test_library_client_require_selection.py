# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The LibraryClient step fails when the library selects no solution.

ClientWriter passes --require-selected-solution to the client for the LibraryClient run only
(the benchmark runs do not use --best-solution), and Tensile fails when that client run exits
non-zero, as it does for the benchmark step (ExitOnFails).
"""

import types
from pathlib import Path
from unittest.mock import patch

import pytest

import tensilelite.ClientWriter as CW
from tensilelite import Tensile as TensileModule
from tensilelite.Common.GlobalParameters import globalParameters

pytestmark = pytest.mark.unit


def _runScript(tmp_path, monkeypatch, forBenchmark):
    monkeypatch.setitem(globalParameters, "MXScaleFormat", None)
    monkeypatch.setitem(globalParameters, "TimingInstrumentation", False)
    monkeypatch.setitem(globalParameters, "PinClocks", False)
    monkeypatch.setitem(globalParameters, "DataInitTypeA", 0)
    monkeypatch.setitem(globalParameters, "DataInitTypeB", 0)
    buildDir = tmp_path / "build"
    buildDir.mkdir()
    with patch.object(CW, "getClientExecutablePath", return_value="/fake/client"):
        scriptName = CW.writeRunScript(
            path=str(buildDir),
            forBenchmark=forBenchmark,
            enableTileSelection=False,
            cxxCompiler="/usr/bin/clang++",
            cCompiler="/usr/bin/clang",
            buildDir=str(buildDir),
            configPaths=["/cfg/A.ini", "/cfg/B.ini"],
        )
    return [
        line
        for line in Path(scriptName).read_text().splitlines()
        if line.startswith("/fake/client")
    ]


def test_library_client_run_requires_a_selected_solution(tmp_path, monkeypatch):
    lines = _runScript(tmp_path, monkeypatch, forBenchmark=False)
    assert lines == [
        "/fake/client --config-file /cfg/A.ini --best-solution --require-selected-solution",
        "/fake/client --config-file /cfg/B.ini --best-solution --require-selected-solution",
    ]


def test_benchmark_run_does_not_get_the_option(tmp_path, monkeypatch):
    lines = _runScript(tmp_path, monkeypatch, forBenchmark=True)
    assert len(lines) == 2
    assert not any("--require-selected-solution" in line for line in lines)


def _executeLibraryClientStep(monkeypatch, tmp_path, returncode):
    monkeypatch.setattr(TensileModule.ClientWriter, "main", lambda *_a, **_kw: returncode)
    TensileModule.executeStepsInConfig(
        config={"LibraryClient": None},
        outputPath=tmp_path,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler=types.SimpleNamespace(default_args=[])),
        isaInfoMap={(9, 5, 0): types.SimpleNamespace()},
        cCompiler="cc",
        debugConfig=types.SimpleNamespace(
            splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False
        ),
        deviceId=0,
        probSolDict={},
    )


def test_failed_library_client_run_fails_tensile(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(globalParameters, "ExitOnFails", 1)
    with pytest.raises(SystemExit) as exc:
        _executeLibraryClientStep(monkeypatch, tmp_path, returncode=1)
    assert exc.value.code != 0
    assert "LibraryClient exited with code 1" in capsys.readouterr().out


@pytest.mark.parametrize("exitOnFails, returncode", [(1, 0), (0, 1)])
def test_library_client_run_passes(monkeypatch, tmp_path, exitOnFails, returncode):
    monkeypatch.setitem(globalParameters, "ExitOnFails", exitOnFails)
    _executeLibraryClientStep(monkeypatch, tmp_path, returncode=returncode)
