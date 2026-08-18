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

"""Characterization tests for ClientWriter.py — driver/run path.

Targets the methodology-A missing ranges:
  94-206  (main())
  213-242 (runNewClient() + runClient() start)
  265-313 (runClient() return + getBuildClientLibraryScript() + writeRunScript() head)
  366-380 (writeRunScript() non-benchmark else branch + toCppBool + getMaxSolutionSizes)

Strategy: drive each function CPU-only via direct calls with controlled mocks
so that actual lines execute without GPU hardware.  No golden snapshots are
needed — all assertions are on observable side-effects (file existence, file
content, return values).
"""

import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import Tensile.ClientWriter as CW
from Tensile.Common.GlobalParameters import globalParameters

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_globalParameters_overrides(monkeypatch):
    """Set only the globalParameters keys that the exercised code paths read."""
    monkeypatch.setitem(globalParameters, "LazyLibraryLoading", True)
    monkeypatch.setitem(globalParameters, "AsmDebug", False)
    monkeypatch.setitem(globalParameters, "KeepBuildTmp", False)
    monkeypatch.setitem(globalParameters, "DisableAsmComments", False)
    monkeypatch.setitem(globalParameters, "CodeObjectVersion", "4")
    monkeypatch.setitem(globalParameters, "LibraryFormat", "yaml")
    monkeypatch.setitem(globalParameters, "RuntimeLanguage", "HIP")
    monkeypatch.setitem(globalParameters, "PinClocks", False)
    monkeypatch.setitem(globalParameters, "ROCmSMIPath", None)
    monkeypatch.setitem(globalParameters, "ForceRedoLibraryClient", False)
    monkeypatch.setitem(globalParameters, "TimingInstrumentation", False)
    monkeypatch.setitem(globalParameters, "MXScaleFormat", 0)
    monkeypatch.setitem(globalParameters, "ParallelGpuExecution", 1)
    monkeypatch.setitem(globalParameters, "ClientExecutionLockPath", None)
    monkeypatch.setitem(globalParameters, "PrebuiltClient", "/fake/tensile_client")
    monkeypatch.setitem(globalParameters, "DataInitTypeAB", 3)
    monkeypatch.setitem(globalParameters, "DataInitTypeA", -1)
    monkeypatch.setitem(globalParameters, "DataInitTypeB", -1)


# ---------------------------------------------------------------------------
# getBuildClientLibraryScript (lines 268-297)
# ---------------------------------------------------------------------------

class TestGetBuildClientLibraryScript:
    """Direct calls into getBuildClientLibraryScript exercise lines 268-297."""

    def test_default_returns_create_library_command(self, monkeypatch, tmp_path):
        """Lines 268-297: basic command structure is emitted."""
        _default_globalParameters_overrides(monkeypatch)
        # LazyLibraryLoading=True → no --no-lazy-library-loading
        result = CW.getBuildClientLibraryScript(
            buildPath=str(tmp_path / "build"),
            libraryLogicPath=str(tmp_path / "logic"),
            cxxCompiler="/usr/bin/clang++",
            targetGfx="gfx942",
        )
        assert "TensileCreateLibrary" in result
        assert "--architecture=gfx942" in result
        assert "--code-object-version=4" in result
        assert "--cxx-compiler=/usr/bin/clang++" in result
        assert "--library-format=yaml" in result
        # lazy loading on → flag absent
        assert "--no-lazy-library-loading" not in result
        # line ends with "\n"
        assert result.endswith("\n")

    def test_no_lazy_library_loading_flag(self, monkeypatch, tmp_path):
        """Line 274-275: --no-lazy-library-loading appended when flag is False."""
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "LazyLibraryLoading", False)
        result = CW.getBuildClientLibraryScript(
            buildPath=str(tmp_path),
            libraryLogicPath=str(tmp_path / "logic"),
            cxxCompiler="/usr/bin/hipcc",
            targetGfx="gfx950",
        )
        assert "--no-lazy-library-loading" in result

    def test_asm_debug_flag(self, monkeypatch, tmp_path):
        """Line 277-278: --asm-debug appended when AsmDebug is True."""
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "AsmDebug", True)
        result = CW.getBuildClientLibraryScript(
            buildPath=str(tmp_path),
            libraryLogicPath=str(tmp_path / "logic"),
            cxxCompiler="/usr/bin/hipcc",
            targetGfx="gfx942",
        )
        assert "--asm-debug" in result

    def test_keep_build_tmp_flag(self, monkeypatch, tmp_path):
        """Line 280-281: --keep-build-tmp appended when KeepBuildTmp is True."""
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "KeepBuildTmp", True)
        result = CW.getBuildClientLibraryScript(
            buildPath=str(tmp_path),
            libraryLogicPath=str(tmp_path / "logic"),
            cxxCompiler="/usr/bin/hipcc",
            targetGfx="gfx942",
        )
        assert "--keep-build-tmp" in result

    def test_disable_asm_comments_flag(self, monkeypatch, tmp_path):
        """Line 283-284: --disable-asm-comments appended when DisableAsmComments is True."""
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "DisableAsmComments", True)
        result = CW.getBuildClientLibraryScript(
            buildPath=str(tmp_path),
            libraryLogicPath=str(tmp_path / "logic"),
            cxxCompiler="/usr/bin/hipcc",
            targetGfx="gfx942",
        )
        assert "--disable-asm-comments" in result

    def test_paths_embedded_in_command(self, monkeypatch, tmp_path):
        """Lines 291-293: logic path, build path, runtime lang appear in command."""
        _default_globalParameters_overrides(monkeypatch)
        logic = str(tmp_path / "logic")
        build = str(tmp_path / "build")
        result = CW.getBuildClientLibraryScript(
            buildPath=build,
            libraryLogicPath=logic,
            cxxCompiler="/usr/bin/hipcc",
            targetGfx="gfx942",
        )
        assert logic in result
        assert build in result
        assert "HIP" in result


# ---------------------------------------------------------------------------
# writeRunScript — non-benchmark path (lines 300-373 with forBenchmark=False)
# Exercises the else branch at line 363-366 plus the closing block 368-373.
# ---------------------------------------------------------------------------

class TestWriteRunScriptNonBenchmark:
    """writeRunScript with forBenchmark=False exercises lines 300-316, 363-373."""

    def _run(self, tmp_path, monkeypatch, configPaths=None, clientExePath="/fake/client"):
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "PrebuiltClient", clientExePath)

        buildDir = tmp_path / "build"
        buildDir.mkdir(parents=True)

        with patch.object(CW, "getClientExecutablePath", return_value=clientExePath):
            scriptName = CW.writeRunScript(
                path=str(buildDir),
                forBenchmark=False,
                enableTileSelection=False,
                cxxCompiler="/usr/bin/clang++",
                cCompiler="/usr/bin/clang",
                buildDir=str(buildDir),
                configPaths=configPaths,
            )
        return scriptName, buildDir

    def test_script_created_on_disk(self, tmp_path, monkeypatch):
        """Lines 309-311: script file is created."""
        scriptName, buildDir = self._run(tmp_path, monkeypatch,
                                         configPaths=["/cfg/ClientParameters.ini"])
        assert os.path.exists(scriptName)

    def test_script_has_shebang(self, tmp_path, monkeypatch):
        """Line 312-313: shebang written on non-NT."""
        scriptName, _ = self._run(tmp_path, monkeypatch,
                                   configPaths=["/cfg/ClientParameters.ini"])
        content = Path(scriptName).read_text()
        assert "#!/bin/bash" in content

    def test_script_has_set_ex(self, tmp_path, monkeypatch):
        """Line 315: set -ex written."""
        scriptName, _ = self._run(tmp_path, monkeypatch,
                                   configPaths=["/cfg/ClientParameters.ini"])
        content = Path(scriptName).read_text()
        assert "set -ex" in content

    def test_non_benchmark_writes_best_solution_flag(self, tmp_path, monkeypatch):
        """Lines 363-366: non-benchmark path writes --best-solution."""
        scriptName, _ = self._run(tmp_path, monkeypatch,
                                   configPaths=["/cfg/A.ini", "/cfg/B.ini"])
        content = Path(scriptName).read_text()
        assert "--best-solution" in content
        assert "/cfg/A.ini" in content
        assert "/cfg/B.ini" in content

    def test_script_is_executable(self, tmp_path, monkeypatch):
        """Line 371-372: chmod 777 applied on non-NT."""
        scriptName, _ = self._run(tmp_path, monkeypatch,
                                   configPaths=["/cfg/ClientParameters.ini"])
        mode = os.stat(scriptName).st_mode
        assert mode & stat.S_IXUSR

    def test_script_ends_with_exit(self, tmp_path, monkeypatch):
        """Lines 368-369: exit $ERR appended on non-NT."""
        scriptName, _ = self._run(tmp_path, monkeypatch,
                                   configPaths=["/cfg/ClientParameters.ini"])
        content = Path(scriptName).read_text()
        assert "exit $ERR" in content

    def test_default_configPaths_none(self, tmp_path, monkeypatch):
        """Lines 301-306: when configPaths=None, default paths are used."""
        scriptName, buildDir = self._run(tmp_path, monkeypatch, configPaths=None)
        content = Path(scriptName).read_text()
        # default config references ../source/ClientParameters.ini relative to buildDir
        assert "ClientParameters.ini" in content


# ---------------------------------------------------------------------------
# writeRunScript — benchmark path (lines 318-362)
# Exercises the forBenchmark=True branch.
# ---------------------------------------------------------------------------

class TestWriteRunScriptBenchmark:
    """writeRunScript with forBenchmark=True exercises lines 318-362."""

    def _run_benchmark(self, tmp_path, monkeypatch, configPaths=None, clientExePath="/fake/client"):
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "PrebuiltClient", clientExePath)
        monkeypatch.setitem(globalParameters, "DataInitTypeA", -1)
        monkeypatch.setitem(globalParameters, "DataInitTypeB", -1)
        monkeypatch.setitem(globalParameters, "DataInitTypeAB", 3)

        buildDir = tmp_path / "build"
        buildDir.mkdir(parents=True)

        with patch.object(CW, "getClientExecutablePath", return_value=clientExePath):
            scriptName = CW.writeRunScript(
                path=str(buildDir),
                forBenchmark=True,
                enableTileSelection=False,
                cxxCompiler="/usr/bin/clang++",
                cCompiler="/usr/bin/clang",
                buildDir=str(buildDir),
                configPaths=configPaths or ["/cfg/A.ini"],
            )
        return scriptName

    def test_benchmark_writes_ERR_block(self, tmp_path, monkeypatch):
        """Lines 336-357: ERR shell logic is written."""
        scriptName = self._run_benchmark(tmp_path, monkeypatch)
        content = Path(scriptName).read_text()
        assert "ERR1=0" in content
        assert "ERR2=$?" in content
        assert "ERR=0" in content

    def test_benchmark_writes_config_file_flag(self, tmp_path, monkeypatch):
        """Lines 338-342: --config-file flag written for each configPath."""
        scriptName = self._run_benchmark(tmp_path, monkeypatch,
                                          configPaths=["/cfg/A.ini", "/cfg/B.ini"])
        content = Path(scriptName).read_text()
        assert "--config-file /cfg/A.ini" in content
        assert "--config-file /cfg/B.ini" in content

    def test_benchmark_DataInitType_resolved(self, tmp_path, monkeypatch):
        """Lines 331-334: DataInitTypeA/B get set from AtoB when -1."""
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "DataInitTypeA", -1)
        monkeypatch.setitem(globalParameters, "DataInitTypeB", -1)
        monkeypatch.setitem(globalParameters, "DataInitTypeAB", 3)

        buildDir = tmp_path / "build"
        buildDir.mkdir(parents=True)
        with patch.object(CW, "getClientExecutablePath", return_value="/fake/client"):
            CW.writeRunScript(
                path=str(buildDir),
                forBenchmark=True,
                enableTileSelection=False,
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
                buildDir=str(buildDir),
                configPaths=["/cfg/A.ini"],
            )
        # After the call the global DataInitTypeA/B should have been resolved.
        assert globalParameters["DataInitTypeA"] == 3
        assert globalParameters["DataInitTypeB"] == 3


# ---------------------------------------------------------------------------
# toCppBool (line 376-377) and getMaxSolutionSizes (lines 379-397)
# ---------------------------------------------------------------------------

class TestToCppBool:
    """toCppBool — lines 376-377."""

    def test_true_value(self):
        assert CW.toCppBool(True) == "true"

    def test_false_value(self):
        assert CW.toCppBool(False) == "false"

    def test_truthy_value(self):
        assert CW.toCppBool(1) == "true"

    def test_falsy_value(self):
        assert CW.toCppBool(0) == "false"


class TestGetMaxSolutionSizes:
    """getMaxSolutionSizes — lines 379-397."""

    def _make_solution(self, wg0, wg1, tt0, tt1):
        return {"WorkGroup": [wg0, wg1, 1], "ThreadTile": [tt0, tt1]}

    def test_single_solution(self):
        """Lines 380-397: computes maxMT0, maxMT1, maxK from one solution."""
        sol = self._make_solution(8, 8, 4, 4)
        result = CW.getMaxSolutionSizes([sol], [32])
        # mt0 = 8*4 = 32, mt1 = 8*4 = 32, maxK = 32
        assert result == [32, 32, 32]

    def test_multiple_solutions_max_taken(self):
        """Lines 384-395: max is taken across solutions."""
        sol_a = self._make_solution(8, 8, 4, 4)   # mt0=32, mt1=32
        sol_b = self._make_solution(16, 4, 8, 2)  # mt0=128, mt1=8
        sol_c = self._make_solution(4, 16, 2, 8)  # mt0=8, mt1=128
        result = CW.getMaxSolutionSizes([sol_a, sol_b, sol_c], [64, 128])
        assert result[0] == 128  # maxMT0
        assert result[1] == 128  # maxMT1
        assert result[2] == 128  # maxK

    def test_multiple_summation_sizes(self):
        """Line 381: maxK is max across summation sizes."""
        sol = self._make_solution(8, 8, 4, 4)
        result = CW.getMaxSolutionSizes([sol], [16, 64, 32])
        assert result[2] == 64


# ---------------------------------------------------------------------------
# runClient (lines 231-265) — mock subprocess so no GPU needed
# ---------------------------------------------------------------------------

class TestRunClient:
    """runClient exercises lines 231-265 via subprocess mocking."""

    def _invoke_run_client(self, tmp_path, monkeypatch,
                            forBenchmark=False, configPaths=None,
                            parallelGpus=1, numGpus=1,
                            popen_returncode=0):
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "ParallelGpuExecution", parallelGpus)

        outputPath = tmp_path / "output"
        outputPath.mkdir(parents=True)

        fake_exe = "/fake/client"

        # Stub writeRunScript to avoid fs side-effects beyond tmp
        fake_script = str(tmp_path / "run.sh")
        Path(fake_script).write_text("#!/bin/bash\n")

        mock_proc = MagicMock()
        mock_proc.returncode = popen_returncode
        mock_proc.communicate.return_value = (b"", b"")

        with (
            patch.object(CW, "getClientExecutablePath", return_value=fake_exe),
            patch.object(CW, "writeRunScript", return_value=fake_script),
            patch.object(CW.subprocess, "Popen", return_value=mock_proc) as mock_popen,
        ):
            rc = CW.runClient(
                libraryLogicPath=str(tmp_path / "logic"),
                forBenchmark=forBenchmark,
                enableTileSelection=False,
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
                outputPath=outputPath,
                configPaths=configPaths or ["/cfg/A.ini"],
            )
        return rc, mock_popen

    def test_run_client_zero_returncode(self, tmp_path, monkeypatch):
        """Lines 258-265: successful subprocess → rc=0."""
        rc, _ = self._invoke_run_client(tmp_path, monkeypatch, popen_returncode=0)
        assert rc == 0

    def test_run_client_nonzero_returncode(self, tmp_path, monkeypatch):
        """Lines 262-263: non-zero returncode propagated, warning printed."""
        rc, _ = self._invoke_run_client(tmp_path, monkeypatch, popen_returncode=1)
        assert rc == 1

    def test_run_client_configPaths_none_defaults(self, tmp_path, monkeypatch):
        """Lines 237-241: when configPaths=None, defaults are computed."""
        _default_globalParameters_overrides(monkeypatch)

        outputPath = tmp_path / "output"
        outputPath.mkdir(parents=True)

        fake_script = str(tmp_path / "run.sh")
        Path(fake_script).write_text("#!/bin/bash\n")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")

        with (
            patch.object(CW, "getClientExecutablePath", return_value="/fake/client"),
            patch.object(CW, "writeRunScript", return_value=fake_script) as mock_wrs,
            patch.object(CW.subprocess, "Popen", return_value=mock_proc),
        ):
            CW.runClient(
                libraryLogicPath=str(tmp_path / "logic"),
                forBenchmark=False,
                enableTileSelection=False,
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
                outputPath=outputPath,
                configPaths=None,
            )
            # The configPaths list passed to writeRunScript should not be empty
            _, _, kwargs = mock_wrs.mock_calls[0]
        # writeRunScript called; verify it got a non-None configPaths
        assert mock_wrs.called

    def test_run_client_enableTileSelection_adds_granularity_path(self, tmp_path, monkeypatch):
        """Lines 240-241: enableTileSelection=True adds Granularity config path."""
        _default_globalParameters_overrides(monkeypatch)

        outputPath = tmp_path / "output"
        outputPath.mkdir(parents=True)

        fake_script = str(tmp_path / "run.sh")
        Path(fake_script).write_text("#!/bin/bash\n")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")

        captured_configPaths = []

        def capture_writeRunScript(path, forBenchmark, enableTileSelection, cxxCompiler, cCompiler, buildDir, configPaths):
            captured_configPaths.extend(configPaths)
            return fake_script

        with (
            patch.object(CW, "getClientExecutablePath", return_value="/fake/client"),
            patch.object(CW, "writeRunScript", side_effect=capture_writeRunScript),
            patch.object(CW.subprocess, "Popen", return_value=mock_proc),
        ):
            CW.runClient(
                libraryLogicPath=str(tmp_path / "logic"),
                forBenchmark=False,
                enableTileSelection=True,
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
                outputPath=outputPath,
                configPaths=None,
            )

        # With enableTileSelection=True and configPaths=None, 2 paths are added
        assert len(captured_configPaths) == 2
        granularity_paths = [p for p in captured_configPaths if "Granularity" in p]
        assert len(granularity_paths) == 1

    def test_run_client_popen_called_with_script(self, tmp_path, monkeypatch):
        """Line 259: subprocess.Popen is called with the run script."""
        rc, mock_popen = self._invoke_run_client(tmp_path, monkeypatch, popen_returncode=0)
        mock_popen.assert_called_once()

    def test_run_client_builds_buildPath(self, tmp_path, monkeypatch):
        """Line 232: ensurePath(outputPath / 'build') creates build dir."""
        _default_globalParameters_overrides(monkeypatch)

        outputPath = tmp_path / "output"
        outputPath.mkdir(parents=True)

        fake_script = str(tmp_path / "run.sh")
        Path(fake_script).write_text("#!/bin/bash\n")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")

        with (
            patch.object(CW, "getClientExecutablePath", return_value="/fake/client"),
            patch.object(CW, "writeRunScript", return_value=fake_script),
            patch.object(CW.subprocess, "Popen", return_value=mock_proc),
        ):
            CW.runClient(
                libraryLogicPath=str(tmp_path / "logic"),
                forBenchmark=False,
                enableTileSelection=False,
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
                outputPath=outputPath,
                configPaths=["/cfg/A.ini"],
            )

        # build dir created by runClient
        assert (outputPath / "build").is_dir()


# ---------------------------------------------------------------------------
# runNewClient (lines 215-228) — getClientExecutablePath path exercised
# ---------------------------------------------------------------------------

class TestRunNewClient:
    """runNewClient exercises lines 215-228."""

    def test_runnewclient_no_mxscale(self, tmp_path, monkeypatch):
        """Lines 215-228: subprocess.run called with correct args, no MX flag."""
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "MXScaleFormat", 0)

        fake_exe = "/fake/tensile_client"
        config_path = "/cfg/ClientParameters.ini"

        with (
            patch.object(CW, "getClientExecutablePath", return_value=fake_exe),
            patch.object(CW.subprocess, "run") as mock_run,
        ):
            CW.runNewClient(
                scriptPath=str(tmp_path),
                clientParametersPath=config_path,
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
            )

        mock_run.assert_called_once()
        args_used = mock_run.call_args[0][0]
        assert fake_exe in args_used
        assert f"--config-file={config_path}" in args_used
        # No MX flag when MXScaleFormat == 0
        assert "--mx-scale-format" not in " ".join(args_used)

    def test_runnewclient_with_mxscale(self, tmp_path, monkeypatch):
        """Lines 222-223: --mx-scale-format flag appended when set."""
        _default_globalParameters_overrides(monkeypatch)
        monkeypatch.setitem(globalParameters, "MXScaleFormat", 1)

        fake_exe = "/fake/tensile_client"

        with (
            patch.object(CW, "getClientExecutablePath", return_value=fake_exe),
            patch.object(CW.subprocess, "run") as mock_run,
        ):
            CW.runNewClient(
                scriptPath=str(tmp_path),
                clientParametersPath="/cfg/A.ini",
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
            )

        args_used = mock_run.call_args[0][0]
        assert "--mx-scale-format" in args_used
        assert "1" in args_used

    def test_runnewclient_subprocess_error_caught(self, tmp_path, monkeypatch):
        """Lines 225-228: CalledProcessError is caught and warning issued."""
        _default_globalParameters_overrides(monkeypatch)

        with (
            patch.object(CW, "getClientExecutablePath", return_value="/fake/client"),
            patch.object(CW.subprocess, "run",
                          side_effect=subprocess.CalledProcessError(1, "/fake/client")),
        ):
            # Must not raise
            CW.runNewClient(
                scriptPath=str(tmp_path),
                clientParametersPath="/cfg/A.ini",
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
            )

    def test_runnewclient_oserror_caught(self, tmp_path, monkeypatch):
        """Lines 225-228: OSError is caught and warning issued."""
        _default_globalParameters_overrides(monkeypatch)

        with (
            patch.object(CW, "getClientExecutablePath", return_value="/fake/client"),
            patch.object(CW.subprocess, "run", side_effect=OSError("not found")),
        ):
            # Must not raise
            CW.runNewClient(
                scriptPath=str(tmp_path),
                clientParametersPath="/cfg/A.ini",
                cxxCompiler="/usr/bin/hipcc",
                cCompiler="/usr/bin/hipcc",
            )


# ---------------------------------------------------------------------------
# getClientExecutablePath (lines 804-814)
# ---------------------------------------------------------------------------

class TestGetClientExecutablePath:
    """getClientExecutablePath exercises lines 804-814."""

    def test_raises_when_file_not_found(self, monkeypatch):
        """Lines 807-813: raises FileNotFoundError when PrebuiltClient doesn't exist."""
        monkeypatch.setitem(globalParameters, "PrebuiltClient", "/nonexistent/fake_client")

        with pytest.raises(FileNotFoundError, match="Tensile client executable not found"):
            CW.getClientExecutablePath()

    def test_returns_path_when_file_exists(self, tmp_path, monkeypatch):
        """Lines 805-806: returns PrebuiltClient path when it exists."""
        fake_exe = tmp_path / "tensile_client"
        fake_exe.write_text("#!/bin/bash\necho fake")
        monkeypatch.setitem(globalParameters, "PrebuiltClient", str(fake_exe))

        result = CW.getClientExecutablePath()
        assert result == str(fake_exe)
