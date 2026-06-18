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
# SPDX-License-Identifier: MIT
################################################################################

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call, MagicMock

from Tensile.ClientExecutable import (
    CMakeEnvironment,
    clientExecutableEnvironment,
    getClientExecutable,
)


@pytest.mark.unit
class TestCMakeEnvironment:
    """Test suite for CMakeEnvironment class."""

    def test_init_stores_source_dir(self):
        """Constructor stores sourceDir."""
        env = CMakeEnvironment("/src", "/build")
        assert env.sourceDir == "/src"

    def test_init_stores_build_dir(self):
        """Constructor stores buildDir."""
        env = CMakeEnvironment("/src", "/build")
        assert env.buildDir == "/build"

    def test_init_stores_empty_options(self):
        """Constructor stores empty options dict when no options provided."""
        env = CMakeEnvironment("/src", "/build")
        assert env.options == {}

    def test_init_stores_options(self):
        """Constructor stores provided options."""
        options = {"CMAKE_BUILD_TYPE": "Release", "OPTION2": "value2"}
        env = CMakeEnvironment("/src", "/build", **options)
        assert env.options == options

    @patch("Tensile.ClientExecutable.print2")
    @patch("Tensile.ClientExecutable.ClientExecutionLock")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("subprocess.check_call")
    def test_generate_calls_cmake_with_no_options(
        self, mock_subprocess, mock_ensure, mock_lock, mock_print
    ):
        """generate() calls cmake with sourceDir when no options."""
        mock_ensure.return_value = "/build"
        mock_lock.return_value.__enter__ = Mock()
        mock_lock.return_value.__exit__ = Mock()

        env = CMakeEnvironment("/src", "/build")
        env.generate()

        expected_args = ["cmake", "/src"]
        mock_subprocess.assert_called_once_with(expected_args, cwd="/build")

    @patch("Tensile.ClientExecutable.print2")
    @patch("Tensile.ClientExecutable.ClientExecutionLock")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("subprocess.check_call")
    def test_generate_calls_cmake_with_options(
        self, mock_subprocess, mock_ensure, mock_lock, mock_print
    ):
        """generate() calls cmake with -D flags for each option."""
        mock_ensure.return_value = "/build"
        mock_lock.return_value.__enter__ = Mock()
        mock_lock.return_value.__exit__ = Mock()

        env = CMakeEnvironment(
            "/src", "/build", CMAKE_BUILD_TYPE="Release", TENSILE_USE_MSGPACK="ON"
        )
        env.generate()

        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == "cmake"
        assert "-D" in call_args
        assert "CMAKE_BUILD_TYPE=Release" in call_args
        assert "-D" in call_args
        assert "TENSILE_USE_MSGPACK=ON" in call_args
        assert call_args[-1] == "/src"

    @patch("Tensile.ClientExecutable.print2")
    @patch("Tensile.ClientExecutable.ClientExecutionLock")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("subprocess.check_call")
    def test_generate_ensures_build_path(
        self, mock_subprocess, mock_ensure, mock_lock, mock_print
    ):
        """generate() ensures build directory exists."""
        mock_ensure.return_value = "/build/created"
        mock_lock.return_value.__enter__ = Mock()
        mock_lock.return_value.__exit__ = Mock()

        env = CMakeEnvironment("/src", "/build")
        env.generate()

        mock_ensure.assert_called_once_with("/build")

    @patch("Tensile.ClientExecutable.print2")
    @patch("Tensile.ClientExecutable.ClientExecutionLock")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("subprocess.check_call")
    def test_generate_uses_lock(
        self, mock_subprocess, mock_ensure, mock_lock, mock_print
    ):
        """generate() uses ClientExecutionLock."""
        mock_ensure.return_value = "/build"
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance

        env = CMakeEnvironment("/src", "/build")
        env.generate()

        mock_lock_instance.__enter__.assert_called_once()
        mock_lock_instance.__exit__.assert_called_once()

    @patch("Tensile.ClientExecutable.print2")
    @patch("Tensile.ClientExecutable.ClientExecutionLock")
    @patch("subprocess.check_call")
    def test_build_calls_make(self, mock_subprocess, mock_lock, mock_print):
        """build() calls make -j in buildDir."""
        mock_lock.return_value.__enter__ = Mock()
        mock_lock.return_value.__exit__ = Mock()

        env = CMakeEnvironment("/src", "/build")
        env.build()

        expected_args = ["make", "-j"]
        mock_subprocess.assert_called_once_with(expected_args, cwd="/build")

    @patch("Tensile.ClientExecutable.print2")
    @patch("Tensile.ClientExecutable.ClientExecutionLock")
    @patch("subprocess.check_call")
    def test_build_uses_lock(self, mock_subprocess, mock_lock, mock_print):
        """build() uses ClientExecutionLock."""
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance

        env = CMakeEnvironment("/src", "/build")
        env.build()

        mock_lock_instance.__enter__.assert_called_once()
        mock_lock_instance.__exit__.assert_called_once()

    def test_builtPath_single_path(self):
        """builtPath() joins single path to buildDir."""
        env = CMakeEnvironment("/src", "/build")
        result = env.builtPath("output")
        assert result == os.path.join("/build", "output")

    def test_builtPath_multiple_paths(self):
        """builtPath() joins multiple paths to buildDir."""
        env = CMakeEnvironment("/src", "/build")
        result = env.builtPath("client", "tensile_client")
        assert result == os.path.join("/build", "client", "tensile_client")

    def test_builtPath_nested_paths(self):
        """builtPath() handles nested directory paths."""
        env = CMakeEnvironment("/src", "/build")
        result = env.builtPath("a", "b", "c", "file.txt")
        assert result == os.path.join("/build", "a", "b", "c", "file.txt")


@pytest.mark.unit
class TestClientExecutableEnvironment:
    """Test suite for clientExecutableEnvironment function."""

    @patch("Tensile.ClientExecutable.SOURCE_PATH", "/tensile/source")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("Tensile.ClientExecutable.globalParameters", {
        "CMakeBuildType": "Release",
        "LibraryFormat": "msgpack",
        "EnableMarker": "ON",
        "ROCmBinPath": "/opt/rocm/bin",
    })
    def test_creates_cmake_environment(self, mock_ensure):
        """clientExecutableEnvironment() creates CMakeEnvironment with correct paths."""
        mock_ensure.return_value = "/build/ensured"

        result = clientExecutableEnvironment("/build", "hipcc", "clang")

        assert isinstance(result, CMakeEnvironment)
        assert result.sourceDir == "/tensile/source"
        assert result.buildDir == "/build/ensured"

    @patch("Tensile.ClientExecutable.SOURCE_PATH", "/tensile/source")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("Tensile.ClientExecutable.globalParameters", {
        "CMakeBuildType": "Debug",
        "LibraryFormat": "yaml",
        "EnableMarker": "OFF",
        "ROCmBinPath": "/opt/rocm/bin",
    })
    def test_sets_cmake_build_type(self, mock_ensure):
        """clientExecutableEnvironment() sets CMAKE_BUILD_TYPE from globalParameters."""
        mock_ensure.return_value = "/build"

        result = clientExecutableEnvironment("/build", "hipcc", "clang")

        assert result.options["CMAKE_BUILD_TYPE"] == "Debug"

    @patch("Tensile.ClientExecutable.SOURCE_PATH", "/tensile/source")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("Tensile.ClientExecutable.globalParameters", {
        "CMakeBuildType": "Release",
        "LibraryFormat": "msgpack",
        "EnableMarker": "ON",
        "ROCmBinPath": "/opt/rocm/bin",
    })
    def test_sets_tensile_options(self, mock_ensure):
        """clientExecutableEnvironment() sets Tensile-specific options."""
        mock_ensure.return_value = "/build"

        result = clientExecutableEnvironment("/build", "hipcc", "clang")

        assert result.options["TENSILE_USE_MSGPACK"] == "ON"
        assert result.options["Tensile_LIBRARY_FORMAT"] == "msgpack"
        assert result.options["Tensile_ENABLE_MARKER"] == "ON"

    @patch("Tensile.ClientExecutable.SOURCE_PATH", "/tensile/source")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("Tensile.ClientExecutable.globalParameters", {
        "CMakeBuildType": "Release",
        "LibraryFormat": "msgpack",
        "EnableMarker": "ON",
        "ROCmBinPath": "/custom/rocm",
    })
    def test_sets_compilers_from_rocm_path(self, mock_ensure):
        """clientExecutableEnvironment() constructs compiler paths from ROCmBinPath."""
        mock_ensure.return_value = "/build"

        result = clientExecutableEnvironment("/build", "hipcc", "clang")

        assert result.options["CMAKE_CXX_COMPILER"] == "/custom/rocm/hipcc"
        assert result.options["CMAKE_C_COMPILER"] == "/custom/rocm/clang"

    @patch("Tensile.ClientExecutable.SOURCE_PATH", "/tensile/source")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("Tensile.ClientExecutable.globalParameters", {
        "CMakeBuildType": "Release",
        "LibraryFormat": "msgpack",
        "EnableMarker": "ON",
        "ROCmBinPath": "/opt/rocm/bin",
    })
    @patch.dict(os.environ, {}, clear=True)
    def test_no_ccache_when_env_not_set(self, mock_ensure):
        """clientExecutableEnvironment() doesn't add ccache options when CCACHE_BASEDIR not set."""
        mock_ensure.return_value = "/build"

        result = clientExecutableEnvironment("/build", "hipcc", "clang")

        assert "CMAKE_C_COMPILER_LAUNCHER" not in result.options
        assert "CMAKE_CXX_COMPILER_LAUNCHER" not in result.options

    @patch("Tensile.ClientExecutable.SOURCE_PATH", "/tensile/source")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("Tensile.ClientExecutable.globalParameters", {
        "CMakeBuildType": "Release",
        "LibraryFormat": "msgpack",
        "EnableMarker": "ON",
        "ROCmBinPath": "/opt/rocm/bin",
    })
    @patch.dict(os.environ, {"CCACHE_BASEDIR": "/tmp/ccache"})
    @patch("builtins.print")
    def test_enables_ccache_when_env_set(self, mock_print, mock_ensure):
        """clientExecutableEnvironment() adds ccache launcher when CCACHE_BASEDIR is set."""
        mock_ensure.return_value = "/build"

        result = clientExecutableEnvironment("/build", "hipcc", "clang")

        assert result.options["CMAKE_C_COMPILER_LAUNCHER"] == "ccache"
        assert result.options["CMAKE_CXX_COMPILER_LAUNCHER"] == "ccache"
        mock_print.assert_called_once_with("Is Using CCACHE")

    @patch("Tensile.ClientExecutable.SOURCE_PATH", "/tensile/source")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("Tensile.ClientExecutable.globalParameters", {
        "CMakeBuildType": "Release",
        "LibraryFormat": "msgpack",
        "EnableMarker": "ON",
        "ROCmBinPath": "/opt/rocm/bin",
    })
    def test_sets_tensile_use_llvm_on_unix(self, mock_ensure):
        """clientExecutableEnvironment() sets TENSILE_USE_LLVM=ON on non-Windows."""
        mock_ensure.return_value = "/build"

        with patch("os.name", "posix"):
            result = clientExecutableEnvironment("/build", "hipcc", "clang")

        assert result.options["TENSILE_USE_LLVM"] == "ON"

    @patch("Tensile.ClientExecutable.SOURCE_PATH", "/tensile/source")
    @patch("Tensile.ClientExecutable.ensurePath")
    @patch("Tensile.ClientExecutable.globalParameters", {
        "CMakeBuildType": "Release",
        "LibraryFormat": "msgpack",
        "EnableMarker": "ON",
        "ROCmBinPath": "/opt/rocm/bin",
    })
    def test_sets_tensile_use_llvm_off_on_windows(self, mock_ensure):
        """clientExecutableEnvironment() sets TENSILE_USE_LLVM=OFF on Windows."""
        mock_ensure.return_value = "/build"

        with patch("os.name", "nt"):
            result = clientExecutableEnvironment("/build", "hipcc", "clang")

        assert result.options["TENSILE_USE_LLVM"] == "OFF"


@pytest.mark.unit
class TestGetClientExecutable:
    """Test suite for getClientExecutable function."""

    def setup_method(self):
        """Reset buildEnv singleton before each test."""
        import Tensile.ClientExecutable
        Tensile.ClientExecutable.buildEnv = None

    @patch("Tensile.ClientExecutable.globalParameters", {"PrebuiltClient": "/prebuilt/client"})
    def test_returns_prebuilt_client_when_set(self):
        """getClientExecutable() returns PrebuiltClient path when set in globalParameters."""
        result = getClientExecutable("hipcc", "clang", Path("/build"))

        assert result == "/prebuilt/client"

    @patch("Tensile.ClientExecutable.globalParameters", {})
    @patch("Tensile.ClientExecutable.clientExecutableEnvironment")
    def test_builds_client_on_first_call(self, mock_env_factory):
        """getClientExecutable() builds client on first call when PrebuiltClient not set."""
        mock_env = Mock()
        mock_env.builtPath.return_value = "/build/client/tensile_client"
        mock_env_factory.return_value = mock_env

        result = getClientExecutable("hipcc", "clang", Path("/build"))

        mock_env.generate.assert_called_once()
        mock_env.build.assert_called_once()
        assert result == "/build/client/tensile_client"

    @patch("Tensile.ClientExecutable.globalParameters", {})
    @patch("Tensile.ClientExecutable.CLIENT_BUILD_DIR", "client_build")
    @patch("Tensile.ClientExecutable.clientExecutableEnvironment")
    def test_uses_correct_build_dir(self, mock_env_factory):
        """getClientExecutable() constructs correct build directory path."""
        mock_env = Mock()
        mock_env.builtPath.return_value = "/build/client/tensile_client"
        mock_env_factory.return_value = mock_env

        getClientExecutable("hipcc", "clang", Path("/my/build"))

        expected_builddir = Path("/my/build") / "client_build"
        mock_env_factory.assert_called_once_with(expected_builddir, "hipcc", "clang")

    @patch("Tensile.ClientExecutable.globalParameters", {})
    @patch("Tensile.ClientExecutable.clientExecutableEnvironment")
    def test_calls_builtPath_with_client_path(self, mock_env_factory):
        """getClientExecutable() calls builtPath with correct client path."""
        mock_env = Mock()
        mock_env.builtPath.return_value = "/build/output/client/tensile_client"
        mock_env_factory.return_value = mock_env

        result = getClientExecutable("hipcc", "clang", Path("/build"))

        mock_env.builtPath.assert_called_once_with("client/tensile_client")
        assert result == "/build/output/client/tensile_client"

    @patch("Tensile.ClientExecutable.globalParameters", {})
    @patch("Tensile.ClientExecutable.clientExecutableEnvironment")
    def test_singleton_pattern_reuses_buildenv(self, mock_env_factory):
        """getClientExecutable() reuses buildEnv on subsequent calls."""
        mock_env = Mock()
        mock_env.builtPath.return_value = "/build/client/tensile_client"
        mock_env_factory.return_value = mock_env

        # First call
        result1 = getClientExecutable("hipcc", "clang", Path("/build"))
        # Second call
        result2 = getClientExecutable("hipcc", "clang", Path("/build"))

        # Should only create environment once
        assert mock_env_factory.call_count == 1
        # But generate and build only called once
        assert mock_env.generate.call_count == 1
        assert mock_env.build.call_count == 1
        # Both calls should return the same result
        assert result1 == result2

    @patch("Tensile.ClientExecutable.globalParameters", {})
    @patch("Tensile.ClientExecutable.clientExecutableEnvironment")
    def test_passes_correct_compilers(self, mock_env_factory):
        """getClientExecutable() passes compiler arguments correctly."""
        mock_env = Mock()
        mock_env.builtPath.return_value = "/build/client/tensile_client"
        mock_env_factory.return_value = mock_env

        getClientExecutable("custom-cxx", "custom-cc", Path("/build"))

        # Verify compilers passed to environment factory
        call_args = mock_env_factory.call_args
        assert call_args[0][1] == "custom-cxx"
        assert call_args[0][2] == "custom-cc"
