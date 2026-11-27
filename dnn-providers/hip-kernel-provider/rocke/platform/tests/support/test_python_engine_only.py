# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke.tests.python_engine_only plugin, which the installed pytest entries
of a provider build without the shared C++ engine load: in a pytest run that
loads it, a lowering that names no backend runs the Python engine without a
fallback, and one that asks for the C++ engine raises instead of falling back.

Each case runs a two-test pytest session in a child process that has no shared
engine (``ROCKE_ONLINE_LIB=none``), with and without the plugin.
"""

from __future__ import annotations

import subprocess

from rocke.core import native_engine
from rocke.core.backend import BACKEND_ENV, CPP_STRICT_ENV
from rocke.tests.lanes import REQUIRE_CPP_ENGINE_ENV
from rocke.tests.run_records import fixture_session_environment
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_LOWERING_CASES = """
import os

import pytest
from rocke.core import backend
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


def _lower():
    kernel = build_universal_gemm(
        UniversalGemmSpec(
            name="python_engine_only_case",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"),
        )
    )
    return lower_kernel_to_llvm(kernel, arch="gfx950", llvm_flavor="llvm20")


def test_a_lowering_that_names_no_backend_runs_the_python_engine():
    backend.reset_cpp_fallbacks()
    assert "python_engine_only_case" in _lower()
    assert backend.resolve_backend() == "python"
    assert backend.cpp_fallbacks() == []


def test_a_cpp_lowering_raises_instead_of_falling_back():
    os.environ["ROCKE_BACKEND"] = "cpp"
    with pytest.raises(backend.BackendError, match="C\\\\+\\\\+ engine is not available"):
        _lower()
"""


def _run_lowering_cases(tmp_path, *, plugin: bool) -> subprocess.CompletedProcess:
    cases = tmp_path / "test_lowering_cases.py"
    cases.write_text(_LOWERING_CASES)
    plugin_arguments = ("-p", "rocke.tests.python_engine_only") if plugin else ()
    env = {
        key: value
        for key, value in rocke_subprocess_env(fixture_session_environment()).items()
        if key not in (BACKEND_ENV, CPP_STRICT_ENV, REQUIRE_CPP_ENGINE_ENV)
    }
    env[native_engine.SHARED_ENGINE_ENV] = native_engine.NO_SHARED_ENGINE
    return subprocess.run(
        rocke_python_command(
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            *plugin_arguments,
            f"--basetemp={tmp_path / 'basetemp'}",
            str(cases),
        ),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_the_plugin_runs_the_python_engine_and_forbids_the_fallback(tmp_path):
    run = _run_lowering_cases(tmp_path, plugin=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "2 passed" in run.stdout, run.stdout


def test_without_the_plugin_the_cpp_backend_falls_back(tmp_path):
    """The default cpp backend, finding no engine, falls back to the Python
    engine: the case that requires no fallback fails, and so does the one that
    requires a cpp lowering to raise."""
    run = _run_lowering_cases(tmp_path, plugin=False)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "2 failed" in run.stdout, run.stdout
