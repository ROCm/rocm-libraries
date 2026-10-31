# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for compiling the helper kernels (Kernels.cpp) while
TensileCreateLibrary generates the assembly kernels."""
import threading
from unittest.mock import MagicMock

import pytest

import tensilelite.TensileCreateLibrary.Run as Run

pytestmark = pytest.mark.unit


def _write(tmp_path):
    return Run.writeSolutionsAndKernelsTCL(
        str(tmp_path),  # outputPath
        MagicMock(),  # asmToolchain
        MagicMock(),  # srcToolchain
        [],  # solutions
        [],  # kernels
        [],  # kernelHelperObjs
        MagicMock(),  # kernelWriterAssembly
        ["gfx942"],  # cmdlineArchs
    )


@pytest.fixture
def builders(monkeypatch):
    monkeypatch.setattr(Run, "rocisa", MagicMock())
    monkeypatch.setattr(Run, "writeHelpers", lambda *a, **kw: None)
    monkeypatch.setattr(Run, "buildAssemblyCodeObjectFiles", lambda *a, **kw: [])
    return monkeypatch


def test_helper_compile_runs_while_assembly_kernels_are_generated(tmp_path, builders):
    compileStarted = threading.Event()
    codegenDone = threading.Event()
    seen = {}

    def compileHelpers(*a, **kw):
        compileStarted.set()
        codegenDone.wait(timeout=10)
        return []

    def generate(*a, **kw):
        seen["compileStartedDuringCodegen"] = compileStarted.wait(timeout=10)
        codegenDone.set()
        return []

    builders.setattr(Run, "buildSourceCodeObjectFiles", compileHelpers)
    builders.setattr(Run, "ParallelMap2", generate)
    _write(tmp_path)

    assert seen["compileStartedDuringCodegen"]


def test_helper_compile_failure_is_raised(tmp_path, builders):
    def compileHelpers(*a, **kw):
        raise RuntimeError("Kernels.cpp failed to compile")

    builders.setattr(Run, "buildSourceCodeObjectFiles", compileHelpers)
    builders.setattr(Run, "ParallelMap2", lambda *a, **kw: [])
    with pytest.raises(RuntimeError, match="Kernels.cpp failed to compile"):
        _write(tmp_path)
