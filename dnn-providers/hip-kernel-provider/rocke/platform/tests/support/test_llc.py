# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-GPU tests for the llc test helpers.

``find_llvm_tool`` looks in ROCm's LLVM (``$ROCM_PATH``, ``/opt/rocm`` when
unset) first, then ``PATH``, and gives ``None`` (so the tests that need the tool
skip) on a host with neither. The compiler-backed checks skip when llc cannot
target the GPU, and fail on any other compiler error."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rocke.tests import llc
from rocke.tests.llc import checked_llc_output, find_llvm_tool, lists_amdgpu_backend

_TOOL = "llvm-objdump"
_LLC = ["llc", "-mtriple=amdgcn-amd-amdhsa"]


def _fake_tool(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _TOOL
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@unittest.skipIf(os.name == "nt", "fake tools are POSIX scripts")
class TestFindLlvmTool(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _find(self, rocm: Path, path: Path) -> str | None:
        env = {"ROCM_PATH": str(rocm), "PATH": str(path)}
        with mock.patch.dict(os.environ, env):
            return find_llvm_tool(_TOOL)

    def test_rocm_before_path(self):
        rocm_tool = _fake_tool(self.root / "rocm" / "llvm" / "bin")
        _fake_tool(self.root / "bin")
        self.assertEqual(self._find(self.root / "rocm", self.root / "bin"), rocm_tool)

    def test_path_when_rocm_lacks_it(self):
        path_tool = _fake_tool(self.root / "bin")
        self.assertEqual(self._find(self.root / "rocm", self.root / "bin"), path_tool)

    def test_absent_is_none(self):
        (self.root / "bin").mkdir()
        self.assertIsNone(self._find(self.root / "rocm", self.root / "bin"))


class TestFindLlvmToolDefaultRoot(unittest.TestCase):
    def test_opt_rocm_when_rocm_path_is_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "ROCM_PATH"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
            llc.shutil, "which", return_value=None
        ) as which:
            self.assertIsNone(find_llvm_tool(_TOOL))
        self.assertEqual(
            [c.args[0] for c in which.call_args_list],
            [str(Path("/opt/rocm") / "llvm" / "bin" / _TOOL), _TOOL],
        )


class TestLlcHelpers(unittest.TestCase):
    def test_amdgpu_backend_is_read_from_the_version_text(self):
        with_amdgpu = (
            "LLVM version 20.0.0git\n  Registered Targets:\n"
            "    amdgcn - AMD GCN GPUs\n    x86-64 - 64-bit X86: EM64T and AMD64\n"
        )
        without_amdgpu = (
            "LLVM version 20.0.0git\n  Registered Targets:\n"
            "    x86-64 - 64-bit X86: EM64T and AMD64\n"
        )
        self.assertTrue(lists_amdgpu_backend(with_amdgpu))
        self.assertFalse(lists_amdgpu_backend(without_amdgpu))

    def test_unknown_processor_is_not_an_error_even_on_a_failed_run(self):
        result = subprocess.CompletedProcess(
            _LLC,
            returncode=1,
            stdout=b"",
            stderr=b"'gfx1154' is not a recognized processor for this target "
            b"(ignoring processor)\n",
        )
        self.assertIsNone(checked_llc_output(result))

    def test_other_compiler_errors_fail(self):
        result = subprocess.CompletedProcess(
            _LLC, returncode=1, stdout=b"", stderr=b"error: invalid register name\n"
        )
        with self.assertRaisesRegex(
            AssertionError,
            "llc failed with exit status 1: error: invalid register name",
        ):
            checked_llc_output(result)

    def test_successful_run_returns_the_output(self):
        result = subprocess.CompletedProcess(
            _LLC, returncode=0, stdout=b"asm", stderr=b""
        )
        self.assertEqual(checked_llc_output(result), b"asm")


if __name__ == "__main__":
    unittest.main()
