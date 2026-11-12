# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-GPU tests for ``rocke.tests.llc.find_llvm_tool``: ROCm's LLVM
(``$ROCM_PATH``, ``/opt/rocm`` when unset) first, then ``PATH``, and ``None``
(so the tests that need the tool skip) on a host with neither."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rocke.tests import llc
from rocke.tests.llc import find_llvm_tool

_TOOL = "llvm-objdump"


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


if __name__ == "__main__":
    unittest.main()
