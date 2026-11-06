# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``tests._llvm_tools.find_llvm_tool``: ROCm's LLVM first, then ``PATH``, and
``None`` (so the ISA tests skip) on a host with neither. CPU only."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._llvm_tools import find_llvm_tool

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


if __name__ == "__main__":
    unittest.main()
