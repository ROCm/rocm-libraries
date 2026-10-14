# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Environment allowlist and refusals, TRITONGEN-ARCH, and objdump resolution. Needs
no triton."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tritongen import _family as F  # noqa: E402
from tritongen import _triton_env as env  # noqa: E402
from tritongen._codeobject import objdump_path, target_arch, verify_arch  # noqa: E402
from tritongen._family import GeneratorError  # noqa: E402


def prepare_under(environ: dict) -> dict:
    """Run ``prepare()`` against a replacement environment; returns that environment."""
    with mock.patch.dict(os.environ, environ, clear=True), mock.patch.object(
        env, "_prepared", False
    ), mock.patch.object(env, "_cache_dirs", []):
        env.prepare()
        return dict(os.environ)


class EnvAllowlistTest(unittest.TestCase):
    def test_refused_variables_raise_env_error(self):
        for name in ("TRITON_INTERPRET", "TRITON_KERNEL_OVERRIDE"):
            with self.subTest(name=name):
                with self.assertRaises(GeneratorError) as raised:
                    prepare_under({name: "1", "HOME": "/x"})
                self.assertTrue(str(raised.exception).startswith(F.ERR_ENV))
                self.assertIn(name, str(raised.exception))

    def test_only_allowlisted_variables_survive(self):
        after = prepare_under(
            {
                "DISABLE_LLVM_OPT": "1",
                "TRITON_F32_DEFAULT": "ieee",
                "LLVM_PASS_PLUGIN_PATH": "/nonexistent",
                "TRITON_HIP_USE_BLOCK_PINGPONG": "1",
                "TRITON_DISABLE_LINE_INFO": "0",
                "HOME": "/x",
                "PATH": "/bin",
                "ROCM_PATH": "/opt/rocm",
            }
        )
        self.assertEqual(
            after,
            {
                "HOME": "/x",
                "PATH": "/bin",
                "ROCM_PATH": "/opt/rocm",
                "TRITON_DISABLE_LINE_INFO": "1",
            },
        )

    def test_manifest_record_carries_no_host_values(self):
        host = {
            "HOME": "/home/someone",
            "PATH": "/opt/odd/bin",
            "ROCM_PATH": "/opt/rocm-x",
        }
        with mock.patch.dict(os.environ, host, clear=True):
            record = json.dumps(env.env_record())
        for value in host.values():
            self.assertNotIn(value, record)
        for name in host:
            self.assertIn(name, record)


class ArchCheckTest(unittest.TestCase):
    def test_target_arch_ignores_feature_suffixes(self):
        self.assertEqual(
            target_arch(["amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-"], "t"), "gfx950"
        )

    def test_object_filed_under_wrong_arch_rejected(self):
        described = {"targets": ["amdgcn-amd-amdhsa--gfx942"]}
        verify_arch(described, "gfx942", "t")
        with self.assertRaises(GeneratorError) as raised:
            verify_arch(described, "gfx950", "t")
        self.assertTrue(str(raised.exception).startswith(F.ERR_ARCH))


class ObjdumpResolutionTest(unittest.TestCase):
    def fake_tool(self, root: Path) -> Path:
        tool = root / "llvm" / "bin" / "llvm-objdump"
        tool.parent.mkdir(parents=True)
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
        return tool

    def test_rocm_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = self.fake_tool(Path(tmp))
            with mock.patch.dict(os.environ, {"ROCM_PATH": tmp}):
                self.assertEqual(objdump_path(), str(tool))

    def test_path_is_the_last_resort_and_absence_is_env_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"ROCM_PATH": tmp, "PATH": tmp}):
                with mock.patch.object(Path, "is_file", return_value=False):
                    with self.assertRaises(GeneratorError) as raised:
                        objdump_path()
                    self.assertTrue(str(raised.exception).startswith(F.ERR_ENV))
                    tool = tmp + "/llvm-objdump"
                    Path(tool).write_text("#!/bin/sh\n")
                    os.chmod(tool, 0o755)
                    self.assertEqual(objdump_path(), tool)


if __name__ == "__main__":
    unittest.main()
