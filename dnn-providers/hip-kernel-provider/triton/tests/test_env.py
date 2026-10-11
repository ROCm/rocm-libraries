# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Environment scrub/refuse, TRITONGEN-ARCH, and objdump resolution. Needs no triton."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators import _contract as C  # noqa: E402
from generators import _triton_env as env  # noqa: E402
from generators._codeobject import (  # noqa: E402
    GeneratorError,
    objdump_path,
    target_arch,
    verify_arch,
)


def prepare_under(environ: dict) -> dict:
    """Run ``prepare()`` against a replacement environment; returns that environment."""
    with mock.patch.dict(os.environ, environ, clear=True), mock.patch.object(
        env, "_prepared", False
    ), mock.patch.object(env, "_cache_dirs", []):
        env.prepare()
        return dict(os.environ)


class EnvScrubTest(unittest.TestCase):
    def test_refused_knobs_raise_env_error(self):
        for name in env.REFUSED_KNOBS:
            with self.assertRaises(GeneratorError) as raised:
                prepare_under({name: "1"})
            self.assertTrue(str(raised.exception).startswith(C.ERR_ENV))

    def test_overriding_knobs_are_cleared_and_others_kept(self):
        after = prepare_under(
            {
                "TRITON_OVERRIDE_ARCH": "gfx900",
                "TRITON_ALWAYS_COMPILE": "1",
                "TRITON_HIP_USE_BLOCK_PINGPONG": "1",
                "AMDGCN_USE_BUFFER_OPS": "1",
                "TRITON_ENABLE_FP_FUSION": "0",
                "TRITON_F32_DEFAULT": "ieee",
                "HOME": "/x",
            }
        )
        for name in (
            "TRITON_OVERRIDE_ARCH",
            "TRITON_ALWAYS_COMPILE",
            "TRITON_HIP_USE_BLOCK_PINGPONG",
            "AMDGCN_USE_BUFFER_OPS",
            "TRITON_ENABLE_FP_FUSION",
        ):
            self.assertNotIn(name, after)
        self.assertEqual(after["TRITON_F32_DEFAULT"], "ieee")
        self.assertEqual(after["HOME"], "/x")
        self.assertEqual(after["TRITON_DISABLE_LINE_INFO"], "1")

    def test_cleared_record_covers_the_predicates(self):
        for entry in env.CLEARED_RECORD:
            sample = entry.replace("*", "X")
            self.assertTrue(env._is_cleared(sample), entry)


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
        self.assertTrue(str(raised.exception).startswith(C.ERR_ARCH))


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
                    self.assertTrue(str(raised.exception).startswith(C.ERR_ENV))
                    tool = tmp + "/llvm-objdump"
                    Path(tool).write_text("#!/bin/sh\n")
                    os.chmod(tool, 0o755)
                    self.assertEqual(objdump_path(), tool)


if __name__ == "__main__":
    unittest.main()
