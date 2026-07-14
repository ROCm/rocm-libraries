# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for GPU detection in gemm_full_benchmark.

Covers the rocm-smi -> amd-smi migration: amd-smi is preferred (rocm-smi is
deprecated as of ROCm 7.0) and the two tools emit different formats
("GPU: 0" vs "GPU[0]"), so each is parsed by its own branch. Tests mock
subprocess so they need no GPU, no amd-smi/rocm-smi binary, and no numpy.
"""

import os
import subprocess
import sys
import types
import unittest
from unittest import mock

# gemm_full_benchmark imports gemm_utils (which pulls in numpy) at module load.
# detect_devices/resolve_devices don't need it, so stub it out to keep the test
# hermetic and dependency-free.
_stub = types.ModuleType("gemm_utils")
_stub.setup_multiple_gemm_dispatchers = lambda *a, **k: None
_stub.expand_sweep = lambda *a, **k: None
sys.modules.setdefault("gemm_utils", _stub)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gemm_full_benchmark as gfb  # noqa: E402


# Sample tool outputs -------------------------------------------------------
AMD_SMI_LIST = """\
GPU: 0
    BDF: 0000:05:00.0
    UUID: afff74a1-0000-1000-8054-e92b0a5d57c8
GPU: 1
    BDF: 0000:06:00.0
    UUID: bfff74a1-0000-1000-8054-e92b0a5d57c9
"""

ROCM_SMI_SHOWID = """\
GPU[0]		: GPU ID: 0x740c
GPU[1]		: GPU ID: 0x740c
"""


class _NoEnvMixin:
    """Clear visibility env vars so detection reaches the smi branches."""

    def setUp(self):
        self._env_patch = mock.patch.dict(
            os.environ, {}, clear=False
        )
        self._env_patch.start()
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def tearDown(self):
        self._env_patch.stop()


class DetectDevicesEnvTest(unittest.TestCase):
    """Env vars short-circuit before any smi subprocess is invoked."""

    def test_hip_visible_devices_wins(self):
        with mock.patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "2,3"}):
            with mock.patch.object(
                gfb.subprocess, "check_output"
            ) as co:
                self.assertEqual(gfb.detect_devices(), ["2", "3"])
                co.assert_not_called()

    def test_cuda_visible_devices_wins(self):
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with mock.patch.object(gfb.subprocess, "check_output") as co:
                self.assertEqual(gfb.detect_devices(), ["1"])
                co.assert_not_called()

    def test_empty_env_is_ignored(self):
        # An empty/whitespace visibility var must not yield a phantom device;
        # detection should fall through to the smi tools.
        with mock.patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": " , "}):
            with mock.patch.object(
                gfb.subprocess, "check_output", return_value=AMD_SMI_LIST
            ):
                self.assertEqual(gfb.detect_devices(), ["0", "1"])


class DetectDevicesSmiTest(_NoEnvMixin, unittest.TestCase):
    def test_amd_smi_preferred(self):
        # When amd-smi succeeds, rocm-smi must never be consulted.
        def fake(cmd, *a, **k):
            if cmd[0] == "amd-smi":
                return AMD_SMI_LIST
            raise AssertionError("rocm-smi should not be called")

        with mock.patch.object(gfb.subprocess, "check_output", side_effect=fake):
            self.assertEqual(gfb.detect_devices(), ["0", "1"])

    def test_falls_back_to_rocm_smi_when_amd_smi_missing(self):
        def fake(cmd, *a, **k):
            if cmd[0] == "amd-smi":
                raise FileNotFoundError("amd-smi not installed")
            return ROCM_SMI_SHOWID

        with mock.patch.object(gfb.subprocess, "check_output", side_effect=fake):
            self.assertEqual(gfb.detect_devices(), ["0", "1"])

    def test_falls_back_when_amd_smi_returns_no_gpus(self):
        # amd-smi present but emits nothing parseable -> use rocm-smi.
        def fake(cmd, *a, **k):
            if cmd[0] == "amd-smi":
                return "No GPUs found\n"
            return ROCM_SMI_SHOWID

        with mock.patch.object(gfb.subprocess, "check_output", side_effect=fake):
            self.assertEqual(gfb.detect_devices(), ["0", "1"])

    def test_default_when_no_tool_available(self):
        with mock.patch.object(
            gfb.subprocess,
            "check_output",
            side_effect=FileNotFoundError("no smi tool"),
        ):
            self.assertEqual(gfb.detect_devices(), ["0"])

    def test_amd_smi_nonzero_exit_falls_back(self):
        def fake(cmd, *a, **k):
            if cmd[0] == "amd-smi":
                raise subprocess.CalledProcessError(1, cmd)
            return ROCM_SMI_SHOWID

        with mock.patch.object(gfb.subprocess, "check_output", side_effect=fake):
            self.assertEqual(gfb.detect_devices(), ["0", "1"])

    def test_amd_smi_ids_sorted_numerically_and_deduped(self):
        # Numeric (not lexicographic) sort: 2 before 10; duplicates collapsed.
        out = "GPU: 10\nGPU: 2\nGPU: 2\nGPU: 1\n"

        def fake(cmd, *a, **k):
            if cmd[0] == "amd-smi":
                return out
            raise AssertionError("rocm-smi should not be called")

        with mock.patch.object(gfb.subprocess, "check_output", side_effect=fake):
            self.assertEqual(gfb.detect_devices(), ["1", "2", "10"])

    def test_rocm_smi_ids_sorted_numerically(self):
        rocm_out = "GPU[10]\t: x\nGPU[2]\t: x\nGPU[1]\t: x\n"

        def fake(cmd, *a, **k):
            if cmd[0] == "amd-smi":
                raise FileNotFoundError
            return rocm_out

        with mock.patch.object(gfb.subprocess, "check_output", side_effect=fake):
            self.assertEqual(gfb.detect_devices(), ["1", "2", "10"])


class ResolveDevicesTest(_NoEnvMixin, unittest.TestCase):
    """resolve_devices builds on detect_devices; verify the migration
    doesn't regress the count/list semantics."""

    def _patch_detect(self, ids):
        return mock.patch.object(gfb, "detect_devices", return_value=list(ids))

    def test_none_returns_all_detected(self):
        with self._patch_detect(["0", "1", "2"]):
            self.assertEqual(gfb.resolve_devices(None), ["0", "1", "2"])

    def test_count_takes_first_n(self):
        with self._patch_detect(["0", "1", "2", "3"]):
            self.assertEqual(gfb.resolve_devices(2), ["0", "1"])

    def test_comma_list_is_explicit_ids(self):
        with self._patch_detect(["0", "1"]):
            self.assertEqual(gfb.resolve_devices("5,7"), ["5", "7"])

    def test_trailing_comma_single_id(self):
        with self._patch_detect(["0", "1", "2"]):
            self.assertEqual(gfb.resolve_devices("5,"), ["5"])


if __name__ == "__main__":
    unittest.main()
