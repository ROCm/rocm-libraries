# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the GPU arch detection chain."""

from unittest.mock import MagicMock, patch

from dnn_benchmarking.metrics import _arch
from dnn_benchmarking.metrics._arch import detect_arch


class TestDetectArchTorchPath:
    def test_torch_returns_gfx942(self):
        fake_props = MagicMock()
        fake_props.gcnArchName = "gfx942:sramecc+:xnack-"
        with patch.object(_arch, "_detect_via_rocminfo", return_value=None):
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_properties", return_value=fake_props
            ):
                assert detect_arch() == "gfx942"

    def test_torch_present_but_no_cuda_falls_through(self):
        with patch("torch.cuda.is_available", return_value=False), patch.object(
            _arch, "_detect_via_rocminfo", return_value="gfx90a"
        ):
            assert detect_arch() == "gfx90a"


class TestDetectArchRocminfoPath:
    def test_rocminfo_returns_gfx_target(self):
        sample = "  Name:  gfx942\n  Marketing Name: AMD Instinct MI300X\n"
        proc = MagicMock(returncode=0, stdout=sample, stderr="")
        with patch.object(_arch, "_detect_via_torch", return_value=None), patch(
            "shutil.which", return_value="/opt/rocm/bin/rocminfo"
        ), patch("subprocess.run", return_value=proc):
            assert detect_arch() == "gfx942"

    def test_rocminfo_missing_returns_fallback(self):
        with patch.object(_arch, "_detect_via_torch", return_value=None), patch(
            "shutil.which", return_value=None
        ):
            assert detect_arch() == "fallback"


class TestDetectArchFallback:
    def test_no_torch_no_rocminfo_returns_fallback(self):
        with patch.object(_arch, "_detect_via_torch", return_value=None), patch.object(
            _arch, "_detect_via_rocminfo", return_value=None
        ):
            assert detect_arch() == "fallback"
