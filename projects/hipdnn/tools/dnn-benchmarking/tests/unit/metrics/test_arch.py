# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the GPU arch detection chain."""

from unittest.mock import MagicMock, patch

from dnn_benchmarking.metrics import arch as _arch
from dnn_benchmarking.metrics.arch import detect_arch


class TestDetectArchTorchPath:
    """Patches ``_arch._detect_via_torch`` directly rather than the
    ``torch.cuda.*`` symbols. The latter form raises
    ``ModuleNotFoundError`` at collection time on hosts without torch —
    a real concern for the unit suite, which is supposed to run cleanly
    on a CI box without ROCm or torch installed."""

    def test_torch_returns_gfx942(self):
        with patch.object(
            _arch, "_detect_via_torch", return_value="gfx942"
        ), patch.object(_arch, "_detect_via_rocminfo", return_value=None):
            assert detect_arch() == "gfx942"

    def test_torch_path_returns_none_falls_through_to_rocminfo(self):
        with patch.object(_arch, "_detect_via_torch", return_value=None), patch.object(
            _arch, "_detect_via_rocminfo", return_value="gfx90a"
        ):
            assert detect_arch() == "gfx90a"


class TestDetectArchRocminfoPath:
    def test_rocminfo_returns_gfx_target(self):
        sample = "  Name:  gfx942\n  Marketing Name: AMD Instinct MI300X\n"
        proc = MagicMock(returncode=0, stdout=sample, stderr="")
        # Patch resolve_rocm_tool (not bare shutil.which) so the test
        # mirrors production resolution: rocminfo is found under
        # $ROCM_PATH/bin even when /opt/rocm/bin isn't on PATH.
        with patch.object(_arch, "_detect_via_torch", return_value=None), patch.object(
            _arch, "resolve_rocm_tool", return_value="/opt/rocm/bin/rocminfo"
        ), patch("subprocess.run", return_value=proc):
            assert detect_arch() == "gfx942"

    def test_rocminfo_missing_returns_unknown(self):
        with patch.object(_arch, "_detect_via_torch", return_value=None), patch.object(
            _arch, "resolve_rocm_tool", return_value=None
        ):
            assert detect_arch() == "unknown"


class TestDetectArchUnknown:
    def test_no_torch_no_rocminfo_returns_unknown(self):
        with patch.object(_arch, "_detect_via_torch", return_value=None), patch.object(
            _arch, "_detect_via_rocminfo", return_value=None
        ):
            assert detect_arch() == "unknown"
