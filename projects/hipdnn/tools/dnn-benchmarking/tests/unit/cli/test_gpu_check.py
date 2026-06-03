# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for CLI GPU availability fallback logic."""

from types import SimpleNamespace
from unittest.mock import patch

from dnn_benchmarking.cli.gpu_check import gpu_is_available


class TestGpuCheck:
    """GPU detection should not stop at a false PyTorch result."""

    def test_falls_back_to_rocm_smi_when_torch_reports_no_cuda(self) -> None:
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        fake_completed = SimpleNamespace(returncode=0)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with patch("subprocess.run", return_value=fake_completed) as run:
                assert gpu_is_available() is True

        run.assert_called_once()

    def test_torch_cuda_true_short_circuits(self) -> None:
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with patch("subprocess.run") as run:
                assert gpu_is_available() is True

        run.assert_not_called()
