# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for metrics.machine_info collector."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from dnn_benchmarking.metrics import machine_info
from dnn_benchmarking.metrics._diagnostic import reset as _reset_warns


@pytest.fixture(autouse=True)
def _reset_warn_state():
    _reset_warns()
    yield
    _reset_warns()


class TestReadCpuModel:
    def test_returns_first_model_name(self):
        cpuinfo = (
            "processor\t: 0\n"
            "vendor_id\t: AuthenticAMD\n"
            "model name\t: AMD EPYC 9654 96-Core Processor\n"
            "cache size\t: 1024 KB\n"
        )
        with patch("builtins.open", mock_open(read_data=cpuinfo)):
            assert machine_info._read_cpu_model() == "AMD EPYC 9654 96-Core Processor"

    def test_returns_none_when_file_missing(self):
        with patch("builtins.open", side_effect=OSError("no such file")):
            assert machine_info._read_cpu_model() is None


class TestCollectMachineInfo:
    def test_returns_all_keys(self):
        # Force amdsmi unavailable so static GPU fields stay None and
        # the test doesn't depend on an actual GPU.
        with patch.object(machine_info, "is_amdsmi_available", return_value=False):
            info = machine_info.collect_machine_info()
        expected_keys = {
            "cpu_model",
            "cpu_count",
            "numa_nodes",
            "total_ram_gb",
            "kernel_version",
            "gpu_compute_units",
            "gpu_hbm_gb",
            "gpu_pcie_link",
            "amdgpu_driver_version",
        }
        assert set(info.keys()) == expected_keys

    def test_never_raises(self):
        # Even if every probe blows up, the function must return a dict.
        with patch.object(machine_info, "_read_cpu_model", side_effect=OSError):
            with patch.object(machine_info, "is_amdsmi_available", return_value=False):
                info = machine_info.collect_machine_info()
        assert isinstance(info, dict)
