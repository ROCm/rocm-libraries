# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for Tensile.ParallelExecution.detectAvailableGpus.

These verify the amd-smi based GPU detection logic (and its hipInfo
fallback) without requiring real GPUs, by mocking subprocess.run.
"""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Importing Tensile.ParallelExecution pulls in Tensile.Common (and thus the
# rocisa bindings). If that chain is unavailable in the current environment,
# skip rather than error at collection time.
try:
    from Tensile.ParallelExecution import detectAvailableGpus
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    detectAvailableGpus = None
    _IMPORT_ERROR = exc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        detectAvailableGpus is None,
        reason=f"Tensile.ParallelExecution import unavailable: {_IMPORT_ERROR}",
    ),
]


def _completed(stdout="", returncode=0):
    """Build a stand-in for subprocess.CompletedProcess."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def _amdsmi_list_json(num_gpus):
    """Mimic the JSON array emitted by `amd-smi list --json`."""
    return json.dumps(
        [
            {
                "gpu": i,
                "bdf": f"0000:{i:02d}:00.0",
                "uuid": f"uuid-{i}",
                "kfd_id": 1000 + i,
                "node_id": i,
                "partition_id": 0,
            }
            for i in range(num_gpus)
        ]
    )


def test_counts_gpus_from_amdsmi_list():
    with patch.object(subprocess, "run", return_value=_completed(_amdsmi_list_json(8))) as m:
        assert detectAvailableGpus() == 8
    # First (and only needed) call should target amd-smi list --json
    called_cmd = m.call_args_list[0].args[0]
    assert called_cmd == ["amd-smi", "list", "--json"]


def test_single_gpu():
    with patch.object(subprocess, "run", return_value=_completed(_amdsmi_list_json(1))):
        assert detectAvailableGpus() == 1


def test_falls_back_to_hipinfo_when_amdsmi_fails():
    def side_effect(cmd, *args, **kwargs):
        if cmd[0] == "amd-smi":
            # Simulate amd-smi not present / erroring out
            raise FileNotFoundError("amd-smi not found")
        if cmd[0] == "hipInfo":
            return _completed("Number of devices:           4\n")
        raise AssertionError(f"unexpected command {cmd}")

    with patch.object(subprocess, "run", side_effect=side_effect):
        assert detectAvailableGpus() == 4


def test_falls_back_to_hipinfo_when_amdsmi_returns_nonzero():
    def side_effect(cmd, *args, **kwargs):
        if cmd[0] == "amd-smi":
            return _completed("", returncode=1)
        if cmd[0] == "hipInfo":
            return _completed("Number of devices: 2\n")
        raise AssertionError(f"unexpected command {cmd}")

    with patch.object(subprocess, "run", side_effect=side_effect):
        assert detectAvailableGpus() == 2


def test_defaults_to_one_when_everything_fails():
    with patch.object(subprocess, "run", side_effect=FileNotFoundError("nothing here")):
        assert detectAvailableGpus() == 1


def test_defaults_to_one_on_malformed_amdsmi_json():
    # amd-smi exits 0 but emits junk; should not raise, should fall through to default
    def side_effect(cmd, *args, **kwargs):
        if cmd[0] == "amd-smi":
            return _completed("not-json", returncode=0)
        if cmd[0] == "hipInfo":
            raise FileNotFoundError("no hipInfo")
        raise AssertionError(f"unexpected command {cmd}")

    with patch.object(subprocess, "run", side_effect=side_effect):
        assert detectAvailableGpus() == 1
