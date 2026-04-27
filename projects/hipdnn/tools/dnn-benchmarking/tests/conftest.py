# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pytest fixtures for dnn-benchmarking tests."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from dnn_benchmarking.execution.timing import GpuTimerInterface


class DummyTorchTimer(GpuTimerInterface):
    """Minimal timer implementation for factory tests.

    This is a test fixture that can be used to mock GPU timing
    without requiring actual GPU hardware.
    """

    @property
    def backend_name(self) -> str:
        return "torch"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def elapsed_ms(self) -> float:
        return 0.0


@pytest.fixture
def sample_conv_fwd_json() -> Dict[str, Any]:
    """Minimal Conv Fwd JSON for testing (matches hipDNN serialization format)."""
    return {
        "name": "sample_conv_fwd_16x16x16x16_k16_3x3",
        "compute_data_type": "float",
        "io_data_type": "float",
        "intermediate_data_type": "float",
        "tensors": [
            {
                "uid": 0,
                "name": "output_y",
                "dims": [16, 16, 16, 16],
                "strides": [4096, 256, 16, 1],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 1,
                "name": "input_x",
                "dims": [16, 16, 16, 16],
                "strides": [4096, 256, 16, 1],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 2,
                "name": "weight",
                "dims": [16, 16, 3, 3],
                "strides": [144, 9, 3, 1],
                "data_type": "float",
                "virtual": False,
            },
        ],
        "nodes": [
            {
                "name": "conv_fprop_node",
                "type": "ConvolutionFwdAttributes",
                "compute_data_type": "unset",
                "inputs": {
                    "x_tensor_uid": 1,
                    "w_tensor_uid": 2,
                },
                "outputs": {"y_tensor_uid": 0},
                "parameters": {
                    "conv_mode": "CROSS_CORRELATION",
                    "pre_padding": [1, 1],
                    "post_padding": [1, 1],
                    "stride": [1, 1],
                    "dilation": [1, 1],
                },
            }
        ],
    }


@pytest.fixture
def sample_matmul_json() -> Dict[str, Any]:
    """Matmul JSON for testing unsupported operation detection."""
    return {
        "name": "sample_matmul",
        "compute_data_type": "float",
        "io_data_type": "float",
        "intermediate_data_type": "float",
        "preferred_engine_id": 1,
        "tensors": [
            {
                "uid": 1,
                "name": "input_a",
                "dims": [16, 16],
                "strides": [16, 1],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 2,
                "name": "input_b",
                "dims": [16, 16],
                "strides": [16, 1],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 3,
                "name": "output_c",
                "dims": [16, 16],
                "strides": [16, 1],
                "data_type": "float",
                "virtual": False,
            },
        ],
        "nodes": [
            {
                "name": "matmul_node",
                "type": "MatmulAttributes",
                "compute_data_type": "float",
                "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                "outputs": {"c_tensor_uid": 3},
            }
        ],
    }


@pytest.fixture
def temp_json_file(tmp_path: Path, sample_conv_fwd_json: Dict[str, Any]) -> Path:
    """Create a temporary JSON file with sample conv fwd graph."""
    json_path = tmp_path / "test_graph.json"
    with open(json_path, "w") as f:
        json.dump(sample_conv_fwd_json, f)
    return json_path


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no AMD GPU available."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("PyTorch GPU not available")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")

    try:
        import hipdnn_frontend as hipdnn

        hipdnn.Handle()
    except Exception:
        pytest.skip("No GPU available or hipdnn_frontend not installed")


def _find_plugin_path() -> str:
    """Find the hipDNN engine plugin directory.

    Searches worktree build dir and standard install locations.
    Returns the path as a string, or None if not found.
    """
    project_root = Path(__file__).parent.parent
    candidates = [
        # Worktree/superbuild: relative to dnn-benchmarking tool
        project_root.parent.parent.parent.parent
        / "dnn-providers"
        / "miopen-provider"
        / "build"
        / "lib"
        / "hipdnn_plugins"
        / "engines",
        # System install
        Path("/opt/rocm/lib/hipdnn_plugins/engines"),
    ]
    for p in candidates:
        if p.is_dir() and any(p.glob("*.so")):
            return str(p)
    return None


@pytest.fixture
def plugin_path():
    """Get the hipDNN engine plugin path, or skip if not found."""
    path = _find_plugin_path()
    if path is None:
        pytest.skip("No hipDNN engine plugin found")
    return path


@pytest.fixture
def plugin_path_cli_args():
    """Return CLI args for --plugin-path, or empty list if not needed."""
    path = _find_plugin_path()
    if path is None:
        return []
    return ["--plugin-path", path]
