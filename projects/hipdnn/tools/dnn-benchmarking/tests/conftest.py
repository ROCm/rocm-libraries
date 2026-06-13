# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pytest fixtures for dnn-benchmarking tests."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from dnn_benchmarking.execution.timing import GpuTimerInterface


def pytest_addoption(parser):
    parser.addoption(
        "--profiling-strict",
        action="store_true",
        default=False,
        help=(
            "run profiling_strict tests that require profiler subprocesses "
            "to produce real artifacts, not just error/skip diagnostics"
        ),
    )
    parser.addoption(
        "--dnn-plugin-paths",
        action="store",
        default=None,
        help=(
            "Comma-separated hipDNN engine plugin directories for GPU tests. "
            "Each directory must exist and contain at least one .so file."
        ),
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--profiling-strict"):
        return

    skip_strict = pytest.mark.skip(
        reason="profiling_strict tests require --profiling-strict"
    )
    for item in items:
        if "profiling_strict" in item.keywords:
            item.add_marker(skip_strict)


def expected_timing_backend() -> str:
    """Return the GPU timing backend the executor selects on this host.

    ROCm uses direct HIP events ("hip"); CUDA uses torch.cuda events
    ("torch"). Lets GPU-generic tests assert the right backend without
    pinning to one platform.
    """
    from dnn_benchmarking.common import torch_support

    return "hip" if torch_support.is_rocm_build() else "torch"


def skip_if_no_gpu_torch() -> None:
    """Skip unless torch can drive a GPU (ROCm or CUDA), for generic tests."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if not torch.cuda.is_available():
        pytest.skip("PyTorch GPU not available")

    # The executor picks the timing backend per platform (HIP on ROCm,
    # torch.cuda on CUDA). HIP timing lazily imports hipdnn_frontend, so a
    # ROCm-torch host without those bindings would error mid-benchmark rather
    # than skip. Require the backend the executor will actually use.
    from dnn_benchmarking.execution import timing

    if expected_timing_backend() not in timing.get_available_backends():
        pytest.skip(
            "resolved GPU timing backend unavailable "
            "(e.g. hipdnn_frontend HIP bindings missing)"
        )


def skip_if_no_rocm_torch() -> None:
    """Skip unless this is a ROCm torch build with usable HIP bindings."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if not torch.cuda.is_available():
        pytest.skip("PyTorch GPU not available")

    if torch.version.hip is None:
        pytest.skip("ROCm PyTorch build required")

    try:
        import hipdnn_frontend as hipdnn

        if hipdnn.hip_get_device_count() <= 0:
            pytest.skip("No HIP GPU available")
    except Exception as e:
        pytest.skip(f"hipdnn_frontend HIP bindings not available: {e}")


def skip_if_no_cuda_torch() -> None:
    """Skip unless this is a CUDA (non-ROCm) torch build with a usable GPU."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if torch.version.cuda is None or torch.version.hip is not None:
        pytest.skip("CUDA PyTorch build required")

    if not torch.cuda.is_available():
        pytest.skip("PyTorch GPU not available")


class DummyHipTimer(GpuTimerInterface):
    """Minimal timer implementation for factory tests.

    This is a test fixture that can be used to mock GPU timing
    without requiring actual GPU hardware.
    """

    def __init__(self, stream: int = 0) -> None:
        self.stream = stream

    @property
    def backend_name(self) -> str:
        return "hip"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def synchronize(self) -> None:
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
def skip_if_no_gpu(plugin_paths):
    """Skip test if no AMD GPU available."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("PyTorch GPU not available")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")

    try:
        import hipdnn_frontend as hipdnn

        hipdnn.set_engine_plugin_paths(plugin_paths, hipdnn.PluginLoadingMode.ABSOLUTE)

        hipdnn.Handle()
    except Exception:
        pytest.skip("No GPU available or hipdnn_frontend not installed")


def _valid_plugin_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.so"))


def _parse_plugin_paths(raw_paths: str) -> List[Path]:
    paths = [Path(path.strip()) for path in raw_paths.split(",") if path.strip()]
    if not paths:
        raise pytest.UsageError("--dnn-plugin-paths requires at least one path")

    invalid_paths = [path for path in paths if not _valid_plugin_dir(path)]
    if invalid_paths:
        formatted_paths = ", ".join(str(path) for path in invalid_paths)
        raise pytest.UsageError(
            "--dnn-plugin-paths entries must be directories containing at "
            f"least one .so file: {formatted_paths}"
        )

    return paths


def _venv_rocm_sdk_plugin_dirs() -> list[Path]:
    """Return ROCm SDK wheel plugin directories from the active venv only."""
    import sys
    import sysconfig

    venv_root = Path(sys.prefix).resolve()
    dirs: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if not value:
            continue
        site_dir = Path(value).resolve()
        if site_dir != venv_root and venv_root not in site_dir.parents:
            continue
        if not site_dir.is_dir():
            continue
        dirs.extend(
            child / "lib" / "hipdnn_plugins" / "engines"
            for child in site_dir.iterdir()
            if child.is_dir() and child.name.startswith("_rocm_sdk_libraries_")
        )
    return dirs


def _find_plugin_paths(pytestconfig) -> Optional[List[str]]:
    """Find hipDNN engine plugin directories.

    Returns explicitly configured plugin paths, then falls back to known build
    and system install locations. Returns None if no plugin directory is found.
    """
    configured_paths = pytestconfig.getoption("--dnn-plugin-paths", default=None)
    if configured_paths:
        return [str(path) for path in _parse_plugin_paths(configured_paths)]

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
        *_venv_rocm_sdk_plugin_dirs(),
        # System install
        Path("/opt/rocm/lib/hipdnn_plugins/engines"),
    ]
    for path in candidates:
        if _valid_plugin_dir(path):
            return [str(path)]
    return None


@pytest.fixture
def plugin_paths(pytestconfig):
    """Get hipDNN engine plugin paths, or skip if none are found."""
    paths = _find_plugin_paths(pytestconfig)
    if paths is None:
        pytest.skip("No hipDNN engine plugin found")
    return paths


@pytest.fixture
def plugin_path(plugin_paths):
    """Get the first hipDNN engine plugin path, or skip if not found."""
    return plugin_paths[0]


@pytest.fixture
def plugin_path_cli_args(plugin_paths):
    """Return CLI args for --plugin-path using the first resolved plugin path."""
    return ["--plugin-path", plugin_paths[0]]
