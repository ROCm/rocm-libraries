# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for PyTorch executor timing on NVIDIA CUDA devices.

CUDA-specific counterpart of test_gpu_timing_pytorch.py: asserts the
executor selects torch.cuda event timing (not HIP), that the
ROCm-specific environment metadata carries its non-ROCm sentinels, and
that the suite header shows CUDA/cuDNN labels (not ROCm) on a CUDA host.
"""

import io
import re
from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.execution.pytorch_buffer_manager import PyTorchCudaBufferManager
from dnn_benchmarking.execution.pytorch_executor import PyTorchCudaExecutor
from dnn_benchmarking.graph.loader import GraphLoader
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.suite_results import collect_environment_info
from tests.conftest import skip_if_no_cuda_torch

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def test_pytorch_gpu_timing_cuda() -> None:
    """Validate PyTorch executor E2E and kernel timings with torch.cuda events."""
    skip_if_no_cuda_torch()

    graph_path = Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"
    if not graph_path.exists():
        pytest.skip(f"Sample graph not found: {graph_path}")

    loader = GraphLoader()
    graph_json = loader.load_json(graph_path)
    tensor_infos = loader.extract_tensor_info(graph_json)

    config = BenchmarkConfig(graph_path=graph_path, warmup_iters=1, benchmark_iters=3)
    executor = PyTorchCudaExecutor(graph_json, config)
    executor.prepare()

    with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
        buffer_manager.allocate_all()
        buffer_manager.fill_inputs_random(seed=42)
        buffer_manager.zero_outputs()

        tensors = buffer_manager.get_tensors()
        executor.warmup(tensors)
        result = executor.benchmark(tensors, graph_name="pytorch_cuda_timing")

    assert result.kernel_timings is not None
    assert len(result.kernel_timings) == 3
    assert len(result.e2e_timings) == 3
    assert all(t > 0.0 for t in result.kernel_timings)
    assert all(t > 0.0 for t in result.e2e_timings)

    tolerance_ms = 0.1
    for e2e_ms, kernel_ms in zip(result.e2e_timings, result.kernel_timings):
        assert e2e_ms + tolerance_ms >= kernel_ms

    assert result.metadata is not None
    assert result.metadata.execution_backend == "pytorch"
    # CUDA uses torch.cuda events, never direct HIP events.
    assert result.metadata.timing_backend == "torch"


def test_cuda_environment_metadata_sentinels() -> None:
    """ROCm-specific metadata carries its non-ROCm sentinels on a CUDA host."""
    skip_if_no_cuda_torch()

    info = collect_environment_info()

    # No ROCm runtime present on a CUDA host.
    assert info["rocm_version"] is None
    # detect_arch() yields the "unknown" sentinel when no gfx target is found.
    assert info["gpu_arch"] == "unknown"

    # The CUDA-side version probes are populated instead. cuda_version is a
    # plain string from torch; cudnn_version is decoded to major.minor.patch.
    assert isinstance(info["cuda_version"], str) and info["cuda_version"]
    assert info["cudnn_version"] is None or re.fullmatch(
        r"\d+\.\d+\.\d+", info["cudnn_version"]
    )


def test_cuda_suite_header_shows_cuda_label_not_rocm() -> None:
    """On a CUDA host the suite header prints CUDA (and cuDNN), never ROCm."""
    skip_if_no_cuda_torch()

    output = io.StringIO()
    Reporter(output=output).print_suite_header(1)
    out = output.getvalue()

    assert "ROCm:" not in out
    assert re.search(r"^CUDA:\s+\S+", out, re.MULTILINE)
