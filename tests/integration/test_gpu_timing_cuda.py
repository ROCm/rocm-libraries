# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for PyTorch GPU timing on NVIDIA devices."""

from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.execution.pytorch_buffer_manager import PyTorchCudaBufferManager
from dnn_benchmarking.execution.pytorch_executor import PyTorchCudaExecutor
from dnn_benchmarking.graph.loader import GraphLoader

pytestmark = [pytest.mark.gpu, pytest.mark.nvidia]


def _skip_if_no_cuda() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if not torch.cuda.is_available():
        pytest.skip("PyTorch GPU not available")

    if torch.version.hip is not None:
        pytest.skip("ROCm build detected; skipping NVIDIA-only test")


def test_pytorch_gpu_timing_cuda() -> None:
    """Validate E2E and kernel timings on NVIDIA GPUs."""
    _skip_if_no_cuda()

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
        result = executor.benchmark(tensors, graph_name="cuda_timing")

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
    assert result.metadata.gpu_backend == "torch"
