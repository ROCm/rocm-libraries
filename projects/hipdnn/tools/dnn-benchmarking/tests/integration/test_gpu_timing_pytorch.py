# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for PyTorch executor timing with direct HIP events.

ROCm-specific: asserts the executor selects the direct HIP event timing
backend. The CUDA counterpart lives in test_gpu_timing_cuda.py.
"""

from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.execution.pytorch_buffer_manager import PyTorchCudaBufferManager
from dnn_benchmarking.execution.pytorch_executor import PyTorchCudaExecutor
from dnn_benchmarking.graph.loader import GraphLoader
from tests.conftest import skip_if_no_rocm_torch

pytestmark = [pytest.mark.gpu, pytest.mark.rocm]


def test_pytorch_gpu_timing_rocm() -> None:
    """Validate PyTorch executor E2E and kernel timings with HIP events."""
    skip_if_no_rocm_torch()

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
        result = executor.benchmark(tensors, graph_name="pytorch_hip_timing")

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
    assert result.metadata.timing_backend == "hip"
