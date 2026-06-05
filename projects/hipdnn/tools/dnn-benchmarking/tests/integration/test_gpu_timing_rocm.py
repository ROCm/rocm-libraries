# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for hipDNN GPU timing on AMD ROCm devices."""

import json
from pathlib import Path
from typing import List

import pytest

from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.execution.buffer_manager import BufferManager
from dnn_benchmarking.execution.executor import Executor
from dnn_benchmarking.graph.loader import GraphLoader

pytestmark = [pytest.mark.gpu, pytest.mark.amd]


def _skip_if_no_rocm(plugin_paths: List[str]) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if not torch.cuda.is_available():
        pytest.skip("PyTorch GPU not available")

    if torch.version.hip is None:
        pytest.skip("CUDA build detected; skipping AMD-only test")

    try:
        import hipdnn_frontend as hipdnn

        hipdnn.set_engine_plugin_paths(plugin_paths, hipdnn.PluginLoadingMode.ABSOLUTE)
        hipdnn.Handle()
    except Exception as e:
        pytest.skip(f"hipdnn_frontend not available or no GPU: {e}")


def test_hipdnn_gpu_timing_rocm(plugin_paths: List[str]) -> None:
    """Validate E2E and kernel timings on AMD ROCm devices using hipDNN."""
    _skip_if_no_rocm(plugin_paths)

    graph_path = Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"
    if not graph_path.exists():
        pytest.skip(f"Sample graph not found: {graph_path}")

    loader = GraphLoader()
    graph_json = loader.load_json(graph_path)
    tensor_infos = loader.extract_tensor_info(graph_json)
    graph_json_str = json.dumps(graph_json)

    config = BenchmarkConfig(graph_path=graph_path, warmup_iters=1, benchmark_iters=3)

    import hipdnn_frontend as hipdnn

    handle = hipdnn.Handle()
    executor = Executor(graph_json_str, config, gpu_backend="torch")
    executor.prepare(handle)

    with BufferManager(tensor_infos) as buffer_manager:
        buffer_manager.allocate_all()
        buffer_manager.fill_inputs_random(seed=42)
        buffer_manager.zero_outputs()

        variant_pack = buffer_manager.create_variant_pack()
        executor.warmup(handle, variant_pack)

        result = executor.benchmark(handle, variant_pack, graph_name="rocm_timing")

    assert result.kernel_timings is not None
    assert len(result.kernel_timings) == 3
    assert len(result.e2e_timings) == 3
    assert all(t > 0.0 for t in result.kernel_timings)
    assert all(t > 0.0 for t in result.e2e_timings)

    tolerance_ms = 0.1
    for e2e_ms, kernel_ms in zip(result.e2e_timings, result.kernel_timings):
        assert e2e_ms + tolerance_ms >= kernel_ms

    assert result.metadata is not None
    assert result.metadata.gpu_backend == "torch"
