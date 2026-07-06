# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for the PyTorch executor benchmark mode.

These exercise the PyTorch execution backend and run on any live GPU
(ROCm or CUDA): the timing backend selected by the executor differs per
platform (HIP events on ROCm, torch.cuda events on CUDA), so assertions
use ``expected_timing_backend()`` rather than hardcoding one.
"""

import json
from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.execution.buffer_manager import generate_input_data
from dnn_benchmarking.execution.pytorch_buffer_manager import PyTorchCudaBufferManager
from dnn_benchmarking.execution.pytorch_executor import (
    PyTorchCudaExecutor,
    PyTorchExecutionError,
)
from dnn_benchmarking.graph.loader import GraphLoader
from tests.conftest import expected_timing_backend, skip_if_no_gpu_torch

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def sample_conv_graph():
    """Load sample conv graph for testing."""
    graph_path = Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"
    if not graph_path.exists():
        pytest.skip(f"Sample graph not found: {graph_path}")
    loader = GraphLoader()
    return loader.load_json(graph_path), graph_path


@pytest.fixture
def sample_matmul_graph():
    """Load sample matmul graph for testing."""
    graph_path = Path(__file__).parent.parent.parent / "graphs" / "sample_matmul.json"
    if not graph_path.exists():
        pytest.skip(f"Sample graph not found: {graph_path}")
    loader = GraphLoader()
    return loader.load_json(graph_path), graph_path


@pytest.fixture
def sample_relu_graph():
    """Load sample relu graph for testing."""
    graph_path = Path(__file__).parent.parent.parent / "graphs" / "sample_relu.json"
    if not graph_path.exists():
        pytest.skip(f"Sample graph not found: {graph_path}")
    loader = GraphLoader()
    return loader.load_json(graph_path), graph_path


class TestPyTorchCudaBufferManager:
    """Tests for PyTorch GPU buffer management."""

    def test_allocate_and_fill(self, sample_conv_graph):
        """Test tensor allocation and random fill."""
        skip_if_no_gpu_torch()

        graph_json, graph_path = sample_conv_graph
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(graph_json)

        with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            tensors = buffer_manager.get_tensors()
            assert len(tensors) > 0

            for uid, tensor in tensors.items():
                assert tensor.is_cuda

    def test_reproducible_with_seed(self, sample_conv_graph):
        """Test that same seed produces same random data."""
        skip_if_no_gpu_torch()

        graph_json, _ = sample_conv_graph
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(graph_json)

        # First run
        with PyTorchCudaBufferManager(tensor_infos) as bm1:
            bm1.allocate_all()
            bm1.fill_inputs_random(seed=42)
            data1 = {uid: bm1.get_input_data(uid) for uid in bm1.get_tensors().keys()}

        # Second run with same seed
        with PyTorchCudaBufferManager(tensor_infos) as bm2:
            bm2.allocate_all()
            bm2.fill_inputs_random(seed=42)
            data2 = {uid: bm2.get_input_data(uid) for uid in bm2.get_tensors().keys()}

        # Compare
        import numpy as np

        for uid in data1:
            if data1[uid] is not None and data2[uid] is not None:
                assert np.allclose(data1[uid], data2[uid])

    def test_load_input_data_uses_shared_input_map(self, sample_conv_graph):
        """Pre-generated inputs can be loaded without regenerating per run."""
        skip_if_no_gpu_torch()

        graph_json, _ = sample_conv_graph
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(graph_json)
        input_data = generate_input_data(tensor_infos, seed=123)

        with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.load_input_data(input_data)

            for uid, expected in input_data.items():
                actual = buffer_manager.get_input_data(uid)
                if actual is not None:
                    assert actual.shape == expected.shape


class TestPyTorchCudaExecutor:
    """Tests for the PyTorch GPU executor."""

    def test_prepare_validates_operations(self, sample_conv_graph):
        """Test that prepare validates graph operations."""
        skip_if_no_gpu_torch()

        graph_json, graph_path = sample_conv_graph
        config = BenchmarkConfig(
            graph_path=graph_path, warmup_iters=1, benchmark_iters=1
        )

        executor = PyTorchCudaExecutor(graph_json, config)
        executor.prepare()  # Should not raise

        assert executor.init_time_ms > 0

    def test_full_benchmark_conv(self, sample_conv_graph):
        """Test full benchmark workflow with convolution graph."""
        skip_if_no_gpu_torch()

        graph_json, graph_path = sample_conv_graph
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=2,
            benchmark_iters=5,
        )

        executor = PyTorchCudaExecutor(graph_json, config)
        executor.prepare()

        with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            tensors = buffer_manager.get_tensors()

            # Run warmup
            executor.warmup(tensors)

            # Run benchmark
            result = executor.benchmark(tensors, graph_name="test_conv")

            # Verify result
            assert len(result.e2e_timings) == 5
            assert result.kernel_timings is not None
            assert len(result.kernel_timings) == 5
            assert result.metadata is not None
            assert result.metadata.execution_backend == "pytorch"
            assert result.metadata.timing_backend == expected_timing_backend()
            assert result.metadata.graph_name == "test_conv"

    def test_full_benchmark_matmul(self, sample_matmul_graph):
        """Test full benchmark workflow with matmul graph."""
        skip_if_no_gpu_torch()

        graph_json, graph_path = sample_matmul_graph
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=2,
            benchmark_iters=5,
        )

        executor = PyTorchCudaExecutor(graph_json, config)
        executor.prepare()

        with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            tensors = buffer_manager.get_tensors()
            executor.warmup(tensors)
            result = executor.benchmark(tensors, graph_name="test_matmul")

            assert len(result.e2e_timings) == 5
            assert result.kernel_timings is not None

    def test_full_benchmark_relu(self, sample_relu_graph):
        """Test full benchmark workflow with relu graph."""
        skip_if_no_gpu_torch()

        graph_json, graph_path = sample_relu_graph
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=2,
            benchmark_iters=5,
        )

        executor = PyTorchCudaExecutor(graph_json, config)
        executor.prepare()

        with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            tensors = buffer_manager.get_tensors()
            executor.warmup(tensors)
            result = executor.benchmark(tensors, graph_name="test_relu")

            assert len(result.e2e_timings) == 5

    @pytest.mark.parametrize(
        "graph_name",
        [
            "sample_conv_dgrad.json",
            "sample_conv_wgrad.json",
            "sample_matmul_batched.json",
            "sample_matmul_broadcast.json",
            "sample_batchnorm_training.json",
            "sample_batchnorm_inference.json",
            "sample_batchnorm_inference_variance.json",
            "sample_batchnorm_backward.json",
            "sample_sdpa.json",
            "sample_mha_sdpa.json",
            "sample_sdpa_backward.json",
            "sample_layernorm.json",
            "sample_rmsnorm.json",
            "sample_rmsnorm_backward.json",
            "sample_reduction.json",
            "sample_resample_fwd.json",
        ],
    )
    def test_full_benchmark_new_reference_graphs(self, graph_name):
        """Test PyTorch benchmark workflow for newly covered reference graphs."""
        skip_if_no_gpu_torch()

        graph_path = Path(__file__).parent.parent.parent / "graphs" / graph_name
        loader = GraphLoader()
        graph_json = loader.load_json(graph_path)
        tensor_infos = loader.extract_tensor_info(graph_json)
        config = BenchmarkConfig(
            graph_path=graph_path, warmup_iters=1, benchmark_iters=1
        )

        executor = PyTorchCudaExecutor(graph_json, config)
        executor.prepare()

        with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()
            tensors = buffer_manager.get_tensors()

            executor.warmup(tensors)
            result = executor.benchmark(tensors, graph_name=graph_name)

        assert len(result.e2e_timings) == 1

    def test_json_export(self, sample_conv_graph, tmp_path):
        """Test that benchmark results can be exported to JSON."""
        skip_if_no_gpu_torch()

        graph_json, graph_path = sample_conv_graph
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=1,
            benchmark_iters=3,
        )

        executor = PyTorchCudaExecutor(graph_json, config)
        executor.prepare()

        with PyTorchCudaBufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            tensors = buffer_manager.get_tensors()
            executor.warmup(tensors)
            result = executor.benchmark(tensors, graph_name="test_export")

            # Export to JSON
            output_path = tmp_path / "results.json"
            result.save_json(str(output_path))

            # Verify JSON content
            with open(output_path) as f:
                data = json.load(f)

            assert "e2e_timings" in data
            assert "kernel_timings" in data
            assert "metadata" in data
            assert data["metadata"]["execution_backend"] == "pytorch"
            assert data["metadata"]["timing_backend"] == expected_timing_backend()

    def test_not_prepared_raises(self, sample_conv_graph):
        """Test that running without prepare raises error."""
        skip_if_no_gpu_torch()

        graph_json, graph_path = sample_conv_graph
        config = BenchmarkConfig(
            graph_path=graph_path, warmup_iters=1, benchmark_iters=1
        )

        executor = PyTorchCudaExecutor(graph_json, config)

        # Should raise without prepare
        with pytest.raises(PyTorchExecutionError, match="not prepared"):
            executor.warmup({})
