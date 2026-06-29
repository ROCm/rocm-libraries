# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for graph execution (requires GPU)."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from dnn_benchmarking.config import BenchmarkConfig
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    SuiteConfig,
    ValidationConfig,
)
from dnn_benchmarking.execution import BufferManager, Executor
from dnn_benchmarking.execution.suite_runner import run_graph_all_providers
from dnn_benchmarking.graph import GraphLoader
from dnn_benchmarking.validation import ArrayComparator, ReferenceProviderRegistry


def _setup_hipdnn(plugin_paths: List[str]):
    """Import hipdnn_frontend with plugin path configured, or skip."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("PyTorch GPU not available")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")

    try:
        import hipdnn_frontend

        hipdnn_frontend.set_engine_plugin_paths(
            plugin_paths, hipdnn_frontend.PluginLoadingMode.ABSOLUTE
        )

        hipdnn_frontend.Handle()
        return hipdnn_frontend
    except Exception as e:
        pytest.skip(f"hipdnn_frontend not available or no GPU: {e}")


@pytest.mark.gpu
class TestExecution:
    """Integration tests for graph execution requiring GPU."""

    @pytest.fixture
    def hipdnn(self, plugin_paths: List[str]):
        """Get hipdnn_frontend module or skip if not available."""
        return _setup_hipdnn(plugin_paths)

    def test_executor_prepare(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test that executor can prepare a graph."""
        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            warmup_iters=1,
            benchmark_iters=1,
            engine_id=1,
        )

        graph_json_str = json.dumps(sample_conv_fwd_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        assert executor.init_time_ms > 0
        assert executor.graph is not None

    def test_buffer_manager_allocate(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test buffer allocation."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            variant_pack = buffer_manager.create_variant_pack()

            # Should have 3 tensors
            assert len(variant_pack) == 3

            # All pointers should be non-zero
            for uid, ptr in variant_pack.items():
                assert ptr != 0, f"Buffer for UID {uid} has null pointer"

    def test_buffer_manager_fill_random(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test filling buffers with random data."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)

            # Check that input data was stored
            input_data = buffer_manager.get_input_data(1)
            assert input_data is not None
            assert input_data.shape == (16, 16, 16, 16)

    def test_full_execution_workflow(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test complete execution workflow: prepare, warmup, benchmark."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            warmup_iters=2,
            benchmark_iters=5,
            engine_id=1,
        )

        graph_json_str = json.dumps(sample_conv_fwd_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()

            # Warmup
            executor.warmup(handle, variant_pack)

            # Benchmark
            result = executor.benchmark(handle, variant_pack)

            # Should have 5 E2E timing values
            assert len(result.e2e_timings) == 5

            # All E2E timings should be positive
            for t in result.e2e_timings:
                assert t > 0

            # Should also have kernel timings (if HIP backend available)
            if result.kernel_timings is not None:
                assert len(result.kernel_timings) == 5
                for t in result.kernel_timings:
                    assert t > 0

            # Get output data (uid=0 for output tensor)
            output_data = buffer_manager.get_output_data(0)
            assert output_data is not None
            assert output_data.shape == (16, 16, 16, 16)

            # Output should not be all zeros after execution
            import numpy as np

            assert not np.allclose(output_data, 0)

    @pytest.mark.xfail(reason="MIOpen plugin doesn't support matmul operations yet")
    def test_matmul_execution_workflow(self, hipdnn) -> None:
        """Test execution workflow for matmul graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_matmul.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=sample_path,
            warmup_iters=2,
            benchmark_iters=5,
            engine_id=1,
        )

        graph_json_str = json.dumps(graph_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()
            executor.warmup(handle, variant_pack)
            result = executor.benchmark(handle, variant_pack)

            assert len(result.e2e_timings) == 5

            # Get output (C matrix: [256, 1024])
            output_data = buffer_manager.get_output_data(3)
            assert output_data is not None
            assert output_data.shape == (256, 1024)

            import numpy as np

            assert not np.allclose(output_data, 0)

    @pytest.mark.xfail(reason="MIOpen plugin doesn't support pointwise operations yet")
    def test_relu_execution_workflow(self, hipdnn) -> None:
        """Test execution workflow for ReLU activation graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_relu.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=sample_path,
            warmup_iters=2,
            benchmark_iters=5,
            engine_id=1,
        )

        graph_json_str = json.dumps(graph_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()
            executor.warmup(handle, variant_pack)
            result = executor.benchmark(handle, variant_pack)

            assert len(result.e2e_timings) == 5

            # Get output (same shape as input: [64, 128, 56, 56])
            output_data = buffer_manager.get_output_data(2)
            assert output_data is not None
            assert output_data.shape == (64, 128, 56, 56)

            import numpy as np

            # ReLU: output should be non-negative
            assert np.all(output_data >= 0)

    @pytest.mark.xfail(reason="MIOpen plugin doesn't support pointwise operations yet")
    def test_add_execution_workflow(self, hipdnn) -> None:
        """Test execution workflow for element-wise add graph."""
        sample_path = Path(__file__).parent.parent.parent / "graphs" / "sample_add.json"

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=sample_path,
            warmup_iters=2,
            benchmark_iters=5,
            engine_id=1,
        )

        graph_json_str = json.dumps(graph_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()
            executor.warmup(handle, variant_pack)
            result = executor.benchmark(handle, variant_pack)

            assert len(result.e2e_timings) == 5

            # Get output (z: [128, 256, 14, 14])
            output_data = buffer_manager.get_output_data(3)
            assert output_data is not None
            assert output_data.shape == (128, 256, 14, 14)

            import numpy as np

            assert not np.allclose(output_data, 0)

    def test_batchnorm_execution_workflow(self, hipdnn) -> None:
        """Test execution workflow for batchnorm inference graph."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_batchnorm.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=sample_path,
            warmup_iters=2,
            benchmark_iters=5,
            engine_id=1,
        )

        graph_json_str = json.dumps(graph_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()
            executor.warmup(handle, variant_pack)
            result = executor.benchmark(handle, variant_pack)

            assert len(result.e2e_timings) == 5

            # Get output (y: [32, 64, 28, 28])
            output_data = buffer_manager.get_output_data(6)
            assert output_data is not None
            assert output_data.shape == (32, 64, 28, 28)

            import numpy as np

            assert not np.allclose(output_data, 0)


@pytest.mark.gpu
class TestExecutionErrors:
    """Tests for execution error handling."""

    @pytest.fixture
    def hipdnn(self, plugin_paths: List[str]):
        """Get hipdnn_frontend module or skip if not available."""
        return _setup_hipdnn(plugin_paths)

    def test_execute_without_prepare_raises(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test that executing without prepare raises error."""
        from dnn_benchmarking.common.exceptions import ExecutionError

        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            warmup_iters=1,
            benchmark_iters=1,
        )

        graph_json_str = json.dumps(sample_conv_fwd_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()

        with pytest.raises(ExecutionError, match="not prepared"):
            executor.warmup(handle, {})

    def test_variant_pack_without_allocate_raises(
        self, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test that creating variant pack without allocation raises error."""
        from dnn_benchmarking.common.exceptions import ExecutionError

        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        buffer_manager = BufferManager(tensor_infos)

        with pytest.raises(ExecutionError, match="not allocated"):
            buffer_manager.create_variant_pack()


@pytest.mark.gpu
class TestPyTorchReferenceValidation:
    """Integration tests for PyTorch reference validation with GPU execution."""

    @pytest.fixture
    def hipdnn(self, plugin_paths: List[str]):
        """Get hipdnn_frontend module or skip if not available."""
        return _setup_hipdnn(plugin_paths)

    @pytest.fixture
    def pytorch_provider(self):
        """Get PyTorch reference provider or skip if not available."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not available")

        provider = ReferenceProviderRegistry.get_provider("pytorch")
        if not provider.is_available():
            pytest.skip("PyTorch provider not available")
        return provider

    def test_conv_fwd_validates_against_pytorch(
        self,
        hipdnn,
        pytorch_provider,
        sample_conv_fwd_json: Dict[str, Any],
    ) -> None:
        """Test that convolution output matches PyTorch reference."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            warmup_iters=2,
            benchmark_iters=5,
            engine_id=1,
        )

        graph_json_str = json.dumps(sample_conv_fwd_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()
            executor.warmup(handle, variant_pack)
            executor.benchmark(handle, variant_pack)

            # Get hipDNN output
            output_data = buffer_manager.get_output_data(0)
            assert output_data is not None

            # Collect input data for PyTorch reference
            input_data = {}
            for tensor_info in tensor_infos:
                if not tensor_info.is_virtual and not tensor_info.is_output:
                    data = buffer_manager.get_input_data(tensor_info.uid)
                    if data is not None:
                        input_data[tensor_info.uid] = data

            # Compute PyTorch reference
            reference_outputs = pytorch_provider.compute_reference(
                sample_conv_fwd_json, input_data
            )

            # Compare
            assert 0 in reference_outputs
            reference_data = reference_outputs[0].data

            # Use relaxed tolerances for floating point comparison
            comparator = ArrayComparator(rtol=1e-3, atol=1e-5)
            result = comparator.compare(
                output_data, reference_data, "hipDNN", "PyTorch"
            )

            assert (
                result.passed
            ), f"hipDNN output does not match PyTorch reference: {result.message}"

    def test_conv_fwd_bfloat16_validates_against_pytorch_when_supported(
        self,
        hipdnn,
        sample_conv_fwd_json: Dict[str, Any],
    ) -> None:
        """BF16 hipDNN output decode validates against a BF16 PyTorch reference."""
        graph_json = json.loads(json.dumps(sample_conv_fwd_json))
        graph_json["name"] = "sample_conv_fwd_bfloat16"
        graph_json["compute_data_type"] = "float"
        graph_json["io_data_type"] = "bfloat16"
        graph_json["intermediate_data_type"] = "bfloat16"
        for tensor in graph_json["tensors"]:
            tensor["data_type"] = "bfloat16"

        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(graph_json)
        config = SuiteConfig(
            warmup_iters=1,
            benchmark_iters=1,
            seed=42,
            validation=ValidationConfig(provider="pytorch"),
            metrics=MetricsConfig(tier="off"),
        )

        result = run_graph_all_providers(
            graph_path=Path("/test/graph_bfloat16.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=hipdnn.Handle(),
        )

        engine_rows = [row for row in result.results if row.role == "engine"]
        successful_rows = [row for row in engine_rows if row.status == "success"]
        if not successful_rows:
            reasons = "; ".join(
                row.skip_reason or row.error_message or "no details"
                for row in engine_rows
            )
            pytest.skip(
                f"No hipDNN engine supports BF16 conv validation graph: {reasons}"
            )

        assert any(
            row.correctness is not None and row.correctness.passed
            for row in successful_rows
        )

    @pytest.mark.parametrize("graph_name", ["sample_sdpa.json", "sample_mha_sdpa.json"])
    def test_sdpa_validates_against_pytorch_when_supported(
        self,
        hipdnn,
        graph_name: str,
    ) -> None:
        """Supported SDPA engines validate against the PyTorch reference."""
        graph_path = Path(__file__).parent.parent.parent / "graphs" / graph_name
        loader = GraphLoader()
        graph_json = loader.load_json(graph_path)
        tensor_infos = loader.extract_tensor_info(graph_json)
        config = SuiteConfig(
            warmup_iters=1,
            benchmark_iters=1,
            seed=42,
            validation=ValidationConfig(provider="pytorch"),
            metrics=MetricsConfig(tier="off"),
        )

        result = run_graph_all_providers(
            graph_path=graph_path,
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=hipdnn.Handle(),
        )

        reference_rows = [row for row in result.results if row.role == "reference"]
        engine_rows = [row for row in result.results if row.role == "engine"]
        successful_rows = [row for row in engine_rows if row.status == "success"]
        if not successful_rows:
            reasons = "; ".join(
                row.skip_reason or row.error_message or "no details"
                for row in engine_rows
            )
            pytest.skip(f"No hipDNN engine supports {graph_name}: {reasons}")

        assert len(reference_rows) == 1
        assert reference_rows[0].status == "success"
        assert any(
            row.correctness is not None and row.correctness.passed
            for row in successful_rows
        )

    def test_suite_validate_pytorch_reports_timed_reference_row(
        self,
        hipdnn,
        sample_conv_fwd_json: Dict[str, Any],
    ) -> None:
        """Suite validation reports PyTorch and hipDNN timing rows side by side."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)
        config = SuiteConfig(
            warmup_iters=1,
            benchmark_iters=2,
            seed=42,
            validation=ValidationConfig(provider="pytorch"),
            metrics=MetricsConfig(tier="off"),
        )

        result = run_graph_all_providers(
            graph_path=Path("/test/graph.json"),
            graph_json=sample_conv_fwd_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=hipdnn.Handle(),
        )

        reference_rows = [row for row in result.results if row.role == "reference"]
        engine_rows = [row for row in result.results if row.role == "engine"]

        assert len(reference_rows) == 1
        assert reference_rows[0].provider == "pytorch"
        assert reference_rows[0].status == "success"
        assert reference_rows[0].e2e_stats is not None
        assert reference_rows[0].gpu_kernel_stats is not None
        assert engine_rows
        assert any(row.status == "success" for row in engine_rows)

    @pytest.mark.xfail(reason="MIOpen plugin doesn't support pointwise operations yet")
    def test_relu_validates_against_pytorch(self, hipdnn, pytorch_provider) -> None:
        """Test that ReLU output matches PyTorch reference."""
        sample_path = (
            Path(__file__).parent.parent.parent / "graphs" / "sample_relu.json"
        )

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=sample_path,
            warmup_iters=2,
            benchmark_iters=5,
            engine_id=1,
        )

        graph_json_str = json.dumps(graph_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()
            executor.warmup(handle, variant_pack)
            executor.benchmark(handle, variant_pack)

            # Get hipDNN output (uid=2 for relu output)
            output_data = buffer_manager.get_output_data(2)
            assert output_data is not None

            # Collect input data
            input_data = {}
            for tensor_info in tensor_infos:
                if not tensor_info.is_virtual and not tensor_info.is_output:
                    data = buffer_manager.get_input_data(tensor_info.uid)
                    if data is not None:
                        input_data[tensor_info.uid] = data

            # Compute PyTorch reference
            reference_outputs = pytorch_provider.compute_reference(
                graph_json, input_data
            )

            # Compare
            assert 2 in reference_outputs
            reference_data = reference_outputs[2].data

            comparator = ArrayComparator(rtol=1e-5, atol=1e-8)
            result = comparator.compare(
                output_data, reference_data, "hipDNN", "PyTorch"
            )

            assert (
                result.passed
            ), f"hipDNN ReLU output does not match PyTorch: {result.message}"

    @pytest.mark.xfail(reason="MIOpen plugin doesn't support pointwise operations yet")
    def test_add_validates_against_pytorch(self, hipdnn, pytorch_provider) -> None:
        """Test that Add output matches PyTorch reference."""
        sample_path = Path(__file__).parent.parent.parent / "graphs" / "sample_add.json"

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)
        tensor_infos = loader.extract_tensor_info(graph_json)

        config = BenchmarkConfig(
            graph_path=sample_path,
            warmup_iters=2,
            benchmark_iters=5,
            engine_id=1,
        )

        graph_json_str = json.dumps(graph_json)
        executor = Executor(graph_json_str, config)

        handle = hipdnn.Handle()
        executor.prepare(handle)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=42)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()
            executor.warmup(handle, variant_pack)
            executor.benchmark(handle, variant_pack)

            # Get hipDNN output (uid=3 for add output)
            output_data = buffer_manager.get_output_data(3)
            assert output_data is not None

            # Collect input data
            input_data = {}
            for tensor_info in tensor_infos:
                if not tensor_info.is_virtual and not tensor_info.is_output:
                    data = buffer_manager.get_input_data(tensor_info.uid)
                    if data is not None:
                        input_data[tensor_info.uid] = data

            # Compute PyTorch reference
            reference_outputs = pytorch_provider.compute_reference(
                graph_json, input_data
            )

            # Compare
            assert 3 in reference_outputs
            reference_data = reference_outputs[3].data

            comparator = ArrayComparator(rtol=1e-5, atol=1e-8)
            result = comparator.compare(
                output_data, reference_data, "hipDNN", "PyTorch"
            )

            assert (
                result.passed
            ), f"hipDNN Add output does not match PyTorch: {result.message}"


@pytest.mark.gpu
class TestHardEngineSelectBindings:
    """Real-backend coverage for the hard-select / read-back Graph bindings.

    Drives a live Graph through create_execution_plan_ext() -> build_plans() ->
    get_execution_plan_engine_id(), exercising the actual nanobind surface and
    the C++ getter (which the executor unit tests stub out).
    """

    @pytest.fixture
    def hipdnn(self, plugin_paths: List[str]):
        """Get hipdnn_frontend module or skip if not available."""
        return _setup_hipdnn(plugin_paths)

    def _built_graph(self, graph_json_str: str, handle):
        """Return a Graph with the operation graph built (no execution plan yet).

        Reuses Executor's graph construction (which normalises the JSON) and
        hands back the raw hipdnn_frontend.Graph so the test can drive the
        execution-plan bindings directly.
        """
        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"), warmup_iters=0, benchmark_iters=1
        )
        executor = Executor(graph_json_str, config)
        executor._build_through_operation_graph(handle)
        return executor._graph

    def test_hard_select_and_read_back_matches(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Hard-selecting a ranked engine builds, and the read-back reports it."""
        handle = hipdnn.Handle()
        graph_json_str = json.dumps(sample_conv_fwd_json)

        discovery_graph = self._built_graph(graph_json_str, handle)
        ranked = [int(e) for e in discovery_graph.get_ranked_engine_ids()]
        assert ranked, "expected at least one ranked engine for the graph"
        engine = ranked[0]

        graph = self._built_graph(graph_json_str, handle)
        result = graph.create_execution_plan_ext(engine)
        assert not result.is_bad(), result.get_message()
        assert not graph.build_plans().is_bad()
        # The getter returns the engine that actually backs the built plan.
        assert graph.get_execution_plan_engine_id() == engine

    def test_hard_select_inapplicable_engine_is_bad(
        self, hipdnn, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Hard-selecting an engine id the backend cannot honor returns is_bad()."""
        handle = hipdnn.Handle()
        graph = self._built_graph(json.dumps(sample_conv_fwd_json), handle)
        result = graph.create_execution_plan_ext(123456789)
        assert result.is_bad()
