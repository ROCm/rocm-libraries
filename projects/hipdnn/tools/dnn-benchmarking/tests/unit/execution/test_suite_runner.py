# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for suite_runner module."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from dnn_benchmarking.execution.suite_runner import (
    run_graph_all_providers,
    discover_providers,
    discover_engines,
    _get_reference_provider,
    _check_correctness,
)
from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    TimingStats,
)


def _make_tensor_info(uid: int, is_output: bool = False, is_virtual: bool = False):
    """Create a mock TensorInfo object."""
    ti = MagicMock()
    ti.uid = uid
    ti.is_output = is_output
    ti.is_virtual = is_virtual
    return ti


def _make_graph_json():
    """Create a minimal graph JSON dict."""
    return {"name": "test_graph", "nodes": [], "tensors": []}


def _make_config(**overrides):
    """Create a SuiteConfig with optional overrides."""
    defaults = {
        "warmup_iters": 2,
        "benchmark_iters": 3,
        "seed": 42,
        "gpu_backend": "none",
    }
    defaults.update(overrides)
    return SuiteConfig(**defaults)


class TestRunGraphAllProviders:
    """Tests for run_graph_all_providers function."""

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_returns_graph_result_with_one_result_per_provider_engine(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 1: run_graph_all_providers returns a GraphResult with one
        ProviderEngineResult per discovered provider/engine combo."""
        mock_disc_providers.return_value = ["providerA", "providerB"]
        mock_disc_engines.return_value = [0, 1]
        mock_get_ref.return_value = None

        # Set up executor mock
        mock_exec = MagicMock()
        mock_exec.init_time_ms = 5.0
        mock_result = MagicMock()
        mock_result.e2e_timings = [1.0, 2.0, 3.0]
        mock_result.kernel_timings = [0.5, 1.0, 1.5]
        mock_result.has_kernel_timings = True
        mock_exec.benchmark.return_value = mock_result
        mock_exec_cls.return_value = mock_exec

        # Set up buffer manager mock
        mock_bm = MagicMock()
        mock_bm.__enter__ = MagicMock(return_value=mock_bm)
        mock_bm.__exit__ = MagicMock(return_value=False)
        mock_bm.create_variant_pack.return_value = {1: 100}
        mock_bm_cls.return_value = mock_bm

        config = _make_config()
        tensor_infos = [_make_tensor_info(1), _make_tensor_info(2, is_output=True)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        assert isinstance(result, GraphResult)
        # 2 providers x 2 engines = 4 results
        assert len(result.results) == 4

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_prepare_failure_records_error_status(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 2: When a provider/engine combo fails during prepare(), it records
        status='error' with error message, no timing data."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0]
        mock_get_ref.return_value = None

        mock_exec = MagicMock()
        mock_exec.prepare.side_effect = ExecutionError("build failed")
        mock_exec_cls.return_value = mock_exec

        config = _make_config()
        tensor_infos = [_make_tensor_info(1)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "error"
        assert "build failed" in r.error_message
        assert r.cpu_build_time_ms is None
        assert r.gpu_kernel_stats is None
        assert r.e2e_stats is None

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_check_support_failure_records_skipped_status(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 3: When a provider/engine combo does not support the graph
        (check_support fails), it records status='skipped' with reason."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0]
        mock_get_ref.return_value = None

        mock_exec = MagicMock()
        mock_exec.prepare.side_effect = ExecutionError(
            "Backend support check failed: not supported"
        )
        mock_exec_cls.return_value = mock_exec

        config = _make_config()
        tensor_infos = [_make_tensor_info(1)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "skipped"
        assert r.skip_reason is not None

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_successful_execution_records_separated_timing(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 4: Successful execution records status='success' with separate
        cpu_build_time_ms, gpu_kernel_stats, e2e_stats."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0]
        mock_get_ref.return_value = None

        mock_exec = MagicMock()
        mock_exec.init_time_ms = 12.5
        mock_result = MagicMock()
        mock_result.e2e_timings = [1.0, 2.0, 3.0]
        mock_result.kernel_timings = [0.5, 1.0, 1.5]
        mock_result.has_kernel_timings = True
        mock_exec.benchmark.return_value = mock_result
        mock_exec_cls.return_value = mock_exec

        mock_bm = MagicMock()
        mock_bm.__enter__ = MagicMock(return_value=mock_bm)
        mock_bm.__exit__ = MagicMock(return_value=False)
        mock_bm.create_variant_pack.return_value = {1: 100}
        mock_bm_cls.return_value = mock_bm

        config = _make_config()
        tensor_infos = [_make_tensor_info(1), _make_tensor_info(2, is_output=True)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        r = result.results[0]
        assert r.status == "success"
        assert r.cpu_build_time_ms == 12.5
        assert isinstance(r.gpu_kernel_stats, TimingStats)
        assert isinstance(r.e2e_stats, TimingStats)

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_cpu_build_time_from_init_time_ms(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 5: cpu_build_time_ms comes from Executor.init_time_ms."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0]
        mock_get_ref.return_value = None

        mock_exec = MagicMock()
        mock_exec.init_time_ms = 42.0
        mock_result = MagicMock()
        mock_result.e2e_timings = [1.0]
        mock_result.kernel_timings = None
        mock_result.has_kernel_timings = False
        mock_exec.benchmark.return_value = mock_result
        mock_exec_cls.return_value = mock_exec

        mock_bm = MagicMock()
        mock_bm.__enter__ = MagicMock(return_value=mock_bm)
        mock_bm.__exit__ = MagicMock(return_value=False)
        mock_bm.create_variant_pack.return_value = {1: 100}
        mock_bm_cls.return_value = mock_bm

        config = _make_config()
        tensor_infos = [_make_tensor_info(1)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        assert result.results[0].cpu_build_time_ms == 42.0


class TestSuiteConfigValidation:
    """Tests for SuiteConfig dataclass validation."""

    def test_valid_config(self):
        """Test 6: SuiteConfig dataclass validates provider/engine filter fields."""
        config = SuiteConfig(warmup_iters=5, benchmark_iters=10)
        assert config.warmup_iters == 5
        assert config.benchmark_iters == 10
        assert config.provider_filter is None
        assert config.engine_filter is None
        assert config.rtol == 1e-5
        assert config.atol == 1e-8
        assert config.gpu_backend == "auto"
        assert config.reference_provider == "pytorch"

    def test_negative_warmup_raises(self):
        """SuiteConfig rejects negative warmup_iters."""
        with pytest.raises(ValueError, match="warmup_iters"):
            SuiteConfig(warmup_iters=-1, benchmark_iters=10)

    def test_zero_benchmark_iters_raises(self):
        """SuiteConfig rejects zero benchmark_iters."""
        with pytest.raises(ValueError, match="benchmark_iters"):
            SuiteConfig(warmup_iters=0, benchmark_iters=0)


class TestProviderFilter:
    """Tests for provider/engine filter behavior."""

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_provider_filter_limits_iteration(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 7: When --provider filter is set, only that provider is iterated."""
        mock_disc_providers.return_value = ["provA", "provB", "provC"]
        mock_disc_engines.return_value = [0]
        mock_get_ref.return_value = None

        mock_exec = MagicMock()
        mock_exec.init_time_ms = 1.0
        mock_result = MagicMock()
        mock_result.e2e_timings = [1.0]
        mock_result.kernel_timings = None
        mock_result.has_kernel_timings = False
        mock_exec.benchmark.return_value = mock_result
        mock_exec_cls.return_value = mock_exec

        mock_bm = MagicMock()
        mock_bm.__enter__ = MagicMock(return_value=mock_bm)
        mock_bm.__exit__ = MagicMock(return_value=False)
        mock_bm.create_variant_pack.return_value = {1: 100}
        mock_bm_cls.return_value = mock_bm

        config = _make_config(provider_filter="provB")
        tensor_infos = [_make_tensor_info(1)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        assert len(result.results) == 1
        assert result.results[0].provider == "provB"

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_engine_filter_limits_iteration(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 8: When --engine filter is set, only that engine ID is iterated per provider."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0, 1, 2]
        mock_get_ref.return_value = None

        mock_exec = MagicMock()
        mock_exec.init_time_ms = 1.0
        mock_result = MagicMock()
        mock_result.e2e_timings = [1.0]
        mock_result.kernel_timings = None
        mock_result.has_kernel_timings = False
        mock_exec.benchmark.return_value = mock_result
        mock_exec_cls.return_value = mock_exec

        mock_bm = MagicMock()
        mock_bm.__enter__ = MagicMock(return_value=mock_bm)
        mock_bm.__exit__ = MagicMock(return_value=False)
        mock_bm.create_variant_pack.return_value = {1: 100}
        mock_bm_cls.return_value = mock_bm

        config = _make_config(engine_filter=2)
        tensor_infos = [_make_tensor_info(1)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        assert len(result.results) == 1
        assert result.results[0].engine_id == 2


class TestNoRetryOnFailure:
    """Tests for no-retry failure behavior (D-10)."""

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_no_retry_on_failure(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 9: No retry on failure -- single attempt per provider/engine combination."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0]
        mock_get_ref.return_value = None

        mock_exec = MagicMock()
        mock_exec.prepare.side_effect = ExecutionError("fail")
        mock_exec_cls.return_value = mock_exec

        config = _make_config()
        tensor_infos = [_make_tensor_info(1)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        # Executor created exactly once (no retry)
        assert mock_exec_cls.call_count == 1
        assert result.results[0].status == "error"


class TestCorrectnessChecking:
    """Tests for correctness checking via ArrayComparator (CORR-01/02)."""

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner._check_correctness")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_tolerance_match_populated_from_comparator(
        self, mock_bm_cls, mock_exec_cls, mock_check_corr, mock_get_ref,
        mock_disc_providers, mock_disc_engines
    ):
        """Test 10: Successful execution populates correctness.tolerance_match
        from ArrayComparator output (per CORR-02)."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0]

        mock_ref = MagicMock()
        mock_get_ref.return_value = mock_ref

        corr_result = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-8,
            max_abs_diff=1e-7,
            max_rel_diff=1e-6,
        )
        mock_check_corr.return_value = corr_result

        mock_exec = MagicMock()
        mock_exec.init_time_ms = 1.0
        mock_result = MagicMock()
        mock_result.e2e_timings = [1.0]
        mock_result.kernel_timings = None
        mock_result.has_kernel_timings = False
        mock_exec.benchmark.return_value = mock_result
        mock_exec_cls.return_value = mock_exec

        mock_bm = MagicMock()
        mock_bm.__enter__ = MagicMock(return_value=mock_bm)
        mock_bm.__exit__ = MagicMock(return_value=False)
        mock_bm.create_variant_pack.return_value = {1: 100}
        mock_bm_cls.return_value = mock_bm

        config = _make_config()
        tensor_infos = [_make_tensor_info(1), _make_tensor_info(2, is_output=True)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.tolerance_match is True
        assert r.correctness.execution_success is True

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_tolerance_match_none_when_ref_unavailable(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 11: When reference provider is unavailable, correctness.tolerance_match
        is None and correctness.error_message explains why."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0]
        mock_get_ref.return_value = None  # No reference provider

        mock_exec = MagicMock()
        mock_exec.init_time_ms = 1.0
        mock_result = MagicMock()
        mock_result.e2e_timings = [1.0]
        mock_result.kernel_timings = None
        mock_result.has_kernel_timings = False
        mock_exec.benchmark.return_value = mock_result
        mock_exec_cls.return_value = mock_exec

        mock_bm = MagicMock()
        mock_bm.__enter__ = MagicMock(return_value=mock_bm)
        mock_bm.__exit__ = MagicMock(return_value=False)
        mock_bm.create_variant_pack.return_value = {1: 100}
        mock_bm_cls.return_value = mock_bm

        config = _make_config()
        tensor_infos = [_make_tensor_info(1)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.tolerance_match is None
        assert r.correctness.execution_success is True
        assert r.correctness.error_message is not None

    @patch("dnn_benchmarking.execution.suite_runner.discover_engines")
    @patch("dnn_benchmarking.execution.suite_runner.discover_providers")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_execution_success_false_on_error(
        self, mock_bm_cls, mock_exec_cls, mock_get_ref, mock_disc_providers, mock_disc_engines
    ):
        """Test 12: correctness.execution_success is False when benchmark errors (CORR-01)."""
        mock_disc_providers.return_value = ["provA"]
        mock_disc_engines.return_value = [0]
        mock_get_ref.return_value = None

        mock_exec = MagicMock()
        mock_exec.prepare.side_effect = ExecutionError("boom")
        mock_exec_cls.return_value = mock_exec

        config = _make_config()
        tensor_infos = [_make_tensor_info(1)]
        graph_json = _make_graph_json()
        handle = MagicMock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=graph_json,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.execution_success is False
        assert r.correctness.tolerance_match is None
