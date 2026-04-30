# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for suite_runner module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.execution.suite_runner import (
    run_graph_all_providers,
    _resolve_engine_name,
    _get_reference_provider,
    _check_correctness,
    _is_support_error,
)
from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.common.exceptions import ExecutionError
from dnn_benchmarking.reporting.statistics import BenchmarkStats
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
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


def _make_bm_mock():
    """Create a BufferManager mock that supports the context-manager protocol."""
    mock_bm = MagicMock()
    mock_bm.__enter__ = MagicMock(return_value=mock_bm)
    mock_bm.__exit__ = MagicMock(return_value=False)
    mock_bm.create_variant_pack.return_value = {1: 100}
    return mock_bm


def _make_exec_factory(
    engine_ids=None,
    init_time_ms: float = 1.0,
    has_kernel_timings: bool = False,
    prepare_side_effect=None,
    discover_side_effect=None,
):
    """Build a factory for Executor() that handles both discovery and execution.

    The first Executor() call (in run_graph_all_providers) is for discovery
    and only uses .discover_engines(); subsequent calls are per-engine and
    use .prepare(), .warmup(), .benchmark(). All instances share the same
    mock by default; override with side_effects when behaviour must differ.
    """

    def make_instance(*args, **kwargs):
        m = MagicMock()
        m.init_time_ms = init_time_ms
        if discover_side_effect is not None:
            m.discover_engines.side_effect = discover_side_effect
        else:
            m.discover_engines.return_value = engine_ids or []
        if prepare_side_effect is not None:
            m.prepare.side_effect = prepare_side_effect
        bench_result = MagicMock()
        bench_result.e2e_timings = [1.0]
        bench_result.kernel_timings = [0.5] if has_kernel_timings else None
        bench_result.has_kernel_timings = has_kernel_timings
        m.benchmark.return_value = bench_result
        return m

    return make_instance


class TestRunGraphAllProviders:
    """Tests for run_graph_all_providers function."""

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_one_result_per_discovered_engine(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """run_graph_all_providers returns one ProviderEngineResult per discovered engine ID."""
        mock_resolve_name.side_effect = lambda eid: f"engine_{eid}"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0, 1, 2], has_kernel_timings=True
        )
        mock_bm_cls.return_value = _make_bm_mock()

        config = _make_config()
        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        assert isinstance(result, GraphResult)
        assert len(result.results) == 3
        assert [r.engine_id for r in result.results] == [0, 1, 2]
        assert [r.provider for r in result.results] == [
            "engine_0",
            "engine_1",
            "engine_2",
        ]

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_prepare_failure_records_error_status(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """When Executor.prepare() fails, the result is status='error' with no timing."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=ExecutionError("build failed"),
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "error"
        assert "build failed" in r.error_message
        assert r.cpu_build_time_ms is None
        assert r.gpu_kernel_stats is None
        assert r.e2e_stats is None

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_check_support_failure_records_skipped_status(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """An ExecutionError that looks like a support-check failure is recorded as skipped."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=ExecutionError(
                "Backend support check failed: not supported"
            ),
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "skipped"
        assert r.skip_reason is not None

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_successful_execution_records_separated_timing(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """Success: status='success' with separate cpu_build_time_ms / gpu_kernel_stats / e2e_stats."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0], init_time_ms=12.5, has_kernel_timings=True
        )
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=_make_config(),
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.status == "success"
        assert r.cpu_build_time_ms == 12.5
        assert isinstance(r.gpu_kernel_stats, BenchmarkStats)
        assert isinstance(r.e2e_stats, BenchmarkStats)

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_cpu_build_time_from_init_time_ms(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """cpu_build_time_ms comes from Executor.init_time_ms."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0], init_time_ms=42.0
        )
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert result.results[0].cpu_build_time_ms == 42.0


class TestDiscoveryFailure:
    """Discovery-level failures are surfaced as graph-level errors."""

    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    def test_discovery_exception_is_recorded_as_graph_error(self, mock_exec_cls):
        """When discover_engines raises, the graph gets a single error entry."""
        mock_exec_cls.side_effect = _make_exec_factory(
            discover_side_effect=ExecutionError("backend rejected graph")
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "error"
        assert "Engine discovery failed" in r.error_message
        assert "backend rejected graph" in r.error_message

    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    def test_empty_discovery_recorded_as_graph_error(self, mock_exec_cls):
        """When discovery returns no engines, surface as a graph-level error."""
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[])

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        assert result.results[0].status == "error"
        assert "No engines discovered" in result.results[0].error_message

    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    def test_no_engines_runtime_error_recorded_as_skipped(self, mock_exec_cls):
        """C++ binding throws RuntimeError when no engines support the graph;
        we classify that as 'skipped' (unsupported) rather than 'error'."""
        mock_exec_cls.side_effect = _make_exec_factory(
            discover_side_effect=RuntimeError(
                "Failed to get ranked engine ids: No engine configurations available for the graph."
            )
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        r = result.results[0]
        assert r.status == "skipped"
        assert "No engine configurations" in (r.skip_reason or "")

    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    def test_engine_filter_excludes_everything(self, mock_exec_cls):
        """When engine_filter excludes every discovered engine, surface as error."""
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0, 1])

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(engine_filter=[99]),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        assert result.results[0].status == "error"
        assert "filter" in result.results[0].error_message.lower()


class TestSuiteConfigValidation:
    """Tests for SuiteConfig dataclass validation."""

    def test_valid_config(self):
        config = SuiteConfig(warmup_iters=5, benchmark_iters=10)
        assert config.warmup_iters == 5
        assert config.benchmark_iters == 10
        assert config.engine_filter is None
        assert config.rtol == 1e-5
        assert config.atol == 1e-8
        assert config.gpu_backend == "auto"
        assert config.reference_provider == "none"

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="warmup_iters"):
            SuiteConfig(warmup_iters=-1, benchmark_iters=10)

    def test_zero_benchmark_iters_raises(self):
        with pytest.raises(ValueError, match="benchmark_iters"):
            SuiteConfig(warmup_iters=0, benchmark_iters=0)

    def test_engine_filter_accepts_list(self):
        config = SuiteConfig(engine_filter=[1, 2, 3])
        assert config.engine_filter == [1, 2, 3]

    def test_engine_filter_empty_list_raises(self):
        with pytest.raises(ValueError, match="engine_filter"):
            SuiteConfig(engine_filter=[])

    def test_engine_filter_accepts_negative_ids(self):
        """Engine IDs are FNV-1a hashes; negative values must be allowed."""
        config = SuiteConfig(engine_filter=[1, -1234567890])
        assert config.engine_filter == [1, -1234567890]

    def test_verbose_default_false(self):
        config = SuiteConfig()
        assert config.verbose is False

    def test_verbose_can_be_set(self):
        config = SuiteConfig(verbose=True)
        assert config.verbose is True

    def test_invalid_gpu_backend_raises(self):
        with pytest.raises(ValueError, match="gpu_backend"):
            SuiteConfig(gpu_backend="bogus")

    def test_invalid_reference_provider_raises(self):
        with pytest.raises(ValueError, match="reference_provider"):
            SuiteConfig(reference_provider="not_a_real_provider")

    def test_default_gpu_backend_and_reference_provider_accepted(self):
        config = SuiteConfig()
        assert config.gpu_backend == "auto"
        assert config.reference_provider == "none"
        for backend in ("torch", "auto", "none"):
            SuiteConfig(gpu_backend=backend)
        for provider in ("none", "pytorch", "cpu_plugin"):
            SuiteConfig(reference_provider=provider)


class TestIsSupportError:
    """Tests for _is_support_error keyword classification."""

    def test_support_check_failed_is_support_error(self):
        assert _is_support_error("Backend support check failed: bad config") is True

    def test_not_supported_is_support_error(self):
        assert _is_support_error("Operation not supported on this engine") is True

    def test_unsupported_is_support_error(self):
        assert _is_support_error("unsupported tensor layout") is True

    def test_no_engine_is_support_error(self):
        assert _is_support_error("no engine available for this graph") is True

    def test_case_insensitive(self):
        assert _is_support_error("UNSUPPORTED") is True
        assert _is_support_error("Not Supported") is True

    def test_unrelated_error_not_support_error(self):
        assert _is_support_error("out of memory") is False

    def test_empty_string_not_support_error(self):
        assert _is_support_error("") is False


class TestEngineFilter:
    """Tests for engine filter behavior."""

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_engine_filter_limits_iteration(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """When --engine filter is set, only that engine ID is iterated."""
        mock_resolve_name.side_effect = lambda eid: f"engine_{eid}"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0, 1, 2])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(engine_filter=[2]),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        assert result.results[0].engine_id == 2

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_engine_filter_list_keeps_intersection(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """engine_filter=[1, 3, 99]: engines 1 and 3 run; 99 (not discovered) is dropped."""
        mock_resolve_name.side_effect = lambda eid: f"engine_{eid}"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0, 1, 2, 3])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(engine_filter=[1, 3, 99]),
            handle=MagicMock(),
        )

        engine_ids = sorted(r.engine_id for r in result.results)
        assert engine_ids == [1, 3]


class TestNoRetryOnFailure:
    """Single attempt per engine -- no automatic retry."""

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_no_retry_on_failure(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """No retry on failure -- single attempt per engine."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=ExecutionError("fail"),
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        # One Executor for discovery + one for the single failed engine.
        assert mock_exec_cls.call_count == 2
        assert result.results[0].status == "error"


class TestCorrectnessChecking:
    """Tests for correctness checking via the reference provider path."""

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner._check_correctness")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_tolerance_match_populated_from_comparator(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_check_corr,
        mock_get_ref,
        mock_resolve_name,
    ):
        """Successful execution populates correctness.tolerance_match from the validator."""
        mock_resolve_name.return_value = "engine_0"

        mock_get_ref.return_value = MagicMock()
        mock_check_corr.return_value = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-8,
            max_abs_diff=1e-7,
            max_rel_diff=1e-6,
        )

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=_make_config(),
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.tolerance_match is True
        assert r.correctness.execution_success is True

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_tolerance_match_none_when_not_requested(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """When --validate is not requested, tolerance_match is None (no correctness performed)."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),  # reference_provider defaults to "none"
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.tolerance_match is None
        assert r.correctness.execution_success is True

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_tolerance_match_false_when_requested_but_unsupported(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """--validate requested but provider doesn't support graph -> tolerance_match=False."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None  # provider unavailable for this graph

        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(reference_provider="pytorch"),
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.tolerance_match is False
        assert r.correctness.execution_success is True
        assert "does not support" in (r.correctness.error_message or "")

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_execution_success_false_on_error(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """correctness.execution_success is False when benchmark errors."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=ExecutionError("boom"),
        )

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
            handle=MagicMock(),
        )

        r = result.results[0]
        assert r.correctness is not None
        assert r.correctness.execution_success is False
        assert r.correctness.tolerance_match is None


class TestCheckCorrectnessOutputCount:
    """_check_correctness returns tolerance_match=False when no outputs are comparable."""

    def test_no_outputs_returns_false(self):
        bm = MagicMock()
        bm.get_input_data.return_value = None
        bm.get_output_data.return_value = None

        ref_provider = MagicMock()
        ref_provider.compute_reference.return_value = {}
        ref_provider.name = "pytorch"

        config = SuiteConfig(reference_provider="pytorch")
        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[],
            graph_json=_make_graph_json(),
            ref_provider=ref_provider,
            config=config,
        )

        assert result.tolerance_match is False
        assert result.execution_success is True
        assert "No output tensors to compare" in (result.error_message or "")


class TestResolveEngineName:
    """Tests for _resolve_engine_name fallback behavior."""

    def test_falls_back_to_hex_when_lookup_fails(self):
        """If hipdnn_frontend isn't importable, the helper falls back to a hex display."""
        # Force the import inside _resolve_engine_name to fail by injecting a
        # missing module entry. We use unittest.mock.patch on builtins.__import__
        # to surgically reject just hipdnn_frontend.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "hipdnn_frontend":
                raise ImportError("simulated missing module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert _resolve_engine_name(0xABC) == "engine_0xabc"
