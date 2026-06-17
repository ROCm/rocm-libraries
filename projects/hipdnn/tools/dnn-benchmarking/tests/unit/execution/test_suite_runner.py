# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for suite_runner module."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from dnn_benchmarking.execution.suite_runner import (
    run_graph_all_providers,
    run_graph_pytorch_backend,
    _resolve_engine_name,
    _get_reference_provider,
    _check_correctness,
    _run_timed_pytorch_row,
    set_plugin_path,
)
from dnn_benchmarking.config.benchmark_config import (
    MetricsConfig,
    ReferenceProviderName,
    SuiteConfig,
    ValidationConfig,
)
from dnn_benchmarking.common.exceptions import ExecutionError, UnsupportedGraphError
from dnn_benchmarking.reporting.statistics import BenchmarkStats
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
)
from dnn_benchmarking.validation.reference_provider import ReferenceOutput


def _make_tensor_info(
    uid: int,
    is_output: bool = False,
    is_virtual: bool = False,
    data_type: str = "float",
    dims=None,
    strides=None,
    value=None,
):
    """Create a mock TensorInfo object."""
    ti = MagicMock()
    ti.uid = uid
    ti.is_output = is_output
    ti.is_virtual = is_virtual
    ti.data_type = data_type
    ti.dims = dims or [1]
    ti.strides = strides or []
    ti.value = value
    ti.is_pass_by_value = value is not None
    ti.storage_elements = 1
    ti.size_bytes = 4
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


class TestPluginPathLoading:
    """Explicit benchmark plugin paths should replace default plugin search paths."""

    def test_set_plugin_path_defaults_to_absolute_loading(self) -> None:
        hipdnn = MagicMock()
        hipdnn.PluginLoadingMode.ABSOLUTE = "absolute"

        set_plugin_path(hipdnn, Path("/plugins/engines"))

        hipdnn.set_engine_plugin_paths.assert_called_once_with(
            ["/plugins/engines"], "absolute"
        )


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
        """An UnsupportedGraphError is recorded as skipped."""
        mock_resolve_name.return_value = "engine_0"
        mock_get_ref.return_value = None

        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0],
            prepare_side_effect=UnsupportedGraphError(
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

    def test_input_generation_exception_recorded_as_graph_error(self):
        """Bad tensor metadata during shared input generation does not abort the suite."""
        with (
            patch("dnn_benchmarking.execution.suite_runner.Executor") as mock_exec_cls,
            patch(
                "dnn_benchmarking.execution.suite_runner._get_reference_provider",
                return_value=None,
            ),
            patch(
                "dnn_benchmarking.execution.suite_runner.generate_input_data",
                side_effect=ValueError("bad tensor strides"),
            ),
        ):
            mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[7])

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
        assert r.provider == "unknown"
        assert "Input data generation failed" in r.error_message
        assert "bad tensor strides" in r.error_message
        assert r.correctness is not None
        assert r.correctness.passed is False
        assert result.engine_ids == [7]

    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    def test_no_engines_unsupported_error_recorded_as_skipped(self, mock_exec_cls):
        """UnsupportedGraphError during discovery is recorded as skipped."""
        mock_exec_cls.side_effect = _make_exec_factory(
            discover_side_effect=UnsupportedGraphError(
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

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_engine_filter_runs_explicit_id_without_discovery(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """Explicit --engine IDs run in CLI order without discovery filtering."""
        mock_resolve_name.side_effect = lambda eid: f"engine_{eid}"
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[0, 1])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(engine_filter=[99]),
            handle=MagicMock(),
        )

        assert len(result.results) == 1
        assert result.results[0].status == "success"
        assert result.results[0].engine_id == 99


class TestSuiteConfigValidation:
    """Tests for SuiteConfig dataclass validation."""

    def test_valid_config(self):
        config = SuiteConfig(warmup_iters=5, benchmark_iters=10)
        assert config.warmup_iters == 5
        assert config.benchmark_iters == 10
        assert config.engine_filter is None
        assert config.validation.rtol is None
        assert config.validation.atol is None
        assert config.validation.tolerance_override is None
        assert config.validation.provider is ReferenceProviderName.NONE

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

    def test_default_reference_provider_accepted(self):
        config = SuiteConfig()
        assert config.validation.provider is ReferenceProviderName.NONE
        for provider in ("none", "pytorch"):
            SuiteConfig(validation=ValidationConfig(provider=provider))


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
        """engine_filter=[1, 3, 99] runs exactly those IDs in caller order."""
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

        engine_ids = [r.engine_id for r in result.results]
        assert engine_ids == [1, 3, 99]

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_same_engine_runs_with_distinct_plugin_paths(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """Repeated engine IDs are separate ordered selections."""
        mock_resolve_name.side_effect = lambda eid: f"engine_{eid}"
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(has_kernel_timings=True)
        mock_bm_cls.return_value = _make_bm_mock()
        hipdnn = MagicMock()
        hipdnn.PluginLoadingMode.ABSOLUTE = "absolute"
        hipdnn.Handle.side_effect = [MagicMock(), MagicMock()]

        with patch.dict("sys.modules", {"hipdnn_frontend": hipdnn}):
            result = run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(
                    engine_filter=[1, 1],
                    plugin_paths=[Path("/plugins/a"), Path("/plugins/b")],
                ),
                handle=None,
            )

        assert [r.engine_id for r in result.results] == [1, 1]
        assert [r.plugin_path for r in result.results] == [
            "/plugins/a",
            "/plugins/b",
        ]
        hipdnn.set_engine_plugin_paths.assert_has_calls(
            [
                call(["/plugins/a"], "absolute"),
                call(["/plugins/b"], "absolute"),
            ]
        )

    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_per_engine_handle_creation_failure_records_error_result(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
    ):
        """A later per-engine handle failure records an error row and continues."""
        mock_resolve_name.side_effect = lambda eid: f"engine_{eid}"
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(has_kernel_timings=True)
        mock_bm_cls.return_value = _make_bm_mock()
        hipdnn = MagicMock()
        hipdnn.PluginLoadingMode.ABSOLUTE = "absolute"
        hipdnn.Handle.side_effect = [MagicMock(), RuntimeError("bad plugin")]

        with patch.dict("sys.modules", {"hipdnn_frontend": hipdnn}):
            result = run_graph_all_providers(
                graph_path=Path("test.json"),
                graph_json=_make_graph_json(),
                tensor_infos=[_make_tensor_info(1)],
                config=_make_config(
                    engine_filter=[1, 2],
                    plugin_paths=[Path("/plugins/a"), Path("/plugins/b")],
                ),
                handle=None,
            )

        assert [r.status for r in result.results] == ["success", "error"]
        assert result.results[0].plugin_path == "/plugins/a"
        assert result.results[1].plugin_path == "/plugins/b"
        assert "bad plugin" in (result.results[1].error_message or "")
        assert result.results[1].correctness is not None
        assert result.results[1].correctness.execution_success is False


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
            config=_make_config(validation=ValidationConfig(provider="pytorch")),
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

    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner._check_correctness")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_validate_pytorch_adds_timed_reference_row(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_check_corr,
        mock_get_ref,
        mock_resolve_name,
        mock_timed_reference,
    ):
        """--validate pytorch adds a timed reference row and reuses its outputs."""
        mock_resolve_name.return_value = "engine_1"
        ref_outputs = {
            2: ReferenceOutput(data=np.array([1.0], dtype=np.float32), tensor_uid=2)
        }
        ref_provider = MagicMock()
        ref_provider.name = "pytorch"
        mock_get_ref.return_value = ref_provider
        timed_result = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="success",
            role="reference",
            e2e_stats=BenchmarkStats.from_timings([2.0]),
            gpu_kernel_stats=BenchmarkStats.from_timings([1.0]),
        )
        mock_timed_reference.return_value = MagicMock(
            result=timed_result,
            outputs=ref_outputs,
        )
        mock_check_corr.return_value = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-6,
        )
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[1])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=_make_config(validation=ValidationConfig(provider="pytorch")),
            handle=MagicMock(),
        )

        assert [r.role for r in result.results] == ["reference", "engine"]
        assert result.results[0].provider == "pytorch"
        assert result.results[0].status == "success"
        ref_provider.compute_reference.assert_not_called()
        assert mock_check_corr.call_args.args[3] is ref_outputs

    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner._check_correctness")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_validate_pytorch_falls_back_when_timed_reference_skips(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_check_corr,
        mock_get_ref,
        mock_resolve_name,
        mock_timed_reference,
    ):
        """A skipped timed PyTorch row does not prevent CPU reference fallback."""
        mock_resolve_name.return_value = "engine_1"
        ref_outputs = {
            2: ReferenceOutput(data=np.array([1.0], dtype=np.float32), tensor_uid=2)
        }
        ref_provider = MagicMock()
        ref_provider.name = "pytorch"
        ref_provider.compute_reference.return_value = ref_outputs
        mock_get_ref.return_value = ref_provider
        skipped_result = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="skipped",
            role="reference",
            skip_reason="PyTorch GPU not available",
        )
        mock_timed_reference.return_value = MagicMock(
            result=skipped_result,
            outputs=None,
        )
        mock_check_corr.return_value = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-6,
        )
        mock_exec_cls.side_effect = _make_exec_factory(engine_ids=[1])
        mock_bm_cls.return_value = _make_bm_mock()

        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=_make_config(validation=ValidationConfig(provider="pytorch")),
            handle=MagicMock(),
        )

        assert result.results[0].role == "reference"
        assert result.results[0].status == "skipped"
        assert ref_provider.compute_reference.call_count == 1
        assert mock_check_corr.call_args.args[3] is ref_outputs

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    @patch("dnn_benchmarking.execution.pytorch_buffer_manager.PyTorchCudaBufferManager")
    def test_timed_pytorch_reference_uses_auto_timing(
        self,
        mock_buffer_manager_cls,
        mock_pytorch_executor_cls,
    ):
        """PyTorch reference rows let the executor resolve timing from runtime."""
        executor = MagicMock()
        executor.init_time_ms = 0.5
        bench_result = MagicMock()
        bench_result.e2e_timings = [1.0, 2.0]
        bench_result.kernel_timings = None
        bench_result.has_kernel_timings = False
        executor.benchmark.return_value = bench_result
        mock_pytorch_executor_cls.return_value = executor

        buffer_manager = _make_bm_mock()
        buffer_manager.get_tensors.return_value = {}
        buffer_manager.get_output_tensors.return_value = []
        mock_buffer_manager_cls.return_value = buffer_manager

        result = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(
                validation=ValidationConfig(provider="pytorch"),
                metrics=MetricsConfig(tier="off"),
            ),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
        )

        mock_pytorch_executor_cls.assert_called_once()
        assert result.result.e2e_stats is not None
        assert result.result.gpu_kernel_stats is None
        assert result.result.status == "success"


@patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
@patch("dnn_benchmarking.execution.pytorch_buffer_manager.PyTorchCudaBufferManager")
def test_timed_pytorch_reference_attaches_manual_reference_warnings(
    mock_buffer_manager_cls,
    mock_executor_cls,
):
    graph_json = {
        "nodes": [
            {
                "name": "rms_bwd",
                "type": "RMSNormBackwardAttributes",
                "inputs": {
                    "dy_tensor_uid": 1,
                    "x_tensor_uid": 1,
                    "scale_tensor_uid": 2,
                    "inv_rms_tensor_uid": 3,
                },
                "outputs": {"dx_tensor_uid": 4, "dscale_tensor_uid": 2},
            }
        ],
        "tensors": [
            {"uid": 1, "dims": [2, 3, 4]},
            {"uid": 2, "dims": [4]},
            {"uid": 3, "dims": [2, 3, 1]},
            {"uid": 4, "dims": [2, 3, 4]},
        ],
    }
    executor = MagicMock()
    executor.init_time_ms = 1.25
    executor.benchmark.return_value = MagicMock(
        e2e_timings=[2.0],
        kernel_timings=[],
        has_kernel_timings=False,
    )
    mock_executor_cls.return_value = executor

    buffer_manager = MagicMock()
    buffer_manager.__enter__.return_value = buffer_manager
    buffer_manager.__exit__.return_value = False
    buffer_manager.get_tensors.return_value = {}
    buffer_manager.get_output_tensors.return_value = [
        _make_tensor_info(4, is_output=True)
    ]
    buffer_manager.get_output_data.return_value = np.zeros((2, 3, 4), dtype=np.float32)
    mock_buffer_manager_cls.return_value = buffer_manager

    timed = _run_timed_pytorch_row(
        graph_path=Path("rms_bwd.json"),
        graph_json=graph_json,
        graph_name="rms_bwd",
        tensor_infos=[_make_tensor_info(4, is_output=True)],
        config=_make_config(warmup_iters=0, benchmark_iters=1),
        input_data={},
        analytical_flops=None,
        analytical_flops_partial=False,
        analytical_io_bytes=None,
    )

    assert timed.result.status == "success"
    assert timed.result.warnings
    assert "RMSNormBackwardAttributes" in timed.result.warnings[0]
    assert "not solely built-in PyTorch operator time" in timed.result.warnings[0]


class TestCheckCorrectnessOutputCount:
    """_check_correctness returns tolerance_match=False when no outputs are comparable."""

    def test_no_outputs_returns_false(self):
        bm = MagicMock()
        bm.get_output_data.return_value = None

        config = SuiteConfig(validation=ValidationConfig(provider="pytorch"))
        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[],
            graph_json=_make_graph_json(),
            ref_outputs={},
            reference_provider_name="pytorch",
            config=config,
        )

        assert result.tolerance_match is False
        assert result.execution_success is True
        assert "No output tensors to compare" in (result.error_message or "")

    def test_missing_reference_output_returns_false(self):
        bm = MagicMock()
        bm.get_output_data.return_value = np.array([0.0], dtype=np.float32)

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[_make_tensor_info(7, is_output=True)],
            graph_json={
                "nodes": [
                    {
                        "type": "SdpaAttributes",
                        "outputs": {"o_tensor_uid": 7},
                    }
                ]
            },
            ref_outputs={},
            reference_provider_name="pytorch",
            config=SuiteConfig(validation=ValidationConfig(provider="pytorch")),
        )

        assert result.tolerance_match is False
        assert "did not produce output tensor UID 7" in (result.error_message or "")

    def test_zero_bf16_sdpa_forward_output_uses_bfloat16_tolerance(self):
        bm = MagicMock()
        bm.get_output_data.return_value = np.zeros((2,), dtype=np.float32)

        ref_outputs = {
            7: ReferenceOutput(
                data=np.ones((2,), dtype=np.float32),
                tensor_uid=7,
            )
        }

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[
                _make_tensor_info(7, is_output=True, data_type="bfloat16"),
            ],
            graph_json={
                "nodes": [
                    {
                        "type": "SdpaAttributes",
                        "outputs": {"o_tensor_uid": 7},
                    }
                ]
            },
            ref_outputs=ref_outputs,
            reference_provider_name="pytorch",
            config=SuiteConfig(validation=ValidationConfig(provider="pytorch")),
        )

        assert result.tolerance_match is False
        assert result.rtol == pytest.approx(1e-2)
        assert result.atol == pytest.approx(1e-3)

    def test_small_bf16_output_difference_exceeds_absolute_floor(self):
        bm = MagicMock()
        bm.get_output_data.return_value = np.zeros((1,), dtype=np.float32)

        ref_outputs = {
            7: ReferenceOutput(
                data=np.array([5e-3], dtype=np.float32),
                tensor_uid=7,
            )
        }

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[
                _make_tensor_info(7, is_output=True, data_type="bfloat16"),
            ],
            graph_json={
                "nodes": [{"type": "PointwiseAttributes", "outputs": {"y": 7}}]
            },
            ref_outputs=ref_outputs,
            reference_provider_name="pytorch",
            config=SuiteConfig(validation=ValidationConfig(provider="pytorch")),
        )

        assert result.tolerance_match is False
        assert result.atol == pytest.approx(1e-3)

    def test_single_explicit_tolerance_overrides_both_values(self):
        bm = MagicMock()
        bm.get_output_data.return_value = np.array([1.0], dtype=np.float32)

        ref_outputs = {
            7: ReferenceOutput(
                data=np.array([1.1], dtype=np.float32),
                tensor_uid=7,
            )
        }

        result = _check_correctness(
            buffer_manager=bm,
            tensor_infos=[_make_tensor_info(7, is_output=True, data_type="bfloat16")],
            graph_json={
                "nodes": [
                    {
                        "type": "SdpaAttributes",
                        "outputs": {"o_tensor_uid": 7},
                    }
                ]
            },
            ref_outputs=ref_outputs,
            reference_provider_name="pytorch",
            config=SuiteConfig(
                validation=ValidationConfig(provider="pytorch", rtol=0.25)
            ),
        )

        assert result.tolerance_match is True
        assert result.rtol == pytest.approx(0.25)
        assert result.atol == pytest.approx(0.25)


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


class TestProfilingPassInvocation:
    """suite_runner.py:521-542 calls the profiling orchestrator after the
    timed pass when any opt-in metric is requested. The orchestrator's
    payload lands on result.extra_metrics; orchestrator exceptions must
    not bubble out as engine errors."""

    def _setup_mocks(self, mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name):
        mock_resolve_name.side_effect = lambda eid: f"engine_{eid}"
        mock_get_ref.return_value = None
        mock_exec_cls.side_effect = _make_exec_factory(
            engine_ids=[0], has_kernel_timings=True
        )
        mock_bm_cls.return_value = _make_bm_mock()

    @patch("dnn_benchmarking.metrics.profiling_orchestrator.run_profiling_passes")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_orchestrator_called_once_and_payload_lands_in_extra_metrics(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
        mock_orch,
    ):
        self._setup_mocks(mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name)
        payload = {
            "pmc": {"set": "basic", "counters": {"GRBM_GUI_ACTIVE": {"sum": 1.0}}}
        }
        mock_orch.return_value = payload

        config = _make_config(metrics=MetricsConfig(tier="basic", pmc_set="basic"))
        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        # The orchestrator runs exactly once per (graph, engine). Two
        # calls here would catch the duplicate-block bug fixed in
        # commit 196a0fb33ca.
        assert mock_orch.call_count == 1
        assert len(result.results) == 1
        assert result.results[0].extra_metrics == payload

    @patch("dnn_benchmarking.metrics.profiling_orchestrator.run_profiling_passes")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_orchestrator_not_called_when_no_opt_in_flag(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
        mock_orch,
    ):
        self._setup_mocks(mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name)

        # Default MetricsConfig() — basic tier, no opt-in source set.
        config = _make_config(metrics=MetricsConfig())
        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        mock_orch.assert_not_called()
        assert result.results[0].extra_metrics is None

    @patch("dnn_benchmarking.metrics.profiling_orchestrator.run_profiling_passes")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_orchestrator_exception_does_not_fail_engine(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
        mock_orch,
        capsys,
    ):
        """Orchestrator failure (tool missing, parse error, anything) must
        keep the timed pass's status='success' — the headline timing data
        already exists; profiling is best-effort."""
        self._setup_mocks(mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name)
        mock_orch.side_effect = RuntimeError("rocprofv3 missing")

        config = _make_config(metrics=MetricsConfig(tier="basic", pmc_set="basic"))
        result = run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        # Engine still passes; extra_metrics stays None.
        assert result.results[0].status == "success"
        assert result.results[0].extra_metrics is None
        # warn_once writes to stderr.
        captured = capsys.readouterr()
        assert "profiling pass failed" in captured.err
        assert "rocprofv3 missing" in captured.err

    @patch("dnn_benchmarking.metrics.profiling_orchestrator.run_profiling_passes")
    @patch("dnn_benchmarking.execution.suite_runner._resolve_engine_name")
    @patch("dnn_benchmarking.execution.suite_runner._get_reference_provider")
    @patch("dnn_benchmarking.execution.suite_runner.Executor")
    @patch("dnn_benchmarking.execution.suite_runner.BufferManager")
    def test_orchestrator_runs_after_buffermanager_teardown(
        self,
        mock_bm_cls,
        mock_exec_cls,
        mock_get_ref,
        mock_resolve_name,
        mock_orch,
    ):
        """Profiling pass must fire *after* the BufferManager context
        exits — only then are the parent's I/O buffers and the
        executor's workspace freed. Without this ordering, the inner
        profiling subprocess allocates its own VRAM on top of the
        parent's still-pinned tensors, which roughly doubles peak VRAM
        and can OOM on large graphs that fit fine on the headline run.
        """
        self._setup_mocks(mock_exec_cls, mock_bm_cls, mock_get_ref, mock_resolve_name)

        # Track __exit__ vs orchestrator invocation order via shared list.
        order: list[str] = []
        bm_instance = mock_bm_cls.return_value
        original_exit = bm_instance.__exit__

        def tracking_exit(*args, **kwargs):
            order.append("bm_exit")
            return original_exit(*args, **kwargs)

        bm_instance.__exit__ = tracking_exit

        def tracking_orch(**kwargs):
            order.append("orch")
            return {"pmc": {"set": "basic"}}

        mock_orch.side_effect = tracking_orch

        config = _make_config(metrics=MetricsConfig(tier="basic", pmc_set="basic"))
        run_graph_all_providers(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1), _make_tensor_info(2, is_output=True)],
            config=config,
            handle=MagicMock(),
        )

        # Strict ordering: bm context must exit BEFORE the orchestrator
        # runs. Reversing this (the pre-fix state) is the bug.
        assert order == [
            "bm_exit",
            "orch",
        ], f"profiling must run after BufferManager teardown; got {order}"


class TestRunGraphPytorchBackend:
    """run_graph_pytorch_backend emits one provider='pytorch' engine row."""

    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    def test_single_engine_row(self, mock_timed_row):
        row = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="success",
            e2e_stats=BenchmarkStats.from_timings([2.0]),
        )
        mock_timed_row.return_value = MagicMock(result=row, outputs=None)

        result = run_graph_pytorch_backend(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
        )

        assert mock_timed_row.call_args.kwargs["role"] == "engine"
        assert result.engine_ids == [0]
        assert [r.provider for r in result.results] == ["pytorch"]
        assert result.results[0].status == "success"

    @patch("dnn_benchmarking.execution.suite_runner.generate_input_data")
    @patch("dnn_benchmarking.execution.suite_runner._run_timed_pytorch_row")
    def test_unsupported_operations_skip_row(
        self, mock_timed_row, mock_gen, monkeypatch
    ):
        import sys
        import types

        import dnn_benchmarking.execution as execution_pkg

        fake_ops = types.SimpleNamespace(
            get_unsupported_operations=lambda graph_json: ["FooAttributes"]
        )
        monkeypatch.setattr(execution_pkg, "pytorch_ops", fake_ops, raising=False)
        monkeypatch.setitem(
            sys.modules, "dnn_benchmarking.execution.pytorch_ops", fake_ops
        )

        result = run_graph_pytorch_backend(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
        )

        mock_timed_row.assert_not_called()
        # Unsupported graphs must be skipped before any input allocation.
        mock_gen.assert_not_called()
        row = result.results[0]
        assert row.status == "skipped"
        assert "unsupported operations" in (row.skip_reason or "")
        assert result.engine_ids == [0]

    @patch("dnn_benchmarking.execution.suite_runner.generate_input_data")
    def test_input_generation_failure_is_error_row(self, mock_gen):
        mock_gen.side_effect = ValueError("boom")

        result = run_graph_pytorch_backend(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            tensor_infos=[_make_tensor_info(1)],
            config=_make_config(),
        )

        row = result.results[0]
        assert row.status == "error"
        assert "Input data generation failed" in (row.error_message or "")


class TestTimedPytorchRowEngineRole:
    """Engine-role rows report failures as errors, not skips."""

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    @patch("dnn_benchmarking.execution.pytorch_buffer_manager.PyTorchCudaBufferManager")
    def test_engine_role_success_has_no_reference_correctness(
        self,
        mock_buffer_manager_cls,
        mock_pytorch_executor_cls,
    ):
        executor = MagicMock()
        executor.init_time_ms = 0.5
        bench_result = MagicMock()
        bench_result.e2e_timings = [1.0, 2.0]
        bench_result.kernel_timings = None
        bench_result.has_kernel_timings = False
        executor.benchmark.return_value = bench_result
        mock_pytorch_executor_cls.return_value = executor

        buffer_manager = _make_bm_mock()
        buffer_manager.get_tensors.return_value = {}
        buffer_manager.get_output_tensors.return_value = []
        mock_buffer_manager_cls.return_value = buffer_manager

        row = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(metrics=MetricsConfig(tier="off")),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
            role="engine",
        )

        assert row.result.status == "success"
        assert row.result.role == "engine"
        assert row.outputs is None
        # Engine rows never run the extra reference-output extraction pass.
        executor.execute_once.assert_not_called()
        assert row.result.correctness is not None
        assert row.result.correctness.tolerance_match is None
        assert "No reference provider requested" in (
            row.result.correctness.error_message or ""
        )

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    def test_engine_role_failure_is_error(self, mock_pytorch_executor_cls):
        mock_pytorch_executor_cls.side_effect = RuntimeError("no GPU")

        row = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(metrics=MetricsConfig(tier="off")),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
            role="engine",
        )

        assert row.result.status == "error"
        assert "no GPU" in (row.result.error_message or "")

    @patch("dnn_benchmarking.execution.pytorch_executor.PyTorchCudaExecutor")
    def test_reference_role_failure_is_skip(self, mock_pytorch_executor_cls):
        mock_pytorch_executor_cls.side_effect = RuntimeError("no GPU")

        row = _run_timed_pytorch_row(
            graph_path=Path("test.json"),
            graph_json=_make_graph_json(),
            graph_name="test_graph",
            tensor_infos=[],
            config=_make_config(metrics=MetricsConfig(tier="off")),
            input_data={},
            analytical_flops=None,
            analytical_flops_partial=False,
            analytical_io_bytes=None,
            role="reference",
        )

        assert row.result.status == "skipped"
        assert "no GPU" in (row.result.skip_reason or "")
