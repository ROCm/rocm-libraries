# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for suite_results module."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dnn_benchmarking.reporting.statistics import BenchmarkStats
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    StatusCounts,
    SuiteMetadata,
    SuiteResult,
    _format_cudnn_version,
    collect_environment_info,
)


class TestBenchmarkStatsToDict:
    """Tests for BenchmarkStats.to_dict (used by suite serialization)."""

    def test_to_dict(self):
        """BenchmarkStats.to_dict includes all stat fields."""
        stats = BenchmarkStats(
            mean_ms=1.0, std_ms=0.1, min_ms=0.5, max_ms=1.5, p95_ms=1.4, p99_ms=1.49
        )
        d = stats.to_dict()
        assert d == {
            "mean_ms": 1.0,
            "median_ms": 0.0,
            "std_ms": 0.1,
            "min_ms": 0.5,
            "max_ms": 1.5,
            "p95_ms": 1.4,
            "p99_ms": 1.49,
            "total_ms": 0.0,
        }


class TestCorrectnessResult:
    """Tests for CorrectnessResult dataclass."""

    def test_serializes_with_passed_rtol_atol(self):
        """CorrectnessResult serializes with passed (bool), rtol, atol fields."""
        cr = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-8,
            max_abs_diff=1e-7,
            max_rel_diff=1e-6,
        )
        d = cr.to_dict()
        assert d["passed"] is True
        assert d["rtol"] == 1e-5
        assert d["atol"] == 1e-8
        assert d["execution_success"] is True
        assert d["tolerance_match"] is True

    def test_passed_property_true(self):
        """passed is True when execution_success=True and tolerance_match=True."""
        cr = CorrectnessResult(
            execution_success=True, tolerance_match=True, rtol=1e-5, atol=1e-8
        )
        assert cr.passed is True

    def test_passed_property_false_when_no_match(self):
        """passed is False when tolerance_match is False."""
        cr = CorrectnessResult(
            execution_success=True, tolerance_match=False, rtol=1e-5, atol=1e-8
        )
        assert cr.passed is False

    def test_passed_property_false_when_execution_failed(self):
        """passed is False when execution_success is False."""
        cr = CorrectnessResult(
            execution_success=False, tolerance_match=None, rtol=1e-5, atol=1e-8
        )
        assert cr.passed is False

    def test_passed_property_false_when_tolerance_none(self):
        """passed is False when tolerance_match is None (no comparison done)."""
        cr = CorrectnessResult(
            execution_success=True, tolerance_match=None, rtol=1e-5, atol=1e-8
        )
        assert cr.passed is False

    def test_execution_success_separate_from_tolerance_match(self):
        """CorrectnessResult includes execution_success (bool) separate
        from tolerance_match (bool)."""
        cr = CorrectnessResult(
            execution_success=True,
            tolerance_match=False,
            rtol=1e-5,
            atol=1e-8,
        )
        d = cr.to_dict()
        assert "execution_success" in d
        assert "tolerance_match" in d
        assert d["execution_success"] is True
        assert d["tolerance_match"] is False

    def test_error_message_included_when_present(self):
        """CorrectnessResult includes error_message in dict when set."""
        cr = CorrectnessResult(
            execution_success=False,
            tolerance_match=None,
            rtol=1e-5,
            atol=1e-8,
            error_message="No ref available",
        )
        d = cr.to_dict()
        assert d["error_message"] == "No ref available"


class TestProviderEngineResult:
    """Tests for ProviderEngineResult dataclass."""

    def test_success_serializes_with_timing_and_correctness(self):
        """ProviderEngineResult with status='success' serializes with
        cpu_build_time_ms, gpu_kernel_stats, e2e_stats, correctness."""
        stats = BenchmarkStats(
            mean_ms=1.0, std_ms=0.1, min_ms=0.5, max_ms=1.5, p95_ms=1.4, p99_ms=1.49
        )
        corr = CorrectnessResult(
            execution_success=True, tolerance_match=True, rtol=1e-5, atol=1e-8
        )
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            cpu_build_time_ms=10.5,
            gpu_kernel_stats=stats,
            e2e_stats=stats,
            correctness=corr,
        )
        d = pe.to_dict()
        assert d["status"] == "success"
        assert d["cpu_build_time_ms"] == 10.5
        assert "gpu_kernel_stats" in d
        assert "e2e_stats" in d
        assert "correctness" in d
        assert d["gpu_kernel_stats"]["mean_ms"] == 1.0

    def test_success_serializes_plugin_path(self):
        stats = BenchmarkStats(
            mean_ms=1.0,
            std_ms=0.1,
            min_ms=0.5,
            max_ms=1.5,
            p95_ms=1.4,
            p99_ms=1.49,
            median_ms=0.9,
        )
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            plugin_path="/plugins/a",
            cpu_build_time_ms=10.5,
            gpu_kernel_stats=stats,
            e2e_stats=stats,
        )

        d = pe.to_dict()

        assert d["plugin_path"] == "/plugins/a"
        assert "comparison_to_baseline" not in d

    def test_reference_role_serializes_and_is_not_legacy_baseline(self):
        stats = BenchmarkStats(
            mean_ms=1.0,
            std_ms=0.1,
            min_ms=0.5,
            max_ms=1.5,
            p95_ms=1.4,
            p99_ms=1.49,
            median_ms=0.9,
        )
        pe = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="success",
            role="reference",
            e2e_stats=stats,
            gpu_kernel_stats=stats,
        )

        d = pe.to_dict()

        assert d["role"] == "reference"
        assert d["provider"] == "pytorch"
        assert "comparison_to_baseline" not in d

    def test_warnings_serialize_for_reference_timing_rows(self):
        pe = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="success",
            role="reference",
            warnings=[
                "RMSNormBackwardAttributes uses a manual formula; "
                "PyTorch reference timing is not solely built-in PyTorch operator time."
            ],
        )

        d = pe.to_dict()

        assert d["warnings"] == pe.warnings

    def test_error_serializes_without_timing(self):
        """ProviderEngineResult with status='error' serializes with
        status, error_message, no timing data."""
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="error",
            error_message="build failed",
        )
        d = pe.to_dict()
        assert d["status"] == "error"
        assert d["error_message"] == "build failed"
        assert "cpu_build_time_ms" not in d
        assert "gpu_kernel_stats" not in d
        assert "e2e_stats" not in d

    def test_skipped_serializes_with_reason(self):
        """ProviderEngineResult with status='skipped' serializes with
        status, skip_reason."""
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="skipped",
            skip_reason="not supported",
        )
        d = pe.to_dict()
        assert d["status"] == "skipped"
        assert d["skip_reason"] == "not supported"
        assert "cpu_build_time_ms" not in d

    def test_error_status_serializes_correctness(self):
        """C-01: error-status results with a populated correctness still
        emit the correctness block in to_dict()."""
        corr = CorrectnessResult.failed(
            rtol=1e-5, atol=1e-8, error_message="build failed"
        )
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=2,
            status="error",
            error_message="build failed",
            correctness=corr,
        )
        d = pe.to_dict()
        assert d["status"] == "error"
        assert d["error_message"] == "build failed"
        assert "correctness" in d
        assert d["correctness"]["execution_success"] is False
        assert d["correctness"]["tolerance_match"] is None
        assert d["correctness"]["error_message"] == "build failed"

    def test_skipped_status_serializes_correctness(self):
        """C-01: skipped-status results with correctness still emit correctness."""
        corr = CorrectnessResult.failed(
            rtol=1e-5, atol=1e-8, error_message="not supported"
        )
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=3,
            status="skipped",
            skip_reason="not supported",
            correctness=corr,
        )
        d = pe.to_dict()
        assert d["status"] == "skipped"
        assert d["skip_reason"] == "not supported"
        assert "correctness" in d
        assert d["correctness"]["execution_success"] is False


class TestCorrectnessFailed:
    """Tests for CorrectnessResult.failed factory (S-01)."""

    def test_failed_factory_sets_expected_fields(self):
        cr = CorrectnessResult.failed(rtol=1e-3, atol=1e-6, error_message="boom")
        assert cr.execution_success is False
        assert cr.tolerance_match is None
        assert cr.rtol == 1e-3
        assert cr.atol == 1e-6
        assert cr.error_message == "boom"
        assert cr.passed is False


class TestGraphResult:
    """Tests for GraphResult dataclass."""

    def test_contains_graph_name_path_results(self):
        """GraphResult contains graph_name, graph_path, list of
        ProviderEngineResult."""
        pe = ProviderEngineResult(
            provider="miopen", engine_id=0, status="success", cpu_build_time_ms=5.0
        )
        gr = GraphResult(
            graph_name="conv_fwd", graph_path="/path/to/conv.json", results=[pe]
        )
        assert gr.graph_name == "conv_fwd"
        assert gr.graph_path == "/path/to/conv.json"
        assert len(gr.results) == 1

    def test_to_dict_nesting(self):
        """GraphResult.to_dict produces correct nesting."""
        pe = ProviderEngineResult(
            provider="miopen", engine_id=0, status="error", error_message="fail"
        )
        gr = GraphResult(graph_name="conv", graph_path="/p/conv.json", results=[pe])
        d = gr.to_dict()
        assert d["graph_name"] == "conv"
        assert d["graph_path"] == "/p/conv.json"
        assert len(d["results"]) == 1
        assert d["results"][0]["status"] == "error"

    def test_count_by_status_buckets_all_outcomes(self):
        """count_by_status returns the correct bucket counts."""
        pass_corr = CorrectnessResult(
            execution_success=True, tolerance_match=True, rtol=1e-5, atol=1e-8
        )
        fail_corr = CorrectnessResult(
            execution_success=True, tolerance_match=False, rtol=1e-5, atol=1e-8
        )
        none_corr = CorrectnessResult(
            execution_success=True, tolerance_match=None, rtol=1e-5, atol=1e-8
        )
        results = [
            # 2 passes (one with explicit pass, one with no comparison)
            ProviderEngineResult(
                provider="p", engine_id=0, status="success", correctness=pass_corr
            ),
            ProviderEngineResult(
                provider="p", engine_id=1, status="success", correctness=none_corr
            ),
            # 1 fail
            ProviderEngineResult(
                provider="p", engine_id=2, status="success", correctness=fail_corr
            ),
            # 1 skipped
            ProviderEngineResult(
                provider="p", engine_id=3, status="skipped", skip_reason="x"
            ),
            # 1 errored
            ProviderEngineResult(
                provider="p", engine_id=4, status="error", error_message="y"
            ),
        ]
        gr = GraphResult(graph_name="g", graph_path="/p.json", results=results)
        counts = gr.count_by_status()
        assert counts.passed == 2
        assert counts.failed == 1
        assert counts.skipped == 1
        assert counts.errored == 1

    def test_count_by_status_success_without_correctness_counts_as_passed(self):
        """A success with correctness=None counts as passed."""
        pe = ProviderEngineResult(
            provider="p", engine_id=0, status="success", correctness=None
        )
        gr = GraphResult(graph_name="g", graph_path="/p.json", results=[pe])
        counts = gr.count_by_status()
        assert counts == StatusCounts(passed=1, failed=0, skipped=0, errored=0)

    def test_count_by_status_excludes_reference_rows(self):
        """Timed reference rows are reported but not counted as engine passes."""
        reference = ProviderEngineResult(
            provider="pytorch",
            engine_id=0,
            status="success",
            role="reference",
        )
        engine = ProviderEngineResult(provider="miopen", engine_id=1, status="success")
        gr = GraphResult(
            graph_name="g", graph_path="/p.json", results=[reference, engine]
        )

        counts = gr.count_by_status()

        assert counts == StatusCounts(passed=1, failed=0, skipped=0, errored=0)

    def test_count_by_status_empty_results(self):
        """An empty results list yields all-zero counts."""
        gr = GraphResult(graph_name="g", graph_path="/p.json", results=[])
        counts = gr.count_by_status()
        assert counts == StatusCounts(passed=0, failed=0, skipped=0, errored=0)


class TestSuiteResult:
    """Tests for SuiteResult and SuiteMetadata dataclasses."""

    def _make_suite_result(self):
        """Create a minimal SuiteResult for testing."""
        meta = SuiteMetadata(
            timestamp="2026-01-01T00:00:00Z",
            hostname="testhost",
            total_graphs=2,
            total_combinations=2,
            pass_combinations=1,
            fail_combinations=1,
            skip_combinations=0,
            error_combinations=0,
            rocm_version="6.0",
            gpu_model="MI300X",
            gpu_arch="gfx942",
            python_version="3.12.3",
            hipdnn_version="0.1.0",
        )
        stats = BenchmarkStats(
            mean_ms=1.0, std_ms=0.1, min_ms=0.5, max_ms=1.5, p95_ms=1.4, p99_ms=1.49
        )
        corr = CorrectnessResult(
            execution_success=True, tolerance_match=True, rtol=1e-5, atol=1e-8
        )
        pe1 = ProviderEngineResult(
            provider="miopen",
            engine_id=0,
            status="success",
            cpu_build_time_ms=5.0,
            gpu_kernel_stats=stats,
            e2e_stats=stats,
            correctness=corr,
        )
        pe2 = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="error",
            error_message="fail",
        )
        gr1 = GraphResult(graph_name="conv", graph_path="/p/conv.json", results=[pe1])
        gr2 = GraphResult(graph_name="relu", graph_path="/p/relu.json", results=[pe2])
        return SuiteResult(metadata=meta, graphs=[gr1, gr2])

    def test_metadata_includes_environment_fields(self):
        """SuiteResult contains metadata with timestamp, hostname,
        total_graphs, pass_count, fail_count, rocm_version, gpu_model,
        python_version, hipdnn_version."""
        sr = self._make_suite_result()
        meta_d = sr.metadata.to_dict()
        assert meta_d["timestamp"] == "2026-01-01T00:00:00Z"
        assert meta_d["hostname"] == "testhost"
        assert meta_d["total_graphs"] == 2
        assert meta_d["total_combinations"] == 2
        assert meta_d["pass_combinations"] == 1
        assert meta_d["fail_combinations"] == 1
        assert meta_d["skip_combinations"] == 0
        assert meta_d["error_combinations"] == 0
        assert meta_d["rocm_version"] == "6.0"
        assert meta_d["gpu_model"] == "MI300X"
        assert meta_d["gpu_arch"] == "gfx942"
        assert meta_d["python_version"] == "3.12.3"
        assert meta_d["hipdnn_version"] == "0.1.0"

    def test_to_dict_graph_first_nesting(self):
        """SuiteResult.to_dict() produces graph-first nesting: top-level
        'graphs' array, each with 'results' array."""
        sr = self._make_suite_result()
        d = sr.to_dict()
        assert "metadata" in d
        assert "graphs" in d
        assert isinstance(d["graphs"], list)
        assert len(d["graphs"]) == 2
        for g in d["graphs"]:
            assert "graph_name" in g
            assert "results" in g
            assert isinstance(g["results"], list)

    def test_save_json_writes_valid_file(self):
        """SuiteResult.save_json(path) writes valid JSON to file."""
        sr = self._make_suite_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        sr.save_json(path)

        loaded = json.loads(Path(path).read_text())
        assert "metadata" in loaded
        assert "graphs" in loaded
        assert len(loaded["graphs"]) == 2

        # Cleanup
        Path(path).unlink()

    def test_timing_stats_include_all_fields(self):
        """SuiteResult.to_dict() timing stats include mean, std, min, max, p95, p99."""
        sr = self._make_suite_result()
        d = sr.to_dict()
        # Get the first successful result
        first_graph = d["graphs"][0]
        first_result = first_graph["results"][0]
        gpu_stats = first_result["gpu_kernel_stats"]
        e2e_stats = first_result["e2e_stats"]

        for stats in [gpu_stats, e2e_stats]:
            assert "mean_ms" in stats
            assert "std_ms" in stats
            assert "min_ms" in stats
            assert "max_ms" in stats
            assert "p95_ms" in stats
            assert "p99_ms" in stats


class TestCollectEnvironmentInfo:
    """Tests for collect_environment_info helper."""

    def test_returns_python_version(self):
        """collect_environment_info always includes python_version."""
        info = collect_environment_info()
        assert "python_version" in info
        assert info["python_version"] is not None
        # Should be x.y.z format
        parts = info["python_version"].split(".")
        assert len(parts) == 3

    def test_includes_gpu_arch_from_detect_arch(self, monkeypatch):
        """gpu_arch is sourced from metrics.arch.detect_arch so the
        JSON output and the rocprof_pmc PMC keying agree on the
        gfx target. Patch the binding in suite_results (where the name
        is now bound at import time), not in arch_mod — patching the
        source module after the name has been imported wouldn't take."""
        from dnn_benchmarking.reporting import suite_results as sr_mod

        monkeypatch.setattr(sr_mod, "detect_arch", lambda: "gfx942")
        info = collect_environment_info()
        assert info["gpu_arch"] == "gfx942"

    def test_includes_cuda_and_cudnn_keys(self):
        """collect_environment_info always exposes the CUDA version keys.

        The values are platform-dependent (populated only on a CUDA
        host), but the keys must always be present so the JSON schema is
        stable across ROCm and CUDA runs."""
        info = collect_environment_info()
        assert "cuda_version" in info
        assert "cudnn_version" in info


class TestFormatCudnnVersion:
    """Tests for the packed-int cuDNN version decoder."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (92000, "9.20.0"),
            (90000, "9.0.0"),
            (91300, "9.13.0"),
            (90201, "9.2.1"),
            (8907, "8.9.7"),  # pre-9 packing scheme
        ],
    )
    def test_decodes_packed_int(self, raw, expected):
        assert _format_cudnn_version(raw) == expected

    @pytest.mark.parametrize("raw", [None, 0])
    def test_missing_version_returns_none(self, raw):
        assert _format_cudnn_version(raw) is None
