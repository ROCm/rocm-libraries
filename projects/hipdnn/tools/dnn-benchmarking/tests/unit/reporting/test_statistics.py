# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for BenchmarkStats and BenchmarkResult."""

import json

import pytest

from dnn_benchmarking.reporting import BenchmarkStats
from dnn_benchmarking.reporting.statistics import BenchmarkMetadata, BenchmarkResult


class TestBenchmarkStats:
    """Tests for BenchmarkStats dataclass."""

    def test_from_timings_basic(self) -> None:
        """Test basic stats calculation."""
        timings = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 3.0
        assert stats.min_ms == 1.0
        assert stats.max_ms == 5.0

    def test_from_timings_single_value(self) -> None:
        """Test stats with single value."""
        timings = [5.0]
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 5.0
        assert stats.std_ms == 0.0
        assert stats.min_ms == 5.0
        assert stats.max_ms == 5.0
        assert stats.p95_ms == 5.0
        assert stats.p99_ms == 5.0

    def test_from_timings_empty_raises(self) -> None:
        """Test that empty timings raises ValueError."""
        with pytest.raises(ValueError, match="timings list cannot be empty"):
            BenchmarkStats.from_timings([])

    def test_from_timings_std_calculation(self) -> None:
        """Test standard deviation calculation (sample std, ddof=1)."""
        timings = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 5.0
        # Sample std (ddof=1): sqrt(32/7) ≈ 2.138
        assert stats.std_ms == pytest.approx(2.1381, rel=0.01)

    def test_from_timings_percentiles(self) -> None:
        """Test percentile calculations."""
        # 100 values from 1 to 100
        timings = list(range(1, 101))
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 50.5
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0
        # For 100 uniformly spaced values, p95 should be around 95.05
        assert stats.p95_ms == pytest.approx(95.05, rel=0.01)
        assert stats.p99_ms == pytest.approx(99.01, rel=0.01)

    def test_from_timings_uniform_values(self) -> None:
        """Test with all identical values."""
        timings = [10.0] * 100
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 10.0
        assert stats.std_ms == 0.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 10.0
        assert stats.p95_ms == 10.0
        assert stats.p99_ms == 10.0


class TestBenchmarkMetadata:
    """Tests for BenchmarkMetadata dataclass."""

    def test_default_values(self) -> None:
        """Test metadata has reasonable defaults."""
        metadata = BenchmarkMetadata()
        assert metadata.graph_name == ""
        assert metadata.graph_path == ""
        assert metadata.warmup_iters == 0
        assert metadata.benchmark_iters == 0
        assert metadata.engine_id == 0
        assert metadata.gpu_backend == ""
        # hostname and timestamp are auto-generated
        assert metadata.hostname != ""
        assert metadata.timestamp != ""

    def test_custom_values(self) -> None:
        """Test metadata with custom values."""
        metadata = BenchmarkMetadata(
            graph_name="test_graph",
            graph_path="/path/to/graph.json",
            warmup_iters=10,
            benchmark_iters=100,
            engine_id=1,
            gpu_backend="torch",
        )
        assert metadata.graph_name == "test_graph"
        assert metadata.graph_path == "/path/to/graph.json"
        assert metadata.warmup_iters == 10
        assert metadata.benchmark_iters == 100
        assert metadata.engine_id == 1
        assert metadata.gpu_backend == "torch"


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass and serialization."""

    def test_has_kernel_timings_false_when_none(self) -> None:
        """Test has_kernel_timings is False when timings are None."""
        result = BenchmarkResult(e2e_timings=[1.0, 2.0], kernel_timings=None)
        assert result.has_kernel_timings is False

    def test_has_kernel_timings_false_when_empty(self) -> None:
        """Test has_kernel_timings is False when timings are empty."""
        result = BenchmarkResult(e2e_timings=[1.0, 2.0], kernel_timings=[])
        assert result.has_kernel_timings is False

    def test_has_kernel_timings_true(self) -> None:
        """Test has_kernel_timings is True when timings exist."""
        result = BenchmarkResult(e2e_timings=[1.0, 2.0], kernel_timings=[0.5, 0.6])
        assert result.has_kernel_timings is True

    def test_gpu_backend_from_metadata(self) -> None:
        """Test gpu_backend property reads from metadata."""
        metadata = BenchmarkMetadata(gpu_backend="torch")
        result = BenchmarkResult(
            e2e_timings=[1.0], kernel_timings=[0.5], metadata=metadata
        )
        assert result.gpu_backend == "torch"

    def test_gpu_backend_empty_without_metadata(self) -> None:
        """Test gpu_backend is empty when no metadata."""
        result = BenchmarkResult(e2e_timings=[1.0])
        assert result.gpu_backend == ""

    def test_to_dict_basic(self) -> None:
        """Test to_dict with basic result."""
        result = BenchmarkResult(
            e2e_timings=[1.0, 2.0, 3.0], kernel_timings=[0.5, 0.6, 0.7]
        )
        data = result.to_dict()
        assert data["e2e_timings"] == [1.0, 2.0, 3.0]
        assert data["kernel_timings"] == [0.5, 0.6, 0.7]
        assert "metadata" not in data or data["metadata"] is None

    def test_to_dict_with_metadata(self) -> None:
        """Test to_dict includes metadata."""
        metadata = BenchmarkMetadata(
            graph_name="test", gpu_backend="torch", benchmark_iters=100
        )
        result = BenchmarkResult(
            e2e_timings=[1.0], kernel_timings=[0.5], metadata=metadata
        )
        data = result.to_dict()
        assert "metadata" in data
        assert data["metadata"]["graph_name"] == "test"
        assert data["metadata"]["gpu_backend"] == "torch"
        assert data["metadata"]["benchmark_iters"] == 100

    def test_to_json(self) -> None:
        """Test to_json produces valid JSON."""
        result = BenchmarkResult(e2e_timings=[1.0, 2.0], kernel_timings=[0.5, 0.6])
        json_str = result.to_json()
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["e2e_timings"] == [1.0, 2.0]
        assert parsed["kernel_timings"] == [0.5, 0.6]

    def test_from_dict(self) -> None:
        """Test from_dict creates correct result."""
        data = {
            "e2e_timings": [1.0, 2.0, 3.0],
            "kernel_timings": [0.5, 0.6, 0.7],
        }
        result = BenchmarkResult.from_dict(data)
        assert result.e2e_timings == [1.0, 2.0, 3.0]
        assert result.kernel_timings == [0.5, 0.6, 0.7]
        assert result.metadata is None

    def test_from_dict_with_metadata(self) -> None:
        """Test from_dict with metadata."""
        data = {
            "e2e_timings": [1.0],
            "kernel_timings": None,
            "metadata": {
                "graph_name": "test_graph",
                "graph_path": "/path/graph.json",
                "warmup_iters": 10,
                "benchmark_iters": 100,
                "engine_id": 1,
                "gpu_backend": "torch",
                "hostname": "test-host",
                "timestamp": "2026-01-20T12:00:00",
            },
        }
        result = BenchmarkResult.from_dict(data)
        assert result.metadata is not None
        assert result.metadata.graph_name == "test_graph"
        assert result.metadata.gpu_backend == "torch"

    def test_round_trip_serialization(self, tmp_path) -> None:
        """Test that results survive JSON round-trip."""
        original = BenchmarkResult(
            e2e_timings=[1.0, 2.0, 3.0],
            kernel_timings=[0.5, 0.6, 0.7],
            metadata=BenchmarkMetadata(
                graph_name="test_graph",
                graph_path="/path/to/graph.json",
                warmup_iters=10,
                benchmark_iters=100,
                engine_id=1,
                gpu_backend="torch",
            ),
        )

        path = tmp_path / "results.json"
        original.save_json(str(path))
        loaded = BenchmarkResult.load_json(str(path))

        assert loaded.e2e_timings == original.e2e_timings
        assert loaded.kernel_timings == original.kernel_timings
        assert loaded.metadata is not None
        assert loaded.metadata.graph_name == original.metadata.graph_name
        assert loaded.metadata.gpu_backend == original.metadata.gpu_backend
        assert loaded.metadata.benchmark_iters == original.metadata.benchmark_iters

    def test_round_trip_no_kernel_timings(self, tmp_path) -> None:
        """Test round-trip with no kernel timings."""
        original = BenchmarkResult(e2e_timings=[1.0, 2.0])

        path = tmp_path / "results.json"
        original.save_json(str(path))
        loaded = BenchmarkResult.load_json(str(path))

        assert loaded.e2e_timings == original.e2e_timings
        assert loaded.kernel_timings is None
