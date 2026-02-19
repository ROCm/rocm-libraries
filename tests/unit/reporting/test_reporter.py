# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for Reporter."""

import io
from pathlib import Path

from dnn_benchmarking.config import ABTestConfig, BenchmarkConfig
from dnn_benchmarking.reporting import BenchmarkStats, Reporter


class TestReporter:
    """Tests for Reporter class."""

    def test_print_header(self) -> None:
        """Test header output format."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            warmup_iters=10,
            benchmark_iters=100,
            engine_id=1,
        )

        reporter.print_header(config, "test_conv_fwd")

        result = output.getvalue()
        assert "hipDNN Benchmark: test_conv_fwd" in result
        assert "/test/graph.json" in result
        assert "Engine ID:  1" in result
        assert "Warmup:     10 iterations" in result
        assert "Benchmark:  100 iterations" in result

    def test_print_init_time(self) -> None:
        """Test initialization time output."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_init_time(45.23)

        result = output.getvalue()
        assert "Initialization:" in result
        assert "45.23 ms" in result

    def test_print_stats(self) -> None:
        """Test statistics output format."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        stats = BenchmarkStats(
            mean_ms=1.234,
            std_ms=0.045,
            min_ms=1.156,
            max_ms=1.456,
            p95_ms=1.312,
            p99_ms=1.398,
        )

        reporter.print_stats(stats)

        result = output.getvalue()
        assert "Execution Statistics:" in result
        assert "Mean:" in result
        assert "1.234" in result
        assert "Std Dev:" in result
        assert "0.045" in result
        assert "P95:" in result
        assert "P99:" in result

    def test_print_validation_passed(self) -> None:
        """Test validation passed output."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_validation(True, "rtol=1e-5, atol=1e-8")

        result = output.getvalue()
        assert "PASSED" in result

    def test_print_validation_failed(self) -> None:
        """Test validation failed output."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_validation(False, "max diff exceeded")

        result = output.getvalue()
        assert "FAILED" in result

    def test_print_validation_skipped(self) -> None:
        """Test validation skipped output."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_validation(True, "Validation skipped - no reference")

        result = output.getvalue()
        assert "SKIPPED" in result

    def test_print_validation_stubbed(self) -> None:
        """Test validation stubbed output."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_validation(
            True, "Validation stubbed - CPU reference not available"
        )

        result = output.getvalue()
        assert "SKIPPED" in result

    def test_print_error(self) -> None:
        """Test error message output."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_error("Something went wrong")

        result = output.getvalue()
        assert "ERROR: Something went wrong" in result


class TestReporterAB:
    """Tests for Reporter A/B testing methods."""

    def test_print_ab_header(self) -> None:
        """Test A/B header output format."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            warmup_iters=10,
            benchmark_iters=100,
        )
        ab_config = ABTestConfig(
            a_path=Path("/path/to/pluginA"),
            a_id=1,
            b_path=Path("/path/to/pluginB"),
            b_id=2,
        )

        reporter.print_ab_header(config, ab_config, "test_conv_fwd")

        result = output.getvalue()
        assert "hipDNN A/B Test: test_conv_fwd" in result
        assert "/test/graph.json" in result
        assert "Configuration A:" in result
        assert "Configuration B:" in result
        assert "/path/to/pluginA" in result
        assert "/path/to/pluginB" in result
        assert "Engine ID:   1" in result
        assert "Engine ID:   2" in result

    def test_print_ab_header_default_paths(self) -> None:
        """Test A/B header with default plugin paths."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
        )
        ab_config = ABTestConfig(a_id=1, b_id=2)

        reporter.print_ab_header(config, ab_config, "test_conv_fwd")

        result = output.getvalue()
        assert "(default)" in result

    def test_print_ab_stats(self) -> None:
        """Test A/B statistics output format."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        stats_a = BenchmarkStats(
            mean_ms=1.234,
            std_ms=0.045,
            min_ms=1.156,
            max_ms=1.456,
            p95_ms=1.312,
            p99_ms=1.398,
        )
        stats_b = BenchmarkStats(
            mean_ms=1.100,
            std_ms=0.035,
            min_ms=1.050,
            max_ms=1.200,
            p95_ms=1.180,
            p99_ms=1.195,
        )

        reporter.print_ab_stats(stats_a, stats_b, 45.0, 42.0)

        result = output.getvalue()
        assert "A" in result
        assert "B" in result
        assert "Init Time:" in result
        assert "Mean:" in result
        assert "Speedup:" in result

    def test_print_ab_stats_speedup_b_faster(self) -> None:
        """Test A/B stats shows B is faster."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        stats_a = BenchmarkStats(
            mean_ms=2.0,
            std_ms=0.1,
            min_ms=1.9,
            max_ms=2.1,
            p95_ms=2.0,
            p99_ms=2.1,
        )
        stats_b = BenchmarkStats(
            mean_ms=1.0,
            std_ms=0.1,
            min_ms=0.9,
            max_ms=1.1,
            p95_ms=1.0,
            p99_ms=1.1,
        )

        reporter.print_ab_stats(stats_a, stats_b, 45.0, 42.0)

        result = output.getvalue()
        assert "B is" in result
        assert "faster" in result

    def test_print_ab_stats_speedup_a_faster(self) -> None:
        """Test A/B stats shows A is faster."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        stats_a = BenchmarkStats(
            mean_ms=1.0,
            std_ms=0.1,
            min_ms=0.9,
            max_ms=1.1,
            p95_ms=1.0,
            p99_ms=1.1,
        )
        stats_b = BenchmarkStats(
            mean_ms=2.0,
            std_ms=0.1,
            min_ms=1.9,
            max_ms=2.1,
            p95_ms=2.0,
            p99_ms=2.1,
        )

        reporter.print_ab_stats(stats_a, stats_b, 45.0, 42.0)

        result = output.getvalue()
        assert "A is" in result
        assert "faster" in result

    def test_print_ab_comparison_passed(self) -> None:
        """Test A/B comparison passed output."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_ab_comparison(True, 1e-7, 1e-6, 1e-5, 1e-8)

        result = output.getvalue()
        assert "PASSED" in result
        assert "rtol=" in result
        assert "atol=" in result

    def test_print_ab_comparison_failed(self) -> None:
        """Test A/B comparison failed output."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_ab_comparison(False, 0.1, 0.05, 1e-5, 1e-8)

        result = output.getvalue()
        assert "FAILED" in result
        assert "Max abs diff:" in result
        assert "Max rel diff:" in result
