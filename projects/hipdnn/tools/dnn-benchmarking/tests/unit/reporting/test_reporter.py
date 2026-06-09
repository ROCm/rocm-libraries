# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for Reporter."""

import io
from pathlib import Path

from dnn_benchmarking.config import BenchmarkConfig
from dnn_benchmarking.reporting import BenchmarkStats, Reporter
from dnn_benchmarking.reporting.suite_results import GraphResult, ProviderEngineResult


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


class TestReporterEngineTable:
    """Tests for the compact per-engine result table."""

    def test_print_graph_result_table_without_comparison_columns(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        graph = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                ProviderEngineResult(
                    provider="engine_1",
                    engine_id=1,
                    status="success",
                    gpu_kernel_stats=BenchmarkStats(
                        mean_ms=1.0,
                        median_ms=0.9,
                        std_ms=0.1,
                        min_ms=0.8,
                        max_ms=1.2,
                        p95_ms=1.1,
                        p99_ms=1.2,
                    ),
                    e2e_stats=BenchmarkStats(
                        mean_ms=2.0,
                        median_ms=1.8,
                        std_ms=0.2,
                        min_ms=1.6,
                        max_ms=2.4,
                        p95_ms=2.2,
                        p99_ms=2.4,
                    ),
                )
            ],
        )

        reporter.print_graph_result_table(graph)

        result = output.getvalue()
        assert "kernel_mean_ms" in result
        assert "kernel_median_ms" in result
        assert "e2e_mean_ms" in result
        assert "e2e_median_ms" in result
        assert "delta_pct" not in result

    def test_print_graph_result_table_with_plugin_path_column(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        graph = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                ProviderEngineResult(
                    provider="engine_2",
                    engine_id=2,
                    status="success",
                    plugin_path="/plugins/b",
                    gpu_kernel_stats=BenchmarkStats(
                        mean_ms=1.0,
                        median_ms=0.9,
                        std_ms=0.1,
                        min_ms=0.8,
                        max_ms=1.2,
                        p95_ms=1.1,
                        p99_ms=1.2,
                    ),
                    e2e_stats=BenchmarkStats(
                        mean_ms=2.0,
                        median_ms=1.8,
                        std_ms=0.2,
                        min_ms=1.6,
                        max_ms=2.4,
                        p95_ms=2.2,
                        p99_ms=2.4,
                    ),
                )
            ],
        )

        reporter.print_graph_result_table(graph)

        result = output.getvalue()
        assert "plugin_path" in result
        assert "/plugins/b" in result
        assert "delta_pct" not in result
        assert "%" not in result
