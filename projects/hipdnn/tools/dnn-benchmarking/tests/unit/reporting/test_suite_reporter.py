# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for Reporter suite-specific methods."""

import io

from dnn_benchmarking.reporting.reporter import Reporter


class TestSuiteReporter:
    """Tests for Reporter suite progress and summary methods."""

    def test_print_suite_header(self) -> None:
        """Test 1: print_suite_header prints banner with total graph count."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_header(5)

        result = output.getvalue()
        assert "=" * Reporter.WIDTH in result
        assert "hipDNN Benchmark Suite: 5 graph(s)" in result

    def test_print_suite_graph_start(self) -> None:
        """Test 2: print_suite_graph_start prints '[1/3] graph_name...' format."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_graph_start(1, 3, "conv_fwd_nchw")

        result = output.getvalue()
        assert "[1/3] conv_fwd_nchw..." in result

    def test_print_suite_graph_start_last_graph(self) -> None:
        """Test 2b: print_suite_graph_start with last graph in sequence."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_graph_start(3, 3, "matmul_fp16")

        result = output.getvalue()
        assert "[3/3] matmul_fp16..." in result

    def test_print_suite_graph_result(self) -> None:
        """Test 3: print_suite_graph_result prints '-> N passed, ...' format."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_graph_result(passed=2, failed=1, skipped=0, errored=0)

        result = output.getvalue()
        assert "  -> 2 passed, 1 failed, 0 skipped, 0 errored" in result

    def test_print_suite_graph_result_all_zeros(self) -> None:
        """Test 3b: print_suite_graph_result with all zeros."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_graph_result(passed=0, failed=0, skipped=0, errored=0)

        result = output.getvalue()
        assert "  -> 0 passed, 0 failed, 0 skipped, 0 errored" in result

    def test_print_suite_graph_error(self) -> None:
        """Test 4: print_suite_graph_error prints inline error for a failed graph."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_graph_error(
            "conv_fwd", "Graph file not found: /path/to/missing.json"
        )

        result = output.getvalue()
        assert "  ERROR: Graph file not found: /path/to/missing.json" in result

    def test_print_suite_summary(self) -> None:
        """Test 5: print_suite_summary prints totals."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_summary(
            total_graphs=3,
            total_combinations=9,
            pass_count=6,
            fail_count=2,
            skip_count=1,
            error_count=0,
        )

        result = output.getvalue()
        assert "Suite Summary:" in result
        assert "Graphs:       3" in result
        assert "Combinations: 9" in result
        assert "Passed:       6" in result
        assert "Failed:       2" in result
        assert "Skipped:      1" in result
        assert "Errors:       0" in result

    def test_print_suite_footer(self) -> None:
        """Test 6: print_suite_footer prints closing banner."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_footer()

        result = output.getvalue()
        assert "=" * Reporter.WIDTH in result

    def test_all_output_goes_to_output_stream(self) -> None:
        """Test 7: All output goes to self._output stream (consistent with Reporter pattern)."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        # Call all suite methods
        reporter.print_suite_header(2)
        reporter.print_suite_graph_start(1, 2, "test_graph")
        reporter.print_suite_graph_result(1, 0, 0, 0)
        reporter.print_suite_graph_error("bad_graph", "Load failed")
        reporter.print_suite_summary(
            total_graphs=2,
            total_combinations=4,
            pass_count=3,
            fail_count=0,
            skip_count=1,
            error_count=0,
        )
        reporter.print_suite_footer()

        result = output.getvalue()
        # All output should be in the StringIO, not stdout
        assert len(result) > 0
        assert "hipDNN Benchmark Suite" in result
        assert "[1/2] test_graph..." in result
        assert "Suite Summary:" in result

    def test_print_suite_header_single_graph(self) -> None:
        """Test header with single graph shows '1 graph(s)'."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_header(1)

        result = output.getvalue()
        assert "hipDNN Benchmark Suite: 1 graph(s)" in result

    def test_print_suite_summary_separator_line(self) -> None:
        """Test summary has separator line before content."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_summary(
            total_graphs=1,
            total_combinations=2,
            pass_count=1,
            fail_count=0,
            skip_count=1,
            error_count=0,
        )

        result = output.getvalue()
        assert "-" * Reporter.WIDTH in result
