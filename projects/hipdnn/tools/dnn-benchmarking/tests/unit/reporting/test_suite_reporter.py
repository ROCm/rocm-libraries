# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for Reporter suite-specific methods."""

import io
from unittest.mock import patch

from dnn_benchmarking.config.benchmark_config import SuiteConfig, ValidationConfig
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.statistics import BenchmarkStats
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    SuiteMetadata,
)


def _make_pe_success(
    engine_id: int = 1,
    correctness: object = None,
    provider: str = "miopen",
) -> ProviderEngineResult:
    """Helper: build a successful ProviderEngineResult with timing."""
    e2e = BenchmarkStats(
        mean_ms=1.234,
        std_ms=0.045,
        min_ms=1.156,
        max_ms=1.456,
        p95_ms=1.312,
        p99_ms=1.398,
    )
    kernel = BenchmarkStats(
        mean_ms=0.500,
        std_ms=0.020,
        min_ms=0.470,
        max_ms=0.540,
        p95_ms=0.520,
        p99_ms=0.535,
    )
    return ProviderEngineResult(
        provider=provider,
        engine_id=engine_id,
        status="success",
        cpu_build_time_ms=45.23,
        e2e_stats=e2e,
        gpu_kernel_stats=kernel,
        correctness=correctness,
    )


class TestSuiteReporter:
    """Tests for Reporter suite progress and summary methods."""

    def test_print_suite_header(self) -> None:
        """print_suite_header prints banner with total graph count."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_header(5)

        result = output.getvalue()
        assert "=" * Reporter.WIDTH in result
        assert "hipDNN Benchmark Suite: 5 graph(s)" in result

    def test_print_suite_graph_start(self) -> None:
        """print_suite_graph_start prints '[1/3] graph_name...' format."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_graph_start(1, 3, "conv_fwd_nchw")

        result = output.getvalue()
        assert "[1/3] conv_fwd_nchw..." in result

    def test_print_suite_graph_start_last_graph(self) -> None:
        """print_suite_graph_start with last graph in sequence."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_graph_start(3, 3, "matmul_fp16")

        result = output.getvalue()
        assert "[3/3] matmul_fp16..." in result

    def test_print_suite_graph_error(self) -> None:
        """print_suite_graph_error prints inline error for a failed graph."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_graph_error(
            "conv_fwd", "Graph file not found: /path/to/missing.json"
        )

        result = output.getvalue()
        assert " ERROR: Graph file not found: /path/to/missing.json" in result

    def test_print_suite_summary(self) -> None:
        """print_suite_summary prints totals from SuiteMetadata."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        meta = SuiteMetadata(
            timestamp="2026-01-01T00:00:00Z",
            hostname="testhost",
            total_graphs=3,
            total_combinations=9,
            pass_combinations=6,
            fail_combinations=2,
            skip_combinations=1,
            error_combinations=0,
        )
        reporter.print_suite_summary(meta)

        result = output.getvalue()
        assert "Suite Summary:" in result
        assert "Graphs:       3" in result
        assert "Combinations: 9" in result
        assert "Passed:       6" in result
        assert "Failed:       2" in result
        assert "Skipped:      1" in result
        assert "Errors:       0" in result

    def test_print_suite_footer(self) -> None:
        """print_suite_footer prints closing banner."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        reporter.print_suite_footer()

        result = output.getvalue()
        assert "=" * Reporter.WIDTH in result

    def test_all_output_goes_to_output_stream(self) -> None:
        """All output goes to self._output stream (consistent with Reporter pattern)."""
        output = io.StringIO()
        reporter = Reporter(output=output)

        # Call all suite methods
        reporter.print_suite_header(2)
        reporter.print_suite_graph_start(1, 2, "test_graph")
        reporter.print_suite_graph_error("bad_graph", "Load failed")
        reporter.print_suite_summary(
            SuiteMetadata(
                timestamp="2026-01-01T00:00:00Z",
                hostname="testhost",
                total_graphs=2,
                total_combinations=4,
                pass_combinations=3,
                fail_combinations=0,
                skip_combinations=1,
                error_combinations=0,
            )
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
            SuiteMetadata(
                timestamp="2026-01-01T00:00:00Z",
                hostname="testhost",
                total_graphs=1,
                total_combinations=2,
                pass_combinations=1,
                fail_combinations=0,
                skip_combinations=1,
                error_combinations=0,
            )
        )

        result = output.getvalue()
        assert "-" * Reporter.WIDTH in result


class TestVerboseReporter:
    """Tests for print_verbose_graph_result (rich per-engine block)."""

    def test_verbose_success_renders_header_init_and_stats(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        gr = GraphResult(
            graph_name="conv1_fwd",
            graph_path="/tmp/conv1_fwd.json",
            results=[_make_pe_success(engine_id=1)],
        )
        reporter.print_verbose_graph_result(
            gr, SuiteConfig(warmup_iters=10, benchmark_iters=100)
        )
        out = output.getvalue()

        assert "hipDNN Benchmark: conv1_fwd" in out
        assert "Engine ID:  1" in out
        assert "Graph build time:" in out
        assert "E2E Execution Statistics:" in out
        assert "Kernel Execution Statistics:" in out
        assert "Mean:" in out

    def test_verbose_renders_block_per_engine(self) -> None:
        """A GraphResult with multiple engines renders one block each."""
        output = io.StringIO()
        reporter = Reporter(output=output)
        gr = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                _make_pe_success(engine_id=0),
                _make_pe_success(engine_id=2),
            ],
        )
        reporter.print_verbose_graph_result(gr, SuiteConfig())
        out = output.getvalue()
        assert "Engine ID:  0" in out
        assert "Engine ID:  2" in out
        # Two distinct rich blocks
        assert out.count("hipDNN Benchmark: g") == 2

    def test_verbose_correctness_pass_renders_passed(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        correctness = CorrectnessResult(
            execution_success=True,
            tolerance_match=True,
            rtol=1e-5,
            atol=1e-8,
            max_abs_diff=1e-6,
            max_rel_diff=1e-6,
        )
        gr = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[_make_pe_success(correctness=correctness)],
        )
        reporter.print_verbose_graph_result(
            gr, SuiteConfig(validation=ValidationConfig(provider="pytorch"))
        )
        out = output.getvalue()
        assert "Reference Validation: PASSED" in out
        assert "pytorch" in out

    def test_verbose_correctness_fail_renders_failed_with_diffs(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        correctness = CorrectnessResult(
            execution_success=True,
            tolerance_match=False,
            rtol=1e-5,
            atol=1e-8,
            max_abs_diff=2.5e-3,
            max_rel_diff=1.7e-2,
        )
        gr = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[_make_pe_success(correctness=correctness)],
        )
        reporter.print_verbose_graph_result(
            gr, SuiteConfig(validation=ValidationConfig(provider="pytorch"))
        )
        out = output.getvalue()
        assert "Reference Validation: FAILED" in out
        assert "Max abs diff:" in out
        assert "Max rel diff:" in out

    def test_verbose_correctness_none_renders_skipped(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        correctness = CorrectnessResult(
            execution_success=True,
            tolerance_match=None,
            rtol=1e-5,
            atol=1e-8,
            error_message="reference provider unavailable",
        )
        gr = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[_make_pe_success(correctness=correctness)],
        )
        reporter.print_verbose_graph_result(
            gr, SuiteConfig(validation=ValidationConfig(provider="pytorch"))
        )
        out = output.getvalue()
        assert "Reference Validation: SKIPPED" in out
        assert "reference provider unavailable" in out

    def test_verbose_error_status_renders_error(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        gr = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                ProviderEngineResult(
                    provider="miopen",
                    engine_id=1,
                    status="error",
                    error_message="boom",
                )
            ],
        )
        reporter.print_verbose_graph_result(gr, SuiteConfig())
        out = output.getvalue()
        assert "ERROR: boom" in out

    def test_verbose_skipped_status_renders_skipped(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        gr = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                ProviderEngineResult(
                    provider="miopen",
                    engine_id=1,
                    status="skipped",
                    skip_reason="not supported",
                )
            ],
        )
        reporter.print_verbose_graph_result(gr, SuiteConfig())
        out = output.getvalue()
        assert "SKIPPED" in out
        assert "not supported" in out

    def test_verbose_header_uses_per_engine_provider_name(self) -> None:
        """Verbose header must render the engine's actual provider, not '(MIOpen)'."""
        output = io.StringIO()
        reporter = Reporter(output=output)
        gr = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                _make_pe_success(engine_id=1, provider="MIOPEN_ENGINE"),
                _make_pe_success(engine_id=2, provider="HIPBLASLT_ENGINE"),
            ],
        )
        reporter.print_verbose_graph_result(gr, SuiteConfig())
        out = output.getvalue()
        assert "Engine ID:  1 (MIOPEN_ENGINE)" in out
        assert "Engine ID:  2 (HIPBLASLT_ENGINE)" in out
        # The legacy literal must not appear anywhere in suite-mode verbose output.
        assert "(MIOpen)" not in out

    def test_verbose_reference_row_uses_reference_header(self) -> None:
        """Timed validation-provider rows must not render as hipDNN engines."""
        output = io.StringIO()
        reporter = Reporter(output=output)
        gr = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                _make_pe_success(engine_id=0, provider="pytorch", correctness=None)
            ],
        )
        gr.results[0].role = "reference"

        reporter.print_verbose_graph_result(
            gr, SuiteConfig(validation=ValidationConfig(provider="pytorch"))
        )
        out = output.getvalue()

        assert "Validation Reference Benchmark: g" in out
        assert "Provider:   pytorch" in out
        assert "Engine ID:" not in out
        assert "Reference: timing baseline (no correctness comparison)" in out

    def test_verbose_reference_row_renders_warnings(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        pe = _make_pe_success(engine_id=0, provider="pytorch", correctness=None)
        pe.role = "reference"
        pe.warnings = [
            "RMSNormBackwardAttributes uses a manual formula; "
            "PyTorch reference timing is not solely built-in PyTorch operator time."
        ]
        gr = GraphResult(graph_name="g", graph_path="/tmp/g.json", results=[pe])

        reporter.print_verbose_graph_result(
            gr, SuiteConfig(validation=ValidationConfig(provider="pytorch"))
        )

        out = output.getvalue()
        assert "Warnings:" in out
        assert "WARNING: RMSNormBackwardAttributes" in out
        assert "Reference: timing baseline" in out

    def test_graph_result_table_renders_warning_column(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        pe = _make_pe_success(engine_id=0, provider="pytorch", correctness=None)
        pe.role = "reference"
        pe.warnings = [
            "manual RMSNorm backward; PyTorch reference timing is not solely "
            "built-in PyTorch operator time."
        ]
        graph = GraphResult(graph_name="g", graph_path="/tmp/g.json", results=[pe])

        reporter.print_graph_result_table(graph)

        out = output.getvalue()
        assert "warnings" in out
        assert "manual RMSNorm backward" in out

    def test_verbose_profiling_renders_when_always_on_metrics_absent(self) -> None:
        """``--metrics-tier off --pmc basic`` leaves every always-on metric
        field unset but still populates ``extra_metrics``. The profiling
        block must render regardless of the always-on suppression — users
        who opted into profiling need to see where their artefacts landed.
        """
        output = io.StringIO()
        reporter = Reporter(output=output)
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            cpu_build_time_ms=12.3,
            e2e_stats=BenchmarkStats(
                mean_ms=1.0,
                std_ms=0.1,
                min_ms=0.9,
                max_ms=1.1,
                p95_ms=1.05,
                p99_ms=1.09,
            ),
            gpu_kernel_stats=BenchmarkStats(
                mean_ms=0.5,
                std_ms=0.05,
                min_ms=0.45,
                max_ms=0.55,
                p95_ms=0.52,
                p99_ms=0.54,
            ),
            extra_metrics={
                "pmc": {
                    "set": "basic",
                    "arch": "gfx942",
                    "counters": {
                        "GRBM_GUI_ACTIVE": {"sum": 999, "mean_per_kernel": 1.0}
                    },
                }
            },
        )
        gr = GraphResult(graph_name="g", graph_path="/tmp/g.json", results=[pe])
        reporter.print_verbose_graph_result(gr, SuiteConfig())
        out = output.getvalue()
        # Always-on block must be suppressed (no Derived Metrics header)…
        assert "Derived Metrics:" not in out
        # …but the profiling block still renders.
        assert "Profiling:" in out
        assert "PMC (basic, gfx942)" in out

    def test_verbose_metrics_render_na_when_no_analytical_model(self) -> None:
        """Ops without an analytical FLOPs model must show N/A, not 0."""
        output = io.StringIO()
        reporter = Reporter(output=output)
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            analytical_flops=None,
            analytical_flops_partial=True,
            analytical_io_bytes=4096,
            gpu_kernel_stats=BenchmarkStats(
                mean_ms=0.5,
                std_ms=0.05,
                min_ms=0.45,
                max_ms=0.55,
                p95_ms=0.52,
                p99_ms=0.54,
            ),
        )
        gr = GraphResult(graph_name="g", graph_path="/tmp/g.json", results=[pe])
        reporter.print_verbose_graph_result(gr, SuiteConfig())
        out = output.getvalue()
        assert "Derived Metrics:" in out
        assert "Analytical FLOPs:     N/A (no analytical model)" in out
        assert "Throughput:           N/A (no analytical model)" in out
        assert "Analytical FLOPs:     0" not in out

    def test_verbose_profiling_surfaces_error_tail_for_each_source(self) -> None:
        """Tool failures in trace/perf/roofline must show in verbose
        output, not only in JSON. Without -o, a silent profiler failure
        is invisible."""
        output = io.StringIO()
        reporter = Reporter(output=output)
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            cpu_build_time_ms=12.3,
            e2e_stats=BenchmarkStats(
                mean_ms=1.0,
                std_ms=0.1,
                min_ms=0.9,
                max_ms=1.1,
                p95_ms=1.05,
                p99_ms=1.09,
            ),
            gpu_kernel_stats=BenchmarkStats(
                mean_ms=0.5,
                std_ms=0.05,
                min_ms=0.45,
                max_ms=0.55,
                p95_ms=0.52,
                p99_ms=0.54,
            ),
            extra_metrics={
                "trace": {"format": "pftrace", "error_tail": "boom", "returncode": 3},
                "perf": {"error_tail": "perf: bad event", "returncode": 2},
                "roofline": {"error_tail": "workload failed", "returncode": 1},
            },
        )
        gr = GraphResult(graph_name="g", graph_path="/tmp/g.json", results=[pe])
        reporter.print_verbose_graph_result(gr, SuiteConfig())
        out = output.getvalue()
        assert "Trace (pftrace):" in out and "rc=3" in out
        assert "CPU (perf):" in out and "rc=2" in out
        assert "Roofline:" in out and "rc=1" in out


class TestPrintHeader:
    """Tests for print_header provider override."""

    def test_print_header_default_is_miopen(self) -> None:
        """When no provider is supplied, the legacy '(MIOpen)' literal is preserved."""
        from pathlib import Path

        from dnn_benchmarking.config.benchmark_config import BenchmarkConfig

        output = io.StringIO()
        reporter = Reporter(output=output)
        cfg = BenchmarkConfig(
            graph_path=Path("/tmp/x.json"),
            warmup_iters=10,
            benchmark_iters=100,
            engine_id=1,
        )
        reporter.print_header(cfg, "graph")
        assert "(MIOpen)" in output.getvalue()

    def test_print_header_uses_provided_provider(self) -> None:
        from pathlib import Path

        from dnn_benchmarking.config.benchmark_config import BenchmarkConfig

        output = io.StringIO()
        reporter = Reporter(output=output)
        cfg = BenchmarkConfig(
            graph_path=Path("/tmp/x.json"),
            warmup_iters=10,
            benchmark_iters=100,
            engine_id=42,
        )
        reporter.print_header(cfg, "graph", provider="HIPBLASLT_ENGINE")
        out = output.getvalue()
        assert "Engine ID:  42 (HIPBLASLT_ENGINE)" in out
        assert "(MIOpen)" not in out

    def test_reference_row_status_renders_reference(self) -> None:
        output = io.StringIO()
        reporter = Reporter(output=output)
        graph = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                ProviderEngineResult(
                    provider="pytorch",
                    engine_id=0,
                    status="success",
                    role="reference",
                    e2e_stats=BenchmarkStats(
                        mean_ms=2.0,
                        median_ms=2.0,
                        std_ms=0.0,
                        min_ms=2.0,
                        max_ms=2.0,
                        p95_ms=2.0,
                        p99_ms=2.0,
                    ),
                )
            ],
        )

        reporter.print_graph_result_table(graph)

        assert "reference" in output.getvalue()


class TestMachineSummaryPlatformLabel:
    """The suite header shows a platform-appropriate accelerator label.

    A CUDA wheel reports cuda_version (and cudnn_version); a ROCm wheel
    reports rocm_version. The header must show only the label that matches
    the running platform — CUDA hosts never print a ROCm line, and ROCm
    hosts never print a CUDA/cuDNN line.
    """

    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    def test_cuda_host_shows_cuda_and_cudnn_not_rocm(self, mock_env) -> None:
        mock_env.return_value = {
            "cpu_model": "Test CPU",
            "gpu_model": "NVIDIA GeForce RTX 5080",
            "rocm_version": None,
            "cuda_version": "13.0",
            "cudnn_version": "9.20.0",
        }
        output = io.StringIO()
        Reporter(output=output).print_suite_header(1)
        out = output.getvalue()

        assert "CUDA:    13.0" in out
        assert "cuDNN:   9.20.0" in out
        assert "ROCm:" not in out

    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    def test_rocm_host_shows_rocm_not_cuda(self, mock_env) -> None:
        mock_env.return_value = {
            "cpu_model": "Test CPU",
            "gpu_model": "AMD Instinct MI300X",
            "rocm_version": "6.2.0",
            "cuda_version": None,
            "cudnn_version": None,
        }
        output = io.StringIO()
        Reporter(output=output).print_suite_header(1)
        out = output.getvalue()

        assert "ROCm:    6.2.0" in out
        assert "CUDA:" not in out
        assert "cuDNN:" not in out
