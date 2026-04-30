# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Console output formatting for benchmark results."""

import sys
from pathlib import Path
from typing import Any, Optional, TextIO

from ..config.benchmark_config import ABTestConfig, BenchmarkConfig, SuiteConfig
from .statistics import BenchmarkStats, CombinedBenchmarkStats
from .suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    SuiteMetadata,
)


class Reporter:
    """Formats and prints benchmark results to console.

    Handles all console output including:
    - Configuration header
    - Initialization timing
    - Execution statistics
    - Validation results
    """

    WIDTH = 80

    def __init__(self, output: TextIO = sys.stdout) -> None:
        """Initialize reporter with output stream.

        Args:
            output: Output stream (default: stdout).
        """
        self._output = output

    def print_header(
        self,
        config: BenchmarkConfig,
        graph_name: str,
        provider: Optional[str] = None,
    ) -> None:
        """Print benchmark configuration header.

        Args:
            config: Benchmark configuration.
            graph_name: Name of the graph being benchmarked.
            provider: Optional engine display name. When set, replaces the
                legacy "(MIOpen)" literal so suite-mode verbose output can
                show the actual engine for each result.
        """
        engine_label = provider if provider else "MIOpen"
        self._print_line("=")
        self._print(f"hipDNN Benchmark: {graph_name}")
        self._print_line("=")
        self._print(f"Graph:      {config.graph_path}")
        self._print(f"Engine ID:  {config.engine_id} ({engine_label})")
        self._print(f"Warmup:     {config.warmup_iters} iterations")
        self._print(f"Benchmark:  {config.benchmark_iters} iterations")
        self._print_line("-")
        self._print("")

    def print_pytorch_header(
        self, config: BenchmarkConfig, graph_name: str, device: str
    ) -> None:
        """Print PyTorch CUDA benchmark configuration header.

        Args:
            config: Benchmark configuration.
            graph_name: Name of the graph being benchmarked.
            device: CUDA device being used.
        """
        self._print_line("=")
        self._print(f"PyTorch CUDA Benchmark: {graph_name}")
        self._print_line("=")
        self._print(f"Graph:      {config.graph_path}")
        self._print(f"Device:     {device}")
        self._print(f"Warmup:     {config.warmup_iters} iterations")
        self._print(f"Benchmark:  {config.benchmark_iters} iterations")
        self._print_line("-")
        self._print("")

    def print_init_time(self, init_time_ms: float) -> None:
        """Print initialization timing.

        Args:
            init_time_ms: Graph initialization time in milliseconds.
        """
        self._print("Initialization:")
        self._print(f"  Graph build time:     {init_time_ms:.2f} ms")
        self._print("")

    def print_stats(self, stats: BenchmarkStats) -> None:
        """Print execution statistics.

        Args:
            stats: Benchmark statistics.
        """
        self._print("Execution Statistics:")
        self._print_stats_block(stats)
        self._print("")

    def print_combined_stats(self, stats: CombinedBenchmarkStats) -> None:
        """Print combined E2E and kernel execution statistics.

        Args:
            stats: Combined benchmark statistics.
        """
        self._print("E2E Execution Statistics:")
        self._print_stats_block(stats.e2e_stats)
        self._print("")

        if stats.kernel_stats:
            self._print("Kernel Execution Statistics:")
            self._print_stats_block(stats.kernel_stats)
        else:
            self._print("Kernel Timing: Not available (PyTorch GPU not available)")
        self._print("")

    def _print_stats_block(self, stats: BenchmarkStats) -> None:
        """Print a statistics block (helper for print_stats/print_combined_stats).

        Args:
            stats: Benchmark statistics.
        """
        self._print(f"  Mean:                 {stats.mean_ms:.3f} ms")
        self._print(f"  Std Dev:              {stats.std_ms:.3f} ms")
        self._print(f"  Min:                  {stats.min_ms:.3f} ms")
        self._print(f"  Max:                  {stats.max_ms:.3f} ms")
        self._print(f"  P95:                  {stats.p95_ms:.3f} ms")
        self._print(f"  P99:                  {stats.p99_ms:.3f} ms")

    def print_validation(self, passed: bool, message: str) -> None:
        """Print validation result.

        Args:
            passed: Whether validation passed.
            message: Validation message.
        """
        status = "PASSED" if passed else "FAILED"
        if "skipped" in message.lower() or "stubbed" in message.lower():
            status = "SKIPPED"

        self._print(f"Validation: {status} ({message})")

    def print_footer(self) -> None:
        """Print benchmark footer."""
        self._print_line("=")

    def print_error(self, message: str) -> None:
        """Print error message.

        Args:
            message: Error message.
        """
        self._print(f"ERROR: {message}")

    def _print(self, text: str) -> None:
        """Print a line of text.

        Args:
            text: Text to print.
        """
        print(text, file=self._output)

    def _print_line(self, char: str) -> None:
        """Print a horizontal line.

        Args:
            char: Character to use for the line.
        """
        print(char * self.WIDTH, file=self._output)

    # A/B Testing Methods

    def print_ab_header(
        self, config: BenchmarkConfig, ab_config: ABTestConfig, graph_name: str
    ) -> None:
        """Print A/B test configuration header.

        Args:
            config: Benchmark configuration.
            ab_config: A/B test configuration.
            graph_name: Name of the graph being benchmarked.
        """
        self._print_line("=")
        self._print(f"hipDNN A/B Test: {graph_name}")
        self._print_line("=")
        self._print(f"Graph:      {config.graph_path}")
        self._print(f"Warmup:     {config.warmup_iters} iterations")
        self._print(f"Benchmark:  {config.benchmark_iters} iterations")
        self._print_line("-")
        self._print("Configuration A:")
        if ab_config.a_path:
            self._print(f"  Plugin Path: {ab_config.a_path}")
        else:
            self._print("  Plugin Path: (default)")
        self._print(f"  Engine ID:   {ab_config.a_id}")
        self._print("Configuration B:")
        if ab_config.b_path:
            self._print(f"  Plugin Path: {ab_config.b_path}")
        else:
            self._print("  Plugin Path: (default)")
        self._print(f"  Engine ID:   {ab_config.b_id}")
        self._print_line("-")
        self._print("")

    def print_ab_stats(
        self,
        stats_a: BenchmarkStats,
        stats_b: BenchmarkStats,
        init_time_a_ms: float,
        init_time_b_ms: float,
    ) -> None:
        """Print side-by-side comparison of A vs B statistics.

        Args:
            stats_a: Statistics for configuration A.
            stats_b: Statistics for configuration B.
            init_time_a_ms: Init time for A in milliseconds.
            init_time_b_ms: Init time for B in milliseconds.
        """
        # Header
        self._print(f"{'':20} {'A':>15} {'B':>15}")
        self._print_line("-")

        # Init times
        self._print(
            f"{'Init Time:':20} {init_time_a_ms:>12.2f} ms {init_time_b_ms:>12.2f} ms"
        )

        # Execution stats
        self._print(
            f"{'Mean:':20} {stats_a.mean_ms:>12.3f} ms {stats_b.mean_ms:>12.3f} ms"
        )
        self._print(
            f"{'Std Dev:':20} {stats_a.std_ms:>12.3f} ms {stats_b.std_ms:>12.3f} ms"
        )
        self._print(
            f"{'Min:':20} {stats_a.min_ms:>12.3f} ms {stats_b.min_ms:>12.3f} ms"
        )
        self._print(
            f"{'Max:':20} {stats_a.max_ms:>12.3f} ms {stats_b.max_ms:>12.3f} ms"
        )
        self._print(
            f"{'P95:':20} {stats_a.p95_ms:>12.3f} ms {stats_b.p95_ms:>12.3f} ms"
        )
        self._print(
            f"{'P99:':20} {stats_a.p99_ms:>12.3f} ms {stats_b.p99_ms:>12.3f} ms"
        )
        self._print_line("-")

        # Calculate speedup
        if stats_a.mean_ms > 0 and stats_b.mean_ms > 0:
            if stats_a.mean_ms > stats_b.mean_ms:
                speedup = (stats_a.mean_ms - stats_b.mean_ms) / stats_a.mean_ms * 100
                self._print(f"Speedup:            B is {speedup:.1f}% faster")
            elif stats_b.mean_ms > stats_a.mean_ms:
                speedup = (stats_b.mean_ms - stats_a.mean_ms) / stats_b.mean_ms * 100
                self._print(f"Speedup:            A is {speedup:.1f}% faster")
            else:
                self._print("Speedup:            A and B are equal")

        self._print("")

    def print_ab_combined_stats(
        self,
        stats_a: CombinedBenchmarkStats,
        stats_b: CombinedBenchmarkStats,
        init_time_a_ms: float,
        init_time_b_ms: float,
    ) -> None:
        """Print side-by-side comparison of A vs B with both E2E and kernel stats.

        Args:
            stats_a: Combined statistics for configuration A.
            stats_b: Combined statistics for configuration B.
            init_time_a_ms: Init time for A in milliseconds.
            init_time_b_ms: Init time for B in milliseconds.
        """
        # E2E Stats section
        self._print("E2E Execution Statistics:")
        self._print(f"{'':20} {'A':>15} {'B':>15}")
        self._print_line("-")

        # Init times
        self._print(
            f"{'Init Time:':20} {init_time_a_ms:>12.2f} ms {init_time_b_ms:>12.2f} ms"
        )

        # E2E execution stats
        self._print_ab_stats_block(stats_a.e2e_stats, stats_b.e2e_stats)
        self._print("")

        # Kernel Stats section (if available)
        if stats_a.kernel_stats and stats_b.kernel_stats:
            self._print("Kernel Execution Statistics:")
            self._print(f"{'':20} {'A':>15} {'B':>15}")
            self._print_line("-")
            self._print_ab_stats_block(stats_a.kernel_stats, stats_b.kernel_stats)
            self._print("")

            # Calculate kernel speedup
            ka, kb = stats_a.kernel_stats, stats_b.kernel_stats
            if ka.mean_ms > 0 and kb.mean_ms > 0:
                if ka.mean_ms > kb.mean_ms:
                    speedup = (ka.mean_ms - kb.mean_ms) / ka.mean_ms * 100
                    self._print(f"Kernel Speedup:     B is {speedup:.1f}% faster")
                elif kb.mean_ms > ka.mean_ms:
                    speedup = (kb.mean_ms - ka.mean_ms) / kb.mean_ms * 100
                    self._print(f"Kernel Speedup:     A is {speedup:.1f}% faster")
                else:
                    self._print("Kernel Speedup:     A and B are equal")
                self._print("")
        else:
            self._print("Kernel Timing: Not available")
            self._print("")

    def _print_ab_stats_block(
        self, stats_a: BenchmarkStats, stats_b: BenchmarkStats
    ) -> None:
        """Print a side-by-side statistics block for A/B comparison.

        Args:
            stats_a: Statistics for configuration A.
            stats_b: Statistics for configuration B.
        """
        self._print(
            f"{'Mean:':20} {stats_a.mean_ms:>12.3f} ms {stats_b.mean_ms:>12.3f} ms"
        )
        self._print(
            f"{'Std Dev:':20} {stats_a.std_ms:>12.3f} ms {stats_b.std_ms:>12.3f} ms"
        )
        self._print(
            f"{'Min:':20} {stats_a.min_ms:>12.3f} ms {stats_b.min_ms:>12.3f} ms"
        )
        self._print(
            f"{'Max:':20} {stats_a.max_ms:>12.3f} ms {stats_b.max_ms:>12.3f} ms"
        )
        self._print(
            f"{'P95:':20} {stats_a.p95_ms:>12.3f} ms {stats_b.p95_ms:>12.3f} ms"
        )
        self._print(
            f"{'P99:':20} {stats_a.p99_ms:>12.3f} ms {stats_b.p99_ms:>12.3f} ms"
        )

    def print_ab_comparison(
        self,
        passed: bool,
        max_abs_diff: float,
        max_rel_diff: float,
        rtol: float,
        atol: float,
    ) -> None:
        """Print A/B accuracy comparison result.

        Args:
            passed: Whether comparison passed.
            max_abs_diff: Maximum absolute difference.
            max_rel_diff: Maximum relative difference.
            rtol: Relative tolerance used.
            atol: Absolute tolerance used.
        """
        status = "PASSED" if passed else "FAILED"
        self._print(f"Accuracy Comparison: {status}")
        self._print(f"  (rtol={rtol:.0e}, atol={atol:.0e})")
        if not passed:
            self._print(f"  Max abs diff: {max_abs_diff:.2e}")
            self._print(f"  Max rel diff: {max_rel_diff:.2e}")

    def print_ab_validation(
        self,
        validation_a: Optional[Any],
        validation_b: Optional[Any],
        rtol: float,
        atol: float,
    ) -> None:
        """Print reference validation results for A/B test.

        Args:
            validation_a: ValidationResult for configuration A, or None.
            validation_b: ValidationResult for configuration B, or None.
            rtol: Relative tolerance used.
            atol: Absolute tolerance used.
        """
        if validation_a is None and validation_b is None:
            return

        self._print("")
        self._print("Reference Validation:")

        if validation_a is not None:
            status_a = "PASSED" if validation_a.passed else "FAILED"
            self._print(f"  Config A vs {validation_a.provider_name}: {status_a}")
            if not validation_a.passed:
                self._print(f"    Max abs diff: {validation_a.max_abs_diff:.2e}")
                self._print(f"    Max rel diff: {validation_a.max_rel_diff:.2e}")

        if validation_b is not None:
            status_b = "PASSED" if validation_b.passed else "FAILED"
            self._print(f"  Config B vs {validation_b.provider_name}: {status_b}")
            if not validation_b.passed:
                self._print(f"    Max abs diff: {validation_b.max_abs_diff:.2e}")
                self._print(f"    Max rel diff: {validation_b.max_rel_diff:.2e}")

        self._print(f"  (rtol={rtol:.0e}, atol={atol:.0e})")

    # Reference Validation Methods

    # Suite Methods

    def print_suite_header(self, total_graphs: int) -> None:
        """Print suite execution header."""
        self._print_line("=")
        self._print(f"hipDNN Benchmark Suite: {total_graphs} graph(s)")
        self._print_line("=")
        self._print("")

    def print_suite_graph_start(self, index: int, total: int, graph_name: str) -> None:
        """Print per-graph progress line at start.

        Format: [1/3] graph_name...
        """
        self._print(f"[{index}/{total}] {graph_name}...")

    def print_suite_graph_result(
        self, passed: int, failed: int, skipped: int, errored: int
    ) -> None:
        """Print per-graph result summary line.

        Format:   -> 2 passed, 1 failed, 0 skipped, 0 errored
        """
        self._print(
            f"  -> {passed} passed, {failed} failed, "
            f"{skipped} skipped, {errored} errored"
        )

    def print_suite_graph_error(self, graph_name: str, error: str) -> None:
        """Print inline error when a graph fails to load/execute.

        Prints error then continues (caller must not abort).
        """
        self._print(f"  ERROR: {error}")

    def print_suite_summary(self, metadata: SuiteMetadata) -> None:
        """Print suite execution summary from suite metadata.

        Args:
            metadata: SuiteMetadata containing graph and combination totals.
        """
        self._print("")
        self._print_line("-")
        self._print("Suite Summary:")
        self._print(f"  Graphs:       {metadata.total_graphs}")
        self._print(f"  Combinations: {metadata.total_combinations}")
        self._print(f"  Passed:       {metadata.pass_combinations}")
        self._print(f"  Failed:       {metadata.fail_combinations}")
        self._print(f"  Skipped:      {metadata.skip_combinations}")
        self._print(f"  Errors:       {metadata.error_combinations}")

    def print_suite_footer(self) -> None:
        """Print suite footer."""
        self._print_line("=")

    def print_verbose_graph_result(
        self, graph_result: GraphResult, suite_config: SuiteConfig
    ) -> None:
        """Render a graph's per-engine results in the rich single-graph format.

        For each ProviderEngineResult, prints a header + init time + execution
        statistics + correctness block, matching the legacy run_benchmark output.
        Used in verbose mode when the unified runner processes a graph.
        """
        for pe in graph_result.results:
            cfg_view = BenchmarkConfig(
                graph_path=Path(graph_result.graph_path),
                warmup_iters=suite_config.warmup_iters,
                benchmark_iters=suite_config.benchmark_iters,
                engine_id=pe.engine_id,
            )
            self.print_header(cfg_view, graph_result.graph_name, provider=pe.provider)

            if pe.cpu_build_time_ms is not None:
                self.print_init_time(pe.cpu_build_time_ms)

            if pe.status == "success":
                self._print_pe_stats(pe)
                if pe.correctness is not None:
                    self._print_pe_correctness(pe.correctness, suite_config)
            elif pe.status == "skipped":
                self._print(f"Status: SKIPPED ({pe.skip_reason or 'no reason given'})")
                self._print("")
            else:  # error
                self.print_error(pe.error_message or "execution failed")
                self._print("")

            self.print_footer()
            self._print("")

    def _print_pe_stats(self, pe: ProviderEngineResult) -> None:
        """Print E2E + kernel stats from a ProviderEngineResult."""
        if pe.e2e_stats is not None:
            self._print("E2E Execution Statistics:")
            self._print_stats_block(pe.e2e_stats)
            self._print("")
        if pe.gpu_kernel_stats is not None:
            self._print("Kernel Execution Statistics:")
            self._print_stats_block(pe.gpu_kernel_stats)
            self._print("")
        elif pe.e2e_stats is not None:
            self._print("Kernel Timing: Not available")
            self._print("")

    def _print_pe_correctness(
        self, correctness: CorrectnessResult, suite_config: SuiteConfig
    ) -> None:
        """Print correctness block from a CorrectnessResult."""
        if correctness.tolerance_match is None:
            reason = correctness.error_message or "no reference comparison performed"
            self._print(f"Reference Validation: SKIPPED ({reason})")
            self._print(f"  Provider: {suite_config.reference_provider}")
            self._print("")
            return

        status = "PASSED" if correctness.tolerance_match else "FAILED"
        self._print(f"Reference Validation: {status}")
        self._print(f"  Provider: {suite_config.reference_provider}")
        self._print(f"  (rtol={correctness.rtol:.0e}, atol={correctness.atol:.0e})")
        if not correctness.tolerance_match:
            if correctness.max_abs_diff is not None:
                self._print(f"  Max abs diff: {correctness.max_abs_diff:.2e}")
            if correctness.max_rel_diff is not None:
                self._print(f"  Max rel diff: {correctness.max_rel_diff:.2e}")
        self._print("")

    def print_reference_validation(
        self,
        provider_name: str,
        passed: bool,
        max_abs_diff: float,
        max_rel_diff: float,
        rtol: float,
        atol: float,
    ) -> None:
        """Print reference validation result.

        Args:
            provider_name: Name of the reference provider used.
            passed: Whether validation passed.
            max_abs_diff: Maximum absolute difference.
            max_rel_diff: Maximum relative difference.
            rtol: Relative tolerance used.
            atol: Absolute tolerance used.
        """
        status = "PASSED" if passed else "FAILED"
        self._print(f"Reference Validation: {status}")
        self._print(f"  Provider: {provider_name}")
        self._print(f"  (rtol={rtol:.0e}, atol={atol:.0e})")
        if not passed:
            self._print(f"  Max abs diff: {max_abs_diff:.2e}")
            self._print(f"  Max rel diff: {max_rel_diff:.2e}")
