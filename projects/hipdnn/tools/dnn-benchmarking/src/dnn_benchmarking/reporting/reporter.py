# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Console output formatting for benchmark results."""

import sys
from pathlib import Path
from typing import Any, List, Optional, TextIO

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

    def print_hipdnn_init_start(self) -> None:
        """Print hipDNN initialization start (no trailing newline)."""
        print("Initializing hipDNN...", end="", flush=True, file=self._output)

    def print_hipdnn_init_done(self) -> None:
        """Print hipDNN initialization completion on the same line."""
        print(" done", flush=True, file=self._output)

    def print_hipdnn_init_newline(self) -> None:
        """End the hipDNN init line (used before printing an error)."""
        print(flush=True, file=self._output)

    def print_running_benchmark(self, total: int) -> None:
        """Print running benchmark status line."""
        self._print(f"Running benchmark on {total} file(s)...")

    def print_suite_header(
        self,
        total_graphs: int,
        tarball_source: Optional[str] = None,
        extra_profiling_runs: int = 0,
    ) -> None:
        """Print suite execution header.

        Includes a one-line machine summary (CPU + GPU + ROCm) collected
        by ``metrics.machine_info`` so console output matches the JSON
        metadata. Failures are silent — ``machine_info`` already routes
        them through ``warn_once``.

        When ``extra_profiling_runs > 0`` the user gets an upfront notice
        of the cost of opt-in profiling so they can size their suite
        accordingly.
        """
        self._print_line("=")
        self._print(f"hipDNN Benchmark Suite: {total_graphs} graph(s)")
        self._print_line("=")
        if tarball_source is not None:
            self._print(f"Source:  {tarball_source} (extracted)")
        self._print_machine_summary()
        if extra_profiling_runs > 0:
            self._print(
                f"Profiling: {extra_profiling_runs} extra workload run(s) "
                "per (graph, engine)"
            )
        self._print("")

    def _print_machine_summary(self) -> None:
        """Print a compact machine identity line if any field is known."""
        try:
            from ..metrics.machine_info import collect_machine_info
            from .suite_results import collect_environment_info
        except ImportError:
            return
        try:
            env = collect_environment_info()
        except Exception:
            return
        cpu = env.get("cpu_model") or "unknown CPU"
        gpu = env.get("gpu_model") or "unknown GPU"
        rocm = env.get("rocm_version") or "unknown ROCm"
        cu = env.get("gpu_compute_units")
        hbm = env.get("gpu_hbm_gb")
        gpu_extras = []
        if cu is not None:
            gpu_extras.append(f"{cu} CUs")
        if hbm is not None:
            gpu_extras.append(f"{hbm:g} GB HBM")
        gpu_label = gpu + (f" ({', '.join(gpu_extras)})" if gpu_extras else "")
        self._print(f"Host:    {cpu}")
        self._print(f"GPU:     {gpu_label}")
        self._print(f"ROCm:    {rocm}")

    def print_suite_graph_start(self, index: int, total: int, graph_name: str) -> None:
        """Print per-graph progress line at start (no trailing newline).

        Format: [1/3] graph_name...
        """
        print(
            f"[{index}/{total}] {graph_name}...", end="", flush=True, file=self._output
        )

    def print_engine_start(self, name: str) -> None:
        """Print the engine name with trailing ellipsis, no newline.

        Format:   miopen_winograd...
        """
        print(f"  {name}...", end="", flush=True, file=self._output)

    def print_engine_result(self, pe: ProviderEngineResult) -> None:
        """Append the outcome to the engine start line, then newline.

        Format:   miopen_winograd...passed
        """
        print(self._pe_outcome(pe), file=self._output)

    def print_suite_graph_error(self, graph_name: str, error: str) -> None:
        """Print inline error on the same line as the graph start, then newline.

        Prints error then continues (caller must not abort).
        """
        self._print(f" ERROR: {error}")

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

        # Suite-end footprint — process RSS and VRAM allocated at the
        # moment the metadata was built. These are flat across the suite
        # (steady-state library + buffer footprint), which is exactly
        # why they belong here and not on every per-engine result.
        footprint_present = any(
            v is not None
            for v in (
                metadata.host_rss_mb,
                metadata.host_ram_available_mb,
                metadata.vram_used_mb,
            )
        )
        if footprint_present:
            self._print("")
            self._print("Suite Footprint:")
            if metadata.host_rss_mb is not None:
                avail = metadata.host_ram_available_mb
                avail_str = (
                    f"  (host avail {self._fmt_mib(avail)})"
                    if avail is not None
                    else ""
                )
                self._print(
                    f"  Host RSS:     {self._fmt_mib(metadata.host_rss_mb)}{avail_str}"
                )
            if metadata.vram_used_mb is not None:
                if metadata.vram_total_mb:
                    self._print(
                        f"  VRAM used:    {self._fmt_mib(metadata.vram_used_mb)}"
                        f" / {self._fmt_mib(metadata.vram_total_mb)}"
                    )
                else:
                    self._print(
                        f"  VRAM used:    {self._fmt_mib(metadata.vram_used_mb)}"
                    )

    def print_suite_footer(self) -> None:
        """Print suite footer."""
        self._print_line("=")

    @staticmethod
    def _pe_outcome(pe: ProviderEngineResult) -> str:
        """Derive a short outcome label for a ProviderEngineResult."""
        if pe.status == "success":
            label = (
                "failed"
                if (
                    pe.correctness is not None
                    and pe.correctness.tolerance_match is False
                )
                else "passed"
            )
            timing = (
                pe.gpu_kernel_stats if pe.gpu_kernel_stats is not None else pe.e2e_stats
            )
            if timing is not None:
                exec_s = timing.total_ms / 1000
                wall_s = pe.elapsed_time_ms / 1000
                return f"{label} (exec {exec_s:.2f}s, elapsed {wall_s:.2f}s)"
            return label
        if pe.status == "skipped":
            return "skipped"
        return "errored"

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
                self._print_pe_metrics(pe)
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

    @staticmethod
    def _fmt_mib(mib: float) -> str:
        """Render a MiB quantity as MiB or GiB depending on magnitude."""
        if mib >= 1024:
            return f"{mib / 1024:.2f} GiB"
        return f"{mib:.1f} MiB"

    def _print_pe_metrics(self, pe: ProviderEngineResult) -> None:
        """Render the always-on metrics block in verbose mode.

        Suppresses the entire section when no metric fields are
        populated (e.g. when the user passed ``--metrics-tier off``)
        so the output stays compact.
        """
        any_present = any(
            v is not None
            for v in (
                pe.workspace_bytes,
                pe.analytical_flops,
                pe.analytical_io_bytes,
                pe.derived_tflops_per_s,
                pe.derived_gbytes_per_s,
                pe.cpu_user_time_per_iter_us,
                pe.cpu_kernel_time_per_iter_us,
                pe.vram_used_mb,
            )
        )
        if not any_present:
            return

        # Pull the unrounded kernel mean used to derive throughput/BW so
        # the printed numbers are reproducible from a single source of
        # truth — without it, the user can't multiply the rounded mean
        # back through the FLOPs total to recover the printed TFLOPs.
        kernel_mean_ms = (
            pe.gpu_kernel_stats.mean_ms if pe.gpu_kernel_stats is not None else None
        )
        derivation_suffix = (
            f"  (kernel mean {kernel_mean_ms:.4f} ms)"
            if kernel_mean_ms is not None
            else ""
        )

        self._print("Derived Metrics:")
        if pe.workspace_bytes is not None:
            self._print(
                f"  Workspace:            {self._fmt_mib(pe.workspace_bytes / 1024 / 1024)}"
            )
        if pe.analytical_flops is not None:
            partial = " (partial)" if pe.analytical_flops_partial else ""
            self._print(f"  Analytical FLOPs:     {pe.analytical_flops:,}{partial}")
        if pe.derived_tflops_per_s is not None:
            self._print(
                f"  Throughput:           {pe.derived_tflops_per_s:.3f} TFLOP/s"
                f"{derivation_suffix}"
            )
        if pe.analytical_io_bytes is not None:
            self._print(
                f"  Analytical I/O:       {self._fmt_mib(pe.analytical_io_bytes / 1024 / 1024)}"
            )
        if pe.derived_gbytes_per_s is not None:
            self._print(
                f"  Bandwidth:            {pe.derived_gbytes_per_s:.2f} GB/s"
                f"{derivation_suffix}"
            )
        if (
            pe.cpu_user_time_per_iter_us is not None
            or pe.cpu_kernel_time_per_iter_us is not None
        ):
            user = (
                pe.cpu_user_time_per_iter_us
                if pe.cpu_user_time_per_iter_us is not None
                else 0.0
            )
            kern = (
                pe.cpu_kernel_time_per_iter_us
                if pe.cpu_kernel_time_per_iter_us is not None
                else 0.0
            )
            self._print(f"  CPU per iter (u/k):   {user:.1f} µs / {kern:.1f} µs")
        if pe.vram_used_mb is not None:
            self._print(f"  VRAM used:            {self._fmt_mib(pe.vram_used_mb)}")
        self._print("")
        self._print_profiling_block(pe)

    def _print_profiling_block(self, pe: ProviderEngineResult) -> None:
        """Render the opt-in profiling artefacts when extra_metrics is set.

        Each line is conditional: a user who runs only --emit-trace
        sees one line, only --perf sees a different one. Full nested
        data is always in the JSON.
        """
        extra = pe.extra_metrics
        if not extra:
            return
        any_present = any(isinstance(extra.get(k), dict) for k in ("trace", "perf"))
        if not any_present:
            return
        self._print("Profiling:")

        trace = extra.get("trace")
        if isinstance(trace, dict):
            path = trace.get("path") or trace.get("db_path")
            fmt = trace.get("format", "?")
            if path:
                self._print(f"  Trace ({fmt}):         {path}")
            elif "skipped" in trace:
                self._print(f"  Trace ({fmt}):         skipped — {trace['skipped']}")

        perf = extra.get("perf")
        if isinstance(perf, dict):
            if "skipped" in perf:
                self._print(f"  CPU (perf):           skipped — {perf['skipped']}")
            else:
                bits = []
                ipc = perf.get("ipc_user")
                if isinstance(ipc, (int, float)):
                    bits.append(f"IPC={ipc:.2f}")
                cu = perf.get("cycles_user")
                if isinstance(cu, (int, float)):
                    bits.append(f"cycles_u={cu:,.0f}")
                iu = perf.get("instructions_user")
                if isinstance(iu, (int, float)):
                    bits.append(f"instr_u={iu:,.0f}")
                tc = perf.get("task_clock_ms")
                if isinstance(tc, (int, float)):
                    bits.append(f"task_clock={tc:.1f}ms")
                if bits:
                    self._print(f"  CPU (perf):           {'  '.join(bits)}")
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

    # CLI progress methods

    def print_extracting(self, source: str) -> None:
        """Print tarball extraction start message."""
        print(f"Extracting {source}...", flush=True, file=self._output)

    def print_extracted_count(self, count: int, source: str) -> None:
        """Print number of graphs extracted from a source."""
        self._print(f"Extracted {count} graph(s) from {source}")

    def print_no_graphs_found(self, pattern: str) -> None:
        """Print message when no graph files match the given pattern."""
        self._print(f"No graph files found matching: {pattern}")

    def print_results_exported(self, path: Path) -> None:
        """Print JSON export confirmation."""
        self._print(f"Results exported to: {path}")

    def print_no_engines_applicable(self) -> None:
        """Print inline note when no engines matched for a graph."""
        print("  no engines applicable", flush=True, file=self._output)

    def print_newline(self) -> None:
        """Print a blank line with flush."""
        print(flush=True, file=self._output)

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
