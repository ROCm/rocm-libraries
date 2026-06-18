# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Console output formatting for benchmark results."""

import sys
from pathlib import Path
from typing import List, Optional, TextIO

from ..config.benchmark_config import BenchmarkConfig, SuiteConfig
from .statistics import BenchmarkStats
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

    def print_reference_header(
        self, config: BenchmarkConfig, graph_name: str, provider: str
    ) -> None:
        """Print a timed validation-provider reference row header."""
        self._print_line("=")
        self._print(f"Validation Reference Benchmark: {graph_name}")
        self._print_line("=")
        self._print(f"Graph:      {config.graph_path}")
        self._print(f"Provider:   {provider}")
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

    def _print_stats_block(self, stats: BenchmarkStats) -> None:
        """Print a statistics block (helper for print_stats).

        Args:
            stats: Benchmark statistics.
        """
        self._print(f"  Mean:                 {stats.mean_ms:.3f} ms")
        self._print(f"  Median:               {stats.median_ms:.3f} ms")
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

    def print_warning(self, message: str) -> None:
        """Print non-fatal warning message.

        Args:
            message: Warning message.
        """
        self._print(f"WARNING: {message}")

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
        cuda = env.get("cuda_version")
        cudnn = env.get("cudnn_version")
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
        # A CUDA wheel reports cuda_version; a ROCm wheel does not. Show the
        # platform-appropriate label so CUDA hosts never print a ROCm line
        # (and vice versa).
        if cuda is not None:
            self._print(f"CUDA:    {cuda}")
            if cudnn is not None:
                self._print(f"cuDNN:   {cudnn}")
        else:
            rocm = env.get("rocm_version") or "unknown ROCm"
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
        if pe.role == "reference" and pe.status == "success":
            label = "reference"
            timing = (
                pe.gpu_kernel_stats if pe.gpu_kernel_stats is not None else pe.e2e_stats
            )
            if timing is not None:
                exec_s = timing.total_ms / 1000
                wall_s = pe.elapsed_time_ms / 1000
                return f"{label} (exec {exec_s:.2f}s, elapsed {wall_s:.2f}s)"
            return label
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

    def print_graph_result_table(self, graph_result: GraphResult) -> None:
        """Render one compact summary row per engine for a graph."""
        if not graph_result.results:
            return

        include_plugin = any(pe.plugin_path for pe in graph_result.results)
        headers = ["engine", "status"]
        if include_plugin:
            headers.append("plugin_path")
        headers.extend(
            [
                "kernel_mean_ms",
                "kernel_median_ms",
                "e2e_mean_ms",
                "e2e_median_ms",
            ]
        )
        include_warnings = any(pe.warnings for pe in graph_result.results)
        if include_warnings:
            headers.append("warnings")
        rows: List[List[str]] = []
        for pe in graph_result.results:
            row = [pe.provider, self._pe_status(pe)]
            if include_plugin:
                row.append(pe.plugin_path or "")
            row.extend(
                [
                    self._fmt_stat(pe.gpu_kernel_stats, "mean_ms"),
                    self._fmt_stat(pe.gpu_kernel_stats, "median_ms"),
                    self._fmt_stat(pe.e2e_stats, "mean_ms"),
                    self._fmt_stat(pe.e2e_stats, "median_ms"),
                ]
            )
            if include_warnings:
                row.append(self._fmt_warnings(pe.warnings))
            rows.append(row)

        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows))
            for i in range(len(headers))
        ]
        self._print("Results:")
        self._print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        self._print("  " + "  ".join("-" * width for width in widths))
        for row in rows:
            self._print(
                "  " + "  ".join(row[i].ljust(widths[i]) for i in range(len(row)))
            )
        self._print("")

    @staticmethod
    def _pe_status(pe: ProviderEngineResult) -> str:
        if pe.role == "reference" and pe.status == "success":
            return "reference"
        if pe.status != "success":
            return pe.status
        if pe.correctness is not None and pe.correctness.tolerance_match is False:
            return "failed"
        return "passed"

    @staticmethod
    def _fmt_warnings(warnings: Optional[List[str]]) -> str:
        if not warnings:
            return ""
        if len(warnings) == 1:
            return warnings[0]
        return f"{warnings[0]} [{len(warnings) - 1} more, see JSON]"

    @staticmethod
    def _fmt_stat(stats: Optional[BenchmarkStats], name: str) -> str:
        if stats is None:
            return "n/a"
        value = getattr(stats, name)
        return f"{value:.3f}"

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
            if pe.role == "reference":
                self.print_reference_header(
                    cfg_view, graph_result.graph_name, pe.provider
                )
            else:
                self.print_header(
                    cfg_view, graph_result.graph_name, provider=pe.provider
                )

            if pe.cpu_build_time_ms is not None:
                self.print_init_time(pe.cpu_build_time_ms)

            if pe.status == "success":
                self._print_pe_stats(pe)
                self._print_pe_metrics(pe)
                # Profiling artefacts render independently of the always-on
                # metrics block — opt-in profiling is valid under
                # --metrics-tier off, and the user should still see where
                # their artefacts landed plus any tool-failure detail.
                self._print_profiling_block(pe)
                self._print_pe_warnings(pe)
                if pe.role == "reference":
                    self._print(
                        "Reference: timing baseline (no correctness comparison)"
                    )
                    self._print("")
                elif pe.correctness is not None:
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

    def _print_pe_warnings(self, pe: ProviderEngineResult) -> None:
        if not pe.warnings:
            return
        self._print("Warnings:")
        for warning in pe.warnings:
            self._print(f"  WARNING: {warning}")
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
        elif pe.analytical_flops_partial:
            # No node could be modelled analytically — show N/A rather than a
            # misleading 0 so users know throughput is unavailable, not zero.
            self._print("  Analytical FLOPs:     N/A (no analytical model)")
            self._print("  Throughput:           N/A (no analytical model)")
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

    def _print_profiling_block(self, pe: ProviderEngineResult) -> None:
        """Render the opt-in profiling artefacts when extra_metrics is set.

        Each line is conditional: a user who runs only --pmc sees one
        line, only --emit-trace sees a different one, and so on. Long
        counter lists fold to ``[N more, see JSON]`` so the console
        block stays compact — full nested data is always in the JSON.
        """
        extra = pe.extra_metrics
        if not extra:
            return
        any_present = any(
            isinstance(extra.get(k), dict) for k in ("trace", "pmc", "perf", "roofline")
        )
        if not any_present:
            return
        self._print("Profiling:")

        trace = extra.get("trace")
        if isinstance(trace, dict):
            # Trace and rocpd db are distinct artefacts — kineto in
            # particular emits both: `path` is the chrome JSON the user
            # opens in Perfetto/Chrome, `db_path` is the rocpd source.
            # Folding them onto one line with a `fmt` suffix would
            # mislabel a database as a trace whenever convert failed and
            # only `db_path` was present.
            fmt = trace.get("format", "?")
            trace_path = trace.get("path")
            db_path = trace.get("db_path")
            if trace_path:
                self._print(f"  Trace ({fmt}):         {trace_path}")
                # Both .pftrace and chrome JSON open in Perfetto.
                self._print("    → drag onto https://ui.perfetto.dev/")
            if db_path:
                self._print(f"  Trace DB:              {db_path}")
                # When the kineto convert failed, the db is all we have;
                # surface the convert command so the user can rerun.
                if not trace_path:
                    self._print(
                        "    → python -m rocpd convert -i <db> "
                        "--output-format chrome -o trace.json"
                    )
            if not trace_path and not db_path and "skipped" in trace:
                self._print(f"  Trace ({fmt}):         skipped — {trace['skipped']}")
            if "error_tail" in trace:
                self._print(
                    f"  Trace ({fmt}):         rocprofv3 errored "
                    f"(rc={trace.get('returncode', '?')})"
                )

        pmc = extra.get("pmc")
        if isinstance(pmc, dict):
            arch = pmc.get("arch", "?")
            pmc_set = pmc.get("set", "?")
            counters = pmc.get("counters") or {}
            db_path = pmc.get("db_path")
            if counters:
                head = list(counters.items())[:3]
                rendered = "  ".join(
                    f"{name}={int(v.get('sum', 0)):,}" for name, v in head
                )
                more = len(counters) - len(head)
                suffix = f"  [{more} more, see JSON]" if more > 0 else ""
                self._print(f"  PMC ({pmc_set}, {arch}):  {rendered}{suffix}")
            elif "skipped" in pmc:
                self._print(f"  PMC ({pmc_set}, {arch}):  skipped — {pmc['skipped']}")
            elif "error_tail" in pmc:
                self._print(
                    f"  PMC ({pmc_set}, {arch}):  rocprofv3 errored "
                    f"(rc={pmc.get('returncode', '?')})"
                )
            # Surface the rocpd db path + analyze hint whenever we
            # captured one, regardless of whether parsing succeeded.
            # The db itself is the full source of truth; aggregates are
            # the convenience layer.
            if db_path:
                self._print(f"  PMC db:               {db_path}")
                self._print(
                    "    → rocprof-compute analyze --path "
                    f"{Path(db_path).parent} "
                    "(see docs/troubleshooting.md for venv setup)"
                )

        perf = extra.get("perf")
        if isinstance(perf, dict):
            if "skipped" in perf:
                self._print(f"  CPU (perf):           skipped — {perf['skipped']}")
            elif "error_tail" in perf:
                self._print(
                    f"  CPU (perf):           errored "
                    f"(rc={perf.get('returncode', '?')})"
                )
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

        roofline = extra.get("roofline")
        if isinstance(roofline, dict):
            # rocprof-compute profile mode emits CSVs (no PDF); the
            # rendered roofline (ASCII / GUI / TUI) comes from a
            # separate `rocprof-compute analyze` pass against
            # `workload_path`. analyze has its own Python deps that
            # would downgrade torch's numpy if installed into the
            # dnn-benchmarking venv — see docs/troubleshooting.md for
            # the separate-venv recipe.
            workload = roofline.get("workload_path")
            csv = roofline.get("roofline_csv")
            if csv:
                self._print(f"  Roofline CSV:         {csv}")
            if workload:
                self._print(
                    f"    → rocprof-compute analyze --path {workload} "
                    "--block 4  (ASCII roofline)"
                )
                self._print(
                    "    → add --gui for interactive web UI "
                    "(see docs/troubleshooting.md for analyze venv setup)"
                )
            if "skipped" in roofline:
                self._print(f"  Roofline:             skipped — {roofline['skipped']}")
            if "error_tail" in roofline:
                self._print(
                    f"  Roofline:             rocprof-compute errored "
                    f"(rc={roofline.get('returncode', '?')})"
                )
        self._print("")

    def _print_pe_correctness(
        self, correctness: CorrectnessResult, suite_config: SuiteConfig
    ) -> None:
        """Print correctness block from a CorrectnessResult."""
        if correctness.tolerance_match is None:
            reason = correctness.error_message or "no reference comparison performed"
            self._print(f"Reference Validation: SKIPPED ({reason})")
            self._print(f"  Provider: {suite_config.validation.provider.value}")
            self._print("")
            return

        status = "PASSED" if correctness.tolerance_match else "FAILED"
        self._print(f"Reference Validation: {status}")
        self._print(f"  Provider: {suite_config.validation.provider.value}")
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
