# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite benchmark CLI runner."""

import argparse
from pathlib import Path
from typing import Any, List, Optional

from ..common.exceptions import ExecutionError, GraphLoadError
from ..config.benchmark_config import MetricsConfig, SuiteConfig
from ..execution.suite_runner import run_graph_all_providers
from ..graph.loader import GraphLoader
from ..reporting.reporter import Reporter
from ..reporting.suite_results import (
    GraphResult,
    ProviderEngineResult,
    SuiteResult,
)
from ..validation.reference_provider import ReferenceProviderRegistry


def _error_graph_result(graph_path: Path, error_message: str) -> GraphResult:
    """Build a GraphResult representing a graph-level setup failure."""
    return GraphResult(
        graph_name=graph_path.stem,
        graph_path=str(graph_path),
        results=[
            ProviderEngineResult(
                provider="unknown",
                engine_id=0,
                status="error",
                error_message=error_message,
            )
        ],
    )


def _run_one_graph(
    graph_path: Path, config: SuiteConfig, handle: Any, reporter: Reporter
) -> GraphResult:
    """Load and run a single graph. Returns a GraphResult (errors included)."""
    try:
        loader = GraphLoader()
        graph_json = loader.load_json(graph_path)
        loader.validate(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)
        result = run_graph_all_providers(
            graph_path, graph_json, tensor_infos, config, handle, reporter=reporter
        )
        if len(result.results) == 0:
            return _error_graph_result(
                graph_path, "No provider/engine combinations matched filters"
            )
        return result
    except (GraphLoadError, ExecutionError) as e:
        return _error_graph_result(graph_path, str(e))


def run_suite_benchmark(
    graph_paths: List[Path],
    config: SuiteConfig,
    output_path: Optional[Path],
    plugin_path: Optional[Path],
    reporter: Reporter,
    tarball_source: Optional[str] = None,
) -> int:
    """Run the benchmark suite and return an exit code.

    Args:
        graph_paths: List of resolved graph file paths to benchmark.
        config: Suite configuration.
        output_path: Optional path to export results as JSON.
        plugin_path: Optional path to plugin .so directory.
        reporter: Reporter instance for console output.
        tarball_source: Optional tarball source path for display.

    Returns:
        Exit code (0 for success, 1 for error, 2 for correctness failure).
    """
    total = len(graph_paths)

    if config.reference_provider != "none":
        try:
            ref = ReferenceProviderRegistry.get_provider(config.reference_provider)
        except ValueError:
            reporter.print_error(
                f"Reference provider '{config.reference_provider}' is not registered."
            )
            return 1
        if not ref.is_available():
            reporter.print_error(
                f"Reference provider '{config.reference_provider}' is not available "
                "(check that its dependencies are installed)."
            )
            return 1

    reporter.print_suite_header(
        total,
        tarball_source=tarball_source,
        extra_profiling_runs=config.metrics.extra_runs_per_engine,
    )

    reporter.print_hipdnn_init_start()
    try:
        import hipdnn_frontend as hipdnn

        if plugin_path is not None:
            hipdnn.set_engine_plugin_paths([str(plugin_path)])
        handle = hipdnn.Handle()
    except ImportError:
        reporter.print_hipdnn_init_newline()
        reporter.print_error(
            "hipdnn_frontend not available. Install hipDNN Python bindings first."
        )
        return 1
    except RuntimeError as e:
        reporter.print_hipdnn_init_newline()
        reporter.print_error(f"Failed to create hipDNN handle: {e}")
        return 1

    reporter.print_hipdnn_init_done()
    reporter.print_running_benchmark(total)

    graph_results: List[GraphResult] = []
    for i, graph_path in enumerate(graph_paths, start=1):
        reporter.print_suite_graph_start(i, total, graph_path.stem)
        reporter.print_newline()
        gr = _run_one_graph(graph_path, config, handle, reporter)
        if gr.is_no_engine_graph():
            reporter.print_no_engines_applicable()
        if config.verbose:
            reporter.print_verbose_graph_result(gr, config)
        graph_results.append(gr)

    suite_result = SuiteResult.from_graph_results(graph_results, total_graphs=total)

    reporter.print_suite_summary(suite_result.metadata)
    reporter.print_suite_footer()

    if output_path is not None:
        suite_result.save_json(str(output_path))

    if suite_result.metadata.fail_combinations > 0:
        return 2
    if suite_result.metadata.error_combinations > 0:
        return 1
    return 0


def run_suite_cli(
    args: argparse.Namespace,
    graph_paths: List[Path],
    reporter: Reporter,
    tarball_source: Optional[str] = None,
) -> int:
    """Validate suite CLI args, build config, and delegate to run_suite_benchmark."""
    try:
        metrics_config = MetricsConfig(
            tier=getattr(args, "metrics_tier", "basic"),
            emit_trace=getattr(args, "emit_trace", None),
            perf=getattr(args, "perf", False),
            profiling_output_dir=getattr(args, "profiling_output_dir", None),
        )
        config = SuiteConfig(
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            seed=args.seed,
            engine_filter=args.engine,
            rtol=args.rtol,
            atol=args.atol,
            gpu_backend="auto",
            reference_provider=args.validate,
            verbose=args.verbose,
            metrics=metrics_config,
            plugin_path=args.plugin_path,
        )
    except ValueError as e:
        reporter.print_error(f"Suite configuration error: {e}")
        return 1

    return run_suite_benchmark(
        graph_paths=graph_paths,
        config=config,
        output_path=args.output,
        plugin_path=args.plugin_path,
        reporter=reporter,
        tarball_source=tarball_source,
    )
