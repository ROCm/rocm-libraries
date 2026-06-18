# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite benchmark CLI dispatch and shared suite loop."""

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..common.exceptions import ExecutionError, GraphLoadError
from ..config.benchmark_config import (
    ExecutionBackendName,
    MetricsConfig,
    ReferenceProviderName,
    SuiteConfig,
    ValidationConfig,
)
from ..graph.loader import GraphLoader
from ..reporting.reporter import Reporter
from ..reporting.suite_results import (
    GraphResult,
    ProviderEngineResult,
    SuiteResult,
)
from ..validation.reference_provider import ReferenceProviderRegistry

LoadedGraphRunner = Callable[
    [Path, Dict[str, Any], list, SuiteConfig, Reporter], GraphResult
]


def _plugin_paths_from_environment() -> Optional[List[Path]]:
    rocm_path = os.environ.get("ROCM_PATH")
    if not rocm_path:
        return None
    return [Path(rocm_path) / "lib" / "hipdnn_plugins" / "engines"]


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
    graph_path: Path,
    config: SuiteConfig,
    reporter: Reporter,
    run_loaded_graph: LoadedGraphRunner,
) -> GraphResult:
    """Load and run a single graph. Returns a GraphResult (errors included)."""
    try:
        loader = GraphLoader()
        graph_json = loader.load_json(graph_path)
        loader.validate(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)
        result = run_loaded_graph(
            graph_path, graph_json, tensor_infos, config, reporter
        )
        if len(result.results) == 0:
            return _error_graph_result(
                graph_path, "No provider/engine combinations matched filters"
            )
        return result
    except (GraphLoadError, ExecutionError) as e:
        return _error_graph_result(graph_path, str(e))


def _run_suite_graphs_after_startup(
    graph_paths: List[Path],
    config: SuiteConfig,
    output_path: Optional[Path],
    reporter: Reporter,
    run_loaded_graph: LoadedGraphRunner,
) -> int:
    """Run loaded-graph callback for each graph and emit suite output."""
    total = len(graph_paths)
    reporter.print_running_benchmark(total)

    graph_results: List[GraphResult] = []
    for i, graph_path in enumerate(graph_paths, start=1):
        reporter.print_suite_graph_start(i, total, graph_path.stem)
        reporter.print_newline()
        gr = _run_one_graph(graph_path, config, reporter, run_loaded_graph)
        if gr.is_no_engine_graph():
            reporter.print_no_engines_applicable()
        if config.verbose:
            reporter.print_verbose_graph_result(gr, config)
        else:
            reporter.print_graph_result_table(gr)
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


def _reference_provider_available(config: SuiteConfig, reporter: Reporter) -> bool:
    if not config.validation.enabled:
        return True
    provider_name = config.validation.provider.value
    try:
        ref = ReferenceProviderRegistry.get_provider(provider_name)
    except ValueError:
        reporter.print_error(f"Reference provider '{provider_name}' is not registered.")
        return False
    if not ref.is_available():
        reporter.print_error(
            f"Reference provider '{provider_name}' is not available "
            "(check that its dependencies are installed)."
        )
        return False
    return True


def run_suite_benchmark(
    graph_paths: List[Path],
    config: SuiteConfig,
    output_path: Optional[Path],
    reporter: Reporter,
    tarball_source: Optional[str] = None,
) -> int:
    """Run the benchmark suite and return an exit code."""
    if not _reference_provider_available(config, reporter):
        return 1

    if config.backend == ExecutionBackendName.PYTORCH:
        from .pytorch_suite_runner import run_pytorch_suite_benchmark

        return run_pytorch_suite_benchmark(
            graph_paths=graph_paths,
            config=config,
            output_path=output_path,
            reporter=reporter,
            tarball_source=tarball_source,
        )

    from .hipdnn_suite_runner import run_hipdnn_suite_benchmark

    return run_hipdnn_suite_benchmark(
        graph_paths=graph_paths,
        config=config,
        output_path=output_path,
        reporter=reporter,
        tarball_source=tarball_source,
    )


def run_suite_cli(
    args: argparse.Namespace,
    graph_paths: List[Path],
    reporter: Reporter,
    tarball_source: Optional[str] = None,
) -> int:
    """Validate suite CLI args, build config, and delegate to run_suite_benchmark."""
    try:
        backend = ExecutionBackendName(args.backend)
        validation = ValidationConfig(
            provider=args.validate,
            rtol=args.rtol,
            atol=args.atol,
        )
        metrics_config = MetricsConfig(
            tier=args.metrics_tier,
            emit_trace=args.emit_trace,
            pmc_set=args.pmc,
            perf=args.perf,
            roofline=args.roofline,
            pmc_allow_multipass=args.pmc_allow_multipass,
            profiling_output_dir=args.profiling_output_dir,
            profiling_timeout_s=args.profiling_timeout,
        )
        if backend is ExecutionBackendName.PYTORCH:
            if args.engine:
                reporter.print_error("--engine is not supported with --backend pytorch")
                return 1
            if args.plugin_path:
                reporter.print_error(
                    "--plugin-path is not supported with --backend pytorch"
                )
                return 1
            if validation.provider is ReferenceProviderName.PYTORCH:
                reporter.print_error(
                    "--validate pytorch is not supported with --backend pytorch "
                    "(the backend would validate against itself)"
                )
                return 1
            if metrics_config.opt_in_pass_requested:
                reporter.print_error(
                    "Profiling options (--pmc, --emit-trace, --perf, "
                    "--roofline) are not supported with --backend pytorch"
                )
                return 1
        # --profiling-output-dir is only meaningful when at least one
        # opt-in profiling source fires. Passing it solo is a silent
        # no-op today; surface that as a soft warning so the user
        # knows to add --pmc / --emit-trace / --perf / --roofline.
        if (
            metrics_config.profiling_output_dir is not None
            and not metrics_config.opt_in_pass_requested
        ):
            reporter.print_warning(
                "--profiling-output-dir set but no opt-in profiling "
                "source requested (--pmc, --emit-trace, --perf, "
                "--roofline); the directory will not be written to"
            )
        plugin_paths = None
        if backend is not ExecutionBackendName.PYTORCH:
            plugin_paths = args.plugin_path or _plugin_paths_from_environment()
        config = SuiteConfig(
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            seed=args.seed,
            engine_filter=args.engine,
            verbose=args.verbose,
            metrics=metrics_config,
            validation=validation,
            plugin_paths=plugin_paths,
            backend=backend,
        )
    except ValueError as e:
        reporter.print_error(f"Suite configuration error: {e}")
        return 1

    return run_suite_benchmark(
        graph_paths=graph_paths,
        config=config,
        output_path=args.output,
        reporter=reporter,
        tarball_source=tarball_source,
    )
