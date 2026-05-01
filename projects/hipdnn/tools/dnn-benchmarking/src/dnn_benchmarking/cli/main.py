# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Main entry point for dnn-benchmark CLI."""

import glob
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional

from ..common.exceptions import ExecutionError, GraphLoadError
from ..config.benchmark_config import (
    ABTestConfig,
    BenchmarkConfig,
    SuiteConfig,
    ValidationConfig,
)
from ..execution.ab_runner import ABRunner
from ..execution.suite_runner import run_graph_all_providers
from ..graph.loader import GraphLoader
from ..reporting.reporter import Reporter
from ..reporting.statistics import CombinedBenchmarkStats
from ..reporting.suite_results import (
    GraphResult,
    ProviderEngineResult,
    SuiteMetadata,
    SuiteResult,
    collect_environment_info,
)
from ..validation.reference_provider import ReferenceProviderRegistry
from .parser import create_parser


def run_pytorch_benchmark(
    config: BenchmarkConfig,
    seed: Optional[int] = None,
    output_path: Optional[Path] = None,
    device: str = "cuda:0",
) -> int:
    """Run PyTorch CUDA benchmark workflow.

    Args:
        config: Benchmark configuration.
        seed: Optional random seed for reproducibility.
        output_path: Optional path to export benchmark results as JSON.
        device: CUDA device to use.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from ..execution.pytorch_buffer_manager import PyTorchCudaBufferManager
    from ..execution.pytorch_executor import PyTorchCudaExecutor, PyTorchExecutionError

    reporter = Reporter()

    try:
        # Load graph (skip hipDNN-specific validation)
        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)

        graph_name = loader.get_graph_name(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)

        # Print header
        reporter.print_pytorch_header(config, graph_name, device)

        # Check PyTorch CUDA availability
        try:
            import torch

            if not torch.cuda.is_available():
                reporter.print_error(
                    "PyTorch GPU not available. "
                    "Install PyTorch with CUDA or ROCm support."
                )
                return 1
        except ImportError:
            reporter.print_error(
                "PyTorch not available. Install with: pip install torch"
            )
            return 1

        # Create executor
        executor = PyTorchCudaExecutor(graph_json, config, device=device)
        executor.prepare()

        reporter.print_init_time(executor.init_time_ms)

        # Allocate buffers
        with PyTorchCudaBufferManager(tensor_infos, device=device) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=seed)
            buffer_manager.zero_outputs()

            tensors = buffer_manager.get_tensors()

            # Run warmup
            executor.warmup(tensors)

            # Run benchmark
            result = executor.benchmark(tensors, graph_name=graph_name)

            # Calculate statistics
            stats = CombinedBenchmarkStats.from_result(result)
            reporter.print_combined_stats(stats)

            # Export results if requested
            if output_path:
                result.save_json(str(output_path))
                print(f"Results exported to: {output_path}")

        reporter.print_footer()
        return 0

    except GraphLoadError as e:
        reporter.print_error(f"Graph load error: {e}")
        return 1

    except PyTorchExecutionError as e:
        reporter.print_error(f"PyTorch execution error: {e}")
        return 1

    except Exception as e:
        reporter.print_error(f"Unexpected error: {e}")
        return 1


def run_ab_test(
    config: BenchmarkConfig,
    ab_config: ABTestConfig,
    seed: Optional[int] = None,
    gpu_backend: Literal["torch", "auto", "none"] = "auto",
    validation_config: Optional[ValidationConfig] = None,
) -> int:
    """Run A/B comparison workflow.

    Args:
        config: Benchmark configuration.
        ab_config: A/B test configuration.
        seed: Optional random seed for reproducibility.
        gpu_backend: GPU timer backend to use (torch, auto, none).
        validation_config: Optional validation configuration for reference checking.

    Returns:
        Exit code (0 for success, 1 for error, 2 for comparison failure).
    """
    reporter = Reporter()

    try:
        # Validate plugin paths if specified
        ab_config.validate_paths()

        # Load and validate graph
        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)
        loader.validate(graph_json)

        graph_name = loader.get_graph_name(graph_json)

        # Print header
        reporter.print_ab_header(config, ab_config, graph_name)

        # Run A/B comparison
        runner = ABRunner(
            graph_json,
            config,
            ab_config,
            gpu_backend=gpu_backend,
            validation_config=validation_config,
        )
        result = runner.run(seed=seed)

        # Compute combined stats from results
        stats_a = CombinedBenchmarkStats.from_result(result.result_a)
        stats_b = CombinedBenchmarkStats.from_result(result.result_b)

        # Print results with both E2E and kernel stats
        reporter.print_ab_combined_stats(
            stats_a,
            stats_b,
            result.init_time_a_ms,
            result.init_time_b_ms,
        )

        reporter.print_ab_comparison(
            result.passed,
            result.max_abs_diff,
            result.max_rel_diff,
            ab_config.rtol,
            ab_config.atol,
        )

        # Print validation results if available
        if validation_config is not None and validation_config.enabled:
            reporter.print_ab_validation(
                result.validation_a,
                result.validation_b,
                validation_config.rtol,
                validation_config.atol,
            )

        reporter.print_footer()

        # Check validation results
        validation_passed = True
        if result.validation_a is not None and not result.validation_a.passed:
            validation_passed = False
        if result.validation_b is not None and not result.validation_b.passed:
            validation_passed = False

        # Return 0 for pass, 2 for comparison or validation failure
        return 0 if (result.passed and validation_passed) else 2

    except GraphLoadError as e:
        reporter.print_error(f"Graph load error: {e}")
        return 1

    except ExecutionError as e:
        reporter.print_error(f"Execution error: {e}")
        return 1

    except ValueError as e:
        reporter.print_error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        reporter.print_error(f"Unexpected error: {e}")
        return 1


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


def _run_one_graph(graph_path: Path, config: SuiteConfig, handle: Any) -> GraphResult:
    """Load and run a single graph. Returns a GraphResult (errors included).

    Per-graph load/validate/execution failures are captured as error entries
    in the returned GraphResult so the suite continues to the next graph.
    """
    try:
        loader = GraphLoader()
        graph_json = loader.load_json(graph_path)
        loader.validate(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)
        result = run_graph_all_providers(
            graph_path, graph_json, tensor_infos, config, handle
        )
        if len(result.results) == 0:
            return _error_graph_result(
                graph_path, "No provider/engine combinations matched filters"
            )
        return result
    except (GraphLoadError, ExecutionError) as e:
        return _error_graph_result(graph_path, str(e))


def _build_suite_metadata(
    graph_results: List[GraphResult], total_graphs: int
) -> SuiteMetadata:
    """Aggregate per-graph counts and environment info into SuiteMetadata."""
    env_info = collect_environment_info()
    total_pass = total_fail = total_skip = total_error = 0
    for gr in graph_results:
        c = gr.count_by_status()
        total_pass += c.passed
        total_fail += c.failed
        total_skip += c.skipped
        total_error += c.errored
    total_combinations = total_pass + total_fail + total_skip + total_error
    return SuiteMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        total_graphs=total_graphs,
        total_combinations=total_combinations,
        pass_combinations=total_pass,
        fail_combinations=total_fail,
        skip_combinations=total_skip,
        error_combinations=total_error,
        rocm_version=env_info.get("rocm_version"),
        gpu_model=env_info.get("gpu_model"),
        python_version=env_info.get("python_version"),
        hipdnn_version=env_info.get("hipdnn_version"),
    )


def run_suite(
    graph_paths: List[Path],
    config: SuiteConfig,
    handle: Any,
) -> SuiteResult:
    """Run benchmark suite and return aggregated results.

    Pure runner: does not print, write files, or check the validation
    startup gate (the caller is responsible for that). Per-graph errors
    are captured in the returned SuiteResult; the function does not raise
    for per-graph load or execution failures.

    Args:
        graph_paths: List of resolved graph file paths.
        config: Suite configuration.
        handle: hipdnn.Handle instance, ready to use.

    Returns:
        Aggregated SuiteResult covering every graph in ``graph_paths``.
    """
    graph_results: List[GraphResult] = [
        _run_one_graph(graph_path, config, handle) for graph_path in graph_paths
    ]
    metadata = _build_suite_metadata(graph_results, total_graphs=len(graph_paths))
    return SuiteResult(metadata=metadata, graphs=graph_results)


def _suite_exit_code(suite_result: SuiteResult) -> int:
    """Derive a CLI exit code from a SuiteResult.

    Returns:
        0 if all passed, 2 if any correctness failures, otherwise 1 if any
        execution errors.
    """
    if suite_result.metadata.fail_combinations > 0:
        return 2
    if suite_result.metadata.error_combinations > 0:
        return 1
    return 0


def _orchestrate_suite_cli(
    graph_paths: List[Path],
    config: SuiteConfig,
    output_path: Optional[Path],
    plugin_path: Optional[Path],
) -> int:
    """CLI orchestration wrapper around run_suite().

    Owns all side effects: validation startup gate, hipdnn import, console
    output via Reporter, and JSON export. Returns the CLI exit code.
    """
    reporter = Reporter()
    total = len(graph_paths)

    # Validation startup gate: if --validate was requested, fail before
    # iterating any graphs when the reference provider isn't registered or
    # isn't available. Otherwise --validate silently degrades to a no-op.
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

    reporter.print_suite_header(total)

    try:
        import hipdnn_frontend as hipdnn

        if plugin_path is not None:
            hipdnn.set_engine_plugin_paths([str(plugin_path)])
        handle = hipdnn.Handle()
    except ImportError:
        reporter.print_error(
            "hipdnn_frontend not available. Install hipDNN Python bindings first."
        )
        return 1

    suite_result = run_suite(graph_paths, config, handle)

    for i, gr in enumerate(suite_result.graphs, start=1):
        reporter.print_suite_graph_start(i, total, gr.graph_name)
        # Pre-execution graph errors come back as a single "unknown" provider
        # entry with an error message and no timing data; surface those via
        # the dedicated graph-error printer so they read like load failures.
        is_setup_error = (
            len(gr.results) == 1
            and gr.results[0].status == "error"
            and gr.results[0].provider == "unknown"
        )
        if is_setup_error:
            reporter.print_suite_graph_error(
                gr.graph_name, gr.results[0].error_message or "unknown error"
            )
            continue

        if config.verbose:
            reporter.print_verbose_graph_result(gr, config)
        else:
            counts = gr.count_by_status()
            reporter.print_suite_graph_result(
                counts.passed, counts.failed, counts.skipped, counts.errored
            )

    if output_path is not None:
        suite_result.save_json(str(output_path))

    reporter.print_suite_summary(suite_result.metadata)
    reporter.print_suite_footer()

    return _suite_exit_code(suite_result)


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args()

    gpu_backend = "none" if args.no_kernel_timing else "auto"

    # Resolve --graph: glob expansion for suite mode.
    # recursive=True so '**' patterns match nested directories.
    resolved_files = sorted(glob.glob(args.graph, recursive=True))

    # Backward compatibility: if raw string is a single existing file
    if not resolved_files and Path(args.graph).is_file():
        resolved_files = [args.graph]

    if not resolved_files:
        print(
            f"No graph files found matching: {args.graph}",
            file=sys.stderr,
        )
        return 1

    # A/B testing mode: --AId or --BId specified (kept as a separate path for now).
    # TODO(follow-up): --output is currently silently ignored in this mode -- run_ab_test
    # has no JSON export. Either add export or reject --output here.
    if args.AId is not None or args.BId is not None:
        if len(resolved_files) > 1:
            print(
                "A/B testing requires a single graph file, not a glob pattern",
                file=sys.stderr,
            )
            return 1

        if args.AId is None or args.BId is None:
            print(
                "A/B testing requires both --AId and --BId to be specified",
                file=sys.stderr,
            )
            return 1

        # --engine is meaningless in A/B mode (configurations come from
        # --AId / --BId). Reject rather than silently using args.engine[0].
        if args.engine:
            print(
                "--engine is not supported in A/B testing mode "
                "(use --AId and --BId instead)",
                file=sys.stderr,
            )
            return 1

        try:
            # engine_id is unused by the A/B path (it uses a_id / b_id
            # from ABTestConfig); pass a benign default.
            config = BenchmarkConfig(
                graph_path=Path(resolved_files[0]),
                warmup_iters=args.warmup,
                benchmark_iters=args.iters,
                engine_id=args.AId,
            )
        except ValueError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            return 1

        try:
            ab_config = ABTestConfig(
                a_path=args.APath,
                a_id=args.AId,
                b_path=args.BPath,
                b_id=args.BId,
                rtol=args.rtol,
                atol=args.atol,
            )
        except ValueError as e:
            print(f"A/B configuration error: {e}", file=sys.stderr)
            return 1

        ab_validation_config = None
        if args.validate != "none":
            try:
                ab_validation_config = ValidationConfig(
                    provider=args.validate,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            except ValueError as e:
                print(f"Validation configuration error: {e}", file=sys.stderr)
                return 1

        return run_ab_test(
            config,
            ab_config,
            seed=args.seed,
            gpu_backend=gpu_backend,
            validation_config=ab_validation_config,
        )

    # PyTorch backend: single-graph only, separate executor (no provider discovery)
    if args.backend == "pytorch":
        if len(resolved_files) > 1:
            print(
                "Suite mode is not supported with --backend pytorch",
                file=sys.stderr,
            )
            return 1
        # PyTorch backend executes one engine; a multi-ID list is ambiguous.
        if args.engine and len(args.engine) > 1:
            print(
                "--engine accepts only a single ID with --backend pytorch "
                "(got: " + ",".join(str(e) for e in args.engine) + ")",
                file=sys.stderr,
            )
            return 1
        try:
            pt_engine_id = args.engine[0] if args.engine else 1
            pt_config = BenchmarkConfig(
                graph_path=Path(resolved_files[0]),
                warmup_iters=args.warmup,
                benchmark_iters=args.iters,
                engine_id=pt_engine_id,
            )
        except ValueError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            return 1
        return run_pytorch_benchmark(
            pt_config,
            seed=args.seed,
            output_path=args.output,
        )

    # Unified hipDNN path: handles 1..N graphs x 1..N engines.
    # Single-graph is just a 1x1 instance; verbose flag selects rich vs summary.
    try:
        suite_config = SuiteConfig(
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            seed=args.seed,
            engine_filter=args.engine,
            rtol=args.rtol,
            atol=args.atol,
            gpu_backend=gpu_backend,
            reference_provider=args.validate,
            verbose=args.verbose,
        )
    except ValueError as e:
        print(f"Suite configuration error: {e}", file=sys.stderr)
        return 1

    return _orchestrate_suite_cli(
        graph_paths=[Path(p) for p in resolved_files],
        config=suite_config,
        output_path=args.output,
        plugin_path=args.plugin_path,
    )


if __name__ == "__main__":
    sys.exit(main())
