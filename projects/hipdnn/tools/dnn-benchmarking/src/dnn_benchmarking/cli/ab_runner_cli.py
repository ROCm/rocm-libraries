# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""A/B comparison CLI runner."""

import argparse
from pathlib import Path
from typing import Literal, Optional

from ..common.exceptions import ExecutionError, GraphLoadError
from ..config.benchmark_config import ABTestConfig, BenchmarkConfig, ValidationConfig
from ..execution.ab_runner import ABRunner
from ..graph.loader import GraphLoader
from ..reporting.reporter import Reporter
from ..reporting.statistics import CombinedBenchmarkStats


def run_ab_benchmark(
    config: BenchmarkConfig,
    ab_config: ABTestConfig,
    reporter: Reporter,
    seed: Optional[int] = None,
    gpu_backend: Literal["torch", "auto", "none"] = "auto",
    validation_config: Optional[ValidationConfig] = None,
) -> int:
    """Run A/B comparison workflow.

    Args:
        config: Benchmark configuration.
        ab_config: A/B test configuration.
        reporter: Reporter instance for console output.
        seed: Optional random seed for reproducibility.
        gpu_backend: GPU timer backend to use (torch, auto, none).
        validation_config: Optional validation configuration for reference checking.

    Returns:
        Exit code (0 for success, 1 for error, 2 for comparison failure).
    """

    try:
        ab_config.validate_paths()

        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)
        loader.validate(graph_json)

        graph_name = loader.get_graph_name(graph_json)

        reporter.print_ab_header(config, ab_config, graph_name)

        runner = ABRunner(
            graph_json,
            config,
            ab_config,
            gpu_backend=gpu_backend,
            validation_config=validation_config,
        )
        result = runner.run(seed=seed)

        stats_a = CombinedBenchmarkStats.from_result(result.result_a)
        stats_b = CombinedBenchmarkStats.from_result(result.result_b)

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

        if validation_config is not None and validation_config.enabled:
            reporter.print_ab_validation(
                result.validation_a,
                result.validation_b,
                validation_config.rtol,
                validation_config.atol,
            )

        reporter.print_footer()

        validation_passed = True
        if result.validation_a is not None and not result.validation_a.passed:
            validation_passed = False
        if result.validation_b is not None and not result.validation_b.passed:
            validation_passed = False

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


def run_ab_cli(args: argparse.Namespace, graph_path: Path, reporter: Reporter) -> int:
    """Validate A/B CLI args, build configs, and delegate to run_ab_benchmark."""

    if args.AId is None or args.BId is None:
        reporter.print_error(
            "A/B testing requires both --AId and --BId to be specified"
        )
        return 1

    if args.engine:
        reporter.print_error(
            "--engine is not supported in A/B testing mode "
            "(use --AId and --BId instead)"
        )
        return 1

    try:
        config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            engine_id=args.AId,
        )
    except ValueError as e:
        reporter.print_error(f"Configuration error: {e}")
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
        reporter.print_error(f"A/B configuration error: {e}")
        return 1

    validation_config = None
    if args.validate != "none":
        try:
            validation_config = ValidationConfig(
                provider=args.validate,
                rtol=args.rtol,
                atol=args.atol,
            )
        except ValueError as e:
            reporter.print_error(f"Validation configuration error: {e}")
            return 1

    return run_ab_benchmark(
        config,
        ab_config,
        reporter,
        seed=args.seed,
        gpu_backend="auto",
        validation_config=validation_config,
    )
