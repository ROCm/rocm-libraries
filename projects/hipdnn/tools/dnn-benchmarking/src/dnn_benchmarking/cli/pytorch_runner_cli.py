# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch CUDA benchmark runner."""

import argparse
from pathlib import Path
from typing import Optional

from ..common.exceptions import GraphLoadError
from ..config.benchmark_config import BenchmarkConfig
from ..graph.loader import GraphLoader
from ..reporting.reporter import Reporter
from ..reporting.statistics import CombinedBenchmarkStats


def run_pytorch_benchmark(
    config: BenchmarkConfig,
    reporter: Reporter,
    seed: Optional[int] = None,
    output_path: Optional[Path] = None,
    device: str = "cuda:0",
) -> int:
    """Run PyTorch CUDA benchmark workflow.

    Args:
        config: Benchmark configuration.
        reporter: Reporter instance for console output.
        seed: Optional random seed for reproducibility.
        output_path: Optional path to export benchmark results as JSON.
        device: CUDA device to use.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from ..execution.pytorch_buffer_manager import PyTorchCudaBufferManager
    from ..execution.pytorch_executor import PyTorchCudaExecutor, PyTorchExecutionError

    try:
        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)

        graph_name = loader.get_graph_name(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)

        reporter.print_pytorch_header(config, graph_name, device)

        try:
            # PyTorch backend must be gated by PyTorch itself: CPU-only torch
            # cannot execute these GPU benchmarks even if ROCm management tools
            # can see a device.
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

        executor = PyTorchCudaExecutor(graph_json, config, device=device)
        executor.prepare()

        reporter.print_init_time(executor.init_time_ms)

        with PyTorchCudaBufferManager(tensor_infos, device=device) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=seed)
            buffer_manager.zero_outputs()

            tensors = buffer_manager.get_tensors()

            executor.warmup(tensors)

            result = executor.benchmark(tensors, graph_name=graph_name)

            stats = CombinedBenchmarkStats.from_result(result)
            reporter.print_combined_stats(stats)

            if output_path:
                result.save_json(str(output_path))
                reporter.print_results_exported(output_path)

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


def run_pytorch_cli(
    args: argparse.Namespace, graph_path: Path, reporter: Reporter
) -> int:
    """Validate PyTorch CLI args, build config, and delegate to run_pytorch_benchmark."""

    if args.engine and len(args.engine) > 1:
        reporter.print_error(
            "--engine accepts only a single ID with --backend pytorch "
            "(got: " + ",".join(str(e) for e in args.engine) + ")"
        )
        return 1

    try:
        config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            engine_id=args.engine[0] if args.engine else 1,
        )
    except ValueError as e:
        reporter.print_error(f"Configuration error: {e}")
        return 1

    return run_pytorch_benchmark(
        config,
        reporter,
        seed=args.seed,
        output_path=args.output,
    )
