# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CLI argument parsing for dnn-benchmarking."""

import argparse
from pathlib import Path

from ..config.benchmark_config import BenchmarkConfig


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for dnn-benchmark CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="dnn-benchmark",
        description=(
            "Benchmarking and validation tool for hipDNN graphs\n\n"
            "WARNING: This tool is in early development and subject to change.\n"
            "Do not use it in build workflows or CI pipelines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dnn-benchmark --graph ./graphs/conv1_fwd.json
  dnn-benchmark --graph ./graphs/conv1_fwd.json --warmup 20 --iters 200
  dnn-benchmark -g ./graphs/conv1_fwd.json -e 1

PyTorch Backend (GPU via PyTorch):
  dnn-benchmark -g ./graph.json --backend pytorch
  dnn-benchmark -g ./graph.json --backend pytorch -o pytorch_results.json

Reference Validation:
  dnn-benchmark -g ./graph.json --validate pytorch
  dnn-benchmark -g ./graph.json --validate pytorch --validate-rtol 1e-3

A/B Testing:
  dnn-benchmark -g ./graph.json --AId 1 --BId 2
  dnn-benchmark -g ./graph.json --APath /path/pluginA --AId 1 --BPath /path/pluginB --BId 2
        """,
    )

    parser.add_argument(
        "--graph",
        "-g",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to JSON-serialized hipDNN graph file",
    )

    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=10,
        metavar="N",
        help="Number of warmup iterations (default: 10)",
    )

    parser.add_argument(
        "--iters",
        "-i",
        type=int,
        default=100,
        metavar="N",
        help="Number of benchmark iterations (default: 100)",
    )

    parser.add_argument(
        "--engine-id",
        "-e",
        type=int,
        default=1,
        metavar="ID",
        help="Engine ID to use (default: 1 for MIOpen)",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducible input data (default: None)",
    )

    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=["hipdnn", "pytorch"],
        default="hipdnn",
        metavar="BACKEND",
        help="Execution backend (default: hipdnn). "
        "Options: hipdnn (AMD GPU via hipDNN), pytorch (GPU via PyTorch)",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help="Export benchmark results to JSON file for offline comparison",
    )
    output_group.add_argument(
        "--gpu-backend",
        type=str,
        choices=["torch", "auto", "none"],
        default="auto",
        metavar="BACKEND",
        help="GPU timer backend (default: auto). "
        "Options: torch (PyTorch CUDA/ROCm), auto, none (E2E only)",
    )

    # A/B Testing arguments
    ab_group = parser.add_argument_group("A/B Testing")
    ab_group.add_argument(
        "--APath",
        type=Path,
        default=None,
        metavar="PATH",
        help="Plugin path for configuration A (default: use system default)",
    )
    ab_group.add_argument(
        "--AId",
        type=int,
        default=None,
        metavar="ID",
        help="Engine ID for configuration A",
    )
    ab_group.add_argument(
        "--BPath",
        type=Path,
        default=None,
        metavar="PATH",
        help="Plugin path for configuration B (default: use system default)",
    )
    ab_group.add_argument(
        "--BId",
        type=int,
        default=None,
        metavar="ID",
        help="Engine ID for configuration B",
    )
    ab_group.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        metavar="TOL",
        help="Relative tolerance for A/B comparison (default: 1e-5)",
    )
    ab_group.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        metavar="TOL",
        help="Absolute tolerance for A/B comparison (default: 1e-8)",
    )

    # Reference Validation arguments
    val_group = parser.add_argument_group("Reference Validation")
    val_group.add_argument(
        "--validate",
        type=str,
        choices=["pytorch", "cpu_plugin", "none"],
        default="none",
        metavar="PROVIDER",
        help="Reference provider for validation (default: none). "
        "Options: pytorch, cpu_plugin, none",
    )
    val_group.add_argument(
        "--validate-rtol",
        type=float,
        default=1e-5,
        metavar="TOL",
        help="Relative tolerance for validation (default: 1e-5)",
    )
    val_group.add_argument(
        "--validate-atol",
        type=float,
        default=1e-8,
        metavar="TOL",
        help="Absolute tolerance for validation (default: 1e-8)",
    )

    return parser


def parse_args(args=None) -> BenchmarkConfig:
    """Parse command line arguments and return BenchmarkConfig.

    Args:
        args: Command line arguments (default: sys.argv).

    Returns:
        BenchmarkConfig with parsed values.

    Raises:
        SystemExit: If arguments are invalid.
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    return BenchmarkConfig(
        graph_path=parsed.graph,
        warmup_iters=parsed.warmup,
        benchmark_iters=parsed.iters,
        engine_id=parsed.engine_id,
    )
