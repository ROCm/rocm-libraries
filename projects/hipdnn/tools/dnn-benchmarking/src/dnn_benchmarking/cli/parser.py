# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CLI argument parsing for dnn-benchmarking."""

import argparse
from pathlib import Path
from typing import List


def _parse_engine_list(s: str) -> List[int]:
    """Parse --engine value as a single ID or comma-separated list of IDs.

    Engine IDs are deterministic FNV-1a hashes of the engine name and may
    be negative when interpreted as signed int64, so we accept any int.
    Duplicates are removed while preserving first-seen order.

    Examples:
      "1"                      -> [1]
      "1,2,3"                  -> [1, 2, 3]
      "1, 2"                   -> [1, 2]
      "1,1,2"                  -> [1, 2]
      "3,1,3,2"                -> [3, 1, 2]
      "-4567890123456789012"   -> [-4567890123456789012]
    """
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise argparse.ArgumentTypeError("--engine requires at least one ID")
    try:
        ids = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--engine expects integer ID(s), got {s!r}")
    # Deduplicate while preserving first-seen order
    seen: set = set()
    deduped: List[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    return deduped


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
  dnn-benchmark -g ./graphs/conv1_fwd.json -v        # verbose per-engine output
  dnn-benchmark -g ./graphs/conv1_fwd.json -e 1,2    # compare engines 1 and 2

PyTorch Backend (GPU via PyTorch):
  dnn-benchmark -g ./graph.json --backend pytorch
  dnn-benchmark -g ./graph.json --backend pytorch -o pytorch_results.json

Reference Validation:
  dnn-benchmark -g ./graph.json --validate pytorch
  dnn-benchmark -g ./graph.json --validate pytorch --rtol 1e-3

A/B Testing:
  dnn-benchmark -g ./graph.json --AId 1 --BId 2
  dnn-benchmark -g ./graph.json --APath /path/pluginA --AId 1 --BPath /path/pluginB --BId 2

Suite Mode (multiple graphs):
  dnn-benchmark -g graphs/                           # all .json/.tar.gz files in directory
  dnn-benchmark --graph 'graphs/*.json' --warmup 10 --iters 100
  dnn-benchmark --graph 'graphs/*.json' -o results.json
  dnn-benchmark --graph 'graphs/*.json' -v           # rich block per (graph, engine)

Tarball Input:
  dnn-benchmark --graph graphs.tar.gz
  dnn-benchmark --graph graphs.tgz -o results.json
        """,
    )

    parser.add_argument(
        "--graph",
        "-g",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more paths, directories, glob patterns (e.g., 'graphs/*.json'), or "
        "tarballs (.tar, .tar.gz, .tgz) containing JSON graph files. "
        "A directory is searched recursively for .json files. "
        "Shell expansion (e.g., Workloads/BNorm/*) is accepted directly.",
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
        "--engine",
        "-e",
        type=_parse_engine_list,
        default=None,
        metavar="IDS",
        help="Engine ID or comma-separated list of IDs to run "
        "(default: all discovered engines). Examples: -e 1, -e 1,2,3",
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
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed per-engine breakdown for each graph "
        "(default: summary table)",
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
    # Comparison tolerances (used by A/B testing, validation, and suite mode)
    comparison_group = parser.add_argument_group("Comparison")
    comparison_group.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        metavar="TOL",
        help="Relative tolerance for output comparison (default: 1e-5)",
    )
    comparison_group.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        metavar="TOL",
        help="Absolute tolerance for output comparison (default: 1e-8)",
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

    # Suite options
    suite_group = parser.add_argument_group("Suite Options")
    suite_group.add_argument(
        "--plugin-path",
        type=Path,
        default=None,
        metavar="DIR",
        help="Path to directory containing hipDNN engine plugin .so files",
    )

    # Metrics options
    metrics_group = parser.add_argument_group("Metrics")
    metrics_group.add_argument(
        "--metrics-tier",
        type=str,
        choices=["basic", "off"],
        default="basic",
        metavar="TIER",
        help=(
            "Always-on metric tier (default: basic). 'basic' adds "
            "analytical FLOPs/IO, workspace size, host CPU rusage + RAM, "
            "amdsmi GPU snapshot, and machine metadata at zero extra "
            "runtime cost. 'off' disables all extra metric collection."
        ),
    )
    metrics_group.add_argument(
        "--emit-trace",
        type=str,
        choices=["pftrace", "kineto"],
        default=None,
        metavar="FORMAT",
        help=(
            "Re-run benchmark under rocprofv3 and export a kernel + "
            "memory-copy trace in the given format. 'kineto' falls back "
            "to pftrace when the rocpd Python module is not importable. "
            "Adds ~1 extra workload run (~5%% kernel-time overhead)."
        ),
    )
    metrics_group.add_argument(
        "--pmc",
        type=str,
        choices=["basic", "memory", "flops", "all"],
        default=None,
        metavar="SET",
        help=(
            "Re-run benchmark under rocprofv3 with the named PMC counter "
            "set. Per-kernel aggregates land in extra_metrics['pmc']. "
            "'all' requires --pmc-allow-multipass. Adds ~1 extra workload "
            "run (~30%% wallclock overhead)."
        ),
    )
    metrics_group.add_argument(
        "--pmc-allow-multipass",
        action="store_true",
        default=False,
        help=(
            "Required to use --pmc all. The unioned counter set crosses "
            "the single-pass replay budget and can hang on small "
            "workloads (Phase 1 testing observed a 4-min hang on a "
            "0.4 s baseline)."
        ),
    )
    metrics_group.add_argument(
        "--perf",
        action="store_true",
        default=False,
        help=(
            "Wrap re-run in 'perf stat -x,' to collect CPU cycles, "
            "instructions, IPC, and task-clock. Kernel-space events drop "
            "silently when /proc/sys/kernel/perf_event_paranoid > 1. "
            "Adds ~1 extra workload run."
        ),
    )
    metrics_group.add_argument(
        "--roofline",
        action="store_true",
        default=False,
        help=(
            "Re-run under 'rocprof-compute profile --roof-only' for a "
            "roofline plot. PDF + db paths land in "
            "extra_metrics['roofline']. Adds ~3 extra workload runs."
        ),
    )
    metrics_group.add_argument(
        "--roofline-data-type",
        type=str,
        choices=["FP32", "FP16", "BF16", "FP64", "INT8"],
        default="FP32",
        metavar="DTYPE",
        help=(
            "Data type for the roofline plot (default: FP32). Passed "
            "through to rocprof-compute --roofline-data-type."
        ),
    )
    metrics_group.add_argument(
        "--profiling-output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Root directory for profiling artefacts (rocpd dbs, "
            "pftraces, perf CSVs, roofline PDFs). Default: "
            "./profiling-output/<utc-timestamp>/."
        ),
    )

    # Hidden re-exec sub-mode: when an opt-in profiling source is
    # requested, the parent process shells out to a fresh CLI invocation
    # under the profiler. The child process picks up these flags to
    # short-circuit setup and run a single (graph, engine) workload.
    parser.add_argument(
        "--internal-profiling-run",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-profiling-engine",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-profiling-graph",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser
