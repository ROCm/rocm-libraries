# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CLI argument parsing for dnn-benchmarking."""

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, FrozenSet, List, Optional


class ConfigKind(str, Enum):
    """Config-file value normalization strategies."""

    SCALAR = "scalar"
    CHOICE = "choice"
    PATH = "path"
    PATH_LIST = "path_list"
    PATH_OR_PATH_LIST = "path_or_path_list"


@dataclass(frozen=True)
class CliOption:
    """Single source of truth for one public CLI option."""

    flags: tuple[str, ...]
    help: str
    dest: str
    default: Any = None
    parser_type: Any = None
    action: Optional[str] = None
    nargs: Any = None
    metavar: Optional[str] = None
    choices: Optional[FrozenSet[Any]] = None
    group: Optional[str] = None
    config_key: Optional[str] = None
    config_kind: Optional[ConfigKind] = None
    config_type: Optional[type] = None
    config_optional: bool = False

    def __post_init__(self) -> None:
        if (self.config_key is None) != (self.config_kind is None):
            raise ValueError(
                f"{self.dest}: config_key and config_kind must be set together"
            )
        if self.config_kind in {ConfigKind.SCALAR, ConfigKind.CHOICE}:
            if self.config_type is None:
                raise ValueError(
                    f"{self.dest}: {self.config_kind.value} requires config_type"
                )
        if self.config_kind is ConfigKind.CHOICE and self.choices is None:
            raise ValueError(f"{self.dest}: choice config fields require choices")

    @property
    def is_configurable(self) -> bool:
        return self.config_key is not None


def _parse_engine_list(s: str) -> List[int]:
    """Parse --engine value as a single ID or comma-separated list of IDs.

    Engine IDs are deterministic FNV-1a hashes of the engine name and may
    be negative when interpreted as signed int64, so we accept any int.
    Duplicate IDs are preserved because each comma-delimited entry is an
    ordered execution selection; this allows comparing the same engine ID
    from different plugin paths.

    Examples:
      "1"                      -> [1]
      "1,2,3"                  -> [1, 2, 3]
      "1, 2"                   -> [1, 2]
      "1,1,2"                  -> [1, 1, 2]
      "3,1,3,2"                -> [3, 1, 3, 2]
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
    return ids


def _parse_plugin_path_list(s: str) -> List[Path]:
    """Parse --plugin-path as a comma-separated list of plugin directories."""
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise argparse.ArgumentTypeError("--plugin-path requires at least one path")
    return [Path(p) for p in parts]


_BACKEND_CHOICES = frozenset({"hipdnn", "pytorch"})
_VALIDATE_CHOICES = frozenset({"pytorch", "cpu_plugin", "none"})
_METRICS_TIER_CHOICES = frozenset({"basic", "off"})
_EMIT_TRACE_CHOICES = frozenset({"pftrace", "kineto"})
_PMC_CHOICES = frozenset({"basic", "memory", "flops", "all"})

CLI_OPTIONS: tuple[CliOption, ...] = (
    CliOption(
        flags=("--graph", "-g"),
        dest="graph",
        nargs="+",
        metavar="PATH",
        help="One or more paths, directories, glob patterns (e.g., 'graphs/*.json'), or "
        "tarballs (.tar, .tar.gz, .tgz) containing JSON graph files. "
        "A directory is searched recursively for .json files. "
        "Shell expansion (e.g., Workloads/BNorm/*) is accepted directly.",
        config_key="graphs",
        config_kind=ConfigKind.PATH_LIST,
    ),
    CliOption(
        flags=("--config",),
        dest="config",
        parser_type=Path,
        metavar="PATH",
        help="TOML benchmark recipe. CLI flags override config values.",
    ),
    CliOption(
        flags=("--warmup", "-w"),
        dest="warmup",
        parser_type=int,
        default=10,
        metavar="N",
        help="Number of warmup iterations (default: 10)",
        config_key="warmup",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
    ),
    CliOption(
        flags=("--iters", "-i"),
        dest="iters",
        parser_type=int,
        default=100,
        metavar="N",
        help="Number of benchmark iterations (default: 100)",
        config_key="iters",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
    ),
    CliOption(
        flags=("--engine", "-e"),
        dest="engine",
        parser_type=_parse_engine_list,
        metavar="IDS",
        help="Engine ID or comma-separated list of IDs to run "
        "(default: all discovered engines). Examples: -e 1, -e 1,2,3",
    ),
    CliOption(
        flags=("--seed", "-s"),
        dest="seed",
        parser_type=int,
        metavar="SEED",
        help="Random seed for reproducible input data (default: None)",
        config_key="seed",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
        config_optional=True,
    ),
    CliOption(
        flags=("--backend", "-b"),
        dest="backend",
        parser_type=str,
        choices=_BACKEND_CHOICES,
        default="hipdnn",
        metavar="BACKEND",
        help="Execution backend (default: hipdnn). "
        "Options: hipdnn (AMD GPU via hipDNN), pytorch (GPU via PyTorch)",
        config_key="backend",
        config_kind=ConfigKind.CHOICE,
        config_type=str,
    ),
    CliOption(
        flags=("--output", "-o"),
        dest="output",
        parser_type=Path,
        metavar="PATH",
        group="Output",
        help="Export benchmark results to JSON file for offline comparison",
        config_key="output",
        config_kind=ConfigKind.PATH,
    ),
    CliOption(
        flags=("-v", "--verbose"),
        dest="verbose",
        action="store_true",
        default=False,
        group="Output",
        help="Show detailed per-engine breakdown for each graph "
        "(default: summary table)",
        config_key="verbose",
        config_kind=ConfigKind.SCALAR,
        config_type=bool,
    ),
    CliOption(
        flags=("--rtol",),
        dest="rtol",
        parser_type=float,
        default=None,
        metavar="TOL",
        group="Reference Comparison",
        help=(
            "Relative tolerance for output comparison (default: dtype-aware; "
            "if set without --atol, also used as absolute tolerance)"
        ),
        config_key="rtol",
        config_kind=ConfigKind.SCALAR,
        config_type=float,
        config_optional=True,
    ),
    CliOption(
        flags=("--atol",),
        dest="atol",
        parser_type=float,
        default=None,
        metavar="TOL",
        group="Reference Comparison",
        help=(
            "Absolute tolerance for output comparison (default: dtype-aware; "
            "if set without --rtol, also used as relative tolerance)"
        ),
        config_key="atol",
        config_kind=ConfigKind.SCALAR,
        config_type=float,
        config_optional=True,
    ),
    CliOption(
        flags=("--validate",),
        dest="validate",
        parser_type=str,
        choices=_VALIDATE_CHOICES,
        default="none",
        metavar="PROVIDER",
        group="Reference Validation",
        help="Reference provider for validation (default: none). "
        "Options: pytorch, cpu_plugin, none. "
        "With pytorch, suite output includes a timed reference row when "
        "PyTorch GPU execution is available.",
        config_key="validate",
        config_kind=ConfigKind.CHOICE,
        config_type=str,
    ),
    CliOption(
        flags=("--plugin-path",),
        dest="plugin_path",
        parser_type=_parse_plugin_path_list,
        metavar="PATHS",
        group="Suite Options",
        help=(
            "Directory containing hipDNN engine plugin .so files, or a "
            "comma-separated list matching --engine order. A single path is "
            "shared by all selected engines."
        ),
        config_key="plugin_path",
        config_kind=ConfigKind.PATH_OR_PATH_LIST,
    ),
    CliOption(
        flags=("--metrics-tier",),
        dest="metrics_tier",
        parser_type=str,
        choices=_METRICS_TIER_CHOICES,
        default="basic",
        metavar="TIER",
        group="Metrics",
        help=(
            "Always-on metric tier (default: basic). 'basic' adds "
            "analytical FLOPs/IO, workspace size, host CPU rusage + RAM, "
            "amdsmi GPU snapshot, and machine metadata at zero extra "
            "runtime cost. 'off' disables all extra metric collection."
        ),
        config_key="metrics_tier",
        config_kind=ConfigKind.CHOICE,
        config_type=str,
    ),
    CliOption(
        flags=("--emit-trace",),
        dest="emit_trace",
        parser_type=str,
        choices=_EMIT_TRACE_CHOICES,
        metavar="FORMAT",
        group="Metrics",
        help=(
            "Re-run benchmark under rocprofv3 and export a kernel + "
            "memory-copy trace in the given format. 'kineto' falls back "
            "to pftrace when the rocpd Python module is not importable. "
            "Adds ~1 extra workload run (~5%% kernel-time overhead)."
        ),
        config_key="emit_trace",
        config_kind=ConfigKind.CHOICE,
        config_type=str,
        config_optional=True,
    ),
    CliOption(
        flags=("--pmc",),
        dest="pmc",
        parser_type=str,
        choices=_PMC_CHOICES,
        metavar="SET",
        group="Metrics",
        help=(
            "Re-run benchmark under rocprofv3 with the named PMC counter "
            "set. Per-kernel aggregates land in extra_metrics['pmc']. "
            "'all' requires --pmc-allow-multipass. Adds ~1 extra workload "
            "run (~30%% wallclock overhead)."
        ),
        config_key="pmc",
        config_kind=ConfigKind.CHOICE,
        config_type=str,
        config_optional=True,
    ),
    CliOption(
        flags=("--pmc-allow-multipass",),
        dest="pmc_allow_multipass",
        action="store_true",
        default=False,
        group="Metrics",
        help=(
            "Required to use --pmc all. The unioned counter set exceeds "
            "the single-pass replay budget on most arches and rocprofv3 "
            "falls back to multi-pass replay, which has been observed to "
            "hang for minutes on sub-second workloads."
        ),
        config_key="pmc_allow_multipass",
        config_kind=ConfigKind.SCALAR,
        config_type=bool,
    ),
    CliOption(
        flags=("--perf",),
        dest="perf",
        action="store_true",
        default=False,
        group="Metrics",
        help=(
            "Wrap re-run in 'perf stat -x,' to collect CPU cycles, "
            "instructions, IPC, and task-clock. Kernel-space events drop "
            "silently when /proc/sys/kernel/perf_event_paranoid > 1. "
            "Adds ~1 extra workload run."
        ),
        config_key="perf",
        config_kind=ConfigKind.SCALAR,
        config_type=bool,
    ),
    CliOption(
        flags=("--roofline",),
        dest="roofline",
        action="store_true",
        default=False,
        group="Metrics",
        help=(
            "Re-run under 'rocprof-compute profile --roof-only' to "
            "capture HBM/compute ceilings. The CSV artefacts "
            "(roofline.csv, sysinfo.csv) and the workload directory "
            "path land in extra_metrics['roofline'] — render the PDF "
            "post-hoc via 'rocprof-compute analyze --path <workload>'. "
            "Adds ~3 extra workload runs."
        ),
        config_key="roofline",
        config_kind=ConfigKind.SCALAR,
        config_type=bool,
    ),
    CliOption(
        flags=("--profiling-output-dir",),
        dest="profiling_output_dir",
        parser_type=Path,
        metavar="DIR",
        group="Metrics",
        help=(
            "Root directory for profiling artefacts (rocpd dbs, "
            "pftraces, perf CSVs, roofline CSVs). Default: "
            "./profiling-output/<utc-timestamp>/."
        ),
        config_key="profiling_output_dir",
        config_kind=ConfigKind.PATH,
    ),
    CliOption(
        flags=("--profiling-timeout",),
        dest="profiling_timeout",
        parser_type=int,
        default=600,
        metavar="SECONDS",
        group="Metrics",
        help=(
            "Wall-clock budget for each external profiler subprocess "
            "(rocprofv3, perf, rocprof-compute, rocpd convert). Default "
            "600 s. A wedged child surfaces as 'timed out after Ns' in "
            "extra_metrics['<source>']['skipped'] instead of hanging the "
            "suite. Bump for known-long workloads (heavy graph under "
            "multi-pass PMC replay). Pass 0 to disable the timeout."
        ),
        config_key="profiling_timeout",
        config_kind=ConfigKind.SCALAR,
        config_type=int,
    ),
)

CONFIG_OPTIONS: tuple[CliOption, ...] = tuple(
    option for option in CLI_OPTIONS if option.is_configurable
)

OPTION_DEFAULTS: dict[str, Any] = {
    option.dest: option.default for option in CLI_OPTIONS
}


def _add_cli_option(
    parser: argparse.ArgumentParser,
    groups: dict[str, Any],
    option: CliOption,
    *,
    suppress_defaults: bool,
) -> None:
    target = groups.get(option.group, parser)
    kwargs: dict[str, Any] = {
        "default": argparse.SUPPRESS if suppress_defaults else option.default,
        "help": option.help,
    }
    if option.dest:
        kwargs["dest"] = option.dest
    if option.action is not None:
        kwargs["action"] = option.action
    if option.parser_type is not None:
        kwargs["type"] = option.parser_type
    if option.nargs is not None:
        kwargs["nargs"] = option.nargs
    if option.metavar is not None:
        kwargs["metavar"] = option.metavar
    if option.choices is not None:
        kwargs["choices"] = sorted(option.choices)
    target.add_argument(*option.flags, **kwargs)


def create_parser(*, suppress_defaults: bool = False) -> argparse.ArgumentParser:
    """Create the argument parser for dnn-benchmark CLI.

    Args:
        suppress_defaults: When True, absent public options are omitted from
            the parsed namespace. The CLI entry point uses this so config-file
            values can be merged as ``defaults < config < explicit CLI``.

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
  dnn-benchmark -g ./graphs/conv1_fwd.json -e 1,2
  dnn-benchmark --config sample_configs/basic.toml.example --graph ./graphs/conv1_fwd.json
  dnn-benchmark --config sample_configs/config.toml.example --iters 500

PyTorch Backend (GPU via PyTorch):
  dnn-benchmark -g ./graph.json --backend pytorch
  dnn-benchmark -g ./graph.json --backend pytorch -o pytorch_results.json

Reference Validation:
  dnn-benchmark -g ./graph.json --validate pytorch
  dnn-benchmark -g ./graph.json --validate pytorch --rtol 1e-3
  dnn-benchmark -g ./graph.json --validate pytorch -v  # includes PyTorch timing row when available

Engine Comparison:
  dnn-benchmark -g ./graph.json --engine 1,2,3
  dnn-benchmark -g ./graph.json --engine 1,2 --plugin-path /path/pluginA,/path/pluginB

Engine IDs:
  hipdnn_list_engines --plugin-dir /path/to/hipdnn_plugins/engines
  (shipped with hipDNN tools, e.g. /opt/rocm/bin/hipdnn_list_engines)

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

    groups = {
        "Output": parser.add_argument_group("Output"),
        "Reference Comparison": parser.add_argument_group("Reference Comparison"),
        "Reference Validation": parser.add_argument_group("Reference Validation"),
        "Suite Options": parser.add_argument_group("Suite Options"),
        "Metrics": parser.add_argument_group("Metrics"),
    }
    for option in CLI_OPTIONS:
        _add_cli_option(parser, groups, option, suppress_defaults=suppress_defaults)

    # --roofline-data-type intentionally absent: rocprof-compute only
    # accepts it under `analyze`, not `profile`. The profile run captures
    # ceilings at the tool's default datatype (FP32); rendering FP16/
    # BF16/etc. PDFs is a post-processing step the user runs themselves
    # against extra_metrics["roofline"]["workload_path"]:
    #   rocprof-compute analyze --path <workload_path> --roofline-data-type FP16

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
