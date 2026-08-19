# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Command-line entry point for the geko executable.

Defines the argparse parser (build_parser), runs validation / normalization
(parse_cli_args), and dispatches into the high-level pipeline functions in
geko.pipeline (run_configure, run_optimize, run_search, run_bench).

Workload may be supplied as a hipBLASLt log YAML, a generator-style tuning
YAML (via load_prepared_config_from_yaml), or a single inline GEMM
description. The three top-level modes -- --tune, --search, and --bench --
are mutually exclusive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import yaml

from geko import logger, _set_log_level
from geko.config_generator.load_input_config import load_prepared_config_from_yaml
from geko.constants import SUPPORTED_ARCH
from geko.paths import resolve_hipblaslt_path
from geko.pipeline import run_bench, run_configure, run_optimize, run_search
from geko.schemas import GemmConfig, GemmType
from geko.utils import parse_devices

# Sample tuning YAML path shown when --list input is missing or invalid.
_SAMPLE_GEMM_LIST_YAML = Path(__file__).resolve().parent / "config_generator" / "config.yaml"


def _alloc_run_root() -> Path:
    """Return a unique cwd-relative run directory (timestamp + microsecond suffix)."""
    base = datetime.now().strftime("geko_%Y%m%d_%H%M%S_%f")
    root = Path(base)
    n = 0
    while root.exists():
        n += 1
        root = Path(f"{base}_{n}")
    return root


def _rows_from_gemm_config_yaml(path: Path, arch: str | None) -> List[dict]:
    """Flatten GemmProblems from load_prepared_config_from_yaml to workload-log dicts."""
    prepared = load_prepared_config_from_yaml(config_path=path, arch=arch)
    problems: List[GemmConfig] = prepared["GemmProblems"]
    rows: List[dict] = []
    for gc in problems:
        rows.extend(gc.workload_log_rows())
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser (mutually exclusive --tune/--search/--bench and workload sources)."""
    parser = argparse.ArgumentParser(
        description="GEKO: tune (GA), search, or benchmark a hipBLASLt GEMM workload log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true", help="GA workflow: configure then optimize")
    mode.add_argument("--search", action="store_true", help="Dense-search tuning workflow")
    mode.add_argument("--bench", action="store_true", help="Benchmark GEMMs from the workload log only")
    workload_src = parser.add_mutually_exclusive_group(required=True)
    workload_src.add_argument(
        "--workload-log",
        type=str,
        metavar="PATH",
        dest="workload",
        help="hipBLASLt GEMM workload log YAML (often produced with HIPBLASLT_LOG_MASK=64)",
    )
    workload_src.add_argument(
        "--list",
        type=str,
        metavar="PATH",
        dest="gemm_config",
        help="Generator tuning YAML (via load_prepared_config_from_yaml): set ARCH in file or pass --arch; "
        "supports multiple sizes per GEMM type and multiple GemmProblems when applicable",
    )
    workload_src.add_argument(
        "--inline",
        nargs=9,
        metavar=("M", "N", "batch", "K", "DataType", "DestDataType", "ComputeDataType", "transA", "transB"),
        help=(
            "Single GEMM: M N batch_count K, Tensile DataType / DestDataType / ComputeDataType "
            "(e.g. B B S), transA and transB each N or T (e.g. --inline 1024 1024 1 1024 B B S N T)"
        ),
    )
    parser.add_argument(
        "--hipblaslt",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "hipBLASLt checkout root (overrides auto-detection and $GEKO_HIPBLASLT_PATH). "
            "Auto-detected from the launcher location when omitted."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=SUPPORTED_ARCH,
        metavar="ARCH",
        help="Target gfx architecture (required with --tune).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=WARNING, 1=INFO, 2=DEBUG",
    )
    parser.add_argument(
        "--devices",
        "-d",
        action="store",
        type=str,
        required=True,
        help=(
            "Comma-separated list of GPU device IDs (e.g., 0,1,2,3). "
        ),
    )
    parser.add_argument(
        "--n_slots",
        "-n",
        action="store",
        type=int,
        default=4,
        help="Max concurrent optimization jobs per device",
    )
    parser.add_argument(
        "--keep_thr",
        type=float,
        default=None,
        help="Keep threshold for workload filtering (default: 0.0 for --tune/--bench, 0.1 for --search)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ductile",
        choices=["ductile", "tensile"],
        help="Tuning backend for configure step (only used with --tune)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        metavar="PATH",
        help="Output directory (defaults to autogenerated run directory)",
    )
    parser.add_argument(
        "--up_thr",
        type=float,
        default=1.03,
        help="Performance uplift threshold for kernel filtering",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.04,
        help="Target benchmark duration in seconds (used with --search)",
    )
    parser.add_argument(
        "--benchmark-duration",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Target seconds per cold and per timed phase when using --bench",
    )
    parser.add_argument(
        "--no_retry",
        action="store_true",
        help="Do not retry failed operations (used with --tune)",
    )
    parser.add_argument(
        "--bench-freq",
        dest="bench_freq",
        action="store_true",
        default=False,
        help=(
            "Enable HIPBLASLT_BENCH_FREQ during hipblaslt-bench runs to collect "
            "clock frequency telemetry. Off by default to avoid the small "
            "collection / output overhead during offline tuning."
        ),
    )
    return parser


@dataclass(frozen=True)
class CliArgs:
    """Normalized CLI flags after filesystem and --tune/--arch checks."""

    tune: bool
    bench: bool
    search: bool
    workload: str | None
    gemm_config: str | None
    inline: tuple[int, int, int, int, str, str, str, str, str] | None
    arch: str | None
    hipblaslt: str | None
    verbose: int
    devices: list[int]
    n_slots: int
    keep_thr: float
    backend: str
    workdir: str | None
    up_thr: float
    duration: float
    benchmark_duration: float
    retry: bool
    bench_freq: bool


def parse_cli_args(argv: Sequence[str] | None) -> CliArgs:
    """Parse argv (default sys.argv); invalid paths or rules call parser.error (SystemExit)."""
    parser = build_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)

    workload = ns.workload
    gemm_config = ns.gemm_config
    inline_raw = ns.inline

    if workload is not None:
        wp = Path(workload)
        if not wp.is_file():
            parser.error(f"--workload-log file not found: {wp}")

    if gemm_config is not None:
        p = Path(gemm_config)
        if not p.is_file():
            parser.error(f"--list file not found: {p} (example: {_SAMPLE_GEMM_LIST_YAML})")

    inline: tuple[int, int, int, int, str, str, str, str, str] | None = None
    if inline_raw is not None:
        m_s, n_s, b_s, k_s, data_t, dest_t, comp_t, ta, tb = inline_raw
        try:
            m_i, n_i, b_i, k_i = int(m_s), int(n_s), int(b_s), int(k_s)
        except ValueError:
            parser.error("--inline: M, N, batch, and K must be integers")
        trans_a = str(ta).strip().upper()
        trans_b = str(tb).strip().upper()
        if trans_a not in ("N", "T") or trans_b not in ("N", "T"):
            parser.error("--inline: transA and transB must each be N or T")
        dt, dd, cd = str(data_t).strip(), str(dest_t).strip(), str(comp_t).strip()
        inline = (m_i, n_i, b_i, k_i, dt, dd, cd, trans_a, trans_b)

    if ns.tune and ns.arch is None:
        parser.error("--arch is required with --tune")

    try:
        devices = parse_devices(ns.devices)
    except ValueError as e:
        parser.error(str(e))

    keep_thr = ns.keep_thr if ns.keep_thr is not None else (0.1 if ns.search else 0.0)

    return CliArgs(
        tune=ns.tune,
        bench=ns.bench,
        search=ns.search,
        workload=workload,
        gemm_config=gemm_config,
        inline=inline,
        arch=ns.arch,
        hipblaslt=ns.hipblaslt,
        verbose=ns.verbose,
        devices=devices,
        n_slots=ns.n_slots,
        keep_thr=keep_thr,
        backend=ns.backend,
        workdir=ns.workdir,
        up_thr=ns.up_thr,
        duration=ns.duration,
        benchmark_duration=ns.benchmark_duration,
        retry=not ns.no_retry,
        bench_freq=ns.bench_freq,
    )


def dispatch(args: CliArgs, anchor: str | None = None) -> int:
    """Materialize a run dir, build a workload YAML if needed, then bench or full tune.

    anchor is a path inside the hipBLASLt checkout (the launcher's location),
    used to auto-detect the hipBLASLt root when --hipblaslt / the environment
    variable are not set. The installed geko console script has no checkout
    anchor, so it relies on --hipblaslt or $GEKO_HIPBLASLT_PATH.
    """
    hipblaslt_path = resolve_hipblaslt_path(
        explicit=args.hipblaslt, anchor=anchor, require_built=True
    )

    run_root = Path(args.workdir) if args.workdir is not None else _alloc_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    run_root_str = str(run_root.resolve())

    _set_log_level(args.verbose)
    logger.info(f"Run directory: {run_root_str}")
    logger.info(f"hipBLASLt path: '{hipblaslt_path}'")

    log_path: Path
    if args.workload is not None:
        log_path = Path(args.workload)
    elif args.gemm_config is not None:
        try:
            rows = _rows_from_gemm_config_yaml(Path(args.gemm_config), args.arch)
        except (ValueError, FileNotFoundError) as e:
            logger.error(str(e))
            logger.error("Example tuning YAML: %s", _SAMPLE_GEMM_LIST_YAML)
            return 1
        log_path = run_root / "synthetic_workload.yaml"
        with log_path.open("w") as f:
            yaml.safe_dump(rows, f, default_flow_style=None, sort_keys=False, width=5000)
    elif args.inline is not None:
        m, n, batch_count, k, data_t, dest_t, comp_t, trans_a, trans_b = args.inline
        try:
            gtype = GemmType.from_tensile(trans_a, trans_b, data_t, dest_t, comp_t)
            rows = GemmConfig(gtype, [[m, n, batch_count, k]]).workload_log_rows()
        except ValueError as e:
            logger.error(str(e))
            return 1
        log_path = run_root / "synthetic_workload.yaml"
        with log_path.open("w") as f:
            yaml.safe_dump(rows, f, default_flow_style=None, sort_keys=False, width=5000)
    else:
        logger.error("Expected exactly one of --workload-log, --list, or --inline (parser bug?)")
        return 1

    if args.bench:
        return run_bench(
            hipblaslt_path,
            str(log_path),
            run_root_str,
            devices=args.devices,
            benchmark_duration=args.benchmark_duration,
            bench_freq=args.bench_freq,
        )
    if args.search:
        run_search(
            hipblaslt_path,
            str(log_path),
            devices=args.devices,
            keep_thr=args.keep_thr,
            up_thr=args.up_thr,
            workdir=run_root_str,
            verbose=args.verbose,
            duration=args.duration,
            bench_freq=args.bench_freq,
        )
        logger.info(f"Search outputs under '{run_root_str}'")
        return 0
    if args.tune:
        run_configure(
            hipblaslt_path,
            str(log_path),
            devices=args.devices,
            keep_thr=args.keep_thr,
            arch=args.arch,
            backend=args.backend,
            workdir=run_root_str,
            verbose=args.verbose,
            bench_freq=args.bench_freq,
        )
        run_optimize(
            hipblaslt_path,
            workdir=run_root_str,
            devices=args.devices,
            n_slots=args.n_slots,
            retry=args.retry,
            verbose=args.verbose,
            bench_freq=args.bench_freq,
        )
        logger.info(f"Tuning outputs under '{run_root_str}'")
        return 0
    raise NotImplementedError("Expected --tune, --search, or --bench (parser bug?)")


def main(argv: Sequence[str] | None = None, anchor: str | None = None) -> int:
    """Console script entry: parse_cli_args → dispatch; maps SystemExit to int code.

    anchor is forwarded to dispatch for hipBLASLt root auto-detection;
    in-tree launchers (bin/geko) pass their location, the installed console
    script leaves it None and relies on --hipblaslt / the env var.
    """
    try:
        return dispatch(parse_cli_args(argv), anchor=anchor)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    raise SystemExit(main())
