# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse

from geko.constants import SUPPORTED_ARCH
from geko.paths import resolve_hipblaslt_path
from geko.pipeline import run_configure


def main() -> None:
    """Generate Tensile tuning configs from a hipBLASLt GEMM log (legacy script entry).

    Parses CLI flags then calls run_configure (summarize + optim.configure).
    """
    parser = argparse.ArgumentParser(
        description="Generate Tensile configuration from hipBLASLt logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gemm_log",
        type=str,
        help="Path to hipBLASLt YAML log file with GEMM operations (collected with HIPBLASLT_LOG_MASK=64). "
             "CSV format is also supported, expecting the same fields as the YAML logs.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        default=0,
        help="GPU device ID for benchmarking",
    )
    parser.add_argument(
        "--keep_thr",
        type=float,
        default=0,
        help=(
            "Percentage threshold for filtering GEMMs by contribution to E2E latency. "
            "Sizes contribute differently (including call count); setting keep_thr = 0 tunes all sizes, "
            "while values > 0 skip sizes whose contribution is below the threshold (e.g., 0.1 skips sizes contributing < 0.1%%)."
        ),
    )
    parser.add_argument(
        "--architecture",
        "-a",
        type=str,
        default="gfx950",
        choices=SUPPORTED_ARCH,
        help="Target architecture",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="ductile",
        choices=[
            "ductile",
            "tensile",
        ],
        help="Tuning backend: Ductile or Tensile",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=None,
        choices=["heuristic", "generic"],
        dest="search_space",
        help="Search space strategy. Defaults to 'generic' for ductile, 'heuristic' for tensile",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        default="workdir",
        help="Working directory for intermediate files and configurations",
    )
    parser.add_argument(
        "--hipblaslt",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "hipBLASLt checkout root (overrides auto-detection and $GEKO_HIPBLASLT_PATH). "
            "Auto-detected from this script's location when omitted."
        ),
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
        "--bench-freq",
        dest="bench_freq",
        action="store_true",
        default=False,
        help=(
            "Enable HIPBLASLT_BENCH_FREQ during hipblaslt-bench runs to collect "
            "clock frequency telemetry. Only relevant when --keep_thr > 0 "
            "(otherwise no benchmarking happens). Off by default."
        ),
    )

    args = parser.parse_args()
    hipblaslt_path = resolve_hipblaslt_path(explicit=args.hipblaslt, anchor=__file__)

    run_configure(
        hipblaslt_path,
        args.gemm_log,
        device=args.device,
        keep_thr=args.keep_thr,
        arch=args.architecture,
        backend=args.backend,
        search_space=args.search_space,
        workdir=args.workdir,
        verbose=args.verbose,
        bench_freq=args.bench_freq,
    )


if __name__ == "__main__":
    main()
