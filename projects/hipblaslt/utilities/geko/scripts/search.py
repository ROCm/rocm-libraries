# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse

from geko.paths import resolve_hipblaslt_path
from geko.pipeline import run_search
from geko.utils import parse_devices


def main() -> None:
    """Run the dense-search (offline tuning) workflow (legacy script entry).

    Parses CLI flags then calls run_search (summarize + search + analyze + final libs).
    """
    parser = argparse.ArgumentParser(
        description="Dense-search (offline tuning) workflow across existing hipBLASLt solutions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gemm_log",
        type=str,
        help="Path to hipBLASLt YAML log file with GEMM operations (collected with HIPBLASLT_LOG_MASK=64). "
             "CSV format is also supported, expecting the same fields as the YAML logs.",
    )
    parser.add_argument(
        "--devices",
        "-d",
        action="store",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of GPU device IDs (e.g., 0,1,2,3). "
             "The first device is also used for the analyze step.",
    )
    parser.add_argument(
        "--keep_thr",
        type=float,
        default=0.1,
        help=(
            "Percentage threshold for filtering GEMMs by contribution to E2E latency. "
            "Sizes contribute differently (including call count); setting keep_thr = 0 searches all sizes, "
            "while values > 0 skip sizes whose contribution is below the threshold (e.g., 0.1 skips sizes contributing < 0.1%%)."
        ),
    )
    parser.add_argument(
        "--up_thr",
        type=float,
        default=1.03,
        help="Performance uplift threshold for kernel filtering",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        default="workdir",
        help="Working directory for intermediate files and outputs",
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
        "--duration",
        type=float,
        default=0.04,
        help="Target dense-search benchmark duration per GEMM (seconds)",
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
            "clock frequency telemetry. Off by default."
        ),
    )

    args = parser.parse_args()
    devices = parse_devices(args.devices)
    hipblaslt_path = resolve_hipblaslt_path(
        explicit=args.hipblaslt, anchor=__file__, require_built=True
    )

    run_search(
        hipblaslt_path,
        args.gemm_log,
        devices=devices,
        keep_thr=args.keep_thr,
        up_thr=args.up_thr,
        workdir=args.workdir,
        verbose=args.verbose,
        duration=args.duration,
        bench_freq=args.bench_freq,
    )


if __name__ == "__main__":
    main()
