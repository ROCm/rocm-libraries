# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse

from geko.paths import resolve_hipblaslt_path
from geko.pipeline import run_optimize
from geko.utils import parse_devices


def main() -> None:
    """Run optimize, merge, benchmark, and filter (legacy script entry).

    Expects workdir produced by configure.py. Delegates to run_optimize.
    """
    parser = argparse.ArgumentParser(
        description="Merge tuning results into a single library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        default="workdir",
        help="Working directory containing tuning configurations. Output of configure.py",
    )
    parser.add_argument(
        "--devices",
        "-d",
        action="store",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of GPU device IDs (e.g., 0,1,2,3)",
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
        "--up_thr",
        type=float,
        default=1.03,
        help="Performance uplift threshold for kernel filtering",
    )
    parser.add_argument(
        "--err_thr",
        type=float,
        default=0.03,
        help="Error threshold for accuracy filtering",
    )
    parser.add_argument(
        "--client_build_dir",
        type=str,
        default="build_tmp",
        help="Directory path for tensilelite client build",
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
        "--no_retry",
        action="store_true",
        help="Do not retry failed operations",
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
            "Enable HIPBLASLT_BENCH_FREQ during the post-optim analyze sweep "
            "to collect clock frequency telemetry. Off by default."
        ),
    )
    args = parser.parse_args()
    devices = parse_devices(args.devices)
    hipblaslt_path = resolve_hipblaslt_path(
        explicit=args.hipblaslt, anchor=__file__, require_built=True
    )

    run_optimize(
        hipblaslt_path,
        workdir=args.workdir,
        devices=devices,
        n_slots=args.n_slots,
        up_thr=args.up_thr,
        err_thr=args.err_thr,
        client_build_dir=args.client_build_dir,
        retry=not args.no_retry,
        verbose=args.verbose,
        bench_freq=args.bench_freq,
    )


if __name__ == "__main__":
    main()
