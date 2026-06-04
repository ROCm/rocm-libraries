################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""CLI for running multiple tuning configs across GPUs with workload-aware scheduling.

Delegates to ``geko.optim.run`` (``geko.optim.optim.run``).
"""

import argparse

from pathlib import Path

from geko import logger, _set_log_level
from geko.optim import run as optim_run
from geko.paths import resolve_hipblaslt_path
from geko.utils import parse_devices


def main(
    tuning_dir: str,
    hipblaslt_path: str | None = None,
    devices: str = "0,1,2,3,4,5,6,7",
    n_slots: int = 4,
    client_build_dir: str = "build_tmp",
    retry: bool = True,
    verbose: int = 1,
) -> None:
    """Run GEMM optimization for all configs in a tuning directory across GPUs.

    hipblaslt_path is auto-detected from this script's location (and
    $GEKO_HIPBLASLT_PATH) when left as None.
    """

    _set_log_level(verbose)

    hipblaslt_path = resolve_hipblaslt_path(explicit=hipblaslt_path, anchor=__file__)

    devices_list = parse_devices(devices)

    logger.info("Starting optimization phase...")
    logger.info(f"hipBLASLt path: '{hipblaslt_path}'")
    logger.info(f"Tuning directory: '{tuning_dir}'")
    logger.info(f"Devices: {devices_list}")
    logger.info(f"Jobs per device: {n_slots}")
    logger.info(f"Retry failed jobs: {retry}")

    optim_run(
        hipblaslt_path=Path(hipblaslt_path),
        tuning_dir=Path(tuning_dir),
        devices=devices_list,
        client_build_dir=Path(client_build_dir),
        n_slots=n_slots,
        retry=retry,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optimization with workload-aware scheduling across GPUs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "tuning_dir",
        type=str,
        help="Directory containing tuning configs (TD output)",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of device IDs (e.g. 0,3,4,5)",
    )
    parser.add_argument(
        "--n_slots",
        "-n",
        type=int,
        default=4,
        help="Max concurrent jobs per device",
    )
    parser.add_argument(
        "--client_build_dir",
        "-c",
        type=str,
        default="build_tmp",
        help="tensilelite build directory for prebuilt client",
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
    args = parser.parse_args()

    main(
        args.tuning_dir,
        hipblaslt_path=args.hipblaslt,
        devices=args.devices,
        n_slots=args.n_slots,
        client_build_dir=args.client_build_dir,
        retry=not args.no_retry,
        verbose=args.verbose,
    )
