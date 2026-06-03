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

import argparse

from geko.constants import SUPPORTED_ARCH
from geko.pipeline import run_configure
from geko.utils import HIPBLASLT_PATH


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
        help="tensilelite backend. Ductile (GA) or Tensile",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        default="workdir",
        help="Working directory for intermediate files and configurations",
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

    run_configure(
        HIPBLASLT_PATH,
        args.gemm_log,
        device=args.device,
        keep_thr=args.keep_thr,
        arch=args.architecture,
        backend=args.backend,
        workdir=args.workdir,
        verbose=args.verbose,
        bench_freq=args.bench_freq,
    )


if __name__ == "__main__":
    main()
