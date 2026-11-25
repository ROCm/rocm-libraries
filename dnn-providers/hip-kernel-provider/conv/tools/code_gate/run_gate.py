#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The conv code-size gate end to end: build the HIPRTC hook, run the compile census (every
catalog config the census graphs admit, on every catalog architecture, nothing compiled), then
compile each distinct config offline and check it (check_code_size.py). No GPU is needed.

    run_gate.py --tests <hip_kernel_provider_tests> --rocm <rocm> -j <cores> [--work <dir>]

Exit status 0 only when every config passes the gate.
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROVIDER = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CONV_TREE = os.path.join(
    PROVIDER, "src", "engines", "kernel_ingestor_engine", "descriptors", "conv"
)
COVERAGE_MAP = os.path.join(
    PROVIDER,
    "src",
    "integration_tests",
    "kernel_ingestor_engine",
    "hip_kernel_provider_conv_coverage_map.json",
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tests", required=True, help="the hip_kernel_provider_tests binary"
    )
    p.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm"))
    p.add_argument("--archs", help="comma-separated; default: the census's list")
    p.add_argument("-j", "--jobs", type=int, default=2)
    p.add_argument("--work", help="keep the hook, records and reports here")
    args = p.parse_args()

    work = args.work or tempfile.mkdtemp(prefix="hkp-code-gate-")
    os.makedirs(work, exist_ok=True)
    llvm = os.path.join(args.rocm, "lib", "llvm", "bin")
    if not os.path.isdir(llvm):
        llvm = os.path.join(args.rocm, "llvm", "bin")
    hook = os.path.join(work, "librtc_hook.so")
    subprocess.run(
        [
            os.path.join(llvm, "clang"),
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            hook,
            os.path.join(HERE, "..", "rdna_generic", "rtc_hook.c"),
            "-ldl",
            "-lpthread",
        ],
        check=True,
    )

    records = os.path.join(work, "rtc.jsonl")
    if os.path.exists(records):
        os.remove(records)
    env = dict(os.environ)
    env.update(
        {
            "LD_PRELOAD": hook,
            "HKP_RTC_DRY": "1",
            "HKP_RTC_LOG": records,
            "HIPDNN_DISABLE_CACHE": "1",
            "HKP_CONV_CENSUS_TREE": CONV_TREE,
            "HKP_CONV_CENSUS_MAP": COVERAGE_MAP,
            "HKP_CONV_CENSUS_OUT": os.path.join(work, "census.jsonl"),
        }
    )
    if args.archs:
        env["HKP_CONV_CENSUS_ARCHS"] = args.archs
    census = subprocess.run(
        [args.tests, "--gtest_filter=TestConvCompileCensus.*"], env=env
    )
    if census.returncode != 0 or not os.path.exists(records):
        print("the compile census failed", file=sys.stderr)
        return 1

    return subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "check_code_size.py"),
            "--configs",
            records,
            "--kernels-root",
            CONV_TREE,
            "--rocm",
            args.rocm,
            "--allow-compile-errors",
            "-j",
            str(args.jobs),
            "--report",
            os.path.join(work, "gate.csv"),
        ]
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
