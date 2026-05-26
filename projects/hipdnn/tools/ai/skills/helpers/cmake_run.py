#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run `cmake --build` (or a test binary) with PATH and ROCM_PATH set for tests.

Why this exists: provider tests on Windows link to ROCm DLLs (amdhip64_7.dll,
MIOpen.dll, hipblas.dll) and to in-tree DLLs in <build>/bin. ctest spawns
test executables via cmd.exe, which does not inherit a bash-set PATH, so
running `cmake --build <target>-unit-check` from bash silently fails the
loader with 0xc0000135. Setting PATH on Python's subprocess env block
propagates through CreateProcessW to grandchildren (cmake -> ninja -> ctest
-> test.exe) and resolves the DLLs without any shell wrapper.

Why ROCM_PATH: providers that JIT-compile device kernels via hiprtc at
runtime (e.g. hip-kernel-provider) need the HIP headers visible on
hiprtc's include search path. hiprtc resolves these via ROCM_PATH; if
it is unset the runtime kernel compile fails with
"hip/hip_fp16.h file not found" and similar.

Two modes:
    --target <name>     Run `cmake --build <build-dir> --target <name>`.
    --binary <path>     Run the test binary directly (use with --gtest-filter).

Usage:
    cmake_run.py --build-dir <path> --target <name> [--jobs N]
                 [--rocm-path <path>] [--rocm-bin <path>]
                 [--extra-bin <path> ...]
    cmake_run.py --build-dir <path> --binary <path> [--gtest-filter <pat>]
                 [--rocm-path <path>] [--rocm-bin <path>]
                 [--extra-bin <path> ...]
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def resolve_rocm_path(explicit, rocm_bin):
    if explicit:
        return explicit
    if rocm_bin:
        return str(Path(rocm_bin).parent)
    if platform.system() != "Windows":
        return "/opt/rocm"
    return None


def build_env(args):
    env = os.environ.copy()
    rocm_path = resolve_rocm_path(args.rocm_path, args.rocm_bin)
    if rocm_path:
        env.setdefault("ROCM_PATH", rocm_path)

    if platform.system() == "Windows":
        bin_dirs = [str(Path(args.build_dir) / "bin")]
        if args.rocm_bin:
            bin_dirs.append(args.rocm_bin)
        bin_dirs.extend(args.extra_bin)
        env["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + env.get("PATH", "")

    return env


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--build-dir", required=True, help="CMake build directory")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target", help="Ninja/cmake target name to build")
    mode.add_argument("--binary", help="Path to a test binary to run directly")
    p.add_argument("--jobs", type=int, help="Parallel job count for --target mode")
    p.add_argument(
        "--gtest-filter",
        help="gtest filter pattern (--binary mode only)",
    )
    p.add_argument(
        "--rocm-path",
        help="ROCm SDK root (sets ROCM_PATH in env). "
        "Defaults to /opt/rocm on Linux; derived from --rocm-bin's parent on Windows.",
    )
    p.add_argument("--rocm-bin", help="Windows: ROCm bin directory to prepend to PATH")
    p.add_argument(
        "--extra-bin",
        action="append",
        default=[],
        help="Windows: additional bin directory to prepend (repeatable)",
    )
    args = p.parse_args()

    if args.gtest_filter and not args.binary:
        p.error("--gtest-filter requires --binary")

    env = build_env(args)

    if args.binary:
        cmd = [args.binary]
        if args.gtest_filter:
            cmd.append(f"--gtest_filter={args.gtest_filter}")
    else:
        cmd = ["cmake", "--build", args.build_dir, "--target", args.target]
        if args.jobs:
            cmd.extend(["-j", str(args.jobs)])

    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
