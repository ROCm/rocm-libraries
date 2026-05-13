#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run `cmake --build` in PowerShell with PATH set for Windows DLL resolution.

On Linux this just runs cmake --build directly without the PowerShell wrapper.

Why a wrapper: provider tests on Windows link to ROCm DLLs (amdhip64_7.dll,
MIOpen.dll, hipblas.dll) and to in-tree DLLs in <build>/bin. ctest spawns
test executables via cmd.exe, which does not inherit a bash-set PATH.
Wrapping the cmake invocation in PowerShell fixes the DLL search path.

Usage:
    pwsh_cmake_run.py --build-dir <path> --target <name> [--jobs N] \\
                      [--rocm-bin <path>] [--extra-bin <path> ...]
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def to_windows_path(p):
    return str(Path(p)).replace("/", "\\")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--build-dir", required=True, help="CMake build directory")
    p.add_argument("--target", required=True, help="Ninja/cmake target name to build")
    p.add_argument("--jobs", type=int, help="Parallel job count (-j)")
    p.add_argument("--rocm-bin", help="Windows: ROCm bin directory to prepend to PATH")
    p.add_argument(
        "--extra-bin",
        action="append",
        default=[],
        help="Windows: additional bin directory to prepend (repeatable)",
    )
    args = p.parse_args()

    if platform.system() != "Windows":
        cmd = ["cmake", "--build", args.build_dir, "--target", args.target]
        if args.jobs:
            cmd.extend(["-j", str(args.jobs)])
        return subprocess.call(cmd)

    bin_dirs = [to_windows_path(f"{args.build_dir}/bin")]
    if args.rocm_bin:
        bin_dirs.append(to_windows_path(args.rocm_bin))
    for extra in args.extra_bin:
        bin_dirs.append(to_windows_path(extra))

    path_prefix = ";".join(bin_dirs) + ";"

    cmake_parts = [
        "cmake",
        "--build",
        to_windows_path(args.build_dir),
        "--target",
        args.target,
    ]
    if args.jobs:
        cmake_parts.extend(["-j", str(args.jobs)])
    cmake_str = " ".join(cmake_parts)

    pwsh_inner = f"$env:PATH = '{path_prefix}' + $env:PATH; {cmake_str}"
    return subprocess.call(["powershell", "-Command", pwsh_inner])


if __name__ == "__main__":
    sys.exit(main())
