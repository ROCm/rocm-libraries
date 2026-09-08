#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run `cmake --build` or a test binary with PATH and ROCM_PATH set for tests.

Why this exists: provider tests on Windows link to ROCm DLLs such as
amdhip64_7.dll, MIOpen.dll, and hipblas.dll, and to in-tree DLLs in
<build>/bin. ctest spawns test executables via cmd.exe, which does not
inherit a bash-set PATH in the form the Win32 loader needs. Setting PATH on
Python's subprocess environment propagates through CreateProcessW to
grandchildren (cmake -> ninja -> ctest -> test.exe) and avoids loader failures
such as 0xc0000135.

Why ROCM_PATH is set: providers that JIT-compile kernels through hiprtc need
HIP headers visible at runtime. hiprtc resolves those headers through
ROCM_PATH; if it is unset, runtime compilation can fail with errors such as
"hip/hip_fp16.h file not found".

Why comgr is staged app-local: on Windows the loader resolves amd_comgr.dll
from the .exe's directory, then System32, then PATH. The driver's stale
System32 comgr outranks the wheel's copy on PATH and breaks MIOpen's runtime
kernel JIT (GCN-assembly Winograd solvers are the common failure, but the
mismatch is not limited to them), so this is done on every Windows run.
Copying the wheel's amd_comgr.dll into <build>/bin (the test exe's own
directory) before launch overrides it. See comgr_stage.py.
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


def resolve_rocm_bin(args, rocm_path):
    """Return the ROCm bin directory, deriving it from rocm_path if needed."""
    if args.rocm_bin:
        return args.rocm_bin
    if rocm_path:
        candidate = Path(rocm_path) / "bin"
        if candidate.exists():
            return str(candidate)
    return None


def stage_comgr_if_windows(args, rocm_bin):
    """Stage the wheel's amd_comgr.dll into <build>/bin on Windows (best effort).

    Imported lazily and guarded so a missing helper or staging error never
    blocks the actual test run; the worst case is the pre-existing comgr issue.
    """
    if platform.system() != "Windows" or args.no_stage_comgr or not rocm_bin:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from comgr_stage import stage_comgr

        stage_comgr(rocm_bin, Path(args.build_dir) / "bin", verbose=True)
    except Exception as error:  # noqa: BLE001 - never fail the test run on staging
        print(f"comgr-stage: skipped ({error})", file=sys.stderr)


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", required=True, help="CMake build directory")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target", help="Ninja/cmake target name to build")
    mode.add_argument("--binary", help="Path to a test binary to run directly")
    p.add_argument("--jobs", type=int, help="Parallel job count for --target mode")
    p.add_argument("--gtest-filter", help="gtest filter pattern (--binary mode only)")
    p.add_argument(
        "--rocm-path",
        help="ROCm SDK root. Defaults to /opt/rocm on Linux.",
    )
    p.add_argument("--rocm-bin", help="Windows: ROCm bin directory to prepend to PATH")
    p.add_argument(
        "--extra-bin",
        action="append",
        default=[],
        help="Windows: additional bin directory to prepend. Repeatable.",
    )
    p.add_argument(
        "--no-stage-comgr",
        action="store_true",
        help="Windows: skip staging the wheel's amd_comgr.dll into <build>/bin",
    )
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="--binary mode: extra argument passed through to the binary. "
        "Repeatable. For flag-like values use --extra-arg=--flag (with '='); "
        "for a whole flag+value run put them after a literal `--` instead.",
    )
    p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments after `--` are passed through to the binary (--binary mode).",
    )
    args = p.parse_args()

    if args.gtest_filter and not args.binary:
        p.error("--gtest-filter requires --binary")

    # argparse.REMAINDER captures the leading `--` separator; drop it.
    passthrough = (
        args.passthrough[1:] if args.passthrough[:1] == ["--"] else args.passthrough
    )
    if (args.extra_arg or passthrough) and not args.binary:
        p.error("--extra-arg / `-- <args>` require --binary")

    if args.binary:
        binary = Path(args.binary)
        if not binary.is_file():
            p.error(
                f"--binary must be a single existing executable path, got {args.binary!r}. "
                "Pass flags via --extra-arg or after `--`, not inside --binary."
            )

    env = build_env(args)

    rocm_path = resolve_rocm_path(args.rocm_path, args.rocm_bin)
    stage_comgr_if_windows(args, resolve_rocm_bin(args, rocm_path))

    if args.binary:
        cmd = [args.binary]
        if args.gtest_filter:
            cmd.append(f"--gtest_filter={args.gtest_filter}")
        cmd.extend(args.extra_arg)
        cmd.extend(passthrough)
    else:
        cmd = ["cmake", "--build", args.build_dir, "--target", args.target]
        if args.jobs:
            cmd.extend(["-j", str(args.jobs)])

    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
