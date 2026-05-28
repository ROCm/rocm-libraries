#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Detect or set up the wheel-based ROCm install on Windows.

Outputs key=value lines to stdout so callers can parse them:
    ROCM_PATH=<forward-slash path>
    CLANG_PATH=<forward-slash path>
    GPU_TARGETS=<arch>

On Linux this is a no-op that just echoes any provided overrides.

Exits non-zero on failure with diagnostics on stderr.
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path


DEFAULT_VENV = Path("D:/develop/latest_wheels")
DEFAULT_ROCM_DEVEL = DEFAULT_VENV / "Lib/site-packages/_rocm_sdk_devel"
DEFAULT_CLANG_BIN = Path("D:/develop/dist/clang/bin")
DEFAULT_GPU_TARGET = "gfx1151"


def emit(rocm_path, clang_path, gpu_targets):
    if rocm_path:
        print(f"ROCM_PATH={Path(rocm_path).as_posix()}")
    if clang_path:
        print(f"CLANG_PATH={Path(clang_path).as_posix()}")
    if gpu_targets:
        print(f"GPU_TARGETS={gpu_targets}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root", required=True, help="Path to the rocm-libraries repository root"
    )
    p.add_argument("--rocm-path", help="Override ROCm SDK devel path")
    p.add_argument("--clang-path", help="Override clang bin directory (Windows only)")
    p.add_argument("--gpu-targets", help="Override GPU target")
    p.add_argument("--sha", help="Optional S3 staging SHA passed to wheel setup")
    args = p.parse_args()

    if platform.system() != "Windows":
        emit(args.rocm_path, None, args.gpu_targets)
        return 0

    rocm_path = Path(args.rocm_path) if args.rocm_path else DEFAULT_ROCM_DEVEL
    clang_path = Path(args.clang_path) if args.clang_path else DEFAULT_CLANG_BIN
    gpu_targets = args.gpu_targets or DEFAULT_GPU_TARGET

    if not args.rocm_path:
        hipcc = rocm_path / "bin" / "hipcc.exe"
        if not hipcc.exists():
            script = (
                Path(args.repo_root)
                / "projects/hipdnn/scripts/windows/wheel_build_setup.ps1"
            )
            if not script.exists():
                print(f"ERROR: wheel setup script not found: {script}", file=sys.stderr)
                return 1
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
            if args.sha:
                cmd.extend(["-SHA", args.sha])
            if args.gpu_targets:
                cmd.extend(["-GpuTarget", args.gpu_targets])
            print(f"Running wheel setup: {' '.join(cmd)}", file=sys.stderr)
            r = subprocess.run(cmd, text=True)
            if r.returncode != 0:
                print(
                    f"ERROR: wheel setup failed (exit {r.returncode})", file=sys.stderr
                )
                return 1
            if not hipcc.exists():
                print(
                    f"ERROR: wheel setup ran but hipcc.exe still missing at {hipcc}",
                    file=sys.stderr,
                )
                return 1

    if not (clang_path / "clang.exe").exists():
        print(f"ERROR: clang.exe not found at {clang_path}", file=sys.stderr)
        print("Pass --clang-path=<your-path> if clang is elsewhere.", file=sys.stderr)
        return 1

    emit(rocm_path, clang_path, gpu_targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
