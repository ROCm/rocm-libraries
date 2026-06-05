#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Provision a wheel-based ROCm install into a Python venv (cross-platform).

This is the cross-platform Python port of
``projects/hipdnn/scripts/windows/wheel_build_setup.ps1`` so the same
wheel-pull workflow can be driven from the build skill on Windows or Linux.

What it does:
  1. Creates (or reuses) a Python virtual environment.
  2. Installs the ROCm SDK wheels into it, from either:
       - the ROCm nightlies index (default), or
       - S3 staging pinned to a specific build SHA (``--sha``).
  3. Runs ``rocm-sdk init`` to materialize the SDK in the venv.
  4. Prints ``KEY=VALUE`` lines so callers can configure CMake:
       ROCM_PATH=<_rocm_sdk_devel dir, forward slashes>
       ROCM_BIN=<ROCM_PATH>/bin
       GPU_TARGETS=<arch>
       CLANG_PATH=<clang bin dir>   (only when provided / on Windows)

By default an existing venv is reused untouched. Pass ``--pull`` to delete and
reinstall it with fresh wheels. The emitted KEY=VALUE lines compose with
``windows_rocm_setup.py`` and ``cmake_run.py`` from these skills.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


IS_WINDOWS = platform.system() == "Windows"

# Per-worktree venv directory name, created at the repository (worktree) root.
# Each git worktree is its own root, so this isolates wheels per worktree: a
# --pull in one worktree never invalidates a build in another. It is gitignored.
WHEEL_DIR_NAME = ".rocm_wheels"

# Fallback venv when no --repo-root/--venv-path is given. Mirrors
# wheel_build_setup.ps1 on Windows; Linux uses the user's home so it works
# without write access to D:\.
DEFAULT_VENV = (
    Path("D:/develop/latest_wheels")
    if IS_WINDOWS
    else (Path.home() / "rocm_wheels_venv")
)
DEFAULT_CLANG_BIN = Path("D:/develop/dist/clang/bin") if IS_WINDOWS else None
DEFAULT_GPU_TARGET = "gfx1151"


def default_venv_path(repo_root):
    """Per-worktree venv at <repo-root>/.rocm_wheels, or the global fallback."""
    if repo_root:
        return Path(repo_root) / WHEEL_DIR_NAME
    return DEFAULT_VENV


# Per-architecture wheel family. Selects the nightlies index and S3 staging
# bucket; pip picks the OS-correct wheel (win_amd64 vs linux) by platform tag.
DEFAULT_FAMILY = "gfx110X-all"
DEFAULT_S3_VERSION = "7.12.0.dev0"
NIGHTLIES_BASE = "https://rocm.nightlies.amd.com/v2"
S3_STAGING_BASE = "https://therock-dev-python.s3.amazonaws.com/v2-staging"


def venv_python(venv_path):
    sub = "Scripts" if IS_WINDOWS else "bin"
    exe = "python.exe" if IS_WINDOWS else "python"
    return venv_path / sub / exe


def venv_console_script(venv_path, name):
    sub = "Scripts" if IS_WINDOWS else "bin"
    exe = f"{name}.exe" if IS_WINDOWS else name
    return venv_path / sub / exe


def run(cmd, **kwargs):
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}", file=sys.stderr)
    return subprocess.run(cmd, check=True, **kwargs)


def create_venv(venv_path, pull):
    if venv_path.exists():
        if not pull:
            print(f"Reusing existing venv at {venv_path}", file=sys.stderr)
            return False
        print(f"Removing existing venv at {venv_path} (--pull)", file=sys.stderr)
        shutil.rmtree(venv_path)
    print(f"Creating venv at {venv_path}", file=sys.stderr)
    run([sys.executable, "-m", "venv", str(venv_path)])
    return True


def install_wheels(py, family, sha, s3_version, index_url):
    pip = [str(py), "-m", "pip", "install"]
    if sha:
        base = f"{S3_STAGING_BASE}/{family}"
        ver = f"{s3_version}%2B{sha}"
        print(f"Installing ROCm wheels from S3 staging (SHA {sha})", file=sys.stderr)
        run(
            pip
            + [
                f"{base}/rocm-{ver}.tar.gz",
                f"{base}/rocm_sdk_core-{ver}-py3-none-win_amd64.whl",
                f"{base}/rocm_sdk_libraries_{family.replace('-', '_')}-{ver}"
                "-py3-none-win_amd64.whl",
                f"{base}/rocm_sdk_devel-{ver}-py3-none-win_amd64.whl",
            ]
        )
    else:
        url = index_url or f"{NIGHTLIES_BASE}/{family}/"
        print(f"Installing ROCm wheels from nightlies: {url}", file=sys.stderr)
        run(pip + ["--index-url", url, "rocm[libraries,devel]"])


def rocm_sdk_init(venv_path, py):
    script = venv_console_script(venv_path, "rocm-sdk")
    cmd = (
        [str(script), "init"]
        if script.exists()
        else [str(py), "-m", "rocm_sdk", "init"]
    )
    print("Initializing ROCm SDK", file=sys.stderr)
    run(cmd)


def locate_devel(py):
    """Return the _rocm_sdk_devel directory inside the venv, OS-independent."""
    result = subprocess.run(
        [
            str(py),
            "-c",
            "import _rocm_sdk_devel, os; "
            "print(os.path.dirname(_rocm_sdk_devel.__file__))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def emit(devel, gpu_targets, clang_path):
    rocm_path = devel.as_posix()
    print(f"ROCM_PATH={rocm_path}")
    print(f"ROCM_BIN={(devel / 'bin').as_posix()}")
    print(f"GPU_TARGETS={gpu_targets}")
    if clang_path:
        print(f"CLANG_PATH={Path(clang_path).as_posix()}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--repo-root",
        help="Worktree/repo root; default venv becomes <repo-root>/.rocm_wheels",
    )
    p.add_argument(
        "--venv-path",
        help="Explicit venv location (overrides the per-worktree default)",
    )
    p.add_argument(
        "--pull",
        action="store_true",
        help="Remove and reinstall the venv with fresh wheels",
    )
    p.add_argument("--sha", help="Pin wheels to an S3 staging build SHA")
    p.add_argument(
        "--s3-version",
        default=DEFAULT_S3_VERSION,
        help=f"Version stem for S3 staging URLs (default {DEFAULT_S3_VERSION})",
    )
    p.add_argument(
        "--rocm-family",
        default=DEFAULT_FAMILY,
        help=f"Wheel architecture family (default {DEFAULT_FAMILY})",
    )
    p.add_argument("--index-url", help="Override the nightlies pip index URL")
    p.add_argument(
        "--gpu-targets",
        default=DEFAULT_GPU_TARGET,
        help=f"GPU target emitted for CMake (default {DEFAULT_GPU_TARGET})",
    )
    p.add_argument(
        "--clang-path",
        default=str(DEFAULT_CLANG_BIN) if DEFAULT_CLANG_BIN else None,
        help="Clang bin directory emitted as CLANG_PATH (Windows toolchain)",
    )
    p.add_argument(
        "--no-init",
        action="store_true",
        help="Skip 'rocm-sdk init' (assume already initialized)",
    )
    args = p.parse_args()

    venv_path = (
        Path(args.venv_path) if args.venv_path else default_venv_path(args.repo_root)
    )
    fresh = create_venv(venv_path, args.pull)
    py = venv_python(venv_path)
    if not py.exists():
        print(f"ERROR: venv python not found at {py}", file=sys.stderr)
        return 1

    if fresh:
        install_wheels(py, args.rocm_family, args.sha, args.s3_version, args.index_url)
        if not args.no_init:
            rocm_sdk_init(venv_path, py)

    try:
        devel = locate_devel(py)
    except subprocess.CalledProcessError:
        print(
            "ERROR: ROCm SDK not importable in the venv. "
            "Re-run with --pull to (re)install the wheels.",
            file=sys.stderr,
        )
        return 1

    emit(devel, args.gpu_targets, args.clang_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
