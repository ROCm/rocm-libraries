#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Detect or provision the wheel-based ROCm install on Windows.

Outputs key=value lines to stdout so callers can parse them:
    ROCM_PATH=<forward-slash path>
    CLANG_PATH=<forward-slash path>
    GPU_TARGETS=<arch>

All human-readable progress goes to stderr so stdout carries only the
KEY=VALUE lines the calling skill parses.

Provisioning (Windows only): when the ROCm SDK devel wheel is not present in
the target venv, this script creates the venv, pip-installs the ROCm SDK
wheels (multi-arch nightlies, or S3 staging when --sha is given), and runs
`python -m rocm_sdk init`. This is a Python port of the install logic in
projects/hipdnn/scripts/windows/wheel_build_setup.ps1; that PowerShell script
is intentionally left in place for its existing consumers (interactive users
and tools/dnn-benchmarking/setup.ps1, which relies on its in-shell venv
activation and the ROCM_WHEEL_VENV it publishes -- neither of which a child
Python process can do for a parent shell).

Clang is NOT provisioned here: the wheel setup never installed it. If clang is
missing, this script errors and points at windows_build_setup.ps1.

On Linux this is a no-op that echoes any provided overrides.
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_VENV = Path("D:/develop/latest_wheels")
# ROCm SDK devel wheel location relative to a venv root.
SITE_DEVEL_SUBPATH = "Lib/site-packages/_rocm_sdk_devel"
DEFAULT_CLANG_BIN = Path("D:/develop/dist/clang/bin")
DEFAULT_GPU_TARGET = "gfx1151"

# Pinned wheel version for S3 staging installs (--sha). Mirrors
# wheel_build_setup.ps1; update both if the staging version string changes.
WHEEL_VERSION = "7.12.0.dev0"
NIGHTLY_INDEX_URL = "https://rocm.nightlies.amd.com/whl-multi-arch/"
S3_STAGING_BASE = "https://therock-dev-python.s3.amazonaws.com/v2-staging"

# GPU targets with published wheels. Matches wheel_build_setup.ps1.
_VERIFIED_TARGET_RE = r"^(gfx115[0-9]|gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])(-all)?)$"


def log(message):
    print(message, file=sys.stderr)


def emit(rocm_path, clang_path, gpu_targets):
    if rocm_path:
        print(f"ROCM_PATH={Path(rocm_path).as_posix()}")
    if clang_path:
        print(f"CLANG_PATH={Path(clang_path).as_posix()}")
    if gpu_targets:
        print(f"GPU_TARGETS={gpu_targets}")


def resolve_artifact_group(target):
    """Family-suffixed artifact group used in S3 wheel URLs (e.g. gfx942 ->
    gfx942-all; gfx1151 stays gfx1151)."""
    low = target.lower()
    if re.match(r"^gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])-all$", low):
        return target
    if re.match(r"^gfx(120[0-9]|110[0-9]|103[0-9]|90[0-9])$", low):
        return f"{target}-all"
    return target


def pip_install_command(venv_python, gpu_target, sha):
    """Build the `pip install` argv for the ROCm SDK wheels."""
    if sha:
        group = resolve_artifact_group(gpu_target)
        libraries_target = group.lower().replace("-", "_")
        base = f"{S3_STAGING_BASE}/{group}"
        # `%2B` is a URL-encoded '+'; pip fetches these URLs verbatim.
        return [
            str(venv_python),
            "-m",
            "pip",
            "install",
            f"{base}/rocm-{WHEEL_VERSION}%2B{sha}.tar.gz",
            f"{base}/rocm_sdk_core-{WHEEL_VERSION}%2B{sha}-py3-none-win_amd64.whl",
            f"{base}/rocm_sdk_libraries_{libraries_target}-{WHEEL_VERSION}%2B{sha}-py3-none-win_amd64.whl",
            f"{base}/rocm_sdk_devel-{WHEEL_VERSION}%2B{sha}-py3-none-win_amd64.whl",
        ]
    # Multi-arch nightlies select the GPU via a bare `device-<arch>` extra.
    device_target = gpu_target.lower()
    if device_target.endswith("-all"):
        device_target = device_target[: -len("-all")]
    return [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--index-url",
        NIGHTLY_INDEX_URL,
        f"rocm[libraries,devel,device-{device_target}]",
    ]


def run(cmd):
    """Run a provisioning subprocess, routing its output to stderr so stdout
    stays reserved for KEY=VALUE. Returns the exit code."""
    log(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, stdout=sys.stderr, stderr=sys.stderr)


def provision(venv_path, gpu_target, sha, force):
    """Create the venv, install ROCm SDK wheels, and init the SDK. Returns 0 on
    success, nonzero on failure."""
    if not re.match(_VERIFIED_TARGET_RE, gpu_target.lower()):
        log(
            f"WARNING: GPU target '{gpu_target}' is not in the verified list "
            "(gfx115x, gfx120x[-all], gfx110x[-all], gfx103x[-all], gfx90x[-all]). "
            "Wheel install may not work."
        )

    if force and venv_path.exists():
        log(f"Removing existing venv at {venv_path} (--provision always)...")
        shutil.rmtree(venv_path)

    venv_python = venv_path / "Scripts" / "python.exe"
    if not venv_python.exists():
        log(f"Creating Python virtual environment at {venv_path}...")
        rc = run([sys.executable, "-m", "venv", str(venv_path)])
        if rc != 0:
            log("ERROR: failed to create virtual environment.")
            return rc

    source = f"S3 staging (SHA {sha})" if sha else "ROCm multi-arch nightlies"
    log(f"Installing ROCm wheels from {source}...")
    rc = run(pip_install_command(venv_python, gpu_target, sha))
    if rc != 0:
        log("ERROR: failed to install ROCm wheels.")
        return rc

    log("Initializing ROCm SDK (python -m rocm_sdk init)...")
    rc = run([str(venv_python), "-m", "rocm_sdk", "init"])
    if rc != 0:
        log("ERROR: failed to initialize ROCm SDK.")
        return rc

    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root", required=True, help="Path to the rocm-libraries repository root"
    )
    p.add_argument(
        "--venv-path",
        default=str(DEFAULT_VENV),
        help=f"ROCm wheel venv root (Windows). Default: {DEFAULT_VENV.as_posix()}",
    )
    p.add_argument(
        "--rocm-path",
        help="Explicit ROCm SDK devel path. Overrides the venv-derived path.",
    )
    p.add_argument(
        "--clang-path",
        help=f"Clang bin directory (Windows). Default: {DEFAULT_CLANG_BIN.as_posix()}",
    )
    p.add_argument("--gpu-targets", help="Override GPU target")
    p.add_argument(
        "--sha", help="Optional S3 staging SHA; installs pinned staging wheels"
    )
    p.add_argument(
        "--provision",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto: provision if the SDK is missing (default). "
        "always: force a fresh wheel pull. never: validate only.",
    )
    args = p.parse_args()

    if platform.system() != "Windows":
        emit(args.rocm_path, None, args.gpu_targets)
        return 0

    venv_path = Path(args.venv_path)
    rocm_path = (
        Path(args.rocm_path) if args.rocm_path else venv_path / SITE_DEVEL_SUBPATH
    )
    clang_path = Path(args.clang_path) if args.clang_path else DEFAULT_CLANG_BIN
    gpu_targets = args.gpu_targets or DEFAULT_GPU_TARGET

    hipcc = rocm_path / "bin" / "hipcc.exe"

    needs_provision = args.provision == "always" or (
        args.provision == "auto" and not hipcc.exists()
    )
    if needs_provision:
        if args.rocm_path:
            # An explicit devel path can't be provisioned into: the venv layout
            # is not implied. Provisioning only manages the venv-derived tree.
            log(
                f"ERROR: --rocm-path was given but hipcc.exe is missing at {hipcc}. "
                "Provisioning only manages --venv-path; supply a valid --rocm-path "
                "or drop it to let this script provision the wheel venv."
            )
            return 1
        rc = provision(
            venv_path, gpu_targets, args.sha, force=args.provision == "always"
        )
        if rc != 0:
            return rc

    if not hipcc.exists():
        log(f"ERROR: hipcc.exe not found at {hipcc}")
        log("Pass --rocm-path=<your-path> if ROCm is elsewhere, or --provision always.")
        return 1

    if not (clang_path / "clang.exe").exists():
        log(f"ERROR: clang.exe not found at {clang_path}")
        log(
            "Clang is a prerequisite and is not provisioned by this script. "
            "Install it via projects/hipdnn/scripts/windows/windows_build_setup.ps1 "
            "or from the LLVM releases, then pass --clang-path if it is elsewhere."
        )
        return 1

    emit(rocm_path, clang_path, gpu_targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
