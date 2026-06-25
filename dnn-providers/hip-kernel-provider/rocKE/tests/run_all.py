#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Cross-platform (Windows + Linux) CI/parent entrypoint for the rocKE engine.
# One command runs: (1) the relative-path contract guard, (2) the byte-identity
# gate, (3) the pytest suite, (4) ctest if a build dir exists. All paths are
# derived relative to this file so the rocKE/ tree is copy-able verbatim.
#
# Usage:
#   python rocKE/tests/run_all.py [--no-guard] [--no-gate] [--no-pytest]
#       [--only SUBSTR] [--build-root DIR]

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROCKE = Path(__file__).resolve().parents[1]  # tests -> rocKE
TESTS = ROCKE / "tests"
TOOLS = ROCKE / "tools"

# Files that may reference an absolute repo path or a path that escapes rocKE/
# break the verbatim-copy contract. Enforce on code/build files only (docs are
# exempt). A clean run is required before the tree is dropped into another repo.
_GUARD_SUFFIXES = {".py", ".cmake", ".toml", ".ini", ".sh", ".cfg"}
_GUARD_NAMES = {"CMakeLists.txt"}
_GUARD_SKIP_DIRS = {".git", ".venv", "__pycache__", "build", "dsl_docs", "examples"}
_FORBIDDEN = [
    re.compile(r"/workspace\b"),
    re.compile(r"rocm-libraries(?:-[a-z-]+)?/"),
    re.compile(r"projects/composablekernel"),
    re.compile(r"dnn-providers/"),
]


def relative_path_guard() -> int:
    """Fail if any code/build file under rocKE/ references an absolute repo path."""
    violations: list[str] = []
    for path in ROCKE.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _GUARD_SKIP_DIRS for part in path.relative_to(ROCKE).parts):
            continue
        if path.suffix not in _GUARD_SUFFIXES and path.name not in _GUARD_NAMES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # this guard file defines the patterns
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat in _FORBIDDEN:
                if pat.search(line):
                    violations.append(
                        f"{path.relative_to(ROCKE)}:{i}: {line.strip()[:100]}"
                    )
    if violations:
        print("RELATIVE-PATH GUARD: FAIL - absolute/repo paths found under rocKE/:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("RELATIVE-PATH GUARD: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE test/validation runner")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-pytest", action="store_true")
    ap.add_argument(
        "--only",
        default="",
        help="restrict byte-identity gate to families containing SUBSTR",
    )
    ap.add_argument(
        "--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_verify")
    )
    args = ap.parse_args()

    status = 0

    if not args.no_guard:
        status |= relative_path_guard()

    if not args.no_gate:
        print("\n== byte-identity gate ==")
        gate = [
            sys.executable,
            str(TOOLS / "check_byte_identity.py"),
            "--build-root",
            args.build_root,
        ]
        if args.only:
            gate += ["--only", args.only]
        status |= subprocess.run(gate).returncode

    if not args.no_pytest:
        print("\n== pytest ==")
        status |= subprocess.run(
            [sys.executable, "-m", "pytest", str(TESTS)], cwd=str(TESTS)
        ).returncode

    build_root = Path(args.build_root)
    # Only ctest when the CTest-registered binaries were actually built (the
    # byte-identity gate builds just `rocke_core`, so a gate-only build dir has the
    # registration file but no test executables -> running ctest there would
    # spuriously fail). Gate on the registered tests only; `rocke_smoke` is an
    # optional build-only target (not an add_test target) so it is not a signal.
    test_bins = [
        build_root / "tests" / b
        for b in ("rocke_ir_serialize_roundtrip", "rocke_tiled_attention_2d_reentrancy")
    ]
    if (build_root / "CTestTestfile.cmake").exists() and any(
        b.exists() for b in test_bins
    ):
        print("\n== ctest ==")
        status |= subprocess.run(
            ["ctest", "--output-on-failure", "--no-tests=ignore"], cwd=str(build_root)
        ).returncode

    print("\nRESULT:", "GREEN" if status == 0 else "RED")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
