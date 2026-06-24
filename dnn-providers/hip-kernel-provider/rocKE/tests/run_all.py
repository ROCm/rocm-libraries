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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROCKE = Path(__file__).resolve().parents[1]  # tests -> rocKE
TESTS = ROCKE / "tests"
TOOLS = ROCKE / "tools"
IR_ARTIFACT_DIFF = TESTS / "instances" / "differential" / "ir_artifact_diff.py"
IR_LOWER_CLI_SRC = TESTS / "core" / "ir_lower_cli.cpp"
CPP_INCLUDE = ROCKE / "Cpp" / "include"

# Files that may reference an absolute repo path or a path that escapes rocKE/
# break the verbatim-copy contract. Enforce on code/build files only (docs are
# exempt). A clean run is required before the tree is dropped into another repo.
_GUARD_SUFFIXES = {".py", ".cmake", ".toml", ".ini", ".sh", ".cfg"}
_GUARD_NAMES = {"CMakeLists.txt"}
_GUARD_SKIP_DIRS = {".git", "__pycache__", "build", "dsl_docs", "examples"}
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
                    violations.append(f"{path.relative_to(ROCKE)}:{i}: {line.strip()[:100]}")
    if violations:
        print("RELATIVE-PATH GUARD: FAIL - absolute/repo paths found under rocKE/:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("RELATIVE-PATH GUARD: PASS")
    return 0


def _cxx() -> str | None:
    for c in ("c++", "clang++", "g++"):
        p = shutil.which(c)
        if p:
            return p
    return None


def arch_sweep(archs: str, build_root: Path) -> int:
    """Multi-arch byte-identity for the COMMON families: build the engine +
    ir_lower_cli, then run the IR-artifact differential at each requested arch
    (e.g. gfx942,gfx1151) so we are not blindsided by only checking gfx950."""
    print(f"\n== multi-arch byte-identity sweep (archs={archs}) ==")
    archive = build_root / "libckc_core.a"
    if not archive.exists():
        subprocess.run(["cmake", "-S", str(ROCKE), "-B", str(build_root),
                        "-DCMAKE_BUILD_TYPE=Release"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["cmake", "--build", str(build_root), "--target", "ckc_core",
                        "-j", str(os.cpu_count() or 1)], check=True, stdout=subprocess.DEVNULL)
    cxx = _cxx()
    if not cxx:
        print("  no C++ compiler found; skipping arch sweep")
        return 0
    cli = build_root / "ir_lower_cli"
    comp = subprocess.run([cxx, "-std=c++20", "-I", str(CPP_INCLUDE), str(IR_LOWER_CLI_SRC),
                           str(archive), "-lm", "-o", str(cli)], capture_output=True, text=True)
    if comp.returncode != 0:
        print("  ir_lower_cli build FAILED:\n" + comp.stderr)
        return 1
    return subprocess.run([sys.executable, str(IR_ARTIFACT_DIFF),
                           "--cli", str(cli), "--arch", archs]).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE test/validation runner")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-pytest", action="store_true")
    ap.add_argument("--only", default="", help="restrict byte-identity gate to families containing SUBSTR")
    ap.add_argument("--build-root", default=str(Path(tempfile.gettempdir()) / "ckc_verify"))
    ap.add_argument("--arch-sweep", default="",
                    help="comma-separated archs to also run the common-family byte-identity "
                         "sweep at (e.g. gfx942,gfx1151). Off by default.")
    args = ap.parse_args()

    status = 0

    if not args.no_guard:
        status |= relative_path_guard()

    if not args.no_gate:
        print("\n== byte-identity gate ==")
        gate = [sys.executable, str(TOOLS / "check_byte_identity.py"), "--build-root", args.build_root]
        if args.only:
            gate += ["--only", args.only]
        status |= subprocess.run(gate).returncode

    if args.arch_sweep:
        status |= arch_sweep(args.arch_sweep, Path(args.build_root))

    if not args.no_pytest:
        print("\n== pytest ==")
        status |= subprocess.run([sys.executable, "-m", "pytest", str(TESTS)], cwd=str(TESTS)).returncode

    build_root = Path(args.build_root)
    # Only ctest when the C++ test binaries were actually built (the byte-identity
    # gate builds just `ckc_core`, so a gate-only build dir has the registration
    # file but no test executables -> running ctest there would spuriously fail).
    test_bins = [build_root / "tests" / b for b in
                 ("ckc_smoke", "ckc_ir_serialize_roundtrip", "ckc_tiled_attention_2d_reentrancy")]
    if (build_root / "CTestTestfile.cmake").exists() and any(b.exists() for b in test_bins):
        print("\n== ctest ==")
        status |= subprocess.run(["ctest", "--output-on-failure", "--no-tests=ignore"],
                                 cwd=str(build_root)).returncode

    print("\nRESULT:", "GREEN" if status == 0 else "RED")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
