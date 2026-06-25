#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The engine definition-of-done in one command: build the C++ engine fresh and
# prove its LLVM-IR emission is byte-identical to the Python engine across every
# kernel family. A green run means the dual-backend contract still holds.
#
# Cross-platform (Windows + Linux) replacement for the legacy check_byte_identity.sh.
# All paths are derived relative to this file so the rocKE/ tree stays copy-able.
#
# Usage:
#   python rocKE/tools/check_byte_identity.py [--ir] [--only SUBSTR]
#       [--build-root DIR] [--ref-pyroot DIR] [--ref-shim DIR]

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[0] if HERE.name != "tools" else HERE.parent  # tools -> rocKE
RUN_DIFF = ROCKE / "tests" / "instances" / "differential" / "run_diff.py"


def _cxx() -> str | None:
    for c in ("c++", "clang++", "g++", "cl"):
        p = shutil.which(c)
        if p:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE byte-identity gate")
    ap.add_argument(
        "--ir", action="store_true", help="also run the IR canonical diff (diagnostic)"
    )
    ap.add_argument(
        "--only",
        default="",
        help="restrict to families containing SUBSTR (comma-separated)",
    )
    ap.add_argument(
        "--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_verify")
    )
    ap.add_argument("--ref-pyroot", default=os.environ.get("ROCKE_REF_PYROOT", ""))
    ap.add_argument("--ref-shim", default=os.environ.get("ROCKE_REF_SHIM", ""))
    args = ap.parse_args()

    build_root = Path(args.build_root)
    archive = build_root / "librocke_core.a"

    print("== building engine archive (fresh) ==")
    print(f"   source : {ROCKE}")
    print(f"   build  : {build_root}")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROCKE),
            "-B",
            str(build_root),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_root),
            "--target",
            "rocke_core",
            "-j",
            str(os.cpu_count() or 1),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if not archive.exists():
        print(f"FATAL: archive not produced: {archive}", file=sys.stderr)
        return 1
    print(f"   archive: {archive}")

    # Freshness/provenance: print the build-id stamped into the archive. Off the
    # emission path, so this never affects the .ll contract. Best-effort.
    cxx = _cxx()
    if cxx and cxx.endswith("cl") is False:
        probe_src = build_root / "_build_id_probe.cpp"
        probe_bin = build_root / "_build_id_probe"
        probe_src.write_text(
            'extern "C" const char* rocke_build_id(void);\n'
            'extern "C" const char* rocke_engine_version(void);\n'
            "#include <cstdio>\n"
            'int main(){ printf("%s %s\\n", rocke_build_id(), rocke_engine_version()); return 0; }\n'
        )
        try:
            subprocess.run(
                [
                    cxx,
                    "-std=c++20",
                    str(probe_src),
                    str(archive),
                    "-lm",
                    "-o",
                    str(probe_bin),
                ],
                check=True,
                stderr=subprocess.DEVNULL,
            )
            out = subprocess.run([str(probe_bin)], capture_output=True, text=True)
            print(f"   build-id: {out.stdout.strip()}")
        except (subprocess.CalledProcessError, OSError):
            print("   build-id: (probe unavailable)")

    only_args = ["--only", args.only] if args.only else []
    ref_args = []
    if args.ref_pyroot:
        ref_args += ["--pyroot", args.ref_pyroot]
    if args.ref_shim:
        ref_args += ["--shim", args.ref_shim]

    def run_gate(label: str, extra: list[str]) -> int:
        print(f"\n== differential gate: {label} ==")
        proc = subprocess.run(
            [
                sys.executable,
                str(RUN_DIFF),
                "--archive",
                str(archive),
                *only_args,
                *ref_args,
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        if "COMPILE_FAIL" in (proc.stdout + proc.stderr):
            print(f"GATE FAIL ({label}): a family failed to compile.", file=sys.stderr)
            return 1
        return proc.returncode

    status = run_gate("LLVM-IR (the contract)", ["--mode", "ll"])
    if args.ir:
        run_gate(
            "IR canonical (diagnostic)", ["--mode", "ir", "--canonical"]
        )  # non-gating

    print()
    if status == 0:
        print(
            "RESULT: GREEN - engine builds and .ll emission is byte-identical to Python."
        )
    else:
        print(
            "RESULT: RED - see the mismatching families above. The two engines disagree."
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
