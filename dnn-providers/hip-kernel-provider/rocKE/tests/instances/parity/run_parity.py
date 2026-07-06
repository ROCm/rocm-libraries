#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_parity.py -- cross-platform (Windows + Linux) replacement for run_parity.sh.
# Builds the C micro-kernel emitter (emit.c) against the engine archive and, for
# each micro kernel, sha256-compares the C-emitted .ll against the Python
# reference (emit.py). PASS = all byte-identical.
#
# The gemm / ir_serialize parity runners that used to live here are subsumed by
# the differential gate (tools/check_byte_identity.py --only gemm, and run_diff
# --mode ir); they were removed to avoid duplicate harnesses.
#
# Usage:
#   python run_parity.py [--archive PATH] [--build-root DIR]
# All paths are derived relative to this file so the tree is copy-able.

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[2]  # parity -> instances -> tests -> rocKE
INCLUDE = ROCKE / "Cpp" / "include"
PYROOT = ROCKE / "Python"
KERNELS = ["scalar", "memory", "forloop", "vector"]


def _cxx() -> str:
    for c in ("c++", "clang++", "g++"):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("no C++ compiler (c++/clang++/g++) found on PATH")


def _ensure_archive(build_root: Path) -> Path:
    archive = build_root / "librocke_core.a"
    if not archive.exists():
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
    return archive


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="micro-kernel C-vs-Python .ll parity")
    ap.add_argument("--archive", default="")
    ap.add_argument(
        "--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_verify")
    )
    args = ap.parse_args()

    out = Path(tempfile.gettempdir()) / "rocke_parity"
    out.mkdir(parents=True, exist_ok=True)
    archive = (
        Path(args.archive) if args.archive else _ensure_archive(Path(args.build_root))
    )
    if not archive.exists():
        print(f"FATAL: engine archive not found: {archive}", file=sys.stderr)
        return 1

    binexe = out / ("emit_c.exe" if os.name == "nt" else "emit_c")
    print(">> compiling C micro-kernel emitter")
    compile_cmd = [
        _cxx(),
        "-std=c++20",
        "-I",
        str(INCLUDE),
        str(HERE / "emit.c"),
        str(archive),
        "-lm",
        "-o",
        str(binexe),
    ]
    cc = subprocess.run(compile_cmd, capture_output=True, text=True)
    if cc.returncode != 0:
        print("C emitter compile FAILED:\n" + cc.stderr, file=sys.stderr)
        return 1

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PYROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    rc = 0
    for k in KERNELS:
        c_ll = out / f"c_{k}.ll"
        py_ll = out / f"py_{k}.ll"
        c_ll.write_bytes(subprocess.run([str(binexe), k], capture_output=True).stdout)
        py_ll.write_bytes(
            subprocess.run(
                [sys.executable, str(HERE / "emit.py"), k], capture_output=True, env=env
            ).stdout
        )
        cs, ps = _sha(c_ll), _sha(py_ll)
        if cs == ps:
            print(f"PASS  {k}  {cs}")
        else:
            print(f"FAIL  {k}  C={cs} PY={ps}")
            rc = 1
    print(">> ALL PARITY CHECKS PASSED" if rc == 0 else ">> PARITY FAILURES PRESENT")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
