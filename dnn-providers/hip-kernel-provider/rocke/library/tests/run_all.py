#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CI / dev entrypoint for the rocKE *library* pytest suite (kernels / builders /
# dispatch), the counterpart to platform/tests/run_all.py. It exists because the
# platform runner cannot collect this tree: platform is a verbatim-copyable tree
# guarded against referencing the library (the one-way library -> platform
# dependency rule), so library CI collection must be driven from here.
#
# Two lanes:
#   * CPU lane (default): every test NOT marked `gpu` -- pure spec/validation/IR
#     lowering + golden-IR byte-stability. No GPU, no comgr, safe on any CI box.
#   * GPU lane (--gpu): additionally runs the `gpu`-marked numeric tests, which
#     self-skip via a device skipif when no gfx942 GPU is present (so `--gpu` on a
#     CPU box is a no-op, not a failure). Use on a gfx942 (MI300X) ROCm runner.
#
# sys.path is handled by tests/conftest.py, so no PYTHONPATH is required. Usage:
#   python rocke/library/tests/run_all.py                 # CPU lane
#   python rocke/library/tests/run_all.py --gpu           # + on-GPU numeric lane
#   python rocke/library/tests/run_all.py --only dense    # -k filter
#   python rocke/library/tests/run_all.py --gpu-only      # ONLY the gpu lane

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE library pytest runner")
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="also run the `gpu`-marked numeric lane (needs a gfx942 GPU; self-skips "
        "if absent)",
    )
    ap.add_argument(
        "--gpu-only",
        action="store_true",
        help="run ONLY the `gpu`-marked lane (implies --gpu)",
    )
    ap.add_argument("--only", default="", help="pytest -k expression to restrict tests")
    ap.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra args passed through to pytest (after --)",
    )
    args = ap.parse_args()

    cmd = [sys.executable, "-m", "pytest", str(TESTS)]
    if args.gpu_only:
        cmd += ["-m", "gpu"]
    elif not args.gpu:
        cmd += ["-m", "not gpu"]  # default CPU lane excludes the GPU numeric tests
    if args.only:
        cmd += ["-k", args.only]
    extra = args.pytest_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd += extra

    print("== rocke library pytest ==")
    print(" ", " ".join(cmd))
    # cwd MUST be the library root, not tests/. `python -m pytest` puts cwd at
    # sys.path[0]; with cwd=tests/ the pre-existing tests/dispatch/attention/
    # package (headers only, from the platform/library split) SHADOWS the real
    # `dispatch.attention` module, and every dispatch test dies at collection with
    # "cannot import name 'AttentionRequest' from 'dispatch.attention'". Rooting at
    # the library instead makes pytest name those modules tests.dispatch.attention.*
    # (tests/__init__.py exists, so importlib walks up to it), which cannot collide.
    # The platform tree sidesteps this by naming its package `dispatch_tests`; the
    # library tree cannot be renamed without churning develop, so it is rooted
    # correctly here instead.
    status = subprocess.run(cmd, cwd=str(TESTS.parent)).returncode
    print("\nRESULT:", "GREEN" if status == 0 else "RED")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
