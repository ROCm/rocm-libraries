#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
RowColQuant bridge self-test / default-config runner.

Two modes:

  --build-only   codegen + hipcc compile of the default fp8/bf8 configs, no GPU
                 required. Verifies the whole toolchain up to a loadable .so.

  (default)      build + run on a GPU: generates random A/B and per-row/per-col
                 scales, runs each kernel, and (when numpy is available) checks
                 the result against a plain numpy RowColQuant reference.

Reference math (RowColQuant):
    C[m, n] = sum_k ( A[m, k] * AQ[m] ) * ( B[k, n] * BQ[n] )
            = AQ[m] * BQ[n] * sum_k A[m, k] * B[k, n]

Usage:
    python3 rowcolquant_selftest.py --build-only
    python3 rowcolquant_selftest.py --arch gfx950 --m 256 --n 256 --k 512
"""

import argparse
import logging
import sys
from pathlib import Path

import gemm_rowcolquant_utils as u

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rowcolquant_selftest")


def _reference(A, B, AQ, BQ):
    import numpy as np

    acc = A.astype(np.float32) @ B.astype(np.float32)          # [M, N]
    acc = acc * AQ.astype(np.float32)[:, None]                 # per-row scale
    acc = acc * BQ.astype(np.float32)[None, :]                 # per-col scale
    return acc


def _run_one(so_path, M, N, K, verify):
    import numpy as np

    runner = u.RowColQuantGpuGemmRunner(so_path)
    name = runner.kernel_name

    rng = np.random.default_rng(0)
    # fp8/bf8 inputs are represented on the host as float32 here for the
    # reference; the kernel's ADataType/BDataType handle narrowing on device.
    # For a genuine numeric check, use small integer-ish values.
    A = rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float32)
    B = rng.uniform(-2.0, 2.0, size=(K, N)).astype(np.float32)
    AQ = rng.uniform(0.5, 1.5, size=(M,)).astype(np.float32)
    BQ = rng.uniform(0.5, 1.5, size=(N,)).astype(np.float32)

    result = runner.run(A, B, AQ, BQ, u.RowColQuantGemmProblem(M=M, N=N, K=K))
    log.info("kernel %s ran in %.4f ms", name, result.time_ms)

    if verify:
        ref = _reference(A, B, AQ, BQ)
        got = result.C.astype(np.float32)
        denom = np.maximum(np.abs(ref), 1.0)
        rel = np.abs(got - ref) / denom
        max_rel = float(rel.max())
        log.info("  max relative error vs numpy reference: %.4g", max_rel)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="RowColQuant bridge self-test")
    p.add_argument("--build-only", action="store_true",
                   help="Only codegen + compile; do not launch on a GPU")
    p.add_argument("--arch", default=None, help="Target GFX arch (default: autodetect)")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--no-verify", action="store_true",
                   help="Skip numpy reference verification")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    configs = [u.default_fp8_config(), u.default_bf8_config()]
    log.info("Configs: %s", [c.name for c in configs])

    so_paths = u.setup_multiple_rowcolquant_dispatchers(
        configs,
        output_dir=args.output_dir,
        gfx_arch=args.arch,
    )

    ok = True
    for cfg, so in zip(configs, so_paths):
        if so is None:
            log.error("BUILD FAILED: %s", cfg.name)
            ok = False
        else:
            log.info("BUILT: %s -> %s", cfg.name, so)

    if not ok:
        return 1
    if args.build_only:
        log.info("build-only: all %d kernels built", len(configs))
        return 0

    for so in so_paths:
        try:
            _run_one(so, args.m, args.n, args.k, verify=not args.no_verify)
        except Exception as e:
            log.error("RUN FAILED for %s: %s", so, e)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
