#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lightweight AOT sweep for the batchnorm2d kernel family.

Modeled on ``benchmark_implicit_gemm_conv.py``'s AOT mode
(``--compile-all`` / ``--run-from-cache``) but scaled down for a family whose
candidate space is a ``(block_size, vec)`` grid of a few dozen entries rather
than conv's tile/pipeline/epilogue product of thousands: there is no
persistent HSACO cache-dir here, no ``kernel_sweep.compile_all`` /
``kernel_cache.KernelCache``. ``--compile-only`` is the lightweight analog of
``--compile-all`` -- build every valid candidate for the given shape without
launching, a CI smoke step that needs the HIP/LLVM toolchain but no live GPU.
The default mode compiles on demand and benchmarks, one
:class:`~benchmarks.common.norm_bench_common.JsonlResultsWriter` row per
candidate, keyed by :func:`~benchmarks.common.norm_bench_common.case_key`.

See :mod:`benchmarks.common.norm_bench_common` for why this does not reuse
``case_results.ResultsWriter`` / ``early_stop.case_key`` (both are typed for
conv) or ``batched_timing`` (built to rank thousands of candidates behind a
chunked device drain -- overkill for this family's candidate count).
"""

from __future__ import annotations

import argparse
import sys

try:
    # Imported before any rocke.runtime module: rocke's HIP context and
    # torch's lazy CUDA init must not race each other -- initializing torch's
    # first, before compile_candidate/time_candidate touch the runtime,
    # avoids a segfault on the first torch.empty(..., device="cuda") call
    # (a real reproduction, not theoretical: see git history for this line).
    import torch  # noqa: F401
except ImportError:
    pass

from benchmarks.common.norm_bench_common import (
    JsonlResultsWriter,
    case_key,
    compile_candidate,
    time_candidate,
)
from benchmarks.common.norm_sweep_space import batchnorm_combos


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2, help="batch size")
    p.add_argument("--c", type=int, default=64, help="channel count")
    p.add_argument("--h", type=int, default=8, help="input height")
    p.add_argument("--w", type=int, default=8, help="input width")
    p.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    p.add_argument("--arch", default="gfx950")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--training", dest="training", action="store_true", default=True,
        help="training-mode forward: compute batch mean/var, update running stats (default)",
    )
    mode.add_argument(
        "--inference", dest="training", action="store_false",
        help="inference-mode forward: normalize with the given running mean/var",
    )
    p.add_argument("--results", default=None, metavar="FILE",
                    help="write FILE (JSON Lines), one row per candidate")
    p.add_argument(
        "--compile-only", action="store_true",
        help="build every valid (block_size, vec) candidate for this shape "
        "and confirm compile success; do not launch (no GPU required)",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument(
        "--timing", choices=("block", "staged"), default="staged",
        help="'staged' (default) is the protocol comparable to MIOpen/hipDNN; "
        "'block' amortises per-call event overhead across a launch loop",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    combos = batchnorm_combos(
        n=args.n, c=args.c, h=args.h, w=args.w,
        dtype=args.dtype, arch=args.arch, training=args.training,
    )
    if not combos:
        print("no valid batchnorm2d candidates for this shape/dtype/arch", file=sys.stderr)
        return 1

    dims = {"n": args.n, "c": args.c, "h": args.h, "w": args.w}
    key = case_key(dims, dtype=args.dtype, family="batchnorm2d",
                    extra=f"training={args.training}")
    writer = JsonlResultsWriter(args.results)

    print(f"batchnorm2d sweep: {key}, {len(combos)} candidate(s)")
    best = None
    for candidate, spec in combos:
        label = candidate.name
        artifact = compile_candidate(spec, candidate, arch=args.arch)
        if args.compile_only:
            print(f"  compiled: {label}")
            continue

        torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        shape = (args.n, args.c, args.h, args.w)
        x = torch.empty(shape, dtype=torch_dtype, device="cuda").uniform_(-1.0, 1.0)
        y = torch.empty_like(x)
        gamma = torch.empty(args.c, dtype=torch_dtype, device="cuda").uniform_(0.5, 1.5)
        beta = torch.empty(args.c, dtype=torch_dtype, device="cuda").uniform_(-0.5, 0.5)
        running_mean = torch.zeros(args.c, dtype=torch.float32, device="cuda")
        running_var = torch.ones(args.c, dtype=torch.float32, device="cuda")

        values = {
            "X": x.data_ptr(),
            "Gamma": gamma.data_ptr(),
            "Beta": beta.data_ptr(),
            "Y": y.data_ptr(),
            "RunningMean": running_mean.data_ptr(),
            "RunningVar": running_var.data_ptr(),
            "N": args.n,
            "C": args.c,
            "H": args.h,
            "W": args.w,
            "eps": 1e-5,
            "momentum": 0.1,
        }
        grid = (args.c, 1, 1)  # one CTA per channel
        block = (spec.block_size, 1, 1)
        ms = time_candidate(
            artifact, grid=grid, block=block, args=values,
            warmup=args.warmup, iters=args.iters, timing=args.timing,
        )
        print(f"  {label}: {ms:.4f} ms")
        writer.row(key=key, candidate=label, block_size=spec.block_size,
                   vec=spec.vec, training=args.training, dtype=args.dtype,
                   ms=ms, **dims)
        if best is None or ms < best[1]:
            best = (label, ms)

    writer.close()
    if best is not None:
        print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
