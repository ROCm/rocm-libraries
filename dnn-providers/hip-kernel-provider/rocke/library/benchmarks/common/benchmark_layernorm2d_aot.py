#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lightweight AOT sweep for the norm2d family (layernorm2d / rmsnorm2d).

Sibling of ``benchmark_batchnorm2d_aot.py``; see that script's and
:mod:`benchmarks.common.norm_bench_common`'s module docstrings for why this is
a scaled-down AOT mode (no persistent HSACO cache-dir, no ``batched_timing``
chunked ranking, a local JSON-Lines writer and case key instead of conv's
``case_results``/``early_stop``) relative to
``benchmark_implicit_gemm_conv.py``.

``--kind`` selects which of the two instances backing the norm2d family
(``layernorm`` or ``rmsnorm``) is swept; both share the same request shape
(``rows``/``cols``) and candidate grid.
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
from benchmarks.common.norm_sweep_space import norm_combos


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=2048, help="M: number of rows")
    p.add_argument("--cols", type=int, default=4096, help="N: per-row width")
    p.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    p.add_argument("--arch", default="gfx950")
    p.add_argument("--kind", choices=("layernorm", "rmsnorm"), default="layernorm")
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

    combos = norm_combos(
        rows=args.rows, cols=args.cols, dtype=args.dtype,
        arch=args.arch, kind=args.kind,
    )
    if not combos:
        print("no valid norm2d candidates for this shape/dtype/arch/kind", file=sys.stderr)
        return 1

    dims = {"rows": args.rows, "cols": args.cols}
    key = case_key(dims, dtype=args.dtype, family="norm2d", extra=f"kind={args.kind}")
    writer = JsonlResultsWriter(args.results)

    print(f"norm2d sweep: {key}, {len(combos)} candidate(s)")
    best = None
    for candidate, spec in combos:
        label = candidate.name
        artifact = compile_candidate(spec, candidate, arch=args.arch)
        if args.compile_only:
            print(f"  compiled: {label}")
            continue

        torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        x = torch.empty(args.rows, args.cols, dtype=torch_dtype, device="cuda").uniform_(-1.0, 1.0)
        y = torch.empty_like(x)
        gamma = torch.empty(args.cols, dtype=torch_dtype, device="cuda").uniform_(0.5, 1.5)
        beta = torch.empty(args.cols, dtype=torch_dtype, device="cuda").uniform_(-0.5, 0.5)

        values = {
            "X": x.data_ptr(),
            "Gamma": gamma.data_ptr(),
            "Beta": beta.data_ptr(),
            "Y": y.data_ptr(),
            "M": args.rows,
            "N": args.cols,
            "eps": 1e-5,
        }
        grid = (args.rows, 1, 1)  # one CTA per row
        block = (spec.block_size, 1, 1)
        ms = time_candidate(
            artifact, grid=grid, block=block, args=values,
            warmup=args.warmup, iters=args.iters, timing=args.timing,
        )
        print(f"  {label}: {ms:.4f} ms")
        writer.row(key=key, candidate=label, block_size=spec.block_size,
                   vec=spec.vec, kind=args.kind, dtype=args.dtype, ms=ms, **dims)
        if best is None or ms < best[1]:
            best = (label, ms)

    writer.close()
    if best is not None:
        print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
