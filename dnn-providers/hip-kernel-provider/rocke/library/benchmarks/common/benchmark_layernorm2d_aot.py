#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lightweight AOT sweep for the norm2d family (layernorm2d / rmsnorm2d).

Sibling of ``benchmark_batchnorm2d_aot.py``; see that script's and
:mod:`benchmarks.common.norm_bench_common`'s module docstrings for why this is
a scaled-down AOT mode (no persistent HSACO cache-dir, a local JSON-Lines
writer and case key instead of conv's ``case_results``/``early_stop``)
relative to ``benchmark_implicit_gemm_conv.py``, and for ``--fast-rank``, the
same chunked-ranking escape hatch conv's sweeps use for a large candidate grid.

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
    VALIDATE_TOL,
    JsonlResultsWriter,
    add_fast_rank_args,
    case_key,
    compile_candidate,
    rank_candidates,
    time_candidate,
)
from benchmarks.common.norm_sweep_space import norm_combos, norm_dynamic_combos


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
        "--timing", choices=("block", "staged", "direct", "ext"), default="staged",
        help="'staged' (default) is the protocol comparable to MIOpen/hipDNN; "
        "'block' amortises per-call event overhead across a launch loop; "
        "'direct' is 'staged' without the StallGate (MIOpen's own protocol "
        "has no equivalent gate), to isolate the gate's own cost from a "
        "candidate that is genuinely launch-bound; 'ext' records the "
        "events on the kernel's own dispatch packet, as MIOpen does, "
        "and is the like-for-like protocol against MIOpen kernel times",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="after picking the fastest candidate, relaunch it once and check "
        "its output against a torch layer_norm/rms reference (the runbook "
        "rule: never report speed without correctness)",
    )
    p.add_argument(
        "--cache-dir", default=None, metavar="DIR", dest="cache_dir",
        help="load/store compiled KernelArtifacts here instead of compiling "
        "inline (see norm_kernel_cache.py); pairs with norm_compile_all.py "
        "pre-warming the cache across many CPU workers before a GPU-pinned "
        "run_norm_shards.py pass",
    )
    dyn = p.add_mutually_exclusive_group()
    dyn.add_argument(
        "--dynamic", dest="dynamic", action="store_true", default=None,
        help="sweep the AOT-generic (dynamic=True) candidates: one compiled "
        "binary per (block_size, vec) tuning config serves any runtime "
        "row width instead of being built fresh per shape -- see "
        "instances/common/layernorm2d.py's LayerNorm2DSpec.dynamic and "
        "norm_kernel_cache.py's dynamic cache-key note. Default for "
        "--kind layernorm (rmsnorm has no dynamic variant yet, so it "
        "defaults to --static)",
    )
    dyn.add_argument(
        "--static", dest="dynamic", action="store_false",
        help="sweep the shape-specialized candidates instead (one compile "
        "per exact rows/cols) -- the old default, kept for comparison/"
        "debugging",
    )
    add_fast_rank_args(p)
    args = p.parse_args(argv)
    if args.dynamic is None:
        args.dynamic = args.kind == "layernorm"
    return args


def _compile(spec, candidate, *, args, dims):
    if args.cache_dir is None:
        return compile_candidate(spec, candidate, arch=args.arch)
    from benchmarks.common.norm_kernel_cache import compile_candidate_cached

    return compile_candidate_cached(
        spec, candidate, arch=args.arch, cache_dir=args.cache_dir,
        family="norm2d", dims={**dims, "kind": args.kind}, dtype=args.dtype,
    )


def _validate_norm2d(artifact, *, grid, block, x, y, gamma, beta, kind, dtype, key, writer) -> None:
    """Relaunch *artifact* once and check ``y`` against a torch reference.

    Only rocke's own output is checked here -- see
    ``norm_shard_worker.py`` for the (timing-only) MIOpen find-mode
    comparison, which carries no correctness claim of its own.
    """
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, synchronize_and_release

    launcher = KernelLauncher.from_artifact(artifact)
    cfg = LaunchConfig(grid=grid, block=block)
    values = {
        "X": x.data_ptr(), "Gamma": gamma.data_ptr(), "Beta": beta.data_ptr(),
        "Y": y.data_ptr(), "M": x.shape[0], "N": x.shape[1], "eps": 1e-5,
    }
    launcher(values, config=cfg)
    synchronize_and_release()

    if kind == "layernorm":
        ref = torch.nn.functional.layer_norm(
            x.float(), (x.shape[1],), gamma.float(), beta.float(), 1e-5
        )
    else:
        var = x.float().pow(2).mean(-1, keepdim=True)
        ref = x.float() * torch.rsqrt(var + 1e-5) * gamma.float()
    max_abs = float((y.float() - ref).abs().max().item())
    tol = VALIDATE_TOL[dtype]
    ok = max_abs <= tol
    print(f"Validate: {'PASS' if ok else 'FAIL'} max_abs={max_abs:.4e} (tol={tol})")
    writer.row(key=key, candidate="__validate__", validate=True, ok=ok, max_abs=max_abs)


def main(argv=None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    if args.dynamic and args.kind != "layernorm":
        print("--dynamic only has a layernorm candidate set (no dynamic rmsnorm yet)",
              file=sys.stderr)
        return 1
    combo_fn = norm_dynamic_combos if args.dynamic else norm_combos
    combos = combo_fn(
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

    if args.compile_only:
        for candidate, spec in combos:
            _compile(spec, candidate, args=args, dims=dims)
            print(f"  compiled: {candidate.name}")
        writer.close()
        return 0

    best = None
    if args.fast_rank:
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

        plans = []
        for candidate, spec in combos:
            artifact = _compile(spec, candidate, args=args, dims=dims)
            block = (spec.block_size, 1, 1)
            plans.append((candidate.name, artifact, grid, block, values))

        best_idx = None
        ranked = rank_candidates(plans, args=args, timing=args.timing)
        for r in ranked:
            if r.ms is None:
                continue
            label = plans[r.index][0]
            spec = combos[r.index][1]
            print(f"  {label}: {r.ms:.4f} ms")
            writer.row(key=key, candidate=label, block_size=spec.block_size,
                       vec=spec.vec, kind=args.kind, dtype=args.dtype, ms=r.ms, **dims)
            if best is None or r.ms < best[1]:
                best = (label, r.ms)
                best_idx = r.index

        if args.validate and best_idx is not None:
            _, artifact, grid, block, _values = plans[best_idx]
            _validate_norm2d(artifact, grid=grid, block=block, x=x, y=y, gamma=gamma,
                              beta=beta, kind=args.kind, dtype=args.dtype, key=key,
                              writer=writer)
    else:
        best_validate = None
        for candidate, spec in combos:
            label = candidate.name
            artifact = _compile(spec, candidate, args=args, dims=dims)

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
                best_validate = (artifact, grid, block, x, y, gamma, beta)

        if args.validate and best is not None:
            artifact, grid, block, x, y, gamma, beta = best_validate
            _validate_norm2d(artifact, grid=grid, block=block, x=x, y=y, gamma=gamma,
                              beta=beta, kind=args.kind, dtype=args.dtype, key=key,
                              writer=writer)

    writer.close()
    if best is not None:
        print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
