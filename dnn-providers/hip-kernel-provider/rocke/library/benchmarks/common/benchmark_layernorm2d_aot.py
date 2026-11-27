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

``--backward`` sweeps the layernorm backward instead (``MIOpenDriver
layernorm* -F 0``): the three AOT-generic kernels of
``instances/common/layernorm2d_bwd.py`` (``dx``, ``dwdb``, ``dwdb_reduce``).
The parts run back to back and share nothing on chip, so each is tuned on
its own -- ``dx`` over its candidates, ``dwdb`` over (candidate, runtime
row-split ``P``), ``dwdb_reduce`` over its candidates per ``P`` -- and then
the best few end-to-end sequences are re-timed as one sample (one event
pair spanning every kernel under ``--timing ext``, as MIOpen times its own
backward). Only those end-to-end rows carry ``ms``; per-part rows carry
``part_ms``, so a per-case "best ``ms``" is always a whole backward.
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
    time_sequence,
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
    p.add_argument(
        "--backward", action="store_true",
        help="sweep the layernorm backward (dX, dgamma, dbeta -- MIOpenDriver "
        "-F 0) instead of the forward; see the module docstring",
    )
    p.add_argument(
        "--num-cus", type=int, default=None, dest="num_cus",
        help="--backward: CU count the default row split P is sized for "
        "(default: the visible device's)",
    )
    p.add_argument(
        "--bwd-finalists", type=int, default=3, dest="bwd_finalists",
        help="--backward: end-to-end sequences re-timed after the per-part sweep",
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


#: ``--validate`` tolerance for the backward, relative to the reference's
#: max magnitude: dgamma / dbeta are sums over every row, so their absolute
#: error grows with ``rows`` while their relative error does not. ~2 ULP of
#: each dtype's mantissa (bf16 2^-7, fp16 2^-10).
BWD_VALIDATE_REL_TOL = {"fp16": 2e-3, "bf16": 1.6e-2}


def _bwd_row_split_options(*, rows: int, default: int):
    """Runtime ``P`` values the backward sweep tries around the default."""
    opts = {max(1, default // 4), max(1, default // 2), default,
            min(rows, default * 2), min(rows, default * 4)}
    return sorted(opts)


def _validate_backward(steps, *, x, dy, gamma, mean, rstd, dx, dw, db, dtype, key, writer):
    """Relaunch one backward sequence and check dX / dgamma / dbeta against
    a torch reference built from the same (rounded) saved mean / rstd."""
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, synchronize_and_release

    for artifact, grid, block, values in steps:
        KernelLauncher.from_artifact(artifact)(values, config=LaunchConfig(grid=grid, block=block))
    synchronize_and_release()

    xf, dyf, gf = x.float(), dy.float(), gamma.float()
    m, r = mean.float()[:, None], rstd.float()[:, None]
    xhat = (xf - m) * r
    dyg = dyf * gf
    ref = {
        "dx": r * (dyg - dyg.mean(-1, keepdim=True)
                   - xhat * (dyg * xhat).mean(-1, keepdim=True)),
        "dgamma": (dyf * xhat).sum(0),
        "dbeta": dyf.sum(0),
    }
    got = {"dx": dx, "dgamma": dw, "dbeta": db}
    tol = BWD_VALIDATE_REL_TOL[dtype]
    rel = {
        name: float((got[name].float() - ref[name]).abs().max().item()
                    / max(float(ref[name].abs().max().item()), 1e-12))
        for name in ref
    }
    ok = all(v <= tol for v in rel.values())
    detail = " ".join(f"{k}={v:.3e}" for k, v in rel.items())
    print(f"Validate: {'PASS' if ok else 'FAIL'} max_rel {detail} (tol={tol})")
    writer.row(key=key, candidate="__validate__", validate=True, ok=ok,
               max_rel=max(rel.values()), **{f"max_rel_{k}": v for k, v in rel.items()})


def _run_backward(args) -> int:
    from rocke.dispatch.families.norm import NormRequest, dynamic_norm_bwd_candidates
    from rocke.instances.common.layernorm2d_bwd import (
        PARTS,
        default_row_splits,
        layernorm2d_bwd_grid,
        layernorm2d_bwd_workspace_bytes,
    )

    rows, cols = args.rows, args.cols
    req = NormRequest(rows=rows, cols=cols, arch=args.arch, kind="layernorm", dtype=args.dtype)
    combos = {
        part: [(c, c.select_spec(req)) for c in dynamic_norm_bwd_candidates(part)
               if c.admits(req)[0]]
        for part in PARTS
    }
    missing = [part for part, cs in combos.items() if not cs]
    if missing:
        print(f"no valid layernorm backward candidates for part(s) {missing} "
              f"at rows={rows} cols={cols} dtype={args.dtype}", file=sys.stderr)
        return 1

    dims = {"rows": rows, "cols": cols}
    key = case_key(dims, dtype=args.dtype, family="norm2d", extra="kind=layernorm_bwd")
    writer = JsonlResultsWriter(args.results)
    n_total = sum(len(cs) for cs in combos.values())
    print(f"norm2d backward sweep: {key}, "
          + ", ".join(f"{len(combos[p])} {p}" for p in PARTS) + " candidate(s)")

    if args.compile_only:
        for part in PARTS:
            for candidate, spec in combos[part]:
                _compile(spec, candidate, args=args, dims=dims)
                print(f"  compiled: {candidate.name}")
        writer.close()
        print(f"compiled {n_total} candidate(s)")
        return 0

    num_cus = args.num_cus or torch.cuda.get_device_properties(0).multi_processor_count
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    x = torch.empty(rows, cols, dtype=torch_dtype, device="cuda").uniform_(-1.0, 1.0)
    dy = torch.empty_like(x).uniform_(-1.0, 1.0)
    gamma = torch.empty(cols, dtype=torch_dtype, device="cuda").uniform_(0.5, 1.5)
    xf = x.float()
    mean = xf.mean(-1).to(torch_dtype)
    rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5).to(torch_dtype)
    dx = torch.empty_like(x)
    dw = torch.empty_like(gamma)
    db = torch.empty_like(gamma)

    def block_of(spec):
        return (spec.block_size, 1, 1)

    # --- dx ---------------------------------------------------------------
    dx_values = {"DY": dy.data_ptr(), "X": x.data_ptr(), "Gamma": gamma.data_ptr(),
                 "Mean": mean.data_ptr(), "Rstd": rstd.data_ptr(), "DX": dx.data_ptr(),
                 "M": rows, "N": cols}
    dx_best = None
    for candidate, spec in combos["dx"]:
        artifact = _compile(spec, candidate, args=args, dims=dims)
        step = (artifact, layernorm2d_bwd_grid(spec, rows=rows, cols=cols), block_of(spec),
                dx_values)
        ms = time_sequence([step], warmup=args.warmup, iters=args.iters, timing=args.timing)
        print(f"  [dx] {candidate.name}: {ms:.4f} ms")
        writer.row(key=key, part="dx", candidate=candidate.name, block_size=spec.block_size,
                   vec=spec.vec, dtype=args.dtype, part_ms=ms, **dims)
        if dx_best is None or ms < dx_best[0]:
            dx_best = (ms, candidate, step)

    # --- dwdb over (candidate, P) -----------------------------------------
    splits_seen = set()
    dwdb_plans = []
    dwdb_arts = [(c, s, _compile(s, c, args=args, dims=dims)) for c, s in combos["dwdb"]]
    for candidate, spec, artifact in dwdb_arts:
        p0 = default_row_splits(rows=rows, cols=cols, block_size=spec.block_size,
                                vec=spec.vec, num_cus=num_cus)
        splits_seen.update(_bwd_row_split_options(rows=rows, default=p0))
    max_splits = max(splits_seen)
    ws = torch.empty(max(1, layernorm2d_bwd_workspace_bytes(cols=cols, splits=max_splits) // 4),
                     dtype=torch.float32, device="cuda")
    for candidate, spec, artifact in dwdb_arts:
        p0 = default_row_splits(rows=rows, cols=cols, block_size=spec.block_size,
                                vec=spec.vec, num_cus=num_cus)
        for splits in _bwd_row_split_options(rows=rows, default=p0):
            values = {"DY": dy.data_ptr(), "X": x.data_ptr(), "Mean": mean.data_ptr(),
                      "Rstd": rstd.data_ptr(), "DW": dw.data_ptr(), "DB": db.data_ptr(),
                      "WS": ws.data_ptr(), "M": rows, "N": cols, "P": splits}
            step = (artifact, layernorm2d_bwd_grid(spec, rows=rows, cols=cols, splits=splits),
                    block_of(spec), values)
            ms = time_sequence([step], warmup=args.warmup, iters=args.iters,
                               timing=args.timing)
            print(f"  [dwdb] {candidate.name} P={splits}: {ms:.4f} ms")
            writer.row(key=key, part="dwdb", candidate=candidate.name, splits=splits,
                       block_size=spec.block_size, vec=spec.vec, unroll=spec.unroll,
                       dtype=args.dtype, part_ms=ms, **dims)
            dwdb_plans.append((ms, candidate, splits, step))

    # --- dwdb_reduce per P > 1 --------------------------------------------
    reduce_best = {}
    reduce_arts = [(c, s, _compile(s, c, args=args, dims=dims))
                   for c, s in combos["dwdb_reduce"]]
    for splits in sorted(p for p in splits_seen if p > 1):
        for candidate, spec, artifact in reduce_arts:
            values = {"WS": ws.data_ptr(), "DW": dw.data_ptr(), "DB": db.data_ptr(),
                      "N": cols, "P": splits}
            step = (artifact, layernorm2d_bwd_grid(spec, rows=rows, cols=cols),
                    block_of(spec), values)
            ms = time_sequence([step], warmup=args.warmup, iters=args.iters,
                               timing=args.timing)
            writer.row(key=key, part="dwdb_reduce", candidate=candidate.name,
                       splits=splits, block_size=spec.block_size, vec=spec.vec,
                       unroll=spec.unroll, dtype=args.dtype, part_ms=ms, **dims)
            if splits not in reduce_best or ms < reduce_best[splits][0]:
                reduce_best[splits] = (ms, candidate, step)
        r_ms, r_cand, _ = reduce_best[splits]
        print(f"  [dwdb_reduce] P={splits}: best {r_cand.name} {r_ms:.4f} ms")

    # --- end-to-end finalists ---------------------------------------------
    def est(plan):
        ms, _c, splits, _s = plan
        return ms + (reduce_best[splits][0] if splits > 1 else 0.0)

    # The per-part estimate cannot see the gap between back-to-back
    # launches, which favours the three-kernel P > 1 plans; the best
    # two-kernel (P == 1) plan always gets an end-to-end look too.
    ranked = sorted(dwdb_plans, key=est)
    finalists = ranked[: max(1, args.bwd_finalists)]
    direct = [plan for plan in ranked if plan[2] == 1]
    if direct and direct[0] not in finalists:
        finalists.append(direct[0])
    best = None
    for plan in finalists:
        _ms, w_cand, splits, w_step = plan
        steps = [dx_best[2], w_step]
        label = f"{dx_best[1].name}+{w_cand.name}+P{splits}"
        r_name = None
        if splits > 1:
            steps.append(reduce_best[splits][2])
            r_name = reduce_best[splits][1].name
            label += f"+{r_name}"
        ms = time_sequence(steps, warmup=args.warmup, iters=args.iters, timing=args.timing)
        print(f"  [e2e] {label}: {ms:.4f} ms (parts est {dx_best[0] + est(plan):.4f})")
        writer.row(key=key, candidate=label, dx=dx_best[1].name, dwdb=w_cand.name,
                   dwdb_reduce=r_name, splits=splits, kind="layernorm_bwd",
                   dtype=args.dtype, ms=ms, **dims)
        if best is None or ms < best[1]:
            best = (label, ms, steps)

    if args.validate:
        _validate_backward(best[2], x=x, dy=dy, gamma=gamma, mean=mean, rstd=rstd,
                           dx=dx, dw=dw, db=db, dtype=args.dtype, key=key, writer=writer)

    writer.close()
    print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    if args.backward:
        if args.kind != "layernorm":
            print("--backward only has a layernorm candidate set", file=sys.stderr)
            return 1
        return _run_backward(args)

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
