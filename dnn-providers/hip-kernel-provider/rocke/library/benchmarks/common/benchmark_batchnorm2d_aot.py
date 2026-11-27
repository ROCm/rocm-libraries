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
conv). ``n_groups > 1``'s masked tail can grow this family's candidate count
into the hundreds for small H*W shapes, at which point a per-candidate
device drain starts to dominate the sweep the same way it does for conv's
much larger grids -- pass ``--fast-rank`` to rank candidates behind one
chunked drain and only time the best ``--fast-rank-finalists`` in full (see
:func:`~benchmarks.common.norm_bench_common.rank_candidates`).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from rocke.numeric.batchnorm_reference import (
    batchnorm_backward_reference,
    batchnorm_forward_reference,
)
from rocke.runtime._norm_buffer import NormBuffer

from benchmarks.common.norm_bench_common import (
    VALIDATE_TOL,
    JsonlResultsWriter,
    add_fast_rank_args,
    case_key,
    compile_candidate,
    rank_candidates,
    time_candidate,
)
from benchmarks.common.norm_sweep_space import (
    batchnorm_combos,
    batchnorm_dynamic_combos,
)


_RNG = np.random.default_rng(0)


def _uniform(shape, dtype, lo, hi):
    return NormBuffer.uniform(shape, dtype, lo, hi, _RNG)


def _fwd_bufs(n, c, h, w, dtype, *, layout="NCHW", saved=True):
    """Device buffers of one forward case (``x gamma beta y rm rv`` + ``mean invstd``)."""
    shape = (n, c, h, w) if layout == "NCHW" else (n, h, w, c)
    x = _uniform(shape, dtype, -1.0, 1.0)
    bufs = {
        "x": x,
        "y": NormBuffer(shape, dtype),
        "gamma": _uniform((c,), dtype, 0.5, 1.5),
        "beta": _uniform((c,), dtype, -0.5, 0.5),
        "rm": NormBuffer((c,), "f32"),
        "rv": NormBuffer.from_host(np.ones(c, dtype=np.float32), "f32"),
    }
    if saved:
        bufs["mean"] = NormBuffer((c,), dtype)
        bufs["invstd"] = NormBuffer((c,), dtype)
    return bufs


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2, help="batch size")
    p.add_argument("--c", type=int, default=64, help="channel count")
    p.add_argument("--h", type=int, default=8, help="input height")
    p.add_argument("--w", type=int, default=8, help="input width")
    p.add_argument(
        "--d",
        type=int,
        default=1,
        help="input depth (NCDHW): folded into the height, the memory layout is identical",
    )
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument(
        "--layout",
        choices=("NCHW", "NHWC"),
        default="NCHW",
        help="NHWC: channels-last candidates (infer / --training / --backward), AOT cache only",
    )
    p.add_argument("--arch", default="gfx950")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--training",
        dest="training",
        action="store_true",
        default=True,
        help="training-mode forward: compute batch mean/var, update running stats (default)",
    )
    mode.add_argument(
        "--inference",
        dest="training",
        action="store_false",
        help="inference-mode forward: normalize with the given running mean/var",
    )
    p.add_argument(
        "--results",
        default=None,
        metavar="FILE",
        help="write FILE (JSON Lines), one row per candidate",
    )
    p.add_argument(
        "--compile-only",
        action="store_true",
        help="build every valid (block_size, vec) candidate for this shape "
        "and confirm compile success; do not launch (no GPU required)",
    )
    p.add_argument(
        "--name-filter",
        default=None,
        metavar="SUBSTR",
        dest="name_filter",
        help="only sweep candidates whose name contains SUBSTR; '=a|b' selects the exact names a, b",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument(
        "--timing",
        choices=("block", "staged"),
        default="staged",
        help="'staged' (default) is the protocol comparable to MIOpen/hipDNN; "
        "'block' amortises per-call event overhead across a launch loop",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="after picking the fastest candidate, relaunch it once and check "
        "its output against a numpy batch-norm reference (the "
        "runbook rule: never report speed without correctness)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        dest="cache_dir",
        help="load/store compiled KernelArtifacts here instead of compiling "
        "inline (see norm_kernel_cache.py); pairs with norm_compile_all.py "
        "pre-warming the cache across many CPU workers before a GPU-pinned "
        "run_norm_shards.py pass",
    )
    dyn = p.add_mutually_exclusive_group()
    dyn.add_argument(
        "--dynamic",
        dest="dynamic",
        action="store_true",
        default=True,
        help="sweep the AOT-generic (dynamic=True) candidates (default): one "
        "compiled binary per tuning config "
        "(block_size/vec/channels_per_block/n_groups/fast_reduce) serves any "
        "runtime shape instead of being built fresh per shape -- see "
        "instances/common/batchnorm2d.py's BatchNorm2DSpec.dynamic and "
        "norm_kernel_cache.py's dynamic cache-key note",
    )
    dyn.add_argument(
        "--static",
        dest="dynamic",
        action="store_false",
        help="sweep the shape-specialized candidates instead (one compile per "
        "exact N/C/H/W) -- the old default, kept for comparison/debugging",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="training only: sweep the dynamic cross-CTA split-reduction candidates "
        "(two launches per candidate: reduce + apply) instead of the single-kernel ones",
    )
    p.add_argument(
        "--backward",
        action="store_true",
        help="backward pass (spatial, saved mean/invstd): sweep the dynamic backward candidates; "
        "combine with --split for the two-launch backward candidates",
    )
    p.add_argument(
        "--mcw",
        action="store_true",
        help="multi-channel-per-wave single-launch candidates (--training or --backward), NCHW, AOT cache only",
    )
    p.add_argument(
        "--split-order",
        default=None,
        metavar="FILE",
        help="with --split: write one line per candidate (label, number of launch "
        "pairs) in launch order, so a kernel trace can be attributed per plan",
    )
    add_fast_rank_args(p)
    return p.parse_args(argv)


def _name_ok(flt, name) -> bool:
    """``--name-filter`` match: substring, or (leading ``=``) one of the ``|``-separated exact names."""
    if flt.startswith("="):
        return name in flt[1:].split("|")
    return flt in name


def _compile(spec, candidate, *, args, dims):
    if args.cache_dir is None:
        return compile_candidate(spec, candidate, arch=args.arch)
    from benchmarks.common.norm_kernel_cache import compile_candidate_cached

    return compile_candidate_cached(
        spec,
        candidate,
        arch=args.arch,
        cache_dir=args.cache_dir,
        family="batchnorm2d",
        dims={**dims, "training": args.training},
        dtype=args.dtype,
    )


def _add_saved_stats(values, *, training, c, dtype):
    """Training candidates save the batch mean/invstd (MIOpen ``-s 1``); give them buffers.

    Returns the tensors so the caller keeps them alive across the launch.
    """
    if not training:
        return ()
    mean = NormBuffer((c,), dtype)
    invstd = NormBuffer((c,), dtype)
    values.update({"Mean": mean.data_ptr(), "InvStd": invstd.data_ptr()})
    return mean, invstd


def _validate_batchnorm(
    artifact,
    *,
    grid,
    block,
    x,
    y,
    gamma,
    beta,
    running_mean,
    running_var,
    training,
    dtype,
    key,
    writer,
) -> None:
    """Relaunch *artifact* once and check ``y`` against a numpy reference.

    Only rocke's own output is checked here -- this is not a MIOpen
    comparison (see ``norm_shard_worker.py`` for MIOpen find-mode timing,
    which is timing-only and carries no correctness claim of its own).
    The running statistics are reset to their initial values (zeros / ones)
    before the relaunch so the reference sees the same pre-update stats the
    kernel's own momentum update reads from, whichever mode is being validated.
    """
    from rocke.runtime.launcher import (
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    running_mean.write(np.zeros(running_mean.shape, dtype=np.float32))
    running_var.write(np.ones(running_var.shape, dtype=np.float32))
    launcher = KernelLauncher.from_artifact(artifact)
    cfg = LaunchConfig(grid=grid, block=block)
    values = {
        "X": x.data_ptr(),
        "Gamma": gamma.data_ptr(),
        "Beta": beta.data_ptr(),
        "Y": y.data_ptr(),
        "RunningMean": running_mean.data_ptr(),
        "RunningVar": running_var.data_ptr(),
        "N": x.shape[0],
        "C": x.shape[1],
        "H": x.shape[2],
        "W": x.shape[3],
        "eps": 1e-5,
        "momentum": 0.1,
    }
    _saved = _add_saved_stats(values, training=training, c=x.shape[1], dtype=dtype)
    launcher(values, config=cfg)
    synchronize_and_release()

    ref = batchnorm_forward_reference(
        x.read(),
        gamma.read(),
        beta.read(),
        np.zeros(x.shape[1]),
        np.ones(x.shape[1]),
        training=training,
        eps=1e-5,
        momentum=0.1,
    )["y"]
    max_abs = float(np.abs(y.read().astype(np.float64) - ref).max())
    tol = VALIDATE_TOL[dtype]
    ok = max_abs <= tol
    print(f"Validate: {'PASS' if ok else 'FAIL'} max_abs={max_abs:.4e} (tol={tol})")
    writer.row(key=key, candidate="__validate__", validate=True, ok=ok, max_abs=max_abs)


def _validate_run(
    run, bufs, *, mode, layout, dtype, key, writer, label, sync=None
) -> bool:
    """Relaunch one runner on freshly reset state and check every output with numpy.

    ``mode`` is ``train`` / ``bwd`` (``infer`` checks ``y`` only).  The running
    statistics go back to zeros / ones first; ``sync`` (read-once barrier words)
    is zeroed.  Prints one ``Validate`` line and writes a ``validate`` row."""
    from rocke.runtime.launcher import synchronize_and_release

    axis = 1 if layout == "NCHW" else -1
    tol = VALIDATE_TOL[dtype]
    x = bufs["x"].read()
    c = bufs["x"].shape[axis]
    errs = {}
    if mode == "bwd":
        run()
        synchronize_and_release()
        ref = batchnorm_backward_reference(
            x,
            bufs["dy"].read(),
            bufs["scale"].read(),
            bufs["mean"].read(),
            bufs["invstd"].read(),
            channel_axis=axis,
        )
        errs["dx"] = float(np.abs(bufs["dx"].read() - ref["dx"]).max())
        for k in ("dscale", "dbias"):
            err = np.abs(bufs[k].read() - ref[k])
            errs[k] = float((err / np.maximum(1.0, np.abs(ref[k]))).max())
        limits = {"dx": 2 * tol, "dscale": 1e-3, "dbias": 1e-3}
    else:
        training = mode == "train"
        bufs["rm"].write(np.zeros(c, dtype=np.float32))
        bufs["rv"].write(np.ones(c, dtype=np.float32))
        if sync is not None:
            sync.zero_()
        run()
        synchronize_and_release()
        ref = batchnorm_forward_reference(
            x,
            bufs["gamma"].read(),
            bufs["beta"].read(),
            np.zeros(c),
            np.ones(c),
            training=training,
            eps=1e-5,
            momentum=0.1,
            channel_axis=axis,
        )
        errs["y"] = float(np.abs(bufs["y"].read() - ref["y"]).max())
        limits = {"y": tol}
        if training:
            errs["running_mean"] = float(
                np.abs(bufs["rm"].read() - ref["running_mean"]).max()
            )
            errs["running_var"] = float(
                np.abs(bufs["rv"].read() - ref["running_var"]).max()
            )
            errs["mean"] = float(np.abs(bufs["mean"].read() - ref["mean"]).max())
            errs["invstd"] = float(np.abs(bufs["invstd"].read() - ref["invstd"]).max())
            limits.update(running_mean=1e-3, running_var=1e-3, mean=tol, invstd=tol)
    ok = all(errs[k] <= limits[k] for k in errs)
    worst = max(errs, key=lambda k: errs[k] / limits[k])
    print(
        f"Validate {label}: {'PASS' if ok else 'FAIL'} worst={worst} err={errs[worst]:.4e} (tol={limits[worst]})"
    )
    writer.row(
        key=key,
        candidate="__validate__",
        validate=True,
        ok=ok,
        label=label,
        **{f"err_{k}": v for k, v in errs.items()},
    )
    return ok


def _main_split(args) -> int:
    """``--split``: time every split (candidate, plan) as one reduce+apply pair."""
    from benchmarks.common.norm_kernel_cache import compile_split_cached
    from benchmarks.common.norm_split_launch import SplitRun, split_plan_label
    from benchmarks.common.norm_sweep_space import batchnorm_split_combos
    from rocke.instances.common.batchnorm2d import (
        batchnorm2d_split_dyn_workspace_floats,
    )
    from rocke.runtime import time_launches_staged
    import statistics

    if not args.training:
        print("--split is training only", file=sys.stderr)
        return 1
    if args.cache_dir is None:
        print("--split needs --cache-dir (AOT cache)", file=sys.stderr)
        return 1
    combos = batchnorm_split_combos(
        n=args.n, c=args.c, h=args.h, w=args.w, dtype=args.dtype, arch=args.arch
    )
    if args.name_filter:
        combos = [
            cp for cp in combos if _name_ok(args.name_filter, split_plan_label(*cp))
        ]
    if not combos:
        print("no valid split candidates for this shape/dtype/arch", file=sys.stderr)
        return 1
    dims = {"n": args.n, "c": args.c, "h": args.h, "w": args.w}
    key = case_key(
        dims, dtype=args.dtype, family="batchnorm2d", extra="training=True,split"
    )
    writer = JsonlResultsWriter(args.results)
    print(f"batchnorm2d split sweep: {key}, {len(combos)} candidate(s)")

    bufs = _fwd_bufs(args.n, args.c, args.h, args.w, args.dtype)
    ws = None
    arts = {}
    order = []
    best, best_run = None, None
    for cand, plan in combos:
        need = batchnorm2d_split_dyn_workspace_floats(args.c, plan)
        if ws is None or ws.numel() < need:
            ws = NormBuffer((need,), "f32")
        for kind in ("reduce", "apply"):
            if (cand.name, kind) not in arts:
                arts[(cand.name, kind)] = compile_split_cached(
                    cand,
                    kind,
                    dtype=args.dtype,
                    arch=args.arch,
                    cache_dir=args.cache_dir,
                )
        run = SplitRun(
            cand,
            plan,
            arts[(cand.name, "reduce")],
            arts[(cand.name, "apply")],
            n=args.n,
            c=args.c,
            h=args.h,
            w=args.w,
            bufs=bufs,
            ws=ws,
        )
        samples = time_launches_staged(run, warmup=args.warmup, iters=args.iters)
        ms = statistics.median(samples)
        label = split_plan_label(cand, plan)
        order.append((label, args.warmup + args.iters))
        print(f"  {label}: {ms:.4f} ms")
        writer.row(
            key=key,
            candidate=label,
            block_size=cand.block_size,
            vec=cand.vec,
            training=True,
            dtype=args.dtype,
            ms=ms,
            **dims,
        )
        if best is None or ms < best[1]:
            best = (label, ms)
            best_run = run
    if args.split_order:
        with open(args.split_order, "w") as fh:
            fh.write("".join(f"{lb}\t{k}\n" for lb, k in order))
    rc = 0
    if (
        args.validate
        and best is not None
        and not _validate_run(
            best_run,
            bufs,
            mode="train",
            layout="NCHW",
            dtype=args.dtype,
            key=key,
            writer=writer,
            label=best[0],
        )
    ):
        rc = 1
    writer.close()
    if best is not None:
        print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return rc


def _bwd_bufs(args, layout="NCHW"):
    dtype = args.dtype
    n, c, h, w = args.n, args.c, args.h, args.w
    shape = (n, c, h, w) if layout == "NCHW" else (n, h, w, c)
    return {
        "x": _uniform(shape, dtype, -1.0, 1.0),
        "dy": _uniform(shape, dtype, -1.0, 1.0),
        "dx": NormBuffer(shape, dtype),
        "scale": _uniform((c,), "f32", 0.5, 1.5),
        "mean": _uniform((c,), "f32", -0.1, 0.1),
        "invstd": _uniform((c,), "f32", 0.8, 1.2),
        "dscale": NormBuffer((c,), "f32"),
        "dbias": NormBuffer((c,), "f32"),
    }


class _BwdLaunch:
    """One compiled single-launch kernel on fixed arguments, callable without arguments."""

    def __init__(self, artifact, values, grid, block_size):
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        self.launcher = KernelLauncher.from_artifact(artifact)
        self.values = values
        self.cfg = LaunchConfig(grid=grid, block=(block_size, 1, 1))

    def __call__(self) -> None:
        self.launcher(self.values, config=self.cfg)


def _main_bwd(args) -> int:
    """``--backward``: time every single-launch candidate, or (``--split``) every
    split pair; the order file / kernel names let a trace attribute the launches."""
    import statistics
    from benchmarks.common.norm_split_launch import (
        SplitBwdRun,
        bwd_values,
        split_plan_label,
    )
    from benchmarks.common.norm_sweep_space import (
        batchnorm_bwd_dynamic_combos,
        batchnorm_bwd_split_combos,
    )
    from rocke.instances.common.batchnorm2d import batchnorm2d_bwd_grid
    from rocke.runtime import time_launches_staged

    n, c, h, w = args.n, args.c, args.h, args.w
    dims = {"n": n, "c": c, "h": h, "w": w}
    bufs = _bwd_bufs(args)
    writer = JsonlResultsWriter(args.results)
    key = case_key(
        dims,
        dtype=args.dtype,
        family="batchnorm2d",
        extra="backward" + (",split" if args.split else ""),
    )
    best, best_run = None, None
    if args.split:
        from benchmarks.common.norm_kernel_cache import compile_split_cached

        if args.cache_dir is None:
            print("--split needs --cache-dir (AOT cache)", file=sys.stderr)
            return 1
        combos = batchnorm_bwd_split_combos(
            n=n, c=c, h=h, w=w, dtype=args.dtype, arch=args.arch
        )
        if args.name_filter:
            combos = [
                cp for cp in combos if _name_ok(args.name_filter, split_plan_label(*cp))
            ]
        if not combos:
            print(
                "no valid split backward candidates for this shape/dtype/arch",
                file=sys.stderr,
            )
            return 1
        print(f"batchnorm2d backward split sweep: {key}, {len(combos)} plan(s)")
        ws, arts, order = None, {}, []
        for cand, plan in combos:
            need = cand.workspace_floats(c, plan)
            if ws is None or ws.numel() < need:
                ws = NormBuffer((need,), "f32")
            for kind in ("reduce", "apply"):
                if (cand.name, kind) not in arts:
                    arts[(cand.name, kind)] = compile_split_cached(
                        cand,
                        kind,
                        dtype=args.dtype,
                        arch=args.arch,
                        cache_dir=args.cache_dir,
                    )
            run = SplitBwdRun(
                cand,
                plan,
                arts[(cand.name, "reduce")],
                arts[(cand.name, "apply")],
                n=n,
                c=c,
                h=h,
                w=w,
                bufs=bufs,
                ws=ws,
            )
            ms = statistics.median(
                time_launches_staged(run, warmup=args.warmup, iters=args.iters)
            )
            label = split_plan_label(cand, plan)
            order.append((label, args.warmup + args.iters))
            print(f"  {label}: {ms:.4f} ms")
            writer.row(key=key, candidate=label, ms=ms, backward=True, **dims)
            if best is None or ms < best[1]:
                best = (label, ms)
                best_run = run
        if args.split_order:
            with open(args.split_order, "w") as fh:
                fh.write("".join(f"{lb}\t{k}\n" for lb, k in order))
    else:
        combos = batchnorm_bwd_dynamic_combos(
            n=n, c=c, h=h, w=w, dtype=args.dtype, arch=args.arch
        )
        if args.name_filter:
            combos = [cs for cs in combos if _name_ok(args.name_filter, cs[0].name)]
        if not combos:
            print(
                "no valid batchnorm2d backward candidates for this shape/dtype/arch",
                file=sys.stderr,
            )
            return 1
        print(f"batchnorm2d backward sweep: {key}, {len(combos)} candidate(s)")
        values = bwd_values(bufs, n=n, c=c, h=h, w=w)
        for candidate, spec in combos:
            artifact = _compile(
                spec, candidate, args=args, dims={**dims, "backward": 1}
            )
            ms = time_candidate(
                artifact,
                grid=batchnorm2d_bwd_grid(c, spec),
                block=(spec.block_size, 1, 1),
                args=values,
                warmup=args.warmup,
                iters=args.iters,
                timing=args.timing,
            )
            print(f"  {candidate.name}: {ms:.4f} ms")
            writer.row(key=key, candidate=candidate.name, ms=ms, backward=True, **dims)
            if best is None or ms < best[1]:
                best = (candidate.name, ms)
                best_run = _BwdLaunch(
                    artifact, values, batchnorm2d_bwd_grid(c, spec), spec.block_size
                )
    rc = 0
    if (
        args.validate
        and best is not None
        and not _validate_run(
            best_run,
            bufs,
            mode="bwd",
            layout="NCHW",
            dtype=args.dtype,
            key=key,
            writer=writer,
            label=best[0],
        )
    ):
        rc = 1
    writer.close()
    if best is not None:
        print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return rc


def _nhwc_bufs(args, mode):
    """Device tensors of one NHWC case (memory order N, H, W, C)."""
    if mode == "bwd":
        return _bwd_bufs(args, layout="NHWC")
    return _fwd_bufs(
        args.n,
        args.c,
        args.h,
        args.w,
        args.dtype,
        layout="NHWC",
        saved=(mode == "train"),
    )


def _main_nhwc(args) -> int:
    """``--layout NHWC``: time every NHWC (candidate, plan) of the requested direction.
    ``--split-order FILE`` gets one ``label<TAB>launches<TAB>kernels-per-launch`` line per entry.
    """
    import statistics
    from benchmarks.common.norm_kernel_cache import compile_nhwc_cached
    from benchmarks.common.norm_split_launch import NhwcRun, nhwc_plan_label
    from benchmarks.common.norm_sweep_space import batchnorm_nhwc_combos
    from rocke.runtime import time_launches_staged

    if args.cache_dir is None:
        print("--layout NHWC needs --cache-dir (AOT cache)", file=sys.stderr)
        return 1
    mode = "bwd" if args.backward else ("train" if args.training else "infer")
    n, c, h, w = args.n, args.c, args.h, args.w
    dims = {"n": n, "c": c, "h": h, "w": w}
    combos = batchnorm_nhwc_combos(
        n=n, c=c, h=h, w=w, dtype=args.dtype, arch=args.arch, mode=mode
    )
    if args.name_filter:
        combos = [
            cp for cp in combos if _name_ok(args.name_filter, nhwc_plan_label(*cp))
        ]
    if not combos:
        print("no valid NHWC candidates for this shape/dtype/arch", file=sys.stderr)
        return 1
    key = case_key(dims, dtype=args.dtype, family="batchnorm2d", extra=f"nhwc,{mode}")
    writer = JsonlResultsWriter(args.results)
    print(f"batchnorm2d NHWC {mode} sweep: {key}, {len(combos)} plan(s)")
    bufs = _nhwc_bufs(args, mode)
    ws, arts, order, best, best_run = None, {}, [], None, None
    for cand, plan in combos:
        need = cand.workspace_floats(c, plan)
        if need and (ws is None or ws.numel() < need):
            ws = NormBuffer((need,), "f32")
        for kind in cand.kinds:
            if (cand.name, kind) not in arts:
                arts[(cand.name, kind)] = compile_nhwc_cached(
                    cand,
                    kind,
                    dtype=args.dtype,
                    arch=args.arch,
                    cache_dir=args.cache_dir,
                )
        run = NhwcRun(
            cand,
            plan,
            [arts[(cand.name, k)] for k in cand.kinds],
            n=n,
            c=c,
            h=h,
            w=w,
            bufs=bufs,
            ws=ws,
        )
        ms = statistics.median(
            time_launches_staged(run, warmup=args.warmup, iters=args.iters)
        )
        label = nhwc_plan_label(cand, plan)
        order.append((label, args.warmup + args.iters, len(cand.kinds)))
        print(f"  {label}: {ms:.4f} ms")
        writer.row(key=key, candidate=label, ms=ms, nhwc=mode, **dims)
        if best is None or ms < best[1]:
            best = (label, ms)
            best_run = run
    if args.split_order:
        with open(args.split_order, "w") as fh:
            fh.write("".join(f"{lb}\t{k}\t{nk}\n" for lb, k, nk in order))
    rc = 0
    if (
        args.validate
        and best is not None
        and not _validate_run(
            best_run,
            bufs,
            mode=mode,
            layout="NHWC",
            dtype=args.dtype,
            key=key,
            writer=writer,
            label=best[0],
        )
    ):
        rc = 1
    writer.close()
    if best is not None:
        print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return rc


def _main_mcw(args) -> int:
    """``--mcw``: time every multi-channel-per-wave candidate of the requested direction.
    ``--split-order FILE`` gets one ``label<TAB>launches<TAB>kernels-per-launch`` line per candidate.
    """
    import statistics
    from benchmarks.common.norm_kernel_cache import compile_mcw_cached
    from benchmarks.common.norm_split_launch import McwRun
    from benchmarks.common.norm_sweep_space import batchnorm_mcw_combos
    from rocke.runtime import time_launches_staged

    if args.cache_dir is None:
        print("--mcw needs --cache-dir (AOT cache)", file=sys.stderr)
        return 1
    mode = "bwd" if args.backward else "train"
    n, c, h, w = args.n, args.c, args.h, args.w
    dims = {"n": n, "c": c, "h": h, "w": w}
    combos = batchnorm_mcw_combos(
        n=n, c=c, h=h, w=w, dtype=args.dtype, arch=args.arch, mode=mode
    )
    if args.name_filter:
        combos = [cd for cd in combos if _name_ok(args.name_filter, cd.name)]
    if not combos:
        print(
            "no valid multi-channel-per-wave candidates for this shape/dtype/arch",
            file=sys.stderr,
        )
        return 1
    key = case_key(dims, dtype=args.dtype, family="batchnorm2d", extra=f"mcw,{mode}")
    writer = JsonlResultsWriter(args.results)
    print(f"batchnorm2d mcw {mode} sweep: {key}, {len(combos)} candidate(s)")
    if mode == "bwd":
        bufs = _bwd_bufs(args)
    else:
        bufs = _fwd_bufs(n, c, h, w, args.dtype)
    order, best, best_run = [], None, None
    for cand in combos:
        art = compile_mcw_cached(
            cand, dtype=args.dtype, arch=args.arch, cache_dir=args.cache_dir
        )
        run = McwRun(cand, art, n=n, c=c, h=h, w=w, bufs=bufs)
        ms = statistics.median(
            time_launches_staged(run, warmup=args.warmup, iters=args.iters)
        )
        order.append((cand.name, args.warmup + args.iters, 1))
        print(f"  {cand.name}: {ms:.4f} ms")
        writer.row(key=key, candidate=cand.name, ms=ms, mcw=mode, **dims)
        if best is None or ms < best[1]:
            best = (cand.name, ms)
            best_run = run
    if args.split_order:
        with open(args.split_order, "w") as fh:
            fh.write("".join(f"{lb}\t{k}\t{nk}\n" for lb, k, nk in order))
    rc = 0
    if (
        args.validate
        and best is not None
        and not _validate_run(
            best_run,
            bufs,
            mode=mode,
            layout="NCHW",
            dtype=args.dtype,
            key=key,
            writer=writer,
            label=best[0],
        )
    ):
        rc = 1
    writer.close()
    if best is not None:
        print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return rc


def main(argv=None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    args.h *= args.d
    args.d = 1
    if args.mcw:
        return _main_mcw(args)
    if args.layout == "NHWC":
        return _main_nhwc(args)
    if args.backward:
        return _main_bwd(args)
    if args.split:
        return _main_split(args)

    combo_fn = batchnorm_dynamic_combos if args.dynamic else batchnorm_combos
    combos = combo_fn(
        n=args.n,
        c=args.c,
        h=args.h,
        w=args.w,
        dtype=args.dtype,
        arch=args.arch,
        training=args.training,
    )
    if args.name_filter:
        combos = [cs for cs in combos if _name_ok(args.name_filter, cs[0].name)]
    if not combos:
        print(
            "no valid batchnorm2d candidates for this shape/dtype/arch", file=sys.stderr
        )
        return 1

    dims = {"n": args.n, "c": args.c, "h": args.h, "w": args.w}
    key = case_key(
        dims, dtype=args.dtype, family="batchnorm2d", extra=f"training={args.training}"
    )
    writer = JsonlResultsWriter(args.results)

    print(f"batchnorm2d sweep: {key}, {len(combos)} candidate(s)")

    if args.compile_only:
        for candidate, spec in combos:
            _compile(spec, candidate, args=args, dims=dims)
            print(f"  compiled: {candidate.name}")
        writer.close()
        return 0

    from rocke.instances.common.batchnorm2d import batchnorm2d_grid

    best = None
    if args.fast_rank:
        fb = _fwd_bufs(args.n, args.c, args.h, args.w, args.dtype, saved=False)
        x, y, gamma, beta = fb["x"], fb["y"], fb["gamma"], fb["beta"]
        running_mean, running_var = fb["rm"], fb["rv"]
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
        _saved = _add_saved_stats(
            values, training=args.training, c=args.c, dtype=args.dtype
        )

        plans = []
        for candidate, spec in combos:
            artifact = _compile(spec, candidate, args=args, dims=dims)
            grid = batchnorm2d_grid(args.c, spec, n=args.n, h=args.h, w=args.w)
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
            writer.row(
                key=key,
                candidate=label,
                block_size=spec.block_size,
                vec=spec.vec,
                training=args.training,
                dtype=args.dtype,
                ms=r.ms,
                **dims,
            )
            if best is None or r.ms < best[1]:
                best = (label, r.ms)
                best_idx = r.index

        if args.validate and best_idx is not None:
            _, artifact, grid, block, _values = plans[best_idx]
            _validate_batchnorm(
                artifact,
                grid=grid,
                block=block,
                x=x,
                y=y,
                gamma=gamma,
                beta=beta,
                running_mean=running_mean,
                running_var=running_var,
                training=args.training,
                dtype=args.dtype,
                key=key,
                writer=writer,
            )
    else:
        best_validate = None
        for candidate, spec in combos:
            label = candidate.name
            artifact = _compile(spec, candidate, args=args, dims=dims)

            fb = _fwd_bufs(args.n, args.c, args.h, args.w, args.dtype, saved=False)
            x, y, gamma, beta = fb["x"], fb["y"], fb["gamma"], fb["beta"]
            running_mean, running_var = fb["rm"], fb["rv"]

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
            _saved = _add_saved_stats(
                values, training=args.training, c=args.c, dtype=args.dtype
            )

            grid = batchnorm2d_grid(args.c, spec, n=args.n, h=args.h, w=args.w)
            block = (spec.block_size, 1, 1)
            ms = time_candidate(
                artifact,
                grid=grid,
                block=block,
                args=values,
                warmup=args.warmup,
                iters=args.iters,
                timing=args.timing,
            )
            print(f"  {label}: {ms:.4f} ms")
            writer.row(
                key=key,
                candidate=label,
                block_size=spec.block_size,
                vec=spec.vec,
                training=args.training,
                dtype=args.dtype,
                ms=ms,
                **dims,
            )
            if best is None or ms < best[1]:
                best = (label, ms)
                best_validate = (
                    artifact,
                    grid,
                    block,
                    x,
                    y,
                    gamma,
                    beta,
                    running_mean,
                    running_var,
                )

        if args.validate and best is not None:
            artifact, grid, block, x, y, gamma, beta, running_mean, running_var = (
                best_validate
            )
            _validate_batchnorm(
                artifact,
                grid=grid,
                block=block,
                x=x,
                y=y,
                gamma=gamma,
                beta=beta,
                running_mean=running_mean,
                running_var=running_var,
                training=args.training,
                dtype=args.dtype,
                key=key,
                writer=writer,
            )

    writer.close()
    if best is not None:
        print(f"Best: {best[0]} ({best[1]:.4f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
