# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Research spike: native-NCHW forward conv vs. transpose + NHWC conv + transpose.

One apples-to-apples forward measurement. An NCHW caller's pipeline is

1. transpose ``X`` NCHW -> NHWC (``batched_transpose2d`` over ``[C, H*W]``),
2. production NHWC :mod:`kernels.common.conv_direct_nongrouped`,
3. transpose ``Y`` NHWK -> NKHW (``[Ho*Wo, K]``),

against two native kernels that read NCHW and write NKHW directly:

* **dgrad-as-fwd** (stride 1 only): :mod:`kernels.common.conv_direct_nchw_dgrad`
  run as the forward conv of the transposed problem (weights flipped and
  transposed at plan time, pad ``KH-1-PAD``);
* **fwd**: :mod:`kernels.common.conv_direct_nchw_fwd`, the same design
  with the stride on the input side (stride 1 and 2).

Weights are repacked once at plan time for every arm and the repack is not
timed. Each arm is tuned with the same protocol:

* **NHWC**: ``nongrouped_knobs`` + ``nongrouped_split_knobs`` for the shape's
  ``Wo`` (the split-C kernels are timed with their reduce pass), the shipped
  gfx950 fwd dispatch instances of the family for the filter/pad/stride, and
  the spec default the earlier forward benchmark used (reported separately as
  ``default``). A lever pass then crosses the screened-fastest configs with
  ``waves_per_eu``, ``double_buffer`` and ``iglp``.
* **native**: a geometry grid, then the same lever pass over its fastest
  configs (``waves_per_eu``, ``pin_prefetch``, LDS pads, ``col_split``).

Every candidate is screened with back-to-back timing, then the ``--topk``
fastest per arm are verified on-device (NaN-poisoned output) against
``torch.nn.functional.conv2d`` in NCHW and timed with ``--timing`` (default:
staged, a median of stalled per-launch samples). The pipeline is verified end
to end into NKHW. Output is ratios only.

Usage (from ``library/``)::

    python -m benchmarks.common.benchmark_fwd_nchw_native_vs_transpose --compile-only
    python -m benchmarks.common.benchmark_fwd_nchw_native_vs_transpose --shapes 0

Run one shape per process for full sweeps (each loads thousands of code
objects). Compiled candidates are cached under ``--cache-dir``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.common.benchmark_dgrad_nchw_proto_vs_transpose import (
    _NATIVE_GRID as _DGRAD_GRID,
)
from benchmarks.common.benchmark_dgrad_nchw_proto_vs_transpose import (
    Cand,
    _fenced,
    _launcher,
    _screen,
    _verify_and_time,
)
from benchmarks.common.benchmark_nchw_proto_vs_transpose import (
    _TRANSPOSE_TILE,
    Shape,
    _bind_transpose,
    _build_transpose,
    _fit_to_arch,
    _time_us,
    _u8,
)
from kernels.common.conv_direct_nchw_dgrad import (
    DirectNchwDgradSpec,
    build_direct_conv_nchw_dgrad,
    is_valid_direct_nchw_dgrad_spec,
)
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_nchw_fwd import (
    DirectNchwFwdSpec,
    build_direct_conv_nchw_fwd,
    is_valid_direct_nchw_fwd_spec,
    with_conflict_free_pad,
)

_ARCH = "gfx950"
_DTYPE = "fp16"
_TOL = 5e-3


def _grid(ns: List[int]) -> List[Shape]:
    out = []
    for N in ns:
        for C in (64, 256):
            for HW in (64, 128):
                for stride in (1, 2):
                    out.append(Shape(N=N, H=HW, W=HW, C=C, K=C, stride=stride))
    return out


def _problem(s: Shape, **kw) -> DirectConvProblem:
    d = dict(
        N=s.N,
        H=s.H,
        W=s.W,
        groups=1,
        cpg=s.C,
        kpg=s.K,
        KH=s.KH,
        KW=s.KW,
        PAD=s.PAD,
        stride=s.stride,
        dtype=_DTYPE,
    )
    d.update(kw)
    return DirectConvProblem(**d)


def _dgrad_problem(s: Shape) -> DirectConvProblem:
    """Stride-1 forward ``s`` as the dgrad prototype's problem (dX = Y)."""
    p = _problem(s)
    return _problem(s, H=p.Ho, W=p.Wo, cpg=s.K, kpg=s.C, PAD=s.KH - 1 - s.PAD, stride=1)


def _probe(stride: int) -> Shape:
    return Shape(N=1, H=64, W=64, C=64, K=64, stride=stride)


# ---------------------------------------------------------------------------
# Candidate sets
# ---------------------------------------------------------------------------

_NHWC_LEVERS = dict(
    waves_per_eu=(None, 2, 3, 4), double_buffer=(False, True), iglp=(None, 0)
)

_FWD_GRID = dict(
    atom=("32x32x16", "16x16x32"),
    tile_j=(32, 64),
    tile_i=(1, 2, 4, 8, 16),
    tile_k=(32, 64, 128),
    ck=(16, 32, 64),
    waves=((1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2)),
    stage_group=(4, 8),
)
_FWD_LEVERS = dict(
    waves_per_eu=(0, 2, 3, 4), pin_prefetch=(False, True), pad_extra=(0, 16)
)


def _kw_key(kw: dict) -> str:
    return repr(sorted(kw.items()))


def _prod_knobs(stride: int, KH: int, PAD: int) -> List[dict]:
    """Shipped gfx950 fwd ``direct_nongrouped`` fp16 instances for the case."""
    path = (
        Path(__file__).resolve().parents[2]
        / "dispatch"
        / "grouped_convolution_instances"
        / "gfx950_fwd.json"
    )
    with open(path) as f:
        data = json.load(f)
    out = []
    for inst in data["instances"]:
        if (
            inst["algorithm"] == "direct_nongrouped"
            and inst["dtype"] == _DTYPE
            and inst["filter"] == [KH, KH]
            and inst["pad"] == [PAD, PAD]
            and inst["stride"] == [stride, stride]
            and inst.get("dilation", [1, 1]) == [1, 1]
        ):
            out.append(dict(inst["knobs"]))
    return out


def _default_nhwc_kw(s: Shape) -> dict:
    """The single spec the earlier forward benchmark timed (spec defaults,
    ``ck`` / ``tile_h`` shrunk until it fits)."""
    from kernels.common.conv_direct_nongrouped import (
        DirectNongroupedConvSpec,
        is_valid_nongrouped_spec,
    )

    spec, _ok = _fit_to_arch(
        DirectNongroupedConvSpec(problem=_problem(s)), _ARCH, is_valid_nongrouped_spec
    )
    return dict(ck=spec.ck, tile_h=spec.tile_h)


def _nhwc_base(shapes: List[Shape]) -> List[dict]:
    from kernels.common.conv_direct_nongrouped import (
        DirectNongroupedConvSpec,
        is_valid_nongrouped_spec,
        nongrouped_knobs,
        nongrouped_split_knobs,
    )

    out, seen = [], set()
    for s in shapes:
        p = _problem(s)
        kws = list(nongrouped_knobs(_ARCH, _DTYPE, Wo=p.Wo))
        kws += list(nongrouped_split_knobs(_ARCH, _DTYPE, Wo=p.Wo))
        kws += _prod_knobs(s.stride, s.KH, s.PAD)
        kws.append(_default_nhwc_kw(s))
        for kw in kws:
            k = _kw_key(kw)
            if k in seen:
                continue
            if is_valid_nongrouped_spec(
                DirectNongroupedConvSpec(problem=p, **kw), _ARCH
            )[0]:
                seen.add(k)
                out.append(kw)
    return out


def _nhwc_levers(kw: dict) -> List[dict]:
    out = []
    for we, db, ig in itertools.product(*_NHWC_LEVERS.values()):
        if kw.get("split_c", 1) > 1 and db:
            continue
        out.append(dict(kw, waves_per_eu=we, double_buffer=db, iglp=ig))
    return out


def _dgrad_specs() -> List[DirectNchwDgradSpec]:
    p = _dgrad_problem(_probe(1))
    g = _DGRAD_GRID
    out = []
    for atom, tj, ti, tc, ck, (wm, wn), sg in itertools.product(
        g["atom"],
        g["tile_j"],
        g["tile_i"],
        g["tile_c"],
        g["ck"],
        g["waves"],
        g["stage_group"],
    ):
        spec = DirectNchwDgradSpec(
            problem=p,
            tile_i=ti,
            tile_j=tj,
            tile_c=tc,
            ck=ck,
            waves_m=wm,
            waves_n=wn,
            atom=atom,
            stage_group=sg,
        )
        if is_valid_direct_nchw_dgrad_spec(spec, _ARCH)[0]:
            out.append(spec)
    return out


def _fwd_specs(stride: int) -> List[DirectNchwFwdSpec]:
    p = _problem(_probe(stride))
    g = _FWD_GRID
    out = []
    for atom, tj, ti, tk, ck, (wm, wn), sg in itertools.product(
        g["atom"],
        g["tile_j"],
        g["tile_i"],
        g["tile_k"],
        g["ck"],
        g["waves"],
        g["stage_group"],
    ):
        spec = DirectNchwFwdSpec(
            problem=p,
            tile_i=ti,
            tile_j=tj,
            tile_k=tk,
            ck=ck,
            waves_m=wm,
            waves_n=wn,
            atom=atom,
            stage_group=sg,
            col_split=stride == 2,
        )
        spec = with_conflict_free_pad(spec)
        if is_valid_direct_nchw_fwd_spec(spec, _ARCH)[0]:
            out.append(spec)
    return out


def _fwd_levers(spec: DirectNchwFwdSpec) -> List[DirectNchwFwdSpec]:
    out = []
    splits = (True, False) if spec.S == 2 else (False,)
    for we, pin, extra, cs in itertools.product(*_FWD_LEVERS.values(), splits):
        s2 = replace(spec, waves_per_eu=we, pin_prefetch=pin, col_split=cs)
        s2 = with_conflict_free_pad(s2)
        s2 = replace(s2, lds_pad=s2.lds_pad + extra)
        if is_valid_direct_nchw_fwd_spec(s2, _ARCH)[0]:
            out.append(s2)
    return out


# ---- parallel compile (top-level workers for pickling) ---------------------


def _compile_nhwc(item):
    kw, stride = item
    try:
        from kernels.common.conv_direct_nongrouped import (
            DirectNongroupedConvSpec,
            build_direct_conv_nongrouped,
        )
        from rocke import compile_kernel

        spec = DirectNongroupedConvSpec(problem=_problem(_probe(stride)), **kw)
        art = compile_kernel(build_direct_conv_nongrouped(spec, arch=_ARCH), arch=_ARCH)
        return _kw_key(kw), (bytes(art.hsaco), art.kernel_name, spec.threads_per_block)
    except Exception:  # noqa: BLE001 - an invalid combination is skipped
        return None


def _compile_dgrad(spec):
    try:
        from rocke import compile_kernel

        art = compile_kernel(build_direct_conv_nchw_dgrad(spec, arch=_ARCH), arch=_ARCH)
        return spec.kernel_name(), (
            bytes(art.hsaco),
            art.kernel_name,
            spec.threads_per_block,
        )
    except Exception:  # noqa: BLE001
        return None


def _compile_fwd(spec):
    try:
        from rocke import compile_kernel

        art = compile_kernel(build_direct_conv_nchw_fwd(spec, arch=_ARCH), arch=_ARCH)
        return spec.kernel_name(), (
            bytes(art.hsaco),
            art.kernel_name,
            spec.threads_per_block,
        )
    except Exception:  # noqa: BLE001
        return None


def _compile_all(items, worker, cache_file: Path, jobs: int) -> Dict[str, tuple]:
    cache: Dict[str, tuple] = {}
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    todo = [it for it, key in items if key not in cache]
    if todo:
        t0 = time.time()
        print(f"  compiling {len(todo)} kernels -> {cache_file.name} ...", flush=True)
        # spawn: the per-shape lever pass compiles after the HIP context exists.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs or os.cpu_count(), mp_context=ctx
        ) as pool:
            for r in pool.map(worker, todo, chunksize=4):
                if r is not None:
                    cache[r[0]] = r[1]
        print(f"  compiled in {time.time() - t0:.0f}s", flush=True)
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
    return cache


# ---------------------------------------------------------------------------
# One shape
# ---------------------------------------------------------------------------


def _top(cands: List[Cand], n: int) -> List[Cand]:
    ok = [c for c in cands if c.screen_us < float("inf")]
    return sorted(ok, key=lambda c: c.screen_us)[:n]


def _run_case(s: Shape, args, cache_dir: Path) -> Optional[dict]:
    import torch

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_nongrouped import (
        DirectNongroupedConvSpec,
        DirectSplitReduceSpec,
        build_direct_conv_split_reduce,
        is_valid_nongrouped_spec,
    )
    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import LaunchConfig

    p = _problem(s)
    N, C, K, H, W = s.N, s.C, s.K, s.H, s.W
    Ho, Wo = p.Ho, p.Wo
    for m in (C, H * W, K, Ho * Wo):
        if m % _TRANSPOSE_TILE:
            print(f"  [skip] {s.label()}: {m} not a multiple of the transpose tile")
            return None

    torch.manual_seed(0)
    X = torch.empty(N, C, H, W, dtype=torch.float16).uniform_(-1.0, 1.0)
    Wk = torch.empty(K, C, s.KH, s.KW, dtype=torch.float16).uniform_(-1.0, 1.0)
    with torch.no_grad():
        ref = torch.nn.functional.conv2d(
            X.float(), Wk.float(), stride=s.stride, padding=s.PAD
        )
    ref_nhwc = ref.permute(0, 2, 3, 1).contiguous()
    floor = float(ref.abs().max().clamp(min=1.0))
    W_krsc = Wk.permute(0, 2, 3, 1).contiguous()  # NHWC conv + native fwd
    W_flip = Wk.flip(2, 3).permute(0, 2, 3, 1).contiguous()  # dgrad-as-fwd

    rt = Runtime()
    x_bytes, w_bytes, y_bytes = X.nbytes, Wk.nbytes, N * K * Ho * Wo * 2
    X_dev = rt.alloc(x_bytes)  # NCHW
    Xt_dev = rt.alloc(x_bytes)  # NHWC
    Wn_dev = rt.alloc(w_bytes)  # KRSC
    Wf_dev = rt.alloc(w_bytes)  # flipped KRSC (dgrad-as-fwd)
    Y_dev = rt.alloc(y_bytes)  # NKHW
    Yh_dev = rt.alloc(y_bytes)  # NHWK
    ws_bytes = 4 * 16 * N * Ho * Wo * K
    WS_dev = rt.alloc(ws_bytes)
    out = torch.empty(N, K, Ho, Wo, dtype=torch.float16)
    out_nhwc = torch.empty(N, Ho, Wo, K, dtype=torch.float16)

    def check(dev, host, want):
        rt.memcpy_d2h(_u8(host), dev, host.nbytes)
        if not torch.isfinite(host.float()).all():
            return False, float("inf")
        rel = float((host.float() - want).abs().max() / floor)
        return rel < _TOL, rel

    def poison(dev):
        host = torch.full((y_bytes // 2,), float("nan"), dtype=torch.float16)
        rt.memcpy_h2d(dev, _u8(host), host.nbytes)

    def timed(c: Optional[Cand]) -> Optional[Cand]:
        if c is not None:
            c.t_us = _time_us(
                c.run, timing=args.timing, warmup=args.warmup, iters=args.iters
            )
        return c

    dsig = conv_direct_args_signature(_DTYPE)
    res: dict = dict(label=s.label())
    try:
        rt.memcpy_h2d(X_dev, _u8(X), x_bytes)
        rt.memcpy_h2d(Wn_dev, _u8(W_krsc), w_bytes)
        rt.memcpy_h2d(Wf_dev, _u8(W_flip), w_bytes)

        # ---- transposes -------------------------------------------------
        in_art, _, in_grid, in_block, in_sig = _build_transpose(C, H * W, N, _ARCH)
        out_art, _, out_grid, out_block, out_sig = _build_transpose(
            Ho * Wo, K, N, _ARCH
        )
        run_in, in_ln, in_vals, in_cfg = _bind_transpose(
            in_art, in_sig, in_grid, in_block, X_dev, Xt_dev, C, H * W, N
        )
        run_out, out_ln, out_vals, out_cfg = _bind_transpose(
            out_art, out_sig, out_grid, out_block, Yh_dev, Y_dev, Ho * Wo, K, N
        )
        _fenced(in_ln, in_vals, in_cfg)()  # X^T is the NHWC arm's input

        # ---- NHWC arm ---------------------------------------------------
        reducers: Dict[int, tuple] = {}

        def nhwc_cand(kw, bins, tag):
            spec = DirectNongroupedConvSpec(problem=p, **kw)
            key = _kw_key(kw)
            if key not in bins or not is_valid_nongrouped_spec(spec, _ARCH)[0]:
                return None
            hsaco, name, block = bins[key]
            ln = _launcher(hsaco, name, dsig, key=f"nhwc{s.stride}:" + key)
            cfg = LaunchConfig(grid=spec.grid(), block=(block, 1, 1))
            sc = spec.split_c
            if sc == 1:
                vals = ConvArgs.from_problem(p).to_launch_values(
                    int(Xt_dev), int(Wn_dev), int(Yh_dev), x_bytes, w_bytes, y_bytes
                )
                c = Cand(
                    f"{tag}|{name}", ln.bind(vals, config=cfg), _fenced(ln, vals, cfg)
                )
            else:
                if sc not in reducers:
                    rs = DirectSplitReduceSpec(dtype=_DTYPE, split_c=sc)
                    ra = compile_kernel(build_direct_conv_split_reduce(rs), arch=_ARCH)
                    rl = _launcher(ra.hsaco, ra.kernel_name, dsig)
                    rv = ConvArgs.from_problem(p).to_launch_values(
                        int(WS_dev),
                        int(Wn_dev),
                        int(Yh_dev),
                        ws_bytes,
                        w_bytes,
                        y_bytes,
                    )
                    rc = LaunchConfig(grid=rs.grid(p), block=(rs.threads, 1, 1))
                    reducers[sc] = (rl.bind(rv, config=rc), _fenced(rl, rv, rc))
                vals = ConvArgs.from_problem(p).to_launch_values(
                    int(Xt_dev), int(Wn_dev), int(WS_dev), x_bytes, w_bytes, ws_bytes
                )
                conv_r, conv_f = ln.bind(vals, config=cfg), _fenced(ln, vals, cfg)
                red_r, red_f = reducers[sc]

                def run(a=conv_r, b=red_r):
                    a()
                    b()

                def run_f(a=conv_f, b=red_f):
                    a()
                    b()

                c = Cand(f"{tag}|{name}", run, run_f)
            c.kw = kw  # type: ignore[attr-defined]
            return c

        nhwc_bins = _compile_all([], None, cache_dir / f"nhwc_s{s.stride}.pkl", 0)
        base_kws = _nhwc_base([s])
        hc = [c for c in (nhwc_cand(kw, nhwc_bins, "base") for kw in base_kws) if c]
        dkw = _default_nhwc_kw(s)
        default_c = nhwc_cand(dkw, nhwc_bins, "default")
        print(f"  nhwc: screening {len(hc)} base candidates", flush=True)
        _screen(hc, args.screen_iters)
        lev_kws, seen = [], {_kw_key(kw) for kw in base_kws}
        for c in _top(hc, args.lever_top):
            for kw in _nhwc_levers(c.kw):  # type: ignore[attr-defined]
                if _kw_key(kw) not in seen:
                    seen.add(_kw_key(kw))
                    lev_kws.append(kw)
        nhwc_bins = _compile_all(
            [((kw, s.stride), _kw_key(kw)) for kw in lev_kws],
            _compile_nhwc,
            cache_dir / f"nhwc_s{s.stride}.pkl",
            args.jobs,
        )
        hl = [c for c in (nhwc_cand(kw, nhwc_bins, "lever") for kw in lev_kws) if c]
        print(f"  nhwc: screening {len(hl)} lever candidates", flush=True)
        _screen(hl, args.screen_iters)
        best_nhwc = _verify_and_time(
            hc + hl,
            lambda: poison(Yh_dev),
            lambda: check(Yh_dev, out_nhwc, ref_nhwc),
            args,
            args.topk,
        )
        if default_c is not None:
            poison(Yh_dev)
            default_c.run_fenced()
            ok, _ = check(Yh_dev, out_nhwc, ref_nhwc)
            default_c = default_c if ok else None

        # ---- native fwd arm ----------------------------------------------
        def fwd_cand(spec, bins, tag):
            sp = replace(spec, problem=p)
            key = spec.kernel_name()
            if key not in bins or not is_valid_direct_nchw_fwd_spec(sp, _ARCH)[0]:
                return None
            hsaco, name, block = bins[key]
            ln = _launcher(hsaco, name, dsig, key="fwd:" + key)
            vals = ConvArgs.from_problem(p).to_launch_values(
                int(X_dev), int(Wn_dev), int(Y_dev), x_bytes, w_bytes, y_bytes
            )
            cfg = LaunchConfig(grid=sp.grid(), block=(block, 1, 1))
            c = Cand(f"{tag}|{key}", ln.bind(vals, config=cfg), _fenced(ln, vals, cfg))
            c.spec = spec  # type: ignore[attr-defined]
            return c

        fpath = cache_dir / f"fwd_s{s.stride}.pkl"
        fwd_bins = _compile_all([], None, fpath, 0)
        fspecs = _fwd_specs(s.stride)
        fc = [c for c in (fwd_cand(sp, fwd_bins, "base") for sp in fspecs) if c]
        print(f"  fwd: screening {len(fc)} base candidates", flush=True)
        _screen(fc, args.screen_iters)
        lev, seen = [], {sp.kernel_name() for sp in fspecs}
        for c in _top(fc, args.lever_top):
            for sp in _fwd_levers(c.spec):  # type: ignore[attr-defined]
                if sp.kernel_name() not in seen:
                    seen.add(sp.kernel_name())
                    lev.append(sp)
        fwd_bins = _compile_all(
            [(sp, sp.kernel_name()) for sp in lev], _compile_fwd, fpath, args.jobs
        )
        fl = [c for c in (fwd_cand(sp, fwd_bins, "lever") for sp in lev) if c]
        print(f"  fwd: screening {len(fl)} lever candidates", flush=True)
        _screen(fl, args.screen_iters)
        best_fwd = _verify_and_time(
            fc + fl,
            lambda: poison(Y_dev),
            lambda: check(Y_dev, out, ref),
            args,
            args.topk,
        )
        best_fwd_base = _verify_and_time(
            fc,
            lambda: poison(Y_dev),
            lambda: check(Y_dev, out, ref),
            args,
            max(1, args.topk // 2),
        )

        # ---- native dgrad-as-fwd arm (stride 1) ---------------------------
        best_dg = None
        if s.stride == 1:
            dp = _dgrad_problem(s)
            dg_bins = _compile_all([], None, cache_dir / "dgrad_as_fwd.pkl", 0)
            dc = []
            for spec in _dgrad_specs():
                sp = replace(spec, problem=dp)
                key = spec.kernel_name()
                if (
                    key not in dg_bins
                    or not is_valid_direct_nchw_dgrad_spec(sp, _ARCH)[0]
                ):
                    continue
                hsaco, name, block = dg_bins[key]
                ln = _launcher(hsaco, name, dsig, key="dg:" + key)
                vals = ConvArgs.from_problem(dp).to_launch_values(
                    int(X_dev), int(Wf_dev), int(Y_dev), x_bytes, w_bytes, y_bytes
                )
                cfg = LaunchConfig(grid=sp.grid(), block=(block, 1, 1))
                dc.append(Cand(key, ln.bind(vals, config=cfg), _fenced(ln, vals, cfg)))
            print(f"  dgrad-as-fwd: screening {len(dc)} candidates", flush=True)
            _screen(dc, args.screen_iters)
            best_dg = _verify_and_time(
                dc,
                lambda: poison(Y_dev),
                lambda: check(Y_dev, out, ref),
                args,
                args.topk,
            )

        # ---- pipeline end to end + final timings (one session) ------------
        pipe_ok = False
        if best_nhwc is not None:
            poison(Y_dev)
            poison(Yh_dev)
            _fenced(in_ln, in_vals, in_cfg)()
            best_nhwc.run_fenced()
            _fenced(out_ln, out_vals, out_cfg)()
            pipe_ok, _ = check(Y_dev, out, ref)
        if not pipe_ok:
            print("  [FAIL] pipeline verification", flush=True)
            return None
        t_in = _time_us(
            run_in, timing=args.timing, warmup=args.warmup, iters=args.iters
        )
        t_out = _time_us(
            run_out, timing=args.timing, warmup=args.warmup, iters=args.iters
        )
        for c in (best_nhwc, default_c, best_fwd, best_fwd_base, best_dg):
            timed(c)

        def one(cv=best_nhwc.run):
            run_in()
            cv()
            run_out()

        t_one = _time_us(one, timing=args.timing, warmup=args.warmup, iters=args.iters)
        synchronize_and_release(0)
    finally:
        for buf in (X_dev, Xt_dev, Wn_dev, Wf_dev, Y_dev, Yh_dev, WS_dev):
            rt.free(buf)

    pipe = t_in + best_nhwc.t_us + t_out
    natives = [c for c in (best_fwd, best_dg) if c is not None]
    best_native = min(natives, key=lambda c: c.t_us) if natives else None
    res.update(
        nhwc=best_nhwc.label,
        fwd=best_fwd.label if best_fwd else None,
        dg=best_dg.label if best_dg else None,
        default=default_c.label if default_c else None,
        # ratios only
        xp_share=(t_in + t_out) / pipe,
        pipe1_over_pipe=t_one / pipe,
        default_over_nhwc=(default_c.t_us / best_nhwc.t_us) if default_c else None,
        xp_share_default=(
            (t_in + t_out) / (t_in + default_c.t_us + t_out) if default_c else None
        ),
    )
    for tag, c in (("fwd", best_fwd), ("fwd_base", best_fwd_base), ("dg", best_dg)):
        if c is None:
            continue
        res[f"spd_{tag}"] = pipe / c.t_us
        res[f"spd1_{tag}"] = t_one / c.t_us
        res[f"{tag}_over_nhwc"] = c.t_us / best_nhwc.t_us
    if best_native is not None:
        res["spd_best"] = pipe / best_native.t_us
        res["best_over_nhwc"] = best_native.t_us / best_nhwc.t_us
        res["best_arm"] = "fwd" if best_native is best_fwd else "dgrad-as-fwd"
    if best_fwd is not None and best_dg is not None:
        res["fwd_over_dg"] = best_fwd.t_us / best_dg.t_us
    return res


def _print(results: List[dict]) -> None:
    print("\n" + "=" * 100)
    print("NATIVE NCHW FWD vs TRANSPOSE + NHWC conv_direct_nongrouped + TRANSPOSE")
    print("spd_* = (x_in + nhwc + x_out) / native  (>1: native wins)")
    print("spd1_* = the three pipeline launches timed as one sample / native")
    print("*_over_nhwc = native / NHWC conv stage alone; xp = transposes share")
    print("=" * 100)
    for r in results:
        print(
            json.dumps(r, default=lambda x: round(x, 3) if isinstance(x, float) else x)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--N", default="1")
    ap.add_argument("--shapes", default=None, help="comma-separated grid indices")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument(
        "--cache-dir",
        default=str(Path(tempfile.gettempdir()) / "rocke_fwd_nchw_spike"),
    )
    ap.add_argument("--timing", choices=("staged", "block"), default="staged")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--screen-iters", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lever-top", type=int, default=12)
    ap.add_argument("--json-out", default=None, help="append result rows (ratios)")
    args = ap.parse_args()

    shapes = _grid([int(n) for n in args.N.split(",")])
    if args.shapes:
        shapes = [shapes[int(i)] for i in args.shapes.split(",")]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.compile_only:
        for st in sorted({s.stride for s in shapes}):
            kws = _nhwc_base([s for s in shapes if s.stride == st])
            print(f"stride {st}: {len(kws)} NHWC base knob sets")
            _compile_all(
                [((kw, st), _kw_key(kw)) for kw in kws],
                _compile_nhwc,
                cache_dir / f"nhwc_s{st}.pkl",
                args.jobs,
            )
            fs = _fwd_specs(st)
            print(f"stride {st}: {len(fs)} native fwd base specs")
            _compile_all(
                [(sp, sp.kernel_name()) for sp in fs],
                _compile_fwd,
                cache_dir / f"fwd_s{st}.pkl",
                args.jobs,
            )
        if any(s.stride == 1 for s in shapes):
            ds = _dgrad_specs()
            print(f"dgrad-as-fwd: {len(ds)} specs")
            _compile_all(
                [(sp, sp.kernel_name()) for sp in ds],
                _compile_dgrad,
                cache_dir / "dgrad_as_fwd.pkl",
                args.jobs,
            )
        return 0

    import torch

    torch.cuda.is_available()  # claim the HIP context for torch first
    print(f"arch={_ARCH} timing={args.timing} warmup={args.warmup} iters={args.iters}")
    results = []
    for s in shapes:
        print(f"\n--- {s.label()} ---", flush=True)
        r = _run_case(s, args, cache_dir)
        if r is not None:
            results.append(r)
            if args.json_out:
                with open(args.json_out, "a") as f:
                    f.write(json.dumps(r) + "\n")
    _print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
