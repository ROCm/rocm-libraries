# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Research spike: native-NCHW dgrad vs. transpose + NHWC dgrad + transpose.

:mod:`kernels.common.conv_direct_nchw_dgrad` is an experimental dgrad kernel
that reads ``dY`` in NKHW and writes ``dX`` in NCHW. An NCHW caller's only
alternative today is the pipeline

1. transpose ``dY`` NKHW -> NHWK (``batched_transpose2d`` over the per-image
   ``[K, Ho*Wo]`` matrix),
2. production NHWC implicit-GEMM dgrad
   (:mod:`kernels.common.conv_implicit_gemm_dgrad`),
3. transpose ``dX`` NHWC -> NCHW (per-image ``[H*W, C]`` matrix).

Weights are repacked once at plan time for both arms (KCYX -> KYXC for the
NHWC kernel, KCYX -> CYXK for the prototype) and the repack is not part of
either timed path; its per-call on-device cost (``permute_nd``, and the
prototype's repack as a plain 2D ``batched_transpose2d`` too) is reported
separately.

Fair tuning
-----------
* **NHWC baseline**: every binary the production AOT generator
  (:mod:`kernels.conv.aot_jobs`) builds for gfx950 fp16 dgrad at the widest
  vectors (A = B = D = 8, which every shape here admits) and that serves the
  shape (``identity_supports_problem``): the stride-1 tap-outer binaries, the
  dY-halo binaries, the tilde binaries (all strides) and the split-K tilde
  covers, every pipeline / epilogue / tile / warp geometry of the cache grid.
  The ungrouped variant of each is built (the cache ships the grouped one,
  which also serves ``groups == 1``; ungrouped is the faster of the two here).
* **Native prototype**: a grid over its knobs (tile, ``ck``, waves, atom,
  staging group).

Each candidate is first screened with back-to-back timing
(:func:`rocke.runtime.launcher.time_launches`); the ``--topk`` fastest per arm
are then verified on-device against a ``torch.nn.grad.conv2d_input``
reference and timed with ``--timing`` (default: staged, a median of stalled
per-launch samples). The best *verified* candidate of each arm is reported.
The full pipeline is also verified end to end (NCHW out) and timed both
stage by stage and as one three-launch sample.

Usage (from ``library/``)::

    python -m benchmarks.common.benchmark_dgrad_nchw_proto_vs_transpose --compile-only
    python -m benchmarks.common.benchmark_dgrad_nchw_proto_vs_transpose --shapes 0,1

Compiled candidates are cached under ``--cache-dir`` (default: a folder in
the system temp directory) so GPU runs can be split per shape.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from benchmarks.common.benchmark_nchw_proto_vs_transpose import (
    _TRANSPOSE_TILE,
    Shape,
    _bind_transpose,
    _build_transpose,
    _time_us,
    _u8,
)
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_nchw_dgrad import (
    DirectNchwDgradSpec,
    build_direct_conv_nchw_dgrad,
    is_valid_direct_nchw_dgrad_spec,
)

_ARCH = "gfx950"
_DTYPE = "fp16"
_TOL = 5e-3  # fp16, relative to max(|ref|, 1); matches test_conv_direct_nchw_dgrad.py


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def _grid(ns: List[int]) -> List[Shape]:
    out = []
    for N in ns:
        for C in (64, 256):
            for HW in (64, 128):
                for stride in (1, 2):
                    out.append(Shape(N=N, H=HW, W=HW, C=C, K=C, stride=stride))
    return out


def _direct_problem(s: Shape) -> DirectConvProblem:
    return DirectConvProblem(
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


def _igemm_problem(s: Shape):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return ConvProblem(
        N=s.N,
        Hi=s.H,
        Wi=s.W,
        C=s.C,
        K=s.K,
        Y=s.KH,
        X=s.KW,
        sH=s.stride,
        sW=s.stride,
        pH=s.PAD,
        pW=s.PAD,
    )


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _nhwc_jobs(shapes: List[Shape]) -> list:
    """Production AOT dgrad jobs (widest vectors) serving any of ``shapes``."""
    import kernels.conv.aot_jobs as aj
    from kernels.conv.kernel_identity import identity_supports_problem
    from rocke.core.arch import ArchTarget

    saved = aj._GROUPED
    aj._GROUPED = (False,)  # ungrouped variants; the halo jobs stay grouped
    try:
        jobs = list(
            aj._direction_jobs(
                "dgrad",
                _ARCH,
                _DTYPE,
                ArchTarget.from_gfx(_ARCH),
                (),
                aj.CACHE_DGRAD_MAX_SUB_GEMMS,
            )
        )
    finally:
        aj._GROUPED = saved
    probs = [_igemm_problem(s) for s in shapes]
    out, seen = [], set()
    for j in jobs:
        k = j.spec_kwargs
        if (
            k.get("vector_size_a") != 8
            or k.get("vector_size_b") != 8
            or j.caps.get("patch")
        ):
            continue
        if not any(identity_supports_problem(j.identity, p)[0] for p in probs):
            continue
        if aj._aot_excluded(j, _ARCH) or not aj._spec_is_valid(j, _ARCH, _DTYPE):
            continue
        key = repr(j.identity)
        if key not in seen:
            seen.add(key)
            out.append(j)
    return out


_NATIVE_GRID = dict(
    atom=("32x32x16", "16x16x32"),
    tile_j=(32, 64),
    tile_i=(1, 2, 4, 8, 16),
    tile_c=(32, 64, 128),
    ck=(16, 32, 64),
    waves=((1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2)),
    stage_group=(1, 4, 8),
)


def _native_specs(stride: int) -> List[DirectNchwDgradSpec]:
    p = _direct_problem(Shape(N=1, H=64, W=64, C=64, K=64, stride=stride))
    out = []
    g = _NATIVE_GRID
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


def _spec_key(spec: DirectNchwDgradSpec) -> str:
    return spec.kernel_name()


def _transposed_problem(s: Shape) -> DirectConvProblem:
    """Stride-1 dgrad as a forward conv: dY (K channels) in, dX (C) out, pad
    ``KH-1-PAD``, weights flipped and transposed (see ``_alt`` below)."""
    return DirectConvProblem(
        N=s.N,
        H=s.H,
        W=s.W,
        groups=1,
        cpg=s.K,
        kpg=s.C,
        KH=s.KH,
        KW=s.KW,
        PAD=s.KH - 1 - s.PAD,
        stride=1,
        dtype=_DTYPE,
    )


def _alt_knobs(shapes: List[Shape]) -> List[dict]:
    """Production ``conv_direct_nongrouped`` sweep grid (``nongrouped_knobs``)
    for the stride-1 shapes' widths, valid on at least one of them."""
    from kernels.common.conv_direct_nongrouped import (
        DirectNongroupedConvSpec,
        is_valid_nongrouped_spec,
        nongrouped_knobs,
    )

    out, seen = [], set()
    for sh in shapes:
        if sh.stride != 1:
            continue
        pt = _transposed_problem(sh)
        for kw in nongrouped_knobs(_ARCH, _DTYPE, Wo=pt.Wo):
            key = _alt_key(kw)
            if key in seen:
                continue
            if is_valid_nongrouped_spec(
                DirectNongroupedConvSpec(problem=pt, **kw), _ARCH
            )[0]:
                seen.add(key)
                out.append(kw)
    return out


def _alt_key(kw: dict) -> str:
    return repr(sorted(kw.items()))


def _alt_probe() -> DirectConvProblem:
    return _transposed_problem(Shape(N=1, H=64, W=64, C=64, K=64, stride=1))


# ---- parallel compile (top-level workers for pickling) ---------------------


def _compile_nhwc(job):
    try:
        import kernels.conv.aot_jobs as aj
        from rocke import compile_kernel

        kernel, meta = aj.kernel_for_job(job, _ARCH, _DTYPE)
        art = compile_kernel(kernel, arch=_ARCH)
        return repr(job.identity), (
            bytes(art.hsaco),
            art.kernel_name,
            meta["block_size"],
        )
    except Exception:  # noqa: BLE001 - an invalid combination is skipped
        return None


def _compile_native(spec):
    try:
        from rocke import compile_kernel

        art = compile_kernel(build_direct_conv_nchw_dgrad(spec, arch=_ARCH), arch=_ARCH)
        return _spec_key(spec), (
            bytes(art.hsaco),
            art.kernel_name,
            spec.threads_per_block,
        )
    except Exception:  # noqa: BLE001
        return None


def _compile_alt(kw):
    try:
        from kernels.common.conv_direct_nongrouped import (
            DirectNongroupedConvSpec,
            build_direct_conv_nongrouped,
        )
        from rocke import compile_kernel

        spec = DirectNongroupedConvSpec(problem=_alt_probe(), **kw)
        art = compile_kernel(build_direct_conv_nongrouped(spec, arch=_ARCH), arch=_ARCH)
        return _alt_key(kw), (bytes(art.hsaco), art.kernel_name, spec.threads_per_block)
    except Exception:  # noqa: BLE001
        return None


def _compile_all(items, worker, cache_file: Path, jobs: int) -> Dict[str, tuple]:
    from concurrent.futures import ProcessPoolExecutor

    cache: Dict[str, tuple] = {}
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    todo = [it for it, key in items if key not in cache]
    if todo:
        t0 = time.time()
        print(f"  compiling {len(todo)} kernels -> {cache_file.name} ...", flush=True)
        with ProcessPoolExecutor(max_workers=jobs or os.cpu_count()) as pool:
            for r in pool.map(worker, todo, chunksize=4):
                if r is not None:
                    cache[r[0]] = r[1]
        print(f"  compiled in {time.time() - t0:.0f}s", flush=True)
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
    return cache


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


@dataclass
class Cand:
    label: str
    run: Callable[[], None]  # one timed call (incl. any per-call zeroing)
    run_fenced: Callable[[], None]  # same, synchronous (for verification)
    screen_us: float = float("inf")
    ok: Optional[bool] = None
    rel: float = float("inf")
    t_us: float = float("nan")


def _screen(cands: List[Cand], iters: int) -> None:
    from rocke.runtime.launcher import time_launches

    for c in cands:
        try:
            c.screen_us = time_launches(c.run, warmup=1, iters=iters) * 1e3
        except Exception as e:  # noqa: BLE001
            print(f"  [screen-fail] {c.label}: {e}", flush=True)
            c.screen_us = float("inf")


def _verify_and_time(
    cands: List[Cand],
    prep: Callable[[], None],
    check: Callable[[], Tuple[bool, float]],
    args,
    topk: int,
) -> Optional[Cand]:
    """Verify the screened-fastest candidates in order and time the first
    ``topk`` that pass; return the fastest verified one. ``prep`` poisons the
    output before each verification launch."""
    order = sorted(
        (c for c in cands if c.screen_us < float("inf")), key=lambda c: c.screen_us
    )
    timed: List[Cand] = []
    for c in order:
        if len(timed) >= topk:
            break
        prep()
        c.run_fenced()
        c.ok, c.rel = check()
        if not c.ok:
            print(f"  [verify-FAIL] {c.label} rel={c.rel:.3e}", flush=True)
            continue
        c.t_us = _time_us(
            c.run, timing=args.timing, warmup=args.warmup, iters=args.iters
        )
        timed.append(c)
    if not timed:
        return None
    return min(timed, key=lambda c: c.t_us)


# One loaded module per binary for the whole run: the same binary serves every
# shape, and loading tens of thousands of modules exhausts the device.
_LAUNCHERS: Dict[Tuple[str, str], object] = {}


def _launcher(hsaco, name, sig, key: Optional[str] = None):
    from rocke.runtime.launcher import KernelLauncher

    if key is not None and (key, name) in _LAUNCHERS:
        return _LAUNCHERS[(key, name)]
    ln = KernelLauncher(hsaco=hsaco, kernel_name=name, signature=sig)
    if key is not None:
        _LAUNCHERS[(key, name)] = ln
    return ln


def _fenced(launcher, values, cfg):
    from rocke.runtime.launcher import LaunchConfig

    def go():
        launcher(
            values, config=LaunchConfig(grid=cfg.grid, block=cfg.block, fence=True)
        )

    return go


def _measure_permute(rt, src_t, perm, args):
    """Time an on-device ``permute_nd`` of ``src_t`` (verified bit-exact)."""
    import torch

    from rocke import compile_kernel
    from rocke.instances.common.permute_nd import (
        PermuteSpec,
        build_permute,
        permute_grid,
        permute_signature,
    )
    from rocke.runtime.launcher import LaunchConfig

    spec = PermuteSpec(x_shape=tuple(src_t.shape), perm=perm, dtype="f16")
    art = compile_kernel(build_permute(spec, arch=_ARCH), arch=_ARCH)
    want = src_t.permute(*perm).contiguous()
    src = rt.alloc(src_t.nbytes)
    dst = rt.alloc(src_t.nbytes)
    try:
        rt.memcpy_h2d(src, _u8(src_t), src_t.nbytes)
        ln = _launcher(art.hsaco, art.kernel_name, permute_signature(spec))
        values = {"X": int(src), "Y": int(dst)}
        cfg = LaunchConfig(grid=permute_grid(spec), block=(spec.block_size, 1, 1))
        _fenced(ln, values, cfg)()
        out = torch.empty_like(want)
        rt.memcpy_d2h(_u8(out), dst, out.nbytes)
        if not torch.equal(out, want):
            return False, float("nan")
        return True, _time_us(
            ln.bind(values, config=cfg),
            timing=args.timing,
            warmup=args.warmup,
            iters=args.iters,
        )
    finally:
        rt.free(src)
        rt.free(dst)


def _measure_transpose_repack(rt, W_kcyx, args):
    """KCYX -> CYXK as one 2D transpose of ``[K, C*KH*KW]`` (verified)."""
    import torch

    K = W_kcyx.shape[0]
    cols = W_kcyx[0].numel()
    if K % _TRANSPOSE_TILE or cols % _TRANSPOSE_TILE:
        return False, float("nan")
    art, _spec, grid, block, sig = _build_transpose(K, cols, 1, _ARCH)
    want = W_kcyx.reshape(K, cols).t().contiguous()
    src = rt.alloc(W_kcyx.nbytes)
    dst = rt.alloc(W_kcyx.nbytes)
    try:
        rt.memcpy_h2d(src, _u8(W_kcyx), W_kcyx.nbytes)
        run, ln, values, cfg = _bind_transpose(
            art, sig, grid, block, src, dst, K, cols, 1
        )
        _fenced(ln, values, cfg)()
        out = torch.empty_like(want)
        rt.memcpy_d2h(_u8(out), dst, out.nbytes)
        if not torch.equal(out, want):
            return False, float("nan")
        return True, _time_us(
            run, timing=args.timing, warmup=args.warmup, iters=args.iters
        )
    finally:
        rt.free(src)
        rt.free(dst)


# ---------------------------------------------------------------------------
# One shape
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    label: str
    native: Optional[Cand]
    nhwc: Optional[Cand]
    alt: Optional[Cand]
    pipe_ok: bool
    pipe_rel: float
    t_in: float
    t_out: float
    t_pipe_one: float
    t_pipe_one_alt: float
    t_repack_nhwc: float
    t_repack_native: float
    t_repack_native_tr: float
    n_native: int
    n_nhwc: int


def _run_case(
    shape: Shape,
    nhwc_jobs,
    nhwc_bins,
    native_specs,
    native_bins,
    alt_knobs,
    alt_bins,
    args,
):
    import torch

    from kernels.common.conv_abi import conv_args_signature, conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm_dgrad import pack_sub_gemm_buffer
    from kernels.conv.kernel_identity import dgrad_sub_gemms, identity_supports_problem
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import LaunchConfig

    dp = _direct_problem(shape)
    ip = _igemm_problem(shape)
    N, C, K, H, W = shape.N, shape.C, shape.K, shape.H, shape.W
    Ho, Wo = dp.Ho, dp.Wo
    assert (Ho, Wo) == (ip.Ho, ip.Wo)
    for m in (K, Ho * Wo, H * W, C):
        if m % _TRANSPOSE_TILE:
            print(f"  [skip] {shape.label()}: {m} not a multiple of the transpose tile")
            return None

    torch.manual_seed(0)
    dY = torch.empty(N, K, Ho, Wo, dtype=torch.float16).uniform_(-1.0, 1.0)
    Wk = torch.empty(K, C, shape.KH, shape.KW, dtype=torch.float16).uniform_(-1.0, 1.0)
    t0 = time.time()
    with torch.no_grad():
        ref = torch.nn.grad.conv2d_input(
            (N, C, H, W), Wk.float(), dY.float(), stride=shape.stride, padding=shape.PAD
        )
    ref_nhwc = ref.permute(0, 2, 3, 1).contiguous()
    floor = float(ref.abs().max().clamp(min=1.0))
    print(f"  reference in {time.time() - t0:.1f}s", flush=True)
    W_cyxk = Wk.permute(1, 2, 3, 0).contiguous()  # native, plan time
    W_kyxc = Wk.permute(0, 2, 3, 1).contiguous()  # NHWC, plan time

    rt = Runtime()
    nbytes = dY.nbytes, Wk.nbytes, N * C * H * W * 2
    dY_dev = rt.alloc(nbytes[0])  # NKHW
    dYt_dev = rt.alloc(nbytes[0])  # NHWK
    Wn_dev = rt.alloc(nbytes[1])  # CYXK (native)
    Wh_dev = rt.alloc(nbytes[1])  # KYXC (NHWC)
    dX_dev = rt.alloc(nbytes[2])  # NCHW (native / pipeline output)
    dXh_dev = rt.alloc(nbytes[2])  # NHWC (dgrad output)
    extra_bufs = []
    out = torch.empty(N, C, H, W, dtype=torch.float16)
    out_nhwc = torch.empty(N, H, W, C, dtype=torch.float16)

    def check(dev, host, want):
        rt.memcpy_d2h(_u8(host), dev, host.nbytes)
        if not torch.isfinite(host.float()).all():
            return False, float("inf")
        rel = float((host.float() - want).abs().max() / floor)
        return rel < _TOL, rel

    def poison(dev):
        host = torch.full((nbytes[2] // 2,), float("nan"), dtype=torch.float16)
        rt.memcpy_h2d(dev, _u8(host), host.nbytes)

    try:
        rt.memcpy_h2d(dY_dev, _u8(dY), nbytes[0])
        rt.memcpy_h2d(Wn_dev, _u8(W_cyxk), nbytes[1])
        rt.memcpy_h2d(Wh_dev, _u8(W_kyxc), nbytes[1])

        # ---- native arm -------------------------------------------------
        dsig = conv_direct_args_signature(_DTYPE)
        nvals = ConvArgs.from_problem(dp).to_launch_values(
            int(dY_dev), int(Wn_dev), int(dX_dev), nbytes[0], nbytes[1], nbytes[2]
        )
        ncands = []
        for spec in native_specs:
            s = replace(spec, problem=dp)
            key = _spec_key(spec)
            if (
                key not in native_bins
                or not is_valid_direct_nchw_dgrad_spec(s, _ARCH)[0]
            ):
                continue
            hsaco, name, block = native_bins[key]
            ln = _launcher(hsaco, name, dsig, key="native:" + key)
            cfg = LaunchConfig(grid=s.grid(), block=(block, 1, 1))
            ncands.append(
                Cand(key, ln.bind(nvals, config=cfg), _fenced(ln, nvals, cfg))
            )
        print(f"  native: screening {len(ncands)} candidates", flush=True)
        _screen(ncands, args.screen_iters)
        best_native = _verify_and_time(
            ncands,
            lambda: poison(dX_dev),
            lambda: check(dX_dev, out, ref),
            args,
            args.topk,
        )

        # ---- NHWC arm ---------------------------------------------------
        isig = conv_args_signature(_DTYPE, direction="dgrad")
        hcands = []
        for job in nhwc_jobs:
            ident = job.identity
            key = repr(ident)
            if key not in nhwc_bins or not identity_supports_problem(ident, ip)[0]:
                continue
            hsaco, name, block = nhwc_bins[key]
            sgs = dgrad_sub_gemms(ident, ip)
            flat = pack_sub_gemm_buffer(sgs, ident.tile_m, ident.tile_n)
            buf = (ctypes.c_int32 * len(flat))(*flat)
            sg_dev = rt.alloc(ctypes.sizeof(buf))
            extra_bufs.append(sg_dev)
            rt.memcpy_h2d(sg_dev, buf, ctypes.sizeof(buf))
            blocks = sum(
                -(-sg.gemm_m // ident.tile_m) * -(-sg.gemm_n // ident.tile_n)
                for sg in sgs
            )
            sk = max(1, ident.split_k)
            vals = ConvArgs.from_problem(
                ip, direction="dgrad", tile_m=ident.tile_m, tile_n=ident.tile_n
            ).to_launch_values(
                int(dYt_dev),
                int(Wh_dev),
                int(dXh_dev),
                nbytes[0],
                nbytes[1],
                nbytes[2],
                sub_gemm_buf=int(sg_dev),
                num_sub_gemms=len(sgs),
            )
            ln = _launcher(hsaco, name, isig, key="nhwc:" + key)
            cfg = LaunchConfig(grid=(blocks, 1, sk), block=(block, 1, 1))
            bound = ln.bind(vals, config=cfg)
            fenced = _fenced(ln, vals, cfg)
            if sk > 1:

                def run(bound=bound):
                    rt.memset_async(dXh_dev, 0, nbytes[2])
                    bound()

                def run_f(fenced=fenced):
                    rt.memset(dXh_dev, 0, nbytes[2])
                    fenced()
            else:
                run, run_f = bound, fenced
            label = f"{name}|sk{sk}|{job.caps}"
            hcands.append(Cand(label, run, run_f))

        # transposes
        in_art, _, in_grid, in_block, in_sig = _build_transpose(K, Ho * Wo, N, _ARCH)
        out_art, _, out_grid, out_block, out_sig = _build_transpose(H * W, C, N, _ARCH)
        run_in, in_ln, in_vals, in_cfg = _bind_transpose(
            in_art, in_sig, in_grid, in_block, dY_dev, dYt_dev, K, Ho * Wo, N
        )
        run_out, out_ln, out_vals, out_cfg = _bind_transpose(
            out_art, out_sig, out_grid, out_block, dXh_dev, dX_dev, H * W, C, N
        )
        _fenced(in_ln, in_vals, in_cfg)()  # dY^T is the NHWC arm's input

        print(f"  nhwc: screening {len(hcands)} candidates", flush=True)
        _screen(hcands, args.screen_iters)
        best_nhwc = _verify_and_time(
            hcands,
            lambda: poison(dXh_dev),
            lambda: check(dXh_dev, out_nhwc, ref_nhwc),
            args,
            args.topk,
        )

        # ---- alt NHWC arm (stride 1): conv_direct_nongrouped over the
        # transposed problem, flipped KRSC weights, same dY^T in / dX NHWC out.
        best_alt = None
        if shape.stride == 1 and alt_knobs:
            from kernels.common.conv_direct_nongrouped import (
                DirectNongroupedConvSpec,
                is_valid_nongrouped_spec,
            )

            pt = _transposed_problem(shape)
            W_flip = Wk.flip(2, 3).permute(1, 2, 3, 0).contiguous()
            Wf_dev = rt.alloc(W_flip.nbytes)
            extra_bufs.append(Wf_dev)
            rt.memcpy_h2d(Wf_dev, _u8(W_flip), W_flip.nbytes)
            avals = ConvArgs.from_problem(pt).to_launch_values(
                int(dYt_dev), int(Wf_dev), int(dXh_dev), nbytes[0], nbytes[1], nbytes[2]
            )
            acands = []
            for kw in alt_knobs:
                key = _alt_key(kw)
                spec = DirectNongroupedConvSpec(problem=pt, **kw)
                if key not in alt_bins or not is_valid_nongrouped_spec(spec, _ARCH)[0]:
                    continue
                hsaco, name, block = alt_bins[key]
                ln = _launcher(hsaco, name, dsig, key="alt:" + key)
                cfg = LaunchConfig(grid=spec.grid(), block=(block, 1, 1))
                acands.append(
                    Cand(name, ln.bind(avals, config=cfg), _fenced(ln, avals, cfg))
                )
            print(f"  alt nhwc direct: screening {len(acands)} candidates", flush=True)
            _screen(acands, args.screen_iters)
            best_alt = _verify_and_time(
                acands,
                lambda: poison(dXh_dev),
                lambda: check(dXh_dev, out_nhwc, ref_nhwc),
                args,
                args.topk,
            )

        pipe_ok, pipe_rel = False, float("inf")
        t_in = t_out = t_one = float("nan")
        if best_nhwc is not None:
            # End to end, NCHW in -> NCHW out, against the same reference.
            poison(dX_dev)
            poison(dXh_dev)
            _fenced(in_ln, in_vals, in_cfg)()
            best_nhwc.run_fenced()
            _fenced(out_ln, out_vals, out_cfg)()
            pipe_ok, pipe_rel = check(dX_dev, out, ref)
            if pipe_ok:
                t_in = _time_us(
                    run_in, timing=args.timing, warmup=args.warmup, iters=args.iters
                )
                t_out = _time_us(
                    run_out, timing=args.timing, warmup=args.warmup, iters=args.iters
                )
                # Re-time the dgrad stage next to the transposes (same session).
                best_nhwc.t_us = _time_us(
                    best_nhwc.run,
                    timing=args.timing,
                    warmup=args.warmup,
                    iters=args.iters,
                )

                def one(dg=best_nhwc.run):
                    run_in()
                    dg()
                    run_out()

                t_one = _time_us(
                    one, timing=args.timing, warmup=args.warmup, iters=args.iters
                )
        t_one_alt = float("nan")
        if best_alt is not None:
            # The alternative pipeline end to end, NCHW out.
            poison(dX_dev)
            poison(dXh_dev)
            _fenced(in_ln, in_vals, in_cfg)()
            best_alt.run_fenced()
            _fenced(out_ln, out_vals, out_cfg)()
            alt_ok, alt_rel = check(dX_dev, out, ref)
            if not alt_ok:
                print(f"  [FAIL] alt pipeline rel={alt_rel:.3e}", flush=True)
                best_alt = None
        if best_alt is not None:
            best_alt.t_us = _time_us(
                best_alt.run, timing=args.timing, warmup=args.warmup, iters=args.iters
            )

            def one_alt(dg=best_alt.run):
                run_in()
                dg()
                run_out()

            t_one_alt = _time_us(
                one_alt, timing=args.timing, warmup=args.warmup, iters=args.iters
            )
        if best_native is not None:
            best_native.t_us = _time_us(
                best_native.run,
                timing=args.timing,
                warmup=args.warmup,
                iters=args.iters,
            )

        # ---- per-call weight repack costs (informational) ---------------
        flat_w = Wk.reshape(K, C, shape.KH * shape.KW)
        ok_rh, t_rh = _measure_permute(rt, flat_w, (0, 2, 1), args)
        ok_rn, t_rn = _measure_permute(rt, flat_w, (1, 2, 0), args)
        ok_rt, t_rt = _measure_transpose_repack(rt, Wk, args)
        if not (ok_rh and ok_rn):
            print("  [FAIL] weight repack mismatch", flush=True)
        synchronize_and_release(0)
    finally:
        for buf in (dY_dev, dYt_dev, Wn_dev, Wh_dev, dX_dev, dXh_dev, *extra_bufs):
            rt.free(buf)

    return CaseResult(
        label=shape.label(),
        native=best_native,
        nhwc=best_nhwc,
        alt=best_alt,
        pipe_ok=pipe_ok,
        pipe_rel=pipe_rel,
        t_in=t_in,
        t_out=t_out,
        t_pipe_one=t_one,
        t_pipe_one_alt=t_one_alt,
        t_repack_nhwc=t_rh,
        t_repack_native=t_rn,
        t_repack_native_tr=t_rt if ok_rt else float("nan"),
        n_native=len(ncands),
        n_nhwc=len(hcands),
    )


def _print(results: List[CaseResult]) -> None:
    hdr = (
        f"{'shape':<26}{'native':>9}{'x_in':>8}{'dgrad':>8}{'x_out':>8}{'pipe':>9}"
        f"{'pipe1':>9}{'speedup':>9}{'nat/dg':>8}{'xp%':>6}"
        f"{'rp_nhwc':>9}{'rp_nat':>8}{'rp_natT':>8}{'alt_dg':>8}{'alt1':>9}{'spd_alt':>8}"
    )
    print("\n" + "=" * len(hdr))
    print("NATIVE NCHW DGRAD vs TRANSPOSE + NHWC DGRAD + TRANSPOSE (times in us)")
    print("pipe = x_in + dgrad + x_out; pipe1 = the three launches as one sample")
    print(
        "speedup = pipe / native (>1: native wins); nat/dg = native / NHWC dgrad stage"
    )
    print(
        "rp_* = per-call on-device weight repack (permute_nd; rp_natT = 2D transpose)"
    )
    print("alt_dg = stride-1 NHWC conv_direct_nongrouped on the transposed problem;")
    print(
        "alt1 = x_in + alt_dg + x_out as one sample; spd_alt = (x_in + alt_dg + x_out) / native"
    )
    print("=" * len(hdr))
    print(hdr)
    for r in results:
        if r.native is None or r.nhwc is None or not r.pipe_ok:
            print(f"{r.label:<26}  (an arm failed verification -- see log)")
            continue
        pipe = r.t_in + r.nhwc.t_us + r.t_out
        print(
            f"{r.label:<26}{r.native.t_us:>9.2f}{r.t_in:>8.2f}{r.nhwc.t_us:>8.2f}"
            f"{r.t_out:>8.2f}{pipe:>9.2f}{r.t_pipe_one:>9.2f}{pipe / r.native.t_us:>8.2f}x"
            f"{r.native.t_us / r.nhwc.t_us:>7.2f}x{100 * (r.t_in + r.t_out) / pipe:>5.0f}%"
            f"{r.t_repack_nhwc:>9.2f}{r.t_repack_native:>8.2f}{r.t_repack_native_tr:>8.2f}"
            + (
                f"{r.alt.t_us:>8.2f}{r.t_pipe_one_alt:>9.2f}"
                f"{(r.t_in + r.alt.t_us + r.t_out) / r.native.t_us:>7.2f}x"
                if r.alt is not None
                else ""
            )
        )
    print("=" * len(hdr))
    for r in results:
        print(f"{r.label}: candidates native={r.n_native} nhwc={r.n_nhwc}")
        if r.native is not None:
            print(f"  best native: {r.native.label} (rel {r.native.rel:.2e})")
        if r.alt is not None:
            print(f"  best alt   : {r.alt.label} (rel {r.alt.rel:.2e})")
        if r.nhwc is not None:
            print(
                f"  best nhwc  : {r.nhwc.label} (rel {r.nhwc.rel:.2e}; pipeline rel {r.pipe_rel:.2e})"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--N", default="1", help="comma-separated batch sizes")
    ap.add_argument(
        "--shapes", default=None, help="comma-separated grid indices (default: all)"
    )
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument(
        "--cache-dir",
        default=str(Path(tempfile.gettempdir()) / "rocke_dgrad_nchw_spike"),
    )
    ap.add_argument("--timing", choices=("staged", "block"), default="staged")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--screen-iters", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument(
        "--no-alt-direct",
        dest="alt_direct",
        action="store_false",
        help="skip the extra stride-1 NHWC arm (conv_direct_nongrouped on the "
        "transposed problem)",
    )
    args = ap.parse_args()

    shapes = _grid([int(n) for n in args.N.split(",")])
    if args.shapes:
        shapes = [shapes[int(i)] for i in args.shapes.split(",")]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    strides = sorted({s.stride for s in shapes})
    nhwc_jobs = {
        st: _nhwc_jobs([s for s in shapes if s.stride == st]) for st in strides
    }
    native_specs = {st: _native_specs(st) for st in strides}
    nhwc_bins, native_bins = {}, {}
    for st in strides:
        print(
            f"stride {st}: {len(nhwc_jobs[st])} NHWC jobs, {len(native_specs[st])} native specs"
        )
        nhwc_bins[st] = _compile_all(
            [(j, repr(j.identity)) for j in nhwc_jobs[st]],
            _compile_nhwc,
            cache_dir / f"nhwc_s{st}.pkl",
            args.jobs,
        )
        native_bins[st] = _compile_all(
            [(s, _spec_key(s)) for s in native_specs[st]],
            _compile_native,
            cache_dir / f"native_s{st}.pkl",
            args.jobs,
        )
    alt_knobs = _alt_knobs(shapes) if args.alt_direct else []
    alt_bins = (
        _compile_all(
            [(kw, _alt_key(kw)) for kw in alt_knobs],
            _compile_alt,
            cache_dir / "alt_direct_s1.pkl",
            args.jobs,
        )
        if alt_knobs
        else {}
    )
    if alt_knobs:
        print(f"stride 1 alt: {len(alt_knobs)} conv_direct_nongrouped knob sets")
    if args.compile_only:
        return 0

    # Claim the HIP context for torch before rocke's runtime touches it.
    import torch

    torch.cuda.is_available()
    print(f"arch={_ARCH} timing={args.timing} warmup={args.warmup} iters={args.iters}")
    results = []
    for s in shapes:
        print(f"\n--- {s.label()} ---", flush=True)
        r = _run_case(
            s,
            nhwc_jobs[s.stride],
            nhwc_bins[s.stride],
            native_specs[s.stride],
            native_bins[s.stride],
            alt_knobs,
            alt_bins,
            args,
        )
        if r is not None:
            results.append(r)
    _print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
