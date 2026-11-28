# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Research spike: native-NCHW wgrad vs. transpose + NHWC wgrad + dW repack.

:mod:`kernels.common.conv_wgrad_nchw_proto` is an experimental prototype that
reads NCHW ``X`` / NKHW ``dY`` and writes KCYX ``dW`` directly. An NCHW caller
can otherwise only use rocke's production NHWC wgrad through a pipeline:

1. transpose ``X``  NCHW -> NHWC  (``batched_transpose2d`` over ``[C, H*W]``)
2. transpose ``dY`` NKHW -> NHWK  (``batched_transpose2d`` over ``[K, Ho*Wo]``)
3. production NHWC wgrad (:mod:`kernels.common.conv_implicit_gemm_wgrad`,
   single-stage packed-atomic split-K or the two-stage f32 workspace path of
   :mod:`kernels.common.conv_implicit_gemm_wgrad_two_stage`), including the
   ``memset`` of the buffer it accumulates into
4. repack ``dW`` KYXC -> KCYX (``permute_nd`` over ``[K, Y*X, C]``)

See :mod:`benchmarks.common.benchmark_nchw_proto_vs_transpose` (the forward
spike) for why one image of an NCHW <-> NHWC swap is exactly one 2D transpose.

Fair tuning
-----------
Both arms are tuned per shape before the measured run:

* **NHWC baseline**: the production sweep's own knob space and helpers
  (``benchmark_implicit_gemm_conv``: the ``_TILE_MN x _TILE_MN x _TILE_K x
  _WARP_MN x _WARP_MN x _WARP_TILE_MN`` grid, every pipeline plus the async
  leg, both epilogues, unsplit and split-K builds, ``--sample``-style random
  thinning with a fixed seed, the exact-tile set appended unthinned, the
  two-stage leg on the split combos), timed at every split-K degree
  ``_split_k_degrees`` offers (the resolver's pick included) with the
  production ``--split-k-prune`` rule. A small curated set around the
  configurations the wgrad case studies found best is always added.
* **Native prototype**: its knob grid (atom, ``tile_m``, ``tile_c``,
  ``tile_r``, wave split, LDS pad) at every split degree.

The quickest ``--finalists`` of each arm are then verified on-device against
an fp32 ``torch.nn.grad.conv2d_weight`` reference (CPU, native NCHW) and
re-timed; the best *verified* one is the arm's result. Every stage of the
pipeline (both transposes, the repack) is verified too, and the full
pipeline's KCYX output is checked against the same reference.

Phases
------
Compilation needs no GPU, so it is a separate phase whose artifacts are
pickled to ``--cache-dir`` (``tempfile.gettempdir()`` by default):

    python -m benchmarks.common.benchmark_wgrad_nchw_proto_vs_transpose --phase build
    flock <lock> python -m benchmarks.common.benchmark_wgrad_nchw_proto_vs_transpose --phase run

``--phase all`` does both in one process. ``--shapes`` selects grid entries by
index. Timing is the median of per-launch staged samples
(:func:`rocke.runtime.stall_gate.time_launches_staged`), as in the forward
spike; the CPU reference does not touch the HIP context.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import os
import pickle
import random
import tempfile
from dataclasses import dataclass
from statistics import median
from typing import Callable, List, Optional, Tuple

_TRANSPOSE_TILE = 64
_TRANSPOSE_VEC = 8
_TOL = 5e-2  # fp16; test_conv_wgrad_correctness.py / test_conv_wgrad_nchw_proto.py
_NATIVE_DEGREES = (1, 2, 4, 8, 16, 32, 64, 128)


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    N: int
    C: int
    K: int
    H: int
    W: int
    stride: int
    KS: int = 3
    PAD: int = 1

    @property
    def Ho(self) -> int:
        return (self.H + 2 * self.PAD - self.KS) // self.stride + 1

    @property
    def Wo(self) -> int:
        return (self.W + 2 * self.PAD - self.KS) // self.stride + 1

    def label(self) -> str:
        return (
            f"N{self.N}C{self.C}K{self.K}H{self.H}W{self.W}"
            f"_{self.KS}x{self.KS}s{self.stride}p{self.PAD}"
        )

    def nhwc_problem(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        return ConvProblem(
            N=self.N,
            Hi=self.H,
            Wi=self.W,
            C=self.C,
            K=self.K,
            Y=self.KS,
            X=self.KS,
            sH=self.stride,
            sW=self.stride,
            pH=self.PAD,
            pW=self.PAD,
        )

    def nchw_problem(self):
        from kernels.common.conv_direct_grouped import DirectConvProblem

        return DirectConvProblem(
            N=self.N,
            H=self.H,
            W=self.W,
            groups=1,
            cpg=self.C,
            kpg=self.K,
            KH=self.KS,
            KW=self.KS,
            PAD=self.PAD,
            stride=self.stride,
            dtype="fp16",
        )


def full_grid() -> List[Shape]:
    return [
        Shape(N=N, C=C, K=C, H=HW, W=HW, stride=s)
        for N in (1, 8)
        for C in (64, 256)
        for HW in (64, 128)
        for s in (1, 2)
    ]


# ---------------------------------------------------------------------------
# Candidate generation + compilation (no GPU)
# ---------------------------------------------------------------------------


def _build_native_one(args_tuple):
    """Picklable worker: validate + build + compile one native spec."""
    spec, arch = args_tuple
    from rocke import compile_kernel
    from kernels.common.conv_wgrad_nchw_proto import (
        build_conv_wgrad_nchw_proto,
        is_valid_wgrad_nchw_proto_spec,
    )

    if not is_valid_wgrad_nchw_proto_spec(spec, arch)[0]:
        return None
    try:
        art = compile_kernel(build_conv_wgrad_nchw_proto(spec, arch=arch), arch=arch)
    except Exception:
        return None
    return spec, art


def native_specs(shape: Shape, arch: str) -> list:
    from kernels.common.conv_wgrad_nchw_proto import (
        WgradNchwProtoSpec,
        is_valid_wgrad_nchw_proto_spec,
    )

    p = shape.nchw_problem()
    out = []
    for atom, tm, tc, tr, (wm, wn), pad, yg in itertools.product(
        ("32x32x16", "16x16x32"),
        (32, 64, 128),
        (16, 32, 64),
        (32, 64, 96, 128),
        ((1, 1), (2, 1), (4, 1), (1, 3), (2, 3), (1, 9)),
        (0, 8),
        (1, 3),
    ):
        spec = WgradNchwProtoSpec(
            problem=p,
            tile_m=tm,
            tile_c=tc,
            tile_r=tr,
            waves_m=wm,
            waves_n=wn,
            atom=atom,
            lds_pad=pad,
            y_groups=yg,
        )
        if is_valid_wgrad_nchw_proto_spec(spec, arch)[0]:
            out.append(spec)
    return out


def nhwc_combos(shape: Shape, arch: str, args) -> Tuple[list, list]:
    """(single-stage combos, two-stage combos) of the production wgrad sweep."""
    from rocke.core.arch import ArchTarget
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _ASYNC_PIPELINE,
        _EPILOGUES,
        _PIPELINES,
        _SPLIT_K_BUILD,
        _TILE_K,
        _TILE_MN,
        _WARP_MN,
        _WARP_TILE_MN,
        WgradCombo,
        _sample_combos,
        _wgrad_exact_tile_combos,
    )

    target = ArchTarget.from_gfx(arch)
    legs = [(pp, False) for pp in _PIPELINES] + [(_ASYNC_PIPELINE, True)]
    sks = (1, _SPLIT_K_BUILD)
    grid = [
        WgradCombo(*g, pipe, epi, asy, sk)
        for g in itertools.product(
            _TILE_MN, _TILE_MN, _TILE_K, _WARP_MN, _WARP_MN, _WARP_TILE_MN
        )
        if g[3] * g[5] <= g[0]
        and g[4] * g[5] <= g[1]
        # A tile larger than the GEMM extent only adds a partial tile; the
        # validator would keep some of them, they never win, and they cost
        # timing budget.
        and g[0] <= max(64, shape.K)
        and g[1] <= max(64, 9 * shape.C)
        for epi in _EPILOGUES
        for pipe, asy in legs
        for sk in sks
    ]
    sampled = _sample_combos(grid, args.sample, args.seed) if args.sample < 1 else grid
    curated = [
        WgradCombo(tm, tn, tk, wm, wn, at, pipe, epi, False, sk)
        for tm, tn in ((64, 64), (64, 128), (128, 64), (128, 128))
        for tk in (32, 64)
        for wm, wn in ((2, 2), (1, 2), (2, 1))
        for at in (16, 32)
        for pipe in ("mem", "compv3")
        for epi in ("default", "cshuffle")
        for sk in sks
        if wm * at <= tm and wn * at <= tn
    ]
    exact = _wgrad_exact_tile_combos(
        wg_M=shape.K,
        wg_N=9 * shape.C,
        wave_size=target.wave_size,
        degrees_fn=lambda _p, _a: sks,
        legs=legs,
        cap=args.exact_tiles,
    )
    single = list(dict.fromkeys(sampled + curated + exact))
    two = list(
        dict.fromkeys(
            c._replace(ws_replicas=r)
            for c in single
            if c.split_k > 1 and not c.async_dma
            for r in ((4, 8, 16) if c in curated else (8,))
        )
    )
    return single, two


@dataclass(frozen=True)
class _Art:
    kernel_name: str
    hsaco: bytes


def _cache_path(args, shape: Shape) -> str:
    return os.path.join(args.cache_dir, f"wgrad_nchw_bench_{shape.label()}.pkl")


def build_phase(shape: Shape, arch: str, args) -> dict:
    import torch
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _build_and_compile_wgrad_one,
        _build_and_compile_wgrad_two_stage_one,
        _build_ir_parallel,
    )
    from benchmarks.common import benchmark_wgrad_nchw_proto_vs_transpose as me

    print(f"[build] {shape.label()}", flush=True)
    nat = native_specs(shape, arch)
    print(f"  native: {len(nat)} valid specs", flush=True)
    nat_built = _build_ir_parallel(
        [(s, arch) for s in nat], me._build_native_one, args.jobs
    )

    p = shape.nhwc_problem()
    single, two = nhwc_combos(shape, arch, args)
    print(
        f"  nhwc: {len(single)} single-stage + {len(two)} two-stage combos", flush=True
    )
    s_built = _build_ir_parallel(
        [(c, p, "fp16", arch) for c in single], _build_and_compile_wgrad_one, args.jobs
    )
    t_built = _build_ir_parallel(
        [(c, p, "fp16", arch) for c in two],
        _build_and_compile_wgrad_two_stage_one,
        args.jobs,
    )

    # Reference: fp32, CPU, native NCHW / NKHW in, KCYX out.
    torch.manual_seed(0)
    X = torch.empty(shape.N, shape.C, shape.H, shape.W).uniform_(-1.0, 1.0).half()
    dY = torch.empty(shape.N, shape.K, shape.Ho, shape.Wo).uniform_(-1.0, 1.0).half()
    torch.set_num_threads(os.cpu_count() or 1)
    ref = torch.nn.grad.conv2d_weight(
        X.float(),
        (shape.K, shape.C, shape.KS, shape.KS),
        dY.float(),
        stride=shape.stride,
        padding=shape.PAD,
    )
    # Keep only what the run phase launches: the artifacts carry IR text.
    data = {
        "native": [(s, _Art(a.kernel_name, a.hsaco)) for s, a in nat_built],
        "single": [(c, s, k, _Art(a.kernel_name, a.hsaco)) for c, s, k, a in s_built],
        "two": [
            (c, s, k, _Art(a1.kernel_name, a1.hsaco), _Art(a2.kernel_name, a2.hsaco))
            for c, s, k, a1, a2 in t_built
        ],
        "X": X,
        "dY": dY,
        "ref": ref,
    }
    with open(_cache_path(args, shape), "wb") as f:
        pickle.dump(data, f)
    print(
        f"  built native={len(nat_built)} single={len(s_built)} two={len(t_built)}",
        flush=True,
    )
    return data


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _time_us(fn: Callable[[], None], warmup: int, iters: int) -> float:
    from rocke.runtime.stall_gate import time_launches_staged

    samples = time_launches_staged(fn, warmup=warmup, iters=iters)
    return median(samples) * 1e3 if samples else float("nan")


def _launcher(art, sig):
    from rocke.runtime.launcher import KernelLauncher

    return KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)


def _try_launcher(art, sig):
    """:func:`_launcher`, or ``None`` (logged) when the code object does not load."""
    try:
        return _launcher(art, sig)
    except Exception as e:  # a candidate whose binary does not load is rejected
        print(f"  [load] {art.kernel_name}: {e}", flush=True)
        return None


def _fenced(cfg):
    from rocke.runtime.launcher import LaunchConfig

    return LaunchConfig(grid=cfg.grid, block=cfg.block, fence=True)


def _transpose(rt, arch, h, w, batch, src, dst):
    """Compile + bind a batched 2D transpose ``[batch, h, w] -> [batch, w, h]``."""
    from rocke import compile_kernel
    from rocke.instances.common.batched_transpose import (
        BatchedTranspose2DSpec,
        batched_transpose2d_grid,
        batched_transpose2d_signature,
        build_batched_transpose2d,
        is_valid_spec,
    )
    from rocke.runtime.launcher import LaunchConfig

    spec = BatchedTranspose2DSpec(
        tile_m=_TRANSPOSE_TILE, tile_n=_TRANSPOSE_TILE, vec=_TRANSPOSE_VEC, dtype="f16"
    )
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok or h % _TRANSPOSE_TILE or w % _TRANSPOSE_TILE:
        raise ValueError(f"transpose [{h},{w}] not supported: {why}")
    art = compile_kernel(build_batched_transpose2d(spec, arch=arch), arch=arch)
    la = _launcher(art, batched_transpose2d_signature(spec))
    vals = {
        "X": int(src),
        "Y": int(dst),
        "H": h,
        "W": w,
        "batch_stride_x": h * w,
        "batch_stride_y": h * w,
    }
    cfg = LaunchConfig(
        grid=batched_transpose2d_grid(h, w, batch, spec), block=(spec.block_size, 1, 1)
    )
    return la, vals, cfg


def _repack(arch, K, taps, C, src, dst):
    """Compile + bind the dW repack KYXC -> KCYX (``[K, taps, C] -> [K, C, taps]``)."""
    from rocke import compile_kernel
    from rocke.instances.common.permute_nd import (
        PermuteSpec,
        build_permute,
        permute_grid,
        permute_signature,
    )
    from rocke.runtime.launcher import LaunchConfig

    spec = PermuteSpec(x_shape=(K, taps, C), perm=(0, 2, 1), dtype="f16")
    art = compile_kernel(build_permute(spec, arch=arch), arch=arch)
    la = _launcher(art, permute_signature(spec))
    vals = {"X": int(src), "Y": int(dst)}
    cfg = LaunchConfig(grid=permute_grid(spec), block=(spec.block_size, 1, 1))
    return la, vals, cfg


def _rel_err(out, ref) -> float:
    import torch

    out = out.float()
    if not torch.isfinite(out).all():
        return float("inf")
    return float((out - ref).abs().max() / ref.abs().max().clamp(min=1.0))


@dataclass
class Cand:
    desc: str
    fn: Callable[[], None]
    out_dev: int  # where dW lands
    layout: str  # "kcyx" or "kyxc"
    quick_us: float = float("nan")
    rel: float = float("inf")
    us: float = float("nan")
    hsaco: bytes = b""


def _tune(cands: List[Cand], rt, shape, ref_kcyx, args, label: str) -> Optional[Cand]:
    """Quick-time every candidate, verify + re-time the fastest finalists."""
    import torch

    from rocke.runtime import synchronize_and_release

    ref_kyxc = ref_kcyx.permute(0, 2, 3, 1).contiguous()
    nbytes = shape.K * shape.C * shape.KS * shape.KS * 2
    for c in cands:
        try:
            c.quick_us = _time_us(c.fn, args.quick_warmup, args.quick_iters)
        except Exception as e:  # a launch failure is a rejected candidate
            print(f"  [{label}] {c.desc}: launch failed ({e})", flush=True)
            c.quick_us = float("nan")
    synchronize_and_release(0)
    ranked = sorted(
        (c for c in cands if c.quick_us == c.quick_us), key=lambda c: c.quick_us
    )
    best = None
    n_fail = 0
    for c in ranked[: args.finalists]:
        rt.memset(c.out_dev, 0xFF, nbytes)  # NaN pattern: unwritten output fails
        c.fn()
        synchronize_and_release(0)
        out = torch.empty(
            (shape.K, shape.C, shape.KS, shape.KS)
            if c.layout == "kcyx"
            else (shape.K, shape.KS, shape.KS, shape.C),
            dtype=torch.float16,
        )
        rt.memcpy_d2h(_u8(out), c.out_dev, nbytes)
        c.rel = _rel_err(out, ref_kcyx if c.layout == "kcyx" else ref_kyxc)
        if c.rel >= _TOL:
            n_fail += 1
            print(f"  [{label}] FAIL verify {c.desc}: rel_err={c.rel:.3e}", flush=True)
            continue
        c.us = _time_us(c.fn, args.warmup, args.iters)
        if best is None or c.us < best.us:
            best = c
    print(
        f"  [{label}] {len(cands)} candidates timed, {min(len(ranked), args.finalists)} "
        f"finalists verified ({n_fail} failed); best: "
        + (f"{best.desc} rel_err={best.rel:.2e}" if best else "none"),
        flush=True,
    )
    return best


# ---------------------------------------------------------------------------
# One shape
# ---------------------------------------------------------------------------


def native_cands(
    native, rt, stream, dY_dev, X_dev, dWn_dev, dY_nbytes, X_nbytes, dw_nb
):
    """Candidates of the implicit-GEMM NCHW prototype at every split degree."""
    from rocke.runtime.launcher import LaunchConfig
    from kernels.common.conv_wgrad_nchw_proto import (
        wgrad_nchw_proto_launch_values,
        wgrad_nchw_proto_signature,
    )

    # ---- native candidates ------------------------------------------
    nat: List[Cand] = []
    sig_n = None
    for spec, art in native:
        sig_n = wgrad_nchw_proto_signature(spec)
        la = _try_launcher(art, sig_n)
        if la is None:
            continue
        for sk in _NATIVE_DEGREES:
            if sk > spec.k_steps:
                continue
            vals = wgrad_nchw_proto_launch_values(
                dY_dev, X_dev, dWn_dev, dY_nbytes, X_nbytes, sk
            )
            cfg = LaunchConfig(grid=spec.grid(sk), block=(spec.threads, 1, 1))
            run = la.bind(vals, config=cfg)

            def fn(_run=run):
                rt.memset_async(dWn_dev, 0, dw_nb, stream)
                _run()

            nat.append(
                Cand(f"{art.kernel_name} spk{sk}", fn, dWn_dev, "kcyx", hsaco=art.hsaco)
            )
    return nat


def nhwc_cands(
    single,
    two,
    rt,
    stream,
    shape,
    arch,
    dYt_dev,
    Xt_dev,
    dWk_dev,
    dY_nbytes,
    X_nbytes,
    dw_nb,
):
    """Production NHWC wgrad candidates (single-stage + two-stage legs).

    Returns ``(candidates, workspace buffer or None)``; the caller frees the
    workspace.
    """
    from rocke.runtime.launcher import LaunchConfig
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _grid_for_wgrad_spec,
        _split_k_degrees,
        _stage2_launch,
    )
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm_wgrad_two_stage import (
        _wgrad_stage1_signature,
        wgrad_stage1_launch_values,
        wgrad_two_stage_workspace_nbytes,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        wgrad_reduce_signature,
    )

    p_nhwc = shape.nhwc_problem()
    ws_dev = None

    # ---- NHWC candidates (production sweep), on transposed data -------
    nh: List[Cand] = []
    sig = conv_args_signature("fp16", direction="wgrad", is_3d=False)
    for combo, spec, rsk, art in single:
        la = _try_launcher(art, sig)
        if la is None:
            continue
        degs = (
            _split_k_degrees(
                p_nhwc,
                arch=arch,
                dtype="fp16",
                tile_m=spec.tile_m,
                tile_n=spec.tile_n,
                tile_k=spec.tile_k,
                block_size=spec.block_size,
                group_merge=spec.group_merge,
                two_stage=False,
            )
            if rsk > 1
            else (1,)
        )
        for sk in degs:
            vals = ConvArgs.from_problem(
                p_nhwc,
                direction="wgrad",
                tile_m=spec.tile_m,
                tile_n=spec.tile_n,
                tile_k=spec.tile_k,
            ).to_launch_values(
                int(dYt_dev),
                int(Xt_dev),
                int(dWk_dev),
                dY_nbytes,
                X_nbytes,
                dw_nb,
                split_k=sk,
            )
            cfg = LaunchConfig(
                grid=_grid_for_wgrad_spec(spec, sk), block=(spec.block_size, 1, 1)
            )
            run = la.bind(vals, config=cfg)
            if sk > 1:

                def fn(_run=run):
                    rt.memset_async(dWk_dev, 0, dw_nb, stream)
                    _run()

            else:
                fn = run
            nh.append(
                Cand(f"{art.kernel_name} spk{sk}", fn, dWk_dev, "kyxc", hsaco=art.hsaco)
            )
    ws_need = max([wgrad_two_stage_workspace_nbytes(s) for _, s, _, _, _ in two] or [0])
    if ws_need:
        ws_dev = rt.alloc(ws_need)
    for combo, spec, rsk, s1, s2 in two:
        l1 = _try_launcher(s1, _wgrad_stage1_signature(spec))
        s2_spec = WgradReduceSpec(
            problem=spec.problem,
            dtype_d=spec.data.dtype_d,
            groups=spec.problem.groups,
            ws_replicas=spec.ws_replicas,
        )
        l2 = _try_launcher(s2, wgrad_reduce_signature(s2_spec))
        if l1 is None or l2 is None:
            continue
        ws_nb = wgrad_two_stage_workspace_nbytes(spec)
        degs = _split_k_degrees(
            p_nhwc,
            arch=arch,
            dtype="fp16",
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            block_size=spec.block_size,
            group_merge=spec.group_merge,
            two_stage=True,
            ws_replicas=spec.ws_replicas,
        )
        for sk in degs:
            v1 = wgrad_stage1_launch_values(
                spec,
                dY_ptr=int(dYt_dev),
                X_ptr=int(Xt_dev),
                dW_ptr=int(dWk_dev),
                dY_bytes=dY_nbytes,
                X_bytes=X_nbytes,
                dW_bytes=dw_nb,
                ws_ptr=int(ws_dev),
                ws_bytes=ws_nb,
                split_k=sk,
            )
            v2, c2 = _stage2_launch(
                s2_spec, ws_ptr=ws_dev, ws_bytes=ws_nb, dw_ptr=dWk_dev, dw_bytes=dw_nb
            )
            c1 = LaunchConfig(
                grid=_grid_for_wgrad_spec(spec, sk), block=(spec.block_size, 1, 1)
            )
            r1 = l1.bind(v1, config=c1)
            r2 = l2.bind(v2, config=c2)

            def fn(_r1=r1, _r2=r2, _nb=ws_nb):
                rt.memset_async(ws_dev, 0, _nb, stream)
                _r1()
                _r2()

            nh.append(
                Cand(f"{s1.kernel_name} spk{sk}", fn, dWk_dev, "kyxc", hsaco=s1.hsaco)
            )
    return nh, ws_dev


@dataclass
class Row:
    label: str
    t_native: float
    t_xT: float
    t_dyT: float
    t_wgrad: float
    t_repack: float
    t_pipe_e2e: float
    native_desc: str
    nhwc_desc: str
    rel_native: float
    rel_pipe: float


def run_phase(shape: Shape, data: dict, arch: str, args) -> Optional[Row]:
    import torch

    from rocke.runtime import resolve_stream, synchronize_and_release
    from rocke.runtime.hip_module import Runtime

    print(f"[run] {shape.label()}", flush=True)
    X, dY, ref = data["X"], data["dY"], data["ref"]
    taps = shape.KS * shape.KS
    dw_nb = shape.K * shape.C * taps * 2
    stream = resolve_stream(0)

    rt = Runtime()
    X_dev = rt.alloc(X.nbytes)
    dY_dev = rt.alloc(dY.nbytes)
    Xt_dev = rt.alloc(X.nbytes)
    dYt_dev = rt.alloc(dY.nbytes)
    dWn_dev = rt.alloc(dw_nb)  # native KCYX
    dWk_dev = rt.alloc(dw_nb)  # NHWC kernel KYXC
    dWr_dev = rt.alloc(dw_nb)  # repacked KCYX
    ws_dev = None
    bufs = [X_dev, dY_dev, Xt_dev, dYt_dev, dWn_dev, dWk_dev, dWr_dev]
    try:
        rt.memcpy_h2d(X_dev, _u8(X), X.nbytes)
        rt.memcpy_h2d(dY_dev, _u8(dY), dY.nbytes)

        # ---- transposes + repack: verify bit-exact against host permutes ---
        HW, HoWo = shape.H * shape.W, shape.Ho * shape.Wo
        xT = _transpose(rt, arch, shape.C, HW, shape.N, X_dev, Xt_dev)
        dyT = _transpose(rt, arch, shape.K, HoWo, shape.N, dY_dev, dYt_dev)
        rp = _repack(arch, shape.K, taps, shape.C, dWk_dev, dWr_dev)
        for la, vals, cfg in (xT, dyT):
            la(vals, config=_fenced(cfg))
        Xt = torch.empty(shape.N, shape.H, shape.W, shape.C, dtype=torch.float16)
        dYt = torch.empty(shape.N, shape.Ho, shape.Wo, shape.K, dtype=torch.float16)
        rt.memcpy_d2h(_u8(Xt), Xt_dev, Xt.nbytes)
        rt.memcpy_d2h(_u8(dYt), dYt_dev, dYt.nbytes)
        ok_t = torch.equal(Xt, X.permute(0, 2, 3, 1)) and torch.equal(
            dYt, dY.permute(0, 2, 3, 1)
        )
        probe = torch.empty(shape.K, shape.KS, shape.KS, shape.C).uniform_(-1, 1).half()
        rt.memcpy_h2d(dWk_dev, _u8(probe), dw_nb)
        rp[0](rp[1], config=_fenced(rp[2]))
        back = torch.empty(shape.K, shape.C, shape.KS, shape.KS, dtype=torch.float16)
        rt.memcpy_d2h(_u8(back), dWr_dev, dw_nb)
        ok_r = torch.equal(back, probe.permute(0, 3, 1, 2))
        if not (ok_t and ok_r):
            print(f"  [FAIL] transpose ok={ok_t} repack ok={ok_r}", flush=True)
            return None

        nat = native_cands(
            data["native"],
            rt,
            stream,
            dY_dev,
            X_dev,
            dWn_dev,
            dY.nbytes,
            X.nbytes,
            dw_nb,
        )
        best_n = _tune(nat, rt, shape, ref, args, "native")
        nh, ws_dev = nhwc_cands(
            data["single"],
            data["two"],
            rt,
            stream,
            shape,
            arch,
            dYt_dev,
            Xt_dev,
            dWk_dev,
            dY.nbytes,
            X.nbytes,
            dw_nb,
        )
        if ws_dev is not None:
            bufs.append(ws_dev)
        best_h = _tune(nh, rt, shape, ref, args, "nhwc")
        if best_n is None or best_h is None:
            return None

        # ---- measured run: full pipeline verified end to end, then staged --
        run_x = xT[0].bind(xT[1], config=xT[2])
        run_dy = dyT[0].bind(dyT[1], config=dyT[2])
        run_rp = rp[0].bind(rp[1], config=rp[2])

        def pipe():
            run_x()
            run_dy()
            best_h.fn()
            run_rp()

        rt.memset(Xt_dev, 0xFF, X.nbytes)
        rt.memset(dYt_dev, 0xFF, dY.nbytes)
        rt.memset(dWr_dev, 0xFF, dw_nb)
        pipe()
        synchronize_and_release(0)
        out = torch.empty(shape.K, shape.C, shape.KS, shape.KS, dtype=torch.float16)
        rt.memcpy_d2h(_u8(out), dWr_dev, dw_nb)
        rel_pipe = _rel_err(out, ref)
        rt.memset(dWn_dev, 0xFF, dw_nb)
        best_n.fn()
        synchronize_and_release(0)
        rt.memcpy_d2h(_u8(out), dWn_dev, dw_nb)
        rel_nat = _rel_err(out, ref)
        if rel_pipe >= _TOL or rel_nat >= _TOL:
            print(
                f"  [FAIL] measured run: rel native={rel_nat:.3e} pipe={rel_pipe:.3e}"
            )
            return None
        # Each stage: median of --iters staged samples, repeated --repeats times
        # round-robin across the stages; the minimum median is reported, so a
        # burst from a foreign GPU process inflates at most one repeat.
        w, it = args.warmup, args.iters
        stages = {
            "native": best_n.fn,
            "xT": run_x,
            "dyT": run_dy,
            "wgrad": best_h.fn,
            "repack": run_rp,
            "e2e": pipe,
        }
        reps = {k: [] for k in stages}
        for _ in range(args.repeats):
            for k, f in stages.items():
                reps[k].append(_time_us(f, w, it))
        print(
            "  repeats (us): "
            + "  ".join(
                f"{k}=" + "/".join(f"{v:.1f}" for v in vs) for k, vs in reps.items()
            ),
            flush=True,
        )
        t = {k: min(vs) for k, vs in reps.items()}
        row = Row(
            label=shape.label(),
            t_native=t["native"],
            t_xT=t["xT"],
            t_dyT=t["dyT"],
            t_wgrad=t["wgrad"],
            t_repack=t["repack"],
            t_pipe_e2e=t["e2e"],
            native_desc=best_n.desc,
            nhwc_desc=best_h.desc,
            rel_native=rel_nat,
            rel_pipe=rel_pipe,
        )
        synchronize_and_release(0)
        if args.isa_dir:
            os.makedirs(args.isa_dir, exist_ok=True)
            for tag, c in (("native", best_n), ("nhwc", best_h)):
                with open(
                    os.path.join(args.isa_dir, f"{shape.label()}_{tag}.hsaco"), "wb"
                ) as f:
                    f.write(c.hsaco)
        return row
    finally:
        for buf in bufs:
            rt.free(buf)


def print_rows(rows: List[Row]) -> None:
    hdr = (
        f"{'shape':<30}{'native':>9}{'x_T':>9}{'dy_T':>9}{'wgrad':>9}{'repack':>9}"
        f"{'pipe_sum':>10}{'pipe_e2e':>10}{'spd_sum':>9}{'spd_e2e':>9}{'nat/wgrad':>11}{'xpose%':>8}"
    )
    print("\n" + "=" * len(hdr))
    print(
        "native NCHW wgrad vs transpose(X) + transpose(dY) + NHWC wgrad + dW repack (us)"
    )
    print(
        "spd_sum = pipe_sum / native, spd_e2e = pipe_e2e / native (>1: native wins); "
        "pipe_sum adds the stages timed one by one, pipe_e2e times the four stages "
        "as one sample; wgrad and native include their memset"
    )
    print("=" * len(hdr))
    print(hdr)
    for r in rows:
        tot = r.t_xT + r.t_dyT + r.t_wgrad + r.t_repack
        print(
            f"{r.label:<30}{r.t_native:>9.2f}{r.t_xT:>9.2f}{r.t_dyT:>9.2f}{r.t_wgrad:>9.2f}"
            f"{r.t_repack:>9.2f}{tot:>10.2f}{r.t_pipe_e2e:>10.2f}{tot / r.t_native:>8.2f}x"
            f"{r.t_pipe_e2e / r.t_native:>8.2f}x"
            f"{r.t_native / r.t_wgrad:>10.2f}x{100 * (r.t_xT + r.t_dyT + r.t_repack) / tot:>7.0f}%"
        )
    print("=" * len(hdr))
    for r in rows:
        print(f"{r.label}: native={r.native_desc} (rel {r.rel_native:.1e})")
        print(f"{'':<{len(r.label)}}  nhwc  ={r.nhwc_desc} (pipe rel {r.rel_pipe:.1e})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--phase", choices=("build", "run", "all"), default="all")
    parser.add_argument(
        "--shapes", default="", help="comma-separated grid indices (default: all)"
    )
    parser.add_argument("--cache-dir", default=tempfile.gettempdir())
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument(
        "--sample",
        type=float,
        default=0.03,
        help="NHWC grid fraction (production --sample)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact-tiles", type=int, default=200)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--quick-warmup", type=int, default=1)
    parser.add_argument("--quick-iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--isa-dir", default="", help="write the best hsaco per arm here"
    )
    args = parser.parse_args()

    shapes = full_grid()
    idx = [int(s) for s in args.shapes.split(",") if s] or list(range(len(shapes)))
    for i in idx:
        print(f"#{i}: {shapes[i].label()}")
    rows: List[Row] = []
    for i in idx:
        shape = shapes[i]
        data = None
        if args.phase in ("build", "all"):
            data = build_phase(shape, args.arch, args)
        if args.phase in ("run", "all"):
            if data is None:
                with open(_cache_path(args, shape), "rb") as f:
                    data = pickle.load(f)
            row = run_phase(shape, data, args.arch, args)
            if row is not None:
                rows.append(row)
            if not args.keep_cache and os.path.exists(_cache_path(args, shape)):
                os.remove(_cache_path(args, shape))
    if rows:
        print_rows(rows)
    return 0


if __name__ == "__main__":
    random.seed(0)
    raise SystemExit(main())
