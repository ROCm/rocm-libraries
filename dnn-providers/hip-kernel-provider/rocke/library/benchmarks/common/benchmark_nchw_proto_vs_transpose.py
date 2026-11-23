# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Research spike: native-NCHW direct conv vs. transpose + NHWC-conv + transpose.

:mod:`kernels.common.conv_direct_nongrouped_nchw_proto` is an experimental
prototype that consumes/produces NCHW tensors directly (see its module
docstring). The only alternative available to an NCHW-layout caller today is
the transpose -> NHWC conv -> transpose pipeline: host-free, on-device,
transpose ``X`` NCHW -> NHWC with
:mod:`rocke.instances.common.batched_transpose`, run the production NHWC
direct conv (:mod:`kernels.common.conv_direct_nongrouped`), then transpose the
output NHWK -> NKHW. This script measures whether the prototype actually wins
end to end, and if not, which pipeline stage dominates.

Transpose kernel choice
------------------------
:mod:`rocke.instances.common.batched_transpose` is a *2D* batched transpose
(``[batch, H, W] -> [batch, W, H]``), not a general 4D permute -- but an
NCHW<->NHWC swap over one image is exactly one 2D transpose once you fold the
other two axes together:

* NCHW, per image ``n``, is already the row-major matrix ``[C, H*W]`` (the
  channel axis has stride ``H*W``, pixels are contiguous within a channel).
  Transposing that matrix gives ``[H*W, C]``, which is exactly NHWC's
  row-major layout (pixel-major, channel-minor). So "transpose NCHW -> NHWC"
  is ``batched_transpose2d`` with its matrix ``H`` parameter bound to the
  channel count and its matrix ``W`` parameter bound to ``H*W``.
* Symmetrically, NHWK per image is the matrix ``[Ho*Wo, K]``; transposing
  gives ``[K, Ho*Wo]`` == NKHW. So "transpose NHWK -> NKHW" is the same
  kernel with ``H`` bound to ``Ho*Wo`` and ``W`` bound to ``K``.

This is simpler and more direct than reaching for
:mod:`rocke.instances.common.permute_nd` (a true N-D permute): the hard
constraints are the same class of thing either way (16-bit dtype only,
tile-divisible extents, no edge masking), but batched_transpose2d needs no
transform-DAG and reuses a kernel this codebase already has a C++/CK Tile
counterpart for. The spatial / channel sizes below are chosen to be multiples
of the transpose tile (64), so every shape in the grid hits it exactly (no
edge masking, which this kernel does not support).

Weight layout is *not* part of the timed pipeline: a real caller would repack
weights once, ahead of inference, not per call. The production NHWC kernel
wants KYXC weights (``[K, KH, KW, C]``); this script derives them once on the
host from the same KCYX reference weights the NCHW prototype and the
reference conv use, via a plain ``torch.permute().contiguous()`` before any
timing starts.

The prototype is measured twice: with torch-native KCYX weights, and with
weights pre-packed into the NHWC kernel's KRSC order (``weight_layout="krsc"``).
For the latter, the cost of doing that repack *per call* on-device
(``permute_nd``, scalar path) is reported separately, so both the
"repack once at plan time" and "repack every call" cases can be read off.

Timing
------
``--timing staged`` (default) is a median of per-launch stalled samples
(:func:`rocke.runtime.stall_gate.time_launches_staged`) -- the same protocol
hipDNN's own benchmark uses, so the numbers are apples-to-apples with other
rocke benchmark scripts. ``--timing block`` is back-to-back launches averaged
(:func:`rocke.runtime.launcher.time_launches`). Defaults mirror the rest of
this directory: ``--warmup 3 --iters 10``.

Usage
-----

    python benchmark_nchw_proto_vs_transpose.py --quick   # 2 shapes, smoke test
    python benchmark_nchw_proto_vs_transpose.py           # full grid
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, replace
from statistics import median
from typing import Callable, Dict, List, Optional, Tuple

from kernels.common.conv_abi import conv_direct_args_signature
from kernels.common.conv_args import ConvArgs
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_nongrouped import (
    DirectNongroupedConvSpec,
    build_direct_conv_nongrouped,
    is_valid_nongrouped_spec,
)
from kernels.common.conv_direct_nongrouped_nchw_proto import (
    DirectNongroupedConvNchwProtoSpec,
    build_direct_conv_nongrouped_nchw_proto,
    is_valid_nchw_proto_spec,
)

# Transpose tile: both matrix extents it is applied to below (C, H*W, K, Ho*Wo)
# must be exact multiples of this (batched_transpose2d does no edge masking).
_TRANSPOSE_TILE = 64
_TRANSPOSE_VEC = 8
_TOL = 5e-3  # fp16, matches test_conv_direct_nongrouped_nchw_proto.py


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


# ---------------------------------------------------------------------------
# Spec fitting (shrink ck / tile_h until the staged tiles fit arch's LDS)
# ---------------------------------------------------------------------------


def _fit_to_arch(spec, arch: str, is_valid: Callable) -> Tuple[object, bool]:
    """Shrink ``spec.ck`` then ``spec.tile_h`` until ``is_valid`` accepts it.

    Works for both ``DirectNongroupedConvSpec`` and
    ``DirectNongroupedConvNchwProtoSpec``: both are frozen dataclasses with
    ``ck``/``tile_h``/``waves_n``/``atom_k`` fields and an ``is_valid_*_spec``
    oracle of the same ``(ok, why)`` shape.
    """
    min_ck = spec.atom_k
    for _ in range(8):
        ok, _why = is_valid(spec, arch)
        if ok:
            return spec, True
        if spec.ck // 2 >= min_ck:
            spec = replace(spec, ck=spec.ck // 2)
            continue
        if spec.tile_h > spec.waves_n and (spec.tile_h // 2) % spec.waves_n == 0:
            spec = replace(spec, tile_h=spec.tile_h // 2)
            continue
        break
    return spec, is_valid(spec, arch)[0]


# ---------------------------------------------------------------------------
# Shape grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    N: int
    H: int
    W: int
    C: int
    K: int
    stride: int
    KH: int = 3
    KW: int = 3
    PAD: int = 1

    def label(self) -> str:
        return (
            f"N{self.N}H{self.H}W{self.W}C{self.C}K{self.K}"
            f"_{self.KH}x{self.KW}s{self.stride}p{self.PAD}"
        )


def _full_grid(N: int) -> List[Shape]:
    shapes = []
    for C in (64, 256):
        for HW in (64, 128):
            for stride in (1, 2):
                shapes.append(Shape(N=N, H=HW, W=HW, C=C, K=C, stride=stride))
    return shapes


def _quick_grid(N: int) -> List[Shape]:
    return [
        Shape(N=N, H=64, W=64, C=64, K=64, stride=1),
        Shape(N=N, H=64, W=64, C=64, K=64, stride=2),
    ]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_us(run: Callable[[], None], *, timing: str, warmup: int, iters: int) -> float:
    """Microseconds for one bound launch, per ``--timing``'s protocol."""
    if timing == "staged":
        from rocke.runtime.stall_gate import time_launches_staged

        samples = time_launches_staged(run, warmup=warmup, iters=iters)
        if not samples:
            return float("nan")
        return median(samples) * 1e3
    from rocke.runtime.launcher import time_launches

    ms = time_launches(run, warmup=warmup, iters=iters)
    return ms * 1e3


# ---------------------------------------------------------------------------
# Transpose stage: NCHW<->NHWC via batched_transpose2d
# ---------------------------------------------------------------------------


def _build_transpose(h_param: int, w_param: int, batch: int, arch: str):
    """Compile a batched 2D transpose over an ``[batch, h_param, w_param]``
    matrix, returning ``(artifact, spec, grid, block, signature)``."""
    from rocke import compile_kernel
    from rocke.instances.common.batched_transpose import (
        BatchedTranspose2DSpec,
        batched_transpose2d_grid,
        batched_transpose2d_signature,
        build_batched_transpose2d,
        is_valid_spec,
    )

    spec = BatchedTranspose2DSpec(
        tile_m=_TRANSPOSE_TILE, tile_n=_TRANSPOSE_TILE, vec=_TRANSPOSE_VEC, dtype="f16"
    )
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid batched-transpose spec: {why}")
    if h_param % spec.tile_m != 0 or w_param % spec.tile_n != 0:
        raise ValueError(
            f"transpose matrix [{h_param}, {w_param}] not divisible by tile "
            f"{spec.tile_m} (no edge masking in this kernel)"
        )
    artifact = compile_kernel(build_batched_transpose2d(spec, arch=arch), arch=arch)
    grid = batched_transpose2d_grid(h_param, w_param, batch, spec)
    block = (spec.block_size, 1, 1)
    sig = batched_transpose2d_signature(spec)
    return artifact, spec, grid, block, sig


def _bind_transpose(artifact, sig, grid, block, x_ptr, y_ptr, h_param, w_param, batch):
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    launcher = KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=sig
    )
    stride_x = h_param * w_param
    stride_y = h_param * w_param
    values = {
        "X": int(x_ptr),
        "Y": int(y_ptr),
        "H": h_param,
        "W": w_param,
        "batch_stride_x": stride_x,
        "batch_stride_y": stride_y,
    }
    cfg = LaunchConfig(grid=grid, block=block)
    return launcher.bind(values, config=cfg), launcher, values, cfg


def _launch_once(launcher, values, cfg) -> None:
    from rocke.runtime.launcher import LaunchConfig

    launcher(values, config=LaunchConfig(grid=cfg.grid, block=cfg.block, fence=True))


# ---------------------------------------------------------------------------
# One shape's measurement
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    label: str
    ok_native: bool
    ok_pipeline: bool
    rel_err_native: float
    rel_err_pipeline: float
    t_native_us: float
    ok_krsc: bool
    rel_err_krsc: float
    t_krsc_us: float
    ok_repack: bool
    t_repack_us: float
    t_in_us: float
    t_conv_us: float
    t_out_us: float
    # KRSC-weight variants keyed by _KRSC_VARIANTS name -> (ok, t_us, tile_h).
    extra: Dict[str, Tuple[bool, float, int]]


# name -> (x_lds_layout, epilogue), all with weight_layout="krsc". "chw_tr"
# needs gfx950 (ds_read_b64_tr_b16); it is skipped elsewhere.
_KRSC_VARIANTS = {
    "krsc+dir": ("pos_c", "direct"),
    "krsc+tr": ("chw_tr", "lds_transpose"),
    "krsc+tr+dir": ("chw_tr", "direct"),
}


def _measure_native(rt, spec, p, X_nchw, W_host, ref, ref_floor, arch, args):
    """Compile, verify against ``ref``, then time one NCHW-prototype spec.

    ``W_host`` is already in ``spec.weight_layout`` order. Returns
    ``(ok, rel_err, t_us)``; ``t_us`` is NaN unless the output verified.
    """
    import torch

    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    artifact = compile_kernel(
        build_direct_conv_nongrouped_nchw_proto(spec, arch=arch), arch=arch
    )
    D = torch.empty(p.N, p.total_k, p.Ho, p.Wo, dtype=torch.float16)
    A_dev = rt.alloc(X_nchw.nbytes)
    B_dev = rt.alloc(W_host.nbytes)
    D_dev = rt.alloc(D.nbytes)
    rel, ok, t = float("inf"), False, float("nan")
    try:
        rt.memcpy_h2d(A_dev, _u8(X_nchw), X_nchw.nbytes)
        rt.memcpy_h2d(B_dev, _u8(W_host), W_host.nbytes)
        D.fill_(float("nan"))
        rt.memcpy_h2d(D_dev, _u8(D), D.nbytes)
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=conv_direct_args_signature(p.dtype),
        )
        values = ConvArgs.from_problem(p).to_launch_values(
            int(A_dev), int(B_dev), int(D_dev), X_nchw.nbytes, W_host.nbytes, D.nbytes
        )
        cfg = LaunchConfig(grid=spec.grid(), block=(spec.threads_per_block, 1, 1))
        launcher(
            values, config=LaunchConfig(grid=cfg.grid, block=cfg.block, fence=True)
        )
        out = torch.empty_like(D)
        rt.memcpy_d2h(_u8(out), D_dev, D.nbytes)
        if torch.isfinite(out.float()).all():
            rel = float((out.float() - ref).abs().max() / ref_floor)
            ok = rel < _TOL
        if ok:
            t = _time_us(
                launcher.bind(values, config=cfg),
                timing=args.timing,
                warmup=args.warmup,
                iters=args.iters,
            )
        synchronize_and_release(0)
    finally:
        for buf in (A_dev, B_dev, D_dev):
            rt.free(buf)
    return ok, rel, t


def _measure_weight_repack(rt, W_kcyx, W_khwc, arch, args):
    """Time an on-device KCYX -> KRSC weight permute (``permute_nd``).

    Viewed as ``[K, C, KH*KW] -> [K, KH*KW, C]``. The innermost axis moves, so
    ``permute_nd`` takes its scalar path -- an upper bound on what a tuned
    repack would cost. Verified bit-exact against the host permute first.
    """
    import torch

    from rocke import compile_kernel
    from rocke.instances.common.permute_nd import (
        PermuteSpec,
        build_permute,
        permute_grid,
        permute_signature,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    K, C, KH, KW = W_kcyx.shape
    spec = PermuteSpec(x_shape=(K, C, KH * KW), perm=(0, 2, 1), dtype="f16")
    art = compile_kernel(build_permute(spec, arch=arch), arch=arch)
    src = rt.alloc(W_kcyx.nbytes)
    dst = rt.alloc(W_kcyx.nbytes)
    ok, t = False, float("nan")
    try:
        rt.memcpy_h2d(src, _u8(W_kcyx), W_kcyx.nbytes)
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=permute_signature(spec),
        )
        values = {"X": int(src), "Y": int(dst)}
        cfg = LaunchConfig(grid=permute_grid(spec), block=(spec.block_size, 1, 1))
        launcher(
            values, config=LaunchConfig(grid=cfg.grid, block=cfg.block, fence=True)
        )
        out = torch.empty_like(W_khwc)
        rt.memcpy_d2h(_u8(out), dst, out.nbytes)
        ok = bool(torch.equal(out, W_khwc))
        if ok:
            t = _time_us(
                launcher.bind(values, config=cfg),
                timing=args.timing,
                warmup=args.warmup,
                iters=args.iters,
            )
        synchronize_and_release(0)
    finally:
        rt.free(src)
        rt.free(dst)
    return ok, t


def _run_case(shape: Shape, arch: str, args) -> Optional[CaseResult]:
    import torch
    import torch.nn.functional as F

    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = DirectConvProblem(
        N=shape.N,
        H=shape.H,
        W=shape.W,
        groups=1,
        cpg=shape.C,
        kpg=shape.K,
        KH=shape.KH,
        KW=shape.KW,
        PAD=shape.PAD,
        stride=shape.stride,
        dtype="fp16",
    )
    Ho, Wo = p.Ho, p.Wo

    # ---- specs (both default to the same tile geometry) --------------------
    proto_spec, proto_ok = _fit_to_arch(
        DirectNongroupedConvNchwProtoSpec(problem=p), arch, is_valid_nchw_proto_spec
    )
    nhwc_spec, nhwc_ok = _fit_to_arch(
        DirectNongroupedConvSpec(problem=p), arch, is_valid_nongrouped_spec
    )
    if not proto_ok or not nhwc_ok:
        print(
            f"  [skip] {shape.label()}: spec does not fit {arch} "
            f"(proto_ok={proto_ok}, nhwc_ok={nhwc_ok})"
        )
        return None

    if (shape.C * shape.H * shape.W) % _TRANSPOSE_TILE or (
        shape.H * shape.W
    ) % _TRANSPOSE_TILE:
        print(f"  [skip] {shape.label()}: transpose-in matrix not tile-divisible")
        return None
    if (Ho * Wo) % _TRANSPOSE_TILE or shape.K % _TRANSPOSE_TILE:
        print(f"  [skip] {shape.label()}: transpose-out matrix not tile-divisible")
        return None

    # ---- reference data ------------------------------------------------
    torch.manual_seed(0)
    X_nchw = torch.empty(p.N, p.total_c, p.H, p.W, dtype=torch.float16).uniform_(
        -1.0, 1.0
    )
    W_kcyx = torch.empty(p.total_k, p.cpg, p.KH, p.KW, dtype=torch.float16).uniform_(
        -1.0, 1.0
    )
    with torch.no_grad():
        ref = F.conv2d(
            X_nchw.float(),
            W_kcyx.float(),
            padding=p.PAD,
            stride=p.stride,
            dilation=(1, 1),
        )
    ref_floor = ref.abs().max().clamp(min=1.0)

    # ---- native: compile once, correctness-check, then time ------------
    rt = Runtime()
    W_khwc = W_kcyx.permute(0, 2, 3, 1).contiguous()
    ok_native, rel_err_native, t_native = _measure_native(
        rt, proto_spec, p, X_nchw, W_kcyx, ref, ref_floor, arch, args
    )
    if not ok_native:
        print(f"  [FAIL] {shape.label()}: native kcyx rel_err={rel_err_native:.3e}")
    krsc_spec, krsc_ok = _fit_to_arch(
        replace(DirectNongroupedConvNchwProtoSpec(problem=p), weight_layout="krsc"),
        arch,
        is_valid_nchw_proto_spec,
    )
    ok_krsc, rel_err_krsc, t_krsc = False, float("inf"), float("nan")
    if krsc_ok:
        ok_krsc, rel_err_krsc, t_krsc = _measure_native(
            rt, krsc_spec, p, X_nchw, W_khwc, ref, ref_floor, arch, args
        )
    if not ok_krsc:
        print(f"  [FAIL] {shape.label()}: native krsc rel_err={rel_err_krsc:.3e}")
    extra: Dict[str, Tuple[bool, float, int]] = {}
    for name, (xl, ep) in _KRSC_VARIANTS.items():
        if xl == "chw_tr" and arch != "gfx950":
            continue
        v_spec, v_ok = _fit_to_arch(
            replace(
                DirectNongroupedConvNchwProtoSpec(problem=p),
                weight_layout="krsc",
                x_lds_layout=xl,
                epilogue=ep,
            ),
            arch,
            is_valid_nchw_proto_spec,
        )
        v_res = (False, float("nan"), v_spec.tile_h)
        if v_ok:
            ok_v, rel_v, t_v = _measure_native(
                rt, v_spec, p, X_nchw, W_khwc, ref, ref_floor, arch, args
            )
            v_res = (ok_v, t_v, v_spec.tile_h)
            if not ok_v:
                print(f"  [FAIL] {shape.label()}: {name} rel_err={rel_v:.3e}")
        extra[name] = v_res
    ok_repack, t_repack = _measure_weight_repack(rt, W_kcyx, W_khwc, arch, args)
    if not ok_repack:
        print(f"  [FAIL] {shape.label()}: on-device KCYX->KRSC weight permute mismatch")

    # ---- transpose pipeline ---------------------------------------------
    nhwc_artifact = compile_kernel(
        build_direct_conv_nongrouped(nhwc_spec, arch=arch), arch=arch
    )
    in_art, in_spec, in_grid, in_block, in_sig = _build_transpose(
        shape.C, shape.H * shape.W, shape.N, arch
    )
    out_art, out_spec, out_grid, out_block, out_sig = _build_transpose(
        Ho * Wo, shape.K, shape.N, arch
    )

    X_dev = rt.alloc(X_nchw.nbytes)
    Xt_dev = rt.alloc(X_nchw.nbytes)  # NHWC, same element count
    Wn_dev = rt.alloc(W_khwc.nbytes)
    Dn_dev = rt.alloc(p.N * Ho * Wo * p.total_k * 2)  # NHWK, fp16
    Dt_dev = rt.alloc(p.N * p.total_k * Ho * Wo * 2)  # NKHW, fp16

    rel_err_pipe = float("inf")
    ok_pipe = False
    t_in = t_conv = t_out = float("nan")
    try:
        rt.memcpy_h2d(X_dev, _u8(X_nchw), X_nchw.nbytes)
        rt.memcpy_h2d(Wn_dev, _u8(W_khwc), W_khwc.nbytes)

        run_in, in_launcher, in_values, in_cfg = _bind_transpose(
            in_art,
            in_sig,
            in_grid,
            in_block,
            X_dev,
            Xt_dev,
            shape.C,
            shape.H * shape.W,
            shape.N,
        )
        run_out, out_launcher, out_values, out_cfg = _bind_transpose(
            out_art,
            out_sig,
            out_grid,
            out_block,
            Dn_dev,
            Dt_dev,
            Ho * Wo,
            shape.K,
            shape.N,
        )

        nhwc_sig = conv_direct_args_signature(p.dtype)
        nhwc_args = ConvArgs.from_problem(p)
        nhwc_launcher = KernelLauncher(
            hsaco=nhwc_artifact.hsaco,
            kernel_name=nhwc_artifact.kernel_name,
            signature=nhwc_sig,
        )
        nhwc_bytes_a = p.N * p.H * p.W * p.total_c * 2
        nhwc_bytes_b = p.total_k * p.KH * p.KW * p.cpg * 2
        nhwc_bytes_d = p.N * Ho * Wo * p.total_k * 2
        nhwc_values = nhwc_args.to_launch_values(
            int(Xt_dev),
            int(Wn_dev),
            int(Dn_dev),
            nhwc_bytes_a,
            nhwc_bytes_b,
            nhwc_bytes_d,
        )
        nhwc_cfg = LaunchConfig(
            grid=nhwc_spec.grid(), block=(nhwc_spec.threads_per_block, 1, 1)
        )
        run_conv = nhwc_launcher.bind(nhwc_values, config=nhwc_cfg)

        # Correctness: run the three stages once, fenced, and compare to the
        # same NCHW reference the native kernel is checked against.
        in_launcher(
            in_values,
            config=LaunchConfig(grid=in_cfg.grid, block=in_cfg.block, fence=True),
        )
        nhwc_launcher(
            nhwc_values,
            config=LaunchConfig(grid=nhwc_cfg.grid, block=nhwc_cfg.block, fence=True),
        )
        out_launcher(
            out_values,
            config=LaunchConfig(grid=out_cfg.grid, block=out_cfg.block, fence=True),
        )
        D_pipe = torch.empty(p.N, p.total_k, Ho, Wo, dtype=torch.float16)
        rt.memcpy_d2h(_u8(D_pipe), Dt_dev, D_pipe.nbytes)
        if torch.isfinite(D_pipe.float()).all():
            rel_err_pipe = float((D_pipe.float() - ref).abs().max() / ref_floor)
            ok_pipe = rel_err_pipe < _TOL

        if ok_pipe:
            t_in = _time_us(
                run_in, timing=args.timing, warmup=args.warmup, iters=args.iters
            )
            t_conv = _time_us(
                run_conv, timing=args.timing, warmup=args.warmup, iters=args.iters
            )
            t_out = _time_us(
                run_out, timing=args.timing, warmup=args.warmup, iters=args.iters
            )
        synchronize_and_release(0)
    finally:
        for buf in (X_dev, Xt_dev, Wn_dev, Dn_dev, Dt_dev):
            rt.free(buf)

    if not ok_pipe:
        print(
            f"  [FAIL] {shape.label()}: transpose-pipeline rel_err={rel_err_pipe:.3e}"
        )

    return CaseResult(
        label=shape.label(),
        ok_native=ok_native,
        ok_pipeline=ok_pipe,
        rel_err_native=rel_err_native,
        rel_err_pipeline=rel_err_pipe,
        t_native_us=t_native,
        ok_krsc=ok_krsc,
        rel_err_krsc=rel_err_krsc,
        t_krsc_us=t_krsc,
        ok_repack=ok_repack,
        t_repack_us=t_repack,
        t_in_us=t_in,
        t_conv_us=t_conv,
        t_out_us=t_out,
        extra=extra,
    )


def _print_summary(results: List[CaseResult]) -> None:
    names = [n for n in _KRSC_VARIANTS if any(n in r.extra for r in results)]
    hdr = (
        f"{'shape':<26}{'kcyx':>9}{'krsc':>9}"
        + "".join(f"{n:>13}" for n in names)
        + f"{'repack':>8}{'x_in':>8}{'conv':>8}{'x_out':>8}{'xpose_tot':>11}{'spd_krsc':>10}"
        + "".join(f"{'spd ' + n:>17}" for n in names)
    )
    print(f"\n{'=' * len(hdr)}")
    print("NCHW-PROTO variants vs TRANSPOSE+NHWC-CONV+TRANSPOSE (times in us)")
    print("speedup = t_transpose_total / t_variant (>1 means the native variant wins)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if not (r.ok_native and r.ok_krsc and r.ok_repack and r.ok_pipeline):
            print(f"{r.label:<26}  (failed correctness check -- see log above)")
            continue
        total = r.t_in_us + r.t_conv_us + r.t_out_us
        ts = [
            r.extra[n][1] if n in r.extra and r.extra[n][0] else float("nan")
            for n in names
        ]
        print(
            f"{r.label:<26}{r.t_native_us:>9.2f}{r.t_krsc_us:>9.2f}"
            + "".join(f"{t:>13.2f}" for t in ts)
            + f"{r.t_repack_us:>8.2f}{r.t_in_us:>8.2f}{r.t_conv_us:>8.2f}{r.t_out_us:>8.2f}"
            + f"{total:>11.2f}{total / r.t_krsc_us:>9.2f}x"
            + "".join(f"{total / t:>16.2f}x" for t in ts)
        )
    print("=" * len(hdr))
    print(
        "tile_h per variant (after LDS fit): "
        + "; ".join(
            f"{r.label}: "
            + ",".join(f"{n}={r.extra[n][2]}" for n in names if n in r.extra)
            for r in results
        )
    )


def main() -> int:
    from rocke.runtime.hip_module import get_device_arch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=None, help="gfx target (default: autodetect)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timing", choices=("staged", "block"), default="staged")
    parser.add_argument(
        "--N", type=int, default=1, help="fixed batch size for the grid"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="2-shape smoke-test grid instead of the full one",
    )
    args = parser.parse_args()

    arch = args.arch or get_device_arch(0)
    if not arch:
        print("error: no ROCm GPU detected", flush=True)
        return 1
    print(
        f"arch={arch} timing={args.timing} warmup={args.warmup} iters={args.iters} N={args.N}"
    )

    shapes = _quick_grid(args.N) if args.quick else _full_grid(args.N)
    results: List[CaseResult] = []
    for shape in shapes:
        print(f"\n--- {shape.label()} ---", flush=True)
        r = _run_case(shape, arch, args)
        if r is not None:
            results.append(r)

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
