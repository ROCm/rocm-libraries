# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tile sweep benchmark for the parametric direct convolution kernels.

Two kernel families are covered:
  cpg == 1  (groups == C == K) — depthwise: ``DirectDepthwiseSpec``, scalar fma.
  cpg >= 4, cpg % 4 == 0      — grouped:   ``DirectConvSpec``, mfma_f32_16x16x16_f16.

The variant is selected automatically from C / groups.

Run examples:
  python benchmark_direct_conv.py --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --groups 64   # depthwise
  python benchmark_direct_conv.py --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --groups 1    # grouped cpg=64
  python benchmark_direct_conv.py --N 8 --Hi 56 --Wi 56 --C 1024 --K 1024 --groups 64 --verify
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from dataclasses import dataclass
from typing import List

os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

from builders.common.conv_reference import conv_reference as _conv_reference
from builders.common.conv_reference import dgrad_reference as _dgrad_reference_shared
from builders.common.conv_reference import wgrad_reference as _wgrad_reference_shared

# ---------------------------------------------------------------------------
# Swept parameter grids
# ---------------------------------------------------------------------------

# Grouped (cpg >= 4) sweep dimensions.
_BLOCK_Q = (16, 32)
_BLOCK_GROUPS = (1, 2, 4, 8, 16)
_DOUBLE_BUFFER = (True, False)

# Grouped MFMA dgrad (transposed-fprop pipeline) sweep dimensions.
# block_h=0 (no H tiling: each workgroup streams the whole image height) is
# a first-class point -- it is the best setting on large-batch shapes, where
# N * q_tiles * g_tiles already fills the device without H tiles. block_h=4
# is the setting dispatch picks for small grids and for 5x5/7x7 filters.
_DGRAD_BLOCK_H = (0, 4, 8, 16)
# (waves_q, waves_k, runtime_k_loop, persistent_grid, fold_k32).
# (1, 1, *, *, True) is the single-wave 16x16x32 path; it reads plain W_T like
# the default path, so it needs no reorganize pre-pass. Combos whose waves_k
# does not divide the K-atom count of the chosen atom width are pruned by
# is_valid_spec for the problem at hand.
_DGRAD_WAVES_COMBOS = (
    (1, 1, False, False, False),
    (1, 1, False, False, True),
    (1, 1, True, False, False),
    (1, 2, False, False, False),
    (1, 4, False, False, False),
    (1, 4, True, False, False),
    (1, 2, False, False, True),
    (1, 6, False, False, True),
)

# Depthwise (cpg == 1) sweep dimensions.
_DW_BLOCK_W = (4, 8, 16, 32)
_DW_BLOCK_WAVES = (1, 2, 4)
# The ho-streaming depthwise dgrad builder unrolls every (ho, r, s, j) tap, so
# block_w at or above this with a filter of 7x7 or larger does not finish
# compiling in a usable time; the sweep skips those combinations.
_DW_DGRAD_STREAM_MAX_BW_LARGE_FILTER = 16
_DW_DGRAD_LARGE_FILTER_TAPS = 49


def _dw_dgrad_windowed_combos(p, arch: str) -> list:
    """Windowed depthwise dgrad knob combinations (stride 1) for the sweep.

    block_w: the whole row when W <= 16, plus tiles of about 8 and 16 columns
    (divisors of W where one is near); waves 1/2/4; f32 FMA, packed channel
    pairs, or dot2 (gfx950); whole H, or an H split when H spans at least four
    filter heights. Each tuple is (block_w, waves, ch_per_lane, block_h, dot2).
    """
    W, H, KH = p.W, p.H, p.KH
    bws = {math.ceil(W / math.ceil(W / t)) for t in (8, 16)}
    if W <= 16:
        bws.add(W)
    modes = [(1, False), (2, False)]
    if arch == "gfx950":
        modes.append((1, True))
    bhs = [0]
    if H >= 4 * KH:
        bhs += sorted({max(2 * KH, 8), H // 2})
    return [
        (bw, wv, cpl, bh, dot2)
        for bw in sorted(bws)
        for wv in _DW_BLOCK_WAVES
        for cpl, dot2 in modes
        for bh in bhs
    ]


# 4c dgrad (cpg == kpg == 4, batched 4x4x4 MFMA) sweep dimensions.
# block_q <= 32 keeps q_tiles_per_wave inside the C++ engine's tile bound.
_DGRAD_4C_BLOCK_Q = (4, 8, 16, 32)
_DGRAD_4C_BLOCK_GROUPS = (16, 32, 64)

# Single-kernel dgrad (fused weight transform, ``dgrad_fused_weights``) sweep
# dimensions for the generic direct-MFMA kernel (block_h as the pre-pass sweep).
_DGRAD_FUSED_BLOCK_H = _DGRAD_BLOCK_H
_DGRAD_FUSED_WAVES_PER_EU = (0, 4)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class Result:
    kernel_name: str
    block_q: int
    block_groups: int
    double_buffer: bool
    ms: float
    tflops: float
    gbps: float
    passed: "bool | None" = None


@dataclass(frozen=True)
class _MfmaDgradPipeline:
    """A planned MFMA dgrad pipeline and the IR of each of its stages."""

    plan: object  # kernels.common.conv_direct_grouped.DirectMfmaDgradPlan
    kernels: tuple


def _mfma_dgrad_label(combo) -> str:
    """Every knob that distinguishes two MFMA dgrad sweep points."""
    if combo[0] == "4c":
        _, bq, bg, tag = combo
        if tag:
            return f"bq={bq} bg={bg} 4c+{tag} MFMA 1-kernel"
        return f"bq={bq} bg={bg} 4c MFMA"
    if combo[0] == "fused":
        _, bq, bg, bh, k32, tag, wpe = combo
        return (
            f"bq={bq} bg={bg} bh={bh}{'+k32' if k32 else ''}+{tag}"
            f"{f'+we{wpe}' if wpe else ''} MFMA 1-kernel"
        )
    bq, bg, bh, wq, wk, rk, pg, k32 = combo
    flags = "".join(tag for tag, on in (("+rk", rk), ("+pg", pg), ("+k32", k32)) if on)
    return f"bq={bq} bg={bg} bh={bh} wq={wq} wk={wk}{flags} MFMA"


def _conv_rule_bad_count(out_t, ref_out, dtype: str) -> int:
    """Elements failing the manifest-runner conv rule (NaN counts as bad).

    ``bad = |D - ref| > tol + tol * |ref|`` with ``tol = 1e-2``; for bf16 the
    reference is first rounded to bf16 (RNE), as the manifest runner does.
    """
    import torch

    tol = 1e-2
    out_f32 = out_t.float().cpu()
    ref_f32 = ref_out.float().cpu()
    if dtype == "bf16":
        ref_f32 = ref_f32.to(torch.bfloat16).float()
    err = (out_f32 - ref_f32).abs()
    ok = err <= tol + tol * ref_f32.abs()
    return int((~ok).sum())


@dataclass
class DepthwiseResult:
    kernel_name: str
    block_w: int
    block_waves: int
    ms: float
    tflops: float
    gbps: float
    passed: "bool | None" = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MIOpen driver command parser
# ---------------------------------------------------------------------------

_MIOPEN_DTYPE_MAP = {
    "conv": "fp32",  # rejected — fp32 not supported
    "convfp16": "fp16",
    "convbfp16": "bf16",
    "convint8": "int8",  # rejected — int8 not supported
}


def parse_miopen_cmd_direct(cmd: str):
    """Parse a MIOpenDriver command string into a ``DirectConvProblem``.

    Supports 2-D NHWC forward (F=1), dgrad (F=2) and wgrad (F=4) convolutions.
    Raises ``ValueError`` for unsupported cases.
    Returns ``(problem, dtype, forw)`` where ``dtype`` is ``"fp16"``, ``"bf16"``, or
    ``"fp32"`` and ``forw`` is the raw MIOpen ``-F`` value.

    Note: ``DirectConvProblem`` requires ``cpg == kpg`` and cpg must be either
    1 (depthwise) or a positive multiple of 4 (grouped).
    """
    import shlex

    tokens = shlex.split(cmd)

    driver_kw = None
    driver_idx = None
    for i, t in enumerate(tokens):
        key = t.split("/")[-1].lower()
        if key in _MIOPEN_DTYPE_MAP:
            driver_kw = key
            driver_idx = i
            break
    if driver_kw is None:
        raise ValueError(
            f"No MIOpenDriver keyword found in command "
            f"(expected one of: {list(_MIOPEN_DTYPE_MAP)})"
        )
    dtype = _MIOPEN_DTYPE_MAP[driver_kw]
    if dtype == "fp32":
        raise ValueError(
            f"fp32 ({driver_kw!r}) is not supported by this benchmark; "
            f"use convfp16 or convbfp16"
        )
    if driver_kw == "convint8":
        raise ValueError(
            "convint8 is not supported by this benchmark; " "use convfp16 or convbfp16"
        )

    sub = argparse.ArgumentParser(add_help=False)
    sub.add_argument("-n", "--n", dest="N", type=int, default=1)
    sub.add_argument("-c", "--c", dest="C", type=int, default=1)
    sub.add_argument("-H", "--H", dest="Hi", type=int, default=1)
    sub.add_argument("-W", "--W", dest="Wi", type=int, default=1)
    sub.add_argument("-k", "--k", dest="K", type=int, default=1)
    sub.add_argument("-y", "--y", dest="Y", type=int, default=1)
    sub.add_argument("-x", "--x", dest="X", type=int, default=1)
    sub.add_argument("-p", "--p", dest="pH", type=int, default=0)
    sub.add_argument("-q", "--q", dest="pW", type=int, default=0)
    sub.add_argument("-u", "--u", dest="sH", type=int, default=1)
    sub.add_argument("-v", "--v", dest="sW", type=int, default=1)
    sub.add_argument("-l", "--l", dest="dH", type=int, default=1)
    sub.add_argument("-j", "--j", dest="dW", type=int, default=1)
    sub.add_argument("-g", "--g", dest="groups", type=int, default=1)
    sub.add_argument("-F", "--F", dest="forw", type=int, default=1)
    sub.add_argument(
        "-in_layout", "--in_layout", dest="in_layout", type=str, default="NHWC"
    )
    sub.add_argument("-m", "--m", dest="_mode", type=str, default="conv")
    sub.add_argument("-t", "--t", dest="_time", type=int, default=0)
    sub.add_argument("-V", "--V", dest="_verify", type=int, default=1)
    sub.add_argument("-_", "--_", dest="_spatial_dim", type=int, default=2)

    miopen_args, _ = sub.parse_known_args(tokens[driver_idx + 1 :])

    layout = miopen_args.in_layout.upper()
    if layout not in ("NHWC", "NWC"):
        raise ValueError(
            f"Layout {layout!r} is not supported; only NHWC/NWC inputs are accepted"
        )

    N = miopen_args.N
    C = miopen_args.C
    K = miopen_args.K
    groups = miopen_args.groups

    if C % groups != 0:
        raise ValueError(f"C={C} is not divisible by groups={groups}")
    if K % groups != 0:
        raise ValueError(f"K={K} is not divisible by groups={groups}")

    cpg = C // groups
    kpg = K // groups
    # For fprop the grouped direct kernels require cpg == kpg.
    # For dgrad and wgrad cpg and kpg may differ; the spec validators enforce the
    # kernel-specific constraints, so we skip the symmetric check here.
    if (
        miopen_args.forw not in (2, 4)
        and cpg != 1
        and cpg != kpg
        and (cpg % 4 != 0 or cpg < 4)
    ):
        raise ValueError(
            f"cpg={cpg} (C/groups) must be 1 (depthwise) or a positive multiple of 4"
        )

    sH = miopen_args.sH
    if miopen_args.sH != miopen_args.sW:
        print(
            f"[warn] sH={miopen_args.sH} != sW={miopen_args.sW}; using sH={miopen_args.sH}",
            file=sys.stderr,
        )
    if miopen_args.pH != miopen_args.pW:
        print(
            f"[warn] pH={miopen_args.pH} != pW={miopen_args.pW}; using pH={miopen_args.pH}",
            file=sys.stderr,
        )

    from kernels.common.conv_direct_grouped import DirectConvProblem

    problem = DirectConvProblem(
        N=N,
        H=miopen_args.Hi,
        W=miopen_args.Wi,
        groups=groups,
        cpg=cpg,
        kpg=kpg,
        KH=miopen_args.Y,
        KW=miopen_args.X,
        PAD=miopen_args.pH,
        stride=sH,
        dtype=dtype if dtype in ("fp16", "bf16") else "fp16",
    )
    return problem, dtype, miopen_args.forw


def _sample_combos(combos: list, frac: float, seed: int) -> list:
    import random

    n = max(1, round(len(combos) * frac))
    rng = random.Random(seed)
    return rng.sample(combos, min(n, len(combos)))


def _compile_one(args_tuple):
    kernel, arch = args_tuple
    from rocke import compile_kernel as _compile_kernel

    artifact = _compile_kernel(kernel, arch=arch)
    return kernel.name, artifact


def _compile_kernels_parallel(kernels, compile_kernel, arch: str, jobs: int) -> dict:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    unique: dict = {}
    for k in kernels:
        if k.name not in unique:
            unique[k.name] = k

    if not unique:
        return {}

    if jobs == 1:
        return {name: compile_kernel(k, arch=arch) for name, k in unique.items()}

    max_workers = os.cpu_count() if jobs == 0 else jobs
    work = [(k, arch) for k in unique.values()]
    print(
        f"Compiling {len(unique)} unique kernels with {max_workers} workers ...",
        flush=True,
    )
    artifact_map: dict = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compile_one, item): item[0].name for item in work}
        done = 0
        for fut in as_completed(futures):
            name, artifact = fut.result()
            artifact_map[name] = artifact
            done += 1
            if done % max(1, len(unique) // 10) == 0 or done == len(unique):
                print(f"  compiled {done}/{len(unique)}", flush=True)
    return artifact_map


def _verify_kernel(
    *,
    rt,
    launcher,
    values: dict,
    grid: tuple,
    block: tuple,
    out_dev,
    out_t,
    ref_out,
    kernel_name: str,
    dump_fail: "str | None",
    u8,
) -> "tuple[bool, bool]":
    import torch

    from rocke.runtime.launcher import LaunchConfig

    rt.memset(out_dev, 0, out_t.nbytes)
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    out_cpu = torch.empty_like(out_t)
    rt.memcpy_d2h(u8(out_cpu), out_dev, out_t.nbytes)

    out_f32 = out_cpu.float().cuda()
    abs_diff = out_f32.sub(ref_out).abs()
    ref_scale = ref_out.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    tol = 5e-2
    status = "PASS" if rel_err < tol else f"FAIL(rel_err={rel_err:.2e})"
    print(f"  verify {kernel_name}: {status}", flush=True)

    if rel_err >= tol and dump_fail:
        import pathlib

        import numpy as np

        dump_dir = pathlib.Path(dump_fail)
        dump_dir.mkdir(parents=True, exist_ok=True)
        diff = out_f32.sub(ref_out)

        def _save(name, t):
            np.savetxt(
                dump_dir / f"{kernel_name}_{name}.txt",
                t.cpu().numpy().flatten(),
                fmt="%.6f",
            )

        _save("out", out_f32)
        _save("ref", ref_out)
        _save("diff", diff)
        max_idx = int(diff.abs().argmax())
        unravel = np.unravel_index(max_idx, diff.shape)
        print(
            f"  [dump] saved to {dump_dir}/  "
            f"max_diff={rel_err:.4e} at index {unravel} (flat {max_idx})\n"
            f"  [dump] out={float(out_f32.flatten()[max_idx]):.6f}  "
            f"ref={float(ref_out.flatten()[max_idx]):.6f}",
            flush=True,
        )
        return True, False

    return False, rel_err < tol


def _conv_reference_grouped(A_t, B_t, p):
    """Grouped conv reference via torch.nn.functional.conv2d.

    Returns a ``torch.Tensor`` on the device. Unannotated on purpose: torch is
    an optional dependency here, so naming it in a signature would either need
    an import this module must not make at all, or a ``TYPE_CHECKING`` one that
    does not resolve in an environment without torch.
    """
    import torch.nn.functional as F


class _DirectConvProblemAdapter:
    """Thin adapter so ``conv_reference.{conv_reference,dgrad_reference}`` can
    consume a ``DirectConvProblem`` without modification.

    ``DirectConvProblem`` uses different attribute names from ``ConvProblem``
    (e.g. ``H``/``W`` vs ``Hi``/``Wi``, ``PAD`` vs ``pH``/``pW``, ``stride``
    vs ``sH``/``sW``).  This class exposes the interface that ``conv_reference``
    expects.
    """

    is_3d = False

    def __init__(self, p):
        self.N = p.N
        self.C = p.total_c
        self.K = p.total_k
        self.Hi = p.H
        self.Wi = p.W
        self.Y = p.KH
        self.X = p.KW
        self.sH = p.stride
        self.sW = p.stride
        self.pH = p.PAD
        self.pW = p.PAD
        self.dH = 1
        self.dW = 1
        self.groups = p.groups


def _print_results(
    results: List[Result],
    top_n_arg: int,
    arch: str,
    p,
    show_verify: bool,
    dtype: str = "fp16",
):
    top_n = min(top_n_arg, len(results))
    width = 100 if show_verify else 88
    print(f"\n{'='*width}")
    print(f"Top {top_n} configurations for {arch} {dtype} {p.short()}")
    print(f"{'='*width}")
    hdr = (
        f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  {'verify':>6}  config"
        if show_verify
        else f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
    )
    print(hdr)
    print("-" * width)
    for rank, r in enumerate(results[:top_n], 1):
        cfg = f"bq={r.block_q:3d} bg={r.block_groups:3d} db={r.double_buffer}"
        if show_verify:
            v = "PASS" if r.passed else "FAIL"
            print(
                f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}"
                f"  {v:>6}  {cfg}"
            )
        else:
            print(
                f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}" f"  {cfg}"
            )
    best = results[0]
    print(f"\nBest: {best.tflops:.1f} TFLOPS — {best.kernel_name}")


def _print_depthwise_results(
    results: "List[DepthwiseResult]",
    top_n_arg: int,
    arch: str,
    p,
    show_verify: bool,
    dtype: str = "fp16",
):
    top_n = min(top_n_arg, len(results))
    width = 96 if show_verify else 84
    print(f"\n{'='*width}")
    print(f"Top {top_n} depthwise configurations for {arch} {dtype} {p.short()}")
    print(f"{'='*width}")
    hdr = (
        f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  {'verify':>6}  config"
        if show_verify
        else f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
    )
    print(hdr)
    print("-" * width)
    for rank, r in enumerate(results[:top_n], 1):
        cfg = f"bw={r.block_w:3d} bwv={r.block_waves}"
        if show_verify:
            v = "PASS" if r.passed else "FAIL"
            print(
                f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}"
                f"  {v:>6}  {cfg}"
            )
        else:
            print(
                f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}" f"  {cfg}"
            )
    best = results[0]
    print(f"\nBest: {best.tflops:.1f} TFLOPS — {best.kernel_name}")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def _run_depthwise_sweep(
    *,
    args,
    problem,
    dtype: str = "fp16",
    arch: str,
    compile_kernel,
    jobs: int,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
) -> "tuple[int, List[DepthwiseResult]]":
    import math
    import torch

    from rocke.helpers.manifest import conv_args_signature
    from kernels.common.conv_direct_grouped import (
        DirectDepthwiseSpec,
        DirectDepthwiseSpatialSpec,
        build_direct_depthwise,
        build_direct_depthwise_spatial,
        is_valid_depthwise_spec,
        is_valid_depthwise_spatial_spec,
    )
    from rocke.runtime.hip_module import HipError

    p = problem
    # Use spatial kernel when groups fit in one wave (better thread utilisation).
    # Derive wave_size from the spec default so this stays correct on wave32 targets
    # (groups == wave_size would leave zero W-positions per wave — not valid).
    _wave_size = DirectDepthwiseSpatialSpec(problem=p).wave_size
    _use_spatial = p.groups < _wave_size

    torch.manual_seed(42)
    _torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    A_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=_torch_dtype).uniform_(-1.0, 1.0)
    B_t = torch.empty(p.total_k, p.KH, p.KW, 1, dtype=_torch_dtype).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=_torch_dtype)

    bytes_xfer = float(A_t.nbytes + B_t.nbytes + D_t.nbytes)
    flop = float(p.flops)
    sig = conv_args_signature(dtype)

    # For the spatial layout block_w is derived from block_waves internally,
    # so sweeping block_w would produce duplicate kernels; use a dummy value.
    if _use_spatial:
        combos = [(None, bw) for bw in _DW_BLOCK_WAVES]
    else:
        combos = list(itertools.product(_DW_BLOCK_W, _DW_BLOCK_WAVES))

    if args.sample is not None:
        total = len(combos)
        combos = _sample_combos(combos, args.sample, args.seed)
        print(
            f"Sampling {len(combos)}/{total} depthwise combinations "
            f"({args.sample*100:.0f}%, seed={args.seed}).",
            flush=True,
        )

    print(
        f"Sweeping {len(combos)} depthwise combinations for {arch} {dtype} {p.short()} ...",
        flush=True,
    )

    n_skipped = 0
    pending = []
    for combo in combos:
        block_w, block_waves = combo
        if _use_spatial:
            spec = DirectDepthwiseSpatialSpec(
                problem=p,
                name="rocke_bench_direct_depthwise_spatial",
                block_waves=block_waves,
            )
            ok, _ = is_valid_depthwise_spatial_spec(spec, arch=arch)
        else:
            spec = DirectDepthwiseSpec(
                problem=p,
                name="rocke_bench_direct_depthwise",
                block_w=block_w,
                block_waves=block_waves,
            )
            ok, _ = is_valid_depthwise_spec(spec, arch=arch)
        if not ok:
            n_skipped += 1
            continue
        try:
            if _use_spatial:
                kernel = build_direct_depthwise_spatial(spec, arch=arch)
            else:
                kernel = build_direct_depthwise(spec, arch=arch)
        except ValueError:
            n_skipped += 1
            continue
        pending.append((combo, spec, kernel))

    artifact_map = _compile_kernels_parallel(
        [k for _, _, k in pending], compile_kernel, arch, jobs
    )
    n_built = len(artifact_map)

    rt = Runtime()
    results: "List[DepthwiseResult]" = []

    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    ref_out = None
    if args.verify or args.dump_fail:
        ref_out = _conv_reference(A_t, B_t, _DirectConvProblemAdapter(p))
        print(
            f"Reference computed via torch ({tuple(ref_out.shape)}, {ref_out.dtype}).",
            flush=True,
        )

    n_run = 0
    for combo, spec, kernel in pending:
        block_w, block_waves = combo
        artifact = artifact_map[kernel.name]

        try:
            launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=sig,
            )
        except HipError as e:
            n_skipped += 1
            print(
                f"[skip] kernel load failed for {artifact.kernel_name}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if _use_spatial:
            q_tiles = math.ceil(p.Wo / spec.block_w)
            g_tiles = 1  # all channels handled within each wavefront
        else:
            q_tiles = math.ceil(p.Wo / block_w)
            g_tiles = math.ceil(p.groups / spec.block_ch)
        grid = (q_tiles, g_tiles, p.N)
        block = (spec.threads_per_block, 1, 1)
        stream = 0
        values = {
            "A": A_dev,
            "B": B_dev,
            "D": D_dev,
            "A_bytes": A_t.nbytes,
            "B_bytes": B_t.nbytes,
            "D_bytes": D_t.nbytes,
        }
        cfg = LaunchConfig(grid=grid, block=block, stream=stream)

        kernel_passed = None
        if args.verify or args.dump_fail:
            rt.memset(D_dev, 0, D_t.nbytes)
            stopped, kernel_passed = _verify_kernel(
                rt=rt,
                launcher=launcher,
                values=values,
                grid=grid,
                block=block,
                out_dev=D_dev,
                out_t=D_t,
                ref_out=ref_out,
                kernel_name=artifact.kernel_name,
                dump_fail=args.dump_fail,
                u8=u8,
            )
            if stopped:
                rt.free(A_dev)
                rt.free(B_dev)
                rt.free(D_dev)
                return 1, []
            rt.memset(D_dev, 0, D_t.nbytes)

        ms = time_launches(
            lambda: launcher(values, config=cfg),
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
        )
        synchronize_and_release(stream)

        cur_tflops = (flop / ms) * 1e-9
        cur_gbps = (bytes_xfer / ms) * 1e-6
        n_run += 1

        results.append(
            DepthwiseResult(
                kernel_name=artifact.kernel_name,
                block_w=block_w,
                block_waves=block_waves,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
                passed=kernel_passed,
            )
        )
        print(
            f"[{n_run:4d}] bw={block_w:3d} bwv={block_waves}"
            f"  {cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)

    print(f"\nSweep done: {n_built} compiled, {n_skipped} skipped.", flush=True)

    if not results:
        print("No valid depthwise configurations found.", file=sys.stderr)
        return 1, []

    results.sort(key=lambda r: r.tflops, reverse=True)
    _print_depthwise_results(results, args.top, arch, p, args.verify, dtype=dtype)
    return 0, results


def _run_sweep(
    *,
    args,
    problem,
    dtype: str = "fp16",
    arch: str,
    compile_kernel,
    jobs: int,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
) -> "tuple[int, List[Result]]":
    import torch

    from rocke.helpers.manifest import conv_args_signature
    from kernels.common.conv_direct_grouped import (
        DirectConvSpec,
        build_direct_conv,
        is_valid_spec,
    )
    from rocke.runtime.hip_module import HipError

    p = problem

    torch.manual_seed(42)
    _torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    A_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=_torch_dtype).uniform_(-1.0, 1.0)
    B_t = torch.empty(p.total_k, p.KH, p.KW, p.cpg, dtype=_torch_dtype).uniform_(
        -1.0, 1.0
    )
    D_t = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=_torch_dtype)

    bytes_xfer = float(A_t.nbytes + B_t.nbytes + D_t.nbytes)
    flop = float(p.flops)
    sig = conv_args_signature(dtype)

    combos = list(itertools.product(_BLOCK_Q, _BLOCK_GROUPS, _DOUBLE_BUFFER))

    if args.sample is not None:
        total = len(combos)
        combos = _sample_combos(combos, args.sample, args.seed)
        print(
            f"Sampling {len(combos)}/{total} combinations "
            f"({args.sample*100:.0f}%, seed={args.seed}).",
            flush=True,
        )

    print(
        f"Sweeping {len(combos)} combinations for {arch} {dtype} {p.short()} "
        f"(cpg={p.cpg}) ...",
        flush=True,
    )

    n_skipped = 0
    pending = []
    for combo in combos:
        block_q, block_groups, double_buffer = combo
        spec = DirectConvSpec(
            problem=p,
            name="rocke_bench_direct_conv",
            block_q=block_q,
            block_groups=block_groups,
            double_buffer=double_buffer,
        )
        ok, _ = is_valid_spec(spec, arch=arch)
        if not ok:
            n_skipped += 1
            continue
        try:
            kernel = build_direct_conv(spec, arch=arch)
        except ValueError:
            n_skipped += 1
            continue
        pending.append((combo, spec, kernel))

    artifact_map = _compile_kernels_parallel(
        [k for _, _, k in pending], compile_kernel, arch, jobs
    )
    n_built = len(artifact_map)

    rt = Runtime()
    results: List[Result] = []

    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    ref_out = None
    if args.verify or args.dump_fail:
        ref_out = _conv_reference(A_t, B_t, _DirectConvProblemAdapter(p))
        print(
            f"Reference computed via torch ({tuple(ref_out.shape)}, {ref_out.dtype}).",
            flush=True,
        )

    n_run = 0
    for combo, spec, kernel in pending:
        block_q, block_groups, double_buffer = combo
        artifact = artifact_map[kernel.name]

        try:
            launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=sig,
            )
        except HipError as e:
            n_skipped += 1
            print(
                f"[skip] kernel load failed for {artifact.kernel_name}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

        q_tiles = (p.Wo + block_q - 1) // block_q
        g_tiles = p.groups // block_groups
        grid = (q_tiles, g_tiles, p.N)
        block = (spec.threads_per_block, 1, 1)
        stream = 0
        values = {
            "A": A_dev,
            "B": B_dev,
            "D": D_dev,
            "A_bytes": A_t.nbytes,
            "B_bytes": B_t.nbytes,
            "D_bytes": D_t.nbytes,
        }
        cfg = LaunchConfig(grid=grid, block=block, stream=stream)

        kernel_passed = None
        if args.verify or args.dump_fail:
            rt.memset(D_dev, 0, D_t.nbytes)
            stopped, kernel_passed = _verify_kernel(
                rt=rt,
                launcher=launcher,
                values=values,
                grid=grid,
                block=block,
                out_dev=D_dev,
                out_t=D_t,
                ref_out=ref_out,
                kernel_name=artifact.kernel_name,
                dump_fail=args.dump_fail,
                u8=u8,
            )
            if stopped:
                rt.free(A_dev)
                rt.free(B_dev)
                rt.free(D_dev)
                return 1, []
            rt.memset(D_dev, 0, D_t.nbytes)

        ms = time_launches(
            lambda: launcher(values, config=cfg),
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
        )
        synchronize_and_release(stream)

        cur_tflops = (flop / ms) * 1e-9
        cur_gbps = (bytes_xfer / ms) * 1e-6
        n_run += 1

        results.append(
            Result(
                kernel_name=artifact.kernel_name,
                block_q=block_q,
                block_groups=block_groups,
                double_buffer=double_buffer,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
                passed=kernel_passed,
            )
        )
        print(
            f"[{n_run:4d}] bq={block_q:3d} bg={block_groups:3d} "
            f"db={double_buffer}  "
            f"{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)

    print(f"\nSweep done: {n_built} compiled, {n_skipped} skipped.", flush=True)

    if not results:
        print("No valid configurations found.", file=sys.stderr)
        return 1, []

    results.sort(key=lambda r: r.tflops, reverse=True)
    _print_results(results, args.top, arch, p, args.verify, dtype=dtype)
    return 0, results


# ---------------------------------------------------------------------------
# Wgrad sweep
# ---------------------------------------------------------------------------


def _run_wgrad_sweep(
    *,
    args,
    problem,
    dtype: str = "fp16",
    arch: str,
    compile_kernel,
    jobs: int,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
) -> "tuple[int, list]":
    """Benchmark the direct wgrad kernel."""
    import torch

    from rocke.helpers.manifest import conv_args_signature
    from kernels.common.conv_direct_grouped import (
        DirectConvWgradSpec,
        build_direct_conv_wgrad,
        is_valid_wgrad_spec,
    )
    from rocke.runtime.hip_module import HipError

    p = problem
    # dY and X carry the problem dtype; dW is fp32 either way -- the
    # split-K reduction lands through fp32 global atomics.
    _torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    torch.manual_seed(42)
    X_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=_torch_dtype).uniform_(-1.0, 1.0)
    dY_t = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=_torch_dtype).uniform_(
        -1.0, 1.0
    )
    dW_t = torch.zeros(p.total_k, p.KH, p.KW, p.cpg, dtype=torch.float32)

    bytes_xfer = float(X_t.nbytes + dY_t.nbytes + dW_t.nbytes)
    flop = float(p.flops)

    sig_wg = conv_args_signature(dtype)

    # (waves_k, waves_c, waves_q). waves_c > 1 is what lets one block cover the
    # whole C axis, which is the difference between reading dY once and reading
    # it once per C tile.
    n_c_tiles_1 = (p.cpg + 15) // 16
    _WAVES = [
        (1, 1, 1),
        (2, 1, 1),
        (4, 1, 1),
        (6, 1, 1),
        (8, 1, 1),
        (2, 1, 2),
        (4, 1, 2),
        (1, 2, 1),
        (2, 2, 1),
        (4, 2, 1),
        (1, n_c_tiles_1, 1),
        (2, n_c_tiles_1, 1),
        (4, n_c_tiles_1, 1),
    ]
    _WAVES = sorted({w for w in _WAVES if w[1] <= n_c_tiles_1})
    _HPB = [30, 60, 120]
    _MK = [32]
    combos = [
        (wk, wc, wq, hpb, mk)
        for wk, wc, wq in _WAVES
        for hpb in _HPB
        for mk in _MK
        if wk * wc <= 16 and wk * wc * wq * 64 <= 1024
    ]

    print(
        f"Sweeping wgrad configurations for {arch} {dtype}→fp32 {p.short()} ...",
        flush=True,
    )

    n_skipped = 0
    pending = []
    for waves_k, waves_c, waves_q, hpb, mk in combos:
        spec = DirectConvWgradSpec(
            problem=p,
            name="rocke_bench_direct_wgrad",
            waves_k=waves_k,
            waves_c=waves_c,
            waves_q=waves_q,
            ho_per_block=hpb,
            mfma_k=mk,
        )
        ok, _ = is_valid_wgrad_spec(spec, arch=arch)
        if not ok:
            n_skipped += 1
            continue
        try:
            kernel = build_direct_conv_wgrad(spec, arch=arch)
        except ValueError:
            n_skipped += 1
            continue
        pending.append(((waves_k, waves_c, waves_q, hpb, mk), spec, kernel))

    artifact_map = _compile_kernels_parallel(
        [k for _, _, k in pending], compile_kernel, arch, jobs
    )
    n_built = len(artifact_map)

    rt = Runtime()
    results = []

    X_dev = rt.alloc(X_t.nbytes)
    dY_dev = rt.alloc(dY_t.nbytes)
    dW_dev = rt.alloc(dW_t.nbytes)
    rt.memcpy_h2d(X_dev, u8(X_t), X_t.nbytes)
    rt.memcpy_h2d(dY_dev, u8(dY_t), dY_t.nbytes)
    rt.memset(dW_dev, 0, dW_t.nbytes)

    # After the Runtime, as in every other sweep here: torch and rocke each
    # bring up their own HIP runtime, and whichever initialises second loses --
    # torch first leaves rocke's hipModuleGetFunction reporting "named symbol
    # not found" for every kernel.
    ref_out_wg = None
    if args.verify or args.dump_fail:
        ref_out_wg = _wgrad_reference_shared(X_t, dY_t, _DirectConvProblemAdapter(p))
        print(
            f"Reference wgrad computed via torch ({tuple(ref_out_wg.shape)}, {ref_out_wg.dtype}).",
            flush=True,
        )

    n_run = 0
    for combo, spec, kernel in pending:
        waves_k, waves_c, waves_q, hpb, mk = combo
        artifact = artifact_map[kernel.name]

        try:
            launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=sig_wg,
            )
        except HipError as e:
            n_skipped += 1
            print(f"[skip] {artifact.kernel_name}: {e}", file=sys.stderr, flush=True)
            continue

        # Grid: bx = (group*n_k_tiles+k_tile)*n_c_tiles + c_tile
        #       by = ho_block, bz = n * n_wo_tiles + wo_tile
        # Delta register ring: each block owns one wo_tile (WO_BLOCK cols), iterates H rows.
        # S-strips reused KH× via register ring → ~12× fewer loads vs old approach.
        n_k_tiles = (p.kpg + spec.block_k - 1) // spec.block_k
        n_c_tiles = (p.cpg + spec.block_c - 1) // spec.block_c
        n_q_blocks = spec.n_q_blocks()  # ceil(n_wo_tiles / waves_q)
        n_hi_blocks = spec.n_ho_blocks()  # ceil(H / ho_per_block)
        grid = (p.groups * n_k_tiles * n_c_tiles, n_hi_blocks, p.N * n_q_blocks)
        block_dim = (spec.threads_per_block, 1, 1)
        values = {
            "A": dY_dev,
            "B": X_dev,
            "D": dW_dev,
            "A_bytes": dY_t.nbytes,
            "B_bytes": X_t.nbytes,
            "D_bytes": dW_t.nbytes,
        }

        kernel_passed = None
        if (args.verify or args.dump_fail) and ref_out_wg is not None:
            # Wgrad outputs fp32 — convert ref to a float16-shaped tensor for _verify_kernel.
            # We keep everything in fp32 and just reuse the verify infrastructure.
            import torch

            dW_ref_t = ref_out_wg.cpu()
            rt.memset(dW_dev, 0, dW_t.nbytes)
            launcher(
                values, config=LaunchConfig(grid=grid, block=block_dim, fence=True)
            )
            dW_cpu = torch.empty_like(dW_t)
            rt.memcpy_d2h(u8(dW_cpu), dW_dev, dW_t.nbytes)
            abs_diff = (dW_cpu.float() - dW_ref_t.float()).abs()
            ref_scale = dW_ref_t.float().abs().max().clamp(min=1.0)
            rel_err = float(abs_diff.max() / ref_scale)
            tol = 5e-2
            kernel_passed = rel_err < tol
            status = "PASS" if kernel_passed else f"FAIL(rel_err={rel_err:.2e})"
            print(f"  verify {artifact.kernel_name}: {status}", flush=True)
            rt.memset(dW_dev, 0, dW_t.nbytes)

        cfg_wg = LaunchConfig(grid=grid, block=block_dim)
        ms = time_launches(
            lambda: launcher(values, config=cfg_wg),
            warmup=args.warmup,
            iters=args.iters,
            stream=0,
        )
        synchronize_and_release(0)
        tflops = flop / ms / 1e9
        gbps = bytes_xfer / ms / 1e6
        passed_str = (
            f"  {'PASS' if kernel_passed else 'FAIL'}"
            if kernel_passed is not None
            else ""
        )
        n_blocks = grid[0] * grid[1] * grid[2]
        results.append(
            {
                "wk": waves_k,
                "wc": waves_c,
                "wq": waves_q,
                "hpb": hpb,
                "mk": mk,
                "ms": ms,
                "tflops": tflops,
                "gbps": gbps,
                "passed": kernel_passed,
                "n_blocks": n_blocks,
            }
        )
        n_run += 1
        print(
            f"[{n_run:4d}] wk={waves_k} wc={waves_c} wq={waves_q} hpb={hpb:2d} mk={mk:2d}"
            f"  blk={n_blocks:6d}  {tflops:6.1f} TFLOPS  {ms:.3f} ms{passed_str}",
            flush=True,
        )

    rt.free(X_dev)
    rt.free(dY_dev)
    rt.free(dW_dev)
    print(f"\nWgrad sweep done: {n_built} compiled, {n_skipped} skipped.", flush=True)

    if not results:
        print("No valid wgrad configurations found.", file=sys.stderr)
        return 1, []

    results.sort(key=lambda r: r["tflops"], reverse=True)
    best = results[0]
    passed_str = (
        f"  {'PASS' if best['passed'] else 'FAIL'}"
        if best["passed"] is not None
        else ""
    )
    print(
        f"\nBest wgrad: wk={best['wk']} wc={best['wc']} wq={best['wq']} hpb={best['hpb']} mk={best['mk']}  "
        f"blocks={best['n_blocks']}  {best['tflops']:.1f} TFLOPS  {best['ms']:.3f} ms{passed_str}",
        flush=True,
    )
    return 0, results


# ---------------------------------------------------------------------------
# MIOpen -F flag → direction string
# ---------------------------------------------------------------------------

# MIOpen -F bitmask: 1=fwd, 2=dgrad, 4=wgrad.
# When multiple bits are set the benchmark picks the highest-priority direction
# (wgrad > dgrad > fwd) so a single command maps to one sweep.
_FORW_TO_DIR = {
    1: "fwd",
    2: "dgrad",
    3: "dgrad",  # fwd+dgrad → dgrad
    4: "wgrad",
    5: "wgrad",  # fwd+wgrad → wgrad
    6: "wgrad",  # dgrad+wgrad → wgrad
    7: "wgrad",  # all → wgrad
}


# ---------------------------------------------------------------------------
# Dgrad sweep
# ---------------------------------------------------------------------------


def _run_dgrad_sweep(
    *,
    args,
    problem,
    dtype: str = "fp16",
    arch: str,
    compile_kernel,
    jobs: int,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
) -> "tuple[int, list]":
    """Benchmark the direct dgrad kernel.

    Dispatches to the depthwise dgrad kernel for cpg=1 (any stride) and to
    the MFMA grouped dgrad kernel for cpg>=4 (stride=1 only).
    """
    import math

    import torch

    from rocke.helpers.manifest import conv_args_signature
    from kernels.common.conv_direct_grouped import (
        DirectConvDgradSpec,
        DirectDepthwiseDgradSpec,
        build_direct_conv_dgrad,
        build_direct_depthwise_dgrad,
        is_valid_dgrad_spec,
        is_valid_depthwise_dgrad_spec,
    )
    from rocke.runtime.hip_module import HipError

    p = problem

    torch.manual_seed(42)
    _torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    dY_t = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=_torch_dtype).uniform_(
        -1.0, 1.0
    )
    W_t = torch.empty(p.total_k, p.KH, p.KW, p.cpg, dtype=_torch_dtype).uniform_(
        -1.0, 1.0
    )
    dX_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=_torch_dtype)

    bytes_xfer = float(dY_t.nbytes + W_t.nbytes + dX_t.nbytes)
    flop = float(p.flops)
    sig = conv_args_signature(dtype)

    is_depthwise = p.cpg == 1

    if is_depthwise:
        # Depthwise dgrad: use ho-streaming kernel (better DRAM efficiency).
        from kernels.common.conv_direct_grouped import (
            DirectDepthwiseDgradStreamSpec,
            DirectDepthwiseDgradWindowedSpec,
            build_direct_depthwise_dgrad_streaming,
            build_direct_depthwise_dgrad_windowed,
            is_valid_depthwise_dgrad_stream_spec,
            is_valid_depthwise_dgrad_win_spec,
        )

        large_filter = p.KH * p.KW >= _DW_DGRAD_LARGE_FILTER_TAPS
        stream_bws = [
            bw
            for bw in _DW_BLOCK_W
            if not (large_filter and bw > _DW_DGRAD_STREAM_MAX_BW_LARGE_FILTER)
        ]
        combos_dw = [
            ("stream", bw, wv)
            for bw, wv in itertools.product(stream_bws, _DW_BLOCK_WAVES)
        ]
        if p.stride == 1:
            combos_dw += [("win",) + c for c in _dw_dgrad_windowed_combos(p, arch)]
        print(
            f"Sweeping {len(combos_dw)} depthwise dgrad combinations for {arch} {dtype} "
            f"{p.short()} (stride={p.stride}) ...",
            flush=True,
        )
        n_skipped = 0
        pending = []
        for combo in combos_dw:
            if combo[0] == "win":
                _, block_w, block_waves, cpl, block_h, dot2 = combo
                spec = DirectDepthwiseDgradWindowedSpec(
                    problem=p,
                    name="rocke_bench_dw_dgrad_win",
                    block_w=block_w,
                    block_waves=block_waves,
                    ch_per_lane=cpl,
                    block_h=block_h,
                    dot2=dot2,
                )
                ok, _ = is_valid_depthwise_dgrad_win_spec(spec, arch=arch)
                build = build_direct_depthwise_dgrad_windowed
            else:
                _, block_w, block_waves = combo
                spec = DirectDepthwiseDgradStreamSpec(
                    problem=p,
                    name="rocke_bench_dw_dgrad",
                    block_w=block_w,
                    block_waves=block_waves,
                )
                ok, _ = is_valid_depthwise_dgrad_stream_spec(spec, arch=arch)
                build = build_direct_depthwise_dgrad_streaming
            if not ok:
                n_skipped += 1
                continue
            try:
                kernel = build(spec, arch=arch)
            except ValueError:
                n_skipped += 1
                continue
            pending.append((combo, spec, kernel))
    else:
        # Grouped dgrad: use the MFMA pipeline (weight pre-pass + fprop) for
        # stride=1, fall back to scalar FMA for stride > 1.
        # H-tiling (block_h) ensures enough blocks/CU even for groups=1.
        from kernels.common.conv_direct_grouped import (
            direct_mfma_dgrad_stage_kernel,
            is_valid_spec as is_valid_fprop_spec,
            make_dgrad_fprop_spec,
            plan_direct_mfma_dgrad,
        )

        use_mfma = p.stride == 1
        dgrad_family = getattr(args, "dgrad_family", "all")

        if use_mfma:
            from dataclasses import replace as dc_replace

            valid_bgs = [bg for bg in _BLOCK_GROUPS if p.groups % bg == 0]
            combos = list(itertools.product(_BLOCK_Q, valid_bgs))
            print(
                f"Sweeping MFMA dgrad over {len(combos)} (block_q, block_groups) pairs x "
                f"{len(_DGRAD_BLOCK_H)} block_h x {len(_DGRAD_WAVES_COMBOS)} wave combos "
                f"for {arch} {dtype} {p.short()} (cpg={p.cpg}, kpg={p.kpg}) ...",
                flush=True,
            )
            n_skipped = 0
            pending = []
            if dgrad_family in ("4c", "fused"):
                combos = []
            for block_q, block_groups in combos:
                for block_h in _DGRAD_BLOCK_H:
                    # One H tile covering the whole image is block_h=0 with
                    # extra runtime guards; skip it rather than time it twice.
                    if block_h >= p.Ho:
                        n_skipped += 1
                        continue
                    for (
                        waves_q,
                        waves_k,
                        use_rk,
                        use_pg,
                        use_k32,
                    ) in _DGRAD_WAVES_COMBOS:
                        if use_k32 and p.kpg % 32 != 0:
                            n_skipped += 1
                            continue
                        fprop_spec = make_dgrad_fprop_spec(
                            p,
                            block_q=block_q,
                            block_groups=block_groups,
                            block_h=block_h,
                            waves_q=waves_q,
                            waves_k=waves_k,
                            runtime_k_loop=use_rk,
                            persistent_grid=use_pg,
                        )
                        if use_k32:
                            fprop_spec = dc_replace(fprop_spec, fold_k32=True)
                        # is_valid_spec runs validate() too (waves_k against the
                        # atom width actually used, LDS footprint).
                        ok, _ = is_valid_fprop_spec(fprop_spec, arch=arch)
                        if not ok:
                            n_skipped += 1
                            continue
                        try:
                            plan = plan_direct_mfma_dgrad(p, fprop_spec)
                            stage_kernels = tuple(
                                direct_mfma_dgrad_stage_kernel(st, arch=arch)
                                for st in plan.stages
                            )
                        except ValueError:
                            n_skipped += 1
                            continue
                        pending.append(
                            (
                                (
                                    block_q,
                                    block_groups,
                                    block_h,
                                    waves_q,
                                    waves_k,
                                    use_rk,
                                    use_pg,
                                    use_k32,
                                ),
                                fprop_spec,
                                _MfmaDgradPipeline(plan=plan, kernels=stage_kernels),
                            )
                        )
            # 4c dgrad: cpg == kpg == 4 runs the batched 4x4x4 MFMA kernel on
            # (dY, W_T) after one transpose pre-pass, or on (dY, W) alone with
            # the fused weight transform.
            from kernels.common.conv_direct_grouped import (
                is_valid_dgrad_4c_problem,
                is_valid_spec_4c,
                make_dgrad_4c_spec,
            )
            from rocke.core.arch import ArchTarget

            _has_tr = ArchTarget.from_gfx(arch).memory.has_ds_read_tr
            ok4c, why4c = is_valid_dgrad_4c_problem(p)
            if dgrad_family in ("all", "4c", "fused") and ok4c:
                # 4c forms: pre-pass pipeline (not in the "fused" family) and
                # the single kernel with the fused weight transform (LDS-staged
                # transpose reads where the target has them, gathers otherwise).
                forms4c = [] if dgrad_family == "fused" else [(False, False)]
                forms4c.append((True, bool(_has_tr)))
                for (block_q, block_groups), (fw4c, wl4c) in itertools.product(
                    itertools.product(_DGRAD_4C_BLOCK_Q, _DGRAD_4C_BLOCK_GROUPS),
                    forms4c,
                ):
                    if p.groups % block_groups != 0:
                        n_skipped += 1
                        continue
                    spec4c = make_dgrad_4c_spec(
                        p,
                        block_q=block_q,
                        block_groups=block_groups,
                        dgrad_fused_weights=fw4c,
                        dgrad_weights_lds=wl4c,
                    )
                    ok, _ = is_valid_spec_4c(spec4c, arch=arch)
                    if not ok:
                        n_skipped += 1
                        continue
                    try:
                        plan = plan_direct_mfma_dgrad(p, spec4c)
                        stage_kernels = tuple(
                            direct_mfma_dgrad_stage_kernel(st, arch=arch)
                            for st in plan.stages
                        )
                    except ValueError:
                        n_skipped += 1
                        continue
                    tag4c = ("fwl" if wl4c else "fw") if fw4c else ""
                    pending.append(
                        (
                            ("4c", block_q, block_groups, tag4c),
                            spec4c,
                            _MfmaDgradPipeline(plan=plan, kernels=stage_kernels),
                        )
                    )
            elif dgrad_family == "4c":
                print(f"[skip] 4c dgrad not applicable: {why4c}", flush=True)

            # Single-kernel generic direct-MFMA dgrad: the main kernel reads W
            # with flipped / k<->c transposed addressing (dgrad_fused_weights),
            # LDS-staged + transpose reads where available, else gathers.
            if dgrad_family in ("all", "fused") and p.kpg % 4 == 0:
                k32_opts = (False, True) if p.kpg % 32 == 0 else (False,)
                for block_q, block_groups in itertools.product(_BLOCK_Q, valid_bgs):
                    for block_h, use_k32, wpe in itertools.product(
                        _DGRAD_FUSED_BLOCK_H, k32_opts, _DGRAD_FUSED_WAVES_PER_EU
                    ):
                        if block_h >= p.Ho:
                            n_skipped += 1
                            continue
                        spec_fw = None
                        for use_lds in (True, False) if _has_tr else (False,):
                            cand = make_dgrad_fprop_spec(
                                p,
                                block_q=block_q,
                                block_groups=block_groups,
                                block_h=block_h,
                                fold_k32=use_k32,
                                dgrad_fused_weights=True,
                                dgrad_weights_lds=use_lds,
                                waves_per_eu=wpe,
                            )
                            try:
                                cand.validate()
                            except ValueError:
                                continue
                            if is_valid_fprop_spec(cand, arch=arch)[0]:
                                spec_fw = cand
                                break
                        if spec_fw is None:
                            n_skipped += 1
                            continue
                        try:
                            plan = plan_direct_mfma_dgrad(p, spec_fw)
                            stage_kernels = tuple(
                                direct_mfma_dgrad_stage_kernel(st, arch=arch)
                                for st in plan.stages
                            )
                        except ValueError:
                            n_skipped += 1
                            continue
                        pending.append(
                            (
                                (
                                    "fused",
                                    block_q,
                                    block_groups,
                                    block_h,
                                    use_k32,
                                    "fwl" if spec_fw.dgrad_weights_lds else "fw",
                                    wpe,
                                ),
                                spec_fw,
                                _MfmaDgradPipeline(plan=plan, kernels=stage_kernels),
                            )
                        )
        else:
            valid_bgs = [bg for bg in _BLOCK_GROUPS if p.groups % bg == 0]
            _DGRAD_BLOCK_Q = (4, 8, 16, 32)
            combos = list(itertools.product(_DGRAD_BLOCK_Q, valid_bgs))
            print(
                f"Sweeping {len(combos)} scalar-FMA dgrad combinations for {arch} {dtype} {p.short()} "
                f"(cpg={p.cpg}, kpg={p.kpg}, stride={p.stride}) ...",
                flush=True,
            )
            n_skipped = 0
            pending = []
            for block_q, block_groups in combos:
                spec = DirectConvDgradSpec(
                    problem=p,
                    name="rocke_bench_direct_dgrad",
                    block_q=block_q,
                    block_groups=block_groups,
                )
                ok, _ = is_valid_dgrad_spec(spec, arch=arch)
                if not ok:
                    n_skipped += 1
                    continue
                try:
                    kernel = build_direct_conv_dgrad(spec, arch=arch)
                except ValueError:
                    n_skipped += 1
                    continue
                pending.append(((block_q, block_groups), spec, kernel))

    # Compile all kernels (every stage of an MFMA pipeline is its own kernel).
    all_kernels = []
    for _, _, kernel_or_pipe in pending:
        if isinstance(kernel_or_pipe, _MfmaDgradPipeline):
            all_kernels.extend(kernel_or_pipe.kernels)
        else:
            all_kernels.append(kernel_or_pipe)
    artifact_map = _compile_kernels_parallel(all_kernels, compile_kernel, arch, jobs)
    n_built = len(artifact_map)

    rt = Runtime()
    results = []

    dY_dev = rt.alloc(dY_t.nbytes)
    W_dev = rt.alloc(W_t.nbytes)
    dX_dev = rt.alloc(dX_t.nbytes)
    rt.memcpy_h2d(dY_dev, u8(dY_t), dY_t.nbytes)
    rt.memcpy_h2d(W_dev, u8(W_t), W_t.nbytes)
    rt.memset(dX_dev, 0, dX_t.nbytes)

    _I32_MAX = (1 << 31) - 1
    _too_large = [
        (name, nb)
        for name, nb in [("dY", dY_t.nbytes), ("W", W_t.nbytes), ("dX", dX_t.nbytes)]
        if nb > _I32_MAX
    ]
    if _too_large:
        desc = ", ".join(f"{n}={nb}" for n, nb in _too_large)
        print(
            f"[skip] problem too large for i32 byte-offset params ({desc}); skipping.",
            flush=True,
        )
        return 0, []

    # Workspaces of the MFMA pipeline, sized once for the largest plan.
    # Buffer roles come from the plan (kernels.common.conv_direct_grouped):
    # ws_wt (plain transposed weights) and ws_coa (coalesced preload layout).
    ws_dev = {}
    _pipes = [k for _, _, k in pending if isinstance(k, _MfmaDgradPipeline)]
    for role in ("ws_wt", "ws_coa"):
        sizes = [
            dict(pp.plan.buffer_bytes)[role]
            for pp in _pipes
            if role in dict(pp.plan.buffer_bytes)
        ]
        if sizes:
            ws_dev[role] = rt.alloc(max(sizes))
    buf_dev = {"dY": dY_dev, "W": W_dev, "dX": dX_dev, **ws_dev}

    ref_out = None
    if args.verify or args.dump_fail:
        ref_out = _dgrad_reference_shared(dY_t, W_t, _DirectConvProblemAdapter(p))
        print(
            f"Reference dgrad computed via torch ({tuple(ref_out.shape)}, {ref_out.dtype}).",
            flush=True,
        )

    _wt_sig = [
        {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "D", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "A_bytes", "type": "i32", "size_bytes": 4},
        {"name": "D_bytes", "type": "i32", "size_bytes": 4},
    ]

    n_run = 0
    for combo, spec, kernel_or_pipe in pending:
        is_mfma_pipe = isinstance(kernel_or_pipe, _MfmaDgradPipeline)

        if is_depthwise:
            if combo[0] == "win":
                _, block_w, block_waves, cpl, block_h, dot2 = combo
                grid = spec.grid()
                label = (
                    f"win bw={block_w:3d} waves={block_waves} cpl={cpl} "
                    f"bh={spec.rows_per_block if spec.h_tiles > 1 else 0}"
                    f"{' dot2' if dot2 else ''}"
                )
            else:
                _, block_w, block_waves = combo
                q_tiles = math.ceil(p.W / block_w)
                g_tiles = math.ceil(p.groups / spec.block_ch)
                grid = (q_tiles, g_tiles, p.N)
                label = f"bw={block_w:3d} waves={block_waves}"
            block_dim = (spec.threads_per_block, 1, 1)
            kernel = kernel_or_pipe
        elif is_mfma_pipe:
            label = _mfma_dgrad_label(combo)
        else:
            block_q, block_groups = combo
            block_ch = spec.block_groups * spec.wave_size
            q_tiles = math.ceil(p.W / block_q)
            c_tiles = math.ceil(p.total_c / block_ch)
            grid = (q_tiles, c_tiles, p.N)
            label = f"bq={block_q:3d} bg={block_groups:3d} scFMA"
            block_dim = (spec.threads_per_block, 1, 1)
            kernel = kernel_or_pipe

        if is_mfma_pipe:
            plan = kernel_or_pipe.plan
            nbytes = dict(plan.buffer_bytes)
            stage_calls = []
            try:
                for st, k in zip(plan.stages, kernel_or_pipe.kernels):
                    art = artifact_map.get(k.name)
                    if art is None:
                        raise KeyError(k.name)
                    if st.b is None:
                        stage_launcher = KernelLauncher(
                            hsaco=art.hsaco,
                            kernel_name=art.kernel_name,
                            signature=_wt_sig,
                        )
                        stage_values = {
                            "A": buf_dev[st.a],
                            "D": buf_dev[st.d],
                            "A_bytes": nbytes[st.a],
                            "D_bytes": nbytes[st.d],
                        }
                    else:
                        stage_launcher = KernelLauncher(
                            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
                        )
                        stage_values = {
                            "A": buf_dev[st.a],
                            "B": buf_dev[st.b],
                            "D": buf_dev[st.d],
                            "A_bytes": nbytes[st.a],
                            "B_bytes": nbytes[st.b],
                            "D_bytes": nbytes[st.d],
                        }
                    stage_calls.append(
                        (
                            stage_launcher,
                            stage_values,
                            LaunchConfig(grid=st.grid, block=st.block),
                        )
                    )
            except (KeyError, HipError) as e:
                n_skipped += 1
                print(f"[skip] {label}: {e}", file=sys.stderr, flush=True)
                continue
            main_name = artifact_map[kernel_or_pipe.kernels[-1].name].kernel_name

            def run_mfma_dgrad(_calls=stage_calls):
                for _launcher, _values, _cfg in _calls:
                    _launcher(_values, config=_cfg)

        else:
            artifact = artifact_map.get(kernel.name)
            if artifact is None:
                n_skipped += 1
                continue
            try:
                launcher = KernelLauncher(
                    hsaco=artifact.hsaco,
                    kernel_name=artifact.kernel_name,
                    signature=sig,
                )
            except HipError as e:
                n_skipped += 1
                print(
                    f"[skip] {artifact.kernel_name}: {e}", file=sys.stderr, flush=True
                )
                continue
            values = {
                "A": dY_dev,
                "B": W_dev,
                "D": dX_dev,
                "A_bytes": dY_t.nbytes,
                "B_bytes": W_t.nbytes,
                "D_bytes": dX_t.nbytes,
            }

        kernel_passed = None
        if args.verify or args.dump_fail:
            if is_mfma_pipe:
                # Poison dX and the workspaces with 0xFF (NaN in fp16 and bf16)
                # so an unwritten output or an unwritten workspace lane that the
                # main kernel reads shows up as a failure instead of a lucky zero.
                rt.memset(dX_dev, 0xFF, dX_t.nbytes)
                for role, ptr in ws_dev.items():
                    if role in nbytes:
                        rt.memset(ptr, 0xFF, nbytes[role])
                run_mfma_dgrad()
                synchronize_and_release(0)
                dX_cpu = torch.empty_like(dX_t)
                rt.memcpy_d2h(u8(dX_cpu), dX_dev, dX_t.nbytes)
                if ref_out is not None:
                    out_f32 = dX_cpu.float()
                    ref_f32 = ref_out.float().cpu()
                    abs_diff = (out_f32 - ref_f32).abs()
                    rel_err = float(abs_diff.max() / ref_f32.abs().max().clamp(min=1.0))
                    n_bad = _conv_rule_bad_count(dX_cpu, ref_out, dtype)
                    kernel_passed = n_bad == 0
                    status = (
                        f"PASS(rel_err={rel_err:.2e})"
                        if kernel_passed
                        else f"FAIL(bad={n_bad}/{dX_cpu.numel()}, rel_err={rel_err:.2e})"
                    )
                    print(f"  verify {main_name} [{label}]: {status}", flush=True)
                rt.memset(dX_dev, 0, dX_t.nbytes)
            else:
                stopped, kernel_passed = _verify_kernel(
                    rt=rt,
                    launcher=launcher,
                    values=values,
                    grid=grid,
                    block=block_dim,
                    out_dev=dX_dev,
                    out_t=dX_t,
                    ref_out=ref_out,
                    kernel_name=artifact.kernel_name,
                    dump_fail=args.dump_fail,
                    u8=u8,
                )
                if stopped:
                    rt.free(dY_dev)
                    rt.free(W_dev)
                    rt.free(dX_dev)
                    for ptr in ws_dev.values():
                        rt.free(ptr)
                    return 1, []
                rt.memset(dX_dev, 0, dX_t.nbytes)

        if is_mfma_pipe:
            ms = time_launches(
                run_mfma_dgrad,
                warmup=args.warmup,
                iters=args.iters,
                stream=0,
            )
        else:
            cfg = LaunchConfig(grid=grid, block=block_dim)
            ms = time_launches(
                lambda: launcher(values, config=cfg),
                warmup=args.warmup,
                iters=args.iters,
                stream=0,
            )
        synchronize_and_release(0)
        tflops = flop / ms / 1e9
        gbps = bytes_xfer / ms / 1e6
        passed_str = (
            f"  {'PASS' if kernel_passed else 'FAIL'}"
            if kernel_passed is not None
            else ""
        )
        results.append(
            {
                "label": label,
                "ms": ms,
                "tflops": tflops,
                "gbps": gbps,
                "passed": kernel_passed,
            }
        )
        n_run += 1
        print(
            f"[{n_run:4d}] {label}  {tflops:6.1f} TFLOPS  {ms:.4f} ms{passed_str}",
            flush=True,
        )

    rt.free(dY_dev)
    rt.free(W_dev)
    rt.free(dX_dev)
    for ptr in ws_dev.values():
        rt.free(ptr)
    print(f"\nDgrad sweep done: {n_built} compiled, {n_skipped} skipped.", flush=True)

    if not results:
        print("No valid dgrad configurations found.", file=sys.stderr)
        return 1, []

    results.sort(key=lambda r: r["tflops"], reverse=True)
    best = results[0]
    passed_str = (
        f"  {'PASS' if best['passed'] else 'FAIL'}"
        if best["passed"] is not None
        else ""
    )
    print(
        f"\nBest: {best['tflops']:.1f} TFLOPS — {best['label']}  {best['ms']:.3f} ms{passed_str}",
        flush=True,
    )
    return 0, results


# ---------------------------------------------------------------------------
# MIOpen -F flag → direction string
# ---------------------------------------------------------------------------


def _miopen_forw_to_direction(forw: int) -> "str | None":
    """Map MIOpen -F value to a direction string, or None if unsupported."""
    return _FORW_TO_DIR.get(forw & 7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tile sweep benchmark for the parametric direct grouped convolution kernel. "
            "The kernel variant is selected automatically from C/groups (cpg)."
        )
    )
    parser.add_argument(
        "--arch",
        default="gfx950",
        help="gfx target (gfx942, gfx950, ...) (default: gfx950)",
    )
    parser.add_argument(
        "--direction",
        default="fwd",
        choices=["fwd", "dgrad", "wgrad"],
        help="convolution direction to benchmark: fwd (default), dgrad, wgrad",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="print top-N results ranked by TFLOPS (default: 10)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="timed iterations (default: 10)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "parallel compile workers (default: 1, serial). "
            "Set to 0 to use os.cpu_count() workers."
        ),
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRAC",
        help="randomly sample FRAC of candidate combinations (e.g. 0.1 for 10%%).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed used by --sample (default: 0)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify each kernel against torch reference before timing",
    )
    parser.add_argument(
        "--dgrad-family",
        default="all",
        choices=["all", "generic", "4c", "fused"],
        dest="dgrad_family",
        help=(
            "grouped stride-1 dgrad: sweep the generic direct-MFMA pre-pass "
            "pipeline, the 4c (cpg=kpg=4, batched 4x4x4 MFMA) forms, the "
            "single-kernel forms with the fused weight transform (generic and "
            "4c), or everything (default: all)"
        ),
    )
    parser.add_argument(
        "--dump-fail",
        default=None,
        metavar="PATH",
        dest="dump_fail",
        help="on the first verify FAIL, dump tensors to PATH/ and stop the sweep.",
    )

    miopen_grp = parser.add_argument_group(
        "MIOpen input",
        "Load the conv problem from a MIOpenDriver command instead of explicit shape flags. "
        "When set, DirectConvProblem is derived from the command; --dtype / shape flags are ignored. "
        "Only forward (fwd) 2-D NHWC convolutions are supported. "
        "cpg must equal kpg and be 1 (depthwise) or a positive multiple of 4 (grouped).",
    )
    miopen_grp.add_argument(
        "--miopen-cmd",
        default=None,
        metavar="CMD",
        help="MIOpenDriver command string, e.g. "
        '"./MIOpenDriver convfp16 -n 8 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 '
        '-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 64 -F 1 -in_layout=NHWC"',
    )
    miopen_grp.add_argument(
        "--miopen-file",
        default=None,
        metavar="FILE",
        help="Path to a file containing one MIOpenDriver command per line; "
        "the benchmark is run once per line (blank lines and # comments ignored).",
    )

    conv = parser.add_argument_group(
        "DirectConvProblem", "convolution shape parameters"
    )
    conv.add_argument("--N", type=int, default=8, help="batch size")
    conv.add_argument("--Hi", type=int, default=56, help="input height")
    conv.add_argument("--Wi", type=int, default=56, help="input width")
    conv.add_argument("--C", type=int, default=64, help="input channels")
    conv.add_argument("--K", type=int, default=64, help="output channels / filters")
    conv.add_argument("--Y", type=int, default=3, help="filter height")
    conv.add_argument("--X", type=int, default=3, help="filter width")
    conv.add_argument("--sH", type=int, default=1, help="vertical stride")
    conv.add_argument("--sW", type=int, default=1, help="horizontal stride")
    conv.add_argument("--pH", type=int, default=1, help="vertical padding")
    conv.add_argument("--pW", type=int, default=1, help="horizontal padding")
    conv.add_argument("--dH", type=int, default=1, help="vertical dilation")
    conv.add_argument("--dW", type=int, default=1, help="horizontal dilation")
    conv.add_argument(
        "--groups",
        "-g",
        type=int,
        default=1,
        help="number of conv groups; C and K must each be divisible by groups (default: 1)",
    )
    conv.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="I/O data type when using shape flags (default: fp16; ignored when using --miopen-cmd/--miopen-file)",
    )

    args = parser.parse_args()

    import ctypes

    from rocke import compile_kernel
    from kernels.common.conv_direct_grouped import DirectConvProblem
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    arch = args.arch

    # Build list of (problem, dtype) cases.
    cases: list  # List[Tuple[DirectConvProblem, str]]
    if args.miopen_file is not None:
        path = args.miopen_file
        lines = open(path).readlines()
        cases = []
        for lineno, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                prob, dt, forw = parse_miopen_cmd_direct(line)
                direction = _miopen_forw_to_direction(forw)
                if direction is None:
                    print(
                        f"[skip] {path}:{lineno}: -F={forw} maps to no supported direction",
                        file=sys.stderr,
                    )
                else:
                    cases.append((prob, dt, direction))
            except ValueError as e:
                print(f"[warn] {path}:{lineno}: skipping — {e}", file=sys.stderr)
        if not cases:
            print(f"error: {path}: no valid cases found", file=sys.stderr)
            return 2
    elif args.miopen_cmd is not None:
        try:
            prob, dt, forw = parse_miopen_cmd_direct(args.miopen_cmd)
        except ValueError as e:
            print(f"error: --miopen-cmd: {e}", file=sys.stderr)
            return 2
        direction = _miopen_forw_to_direction(forw)
        if direction is None:
            print(
                f"error: --miopen-cmd: -F={forw} maps to no supported direction "
                f"(use 1=fwd, 2=dgrad, 4=wgrad)",
                file=sys.stderr,
            )
            return 2
        cases = [(prob, dt, direction)]
    else:
        if args.C % args.groups != 0:
            print(
                f"error: C={args.C} is not divisible by groups={args.groups}",
                file=sys.stderr,
            )
            return 2
        if args.K % args.groups != 0:
            print(
                f"error: K={args.K} is not divisible by groups={args.groups}",
                file=sys.stderr,
            )
            return 2

        cpg = args.C // args.groups
        kpg = args.K // args.groups

        # For fwd direction, cpg == kpg is required. For dgrad it need not hold.
        if args.direction == "fwd" and cpg != kpg:
            print(
                f"error: cpg={cpg} != kpg={kpg}; forward direct grouped conv requires C/groups == K/groups",
                file=sys.stderr,
            )
            return 2

        if args.direction != "dgrad" and cpg != 1 and (cpg % 4 != 0 or cpg < 4):
            print(
                f"error: cpg={cpg} (C/groups={args.C}/{args.groups}) must be 1 (depthwise) "
                f"or a positive multiple of 4 (grouped)",
                file=sys.stderr,
            )
            return 2

        if args.sH != args.sW:
            print(
                f"warning: sH={args.sH} != sW={args.sW}; using sH={args.sH}",
                file=sys.stderr,
            )
        if args.pH != args.pW:
            print(
                f"warning: pH={args.pH} != pW={args.pW}; using pH={args.pH}",
                file=sys.stderr,
            )

        problem = DirectConvProblem(
            N=args.N,
            H=args.Hi,
            W=args.Wi,
            groups=args.groups,
            cpg=cpg,
            kpg=kpg,
            KH=args.Y,
            KW=args.X,
            PAD=args.pH,
            stride=args.sH,
            dtype=args.dtype,
        )
        cases = [(problem, args.dtype, args.direction)]

    _common = dict(
        args=args,
        arch=arch,
        compile_kernel=compile_kernel,
        jobs=args.jobs,
        synchronize_and_release=synchronize_and_release,
        time_launches=time_launches,
        Runtime=Runtime,
        KernelLauncher=KernelLauncher,
        LaunchConfig=LaunchConfig,
        u8=_u8,
    )

    all_rc = 0
    for case_idx, (problem, dtype, direction) in enumerate(cases):
        if len(cases) > 1:
            print(f"\n{'#'*72}", flush=True)
            print(
                f"# Case {case_idx + 1}/{len(cases)}: {problem.short()} dtype={dtype} dir={direction}",
                flush=True,
            )
            print(f"{'#'*72}", flush=True)

        cpg = problem.cpg
        if direction == "dgrad":
            rc, _ = _run_dgrad_sweep(problem=problem, dtype=dtype, **_common)
        elif direction == "wgrad":
            rc, _ = _run_wgrad_sweep(problem=problem, dtype=dtype, **_common)
        elif cpg == 1:
            rc, _ = _run_depthwise_sweep(problem=problem, dtype=dtype, **_common)
        else:
            rc, _ = _run_sweep(problem=problem, dtype=dtype, **_common)
        all_rc = all_rc or rc

    return all_rc


if __name__ == "__main__":
    raise SystemExit(main())
