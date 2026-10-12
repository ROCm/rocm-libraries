# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tile sweep benchmark for the parametric direct convolution kernels.

Three kernel families are covered:
  cpg == 1  (groups == C == K) — depthwise:   ``DirectDepthwiseSpec``, scalar fma.
  groups == 1                  — non-grouped: ``DirectNongroupedConvSpec``, LDS halo
                                 reuse + mfma_f32_32x32x16 / 16x16x32.
  cpg >= 4, cpg % 4 == 0       — grouped:     ``DirectConvSpec``, mfma_f32_16x16x16.

The variant is selected automatically from C / groups.

Run examples:
  python benchmark_direct_conv.py --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --groups 64   # depthwise
  python benchmark_direct_conv.py --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --groups 1    # non-grouped
  python benchmark_direct_conv.py --N 8 --Hi 56 --Wi 56 --C 1024 --K 1024 --groups 64 --verify

Kernel cache. Without cache flags every run builds its own kernels. The
kernels take N, H, W and groups as kernel arguments (the non-grouped one also C
and K), so they can instead be compiled once and reused for any matching shape:
  python benchmark_direct_conv.py --compile-all --cache-dir ./kernel_cache --jobs 0
  python benchmark_direct_conv.py --run-from-cache ./kernel_cache --N 8 --Hi 56 --Wi 56 \
      --C 1024 --K 1024 --groups 64 --verify
The capability list (filters, strides, paddings, channels per group) lives in
benchmarks/common/direct_kernel_sweep.py.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from typing import List

os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

from benchmarks.common.early_stop import EarlyStop, add_early_stop_arg
from builders.common.conv_reference import conv_reference as _conv_reference
from builders.common.conv_reference import dgrad_reference as _dgrad_reference_shared
from builders.common.conv_reference import wgrad_reference as _wgrad_reference_shared

# ---------------------------------------------------------------------------
# Swept parameter grids -- one source for this JIT sweep and the AOT cache
# ---------------------------------------------------------------------------

from benchmarks.common.direct_kernel_sweep import (
    BLOCK_GROUPS as _BLOCK_GROUPS,
    BLOCK_Q as _BLOCK_Q,
    DGRAD_BLOCK_H as _DGRAD_BLOCK_H,
    DGRAD_BLOCK_Q as _DGRAD_BLOCK_Q,
    DGRAD_WAVES as _DGRAD_WAVES,
    DOUBLE_BUFFER as _DOUBLE_BUFFER,
    DW_BLOCK_W_DGRAD as _DW_BLOCK_W_DGRAD,
    DW_BLOCK_W_FWD as _DW_BLOCK_W_FWD,
    DW_BLOCK_WAVES as _DW_BLOCK_WAVES,
    DW_TILED_BLOCK_H as _DW_TILED_BLOCK_H,
    DW_TILED_BLOCK_W as _DW_TILED_BLOCK_W,
    DW_TILED_BLOCK_WAVES as _DW_TILED_BLOCK_WAVES,
    DW_TILED_UNROLL as _DW_TILED_UNROLL,
    DW_COL_BLOCK_H as _DW_COL_BLOCK_H,
    DW_COL_BLOCK_W as _DW_COL_BLOCK_W,
)

# The non-grouped (groups == 1) geometry sweep lives next to the kernel, in
# ``kernels.common.conv_direct_nongrouped.nongrouped_specs``, because the useful
# tile widths depend on Wo.


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


@dataclass
class NonGroupedResult:
    kernel_name: str
    label: str
    ms: float
    tflops: float
    gbps: float
    passed: "bool | None" = None


@dataclass
class DepthwiseResult:
    kernel_name: str
    variant: str
    block_w: int
    block_waves: int
    ms: float
    tflops: float
    gbps: float
    passed: "bool | None" = None
    # The config column for an entry _dw_config_label cannot describe (the
    # output-stationary kernel's tile).
    label: "str | None" = None
    # Output-row tile of the col variant; None for preload and spatial.
    block_h: "int | None" = None


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

    dilated = miopen_args.dH != 1 or miopen_args.dW != 1
    if dilated and (miopen_args.forw != 1 or groups != 1):
        raise ValueError(
            f"direct conv computes a dilated convolution only forward and "
            f"ungrouped (got -l {miopen_args.dH} -j {miopen_args.dW}, "
            f"-F {miopen_args.forw}, -g {groups})"
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
        dil_h=miopen_args.dH,
        dil_w=miopen_args.dW,
    )
    return problem, dtype, miopen_args.forw


def _dw_tiled_label(block_h, block_w, block_waves, unroll_rows) -> str:
    """An output-stationary depthwise entry's knobs."""
    ur = " ur" if unroll_rows else ""
    return f"tiled   bh={block_h:>4d} bw={block_w:>3d} bwv={block_waves}{ur}"


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
        self.dH = p.dil_h
        self.dW = p.dil_w
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
        cfg = r.label or (
            f"{r.variant:<7} bh={str(r.block_h):>4s} bw={str(r.block_w):>3s} "
            f"bwv={r.block_waves}"
        )
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
    import torch

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_grouped import (
        DirectDepthwiseColSpec,
        DirectDepthwiseSpec,
        DirectDepthwiseSpatialSpec,
        DirectDepthwiseTiledSpec,
        direct_launch_geometry,
        build_direct_depthwise,
        build_direct_depthwise_col,
        build_direct_depthwise_spatial,
        build_direct_depthwise_tiled,
        depthwise_stream_register_reason,
        is_valid_depthwise_col_spec,
        is_valid_depthwise_spec,
        is_valid_depthwise_spatial_spec,
        is_valid_depthwise_tiled_spec,
    )
    from rocke.runtime.hip_module import HipError

    p = problem
    # Use spatial kernel when groups fit in one wave (better thread utilisation).
    # Derive wave_size from the spec default so this stays correct on wave32 targets
    # (groups == wave_size would leave zero W-positions per wave — not valid).
    _wave_size = DirectDepthwiseSpatialSpec(problem=p).wave_size
    _use_spatial = p.groups < _wave_size

    # All three variants take fp16 and bf16 -- preload and spatial off
    # problem.dtype, col off DirectDepthwiseColSpec.dtype -- so every dtype
    # sweeps the full bake-off rather than a subset of the variants.
    _torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

    torch.manual_seed(42)
    A_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=_torch_dtype).uniform_(-1.0, 1.0)
    B_t = torch.empty(p.total_k, p.KH, p.KW, 1, dtype=_torch_dtype).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=_torch_dtype)

    bytes_xfer = float(A_t.nbytes + B_t.nbytes + D_t.nbytes)
    flop = float(p.flops)
    # Direct conv is AOT: the whole shape travels as kernargs.
    sig = conv_direct_args_signature(dtype)
    _direct_args = ConvArgs.from_problem(p)

    # For the spatial layout block_w is derived from block_waves internally,
    # so sweeping block_w would produce duplicate kernels; use a dummy value.
    # Combos are (variant, block_h, block_w, block_waves, unroll_rows); block_h
    # is the col/tiled output-row tile and unroll_rows is tiled-only.
    _col_combos = [
        ("col", bh, bw, bwv, None)
        for bh, bw, bwv in itertools.product(
            _DW_COL_BLOCK_H, _DW_COL_BLOCK_W, _DW_BLOCK_WAVES
        )
    ]
    if _use_spatial:
        combos = [("spatial", None, None, bw, None) for bw in _DW_BLOCK_WAVES]
    else:
        combos = [
            ("preload", None, bw, bwv, None)
            for bw, bwv in itertools.product(_DW_BLOCK_W_FWD, _DW_BLOCK_WAVES)
        ] + _col_combos
    # The output-stationary kernel serves any group count.
    combos += [
        ("tiled", bh, bw, wv, ur)
        for bw, bh, wv, ur in itertools.product(
            _DW_TILED_BLOCK_W,
            _DW_TILED_BLOCK_H,
            _DW_TILED_BLOCK_WAVES,
            _DW_TILED_UNROLL,
        )
    ]

    if args.sample is not None:
        total = len(combos)
        combos = _sample_combos(combos, args.sample, args.seed)
        print(
            f"Sampling {len(combos)}/{total} depthwise combinations "
            f"({args.sample*100:.0f}%, seed={args.seed}).",
            flush=True,
        )

    print(
        f"Sweeping {len(combos)} depthwise combinations for {arch} {dtype} "
        f"{p.short()} ...",
        flush=True,
    )

    n_skipped = 0
    pending = []
    for combo in combos:
        variant, block_h, block_w, block_waves, unroll_rows = combo
        if variant == "tiled":
            spec = DirectDepthwiseTiledSpec(
                problem=p,
                name="rocke_bench_direct_depthwise_tiled",
                block_w=block_w,
                block_h=block_h,
                block_waves=block_waves,
                unroll_rows=unroll_rows,
            )
            ok, _ = is_valid_depthwise_tiled_spec(spec, arch=arch)
            build = build_direct_depthwise_tiled
        elif variant == "spatial":
            spec = DirectDepthwiseSpatialSpec(
                problem=p,
                name="rocke_bench_direct_depthwise_spatial",
                block_waves=block_waves,
            )
            ok, _ = is_valid_depthwise_spatial_spec(spec, arch=arch)
            build = build_direct_depthwise_spatial
        elif variant == "col":
            spec = DirectDepthwiseColSpec(
                problem=p,
                name="rocke_bench_direct_depthwise_col",
                block_h=block_h,
                block_w=block_w,
                block_waves=block_waves,
                dtype=dtype,
            )
            ok, _ = is_valid_depthwise_col_spec(spec, arch=arch)
            build = build_direct_depthwise_col
        else:
            spec = DirectDepthwiseSpec(
                problem=p,
                name="rocke_bench_direct_depthwise",
                block_w=block_w,
                block_waves=block_waves,
            )
            ok, _ = is_valid_depthwise_spec(spec, arch=arch)
            build = build_direct_depthwise
        # A spilling row-streaming kernel takes minutes to build and is left to
        # the output-stationary kernel (as in the AOT grid).
        if variant in ("preload", "spatial"):
            ok = ok and depthwise_stream_register_reason(spec) is None
        if not ok:
            n_skipped += 1
            continue
        try:
            kernel = build(spec, arch=arch)
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

    _stop = EarlyStop.for_case(args, problem, dtype, "fwd")
    n_run = 0
    for combo, spec, kernel in pending:
        variant, block_h, block_w, block_waves, unroll_rows = combo
        tiled_label = None
        if variant == "tiled":
            tiled_label = _dw_tiled_label(block_h, block_w, block_waves, unroll_rows)
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

        grid, block = direct_launch_geometry(spec)
        stream = 0
        values = _direct_args.to_launch_values(
            int(A_dev),
            int(B_dev),
            int(D_dev),
            A_t.nbytes,
            B_t.nbytes,
            D_t.nbytes,
        )
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

        ms = _stop.measure(
            launcher.bind(values, config=cfg),
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
            passed=kernel_passed,
        )
        if ms is None:
            _stop.report(artifact.kernel_name)
            continue
        synchronize_and_release(stream)

        cur_tflops = (flop / ms) * 1e-9
        cur_gbps = (bytes_xfer / ms) * 1e-6
        n_run += 1

        results.append(
            DepthwiseResult(
                kernel_name=artifact.kernel_name,
                variant=variant,
                block_w=block_w,
                block_waves=block_waves,
                block_h=block_h,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
                passed=kernel_passed,
                label=tiled_label,
            )
        )
        print(
            f"[{n_run:4d}] "
            + (
                tiled_label
                or f"{variant:<7} bh={str(block_h):>4s} bw={str(block_w):>3s} "
                f"bwv={block_waves}"
            )
            + f"  {cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)

    print(f"\nSweep done: {n_built} compiled, {n_skipped} skipped.", flush=True)

    if not results:
        print("No valid depthwise configurations found.", file=sys.stderr)
        if _stop.summary():
            print(_stop.summary(), file=sys.stderr)
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

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_grouped import (
        DirectConvSpec,
        build_direct_conv,
        direct_launch_geometry,
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
    # Direct conv is AOT: the whole shape travels as kernargs.
    sig = conv_direct_args_signature(dtype)
    _direct_args = ConvArgs.from_problem(p)

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

    _stop = EarlyStop.for_case(args, problem, dtype, "fwd")
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

        grid, block = direct_launch_geometry(spec)
        stream = 0
        values = _direct_args.to_launch_values(
            int(A_dev),
            int(B_dev),
            int(D_dev),
            A_t.nbytes,
            B_t.nbytes,
            D_t.nbytes,
        )
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

        ms = _stop.measure(
            launcher.bind(values, config=cfg),
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
            passed=kernel_passed,
        )
        if ms is None:
            _stop.report(artifact.kernel_name)
            continue
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
        if _stop.summary():
            print(_stop.summary(), file=sys.stderr)
        return 1, []

    results.sort(key=lambda r: r.tflops, reverse=True)
    _print_results(results, args.top, arch, p, args.verify, dtype=dtype)
    return 0, results


# ---------------------------------------------------------------------------
# Non-grouped (groups == 1) sweep
# ---------------------------------------------------------------------------


def _run_nongrouped_sweep(
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
) -> "tuple[int, List[NonGroupedResult]]":
    """Sweep :class:`DirectNongroupedConvSpec` over tile / wave / swizzle geometry.

    The grouped ``DirectConvSpec`` draws its parallelism from the ``groups``
    axis and collapses to a single wave per output row when ``groups == 1``,
    so it is not a usable fallback here.
    """
    import torch

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_nongrouped import (
        build_direct_conv_nongrouped,
        nongrouped_specs,
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
    # Direct conv is AOT: the whole shape travels as kernargs.
    sig = conv_direct_args_signature(dtype)

    specs = nongrouped_specs(p, arch=arch, name="rocke_bench_direct_conv_nongrouped")
    if args.sample is not None:
        total = len(specs)
        specs = _sample_combos(specs, args.sample, args.seed)
        print(
            f"Sampling {len(specs)}/{total} combinations "
            f"({args.sample*100:.0f}%, seed={args.seed}).",
            flush=True,
        )

    print(
        f"Sweeping {len(specs)} non-grouped combinations for {arch} {dtype} "
        f"{p.short()} (C={p.cpg}, K={p.kpg}) ...",
        flush=True,
    )

    n_skipped = 0
    pending = []
    for spec in specs:
        try:
            kernel = build_direct_conv_nongrouped(spec, arch=arch)
        except ValueError:
            n_skipped += 1
            continue
        pending.append((spec, kernel))

    if not pending:
        print("No valid non-grouped configurations for this shape.", file=sys.stderr)
        return 1, []

    artifact_map = _compile_kernels_parallel(
        [k for _, k in pending], compile_kernel, arch, jobs
    )
    n_built = len(artifact_map)

    rt = Runtime()
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

    values = ConvArgs.from_problem(p).to_launch_values(
        int(A_dev),
        int(B_dev),
        int(D_dev),
        A_t.nbytes,
        B_t.nbytes,
        D_t.nbytes,
    )

    results: List[NonGroupedResult] = []
    _stop = EarlyStop(args.early_stop, args.early_stop_after)
    n_run = 0
    for spec, kernel in pending:
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
                f"[skip] kernel load failed for {artifact.kernel_name}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

        grid = spec.grid()
        block = (spec.threads_per_block, 1, 1)
        stream = 0
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

        ms = _stop.measure(
            lambda: launcher(values, config=cfg),
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
            passed=kernel_passed,
        )
        if ms is None:
            _stop.report(artifact.kernel_name)
            continue
        synchronize_and_release(stream)

        label = (
            f"th{spec.tile_h} tw{spec.tile_w} tk{spec.tile_k} ck{spec.ck} "
            f"w{spec.waves_m}x{spec.waves_n} a{spec.atom} "
            f"wgm{spec.swizzle_wgm} iglp{spec.iglp} we{spec.waves_per_eu}"
        )
        n_run += 1
        results.append(
            NonGroupedResult(
                kernel_name=artifact.kernel_name,
                label=label,
                ms=ms,
                tflops=(flop / ms) * 1e-9,
                gbps=(bytes_xfer / ms) * 1e-6,
                passed=kernel_passed,
            )
        )
        print(
            f"[{n_run:4d}] {label}  {results[-1].tflops:6.1f} TFLOPS  {ms:.3f} ms",
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
    top_n = min(args.top, len(results))
    width = 110
    print(f"\n{'='*width}")
    print(f"Top {top_n} non-grouped configurations for {arch} {dtype} {p.short()}")
    print(f"{'='*width}")
    print(
        f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  "
        + (f"{'verify':>6}  " if args.verify else "")
        + "config"
    )
    print("-" * width)
    for rank, r in enumerate(results[:top_n], 1):
        v = f"{'PASS' if r.passed else 'FAIL':>6}  " if args.verify else ""
        print(
            f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}  {v}{r.label}"
        )
    print(f"\nBest: {results[0].tflops:.1f} TFLOPS — {results[0].kernel_name}")
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

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_grouped import (
        DirectConvWgradSpec,
        build_direct_conv_wgrad,
        direct_launch_geometry,
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

    # Direct conv is AOT: the whole shape travels as kernargs; D is fp32 dW.
    sig_wg = conv_direct_args_signature(dtype, direction="wgrad")
    _direct_args = ConvArgs.from_problem(p, direction="wgrad")

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

    _stop = EarlyStop.for_case(args, problem, dtype, "wgrad")
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

        # Delta register ring: each block owns one wo_tile (WO_BLOCK cols),
        # iterates H rows; S-strips are reused KH times via the register ring.
        grid, block_dim = direct_launch_geometry(spec)
        values = _direct_args.to_launch_values(
            int(dY_dev),
            int(X_dev),
            int(dW_dev),
            dY_t.nbytes,
            X_t.nbytes,
            dW_t.nbytes,
        )

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
        ms = _stop.measure(
            launcher.bind(values, config=cfg_wg),
            warmup=args.warmup,
            iters=args.iters,
            stream=0,
            passed=kernel_passed,
        )
        if ms is None:
            _stop.report(artifact.kernel_name)
            continue
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
        if _stop.summary():
            print(_stop.summary(), file=sys.stderr)
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
    import torch

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_grouped import (
        DirectConvDgradSpec,
        DirectReorganizeWeightsSpec,
        DirectTransposeWeightsDgradSpec,
        direct_launch_geometry,
        build_direct_conv_dgrad,
        is_valid_dgrad_spec,
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
    # Direct conv is AOT: the whole shape travels as kernargs.
    sig = conv_direct_args_signature(dtype, direction="dgrad")
    _direct_args = ConvArgs.from_problem(p, direction="dgrad")
    fprop_sig = conv_direct_args_signature(dtype)

    is_depthwise = p.cpg == 1

    if is_depthwise:
        # Depthwise dgrad: use ho-streaming kernel (better DRAM efficiency).
        from kernels.common.conv_direct_grouped import (
            DirectDepthwiseDgradStreamSpec,
            build_direct_depthwise_dgrad_streaming,
            is_valid_depthwise_dgrad_stream_spec,
        )

        combos_dw = list(itertools.product(_DW_BLOCK_W_DGRAD, _DW_BLOCK_WAVES))
        print(
            f"Sweeping {len(combos_dw)} depthwise dgrad combinations for {arch} {dtype} "
            f"{p.short()} (stride={p.stride}) ...",
            flush=True,
        )
        n_skipped = 0
        pending = []
        for block_w, block_waves in combos_dw:
            spec = DirectDepthwiseDgradStreamSpec(
                problem=p,
                name="rocke_bench_dw_dgrad",
                block_w=block_w,
                block_waves=block_waves,
            )
            ok, _ = is_valid_depthwise_dgrad_stream_spec(spec, arch=arch)
            if not ok:
                n_skipped += 1
                continue
            try:
                kernel = build_direct_depthwise_dgrad_streaming(spec, arch=arch)
            except ValueError:
                n_skipped += 1
                continue
            pending.append(((block_w, block_waves), spec, kernel))
    else:
        # Grouped dgrad: use the 2-kernel MFMA pipeline (transpose + fprop) for
        # stride=1, fall back to scalar FMA for stride > 1.
        # H-tiling (block_h) ensures enough blocks/CU even for groups=1.
        from kernels.common.conv_direct_grouped import (
            make_dgrad_fprop_spec,
            build_direct_transpose_weights_dgrad,
            DirectTransposeWeightsDgradSpec,
            build_direct_reorganize_weights,
            DirectReorganizeWeightsSpec,
            build_direct_mfma_dgrad,
            direct_dgrad_workspace_bytes,
            direct_dgrad_coalesced_workspace_bytes,
            is_valid_spec as is_valid_fprop_spec,
            build_direct_conv,
        )

        use_mfma = p.stride == 1

        if use_mfma:
            valid_bgs = [bg for bg in _BLOCK_GROUPS if p.groups % bg == 0]
            combos = list(itertools.product(_BLOCK_Q, valid_bgs))
            print(
                f"Sweeping {len(combos)} MFMA dgrad combinations for {arch} {dtype} {p.short()} "
                f"(cpg={p.cpg}, kpg={p.kpg}) ...",
                flush=True,
            )
            # Sweep (block_q, block_h, waves_q, waves_k, runtime_k_loop) combos.
            # runtime_k_loop=True: loads 1 K-atom at a time → ~55 VGPRs → 4 blks/CU.
            # waves_k>1 + preload: loads all K-atoms at once → ~200 VGPRs → 1 blk/CU.
            n_skipped = 0
            pending = []
            for block_q, block_groups in combos:
                for block_h in _DGRAD_BLOCK_H:
                    for waves_q, waves_k, use_rk, use_pg, use_k32 in _DGRAD_WAVES:
                        if block_q // waves_q < 16:
                            n_skipped += 1
                            continue
                        # fold_k32 requires kpg divisible by 32 AND N_K_ATOMS_32 divisible by waves_k.
                        # Without fold_k32 use ceil(kpg/16) atoms; still require divisibility by waves_k.
                        if use_k32:
                            if p.kpg % 32 != 0:
                                n_skipped += 1
                                continue
                            N_KA = p.kpg // 32
                        else:
                            N_KA = (p.kpg + 15) // 16
                        if N_KA == 0 or N_KA % waves_k != 0:
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
                        from dataclasses import replace as dc_replace

                        if use_k32:
                            fprop_spec = dc_replace(fprop_spec, fold_k32=True)
                        ok, _ = is_valid_fprop_spec(fprop_spec, arch=arch)
                        if not ok:
                            n_skipped += 1
                            continue
                        try:
                            kt1 = build_direct_transpose_weights_dgrad(
                                DirectTransposeWeightsDgradSpec(problem=p), arch=arch
                            )
                            kt2 = build_direct_reorganize_weights(
                                DirectReorganizeWeightsSpec(
                                    problem=p, fold_k32=use_k32
                                ),
                                arch=arch,
                            )
                            kf = build_direct_conv(fprop_spec, arch=arch)
                            kt = (kt1, kt2)  # two-step transpose pipeline
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
                                (kt, kf),
                            )
                        )
        else:
            valid_bgs = [bg for bg in _BLOCK_GROUPS if p.groups % bg == 0]
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

    # Compile all kernels (flatten tuples for MFMA pipeline).
    all_kernels = []
    for _, _, kernel_or_pair in pending:
        if isinstance(kernel_or_pair, tuple) and isinstance(kernel_or_pair[0], tuple):
            # (kt1, kt2), kf structure
            kt1, kt2 = kernel_or_pair[0]
            kf_k = kernel_or_pair[1]
            all_kernels.extend([kt1, kt2, kf_k])
        elif isinstance(kernel_or_pair, tuple):
            all_kernels.extend(kernel_or_pair)
        else:
            all_kernels.append(kernel_or_pair)
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

    # Workspaces for 3-kernel MFMA pipeline:
    # wt_dev:  W_T (step 1, simple transposed weights)
    # wt_coa:  W_coa (step 2, reorganized coalesced weights)
    # The fprop kernel reads from wt_coa.
    wt_dev = None
    wt_coa = None
    if not is_depthwise and use_mfma:
        from kernels.common.conv_direct_grouped import (
            direct_dgrad_workspace_bytes,
            direct_dgrad_coalesced_workspace_bytes,
        )

        # Use the LARGEST possible workspace (max over all fold_k32 combinations).
        wt_dev = rt.alloc(direct_dgrad_workspace_bytes(p))
        _wt_coa_max = max(
            direct_dgrad_coalesced_workspace_bytes(p, fold_k32=False),
            direct_dgrad_coalesced_workspace_bytes(p, fold_k32=True),
        )
        wt_coa = rt.alloc(_wt_coa_max)

    ref_out = None
    if args.verify or args.dump_fail:
        ref_out = _dgrad_reference_shared(dY_t, W_t, _DirectConvProblemAdapter(p))
        print(
            f"Reference dgrad computed via torch ({tuple(ref_out.shape)}, {ref_out.dtype}).",
            flush=True,
        )

    _stop = EarlyStop.for_case(args, problem, dtype, "dgrad")
    n_run = 0
    for combo, spec, kernel_or_pair in pending:
        is_mfma_pair = isinstance(kernel_or_pair, tuple)

        if is_depthwise:
            block_w, block_waves = combo
            grid, block_dim = direct_launch_geometry(spec)
            label = f"bw={block_w:3d} waves={block_waves}"
            kernel = kernel_or_pair
        elif is_mfma_pair:
            # 3-kernel MFMA pipeline: transpose → reorganize → fprop.
            (kt1, kt2), kf = kernel_or_pair
            (
                block_q,
                block_groups,
                block_h,
                waves_q,
                waves_k,
                use_rk,
                use_pg,
                use_k32,
            ) = combo
            # Step 1: simple transpose W_orig → W_T (scalar per thread)
            t1_grid, _ = direct_launch_geometry(
                DirectTransposeWeightsDgradSpec(problem=p)
            )
            # Step 2: reorganize W_T → W_coa (vec4 per thread, coalesced stores)
            t2_grid, _ = direct_launch_geometry(
                DirectReorganizeWeightsSpec(problem=p, fold_k32=use_k32)
            )
            # Step 3: fprop with coalesced preloads (persistent grids included)
            f_grid, block_dim = direct_launch_geometry(spec)
            rk_tag = "+rk" if use_rk else ""
            pg_tag = "+pg" if use_pg else ""
            k32_tag = "+k32" if use_k32 else ""
            label = f"bq={block_q} bh={block_h} wq={waves_q} wk={waves_k}{rk_tag}{pg_tag}{k32_tag} MFMA"
            kernel = kf
            kt = (kt1, kt2)  # for artifact lookup below
        else:
            block_q, block_groups = combo
            grid, block_dim = direct_launch_geometry(spec)
            label = f"bq={block_q:3d} bg={block_groups:3d} scFMA"
            kernel = kernel_or_pair

        artifact = artifact_map.get(kernel.name)
        if artifact is None:
            n_skipped += 1
            continue

        # The MFMA pipeline's last step is a *forward* kernel run on the
        # transposed problem (dY -> dX), so it takes the forward ABI with that
        # problem's extents -- not the dgrad ABI of the original problem.
        # The two only coincide numerically when cpg == kpg.
        fprop_args = ConvArgs.from_problem(spec.problem) if is_mfma_pair else None
        try:
            launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=fprop_sig if is_mfma_pair else sig,
            )
        except HipError as e:
            n_skipped += 1
            print(f"[skip] {artifact.kernel_name}: {e}", file=sys.stderr, flush=True)
            continue

        # For 3-kernel MFMA pipeline: load transpose1 and reorganize launchers.
        t1_launcher = t2_launcher = None
        if is_mfma_pair:
            _wt_sig = [
                {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
                {"name": "D", "type": "ptr<f16, global>", "size_bytes": 8},
                {"name": "A_bytes", "type": "i32", "size_bytes": 4},
                {"name": "D_bytes", "type": "i32", "size_bytes": 4},
            ]
            kt1_art = artifact_map.get(kt[0].name)
            kt2_art = artifact_map.get(kt[1].name)
            if kt1_art is None or kt2_art is None:
                n_skipped += 1
                continue
            try:
                t1_launcher = KernelLauncher(
                    hsaco=kt1_art.hsaco,
                    kernel_name=kt1_art.kernel_name,
                    signature=_wt_sig,
                )
                t2_launcher = KernelLauncher(
                    hsaco=kt2_art.hsaco,
                    kernel_name=kt2_art.kernel_name,
                    signature=_wt_sig,
                )
            except HipError as e:
                n_skipped += 1
                print(f"[skip] transpose {e}", file=sys.stderr, flush=True)
                continue

        if is_mfma_pair:
            wt1_nbytes = direct_dgrad_workspace_bytes(p)
            wt2_nbytes = direct_dgrad_coalesced_workspace_bytes(p, fold_k32=use_k32)
            t1_values = {
                "A": W_dev,
                "D": wt_dev,
                "A_bytes": W_t.nbytes,
                "D_bytes": wt1_nbytes,
            }
            t2_values = {
                "A": wt_dev,
                "D": wt_coa,
                "A_bytes": wt1_nbytes,
                "D_bytes": wt2_nbytes,
            }
            # wk>1 OR runtime_k_loop=True: coalesced preload reads W_coa.
            # wk=1 and not runtime_k_loop: runtime loops expect W_T (wt_dev).
            if waves_k > 1 or use_rk:
                f_values = fprop_args.to_launch_values(
                    int(dY_dev),
                    int(wt_coa),
                    int(dX_dev),
                    dY_t.nbytes,
                    wt2_nbytes,
                    dX_t.nbytes,
                )

                def run_mfma_dgrad():
                    t1_launcher(
                        t1_values, config=LaunchConfig(grid=t1_grid, block=(64, 1, 1))
                    )
                    t2_launcher(
                        t2_values, config=LaunchConfig(grid=t2_grid, block=(64, 1, 1))
                    )
                    launcher(
                        f_values, config=LaunchConfig(grid=f_grid, block=block_dim)
                    )

            else:
                f_values = fprop_args.to_launch_values(
                    int(dY_dev),
                    int(wt_dev),
                    int(dX_dev),
                    dY_t.nbytes,
                    wt1_nbytes,
                    dX_t.nbytes,
                )

                def run_mfma_dgrad():
                    t1_launcher(
                        t1_values, config=LaunchConfig(grid=t1_grid, block=(64, 1, 1))
                    )
                    launcher(
                        f_values, config=LaunchConfig(grid=f_grid, block=block_dim)
                    )

            values = None
        else:
            values = _direct_args.to_launch_values(
                int(dY_dev),
                int(W_dev),
                int(dX_dev),
                dY_t.nbytes,
                W_t.nbytes,
                dX_t.nbytes,
            )

        kernel_passed = None
        if args.verify or args.dump_fail:
            if is_mfma_pair:
                # Verify: run both kernels then compare dX to reference.
                import torch

                rt.memset(dX_dev, 0, dX_t.nbytes)
                run_mfma_dgrad()
                synchronize_and_release(0)
                dX_cpu = torch.empty_like(dX_t)
                rt.memcpy_d2h(u8(dX_cpu), dX_dev, dX_t.nbytes)
                if ref_out is not None:
                    out_f32 = dX_cpu.float()
                    ref_f32 = ref_out.float().cpu()
                    abs_diff = (out_f32 - ref_f32).abs()
                    rel_err = float(abs_diff.max() / ref_f32.abs().max().clamp(min=1.0))
                    tol = 5e-2
                    kernel_passed = rel_err < tol
                    status = "PASS" if kernel_passed else f"FAIL(rel_err={rel_err:.2e})"
                    print(f"  verify {artifact.kernel_name}: {status}", flush=True)
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
                    if wt_dev:
                        rt.free(wt_dev)
                    return 1, []
                rt.memset(dX_dev, 0, dX_t.nbytes)

        if is_mfma_pair:
            ms = _stop.measure(
                run_mfma_dgrad,
                warmup=args.warmup,
                iters=args.iters,
                stream=0,
                passed=kernel_passed,
            )
        else:
            cfg = LaunchConfig(grid=grid, block=block_dim)
            ms = _stop.measure(
                launcher.bind(values, config=cfg),
                warmup=args.warmup,
                iters=args.iters,
                stream=0,
                passed=kernel_passed,
            )
        if ms is None:
            _stop.report(label)
            continue
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
            f"[{n_run:4d}] {label}  {tflops:6.1f} TFLOPS  {ms:.3f} ms{passed_str}",
            flush=True,
        )

    rt.free(dY_dev)
    rt.free(W_dev)
    rt.free(dX_dev)
    if wt_dev:
        rt.free(wt_dev)
    if wt_coa:
        rt.free(wt_coa)
    print(f"\nDgrad sweep done: {n_built} compiled, {n_skipped} skipped.", flush=True)

    if not results:
        print("No valid dgrad configurations found.", file=sys.stderr)
        if _stop.summary():
            print(_stop.summary(), file=sys.stderr)
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
# Cache modes: --compile-all and --run-from-cache
# ---------------------------------------------------------------------------


def _cache_directions(args) -> "tuple[str, ...] | None":
    if not args.directions:
        return None
    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())
    for d in directions:
        if d not in ("fwd", "dgrad"):
            raise ValueError(f"direct conv has no {d!r} kernel; use fwd and/or dgrad")
    return directions


def _cache_dispatch(args, arch: str, cases: list) -> int:
    """Handle the two cache modes.

    ``--compile-all`` builds every kernel in the capability list of
    :mod:`benchmarks.common.direct_kernel_sweep` and never touches a GPU; the
    binaries take N, H, W and groups as kernargs, so the same cache serves
    every later shape with matching filter/stride/padding/channels.
    ``--run-from-cache`` runs the parsed cases on the cached kernels that fit.
    """
    from pathlib import Path

    from benchmarks.common.direct_kernel_sweep import compile_all_direct
    from benchmarks.common.kernel_cache import KernelCache
    from benchmarks.common.kernel_sweep import describe_cache
    from rocke.core.arch import ArchTarget

    try:
        directions = _cache_directions(args)
    except ValueError as e:
        print(f"error: --directions: {e}", file=sys.stderr)
        return 2

    if args.compile_all:
        directions = directions or args.compile_directions
        if "wgrad" in directions:
            print(
                "error: --compile-all: direct conv caches fwd and dgrad kernels "
                "only (no wgrad)",
                file=sys.stderr,
            )
            return 2
        cache_dir = Path(args.cache_dir) if args.cache_dir else Path("./kernel_cache")
        rc = 0
        for dtype in args.compile_dtypes:
            rc = max(
                rc,
                compile_all_direct(
                    cache=KernelCache(cache_dir, arch),
                    arch=arch,
                    target=ArchTarget.from_gfx(arch),
                    directions=directions,
                    jobs=os.cpu_count() if args.jobs == 0 else max(1, args.jobs),
                    limit=args.limit,
                    dtype=dtype,
                ),
            )
        return rc

    cache = KernelCache(Path(args.run_from_cache), arch)
    # Only the direct kernels: the implicit-GEMM entries sharing the cache
    # are far more numerous and this run never launches them.
    rc = describe_cache(
        cache, directions=("direct_fwd", "direct_dgrad", "direct_dgrad_helper")
    )
    if rc:
        return rc
    return _run_from_cache(args, arch, cases, cache, directions)


def _run_from_cache(args, arch: str, cases: list, cache, directions) -> int:
    """Benchmark every cached kernel that can run each case. Nothing is compiled.

    Each case runs in its own direction (``--direction`` or the MIOpen ``-F``
    flag) unless ``--directions`` names the directions explicitly.
    """
    import ctypes

    import torch

    from benchmarks.common.direct_kernel_sweep import direct_plans, launch_values
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    from benchmarks.common.direct_kernel_sweep import DIRECT_DTYPES

    overall_rc = 0
    for case_idx, (p, dtype, case_direction) in enumerate(cases, 1):
        if dtype not in DIRECT_DTYPES:
            print(
                f"Case {case_idx} {p.short()}: direct conv builds "
                f"{'/'.join(DIRECT_DTYPES)} (got {dtype})"
            )
            continue
        torch_dt = torch.bfloat16 if dtype == "bf16" else torch.float16
        for direction in directions or (case_direction,):
            plans, rejected = direct_plans(cache, p, direction, arch)
            if not plans:
                why = (
                    rejected[0][1]
                    if rejected
                    else f"no cached {dtype} kernel has these capabilities; "
                    f"build them with --compile-all --dtype {dtype}"
                )
                print(
                    f"Case {case_idx} {p.short()} {direction}: no runnable kernel "
                    f"(KH={p.KH} PAD={p.PAD} stride={p.stride} cpg={p.cpg} kpg={p.kpg}: {why})",
                    flush=True,
                )
                continue
            print(
                f"\nCase {case_idx}: {p.short()} {direction} -- {len(plans)} cached kernels"
                + (f", {len(rejected)} rejected for this shape" if rejected else ""),
                flush=True,
            )

            torch.manual_seed(42)
            x_shape = (p.N, p.H, p.W, p.total_c)
            y_shape = (p.N, p.Ho, p.Wo, p.total_k)
            W_t = torch.empty(p.total_k, p.KH, p.KW, p.cpg, dtype=torch_dt).uniform_(
                -1, 1
            )
            if direction == "fwd":
                in_name, out_name = "x", "y"
                In_t = torch.empty(*x_shape, dtype=torch_dt).uniform_(-1, 1)
                Out_t = torch.empty(*y_shape, dtype=torch_dt)
            else:
                in_name, out_name = "dy", "dx"
                In_t = torch.empty(*y_shape, dtype=torch_dt).uniform_(-1, 1)
                Out_t = torch.empty(*x_shape, dtype=torch_dt)

            ref_out = None
            if args.verify:
                ref_fn = (
                    _conv_reference if direction == "fwd" else _dgrad_reference_shared
                )
                ref_out = ref_fn(In_t, W_t, _DirectConvProblemAdapter(p))

            rt = Runtime()
            ptrs = {
                in_name: rt.alloc(In_t.nbytes),
                "w": rt.alloc(W_t.nbytes),
                out_name: rt.alloc(Out_t.nbytes),
            }
            sizes = {in_name: In_t.nbytes, "w": W_t.nbytes, out_name: Out_t.nbytes}
            rt.memcpy_h2d(ptrs[in_name], _u8(In_t), In_t.nbytes)
            rt.memcpy_h2d(ptrs["w"], _u8(W_t), W_t.nbytes)
            # Scratch buffers are sized for the largest plan and shared.
            for plan in plans:
                for name, nbytes in plan.workspaces.items():
                    sizes[name] = max(sizes.get(name, 0), nbytes)
            for name in set(sizes) - set(ptrs):
                ptrs[name] = rt.alloc(sizes[name])

            flop = float(p.flops)
            bytes_xfer = float(In_t.nbytes + W_t.nbytes + Out_t.nbytes)
            results = []
            _stop = EarlyStop.for_case(args, p, dtype, direction)
            for plan in plans:
                try:
                    launches = [
                        (
                            KernelLauncher(
                                hsaco=st.hsaco_path.read_bytes(),
                                kernel_name=st.kernel_name,
                                signature=st.signature,
                            ),
                            launch_values(st, ptrs, sizes),
                            LaunchConfig(grid=st.grid, block=st.block, stream=0),
                        )
                        for st in plan.steps
                    ]
                except HipError as e:
                    print(f"  [skip] {plan.label}: {e}", flush=True)
                    continue

                # Each step packed once: a timed loop that packs per launch
                # times the host, not a short kernel.
                bound = [
                    launcher.bind(values, config=cfg)
                    for launcher, values, cfg in launches
                ]

                def run(bound=bound):
                    for launch in bound:
                        launch()

                passed = None
                if ref_out is not None:
                    rt.memset(ptrs[out_name], 0, Out_t.nbytes)
                    run()
                    synchronize_and_release(0)
                    out_cpu = torch.empty_like(Out_t)
                    rt.memcpy_d2h(_u8(out_cpu), ptrs[out_name], Out_t.nbytes)
                    ref_f32 = ref_out.float().cpu()
                    rel_err = float(
                        (out_cpu.float() - ref_f32).abs().max()
                        / ref_f32.abs().max().clamp(min=1.0)
                    )
                    passed = rel_err < 5e-2
                    if not passed:
                        overall_rc = 1
                        print(
                            f"  verify {plan.label}: FAIL(rel_err={rel_err:.2e})",
                            flush=True,
                        )

                try:
                    ms = _stop.measure(
                        run,
                        warmup=args.warmup,
                        iters=args.iters,
                        stream=0,
                        passed=passed,
                    )
                except (HipError, RuntimeError) as e:
                    print(f"  [skip] {plan.label}: {e}", flush=True)
                    continue
                if ms is None:
                    _stop.report(plan.label)
                    continue
                synchronize_and_release(0)
                results.append(
                    (flop / ms / 1e9, bytes_xfer / ms / 1e6, ms, passed, plan.label)
                )

            for ptr in ptrs.values():
                rt.free(ptr)
            if not results:
                print("  no cached kernel launched successfully", flush=True)
                if _stop.summary():
                    print(f"  {_stop.summary()}", flush=True)
                overall_rc = overall_rc or 1
                continue
            results.sort(key=lambda r: r[0], reverse=True)
            print(
                f"  {'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  {'verify':>6}  kernel"
            )
            for rank, (tflops, gbps, ms, passed, label) in enumerate(
                results[: args.top], 1
            ):
                verdict = "-" if passed is None else ("PASS" if passed else "FAIL")
                print(
                    f"  {rank:>4}  {tflops:>7.1f}  {ms:>8.3f}  {gbps:>7.1f}  {verdict:>6}  {label}"
                )
            # The line benchmark_conv_compare.py reads, as the JIT sweep prints.
            tflops, _, ms, _, label = results[0]
            print(f"\nBest: {tflops:.1f} TFLOPS — {label}  {ms:.3f} ms", flush=True)
    return overall_rc


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
        default=None,
        choices=["fwd", "dgrad", "wgrad"],
        help="convolution direction to benchmark: fwd (default), dgrad, wgrad. "
        "With --compile-all: build only this direction (default: fwd and dgrad).",
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
    add_early_stop_arg(parser)
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
        default=None,
        help="I/O data type when using shape flags (default: fp16; cases from "
        "--miopen-cmd/--miopen-file carry their own dtype). --compile-all builds "
        "the cache for every data type unless --dtype names one.",
    )

    cache_grp = parser.add_argument_group(
        "Kernel cache",
        "Direct conv kernels take N, H, W and groups as kernel arguments and bake "
        "the filter size, stride, padding and per-group channel counts. A cache "
        "holds one binary per supported combination of those (see "
        "benchmarks/common/direct_kernel_sweep.py) and serves any shape that "
        "matches one.",
    )
    cache_grp.add_argument(
        "--compile-all",
        action="store_true",
        dest="compile_all",
        help="compile every supported direct kernel for --arch across --directions "
        "and save the HSACOs to --cache-dir. No problem is needed and no GPU is "
        "used. Parallelised with --jobs.",
    )
    cache_grp.add_argument(
        "--run-from-cache",
        default=None,
        metavar="DIR",
        dest="run_from_cache",
        help="load pre-compiled HSACOs from DIR, keep the ones that can run the "
        "requested shape, and benchmark those. Nothing is compiled.",
    )
    cache_grp.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        dest="cache_dir",
        help="directory for the HSACO cache written by --compile-all "
        "(default: ./kernel_cache).",
    )
    cache_grp.add_argument(
        "--directions",
        default=None,
        dest="directions",
        help="comma-separated directions (fwd, dgrad) to build or run. "
        "--compile-all defaults to --direction if given, else both; "
        "--run-from-cache defaults to each case's own direction.",
    )
    cache_grp.add_argument(
        "--limit",
        type=int,
        default=None,
        dest="limit",
        help="--compile-all only: stop after building this many kernels (smoke tests).",
    )

    args = parser.parse_args()
    # --dtype unset: one fp16 run for shape flags, every dtype for --compile-all.
    from benchmarks.common.direct_kernel_sweep import DIRECT_DTYPES

    args.compile_dtypes = (args.dtype,) if args.dtype is not None else DIRECT_DTYPES
    if args.dtype is None:
        args.dtype = "fp16"
    # --direction likewise: fwd for shape flags, both cached directions for
    # --compile-all unless one is named.
    args.compile_directions = (
        (args.direction,) if args.direction is not None else ("fwd", "dgrad")
    )
    if args.direction is None:
        args.direction = "fwd"

    arch = args.arch

    # Compiling the cache needs no problem, so it runs before any shape flag
    # is looked at.
    if args.compile_all:
        return _cache_dispatch(args, arch, [])

    from kernels.common.conv_direct_grouped import DirectConvProblem

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

        # The grouped/depthwise fwd kernels require cpg == kpg; the non-grouped
        # (groups == 1) fwd kernel takes any C/K, and dgrad need not hold it.
        if args.direction == "fwd" and args.groups != 1 and cpg != kpg:
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

        if (args.dH != 1 or args.dW != 1) and (
            args.direction != "fwd" or args.groups != 1
        ):
            print(
                f"error: direct conv computes a dilated convolution only forward "
                f"and ungrouped (got dH={args.dH}, dW={args.dW})",
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
            dil_h=args.dH,
            dil_w=args.dW,
        )
        cases = [(problem, args.dtype, args.direction)]

    if args.run_from_cache:
        return _cache_dispatch(args, arch, cases)

    import ctypes

    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

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
        elif problem.groups == 1:
            rc, _ = _run_nongrouped_sweep(problem=problem, dtype=dtype, **_common)
        else:
            rc, _ = _run_sweep(problem=problem, dtype=dtype, **_common)
        all_rc = all_rc or rc

    return all_rc


if __name__ == "__main__":
    raise SystemExit(main())
