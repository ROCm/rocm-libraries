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
import os
import sys
from dataclasses import dataclass
from typing import List

os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

# ---------------------------------------------------------------------------
# Swept parameter grids
# ---------------------------------------------------------------------------

# Grouped (cpg >= 4) sweep dimensions.
_BLOCK_Q = (16, 32)
_BLOCK_GROUPS = (1, 2, 4, 8, 16)
_DOUBLE_BUFFER = (True, False)

# Depthwise (cpg == 1) sweep dimensions.
_DW_BLOCK_W = (4, 8, 16, 32)
_DW_BLOCK_WAVES = (1, 2, 4)


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


def _conv_reference_grouped(A_t, B_t, p) -> "torch.Tensor":
    """Grouped conv reference via torch.nn.functional.conv2d."""
    import torch
    import torch.nn.functional as F

    A_nchw = A_t.permute(0, 3, 1, 2).float()
    B_kcrs = B_t.permute(0, 3, 1, 2).float()
    out_nchw = F.conv2d(A_nchw, B_kcrs, padding=p.PAD, stride=p.stride, groups=p.groups)
    return out_nchw.permute(0, 2, 3, 1).contiguous().cuda()


def _print_results(
    results: List[Result], top_n_arg: int, arch: str, p, show_verify: bool
):
    top_n = min(top_n_arg, len(results))
    width = 100 if show_verify else 88
    print(f"\n{'='*width}")
    print(f"Top {top_n} configurations for {arch} fp16 {p.short()}")
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
    results: "List[DepthwiseResult]", top_n_arg: int, arch: str, p, show_verify: bool
):
    top_n = min(top_n_arg, len(results))
    width = 96 if show_verify else 84
    print(f"\n{'='*width}")
    print(f"Top {top_n} depthwise configurations for {arch} fp16 {p.short()}")
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
    from rocke.instances.common.conv_direct_grouped import (
        DirectDepthwiseSpec,
        build_direct_depthwise,
        is_valid_depthwise_spec,
    )
    from rocke.runtime.hip_module import HipError

    p = problem

    torch.manual_seed(42)
    A_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=torch.float16).uniform_(-1.0, 1.0)
    B_t = torch.empty(p.total_k, p.KH, p.KW, 1, dtype=torch.float16).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.H, p.W, p.total_k, dtype=torch.float16)

    bytes_xfer = float(A_t.nbytes + B_t.nbytes + D_t.nbytes)
    flop = float(p.flops)
    sig = conv_args_signature("fp16")

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
        f"Sweeping {len(combos)} depthwise combinations for {arch} fp16 {p.short()} ...",
        flush=True,
    )

    n_skipped = 0
    pending = []
    for combo in combos:
        block_w, block_waves = combo
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
        ref_out = _conv_reference_grouped(A_t, B_t, p)
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

        q_tiles = math.ceil(p.W / block_w)
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
    _print_depthwise_results(results, args.top, arch, p, args.verify)
    return 0, results


def _run_sweep(
    *,
    args,
    problem,
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
    from rocke.instances.common.conv_direct_grouped import (
        DirectConvSpec,
        build_direct_conv,
        is_valid_spec,
    )
    from rocke.runtime.hip_module import HipError

    p = problem

    torch.manual_seed(42)
    A_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=torch.float16).uniform_(-1.0, 1.0)
    B_t = torch.empty(p.total_k, p.KH, p.KW, p.cpg, dtype=torch.float16).uniform_(
        -1.0, 1.0
    )
    D_t = torch.empty(p.N, p.H, p.W, p.total_k, dtype=torch.float16)

    bytes_xfer = float(A_t.nbytes + B_t.nbytes + D_t.nbytes)
    flop = float(p.flops)
    sig = conv_args_signature("fp16")

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
        f"Sweeping {len(combos)} combinations for {arch} fp16 {p.short()} "
        f"(cpg={p.cpg}) ...",
        flush=True,
    )

    n_skipped = 0
    pending = []
    for combo in combos:
        block_q, block_groups, double_buffer = combo
        if p.groups % block_groups != 0:
            n_skipped += 1
            continue
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
        ref_out = _conv_reference_grouped(A_t, B_t, p)
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

        q_tiles = (p.W + block_q - 1) // block_q
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
    _print_results(results, args.top, arch, p, args.verify)
    return 0, results


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
        "--dump-fail",
        default=None,
        metavar="PATH",
        dest="dump_fail",
        help="on the first verify FAIL, dump tensors to PATH/ and stop the sweep.",
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

    args = parser.parse_args()

    import ctypes

    from rocke import compile_kernel
    from rocke.instances.common.conv_direct_grouped import DirectConvProblem
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    arch = args.arch

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

    if cpg != kpg:
        print(
            f"error: cpg={cpg} != kpg={kpg}; direct grouped conv requires C/groups == K/groups",
            file=sys.stderr,
        )
        return 2

    # cpg == 1: depthwise.  cpg >= 4 and multiple of 4: grouped.
    if cpg != 1 and (cpg % 4 != 0 or cpg < 4):
        print(
            f"error: cpg={cpg} (C/groups={args.C}/{args.groups}) must be 1 (depthwise) "
            f"or a positive multiple of 4 (grouped)",
            file=sys.stderr,
        )
        return 2

    if cpg == 1 and args.sH != 1:
        print(
            f"error: depthwise kernel requires stride=1 (got sH={args.sH})",
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
    )

    if cpg == 1:
        rc, _ = _run_depthwise_sweep(problem=problem, **_common)
    else:
        rc, _ = _run_sweep(problem=problem, **_common)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
