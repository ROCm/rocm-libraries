# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tile sweep benchmark for direct grouped convolution (4c, 8c, 16c, 32c variants).

Covers all four channel-count variants introduced by the direct grouped conv
implementation (matching the CK Tile PR #8347 kernel family):

  4c  (cpg=kpg=4)  — mfma_f32_4x4x4_f16,    gfx942 + gfx950
  8c  (cpg=kpg=8)  — mfma_f32_16x16x16_f16,  gfx942 + gfx950
  16c (cpg=kpg=16) — mfma_f32_16x16x16_f16,  gfx942 + gfx950
  32c (cpg=kpg=32) — mfma_f32_32x32x8_f16,   gfx950 only

Swept dimensions per variant:

  4c:
    block_q      : 4, 8, 16
    block_groups : 16, 32, 64

  8c:
    block_q      : 16, 32
    block_groups : 4, 8, 16
    double_buffer: True, False

  16c:
    block_q      : 16, 32
    block_groups : 4, 8, 16
    double_buffer: True, False
    fold_k32     : True, False

  32c:
    block_q      : 32, 64
    block_groups : 2, 4, 8
    double_buffer: True, False

Run examples:
  python benchmark_direct_conv.py --variant 16c --N 8 --H 56 --W 56 --groups 64
  python benchmark_direct_conv.py --variant 4c  --N 8 --H 56 --W 56 --groups 256
  python benchmark_direct_conv.py --variant 8c  --N 8 --H 56 --W 56 --groups 128
  python benchmark_direct_conv.py --variant 32c --N 8 --H 56 --W 56 --groups 32
  python benchmark_direct_conv.py --variant all --N 8 --H 56 --W 56 --groups 64
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

_BLOCK_Q_4C = (4, 8, 16)
_BLOCK_GROUPS_4C = (16, 32, 64)

_BLOCK_Q_8C = (16, 32)
_BLOCK_GROUPS_8C = (4, 8, 16)
_DOUBLE_BUFFER_8C = (True, False)

_BLOCK_Q_16C = (16, 32)
_BLOCK_GROUPS_16C = (4, 8, 16)
_DOUBLE_BUFFER_16C = (True, False)
_FOLD_K32_16C = (True, False)

_BLOCK_Q_32C = (32, 64)
_BLOCK_GROUPS_32C = (2, 4, 8)
_DOUBLE_BUFFER_32C = (True, False)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class Result:
    kernel_name: str
    variant: str
    block_q: int
    block_groups: int
    double_buffer: "bool | None"
    fold_k32: "bool | None"
    ms: float
    tflops: float
    gbps: float
    passed: "bool | None" = None


# ---------------------------------------------------------------------------
# Shared helpers
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

    # A: (N, H, W, C) → (N, C, H, W)
    # B: (K_total, KH, KW, cpg) → (K_total, cpg, KH, KW)
    A_nchw = A_t.permute(0, 3, 1, 2).float()
    B_kcrs = B_t.permute(0, 3, 1, 2).float()
    out_nchw = F.conv2d(A_nchw, B_kcrs, padding=p.PAD, stride=p.stride, groups=p.groups)
    return out_nchw.permute(0, 2, 3, 1).contiguous().cuda()


def _make_tensors(p, dtype=None):
    import torch

    _dtype = dtype or torch.float16
    torch.manual_seed(42)
    A_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=_dtype).uniform_(-1.0, 1.0)
    B_t = torch.empty(p.total_k, p.KH, p.KW, p.cpg, dtype=_dtype).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.H, p.W, p.total_k, dtype=_dtype)
    return A_t, B_t, D_t


def _run_generic_sweep(
    *,
    args,
    problem,
    arch: str,
    variant: str,
    combos: list,
    combo_label_fn,
    spec_fn,
    build_fn,
    valid_fn,
    grid_fn,
    compile_kernel,
    jobs: int,
    synchronize_and_release,
    time_launches,
    Runtime,
    KernelLauncher,
    LaunchConfig,
    u8,
) -> "tuple[int, List[Result]]":
    """Generic sweep driver shared by all four variants."""
    import torch

    from rocke.helpers.manifest import conv_args_signature
    from rocke.runtime.hip_module import HipError

    _u8 = u8
    p = problem

    A_t, B_t, D_t = _make_tensors(p)
    bytes_xfer = float(A_t.nbytes + B_t.nbytes + D_t.nbytes)
    flop = float(p.flops)
    sig = conv_args_signature("fp16")

    if args.sample is not None:
        total = len(combos)
        combos = _sample_combos(combos, args.sample, args.seed)
        print(
            f"Sampling {len(combos)}/{total} {variant} combinations "
            f"({args.sample*100:.0f}%, seed={args.seed}).",
            flush=True,
        )

    print(
        f"Sweeping {len(combos)} {variant} combinations for {arch} fp16 {p.short()} ...",
        flush=True,
    )

    n_skipped = 0
    pending = []
    for combo in combos:
        spec = spec_fn(combo)
        ok, reason = valid_fn(spec, arch=arch)
        if not ok:
            n_skipped += 1
            continue
        try:
            kernel = build_fn(spec, arch=arch)
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
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    ref_out = None
    if args.verify or args.dump_fail:
        ref_out = _conv_reference_grouped(A_t, B_t, p)
        print(
            f"{variant} reference computed via torch "
            f"({tuple(ref_out.shape)}, {ref_out.dtype}).",
            flush=True,
        )

    n_run = 0
    for combo, spec, kernel in pending:
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

        grid = grid_fn(spec, p)
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
                u8=_u8,
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

        label = combo_label_fn(combo)
        results.append(
            Result(
                kernel_name=artifact.kernel_name,
                variant=variant,
                **label,
                ms=ms,
                tflops=cur_tflops,
                gbps=cur_gbps,
                passed=kernel_passed,
            )
        )
        extra = (
            f"db={label['double_buffer']} " if label["double_buffer"] is not None else ""
        )
        fk = (
            f"k32={label['fold_k32']} " if label["fold_k32"] is not None else ""
        )
        print(
            f"[{n_run:4d}] bq={label['block_q']:3d} bg={label['block_groups']:3d} "
            f"{extra}{fk}{cur_tflops:6.1f} TFLOPS  {ms:.3f} ms",
            flush=True,
        )

    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)

    print(f"\n{variant} sweep done: {n_built} compiled, {n_skipped} skipped.", flush=True)

    if not results:
        print(f"No valid {variant} configurations found.", file=sys.stderr)
        return 1, []

    results.sort(key=lambda r: r.tflops, reverse=True)
    _print_results(results, args.top, arch, "fp16", p, variant, args.verify)
    return 0, results


# ---------------------------------------------------------------------------
# Per-variant wrappers
# ---------------------------------------------------------------------------


def _run_4c_sweep(*, args, problem, arch, compile_kernel, jobs, **kw) -> "tuple[int, List[Result]]":
    from rocke.instances.common.conv_direct_grouped import (
        DirectConv4cSpec,
        build_direct_conv_4c,
        is_valid_spec_4c,
    )

    p = problem
    combos = list(itertools.product(_BLOCK_Q_4C, _BLOCK_GROUPS_4C))

    def spec_fn(combo):
        block_q, block_groups = combo
        return DirectConv4cSpec(
            problem=p, name="rocke_bench_direct_conv_4c",
            block_q=block_q, block_groups=block_groups,
        )

    def label_fn(combo):
        block_q, block_groups = combo
        return dict(block_q=block_q, block_groups=block_groups,
                    double_buffer=None, fold_k32=None)

    def grid_fn(spec, p):
        q_tiles = (p.W + spec.block_q - 1) // spec.block_q
        g_tiles = p.groups // spec.block_groups
        return (q_tiles, g_tiles, p.N)

    return _run_generic_sweep(
        args=args, problem=problem, arch=arch, variant="4c", combos=combos,
        combo_label_fn=label_fn, spec_fn=spec_fn,
        build_fn=build_direct_conv_4c, valid_fn=is_valid_spec_4c, grid_fn=grid_fn,
        compile_kernel=compile_kernel, jobs=jobs, **kw,
    )


def _run_8c_sweep(*, args, problem, arch, compile_kernel, jobs, **kw) -> "tuple[int, List[Result]]":
    from rocke.instances.common.conv_direct_grouped import (
        DirectConv8cSpec,
        build_direct_conv_8c,
        is_valid_spec_8c,
    )

    p = problem
    combos = list(itertools.product(_BLOCK_Q_8C, _BLOCK_GROUPS_8C, _DOUBLE_BUFFER_8C))

    def spec_fn(combo):
        block_q, block_groups, double_buffer = combo
        return DirectConv8cSpec(
            problem=p, name="rocke_bench_direct_conv_8c",
            block_q=block_q, block_groups=block_groups, double_buffer=double_buffer,
        )

    def label_fn(combo):
        block_q, block_groups, double_buffer = combo
        return dict(block_q=block_q, block_groups=block_groups,
                    double_buffer=double_buffer, fold_k32=None)

    def grid_fn(spec, p):
        q_tiles = (p.W + spec.block_q - 1) // spec.block_q
        g_tiles = p.groups // spec.block_groups
        return (q_tiles, g_tiles, p.N)

    return _run_generic_sweep(
        args=args, problem=problem, arch=arch, variant="8c", combos=combos,
        combo_label_fn=label_fn, spec_fn=spec_fn,
        build_fn=build_direct_conv_8c, valid_fn=is_valid_spec_8c, grid_fn=grid_fn,
        compile_kernel=compile_kernel, jobs=jobs, **kw,
    )


def _run_16c_sweep(*, args, problem, arch, compile_kernel, jobs, **kw) -> "tuple[int, List[Result]]":
    from rocke.instances.common.conv_direct_grouped import (
        DirectConv16cSpec,
        build_direct_conv_16c,
        is_valid_spec_16c,
    )

    p = problem
    combos = list(
        itertools.product(_BLOCK_Q_16C, _BLOCK_GROUPS_16C, _DOUBLE_BUFFER_16C, _FOLD_K32_16C)
    )

    def spec_fn(combo):
        block_q, block_groups, double_buffer, fold_k32 = combo
        return DirectConv16cSpec(
            problem=p, name="rocke_bench_direct_conv_16c",
            block_q=block_q, block_groups=block_groups,
            double_buffer=double_buffer, fold_k32=fold_k32,
        )

    def label_fn(combo):
        block_q, block_groups, double_buffer, fold_k32 = combo
        return dict(block_q=block_q, block_groups=block_groups,
                    double_buffer=double_buffer, fold_k32=fold_k32)

    def grid_fn(spec, p):
        q_tiles = (p.W + spec.block_q - 1) // spec.block_q
        g_tiles = p.groups // spec.block_groups
        return (q_tiles, g_tiles, p.N)

    return _run_generic_sweep(
        args=args, problem=problem, arch=arch, variant="16c", combos=combos,
        combo_label_fn=label_fn, spec_fn=spec_fn,
        build_fn=build_direct_conv_16c, valid_fn=is_valid_spec_16c, grid_fn=grid_fn,
        compile_kernel=compile_kernel, jobs=jobs, **kw,
    )


def _run_32c_sweep(*, args, problem, arch, compile_kernel, jobs, **kw) -> "tuple[int, List[Result]]":
    from rocke.instances.common.conv_direct_grouped import (
        DirectConv32cSpec,
        build_direct_conv_32c,
        is_valid_spec_32c,
    )

    p = problem
    combos = list(itertools.product(_BLOCK_Q_32C, _BLOCK_GROUPS_32C, _DOUBLE_BUFFER_32C))

    def spec_fn(combo):
        block_q, block_groups, double_buffer = combo
        return DirectConv32cSpec(
            problem=p, name="rocke_bench_direct_conv_32c",
            block_q=block_q, block_groups=block_groups, double_buffer=double_buffer,
        )

    def label_fn(combo):
        block_q, block_groups, double_buffer = combo
        return dict(block_q=block_q, block_groups=block_groups,
                    double_buffer=double_buffer, fold_k32=None)

    def grid_fn(spec, p):
        q_tiles = (p.W + spec.block_q - 1) // spec.block_q
        g_tiles = p.groups // spec.block_groups
        return (q_tiles, g_tiles, p.N)

    return _run_generic_sweep(
        args=args, problem=problem, arch=arch, variant="32c", combos=combos,
        combo_label_fn=label_fn, spec_fn=spec_fn,
        build_fn=build_direct_conv_32c, valid_fn=is_valid_spec_32c, grid_fn=grid_fn,
        compile_kernel=compile_kernel, jobs=jobs, **kw,
    )


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------


def _print_results(results, top_n_arg, arch, dtype, p, variant, show_verify):
    top_n = min(top_n_arg, len(results))
    width = 100 if show_verify else 88
    print(f"\n{'='*width}")
    print(f"Top {top_n} {variant} configurations for {arch} {dtype} {p.short()}")
    print(f"{'='*width}")
    hdr = (
        f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  {'verify':>6}  config"
        if show_verify
        else f"{'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GBps':>7}  config"
    )
    print(hdr)
    print("-" * width)
    for rank, r in enumerate(results[:top_n], 1):
        parts = [f"bq={r.block_q:3d}", f"bg={r.block_groups:3d}"]
        if r.double_buffer is not None:
            parts.append("db" if r.double_buffer else "sb")
        if r.fold_k32 is not None:
            parts.append("k32" if r.fold_k32 else "k16")
        cfg_str = " ".join(parts)
        if show_verify:
            v = "PASS" if r.passed else "FAIL"
            print(
                f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}"
                f"  {v:>6}  {cfg_str}"
            )
        else:
            print(
                f"{rank:>4}  {r.tflops:>7.1f}  {r.ms:>8.3f}  {r.gbps:>7.1f}"
                f"  {cfg_str}"
            )
    best = results[0]
    print(f"\nBest: {best.tflops:.1f} TFLOPS — {best.kernel_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_VARIANT_CPG = {"4c": 4, "8c": 8, "16c": 16, "32c": 32}
_SWEEP_FNS = {
    "4c": _run_4c_sweep,
    "8c": _run_8c_sweep,
    "16c": _run_16c_sweep,
    "32c": _run_32c_sweep,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tile sweep benchmark for direct grouped convolution "
            "(4c, 8c, 16c, 32c variants)"
        )
    )
    parser.add_argument(
        "--variant",
        default="16c",
        choices=["4c", "8c", "16c", "32c", "all"],
        help=(
            "kernel variant: "
            "4c (cpg=kpg=4), 8c (cpg=kpg=8), 16c (cpg=kpg=16), "
            "32c (cpg=kpg=32, gfx950 only), or all (default: 16c)"
        ),
    )
    parser.add_argument(
        "--arch",
        default="gfx950",
        help="gfx target (gfx942, gfx950, ...) (default: gfx950)",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="print top-N results ranked by TFLOPS (default: 10)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="timed iterations (default: 10)"
    )
    parser.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help=(
            "parallel compile workers (default: 1, serial). "
            "Set to 0 to use os.cpu_count() workers."
        ),
    )
    parser.add_argument(
        "--sample", type=float, default=None, metavar="FRAC",
        help="randomly sample FRAC of candidate combinations (e.g. 0.1 for 10%%).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed used by --sample (default: 0)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="verify each kernel against torch reference before timing",
    )
    parser.add_argument(
        "--dump-fail", default=None, metavar="PATH", dest="dump_fail",
        help=(
            "on the first verify FAIL, dump tensors to PATH/ and stop the sweep."
        ),
    )

    shape_grp = parser.add_argument_group("DirectConvProblem", "shape parameters")
    shape_grp.add_argument("--N", type=int, default=8, help="batch size (default: 8)")
    shape_grp.add_argument("--H", type=int, default=56, help="spatial height (default: 56)")
    shape_grp.add_argument("--W", type=int, default=56, help="spatial width (default: 56)")
    shape_grp.add_argument(
        "--groups", type=int, default=64,
        help=(
            "number of conv groups. Must be divisible by the block_groups values "
            "swept for each variant. (default: 64)"
        ),
    )
    shape_grp.add_argument("--KH", type=int, default=3, help="filter height (default: 3)")
    shape_grp.add_argument("--KW", type=int, default=3, help="filter width (default: 3)")
    shape_grp.add_argument("--PAD", type=int, default=1, help="padding (default: 1)")
    shape_grp.add_argument("--stride", type=int, default=1, help="stride (default: 1)")

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
    variants_to_run = (
        list(_SWEEP_FNS.keys()) if args.variant == "all" else [args.variant]
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

    all_rc = 0
    for variant in variants_to_run:
        if len(variants_to_run) > 1:
            print(f"\n{'#'*72}", flush=True)
            print(f"# Variant: {variant} (cpg=kpg={_VARIANT_CPG[variant]})", flush=True)
            print(f"{'#'*72}", flush=True)

        cpg = _VARIANT_CPG[variant]
        problem = DirectConvProblem(
            N=args.N,
            H=args.H,
            W=args.W,
            groups=args.groups,
            cpg=cpg,
            kpg=cpg,
            KH=args.KH,
            KW=args.KW,
            PAD=args.PAD,
            stride=args.stride,
        )

        sweep_fn = _SWEEP_FNS[variant]
        rc, _ = sweep_fn(problem=problem, **_common)
        all_rc = all_rc or rc

    return all_rc


if __name__ == "__main__":
    raise SystemExit(main())
