# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prototype: split a forward 3-D implicit-GEMM conv along output D across
multiple MI300A GPUs, and measure whether the halo-exchange + launch overhead
is small enough for the split to pay off.

Context: three NDHWC bf16 conv shapes that run single-GPU in 1.8-3.8 ms are
all compute-bound at ~495-520 TFLOPS (near the MI300A bf16 dense roofline).
A back-of-envelope model argued that
splitting along ``D`` (kernel depth 3, stride 1, pad 0) only needs a thin
2-D halo (``H*W*C`` per boundary plane) against the 3-D compute volume, so
the split should pay off. This script builds the actual multi-GPU split and
measures it, with a mandatory correctness gate (never report a speedup
without checking the stitched output against the single-GPU reference --
see ``platform/dsl_docs/optimization/optimization_runbook.md`` S1.4/7.3).

Deliberately standalone/example-only, like
``examples/gfx950/conv_dgrad/run_one_dgrad.py``: no production dispatch
wiring, no byte-identity concerns (same arch, same engine, not touching
emission).

Geometry: output D is split into P equal contiguous chunks (Do is divisible
by 2 and 4 for all three shapes here). Output row ``o`` reads input rows
``[o, o+2]`` (kernel depth 3, stride 1, pad 0), so each shard's local conv
problem needs ``chunk + 2`` input rows: its own ``chunk`` owned rows plus the
first 2 rows the *next* shard owns (the halo). Only the last shard needs no
halo (its tail is the true end of the tensor).

Untimed setup uploads each shard's own ``chunk`` input rows directly (models
"data already resident", the steady-state situation in a real pipeline).
The *timed* region is: (1) halo copy -- the 2 missing rows pulled from the
neighbor device via a cross-device ``torch.Tensor.copy_`` (torch's own P2P /
staged-copy path; rocke has no peer-access helper of its own, see
``platform/dsl_docs/runtime/limitations.md``), (2) each device's local conv
launch, (3) a synchronize barrier. Host wall-clock, median over iters.

Usage:
    python3 multi_gpu_split_d.py --arch gfx942 --shapes A,B,C --splits 2,4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass


# Three >1ms NDHWC bf16 forward conv shapes (F=1).
_SHAPES = {
    "A": dict(C=64, K=64, Di=258, Hi=130, Wi=130),
    "B": dict(C=128, K=64, Di=258, Hi=130, Wi=130),
    "C": dict(C=256, K=128, Di=130, Hi=66, Wi=66),
}

_TOL_BF16 = 5e-2  # same convention as library/tests/test_conv_fwd_correctness.py
_Z = 3  # fil_d = 3 for all three shapes


@dataclass
class ShardPlan:
    device: int
    out_lo: int  # owned output-D range [out_lo, out_hi)
    out_hi: int
    chunk: int  # out_hi - out_lo == owned input rows too
    has_halo: bool  # False only for the last shard


def _plan_shards(Di: int, Do: int, P: int) -> list[ShardPlan]:
    assert Do % P == 0, f"Do={Do} not divisible by P={P}"
    assert Di == Do + (_Z - 1), f"unexpected Di/Do relationship Di={Di} Do={Do}"
    chunk = Do // P
    return [
        ShardPlan(device=d, out_lo=d * chunk, out_hi=(d + 1) * chunk, chunk=chunk, has_halo=d < P - 1)
        for d in range(P)
    ]


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="gfx942")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    ap.add_argument("--shapes", default="A,B,C")
    ap.add_argument("--splits", default="2,4")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tile-m", type=int, default=64)
    ap.add_argument("--tile-n", type=int, default=64)
    ap.add_argument("--tile-k", type=int, default=64)
    ap.add_argument("--warp-m", type=int, default=2)
    ap.add_argument("--warp-n", type=int, default=2)
    ap.add_argument("--warp-tile-mn", type=int, default=32)
    ap.add_argument("--pipeline", default="mem")
    ap.add_argument("--epilogue", default="cshuffle")
    return ap.parse_args(argv)


def _build_fwd_spec(problem, a, tgt, atom):
    from kernels.common.conv_implicit_gemm import (
        ConvDataSpec,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
        is_valid_spec,
    )

    spec = ImplicitGemmConvSpec(
        problem=problem,
        name="rocke_mgpu_fwd",
        data=ConvDataSpec(dtype_a=a.dtype, dtype_b=a.dtype, dtype_d=a.dtype),
        tile_m=a.tile_m,
        tile_n=a.tile_n,
        tile_k=a.tile_k,
        warp_m=a.warp_m,
        warp_n=a.warp_n,
        warp_tile_m=a.warp_tile_mn,
        warp_tile_n=a.warp_tile_mn,
        warp_tile_k=atom.k,
        wave_size=tgt.wave_size,
        pipeline=a.pipeline,
        epilogue=a.epilogue,
    )
    ok, why = is_valid_spec(spec, arch=a.arch)
    if not ok:
        raise ValueError(f"invalid fwd spec: {why}")
    spec.validate()
    kernel = build_implicit_gemm_conv(spec, arch=a.arch)
    return spec, kernel


def _make_problem(ConvProblem, C, K, Di, Hi, Wi):
    return ConvProblem(
        N=1, C=C, K=K, Hi=Hi, Wi=Wi, Y=3, X=3,
        pH=0, pW=0, sH=1, sW=1, dH=1, dW=1, groups=1,
        Di=Di, Z=_Z, sD=1, pD=0, dD=1,
    )


def _launch_values(ConvArgs, problem, spec, X_t, W_t, Y_t):
    return ConvArgs.from_problem(
        problem, direction="fwd", tile_m=spec.tile_m, tile_n=spec.tile_n
    ).to_launch_values(
        X_t.data_ptr(), W_t.data_ptr(), Y_t.data_ptr(),
        X_t.numel() * X_t.element_size(),
        W_t.numel() * W_t.element_size(),
        Y_t.numel() * Y_t.element_size(),
    )


def run_one(shape_name: str, dims: dict, P: int, a) -> None:
    import torch

    from benchmarks.common.benchmark_implicit_gemm_conv import _grid_for_spec
    from builders.common.conv_reference import conv_reference
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import ConvProblem
    from rocke import compile_kernel
    from rocke.core.arch import ArchTarget
    from rocke.runtime import time_launches
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    tgt = ArchTarget.from_gfx(a.arch)
    family = "wmma" if tgt.wave_size == 32 else "mma"
    atom = tgt.mma.select_largest_k(
        family=family, a_dtype=a.dtype, b_dtype=a.dtype, c_dtype="fp32",
        m=a.warp_tile_mn, n=a.warp_tile_mn, k_max=a.tile_k,
    )
    if atom is None:
        print(f"[{shape_name} P={P}] no {family} atom available, skipping", file=sys.stderr)
        return

    C, K, Di, Hi, Wi = dims["C"], dims["K"], dims["Di"], dims["Hi"], dims["Wi"]
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]

    full_problem = _make_problem(ConvProblem, C, K, Di, Hi, Wi)
    Do = full_problem.Do

    ndev = torch.cuda.device_count()
    if P > ndev:
        print(f"[{shape_name} P={P}] only {ndev} GPUs visible, skipping", file=sys.stderr)
        return

    torch.manual_seed(1234)
    X = torch.empty(1, Di, Hi, Wi, C).uniform_(-1, 1).to(torch_dtype)
    W = torch.empty(K, _Z, 3, 3, C).uniform_(-1, 1).to(torch_dtype)

    # ---- P=1 baseline (single-GPU sanity check) ----
    with torch.cuda.device(0):
        spec0, kernel0 = _build_fwd_spec(full_problem, a, tgt, atom)
        art0 = compile_kernel(kernel0, arch=a.arch)
        sig_fwd = conv_args_signature(a.dtype, direction="fwd", is_3d=True)
        X0 = X.to("cuda:0").contiguous()
        W0 = W.to("cuda:0").contiguous()
        Y0 = torch.empty(1, Do, full_problem.Ho, full_problem.Wo, K, dtype=torch_dtype, device="cuda:0")
        values0 = _launch_values(ConvArgs, full_problem, spec0, X0, W0, Y0)
        grid0 = _grid_for_spec(spec0, full_problem)
        launcher0 = KernelLauncher(hsaco=art0.hsaco, kernel_name=art0.kernel_name, signature=sig_fwd)
        cfg0 = LaunchConfig(grid=grid0, block=(spec0.launch_block_size, 1, 1), stream=0)
        baseline_ms = time_launches(lambda: launcher0(values0, config=cfg0), warmup=a.warmup, iters=a.iters, stream=0)
        torch.cuda.synchronize(0)
        baseline_tflops = (float(full_problem.flops) / baseline_ms) * 1e-9

    # ---- correctness reference (fp32, full problem) ----
    ref = conv_reference(X, W, full_problem, out_dtype=torch_dtype)  # [1,Do,Ho,Wo,K]

    # ---- build the P-way shard plan + per-device spec/launcher ----
    shards = _plan_shards(Di, Do, P)
    chunk = shards[0].chunk
    shard_problem = _make_problem(ConvProblem, C, K, chunk + (_Z - 1), Hi, Wi)

    specs, arts, launchers = {}, {}, {}
    X_dev, W_dev, Y_dev = {}, {}, {}

    for d in range(P):
        with torch.cuda.device(d):
            sp, kern = _build_fwd_spec(shard_problem, a, tgt, atom)
            art = compile_kernel(kern, arch=a.arch)
            specs[d], arts[d] = sp, art
            launchers[d] = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig_fwd)
            s = shards[d]
            local_rows = chunk + (_Z - 1)
            buf = torch.empty(1, local_rows, Hi, Wi, C, dtype=torch_dtype, device=f"cuda:{d}")
            if s.has_halo:
                # own chunk rows now; the last (Z-1) rows are filled by the
                # timed halo_exchange() every call, from the next shard.
                buf[:, :chunk].copy_(X[:, s.out_lo : s.out_hi].to(f"cuda:{d}"))
            else:
                # last shard: the trailing (Z-1) rows are nobody else's --
                # upload the whole local window once, no halo needed.
                buf[:, :local_rows].copy_(X[:, s.out_lo : s.out_lo + local_rows].to(f"cuda:{d}"))
            X_dev[d] = buf
            W_dev[d] = W.to(f"cuda:{d}").contiguous()
            Y_dev[d] = torch.empty(
                1, shard_problem.Do, shard_problem.Ho, shard_problem.Wo, K,
                dtype=torch_dtype, device=f"cuda:{d}",
            )

    can_p2p = P >= 2 and torch.cuda.can_device_access_peer(0, 1)

    def halo_exchange():
        for d in range(P - 1):
            # the (Z-1) rows shard d is missing are exactly the first (Z-1)
            # owned rows of shard d+1.
            tail = X_dev[d + 1][:, : _Z - 1].to(f"cuda:{d}")
            X_dev[d][:, chunk:].copy_(tail)

    def launch_all():
        for d in range(P):
            with torch.cuda.device(d):
                values = _launch_values(ConvArgs, shard_problem, specs[d], X_dev[d], W_dev[d], Y_dev[d])
                grid = _grid_for_spec(specs[d], shard_problem)
                cfg = LaunchConfig(grid=grid, block=(specs[d].launch_block_size, 1, 1), stream=0)
                launchers[d](values, config=cfg)

    def sync_all():
        for d in range(P):
            torch.cuda.synchronize(d)

    # ---- correctness gate before any timing is trusted ----
    halo_exchange()
    launch_all()
    sync_all()
    max_rel_err = 0.0
    for d in range(P):
        s = shards[d]
        out = Y_dev[d].float().cpu()
        ref_slice = ref[:, s.out_lo : s.out_hi].float().cpu()
        abs_diff = (out - ref_slice).abs()
        ref_scale = ref_slice.abs().max().clamp(min=1.0)
        rel_err = float(abs_diff.max() / ref_scale)
        max_rel_err = max(max_rel_err, rel_err)
    passed = max_rel_err < _TOL_BF16
    if not passed:
        print(
            f"[{shape_name} P={P}] CORRECTNESS FAIL rel_err={max_rel_err:.3e} "
            f"> tol={_TOL_BF16:.1e} -- not reporting timings",
            file=sys.stderr,
        )
        return

    # ---- timed region: halo exchange + launch + sync, host wall-clock ----
    for _ in range(a.warmup):
        halo_exchange()
        launch_all()
        sync_all()

    wall_times, halo_times = [], []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        halo_exchange()
        sync_all()
        t1 = time.perf_counter()
        launch_all()
        sync_all()
        t2 = time.perf_counter()
        halo_times.append((t1 - t0) * 1e3)
        wall_times.append((t2 - t0) * 1e3)

    halo_ms = statistics.median(halo_times)
    wall_ms = statistics.median(wall_times)
    compute_ms = wall_ms - halo_ms
    per_device_tflops = (float(shard_problem.flops) / compute_ms) * 1e-9 if compute_ms > 0 else float("nan")

    ideal_ms = baseline_ms / P
    speedup_vs_baseline = baseline_ms / wall_ms
    pct_of_ideal = (ideal_ms / wall_ms) * 100.0

    print(
        f"[{shape_name} P={P}] baseline={baseline_ms:.4f}ms "
        f"({baseline_tflops:.0f} TFLOPS) | multi-gpu wall={wall_ms:.4f}ms "
        f"(halo={halo_ms:.4f}ms, compute~{compute_ms:.4f}ms, "
        f"{per_device_tflops:.0f} TFLOPS/dev) | speedup={speedup_vs_baseline:.2f}x "
        f"({pct_of_ideal:.0f}% of ideal {P}x) | "
        f"p2p={'yes' if can_p2p else 'no/staged'} | "
        f"correctness rel_err={max_rel_err:.2e} (tol {_TOL_BF16:.0e}) PASS"
    )


def main(argv=None) -> int:
    a = _parse_args(argv)
    shape_names = [s.strip() for s in a.shapes.split(",") if s.strip()]
    splits = [int(s.strip()) for s in a.splits.split(",") if s.strip()]
    for shape_name in shape_names:
        dims = _SHAPES[shape_name]
        for P in splits:
            run_one(shape_name, dims, P, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
