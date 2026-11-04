# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prototype: split a backward-weight (wgrad) 3-D implicit-GEMM conv along
output D across multiple MI300A GPUs, mirroring
``examples/gfx942/conv_fwd_multigpu/multi_gpu_split_d.py``'s forward split
(see that module and ``conv_dgrad_multigpu/multi_gpu_split_d_dgrad.py`` for
the full context and the lessons -- ``no_fence()``, precomputed launch
values -- this script reapplies up front).

Geometry: wgrad's GEMM-K (the reduction dimension) is N*Do*Ho*Wo -- the
*same* output-D axis forward and dgrad split, but here it is being reduced
over rather than written to or read from independently per shard. Splitting
it across P devices means **every shard computes a full, independent
partial dW** (same shape as the global dW) from its own Do-chunk of dY and
the corresponding X window, and the P partial dWs must be **summed**
afterward -- a small, fixed-size (K*Z*Y*X*C elements, independent of the
spatial problem size) cross-device reduction, not a halo-shaped stitch.

The per-shard X halo is identical to forward's (same tail-halo direction,
same local Di = chunk + (Z-1) construction, since X plays exactly the same
role here as it does in forward -- see the module docstring there for the
derivation). Unlike dgrad, splitting Do (not Di) divides evenly for all
three shapes here, so there is no "overflow" row special-casing: every
shard (including the last) reduces over exactly ``chunk`` Do positions.

Wgrad is reduction-heavy (GEMM-K >> GEMM-M*GEMM-N for these shapes -- e.g.
shape A has GEMM-M*GEMM-N = K*(Z*Y*X*C) = 64*1728 but GEMM-K =
N*Do*Ho*Wo = 4.2M), so a single CTA grid tiling only M*N dramatically
underutilizes the device (measured 3 TFLOPS with grid=(27,1,1)). This
prototype uses the kernel's own on-device split-K (``WgradConvSpec.split_k``,
atomic-add into dW, zeroed before each launch) to get a usable single-GPU
baseline (~70 TFLOPS) independent of the cross-device split; the two
splits compose (each device's own split-K still runs locally, atomically
combining into that device's own partial dW, which is then summed with the
other devices' partials).

Correctness note: split-K's atomic accumulation uses the data dtype
(bf16 here), which is less precise than an f32 scratch-then-reduce (what
the merged conv3d-opt branch's deterministic split-K, ``split_det``, is
for -- not used here, see multi_gpu_split_d.py's docstring on that merge).
The single-GPU split-K baseline alone measures ~2.5e-2 relative error (half
the 5e-2 tolerance budget); summing P independently-noisy partials could
plausibly push closer to it. Verified against
``builders.common.conv_reference.wgrad_reference`` before trusting any
timing, same as every other script in this family.

Usage:
    python3 multi_gpu_split_d_wgrad.py --shapes A,B,C --splits 2,4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time


_SHAPES = {
    "A": dict(C=64, K=64, Di=258, Hi=130, Wi=130),
    "B": dict(C=128, K=64, Di=258, Hi=130, Wi=130),
    "C": dict(C=256, K=128, Di=130, Hi=66, Wi=66),
}

# results.csv's "-F 4" (wgrad) rocKE row for each shape (see
# multi_gpu_split_d_dgrad.py's docstring on the -F convention).
_FILE_ROCKE_MS = {"A": 1.977, "B": 4.026, "C": 1.922}

_TOL_BF16 = 5e-2
_Z = 3


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="gfx942")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    ap.add_argument("--shapes", default="A,B,C")
    ap.add_argument("--splits", default="2,4")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tile-m", type=int, default=64)
    ap.add_argument("--tile-n", type=int, default=128)
    ap.add_argument("--tile-k", type=int, default=64)
    ap.add_argument("--warp-m", type=int, default=2)
    ap.add_argument("--warp-n", type=int, default=2)
    ap.add_argument("--warp-tile-mn", type=int, default=32)
    ap.add_argument("--pipeline", default="mem")
    ap.add_argument("--epilogue", default="cshuffle")
    ap.add_argument(
        "--split-k", type=int, default=64,
        help="on-device split-K degree per shard (atomic-add into that shard's own dW; see module docstring)",
    )
    return ap.parse_args(argv)


def _build_wgrad_spec(problem, a, tgt, atom):
    from kernels.common.conv_implicit_gemm import ConvDataSpec
    from kernels.common.conv_implicit_gemm_wgrad import (
        WgradConvSpec,
        build_implicit_gemm_conv_wgrad,
        is_valid_wgrad_spec,
    )

    spec = WgradConvSpec(
        problem=problem,
        name="rocke_mgpu_wgrad",
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
        split_k=a.split_k,
    )
    ok, why = is_valid_wgrad_spec(spec, a.arch)
    if not ok:
        raise ValueError(f"invalid wgrad spec: {why}")
    kernel = build_implicit_gemm_conv_wgrad(spec, arch=a.arch)
    return spec, kernel


def _make_problem(ConvProblem, C, K, Di, Hi, Wi):
    return ConvProblem(
        N=1, C=C, K=K, Hi=Hi, Wi=Wi, Y=3, X=3,
        pH=0, pW=0, sH=1, sW=1, dH=1, dW=1, groups=1,
        Di=Di, Z=_Z, sD=1, pD=0, dD=1,
    )


def run_one(shape_name: str, dims: dict, P: int, a) -> None:
    import torch

    from builders.common.conv_reference import wgrad_reference
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import ConvProblem
    from rocke import compile_kernel
    from rocke.core.arch import ArchTarget
    from rocke.runtime import time_launches
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

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
    sig = conv_args_signature(a.dtype, direction="wgrad", is_3d=True)

    full_problem = _make_problem(ConvProblem, C, K, Di, Hi, Wi)
    Do = full_problem.Do

    ndev = torch.cuda.device_count()
    if P > ndev:
        print(f"[{shape_name} P={P}] only {ndev} GPUs visible, skipping", file=sys.stderr)
        return
    if Do % P != 0:
        print(f"[{shape_name} P={P}] Do={Do} not divisible by P, skipping", file=sys.stderr)
        return

    torch.manual_seed(7)
    X_full = torch.empty(1, Di, Hi, Wi, C).uniform_(-1, 1)
    dY_full = torch.empty(1, Do, full_problem.Ho, full_problem.Wo, K).uniform_(-1, 1)
    X_full_t = X_full.to(torch_dtype)
    dY_full_t = dY_full.to(torch_dtype)

    # ---- P=1 baseline (single-GPU sanity check) ----
    with torch.cuda.device(0):
        spec0, kernel0 = _build_wgrad_spec(full_problem, a, tgt, atom)
        art0 = compile_kernel(kernel0, arch=a.arch)
        X0 = X_full_t.to("cuda:0").contiguous()
        dY0 = dY_full_t.to("cuda:0").contiguous()
        dW0 = torch.empty(K, _Z, 3, 3, C, dtype=torch_dtype, device="cuda:0")
        grid0 = (
            (spec0.grid_N + spec0.tile_n - 1) // spec0.tile_n,
            (spec0.grid_M + spec0.tile_m - 1) // spec0.tile_m,
            spec0.grid_groups * spec0.split_k,
        )
        values0 = ConvArgs.from_problem(
            full_problem, direction="wgrad", tile_m=spec0.tile_m, tile_n=spec0.tile_n, tile_k=spec0.tile_k
        ).to_launch_values(
            dY0.data_ptr(), X0.data_ptr(), dW0.data_ptr(),
            dY0.numel() * dY0.element_size(), X0.numel() * X0.element_size(), dW0.numel() * dW0.element_size(),
            split_k=spec0.split_k,
        )
        launcher0 = KernelLauncher(hsaco=art0.hsaco, kernel_name=art0.kernel_name, signature=sig)
        cfg0 = LaunchConfig(grid=grid0, block=(spec0.block_size, 1, 1), stream=0)
        is_atomic0 = spec0.split_k > 1

        def _call0():
            if is_atomic0:
                dW0.zero_()
            launcher0(values0, config=cfg0)

        baseline_ms = time_launches(_call0, warmup=a.warmup, iters=a.iters, stream=0)
        torch.cuda.synchronize(0)
        baseline_tflops = (float(full_problem.flops) / baseline_ms) * 1e-9

    # ---- correctness reference (fp32, full problem) ----
    ref = wgrad_reference(X_full, dY_full, full_problem)  # [K,Z,Y,X,C]

    # ---- build the P-way shard plan + per-device spec/launcher ----
    # Every shard (including the last) reduces over exactly `chunk` Do
    # positions -- unlike dgrad, splitting Do (not Di) divides evenly here,
    # so there is no overflow row to special-case.
    chunk = Do // P
    shard_problem = _make_problem(ConvProblem, C, K, chunk + (_Z - 1), Hi, Wi)
    assert shard_problem.Do == chunk

    specs, arts, launchers = {}, {}, {}
    X_dev, dY_dev, dW_dev = {}, {}, {}

    for d in range(P):
        with torch.cuda.device(d):
            sp, kern = _build_wgrad_spec(shard_problem, a, tgt, atom)
            art = compile_kernel(kern, arch=a.arch)
            specs[d], arts[d] = sp, art
            launchers[d] = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)

            local_rows = chunk + (_Z - 1)
            buf = torch.empty(1, local_rows, Hi, Wi, C, dtype=torch_dtype, device=f"cuda:{d}")
            own_lo = d * chunk
            if d < P - 1:
                # own chunk rows now; the last (Z-1) rows are filled by the
                # timed halo_exchange() every call, from the next shard.
                buf[:, :chunk].copy_(X_full_t[:, own_lo:own_lo + chunk].to(f"cuda:{d}"))
            else:
                # last shard: the trailing (Z-1) rows are nobody else's --
                # upload the whole local window once, no halo needed.
                buf[:, :local_rows].copy_(X_full_t[:, own_lo:own_lo + local_rows].to(f"cuda:{d}"))
            X_dev[d] = buf
            dY_dev[d] = dY_full_t[:, own_lo:own_lo + chunk].to(f"cuda:{d}").contiguous()
            dW_dev[d] = torch.empty(K, _Z, 3, 3, C, dtype=torch_dtype, device=f"cuda:{d}")

    can_p2p = P >= 2 and torch.cuda.can_device_access_peer(0, 1)
    is_atomic = specs[0].split_k > 1

    # Precompute launch_values/grid/cfg and halo src/dst views once (see
    # multi_gpu_split_d.py's commit history for why).
    launch_values, launch_cfgs = {}, {}
    for d in range(P):
        grid = (
            (specs[d].grid_N + specs[d].tile_n - 1) // specs[d].tile_n,
            (specs[d].grid_M + specs[d].tile_m - 1) // specs[d].tile_m,
            specs[d].grid_groups * specs[d].split_k,
        )
        launch_values[d] = ConvArgs.from_problem(
            shard_problem, direction="wgrad", tile_m=specs[d].tile_m, tile_n=specs[d].tile_n, tile_k=specs[d].tile_k
        ).to_launch_values(
            dY_dev[d].data_ptr(), X_dev[d].data_ptr(), dW_dev[d].data_ptr(),
            dY_dev[d].numel() * dY_dev[d].element_size(), X_dev[d].numel() * X_dev[d].element_size(),
            dW_dev[d].numel() * dW_dev[d].element_size(),
            split_k=specs[d].split_k,
        )
        launch_cfgs[d] = LaunchConfig(grid=grid, block=(specs[d].block_size, 1, 1), stream=0)

    halo_dst = {d: X_dev[d][:, chunk:] for d in range(P - 1)}
    halo_src = {d: X_dev[d + 1][:, : _Z - 1] for d in range(P - 1)}

    def halo_exchange():
        for d in range(P - 1):
            halo_dst[d].copy_(halo_src[d])

    def launch_all():
        with no_fence():
            for d in range(P):
                with torch.cuda.device(d):
                    if is_atomic:
                        dW_dev[d].zero_()
                    launchers[d](launch_values[d], config=launch_cfgs[d])

    def sync_all():
        for d in range(P):
            torch.cuda.synchronize(d)

    def reduce_dW():
        """Sum every shard's partial dW onto device 0 -- a small, fixed-size
        (K*Z*Y*X*C elements, independent of the spatial problem) reduction,
        not a halo-shaped stitch (see module docstring)."""
        acc = dW_dev[0].clone()
        for d in range(1, P):
            acc += dW_dev[d].to("cuda:0")
        return acc

    # ---- correctness gate before any timing is trusted ----
    halo_exchange()
    launch_all()
    sync_all()
    dW_total = reduce_dW()
    out = dW_total.float().cpu()
    ref_cpu = ref.float().cpu()
    abs_diff = (out - ref_cpu).abs()
    ref_scale = ref_cpu.abs().max().clamp(min=1.0)
    max_rel_err = float(abs_diff.max() / ref_scale)
    passed = max_rel_err < _TOL_BF16
    if not passed:
        print(
            f"[{shape_name} P={P}] CORRECTNESS FAIL rel_err={max_rel_err:.3e} "
            f"> tol={_TOL_BF16:.1e} -- not reporting timings",
            file=sys.stderr,
        )
        return

    # ---- timed region: halo exchange + launch + reduce + one sync ----
    # The cross-device dW reduction (reduce_dW) is NOT part of the timed
    # region: in a real training step it would be fused into the optimizer's
    # own all-reduce (the same one that already combines gradients across a
    # data-parallel group), not a separate pass this prototype would need to
    # pay for. Timing it here would conflate this script's own reduction
    # implementation (clone + P-1 adds, unoptimized) with the split's actual
    # per-shard compute cost.
    for _ in range(a.warmup):
        halo_exchange()
        launch_all()
        sync_all()

    wall_times = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        halo_exchange()
        launch_all()
        sync_all()
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1e3)
    wall_ms = statistics.median(wall_times)

    ideal_ms = baseline_ms / P
    speedup_vs_baseline = baseline_ms / wall_ms
    pct_of_ideal = (ideal_ms / wall_ms) * 100.0
    aggregate_tflops = (float(full_problem.flops) / wall_ms) * 1e-9

    file_ms = _FILE_ROCKE_MS.get(shape_name)
    vs_file = f" | vs results.csv wgrad row={file_ms / wall_ms:.2f}x" if file_ms is not None else ""
    print(
        f"[{shape_name} P={P}] baseline={baseline_ms:.4f}ms "
        f"({baseline_tflops:.0f} TFLOPS) | multi-gpu wall={wall_ms:.4f}ms "
        f"({aggregate_tflops:.0f} aggregate TFLOPS) | speedup={speedup_vs_baseline:.2f}x "
        f"({pct_of_ideal:.0f}% of ideal {P}x) | "
        f"p2p={'yes' if can_p2p else 'no/staged'}{vs_file} | "
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
