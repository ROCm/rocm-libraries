# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prototype: split a backward-data (dgrad) 3-D implicit-GEMM conv along
output D across multiple MI300A GPUs, mirroring
``examples/gfx942/conv_fwd_multigpu/multi_gpu_split_d.py``'s forward split.

Context: ``results.csv`` compares rocKE against MIOpen/Triton for the same
three NDHWC bf16 conv shapes at three MIOpenDriver ``-F`` values --
``-F 1`` = forward, ``-F 2`` = backward-data (dgrad), ``-F 4`` =
backward-weight (wgrad) (see ``kernels/conv/conv_pins.py``'s
``_DIRECTIONS = {1: "fwd", 2: "dgrad", 4: "wgrad"}``). The forward prototype
only ever compared against the ``-F 1`` rows; this script does the same
split-D idea for the ``-F 2`` (dgrad) rows.

Geometry, mirrored but reversed: dgrad computes dX (shape like forward's
input, Di rows) from dY (shape like forward's output, Do rows) and W. Di =
Do + (Z-1) is still true, but the *direction* of the halo flips relative to
forward: dX row i reads dY rows [i-(Z-1), i] (stride=1, pad=0), so a shard
owning a contiguous dX range needs (Z-1) *extra dY rows from its
predecessor* (a head halo), not its successor. Concretely, for a shard
owning dY range (its own, no overlap) [d*chunk, (d+1)*chunk):
  - shard 0: no halo. Local problem has Do_local = chunk (own dY rows
    only); its natural Di_local = Do_local + (Z-1) local dX outputs, of
    which only the first `chunk` are kept (the kernel's own right-edge
    zero-padding logic produces garbage for the rest, which shard 1
    recomputes correctly using real data).
  - shard d > 0: Do_local = (Z-1) head-halo rows (copied from shard d-1's
    own last (Z-1) dY rows) + its own chunk. Its natural
    Di_local = Do_local + (Z-1) local dX outputs; the first (Z-1) are
    discarded (they belong to shard d-1's kept range) and the rest are
    kept (the full remainder for the last shard, since Di - P*chunk extra
    rows only exist once, owned entirely by the last shard -- same
    convention as the forward prototype).
This was derived from the dX[i] = sum_z dY[i-z]*W[z] relation and verified
against ``builders.common.conv_reference.dgrad_reference`` before trusting
any timing (same correctness-before-speed rule as the forward script).

Unlike forward's halo (a thin slice of the *input*, already touched by
nothing else), dgrad's dY shards are each device's own disjoint data (no
double-buffering question): only the (Z-1)-row head halo is ever
cross-device.

Deliberately standalone/example-only, same spirit as the forward script:
no production dispatch wiring. Applies the lessons already learned there:
KernelLauncher's fenced-by-default launch must be wrapped in ``no_fence()``
for true multi-GPU overlap, and ``launch_values``/halo src-dst views should
be precomputed once per shard rather than rebuilt every timed iteration.

Usage:
    python3 multi_gpu_split_d_dgrad.py --shapes A,B,C --splits 2,4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass


_SHAPES = {
    "A": dict(C=64, K=64, Di=258, Hi=130, Wi=130),
    "B": dict(C=128, K=64, Di=258, Hi=130, Wi=130),
    "C": dict(C=256, K=128, Di=130, Hi=66, Wi=66),
}

# results.csv's "-F 2" (dgrad) rocKE row for each shape, for comparison only
# (results.csv is external to this repo; see multi_gpu_split_d.py's
# _FILE_ROCKE_MS for the analogous forward numbers).
_FILE_ROCKE_MS = {"A": 1.968, "B": 3.930, "C": 1.834}

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
    ap.add_argument("--tile-m", type=int, default=128)
    ap.add_argument("--tile-n", type=int, default=64)
    ap.add_argument("--tile-k", type=int, default=32)
    ap.add_argument("--warp-m", type=int, default=2)
    ap.add_argument("--warp-n", type=int, default=2)
    ap.add_argument("--warp-tile-mn", type=int, default=16)
    ap.add_argument("--pipeline", default="mem")
    ap.add_argument("--epilogue", default="cshuffle")
    return ap.parse_args(argv)


@dataclass
class ShardPlan:
    device: int
    head_halo: int  # (Z-1) for every shard but 0, else 0
    own_chunk: int  # dY rows this shard owns directly (no overlap with any other shard)
    is_last: bool


def _plan_shards(Do: int, P: int) -> list[ShardPlan]:
    assert Do % P == 0, f"Do={Do} not divisible by P={P}"
    chunk = Do // P
    return [
        ShardPlan(device=d, head_halo=0 if d == 0 else (_Z - 1), own_chunk=chunk, is_last=(d == P - 1))
        for d in range(P)
    ]


def _build_dgrad_spec(local_problem, a, tgt, atom, vec):
    from kernels.common.conv_implicit_gemm import ConvDataSpec
    from kernels.common.conv_implicit_gemm_dgrad import (
        DgradConvSpec,
        build_implicit_gemm_conv_dgrad,
        is_valid_dgrad_spec,
    )

    vec_a, vec_b, vec_c = vec
    spec = DgradConvSpec(
        problem=local_problem,
        name="rocke_mgpu_dgrad",
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
        split_k=1,
        vector_size_a=vec_a,
        vector_size_b=vec_b,
        vector_size_c=vec_c,
    )
    ok, why = is_valid_dgrad_spec(spec, arch=a.arch)
    if not ok:
        raise ValueError(f"invalid dgrad spec: {why}")
    kernel = build_implicit_gemm_conv_dgrad(spec, arch=a.arch)
    return spec, kernel


def _make_problem(ConvProblem, C, K, Di, Hi, Wi):
    return ConvProblem(
        N=1, C=C, K=K, Hi=Hi, Wi=Wi, Y=3, X=3,
        pH=0, pW=0, sH=1, sW=1, dH=1, dW=1, groups=1,
        Di=Di, Z=_Z, sD=1, pD=0, dD=1,
    )


def run_one(shape_name: str, dims: dict, P: int, a) -> None:
    import ctypes
    import struct as _struct

    import torch

    from builders.common.conv_reference import dgrad_reference
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import ConvProblem
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec, pack_sub_gemm_buffer
    from rocke import compile_kernel
    from rocke.core.arch import ArchTarget
    from rocke.runtime import time_launches
    from rocke.runtime.hip_module import Runtime
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
    sig = conv_args_signature(a.dtype, direction="dgrad", is_3d=True)

    full_problem = _make_problem(ConvProblem, C, K, Di, Hi, Wi)
    Do = full_problem.Do
    vec = DgradConvSpec.default_vector_sizes(full_problem.cpg, full_problem.kpg, a.dtype)

    ndev = torch.cuda.device_count()
    if P > ndev:
        print(f"[{shape_name} P={P}] only {ndev} GPUs visible, skipping", file=sys.stderr)
        return

    torch.manual_seed(42)
    dY_full = torch.empty(1, Do, full_problem.Ho, full_problem.Wo, K).uniform_(-1, 1)
    W_full = torch.empty(K, _Z, 3, 3, C).uniform_(-1, 1)
    dY_full_t = dY_full.to(torch_dtype)
    W_full_t = W_full.to(torch_dtype)

    # ---- P=1 baseline (single-GPU sanity check) ----
    with torch.cuda.device(0):
        spec0, kernel0 = _build_dgrad_spec(full_problem, a, tgt, atom, vec)
        art0 = compile_kernel(kernel0, arch=a.arch)
        dY0 = dY_full_t.to("cuda:0").contiguous()
        W0 = W_full_t.to("cuda:0").contiguous()
        dX0 = torch.empty(1, Di, Hi, Wi, C, dtype=torch_dtype, device="cuda:0")
        sub_gemms0 = spec0.compute_sub_gemms()
        buf0 = pack_sub_gemm_buffer(sub_gemms0, spec0.tile_m, spec0.tile_n)
        buf0_bytes = _struct.pack(f"{len(buf0)}i", *buf0)
        rt0 = Runtime()
        sgbuf0 = rt0.alloc(len(buf0_bytes))
        rt0.memcpy_h2d(sgbuf0, (ctypes.c_uint8 * len(buf0_bytes)).from_buffer_copy(buf0_bytes), len(buf0_bytes))
        values0 = ConvArgs.from_problem(full_problem, direction="dgrad", tile_m=spec0.tile_m, tile_n=spec0.tile_n).to_launch_values(
            dY0.data_ptr(), W0.data_ptr(), dX0.data_ptr(),
            dY0.numel() * dY0.element_size(), W0.numel() * W0.element_size(), dX0.numel() * dX0.element_size(),
            sub_gemm_buf=int(sgbuf0), num_sub_gemms=len(sub_gemms0),
        )
        grid0 = (sub_gemms0[-1].block_end, 1, spec0.split_k)
        launcher0 = KernelLauncher(hsaco=art0.hsaco, kernel_name=art0.kernel_name, signature=sig)
        cfg0 = LaunchConfig(grid=grid0, block=(spec0.launch_block_size, 1, 1), stream=0)
        baseline_ms = time_launches(lambda: launcher0(values0, config=cfg0), warmup=a.warmup, iters=a.iters, stream=0)
        torch.cuda.synchronize(0)
        baseline_tflops = (float(full_problem.flops) / baseline_ms) * 1e-9

    # ---- correctness reference (fp32, full problem) ----
    ref = dgrad_reference(dY_full, W_full, full_problem)  # [1,Di,Hi,Wi,C]

    # ---- build the P-way shard plan + per-device spec/launcher ----
    shards = _plan_shards(Do, P)
    chunk = shards[0].own_chunk

    specs, arts, launchers = {}, {}, {}
    dY_dev, W_dev, dX_dev = {}, {}, {}
    sgbufs = {}
    keep_lo, keep_hi = {}, {}

    for d in range(P):
        s = shards[d]
        with torch.cuda.device(d):
            Do_local = s.head_halo + s.own_chunk
            Di_local = Do_local + (_Z - 1)
            local_problem = _make_problem(ConvProblem, C, K, Di_local, Hi, Wi)
            assert local_problem.Do == Do_local
            sp, kern = _build_dgrad_spec(local_problem, a, tgt, atom, vec)
            art = compile_kernel(kern, arch=a.arch)
            specs[d], arts[d] = sp, art
            launchers[d] = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)

            dY_local = torch.empty(1, Do_local, full_problem.Ho, full_problem.Wo, K, dtype=torch_dtype, device=f"cuda:{d}")
            own_lo = d * chunk
            dY_local[:, s.head_halo:].copy_(dY_full_t[:, own_lo:own_lo + s.own_chunk].to(f"cuda:{d}"))
            dY_dev[d] = dY_local
            W_dev[d] = W_full_t.to(f"cuda:{d}").contiguous()
            dX_dev[d] = torch.empty(1, Di_local, Hi, Wi, C, dtype=torch_dtype, device=f"cuda:{d}")

            sub_gemms = sp.compute_sub_gemms()
            bufi = pack_sub_gemm_buffer(sub_gemms, sp.tile_m, sp.tile_n)
            buf_bytes = _struct.pack(f"{len(bufi)}i", *bufi)
            rt = Runtime()
            sgbuf = rt.alloc(len(buf_bytes))
            rt.memcpy_h2d(sgbuf, (ctypes.c_uint8 * len(buf_bytes)).from_buffer_copy(buf_bytes), len(buf_bytes))
            sgbufs[d] = sgbuf

            keep_lo[d] = s.head_halo
            keep_hi[d] = Di_local if s.is_last else (s.head_halo + s.own_chunk)

    can_p2p = P >= 2 and torch.cuda.can_device_access_peer(0, 1)

    # Precompute launch_values/grid/cfg and halo src/dst views once (see
    # multi_gpu_split_d.py's commit history for why: rebuilding these every
    # timed iteration is a fixed host cost that does not shrink with the
    # per-shard kernel, and can dominate at higher P).
    launch_values, launch_cfgs = {}, {}
    for d in range(P):
        s = shards[d]
        Do_local = s.head_halo + s.own_chunk
        local_problem = _make_problem(ConvProblem, C, K, Do_local + (_Z - 1), Hi, Wi)
        sub_gemms = specs[d].compute_sub_gemms()
        grid = (sub_gemms[-1].block_end, 1, specs[d].split_k)
        launch_values[d] = ConvArgs.from_problem(
            local_problem, direction="dgrad", tile_m=specs[d].tile_m, tile_n=specs[d].tile_n
        ).to_launch_values(
            dY_dev[d].data_ptr(), W_dev[d].data_ptr(), dX_dev[d].data_ptr(),
            dY_dev[d].numel() * dY_dev[d].element_size(), W_dev[d].numel() * W_dev[d].element_size(),
            dX_dev[d].numel() * dX_dev[d].element_size(),
            sub_gemm_buf=int(sgbufs[d]), num_sub_gemms=len(sub_gemms),
        )
        launch_cfgs[d] = LaunchConfig(grid=grid, block=(specs[d].launch_block_size, 1, 1), stream=0)

    halo_dst = {d: dY_dev[d][:, : shards[d].head_halo] for d in range(1, P)}
    halo_src = {
        d: dY_dev[d - 1][:, shards[d - 1].head_halo + shards[d - 1].own_chunk - shards[d].head_halo:
                         shards[d - 1].head_halo + shards[d - 1].own_chunk]
        for d in range(1, P)
    }

    def halo_exchange():
        for d in range(1, P):
            halo_dst[d].copy_(halo_src[d])

    def launch_all():
        with no_fence():
            for d in range(P):
                with torch.cuda.device(d):
                    launchers[d](launch_values[d], config=launch_cfgs[d])

    def sync_all():
        for d in range(P):
            torch.cuda.synchronize(d)

    # ---- correctness gate before any timing is trusted ----
    halo_exchange()
    launch_all()
    sync_all()
    max_rel_err = 0.0
    for d in range(P):
        out = dX_dev[d][:, keep_lo[d]:keep_hi[d]].float().cpu()
        global_lo = d * chunk
        global_hi = global_lo + (keep_hi[d] - keep_lo[d])
        ref_slice = ref[:, global_lo:global_hi].float().cpu()
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

    # ---- timed region: halo exchange + launch + one sync, no mid-barrier ----
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
    vs_file = f" | vs results.csv dgrad row={file_ms / wall_ms:.2f}x" if file_ms is not None else ""
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
