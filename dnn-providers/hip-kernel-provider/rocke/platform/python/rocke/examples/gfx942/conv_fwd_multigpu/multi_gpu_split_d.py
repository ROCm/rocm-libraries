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

Measured finding (2 MI300A GCDs, bf16, all three shapes): the halo
hypothesis was right -- halo copy is 2-4% of wall time, nowhere near the
bottleneck -- and with a correct launch, the split pays off: 1.64x-1.74x
speedup over P=1 (82-87% of ideal 2x), correctness PASS.

That "correct launch" fix is the load-bearing part, and it was not obvious:
``KernelLauncher`` defaults to a *fenced* launch
(``LaunchConfig.fence=True``, see ``rocke/runtime/launcher.py``) -- every
call does a synchronous ``hipStreamSynchronize`` before returning, so a
Python loop issuing ``launcher(...)`` once per device never lets the two
devices' kernels overlap: each call blocks the host until its own device's
kernel is done before the next device's launch can even be issued, making
P devices strictly additive instead of concurrent. ``launch_all()`` below
wraps the per-device loop in ``rocke.runtime.launcher.no_fence()`` (the
same thing :func:`~rocke.runtime.time_launches` does internally) to get a
true fire-and-forget async launch on each device.

This was misdiagnosed once already in this file's history: an earlier
version of this docstring blamed "node-level power/clock contention
between GCDs" for a measured ~2x collapse in concurrent per-device
throughput with fencing left on. That was wrong -- clock/hwmon sampling
during the concurrent run showed both GPUs at full boost the whole time,
and a two-separate-OS-process control (each pinned to one device, no
shared Python runtime) hit full isolated TFLOPS on both simultaneously,
which is what pointed at an in-process software serialization point
rather than real hardware contention. If a `launch_all()`-shaped loop in
another prototype shows a mysterious ~Px slowdown under concurrency,
check the fence default before suspecting the node.

Usage:
    python3 multi_gpu_split_d.py --arch gfx942 --shapes A,B,C --splits 2,4

Baseline tile configs with ``--no-advanced`` (plain tile/warp sweep, see
``benchmarks/common/benchmark_implicit_gemm_conv.py``; this is what the first
version of this script shipped with, ~230-250 TFLOPS/device):
    A (C=64,K=64,Di=258):   tile=128x64x32  warp=4x1 atom=16x16x16 mem/cshuffle
    B (C=128,K=64,Di=258):  tile=128x64x32  warp=2x2 atom=16x16x16 mem/cshuffle
    C (C=256,K=128,Di=130): tile=128x128x32 warp=4x2 atom=16x16x16 compv4/cshuffle

``--advanced`` (default): the ``users/barkocot/conv-multigpu`` branch merged
``users/aledudek/conv3d-opt`` (the source of the external ``results.csv``
comparison this prototype was built to beat -- see PR #14711), which adds
~60 new opt-in ``ImplicitGemmConvSpec`` fields (register-staged K-loop
prefetch, LDS XOR-swizzled tiles, x-halo input reuse, chiplet/XCD-aware grid
remap, output-plane band tile order, ...). None of that space is swept by
``benchmark_implicit_gemm_conv.py`` for forward, and the full AOT grid
(``kernels/conv/aot_jobs.py``) is >1.6M combinations even scoped to 3-D --
not something to search blind. ``_ADVANCED_TILE`` / ``_ADVANCED_KNOBS``
below are a *time-boxed* hand search (not exhaustive) over the most
promising subset (``uniform_k_decode``, ``reg_prefetch``, ``lds_xor``,
``uk_xhalo``, ``epi_quad_xpose``, ``xcd_remap=6`` -- this node's real XCD
count per ``/sys/class/kfd/kfd/topology/nodes/*/properties:num_xcc``,
``rpf_setprio``, ``defer_mfma``), applied to both the P=1 baseline and every
shard. Measured per-device gain over the plain-tile baseline: A/B ~253->417
TFLOPS (+65%), C ~192->350 TFLOPS (+82%, smaller tile wins for the smaller
spatial shape). Combined with a correct (``no_fence``) 2-way split, this
beats the external comparison's single-GPU numbers on all three shapes:

    A: 1.37x faster than results.csv's rocKE row
    B: 1.46x faster
    C: 1.11x faster

all with the same correctness gate PASS. Treat this as a validated starting
point, not a ceiling: a proper (non-time-boxed) sweep of the full opt-in
space, or getting aledudek's own tuning notes/pins for these exact shapes,
would likely do better still.
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

# (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn) per shape, found by the
# time-boxed hand search described in the module docstring's --advanced
# section. Applied to both the P=1 baseline and every shard.
_ADVANCED_TILE = {
    "A": (256, 64, 32, 4, 1, 16),
    "B": (256, 64, 32, 4, 1, 16),
    "C": (128, 64, 32, 4, 1, 16),
}
# Opt-in ImplicitGemmConvSpec fields from the merged conv3d-opt branch; see
# kernels/common/conv_implicit_gemm.py for each field's semantics. Same combo
# won for all three shapes in the hand search.
_ADVANCED_KNOBS = dict(
    uniform_k_decode=True,
    reg_prefetch=1,
    lds_xor=True,
    uk_xhalo=4,
    epi_quad_xpose=3,
    xcd_remap=6,
    rpf_setprio=1,
    defer_mfma=1,
    vector_size_a=8,
    vector_size_b=8,
    vector_size_c=8,
)

# The external comparison's "rocKE time (ms)" column for each shape's F=1
# (forward) row, repo-root results.csv -- the number this prototype exists to
# beat. Printed alongside each run's own measurement, not baked into any
# pass/fail gate: results.csv is an external artifact, not part of this repo.
_FILE_ROCKE_MS = {"A": 1.88, "B": 3.751, "C": 1.788}

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
    # A shard's conv problem is a different (smaller) shape than the P=1
    # baseline's, so the tile config that is fastest for one is not
    # generally fastest for the other (confirmed empirically: a shard-sized
    # problem tuned on its own merits reaches ~80% of the full problem's
    # TFLOPS, vs ~37% when it's forced to reuse the baseline's config).
    # These default to the baseline value and only need overriding when a
    # separate sweep of the shard shape found a better config.
    ap.add_argument("--shard-tile-m", type=int, default=None)
    ap.add_argument("--shard-tile-n", type=int, default=None)
    ap.add_argument("--shard-tile-k", type=int, default=None)
    ap.add_argument("--shard-warp-m", type=int, default=None)
    ap.add_argument("--shard-warp-n", type=int, default=None)
    ap.add_argument("--shard-warp-tile-mn", type=int, default=None)
    ap.add_argument("--shard-pipeline", default=None)
    ap.add_argument("--shard-epilogue", default=None)
    ap.add_argument(
        "--advanced", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "use the per-shape _ADVANCED_TILE/_ADVANCED_KNOBS combo from the "
            "merged conv3d-opt branch instead of the plain --tile-m/etc CLI "
            "knobs (default: on; see the module docstring). --no-advanced "
            "falls back to the original plain-tile-sweep behavior."
        ),
    )
    a = ap.parse_args(argv)
    for name in (
        "tile_m", "tile_n", "tile_k", "warp_m", "warp_n",
        "warp_tile_mn", "pipeline", "epilogue",
    ):
        shard_name = f"shard_{name}"
        if getattr(a, shard_name) is None:
            setattr(a, shard_name, getattr(a, name))
    return a


def _shard_args(a):
    """View of ``a`` with the ``--shard-*`` tile/pipeline knobs in the plain
    ``tile_m``/``pipeline``/... slots, so :func:`_build_fwd_spec` and the atom
    lookup can read it exactly like the baseline args."""
    from copy import copy

    sa = copy(a)
    sa.tile_m, sa.tile_n, sa.tile_k = a.shard_tile_m, a.shard_tile_n, a.shard_tile_k
    sa.warp_m, sa.warp_n, sa.warp_tile_mn = a.shard_warp_m, a.shard_warp_n, a.shard_warp_tile_mn
    sa.pipeline, sa.epilogue = a.shard_pipeline, a.shard_epilogue
    return sa


def _build_fwd_spec(problem, a, tgt, atom, *, extra_knobs=None):
    from kernels.common.conv_implicit_gemm import (
        ConvDataSpec,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
        is_valid_spec_for_problem,
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
        **(extra_knobs or {}),
    )
    # Several --advanced knobs (uk_xhalo, epi_quad_xpose, ...) have validity
    # conditions on the problem shape (e.g. Wo divisibility), not just the
    # arch, so this checks against the concrete problem rather than
    # is_valid_spec's arch-only check.
    ok, why = is_valid_spec_for_problem(spec, problem, a.arch)
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
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

    extra_knobs = {}
    if a.advanced and shape_name in _ADVANCED_TILE:
        from copy import copy

        a = copy(a)
        tm, tn, tk, wm, wn, wt = _ADVANCED_TILE[shape_name]
        a.tile_m, a.tile_n, a.tile_k = tm, tn, tk
        a.warp_m, a.warp_n, a.warp_tile_mn = wm, wn, wt
        a.shard_tile_m, a.shard_tile_n, a.shard_tile_k = tm, tn, tk
        a.shard_warp_m, a.shard_warp_n, a.shard_warp_tile_mn = wm, wn, wt
        extra_knobs = dict(_ADVANCED_KNOBS)

    tgt = ArchTarget.from_gfx(a.arch)
    family = "wmma" if tgt.wave_size == 32 else "mma"
    atom = tgt.mma.select_largest_k(
        family=family, a_dtype=a.dtype, b_dtype=a.dtype, c_dtype="fp32",
        m=a.warp_tile_mn, n=a.warp_tile_mn, k_max=a.tile_k,
    )
    a_shard = _shard_args(a)
    atom_shard = tgt.mma.select_largest_k(
        family=family, a_dtype=a.dtype, b_dtype=a.dtype, c_dtype="fp32",
        m=a_shard.warp_tile_mn, n=a_shard.warp_tile_mn, k_max=a_shard.tile_k,
    )
    if atom is None or atom_shard is None:
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
        spec0, kernel0 = _build_fwd_spec(full_problem, a, tgt, atom, extra_knobs=extra_knobs)
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
            sp, kern = _build_fwd_spec(shard_problem, a_shard, tgt, atom_shard, extra_knobs=extra_knobs)
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
        # KernelLauncher defaults to a *fenced* (synchronous,
        # hipStreamSynchronize-after-every-call) launch -- see
        # rocke.runtime.launcher.LaunchConfig.fence. Outside no_fence(), each
        # call below would block the host until that device's kernel
        # finished before the next device's launch could even be issued,
        # serializing the P devices' compute instead of overlapping it (this
        # was confirmed to fully explain an earlier, wrong "node-level power
        # contention" diagnosis: per-device throughput looked like it halved
        # under concurrency, but it was purely this fence, not the GPUs).
        with no_fence():
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

    file_ms = _FILE_ROCKE_MS.get(shape_name)
    vs_file = f" | vs results.csv rocKE row={file_ms / wall_ms:.2f}x" if file_ms is not None else ""
    print(
        f"[{shape_name} P={P}] baseline={baseline_ms:.4f}ms "
        f"({baseline_tflops:.0f} TFLOPS) | multi-gpu wall={wall_ms:.4f}ms "
        f"(halo={halo_ms:.4f}ms, compute~{compute_ms:.4f}ms, "
        f"{per_device_tflops:.0f} TFLOPS/dev) | speedup={speedup_vs_baseline:.2f}x "
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
