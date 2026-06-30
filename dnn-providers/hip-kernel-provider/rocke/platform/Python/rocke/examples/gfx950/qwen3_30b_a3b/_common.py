# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared constants, timing helpers, and kernel-building utilities.

Imported by every numbered script in this folder. Keep this module
import-time side-effect free so scripts can import it before touching
the GPU.
"""

from __future__ import annotations

import os
import struct
import sys
from statistics import median
from typing import Callable, Optional

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKE_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
AITER_PATH = os.environ.get("AITER_PATH", "")

for _p in (ROCKE_PATH, AITER_PATH):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# ── Qwen3-30B-A3B model constants ─────────────────────────────────────────────
BATCH = 2  # decode batch (num active sequences)
NHEAD_Q = 32  # query heads
NHEAD_K = 4  # KV heads  (GQA ratio = 8)
HEAD_DIM = 64
HIDDEN = 2048  # hidden size
MOE_INTER = 768  # per-expert intermediate size
NUM_EXPERTS = 128
TOPK = 8  # experts per token
BLOCK_SIZE = 16  # paged-KV block size
DTYPE = torch.bfloat16
ISA = "amdgcn-amd-amdhsa--gfx950"

# ── Timing knobs ──────────────────────────────────────────────────────────────
WARMUP = 10  # untimed warmup launches (ensures JIT cache is hot)
ITERS = 200  # launches per timing sample (batched under one event pair)
REPEATS = 5  # independent samples; we report the median


# ── Timing helper ─────────────────────────────────────────────────────────────
def ms(
    fn: Callable, warmup: int = WARMUP, iters: int = ITERS, repeats: int = REPEATS
) -> float:
    """Return median per-launch time in **milliseconds**.

    Design rationale — why we batch iters under a single event pair
    ---------------------------------------------------------------
    torch.cuda.Event recording has ~2–5µs host-side overhead per pair
    on gfx950/ROCm 7.  For kernels shorter than ~20µs that overhead
    dominates if you record one event pair per launch.  Instead we
    bracket ``iters`` launches with a single pair and divide:

        start.record()
        for _ in range(iters): fn()
        end.record()
        torch.cuda.synchronize()
        sample = start.elapsed_time(end) / iters  # ms

    We repeat this ``repeats`` times and take the median to avoid
    one-off warm-up or OS-scheduling spikes.

    All kernels, baselines and DSL alike, use the same timer so the
    comparison is apples-to-apples.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) / iters)
    return median(samples)


def speedup(baseline_ms: float, dsl_ms: float) -> float:
    if baseline_ms != baseline_ms or dsl_ms != dsl_ms:  # NaN guard
        return float("nan")
    return baseline_ms / dsl_ms


# ── CUDA-graph capture helper ─────────────────────────────────────────────────
def capture_graph(fn: Callable, warmup: int = 3) -> Optional[torch.cuda.CUDAGraph]:
    """Capture ``fn`` (a no-arg callable that only touches preallocated
    tensors) into a HIP/CUDA graph.  Returns the graph, or None if
    capture fails.

    Background — why graph capture removes dispatch overhead
    --------------------------------------------------------
    Every GPU kernel launch goes through the HIP runtime's command
    submission path: the CPU builds a packet descriptor, writes it to
    the HSA queue, and the GPU's hardware scheduler picks it up.  On
    gfx950/ROCm 7 this costs roughly 5–8µs of CPU wall-clock time per
    kernel (measured with ``hipEventRecord`` bracketing a single
    ``rt.launch()``).  For tiny kernels (<10µs GPU time) that overhead
    is visible in the per-launch timer.

    CUDA/HIP graph capture replaces the per-launch command submission
    with a single ``hipGraphLaunch`` that replays the full recorded
    command stream.  The replay overhead is ~0.4–0.5µs regardless of
    how many kernels are in the graph, so a 3-kernel pipeline goes from
    15–24µs dispatch overhead to ~0.5µs.

    Constraint: tensor storage must not be reallocated between capture
    and replay.  Callers must preallocate all outputs and pass stable
    data pointers into ``fn``.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g
    except Exception:
        return None


# ── Low-level kernel build utilities ──────────────────────────────────────────
def _ensure_rocke() -> None:
    if ROCKE_PATH not in sys.path:
        sys.path.insert(0, ROCKE_PATH)


def build_gemm_kernel(
    M: int,
    N: int,
    K: int,
    tile_k: int = 512,
    chiplet_wgm: int = 4,
    chiplet_chunk_size: int = 64,
):
    """Compile and return a callable ``run(Ap, Bp, Cp)`` for a BF16 RCR GEMM.

    Layout: A is (M, K) row-major, B is (N, K) row-major (weight layout
    used by LinearBase), C is (M, N) row-major.  The DSL RCR layout
    means B is read as its transpose — matching ``torch.matmul(A, B.T)``.

    Tile family: DTLA (Direct-To-LDS A) + ``compv3`` memory pipeline +
    ``cshuffle`` epilogue + chiplet swizzle.  This is the family
    identified as the winner for M=2 skinny decode GEMMs in the
    ``gemm_perf_skinny_decode`` runbook (scripts 12–22).

    Returns ``(run_fn, rt, c_buf)`` where ``c_buf`` is a pre-allocated
    output tensor of shape (M, N) on CUDA.
    """
    _ensure_rocke()
    from rocke.instances.common.gemm_universal import (
        UniversalGemmSpec,
        TileSpec,
        TraitSpec,
        DataSpec,
        build_universal_gemm,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    tile = TileSpec(
        tile_m=16,
        tile_n=16,
        tile_k=tile_k,
        warp_m=1,
        warp_n=1,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = TraitSpec(
        pipeline="mem",
        scheduler="interwave",
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
        direct_to_lds=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        chiplet_swizzle=True,
        chiplet_wgm=chiplet_wgm,
        chiplet_num_xcds=8,
        chiplet_chunk_size=chiplet_chunk_size,
    )
    data = DataSpec(
        dtype_a="bf16",
        dtype_b="bf16",
        dtype_c="bf16",
        dtype_acc="fp32",
        layout="RCR",
    )
    spec = UniversalGemmSpec(
        name=f"qwen_gemm_m{M}n{N}k{K}",
        tile=tile,
        trait=trait,
        data=data,
    )
    art = compile_kernel(build_universal_gemm(spec), isa=ISA)
    bm, bn = tile.tile_m, tile.tile_n
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (spec.block_size, 1, 1)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    c_buf = torch.empty((M, N), dtype=DTYPE, device="cuda")

    def run(Ap: int, Bp: int, Cp: int) -> None:
        rt.launch(
            fn, grid, block, struct.pack("<QQQiii", Ap, Bp, Cp, M, N, K), stream=0
        )

    return run, rt, c_buf


def print_result(
    label: str, baseline_us: float, dsl_us: float, notes: str = ""
) -> None:
    spd = speedup(baseline_us / 1000, dsl_us / 1000)
    marker = "✓" if spd >= 1.0 else "✗"
    note_str = f"  [{notes}]" if notes else ""
    print(
        f"  {marker} {label:35s}  baseline={baseline_us:7.2f}µs  "
        f"dsl={dsl_us:7.2f}µs  speedup={spd:.3f}×{note_str}"
    )
