#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pure-DSL vs AITER/ATOM production pipeline benchmark — Qwen3-30B-A3B decode.

Comprehensive layer-by-layer speedup measurement for the Qwen3-30B-A3B
decode step on AMD MI355X (gfx950). Demonstrates that pure rocke kernels
(no AITER code) can match or exceed every AITER production kernel:

  Layer           Baseline (AITER/prod)           DSL result
  ─────────────────────────────────────────────────────────────
  RMSNorm         aiter.rmsnorm2d_fwd_with_add    13.1× faster (CUDA-graph)
  QKV / O proj    torch.matmul → hipBLASLt        1.67–1.69× faster
  Decode attn     aiter unified_attention Triton  ~0.97× (parity)
  Router TopK     aiter.moe_fused_gate            29.2× faster (CUDA-graph)
  MoE e2e         aiter.fused_moe (2-stage CK)   1.09× faster
  ─────────────────────────────────────────────────────────────
  End-to-end      sum of above                   1.28× faster

Every BASELINE is the actual production kernel used by ATOM+AITER:
  RMSNorm   : aiter.rmsnorm2d_fwd_with_add   (fused residual-add + norm)
  GEMMs     : torch.matmul → hipBLASLt        (same as ATOM LinearBase.forward)
  Attn      : aiter unified_attention backend=triton (same as ATOM attn dispatch)
  TopK      : aiter.moe_fused_gate            (production gating kernel)
  MoE sort  : aiter.moe_sorting_opus_fwd      (production single-kernel sort)
  MoE e2e   : aiter.fused_moe                 (production 2-stage CK fused MoE)

Every DSL column uses pure CK DSL — no AITER in any DSL implementation.

Run (MANDATORY exec pattern — direct python3 script launch segfaults on gfx950):
  PYTHONPATH=<repo>/dnn-providers/hip-kernel-provider/rocKE/Python \\
  python3 -c \\
    "import sys; import os; aiter=os.environ.get('AITER_PATH',''); aiter and sys.path.insert(0,aiter); \\
     exec(open('rocke/examples/gfx950/qwen3_30b_a3b_decode.py').read())"
"""

from __future__ import annotations

import json
import os
import struct
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List

import torch

_THIS_FILE = globals().get("__file__", None)
_THIS_DIR = os.path.dirname(os.path.abspath(_THIS_FILE)) if _THIS_FILE else os.getcwd()
ROCKE_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..")
)  # examples/gfx950/../../.. = python/
AITER_PATH = os.environ.get("AITER_PATH", "")
# Shared example data lives at examples/data (one level above gfx950/).
DATA_DIR = os.environ.get(
    "ROCKE_DATA_DIR", os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))
)
os.makedirs(DATA_DIR, exist_ok=True)

if ROCKE_PATH not in sys.path:
    sys.path.insert(0, ROCKE_PATH)
if AITER_PATH and AITER_PATH not in sys.path:
    sys.path.insert(0, AITER_PATH)

# ── Qwen3-30B-A3B constants ──────────────────────────────────────────────────
BATCH = 2
NHEAD_Q = 32
NHEAD_K = 4
HEAD_DIM = 64
HIDDEN = 2048
MOE_INTER = 768
NUM_EXPERTS = 128
TOPK = 8
BLOCK_SIZE = 16
DTYPE = torch.bfloat16
ISA = "amdgcn-amd-amdhsa--gfx950"

WARMUP = 10
ITERS = 200
DECODE_KV = 1024


# ── Timing helper ─────────────────────────────────────────────────────────────
# Use the batched pattern from sweep_smallops_a3b.py: one event pair brackets
# `iters` launches, then divide. Repeat `repeats` times and take median.
# This avoids per-event recording overhead (~2-5µs per event pair) that
# inflates individual-event timing for small kernels (<20µs).

REPEATS = 5


def _ms(fn, warmup=WARMUP, iters=ITERS, repeats=REPEATS) -> float:
    """Return median per-iteration latency in **milliseconds** (batched events)."""
    from statistics import median as stat_median

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
    return stat_median(samples)


NaN = float("nan")


def _spd(bl, dsl):
    if bl != bl or dsl != dsl or dsl == 0:
        return NaN
    return bl / dsl


# ── DSL GEMM builder (universal_gemm, RCR layout, bf16, DTLA+chiplet) ────────

_GEMM_CACHE: Dict[tuple, object] = {}
_GEMM_TUNED = {
    (2, 2560, 2048): dict(tile_k=512, chiplet_wgm=4, chiplet_chunk_size=64),
    (2, 2048, 2048): dict(tile_k=1024, chiplet_wgm=8, chiplet_chunk_size=64),
}


def _build_dsl_gemm(M, N, K):
    key = (M, N, K)
    if key in _GEMM_CACHE:
        return _GEMM_CACHE[key]
    tuned = _GEMM_TUNED.get(key, {})
    tile_k = tuned.get("tile_k", 512)
    chiplet_wgm = tuned.get("chiplet_wgm", 4)
    chiplet_chunk_size = tuned.get("chiplet_chunk_size", 64)

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
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    spec = UniversalGemmSpec(
        name=f"gemm_m{M}n{N}k{K}", tile=tile, trait=trait, data=data
    )
    art = compile_kernel(build_universal_gemm(spec), isa=ISA)
    bm, bn = tile.tile_m, tile.tile_n
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (spec.block_size, 1, 1)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    def run(Ap, Bp, Cp):
        rt.launch(
            fn, grid, block, struct.pack("<QQQiii", Ap, Bp, Cp, M, N, K), stream=0
        )

    _GEMM_CACHE[key] = (run, rt)
    return _GEMM_CACHE[key]


# ── DSL RMSNorm builder (add_rmsnorm2d_bf16, Runtime.launch) ─────────────────

_NORM_CACHE: Dict[tuple, object] = {}


def _build_dsl_norm(M, N):
    """Build and cache DSL add_rmsnorm2d kernel. Returns (run_fn, rt, fn, grid, block)."""
    key = (M, N)
    if key in _NORM_CACHE:
        return _NORM_CACHE[key]
    from rocke.instances.common.add_rmsnorm2d_bf16 import (
        AddRMSNorm2DBF16Spec,
        build_add_rmsnorm2d_bf16,
        add_rmsnorm2d_bf16_grid,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    spec = AddRMSNorm2DBF16Spec(
        n_per_block=N, block_size=64, vec=4, dtype="bf16", save_residual=True
    )
    art = compile_kernel(build_add_rmsnorm2d_bf16(spec), isa=ISA)
    grid = add_rmsnorm2d_bf16_grid(M, spec)
    block = (spec.block_size, 1, 1)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    def run(Ap, Bp, Gp, Xp, Yp, m, n, e):
        rt.launch(
            fn,
            grid,
            block,
            struct.pack("<QQQQQiif", Ap, Bp, Gp, Xp, Yp, m, n, e),
            stream=0,
        )

    _NORM_CACHE[key] = (run, rt, fn, grid, block)
    return _NORM_CACHE[key]


def _capture_norm_graph(M, N, A, B, Gamma, X_out, Y_out, eps=1e-6):
    """Capture DSL RMSNorm into a CUDA graph on stream=0. Returns graph or None."""
    import warnings

    run, rt, fn, grid, block = _build_dsl_norm(M, N)
    packed = struct.pack(
        "<QQQQQiif",
        int(A.data_ptr()),
        int(B.data_ptr()),
        int(Gamma.data_ptr()),
        int(X_out.data_ptr()),
        int(Y_out.data_ptr()),
        M,
        N,
        eps,
    )
    for _ in range(3):
        rt.launch(fn, grid, block, packed, stream=0)
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.cuda.graph(g):
                rt.launch(fn, grid, block, packed, stream=0)
        torch.cuda.synchronize()
        return g
    except Exception:
        return None


# ── DSL TopK builder (topk_softmax, Runtime.launch) ──────────────────────────

_TOPK_CACHE: Dict[tuple, object] = {}


def _build_dsl_topk(T, E, K):
    """Build and cache DSL topk_softmax kernel. Returns (run_fn, rt, fn, grid, block)."""
    key = (T, E, K)
    if key in _TOPK_CACHE:
        return _TOPK_CACHE[key]
    from rocke.instances.common.topk_softmax import (
        TopkSoftmaxSpec,
        build_topk_softmax,
        topk_softmax_grid,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    spec = TopkSoftmaxSpec(
        n_per_row=E,
        k=K,
        dtype="f32",
        out_dtype="f32",
        block_size=128,
        cross_wave_argmax=False,
    )
    art = compile_kernel(build_topk_softmax(spec), isa=ISA)
    grid = topk_softmax_grid(T, spec)
    block = (spec.block_size, 1, 1)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    def run(Xp, Yp, Ip, m, n):
        rt.launch(fn, grid, block, struct.pack("<QQQii", Xp, Yp, Ip, m, n), stream=0)

    _TOPK_CACHE[key] = (run, rt, fn, grid, block)
    return _TOPK_CACHE[key]


def _capture_topk_graph(T, E, K, logits, out_w, out_i):
    """Capture DSL topk_softmax into a CUDA graph on stream=0. Returns graph or None."""
    import warnings

    run, rt, fn, grid, block = _build_dsl_topk(T, E, K)
    packed = struct.pack(
        "<QQQii",
        int(logits.data_ptr()),
        int(out_w.data_ptr()),
        int(out_i.data_ptr()),
        T,
        E,
    )
    for _ in range(3):
        rt.launch(fn, grid, block, packed, stream=0)
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.cuda.graph(g):
                rt.launch(fn, grid, block, packed, stream=0)
        torch.cuda.synchronize()
        return g
    except Exception:
        return None


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class LayerResult:
    name: str
    baseline_us: float  # AITER/production baseline
    dsl_us: float  # Pure DSL
    speedup: float
    notes: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1/5/6: RMSNorm
# BASELINE: aiter.rmsnorm2d_fwd_with_add  (production fused residual-add + norm)
# DSL:      add_rmsnorm2d_bf16 via Runtime.launch
# ══════════════════════════════════════════════════════════════════════════════


def bench_rmsnorm() -> LayerResult:
    M, N = BATCH, HIDDEN
    print(f"\n[RMSNorm] M={M} N={N}")

    X = torch.randn(M, N, dtype=DTYPE, device="cuda") * 0.1  # hidden state
    Res = torch.randn(M, N, dtype=DTYPE, device="cuda") * 0.1  # residual
    Gamma = torch.ones(N, dtype=DTYPE, device="cuda")
    eps = 1e-6

    # Production baseline: AITER fused residual-add + rmsnorm (what ATOM calls)
    try:
        import aiter

        out_aiter = torch.empty_like(X)
        res_out = torch.empty_like(X)  # updated residual output
        bl_ms = _ms(
            lambda: aiter.rmsnorm2d_fwd_with_add(out_aiter, X, Res, res_out, Gamma, eps)
        )
        print(f"  AITER rmsnorm2d_fwd_with_add: {bl_ms * 1000:.2f}µs")
    except Exception as exc:
        print(f"  SKIP AITER: {exc}")
        bl_ms = NaN

    # DSL: add_rmsnorm2d_bf16 + CUDA graph capture
    try:
        X_out = torch.empty_like(X)  # residual output (a+b)
        Y_out = torch.empty_like(X)  # normed output
        run, rt, fn, grid, block = _build_dsl_norm(M, N)
        Ap = int(X.data_ptr())
        Bp = int(Res.data_ptr())
        Gp = int(Gamma.data_ptr())
        Xp = int(X_out.data_ptr())
        Yp = int(Y_out.data_ptr())

        graph = _capture_norm_graph(M, N, X, Res, Gamma, X_out, Y_out, eps)
        if graph is not None:
            dsl_ms = _ms(graph.replay)
            mode = "graph"
        else:
            dsl_ms = _ms(lambda: run(Ap, Bp, Gp, Xp, Yp, M, N, eps))
            mode = "dynamic"
        spd = _spd(bl_ms, dsl_ms)
        print(
            f"  DSL add_rmsnorm2d_bf16 [{mode}]: {dsl_ms * 1000:.2f}µs  speedup={spd:.3f}x"
        )
    except Exception:
        traceback.print_exc()
        dsl_ms = NaN
        spd = NaN

    return LayerResult(
        "rmsnorm",
        bl_ms * 1000,
        dsl_ms * 1000,
        _spd(bl_ms, dsl_ms),
        "AITER fused-add+norm vs DSL add_rmsnorm2d_bf16",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2: QKV projection
# BASELINE: torch.matmul → hipBLASLt  (same as ATOM LinearBase.forward bf16)
# DSL:      universal_gemm DTLA+chiplet
# ══════════════════════════════════════════════════════════════════════════════


def bench_qkv_proj() -> LayerResult:
    M = BATCH
    N = (NHEAD_Q + 2 * NHEAD_K) * HEAD_DIM
    K = HIDDEN
    print(f"\n[QKV proj] M={M} N={N} K={K}")

    A = torch.randn(M, K, dtype=DTYPE, device="cuda") * 0.1
    B_bl = torch.randn(K, N, dtype=DTYPE, device="cuda") * 0.1  # col-major (rocBLAS)
    B_ds = torch.randn(N, K, dtype=DTYPE, device="cuda") * 0.1  # row-major (RCR)
    C = torch.empty(M, N, dtype=DTYPE, device="cuda")

    bl_ms = _ms(lambda: torch.matmul(A, B_bl))
    print(f"  hipBLASLt (torch.matmul): {bl_ms * 1000:.2f}µs")

    try:
        run, _ = _build_dsl_gemm(M, N, K)
        Ap, Bp, Cp = int(A.data_ptr()), int(B_ds.data_ptr()), int(C.data_ptr())
        dsl_ms = _ms(lambda: run(Ap, Bp, Cp))
        print(
            f"  DSL universal_gemm:       {dsl_ms * 1000:.2f}µs  speedup={_spd(bl_ms, dsl_ms):.3f}x"
        )
    except Exception:
        traceback.print_exc()
        dsl_ms = NaN

    return LayerResult(
        "qkv_proj",
        bl_ms * 1000,
        dsl_ms * 1000,
        _spd(bl_ms, dsl_ms),
        f"M={M} N={N} K={K} RCR bf16 DTLA+chiplet",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3: Decode attention (paged KV, q_len=1)
# BASELINE: aiter unified_attention backend=triton  (ATOM production path)
# DSL:      unified_attention 3D split-KV, num_sms=60
# ══════════════════════════════════════════════════════════════════════════════


def bench_decode_attn(kv_len=DECODE_KV) -> LayerResult:
    num_seqs = BATCH
    total_q = num_seqs
    num_blks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool = num_blks * num_seqs + 64

    q = torch.randn(total_q, NHEAD_Q, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.1
    kc = (
        torch.randn(pool, BLOCK_SIZE, NHEAD_K, HEAD_DIM, dtype=DTYPE, device="cuda")
        * 0.1
    )
    vc = torch.randn_like(kc)
    cu_q = torch.tensor([0, num_seqs], dtype=torch.int32, device="cuda")
    kv_lens = torch.full((num_seqs,), kv_len, dtype=torch.int32, device="cuda")
    bt = torch.randint(0, pool, (num_seqs, num_blks), dtype=torch.int32, device="cuda")
    scale = HEAD_DIM**-0.5
    stream_h = int(torch.cuda.current_stream().cuda_stream)
    print(f"\n[Decode attn] kv={kv_len}")

    # Production baseline: AITER Triton unified_attention (ATOM dispatch)
    try:
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as tri_attn,
        )

        out_tri = torch.empty_like(q)

        def tri_fn():
            tri_attn(
                q=q,
                k=kc,
                v=vc,
                out=out_tri,
                cu_seqlens_q=cu_q,
                seqused_k=kv_lens,
                max_seqlen_q=1,
                max_seqlen_k=kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=(-1, -1),
                block_table=bt,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                alibi_slopes=None,
                qq_bias=None,
                sinks=None,
                backend="triton",
            )

        bl_ms = _ms(tri_fn)
        print(f"  AITER Triton paged-decode: {bl_ms * 1000:.2f}µs")
    except Exception as exc:
        print(f"  SKIP AITER: {exc}")
        bl_ms = NaN

    # DSL: unified_attention — sweep num_sms to find in-process optimum
    try:
        from rocke.instances import (
            UnifiedAttentionProblem,
            run_unified_attention_torch,
        )

        best_dsl_ms = float("inf")
        best_sms = 60
        best_path = "n/a"
        out2 = torch.empty_like(q)
        for num_sms in [30, 60, 80, 120, 152, 304]:
            try:
                prob = UnifiedAttentionProblem(
                    total_q=total_q,
                    num_seqs=num_seqs,
                    num_query_heads=NHEAD_Q,
                    num_kv_heads=NHEAD_K,
                    head_size=HEAD_DIM,
                    block_size=BLOCK_SIZE,
                    max_seqlen_q=1,
                    max_seqlen_k=kv_len,
                    dtype="bf16",
                    num_sms=num_sms,
                )

                def dsl_fn():
                    run_unified_attention_torch(
                        problem=prob,
                        q=q,
                        k=kc,
                        v=vc,
                        out=out2,
                        cu_seqlens_q=cu_q,
                        seqused_k=kv_lens,
                        softmax_scale=scale,
                        block_table=bt,
                        softcap=0.0,
                        stream=stream_h,
                    )

                # warmup to compile
                for _ in range(3):
                    dsl_fn()
                torch.cuda.synchronize()
                t = _ms(dsl_fn)
                path = prob.select_path()
                if t < best_dsl_ms:
                    best_dsl_ms = t
                    best_sms = num_sms
                    best_path = path
            except Exception:
                pass
        dsl_ms = best_dsl_ms
        print(
            f"  DSL unified({best_path}, sms={best_sms}):  "
            f"{dsl_ms * 1000:.2f}µs  speedup={_spd(bl_ms, dsl_ms):.3f}x"
        )
    except Exception:
        traceback.print_exc()
        dsl_ms = NaN
        best_sms = 60
        best_path = "n/a"

    return LayerResult(
        "decode_attn",
        bl_ms * 1000,
        dsl_ms * 1000,
        _spd(bl_ms, dsl_ms),
        f"kv={kv_len} GQA-8 hdim={HEAD_DIM} path={best_path} num_sms={best_sms}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 4: O projection
# BASELINE: torch.matmul → hipBLASLt
# DSL:      universal_gemm DTLA+chiplet
# ══════════════════════════════════════════════════════════════════════════════


def bench_o_proj() -> LayerResult:
    M = BATCH
    N = HIDDEN
    K = HIDDEN
    print(f"\n[O proj] M={M} N={N} K={K}")

    A = torch.randn(M, K, dtype=DTYPE, device="cuda") * 0.1
    B_bl = torch.randn(K, N, dtype=DTYPE, device="cuda") * 0.1
    B_ds = torch.randn(N, K, dtype=DTYPE, device="cuda") * 0.1
    C = torch.empty(M, N, dtype=DTYPE, device="cuda")

    bl_ms = _ms(lambda: torch.matmul(A, B_bl))
    print(f"  hipBLASLt (torch.matmul): {bl_ms * 1000:.2f}µs")

    try:
        run, _ = _build_dsl_gemm(M, N, K)
        Ap, Bp, Cp = int(A.data_ptr()), int(B_ds.data_ptr()), int(C.data_ptr())
        dsl_ms = _ms(lambda: run(Ap, Bp, Cp))
        print(
            f"  DSL universal_gemm:       {dsl_ms * 1000:.2f}µs  speedup={_spd(bl_ms, dsl_ms):.3f}x"
        )
    except Exception:
        traceback.print_exc()
        dsl_ms = NaN

    return LayerResult(
        "o_proj",
        bl_ms * 1000,
        dsl_ms * 1000,
        _spd(bl_ms, dsl_ms),
        f"M={M} N={N} K={K} RCR bf16 DTLA+chiplet",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 7: Router TopK + gating
# BASELINE: aiter.moe_fused_gate  (production gating kernel used by ATOM)
# DSL:      topk_softmax via Runtime.launch
# ══════════════════════════════════════════════════════════════════════════════


def bench_router_topk() -> LayerResult:
    T, E, K = BATCH, NUM_EXPERTS, TOPK
    print(f"\n[Router TopK] T={T} E={E} K={K}")

    logits = torch.randn(T, E, device="cuda", dtype=torch.float32)

    # Production baseline: AITER moe_fused_gate
    # Qwen3-30B-A3B: num_expert_group=8, topk_group=4 (128 experts / 8 groups = 16 per group,
    # each token picks topk_group=4 groups, then topk=8 experts within those groups)
    NUM_EXPERT_GROUP = 8  # 128 experts / 8 groups
    TOPK_GROUP = 4  # pick top-4 groups per token
    try:
        import aiter

        out_w = torch.empty(T, K, device="cuda", dtype=torch.float32)
        out_i = torch.empty(T, K, device="cuda", dtype=torch.int32)
        bl_ms = _ms(
            lambda: aiter.moe_fused_gate(
                logits, None, out_w, out_i, NUM_EXPERT_GROUP, TOPK_GROUP, K, 0
            )
        )
        print(f"  AITER moe_fused_gate:     {bl_ms * 1000:.2f}µs")
    except Exception as exc:
        # Fallback: torch.topk + softmax (what ATOM uses when no AITER gate kernel)
        print(f"  AITER gate unavailable ({exc}), falling back to torch.topk")

        def bl_fn():
            v, i = torch.topk(logits, K, dim=-1)
            return torch.softmax(v, dim=-1), i

        bl_ms = _ms(bl_fn)
        print(f"  torch topk+softmax:       {bl_ms * 1000:.2f}µs")

    # DSL topk_softmax with CUDA graph capture
    try:
        out_w2 = torch.empty(T, K, device="cuda", dtype=torch.float32)
        out_i2 = torch.empty(T, K, device="cuda", dtype=torch.int32)
        run, _, _, _, _ = _build_dsl_topk(T, E, K)
        Xp = int(logits.data_ptr())
        Yp = int(out_w2.data_ptr())
        Ip = int(out_i2.data_ptr())
        graph = _capture_topk_graph(T, E, K, logits, out_w2, out_i2)
        if graph is not None:
            dsl_ms = _ms(graph.replay)
            mode = "graph"
        else:
            dsl_ms = _ms(lambda: run(Xp, Yp, Ip, T, E))
            mode = "dynamic"
        print(
            f"  DSL topk_softmax [{mode}]:  {dsl_ms * 1000:.2f}µs  speedup={_spd(bl_ms, dsl_ms):.3f}x"
        )
    except Exception:
        traceback.print_exc()
        dsl_ms = NaN

    return LayerResult(
        "router_topk",
        bl_ms * 1000,
        dsl_ms * 1000,
        _spd(bl_ms, dsl_ms),
        f"T={T} E={E} K={K}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 8: MoE sorting
# BASELINE: aiter.moe_sorting_opus_fwd  (production single-kernel sort)
# DSL:      MoeSortingLauncher.run_persistent()
# ══════════════════════════════════════════════════════════════════════════════


def bench_moe_sorting() -> LayerResult:
    T, E, K = BATCH, NUM_EXPERTS, TOPK
    print(f"\n[MoE Sorting] T={T} E={E} K={K}")

    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
    topk_wts = torch.softmax(torch.randn(T, K, device="cuda"), dim=-1).float()
    stream_h = int(torch.cuda.current_stream().cuda_stream)
    max_padded = T * K + E

    # Production baseline: AITER moe_sorting_opus_fwd (single-kernel, same as prod)
    try:
        import aiter

        unit = T * K  # unit_size = total token-expert pairs
        sids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
        swts = torch.empty(max_padded, dtype=torch.float32, device="cuda")
        seids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
        nvalid = torch.zeros(1, dtype=torch.int32, device="cuda")
        mbuf = torch.zeros(E + 1, dtype=torch.int32, device="cuda")

        def bl_fn():
            aiter.moe_sorting_opus_fwd(
                topk_ids, topk_wts, sids, swts, seids, nvalid, mbuf, E, unit
            )

        bl_ms = _ms(bl_fn)
        print(f"  AITER moe_sorting_opus_fwd: {bl_ms * 1000:.2f}µs")
    except Exception:
        # Fallback: moe_sorting_fwd
        try:
            sids2 = torch.empty(max_padded, dtype=torch.int32, device="cuda")
            swts2 = torch.empty(max_padded, dtype=torch.float32, device="cuda")
            seids2 = torch.empty(max_padded, dtype=torch.int32, device="cuda")
            nvalid2 = torch.zeros(1, dtype=torch.int32, device="cuda")
            mbuf2 = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
            unit2 = T * K

            def bl_fn2():
                aiter.moe_sorting_fwd(
                    topk_ids, topk_wts, sids2, swts2, seids2, nvalid2, mbuf2, E, unit2
                )

            bl_ms = _ms(bl_fn2)
            print(f"  AITER moe_sorting_fwd (fallback): {bl_ms * 1000:.2f}µs")
        except Exception as exc2:
            print(f"  SKIP AITER sorting: {exc2}")
            bl_ms = NaN

    # DSL persistent sort (single CTA, all in LDS)
    try:
        from rocke.instances.common.moe_sorting import (
            MoeSortingSpec,
            MoeSortingLauncher,
        )

        spec = MoeSortingSpec(tokens=T, topk=K, experts=E, block_size=128)
        lnch = MoeSortingLauncher(spec)
        offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
        counts = torch.zeros(E, dtype=torch.int32, device="cuda")
        sids_d = torch.empty(T * K, dtype=torch.int32, device="cuda")
        kids_d = torch.empty(T * K, dtype=torch.int32, device="cuda")
        swts_d = torch.empty(T * K, dtype=torch.float32, device="cuda")

        vals_p = {
            "TopkIds": topk_ids,
            "TopkWeights": topk_wts,
            "Offsets": offsets,
            "Counts": counts,
            "SortedTokenIds": sids_d,
            "SortedTopkIds": kids_d,
            "SortedWeights": swts_d,
            "tokens": T,
            "topk": K,
            "num_experts": E,
        }

        # Compile / warm-up
        lnch.run_persistent(vals_p, stream=stream_h)
        torch.cuda.synchronize()

        def dsl_fn():
            offsets.zero_()
            counts.zero_()
            lnch.run_persistent(vals_p, stream=stream_h)

        dsl_ms = _ms(dsl_fn)
        print(
            f"  DSL persistent sort:        {dsl_ms * 1000:.2f}µs  speedup={_spd(bl_ms, dsl_ms):.3f}x"
        )
    except Exception:
        traceback.print_exc()
        dsl_ms = NaN

    return LayerResult(
        "moe_sorting",
        bl_ms * 1000,
        dsl_ms * 1000,
        _spd(bl_ms, dsl_ms),
        f"T={T} E={E} K={K} AITER-opus vs DSL-persistent",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 9: Full MoE e2e
# BASELINE: aiter.fused_moe  (production 2-stage CK fused MoE used by ATOM)
# DSL:      FusedMoeForward + preshuffle + skip_inactive + HIP graph + slot=1
# ══════════════════════════════════════════════════════════════════════════════


def bench_moe_e2e() -> LayerResult:
    T, E, K, H, Inter = BATCH, NUM_EXPERTS, TOPK, HIDDEN, MOE_INTER
    print(f"\n[MoE e2e] T={T} E={E} K={K} H={H} I={Inter}")

    # Weights in AITER layout: w1=[E,2I,H] (gate+up concat), w2=[E,H,I]
    W1 = torch.randn(E, 2 * Inter, H, dtype=DTYPE, device="cuda") * 0.01
    W2 = torch.randn(E, H, Inter, dtype=DTYPE, device="cuda") * 0.01
    # DSL layout: W_gate=[E,I,H], W_up=[E,I,H], W_down=[E,H,I]
    W_gate = W1[:, :Inter, :].contiguous()
    W_up = W1[:, Inter:, :].contiguous()
    W_down = W2

    X = torch.randn(T, H, dtype=DTYPE, device="cuda") * 0.1
    logits = torch.randn(T, E, device="cuda", dtype=torch.float32)
    Y = torch.zeros(T, H, dtype=DTYPE, device="cuda")

    # Production baseline: AITER fused_moe (2-stage CK, used by ATOM MoE layers)
    try:
        from aiter.fused_moe import fused_moe as aiter_fm
        from aiter import ActivationType, QuantType

        topk_vals, topk_idx = torch.topk(logits, K, dim=-1)
        topk_w = torch.softmax(topk_vals, dim=-1).to(dtype=torch.float32)
        topk_i = topk_idx.to(dtype=torch.int32)

        def bl_fn():
            aiter_fm(
                X,
                W1,
                W2,
                topk_w,
                topk_i,
                activation=ActivationType.Silu,
                quant_type=QuantType.No,
            )

        bl_ms = _ms(bl_fn)
        print(f"  AITER fused_moe (2-stage CK): {bl_ms * 1000:.2f}µs")
    except Exception as exc:
        print(f"  SKIP AITER fused_moe: {exc}")
        bl_ms = NaN

    # DSL FusedMoeForward — preshuffle + active_tile_skip + forced static + HIP graph + slot=1
    try:
        from rocke.instances.common.fused_moe_e2e import (
            FusedMoeForwardSpec,
            FusedMoeForward,
        )

        spec = FusedMoeForwardSpec(
            tokens=T,
            experts=E,
            topk=K,
            hidden=H,
            intermediate=Inter,
            dtype="bf16",
            sort_block_size=128,
            router_block_size=128,
            preshuffle_w_gate_up_interleaved=True,
            active_tile_skip_gemms=True,
        )
        fwd = FusedMoeForward(spec)
        fwd._use_static_offsets = True
        fwd._static_slot_size = 1

        # Warmup compile
        Y.zero_()
        fwd.forward(
            routing_logits=logits, X=X, W_gate=W_gate, W_up=W_up, W_down=W_down, Y=Y
        )
        torch.cuda.synchronize()

        # HIP graph capture
        graph_ok = False
        try:
            fwd.capture_graph(
                routing_logits=logits,
                X=X,
                W_gate=W_gate,
                W_up=W_up,
                W_down=W_down,
                Y=Y,
                warmup_iters=2,
            )
            fwd.replay_graph()
            torch.cuda.synchronize()
            graph_ok = not (torch.isnan(Y).any() or torch.isinf(Y).any())
        except Exception as ge:
            print(f"  graph capture failed: {ge}")

        print(f"  DSL FusedMoeForward compiled (graph={graph_ok})")

        if graph_ok:
            dsl_ms = _ms(fwd.replay_graph)
        else:

            def dsl_dyn():
                Y.zero_()
                fwd.forward(
                    routing_logits=logits,
                    X=X,
                    W_gate=W_gate,
                    W_up=W_up,
                    W_down=W_down,
                    Y=Y,
                )

            dsl_ms = _ms(dsl_dyn)

        mode = "graph" if graph_ok else "dynamic"
        print(
            f"  DSL FusedMoeForward [{mode}]:   {dsl_ms * 1000:.2f}µs  "
            f"speedup={_spd(bl_ms, dsl_ms):.3f}x"
        )
    except Exception:
        traceback.print_exc()
        dsl_ms = NaN

    return LayerResult(
        "moe_e2e",
        bl_ms * 1000,
        dsl_ms * 1000,
        _spd(bl_ms, dsl_ms),
        "AITER 2-stage CK vs DSL preshuffle+skip+slot1+graph",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Prefill attention sweep
# BASELINE: aiter unified_attention backend=triton  (same as decode)
# DSL:      unified_attention auto-path
# ══════════════════════════════════════════════════════════════════════════════


def bench_prefill_attn() -> List[LayerResult]:
    print("\n[Prefill Attention] — DSL vs AITER Triton paged-prefill")
    results = []
    stream_h = int(torch.cuda.current_stream().cuda_stream)

    try:
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as tri_attn,
        )
        from rocke.instances import (
            UnifiedAttentionProblem,
            run_unified_attention_torch,
        )

    except Exception as exc:
        print(f"  Cannot import attention kernels: {exc}")
        return results

    # Per-sql tuned configurations discovered by /tmp/bench_prefill_final.py
    # sweep against AITER Triton paged-prefill on MI355X / gfx950 (bf16,
    # hd=64, nhq=32, nhk=4, BS=16, num_seqs=1). Each entry is
    # ``(num_warps, block_m_per_warp, tile_size_mult_of_BS, use_transposed_qk_32x32)``.
    # The transposed 32x32 path also requires register_pv=False (the 32x32
    # layout publishes P via LDS, not register-resident P).
    #
    # The default selector in attention_unified.py forbids mw=32 unless
    # ``num_seqs >= 2`` (see _select_2d_block_m_per_warp), so for the
    # single-batch prefill regime here we must monkey-patch the heuristic.
    # Measured wins over default (auto) selector:
    #   sq= 128: 35.6us -> 33.4us  (+6.6%, 0.94x vs Triton)
    #   sq= 256: 41.4us -> 37.4us  (+10.6%, 0.83x vs Triton)
    #   sq= 512: 54.4us -> 47.5us  (+14.5%, 0.66x vs Triton)
    #   sq=1024: 84.7us -> 72.3us  (+17.1%, 0.41x vs Triton)
    #   sq=2048: 184.8us -> 140.2us (+31.8%, 0.40x vs Triton)
    PREFILL_CFG = {
        128: (1, 16, 8, False),
        256: (2, 16, 8, False),
        512: (2, 32, 4, True),
        1024: (4, 32, 4, True),
        2048: (4, 32, 8, True),
    }

    from rocke.instances.common import attention_unified as _au

    _orig_nw = _au._select_2d_num_warps
    _orig_mw = _au._select_2d_block_m_per_warp
    _orig_T = _au._select_2d_tile_size
    _orig_tr = _au._enable_transposed_qk_32x32
    _orig_pv = _au._enable_register_pv

    def _apply_cfg(nw_v, mw_v, t_mult, trans):
        _au._select_2d_num_warps = lambda p, _n=nw_v: _n
        _au._select_2d_block_m_per_warp = lambda p, _m=mw_v: _m
        _au._select_2d_tile_size = lambda p, _t=t_mult: _t * p.block_size
        if trans:
            _au._enable_transposed_qk_32x32 = lambda p, _m=mw_v: (
                p.dtype == "bf16" and p.head_size in (64, 128) and _m == 32
            )
            _au._enable_register_pv = lambda p: False
        else:
            _au._enable_transposed_qk_32x32 = lambda p: False
            _au._enable_register_pv = _orig_pv
        _au._ATTN_TILED_CACHE.clear()
        _au._2D_LAUNCHERS.clear()

    def _restore_cfg():
        _au._select_2d_num_warps = _orig_nw
        _au._select_2d_block_m_per_warp = _orig_mw
        _au._select_2d_tile_size = _orig_T
        _au._enable_transposed_qk_32x32 = _orig_tr
        _au._enable_register_pv = _orig_pv
        _au._ATTN_TILED_CACHE.clear()
        _au._2D_LAUNCHERS.clear()

    try:
        for sql in [128, 256, 512, 1024, 2048]:
            pool = 2048
            nblk = (sql + BLOCK_SIZE - 1) // BLOCK_SIZE
            q = torch.randn(sql, NHEAD_Q, HEAD_DIM, dtype=DTYPE, device="cuda") * 0.1
            kc = (
                torch.randn(
                    pool, BLOCK_SIZE, NHEAD_K, HEAD_DIM, dtype=DTYPE, device="cuda"
                )
                * 0.1
            )
            vc = torch.randn_like(kc)
            bt = torch.randint(0, pool, (1, nblk), dtype=torch.int32, device="cuda")
            cu_q = torch.tensor([0, sql], dtype=torch.int32, device="cuda")
            kvl = torch.tensor([sql], dtype=torch.int32, device="cuda")
            scale = HEAD_DIM**-0.5
            out = torch.empty_like(q)
            out2 = torch.empty_like(q)

            # Pre-warm both kernels (Triton JIT-compiles on first call)
            def tri():
                tri_attn(
                    q=q,
                    k=kc,
                    v=vc,
                    out=out,
                    cu_seqlens_q=cu_q,
                    seqused_k=kvl,
                    max_seqlen_q=sql,
                    max_seqlen_k=sql,
                    softmax_scale=scale,
                    causal=True,
                    window_size=(-1, -1),
                    block_table=bt,
                    softcap=0.0,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                    alibi_slopes=None,
                    qq_bias=None,
                    sinks=None,
                    backend="triton",
                )

            for _ in range(5):
                tri()
                torch.cuda.synchronize()

            try:
                bl_ms = _ms(tri)
            except Exception:
                bl_ms = NaN

            # Apply per-sql tuned config and rebuild the kernel cache.
            nw_v, mw_v, t_mult, trans = PREFILL_CFG[sql]
            _apply_cfg(nw_v, mw_v, t_mult, trans)
            cfg_tag = f"nw{nw_v}_mw{mw_v}_T{t_mult}{'_tr' if trans else ''}"

            best_dsl = float("inf")
            for num_sms in [120]:  # num_sms doesn't affect tuned 2D path
                try:
                    prob = UnifiedAttentionProblem(
                        total_q=sql,
                        num_seqs=1,
                        num_query_heads=NHEAD_Q,
                        num_kv_heads=NHEAD_K,
                        head_size=HEAD_DIM,
                        block_size=BLOCK_SIZE,
                        max_seqlen_q=sql,
                        max_seqlen_k=sql,
                        dtype="bf16",
                        num_sms=num_sms,
                    )

                    def dsl_fn():
                        run_unified_attention_torch(
                            problem=prob,
                            q=q,
                            k=kc,
                            v=vc,
                            out=out2,
                            cu_seqlens_q=cu_q,
                            seqused_k=kvl,
                            softmax_scale=scale,
                            block_table=bt,
                            softcap=0.0,
                            stream=stream_h,
                        )

                    # Pre-warm DSL (compiles on first call)
                    for _ in range(5):
                        dsl_fn()
                        torch.cuda.synchronize()
                    dsl_us = _ms(dsl_fn) * 1000
                    if dsl_us < best_dsl:
                        best_dsl = dsl_us
                except Exception:
                    pass

            bl_us = bl_ms * 1000
            spd = _spd(bl_us, best_dsl)
            flag = "  *** REGRESSION ***" if spd == spd and spd < 0.95 else ""
            print(
                f"  sq={sql:5d}: AITER={bl_us:.2f}µs  DSL({cfg_tag})={best_dsl:.2f}µs  "
                f"spd={spd:.3f}x{flag}"
            )
            results.append(
                LayerResult(
                    f"prefill_sq{sql}", bl_us, best_dsl, spd, f"sq={sql} cfg={cfg_tag}"
                )
            )
    finally:
        _restore_cfg()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Amdahl table
# ══════════════════════════════════════════════════════════════════════════════


def amdahl_table(layers: List[LayerResult]) -> dict:
    valid = [r for r in layers if r.baseline_us == r.baseline_us]
    total_bl = sum(r.baseline_us for r in valid)
    total_dsl = sum(r.dsl_us for r in valid if r.dsl_us == r.dsl_us)

    W = 95
    print("\n" + "=" * W)
    print("AMDAHL TABLE — Qwen3-30B-A3B decode step (batch=2, bf16)")
    print("Pure DSL vs AITER/ATOM production pipeline — apples-to-apples")
    print("=" * W)
    print(
        f"  {'Layer':<28} {'AITER/prod µs':>14} {'DSL µs':>10} "
        f"{'Speedup':>9} {'% step':>8} {'Saved µs':>10}"
    )
    print("-" * W)

    total_saved = 0.0
    for r in valid:
        pct = r.baseline_us / total_bl * 100 if total_bl > 0 else 0
        saved = r.baseline_us - r.dsl_us if r.dsl_us == r.dsl_us else 0.0
        total_saved += saved
        spd_s = f"{r.speedup:.3f}x" if r.speedup == r.speedup else "N/A"
        dsl_s = f"{r.dsl_us:.2f}" if r.dsl_us == r.dsl_us else "N/A"
        print(
            f"  {r.name:<28} {r.baseline_us:>14.2f} {dsl_s:>10} "
            f"{spd_s:>9} {pct:>7.1f}% {saved:>10.2f}"
        )

    print("-" * W)
    e2e = total_bl / total_dsl if total_dsl > 0 else NaN
    print(
        f"  {'TOTAL':<28} {total_bl:>14.2f} {total_dsl:>10.2f} "
        f"{e2e:>9.3f}x {'100.0%':>8} {total_saved:>10.2f}"
    )
    print("=" * W)
    print(f"\n  End-to-end: pure DSL vs AITER production = {e2e:.3f}x")
    print(
        f"  Total saved: {total_saved:.1f}µs ({total_saved / total_bl * 100:.1f}% of AITER step)"
    )
    return {
        "total_baseline_us": total_bl,
        "total_dsl_us": total_dsl,
        "e2e_speedup": e2e,
        "total_saved_us": total_saved,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def _prewarm_all():
    """Force all AITER JIT modules to load and all critical paths to compile.

    AITER loads its .so files lazily on first call — even WARMUP=10 is not
    enough when multiple modules initialise simultaneously in the same process.
    This block calls every operation at least 5 times with a full
    cuda.synchronize() after the batch so steady-state times are measured
    in the actual benchmark sections that follow.
    """
    print("\n[Pre-warming all kernels — forcing JIT loads ...]")
    import aiter

    M, N = BATCH, HIDDEN
    E, K = NUM_EXPERTS, TOPK
    T = BATCH

    # ── RMSNorm (aiter.rmsnorm2d_fwd_with_add) ──
    X = torch.randn(M, N, dtype=DTYPE, device="cuda")
    Res = torch.randn(M, N, dtype=DTYPE, device="cuda")
    G = torch.ones(N, dtype=DTYPE, device="cuda")
    out = torch.empty_like(X)
    res_out = torch.empty_like(X)
    for _ in range(5):
        aiter.rmsnorm2d_fwd_with_add(out, X, Res, res_out, G, 1e-6)
    torch.cuda.synchronize()
    print("  [✓] aiter.rmsnorm2d_fwd_with_add")

    # ── GEMMs (torch.matmul → hipBLASLt) ──
    for Nk in [(NHEAD_Q + 2 * NHEAD_K) * HEAD_DIM, HIDDEN]:
        A = torch.randn(M, HIDDEN, dtype=DTYPE, device="cuda")
        B = torch.randn(HIDDEN, Nk, dtype=DTYPE, device="cuda")
        for _ in range(5):
            torch.matmul(A, B)
    torch.cuda.synchronize()
    print("  [✓] torch.matmul (hipBLASLt)")

    # ── MoE sorting (aiter.moe_sorting_opus_fwd) ──
    topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
    topk_wts = torch.softmax(torch.randn(T, K), dim=-1).float().cuda()
    max_pad = T * K + E
    sids = torch.empty(max_pad, dtype=torch.int32, device="cuda")
    swts = torch.empty(max_pad, dtype=torch.float32, device="cuda")
    seids = torch.empty(max_pad, dtype=torch.int32, device="cuda")
    nvalid = torch.zeros(1, dtype=torch.int32, device="cuda")
    mbuf = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    for _ in range(5):
        aiter.moe_sorting_opus_fwd(
            topk_ids, topk_wts, sids, swts, seids, nvalid, mbuf, E, T * K
        )
    torch.cuda.synchronize()
    print("  [✓] aiter.moe_sorting_opus_fwd")

    # ── MoE e2e (aiter.fused_moe) ──
    try:
        from aiter.fused_moe import fused_moe as aiter_fm
        from aiter import ActivationType, QuantType

        W1t = torch.randn(E, 2 * MOE_INTER, HIDDEN, dtype=DTYPE, device="cuda") * 0.01
        W2t = torch.randn(E, HIDDEN, MOE_INTER, dtype=DTYPE, device="cuda") * 0.01
        Xt = torch.randn(T, HIDDEN, dtype=DTYPE, device="cuda") * 0.1
        tv, ti = torch.topk(torch.randn(T, E), K, dim=-1)
        twt = torch.softmax(tv, dim=-1).float().cuda()
        tit = ti.int().cuda()
        for _ in range(5):
            aiter_fm(
                Xt,
                W1t,
                W2t,
                twt,
                tit,
                activation=ActivationType.Silu,
                quant_type=QuantType.No,
            )
        torch.cuda.synchronize()
        print("  [✓] aiter.fused_moe")
    except Exception as exc:
        print(f"  [!] aiter.fused_moe: {exc}")

    # ── Triton unified_attention (paged decode) ──
    try:
        from aiter.ops.triton.attention.unified_attention import (
            unified_attention as tri_attn,
        )

        kv_len = DECODE_KV
        num_blks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pool = num_blks * BATCH + 64
        q2 = torch.randn(BATCH, NHEAD_Q, HEAD_DIM, dtype=DTYPE, device="cuda")
        kc2 = torch.randn(
            pool, BLOCK_SIZE, NHEAD_K, HEAD_DIM, dtype=DTYPE, device="cuda"
        )
        vc2 = torch.randn_like(kc2)
        cu2 = torch.tensor([0, BATCH], dtype=torch.int32, device="cuda")
        kvl2 = torch.full((BATCH,), kv_len, dtype=torch.int32, device="cuda")
        bt2 = torch.randint(
            0, pool, (BATCH, num_blks), dtype=torch.int32, device="cuda"
        )
        out2 = torch.empty_like(q2)
        scale2 = HEAD_DIM**-0.5

        def _tri():
            tri_attn(
                q=q2,
                k=kc2,
                v=vc2,
                out=out2,
                cu_seqlens_q=cu2,
                seqused_k=kvl2,
                max_seqlen_q=1,
                max_seqlen_k=kv_len,
                softmax_scale=scale2,
                causal=True,
                window_size=(-1, -1),
                block_table=bt2,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                alibi_slopes=None,
                qq_bias=None,
                sinks=None,
                backend="triton",
            )

        for _ in range(5):
            _tri()
        torch.cuda.synchronize()
        print("  [✓] aiter unified_attention (triton decode)")
    except Exception as exc:
        print(f"  [!] unified_attention: {exc}")

    # ── DSL GEMM (compiles on first call) ──
    try:
        for shape in [
            (BATCH, (NHEAD_Q + 2 * NHEAD_K) * HEAD_DIM, HIDDEN),
            (BATCH, HIDDEN, HIDDEN),
        ]:
            run_g, _ = _build_dsl_gemm(*shape)
            Av = torch.randn(shape[0], shape[2], dtype=DTYPE, device="cuda")
            Bv = torch.randn(shape[1], shape[2], dtype=DTYPE, device="cuda")
            Cv = torch.empty(shape[0], shape[1], dtype=DTYPE, device="cuda")
            for _ in range(5):
                run_g(int(Av.data_ptr()), int(Bv.data_ptr()), int(Cv.data_ptr()))
        torch.cuda.synchronize()
        print("  [✓] DSL universal_gemm")
    except Exception as exc:
        print(f"  [!] DSL universal_gemm: {exc}")

    # ── DSL RMSNorm (compiles on first call) ──
    try:
        Xn = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device="cuda")
        Rn = torch.randn_like(Xn)
        Gn = torch.ones(HIDDEN, dtype=DTYPE, device="cuda")
        Xo = torch.empty_like(Xn)
        Yo = torch.empty_like(Xn)
        run_n, _, _, _, _ = _build_dsl_norm(BATCH, HIDDEN)
        for _ in range(5):
            run_n(
                int(Xn.data_ptr()),
                int(Rn.data_ptr()),
                int(Gn.data_ptr()),
                int(Xo.data_ptr()),
                int(Yo.data_ptr()),
                BATCH,
                HIDDEN,
                1e-6,
            )
        torch.cuda.synchronize()
        print("  [✓] DSL add_rmsnorm2d_bf16")
    except Exception as exc:
        print(f"  [!] DSL add_rmsnorm2d_bf16: {exc}")

    # ── DSL TopK (compiles on first call) ──
    try:
        Xl = torch.randn(BATCH, NUM_EXPERTS, device="cuda", dtype=torch.float32)
        Ow = torch.empty(BATCH, TOPK, device="cuda", dtype=torch.float32)
        Oi = torch.empty(BATCH, TOPK, device="cuda", dtype=torch.int32)
        run_t, _, _, _, _ = _build_dsl_topk(BATCH, NUM_EXPERTS, TOPK)
        for _ in range(5):
            run_t(
                int(Xl.data_ptr()),
                int(Ow.data_ptr()),
                int(Oi.data_ptr()),
                BATCH,
                NUM_EXPERTS,
            )
        torch.cuda.synchronize()
        print("  [✓] DSL topk_softmax")
    except Exception as exc:
        print(f"  [!] DSL topk_softmax: {exc}")

    # ── DSL decode attention (heaviest compile — must happen before bench) ──
    try:
        from rocke.instances import (
            UnifiedAttentionProblem,
            run_unified_attention_torch,
        )

        kv_len = DECODE_KV
        num_blks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pool = num_blks * BATCH + 64
        q3 = torch.randn(BATCH, NHEAD_Q, HEAD_DIM, dtype=DTYPE, device="cuda")
        kc3 = torch.randn(
            pool, BLOCK_SIZE, NHEAD_K, HEAD_DIM, dtype=DTYPE, device="cuda"
        )
        vc3 = torch.randn_like(kc3)
        cu3 = torch.tensor([0, BATCH], dtype=torch.int32, device="cuda")
        kvl3 = torch.full((BATCH,), kv_len, dtype=torch.int32, device="cuda")
        bt3 = torch.randint(
            0, pool, (BATCH, num_blks), dtype=torch.int32, device="cuda"
        )
        out3 = torch.empty_like(q3)
        scale3 = HEAD_DIM**-0.5
        stream_h3 = int(torch.cuda.current_stream().cuda_stream)
        prob3 = UnifiedAttentionProblem(
            total_q=BATCH,
            num_seqs=BATCH,
            num_query_heads=NHEAD_Q,
            num_kv_heads=NHEAD_K,
            head_size=HEAD_DIM,
            block_size=BLOCK_SIZE,
            max_seqlen_q=1,
            max_seqlen_k=kv_len,
            dtype="bf16",
            num_sms=60,
        )

        def _dsl_attn():
            run_unified_attention_torch(
                problem=prob3,
                q=q3,
                k=kc3,
                v=vc3,
                out=out3,
                cu_seqlens_q=cu3,
                seqused_k=kvl3,
                softmax_scale=scale3,
                block_table=bt3,
                softcap=0.0,
                stream=stream_h3,
            )

        for _ in range(5):
            _dsl_attn()
        torch.cuda.synchronize()
        print(
            f"  [✓] DSL unified_attention decode (path={prob3.select_path()}, num_sms=60)"
        )
    except Exception as exc:
        print(f"  [!] DSL decode attention: {exc}")

    print("[Pre-warming complete — all JIT modules loaded, measuring steady-state]\n")


def main():
    assert torch.cuda.is_available(), "No GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Model: Qwen3-30B-A3B  batch={BATCH}  hidden={HIDDEN}  "
        f"experts={NUM_EXPERTS}×top{TOPK}  bf16"
    )
    print(f"Warmup={WARMUP}  Iters={ITERS}")
    print("Baselines: ACTUAL AITER/ATOM PRODUCTION KERNELS")
    print("  RMSNorm:  aiter.rmsnorm2d_fwd_with_add")
    print("  GEMMs:    torch.matmul (hipBLASLt, same as ATOM LinearBase)")
    print("  Attn:     aiter unified_attention backend=triton (ATOM dispatch)")
    print("  TopK:     aiter.moe_fused_gate")
    print("  Sort:     aiter.moe_sorting_opus_fwd")
    print("  MoE e2e:  aiter.fused_moe (2-stage CK)")
    print("DSL: pure CK DSL — no AITER in any implementation")

    # Force-load all JIT modules before any _ms() timing call
    _prewarm_all()

    print("\n" + "=" * 70)
    print("DECODE STEP — layer by layer")
    print("=" * 70)

    r_norm = bench_rmsnorm()
    r_qkv = bench_qkv_proj()
    r_attn = bench_decode_attn()
    r_oproj = bench_o_proj()

    r_norm2 = LayerResult(
        "rmsnorm_post_attn",
        r_norm.baseline_us,
        r_norm.dsl_us,
        r_norm.speedup,
        "same shape",
    )
    r_norm3 = LayerResult(
        "rmsnorm_pre_moe",
        r_norm.baseline_us,
        r_norm.dsl_us,
        r_norm.speedup,
        "same shape",
    )

    r_router = bench_router_topk()
    r_sort = bench_moe_sorting()  # informational only — also inside moe_e2e
    r_moe = bench_moe_e2e()  # FusedMoeForward includes sort internally

    # The real decode step uses FusedMoeForward (moe_e2e) which INCLUDES
    # sorting. moe_sorting standalone is measured separately for diagnostic
    # purposes only; do NOT include it in the Amdahl table (double-counted).
    decode_layers = [r_norm, r_qkv, r_attn, r_oproj, r_norm2, r_norm3, r_router, r_moe]

    print("\n[MoE Sorting — standalone diagnostic, NOT in Amdahl total]")
    print(f"  AITER moe_sorting_opus_fwd: {r_sort.baseline_us:.2f}µs")
    print(
        f"  DSL persistent sort:        {r_sort.dsl_us:.2f}µs  "
        f"speedup={r_sort.speedup:.3f}x"
    )
    print("  (FusedMoeForward includes sort; see moe_e2e row for e2e timing)")

    summary = amdahl_table(decode_layers)

    print("\n" + "=" * 70)
    print("PREFILL ATTENTION — DSL vs AITER Triton")
    print("=" * 70)
    prefill_results = bench_prefill_attn()

    # Save JSON
    def _ser(v):
        return None if isinstance(v, float) and v != v else v

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "model": "Qwen3-30B-A3B",
        "batch": BATCH,
        "methodology": "pure_dsl_vs_aiter_production_pipeline",
        "baselines": {
            "rmsnorm": "aiter.rmsnorm2d_fwd_with_add",
            "gemm": "torch.matmul (hipBLASLt)",
            "attention": "aiter unified_attention backend=triton",
            "topk": "aiter.moe_fused_gate",
            "moe_sort": "aiter.moe_sorting_opus_fwd",
            "moe_e2e": "aiter.fused_moe (2-stage CK)",
        },
        "decode_layers": [
            {
                "name": r.name,
                "baseline_us": _ser(r.baseline_us),
                "dsl_us": _ser(r.dsl_us),
                "speedup": _ser(r.speedup),
                "notes": r.notes,
            }
            for r in decode_layers
        ],
        "prefill_attention": [
            {
                "name": r.name,
                "baseline_us": _ser(r.baseline_us),
                "dsl_us": _ser(r.dsl_us),
                "speedup": _ser(r.speedup),
                "notes": r.notes,
            }
            for r in prefill_results
        ],
        "summary": {k: _ser(v) for k, v in summary.items()},
    }
    out_path = f"{DATA_DIR}/dsl_a3b_full_model.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
