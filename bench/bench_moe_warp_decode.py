# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Warp-decode vs AITER fused_moe benchmark (FP8 block-scale, gfx950).

Compares three "MoE block" paths on shared weights/topk inputs:

  1. AITER       : fused_topk + aiter.fused_moe(QuantType.per_1x128)
     (internally: moe_sorting + per-1x128 FP8 activation quant + fmoe_fp8_blockscale_g1u1)

  2. WD-FP8      : fused_topk + per-1x128 FP8 quant + warp_decode_gate_up_fp8
                   + warp_decode_down_reduce

  3. WD-BF16     : fused_topk + warp_decode_gate_up_bf16 + warp_decode_down_reduce
                   (BF16 activations straight into the gate/up kernel -- no quant)

Sweeps DeepSeek-V3-like (HIDDEN=7168, INTER=2048) and MiniMax-like
(HIDDEN=3072, INTER=1536) shapes for B in {1,2,4,8,16,32,64} with E=256,
TOPK=8. Reports total us, per-stage us (topk, quant, gate_up, down_reduce),
and GB/s per path, plus a correctness column vs. torch_moe_blockscale.

Run:
    python3 bench_moe_warp_decode.py
    python3 bench_moe_warp_decode.py --iters 30 --warmup 3 --shapes deepseek
    python3 bench_moe_warp_decode.py --csv bench.csv

Requires aiter, the warp_decode_ext torch extension in this directory, and a
gfx950 GPU.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

# Make sure warp_decode_ext (in this directory) is importable.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import aiter  # noqa: E402
from aiter import QuantType, get_hip_quant, dtypes  # noqa: E402
from aiter.fused_moe import fused_topk, fused_moe  # noqa: E402
from aiter.test_common import run_perftest, checkAllclose  # noqa: E402
from aiter import pertoken_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from einops import rearrange  # noqa: E402

import warp_decode_ext  # noqa: E402


# ---------------------------------------------------------------------------
# Shapes & config
# ---------------------------------------------------------------------------

@dataclass
class ShapeCfg:
    name: str
    HIDDEN: int
    INTER: int
    E: int = 256
    TOPK: int = 8


SHAPES = {
    "deepseek": ShapeCfg("deepseek-v3", HIDDEN=7168, INTER=2048),
    "minimax":  ShapeCfg("minimax",     HIDDEN=3072, INTER=1536),
}

DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------
# Weight / scale preparation
# ---------------------------------------------------------------------------

BLOCK = 128
FP8_DTYPE = dtypes.fp8          # on gfx950 this is torch.float8_e4m3fn (OCP)
BF16 = torch.bfloat16
F32  = torch.float32
I32  = torch.int32


def _block_quant(w_bf16: torch.Tensor,
                 block_n: int = BLOCK,
                 block_k: int = BLOCK):
    """Block-[block_n x block_k] FP8 quantize a 3D weight tensor [E, N, K].

    Returns (w_fp8[E,N,K], w_scale[E, N/block_n, K/block_k]).
    """
    assert w_bf16.dim() == 3
    E, N, K = w_bf16.shape
    assert N % block_n == 0 and K % block_k == 0, \
        f"weight shape {w_bf16.shape} not divisible by block {(block_n, block_k)}"
    tmp = rearrange(
        w_bf16.view(E, N // block_n, block_n, K // block_k, block_k),
        "e nbn bn nbk bk -> e nbn nbk (bn bk)",
    ).contiguous()
    w_q, w_scale = pertoken_quant(tmp, quant_dtype=FP8_DTYPE)
    w_q = rearrange(
        w_q.view(E, N // block_n, K // block_k, block_n, block_k),
        "e nbn nbk bn bk -> e (nbn bn) (nbk bk)",
    ).contiguous()
    w_scale = w_scale.view(E, N // block_n, K // block_k).contiguous()
    return w_q, w_scale


@dataclass
class WeightPack:
    shape: ShapeCfg

    # BF16 weights (kept for reference / torch_moe_blockscale)
    w_gate_bf16: torch.Tensor  # [E, INTER, HIDDEN]
    w_up_bf16: torch.Tensor
    w_down_bf16: torch.Tensor  # [E, HIDDEN, INTER]

    # Per-block FP8 weights (gate/up split)
    w_gate_fp8: torch.Tensor   # [E, INTER, HIDDEN] fp8
    w_up_fp8: torch.Tensor
    w_down_fp8: torch.Tensor   # [E, HIDDEN, INTER] fp8

    # Warp-decode scales (contiguous, per-(N/128, K/128))
    w_gate_scale_wd: torch.Tensor  # [E*INTER/128, HIDDEN/128] fp32
    w_up_scale_wd:   torch.Tensor
    w_down_scale_wd: torch.Tensor  # [E*HIDDEN/128, INTER/128] fp32

    # AITER fused-MoE tensors (gate|up interleaved along N axis)
    w1_fp8: torch.Tensor           # [E, 2*INTER, HIDDEN] fp8
    w1_scale: torch.Tensor         # [E, 2*INTER/128, HIDDEN/128] fp32
    w2_fp8: torch.Tensor           # [E, HIDDEN, INTER] fp8
    w2_scale: torch.Tensor         # [E, HIDDEN/128, INTER/128] fp32


def build_weights(shape: ShapeCfg, device: str = "cuda", seed: int = 123) -> WeightPack:
    torch.manual_seed(seed)
    HIDDEN, INTER, E = shape.HIDDEN, shape.INTER, shape.E
    assert INTER % BLOCK == 0 and HIDDEN % BLOCK == 0, \
        f"INTER={INTER}, HIDDEN={HIDDEN} must both be divisible by {BLOCK}"

    w_gate_bf16 = (torch.randn(E, INTER, HIDDEN, dtype=BF16, device=device) / 10.0)
    w_up_bf16   = (torch.randn(E, INTER, HIDDEN, dtype=BF16, device=device) / 10.0)
    w_down_bf16 = (torch.randn(E, HIDDEN, INTER, dtype=BF16, device=device) / 10.0)

    w_gate_fp8, w_gate_scale = _block_quant(w_gate_bf16)
    w_up_fp8,   w_up_scale   = _block_quant(w_up_bf16)
    w_down_fp8, w_down_scale = _block_quant(w_down_bf16)

    # Warp-decode wants flat [E*N/128, K/128] scales.
    w_gate_scale_wd = w_gate_scale.reshape(E * (INTER // BLOCK), HIDDEN // BLOCK).contiguous()
    w_up_scale_wd   = w_up_scale.reshape(  E * (INTER // BLOCK), HIDDEN // BLOCK).contiguous()
    w_down_scale_wd = w_down_scale.reshape(E * (HIDDEN // BLOCK), INTER // BLOCK).contiguous()

    # AITER wants gate/up concatenated along N axis:
    #   w1[E, 2*INTER, HIDDEN] with matching w1_scale[E, 2*INTER/128, HIDDEN/128].
    # fused_moe's per_1x128 2-stage path expects shuffled weights (16,16 layout),
    # same as test_moe_blockscale.py::asm_moe_test.
    w1_unsh = torch.cat([w_gate_fp8, w_up_fp8], dim=1).contiguous()
    w1_fp8   = shuffle_weight(w1_unsh, (16, 16))
    w1_scale = torch.cat([w_gate_scale, w_up_scale], dim=1).contiguous()
    w2_fp8   = shuffle_weight(w_down_fp8.contiguous(), (16, 16))
    w2_scale = w_down_scale.contiguous()

    return WeightPack(
        shape=shape,
        w_gate_bf16=w_gate_bf16, w_up_bf16=w_up_bf16, w_down_bf16=w_down_bf16,
        w_gate_fp8=w_gate_fp8,   w_up_fp8=w_up_fp8,   w_down_fp8=w_down_fp8,
        w_gate_scale_wd=w_gate_scale_wd,
        w_up_scale_wd=w_up_scale_wd,
        w_down_scale_wd=w_down_scale_wd,
        w1_fp8=w1_fp8,   w1_scale=w1_scale,
        w2_fp8=w2_fp8,   w2_scale=w2_scale,
    )


# ---------------------------------------------------------------------------
# Per-path wrappers
# ---------------------------------------------------------------------------

_hip_quant_per_1x128 = get_hip_quant(QuantType.per_1x128)


def aiter_moe_block(hidden_states: torch.Tensor,
                    gating: torch.Tensor,
                    wp: WeightPack) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    out = fused_moe(
        hidden_states,
        wp.w1_fp8,
        wp.w2_fp8,
        topk_weights,
        topk_ids,
        quant_type=QuantType.per_1x128,
        w1_scale=wp.w1_scale.view(wp.shape.E, -1),
        w2_scale=wp.w2_scale.view(wp.shape.E, -1),
    )
    return out


def wd_fp8_moe_block(hidden_states: torch.Tensor,
                     gating: torch.Tensor,
                     wp: WeightPack,
                     gate_up_func: Callable = warp_decode_ext.warp_decode_gate_up_fp8,
                     down_func: Callable = warp_decode_ext.warp_decode_down_reduce) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    x_fp8, x_scale = _hip_quant_per_1x128(hidden_states, quant_dtype=FP8_DTYPE)

    B = hidden_states.shape[0]
    inter = torch.empty(
        (B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device
    )
    gate_up_func(
        x_fp8, x_scale,
        wp.w_gate_fp8, wp.w_gate_scale_wd,
        wp.w_up_fp8,   wp.w_up_scale_wd,
        topk_ids.to(I32).contiguous(),
        inter,
    )
    y = torch.empty((B, wp.shape.HIDDEN), dtype=BF16, device=hidden_states.device)
    down_func(
        inter, wp.w_down_fp8, wp.w_down_scale_wd,
        topk_ids.to(I32).contiguous(),
        topk_weights.to(F32).contiguous(),
        y,
    )
    return y


def wd_bf16_moe_block(hidden_states: torch.Tensor,
                      gating: torch.Tensor,
                      wp: WeightPack,
                      gate_up_func: Callable = warp_decode_ext.warp_decode_gate_up_bf16,
                      down_func: Callable = warp_decode_ext.warp_decode_down_reduce) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)

    B = hidden_states.shape[0]
    inter = torch.empty(
        (B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device
    )
    gate_up_func(
        hidden_states,
        wp.w_gate_fp8, wp.w_gate_scale_wd,
        wp.w_up_fp8,   wp.w_up_scale_wd,
        topk_ids.to(I32).contiguous(),
        inter,
    )
    y = torch.empty((B, wp.shape.HIDDEN), dtype=BF16, device=hidden_states.device)
    down_func(
        inter, wp.w_down_fp8, wp.w_down_scale_wd,
        topk_ids.to(I32).contiguous(),
        topk_weights.to(F32).contiguous(),
        y,
    )
    return y


# ---------------------------------------------------------------------------
# Torch reference (correctness)
# ---------------------------------------------------------------------------

def torch_moe_blockscale_ref(hidden_states: torch.Tensor,
                             gating: torch.Tensor,
                             wp: WeightPack) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    E, INTER, HIDDEN = wp.shape.E, wp.shape.INTER, wp.shape.HIDDEN
    B = hidden_states.shape[0]
    compute = F32

    x = hidden_states.to(compute)
    w_gate = wp.w_gate_fp8.to(compute)
    w_up   = wp.w_up_fp8.to(compute)
    w_down = wp.w_down_fp8.to(compute)

    def _apply_scale_3d(w_q: torch.Tensor, scale: torch.Tensor):
        # scale: [E, N/128, K/128] -> broadcast to [E, N, K]
        return w_q * scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)

    w_gate = _apply_scale_3d(w_gate,
                             wp.w_gate_scale_wd.view(E, INTER // BLOCK, HIDDEN // BLOCK))
    w_up   = _apply_scale_3d(w_up,
                             wp.w_up_scale_wd.view(E, INTER // BLOCK, HIDDEN // BLOCK))
    w_down = _apply_scale_3d(w_down,
                             wp.w_down_scale_wd.view(E, HIDDEN // BLOCK, INTER // BLOCK))

    out = torch.zeros(B, wp.shape.TOPK, HIDDEN, dtype=compute, device=x.device)
    for b in range(B):
        for k in range(wp.shape.TOPK):
            e = topk_ids[b, k].item()
            g = x[b] @ w_gate[e].t()
            u = x[b] @ w_up[e].t()
            inter = F.silu(g) * u
            out[b, k] = inter @ w_down[e].t()
    return (out * topk_weights.view(B, -1, 1).to(compute)).sum(dim=1).to(BF16)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

@dataclass
class StageTimings:
    topk_us: float = 0.0
    quant_us: float = 0.0          # per-1x128 FP8 activation quant (AITER internal / WD-FP8)
    gate_up_us: float = 0.0
    down_us: float = 0.0
    total_us: float = 0.0          # full path incl topk
    core_us: float = 0.0           # full path excluding topk
    err: Optional[dict] = None     # correctness (optional)


def _time(func: Callable, *args, iters: int, warmup: int, **kwargs) -> tuple:
    out, us = run_perftest(func, *args, num_iters=iters, num_warmup=warmup, **kwargs)
    return out, us


def bench_aiter(hidden_states, gating, wp, iters, warmup) -> StageTimings:
    tt = StageTimings()

    def path(h, g):
        return aiter_moe_block(h, g, wp)

    _, tt.total_us = _time(path, hidden_states, gating, iters=iters, warmup=warmup)

    # Pre-computed topk cost
    _, tt.topk_us = _time(
        lambda h, g: fused_topk(h, g, wp.shape.TOPK, True),
        hidden_states, gating, iters=iters, warmup=warmup,
    )

    # Quant cost (what fused_moe does internally before the 1-stage ASM kernel).
    # We time per-1x128 quant on the raw hidden_states as a proxy for the
    # pre-kernel quant stage AITER pays.
    _, tt.quant_us = _time(
        lambda h: _hip_quant_per_1x128(h, quant_dtype=FP8_DTYPE),
        hidden_states, iters=iters, warmup=warmup,
    )

    tt.core_us = tt.total_us - tt.topk_us
    return tt


def bench_wd_fp8(hidden_states,
                 gating,
                 wp,
                 iters,
                 warmup,
                 gate_up_func: Callable = warp_decode_ext.warp_decode_gate_up_fp8,
                 down_func: Callable = warp_decode_ext.warp_decode_down_reduce) -> StageTimings:
    tt = StageTimings()

    def path(h, g):
        return wd_fp8_moe_block(h, g, wp, gate_up_func=gate_up_func, down_func=down_func)

    _, tt.total_us = _time(path, hidden_states, gating, iters=iters, warmup=warmup)

    _, tt.topk_us = _time(
        lambda h, g: fused_topk(h, g, wp.shape.TOPK, True),
        hidden_states, gating, iters=iters, warmup=warmup,
    )

    _, tt.quant_us = _time(
        lambda h: _hip_quant_per_1x128(h, quant_dtype=FP8_DTYPE),
        hidden_states, iters=iters, warmup=warmup,
    )

    # Pre-compute routing + quant inputs for the kernel-only timings.
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    x_fp8, x_scale = _hip_quant_per_1x128(hidden_states, quant_dtype=FP8_DTYPE)
    B = hidden_states.shape[0]
    inter = torch.empty((B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device)
    y     = torch.empty((B, wp.shape.HIDDEN),               dtype=BF16, device=hidden_states.device)
    router_ids = topk_ids.to(I32).contiguous()
    router_wts = topk_weights.to(F32).contiguous()

    _, tt.gate_up_us = _time(
        gate_up_func,
        x_fp8, x_scale,
        wp.w_gate_fp8, wp.w_gate_scale_wd,
        wp.w_up_fp8,   wp.w_up_scale_wd,
        router_ids, inter,
        iters=iters, warmup=warmup,
    )

    _, tt.down_us = _time(
        down_func,
        inter, wp.w_down_fp8, wp.w_down_scale_wd,
        router_ids, router_wts, y,
        iters=iters, warmup=warmup,
    )

    tt.core_us = tt.total_us - tt.topk_us
    return tt


def bench_wd_bf16(hidden_states,
                  gating,
                  wp,
                  iters,
                  warmup,
                  gate_up_func: Callable = warp_decode_ext.warp_decode_gate_up_bf16,
                  down_func: Callable = warp_decode_ext.warp_decode_down_reduce) -> StageTimings:
    tt = StageTimings()

    def path(h, g):
        return wd_bf16_moe_block(h, g, wp, gate_up_func=gate_up_func, down_func=down_func)

    _, tt.total_us = _time(path, hidden_states, gating, iters=iters, warmup=warmup)

    _, tt.topk_us = _time(
        lambda h, g: fused_topk(h, g, wp.shape.TOPK, True),
        hidden_states, gating, iters=iters, warmup=warmup,
    )

    # No activation quant for bf16 path.
    tt.quant_us = 0.0

    # Kernel-only stage timings.
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    B = hidden_states.shape[0]
    inter = torch.empty((B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device)
    y     = torch.empty((B, wp.shape.HIDDEN),               dtype=BF16, device=hidden_states.device)
    router_ids = topk_ids.to(I32).contiguous()
    router_wts = topk_weights.to(F32).contiguous()

    _, tt.gate_up_us = _time(
        gate_up_func,
        hidden_states,
        wp.w_gate_fp8, wp.w_gate_scale_wd,
        wp.w_up_fp8,   wp.w_up_scale_wd,
        router_ids, inter,
        iters=iters, warmup=warmup,
    )

    _, tt.down_us = _time(
        down_func,
        inter, wp.w_down_fp8, wp.w_down_scale_wd,
        router_ids, router_wts, y,
        iters=iters, warmup=warmup,
    )

    tt.core_us = tt.total_us - tt.topk_us
    return tt


# ---------------------------------------------------------------------------
# Bandwidth (approximate)
# ---------------------------------------------------------------------------

def path_bytes(B: int, shape: ShapeCfg, path: str) -> float:
    """Approximate bytes moved for the full MoE block for a given path.

    Reads:
      - gate_up: B*TOPK*INTER activations read from hidden_states (x once per top_k
        output) + 2 * B*TOPK*INTER*HIDDEN weight bytes (gate & up)
      - down   : B*TOPK*INTER intermediate + B*TOPK*INTER*HIDDEN w_down bytes
    Writes:
      - intermediate [B,TOPK,INTER] bf16
      - y [B, HIDDEN] bf16

    Per-element bytes: fp8=1, bf16=2, fp32 scales ignored (negligible).
    """
    HIDDEN, INTER, TOPK = shape.HIDDEN, shape.INTER, shape.TOPK
    x_elem = 1 if path.startswith("wd_fp8") or path == "aiter" else 2   # fp8 vs bf16
    w_elem = 1
    inter_elem = 2
    y_elem = 2

    # gate_up: x read per output element (with kVector loads the compiler reuses
    # across the two matmuls, but it is still two reads effectively per hidden elt)
    gate_up_x = B * TOPK * INTER * HIDDEN * x_elem
    gate_up_w = 2.0 * B * TOPK * INTER * HIDDEN * w_elem
    gate_up_y = B * TOPK * INTER * inter_elem

    down_x = B * TOPK * INTER * inter_elem
    down_w = B * TOPK * INTER * HIDDEN * w_elem
    down_y = B * HIDDEN * y_elem

    return gate_up_x + gate_up_w + gate_up_y + down_x + down_w + down_y


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(args):
    device = "cuda"
    shapes = [SHAPES[k] for k in args.shapes]
    batches = tuple(args.batches)

    rows = []

    header = (
        f"{'shape':<14} {'B':>4} {'path':<9} "
        f"{'total_us':>10} {'topk_us':>9} {'quant_us':>10} "
        f"{'gate_up_us':>12} {'down_us':>10} {'core_us':>10} "
        f"{'GB/s':>8} {'ratio':>7} {'err/cos':>14}"
    )
    print(header)
    print("-" * len(header))

    for shape in shapes:
        wp = build_weights(shape, device=device)
        torch.cuda.synchronize()

        for B in batches:
            torch.manual_seed(1000 + B)
            hidden_states = torch.randn(B, shape.HIDDEN, dtype=BF16, device=device) * 0.1
            gating = torch.randn(B, shape.E, dtype=BF16, device=device)

            iters, warmup = args.iters, args.warmup

            path_specs = {
                "wd_fp8": (
                    bench_wd_fp8,
                    dict(gate_up_func=warp_decode_ext.warp_decode_gate_up_fp8,
                         down_func=warp_decode_ext.warp_decode_down_reduce),
                    lambda h, g: wd_fp8_moe_block(h, g, wp),
                ),
                "wd_bf16": (
                    bench_wd_bf16,
                    dict(gate_up_func=warp_decode_ext.warp_decode_gate_up_bf16,
                         down_func=warp_decode_ext.warp_decode_down_reduce),
                    lambda h, g: wd_bf16_moe_block(h, g, wp),
                ),
                "wd_fp8_lds": (
                    bench_wd_fp8,
                    dict(gate_up_func=warp_decode_ext.warp_decode_gate_up_fp8_lds,
                         down_func=warp_decode_ext.warp_decode_down_reduce_lds),
                    lambda h, g: wd_fp8_moe_block(
                        h,
                        g,
                        wp,
                        gate_up_func=warp_decode_ext.warp_decode_gate_up_fp8_lds,
                        down_func=warp_decode_ext.warp_decode_down_reduce_lds),
                ),
                "wd_bf16_lds": (
                    bench_wd_bf16,
                    dict(gate_up_func=warp_decode_ext.warp_decode_gate_up_bf16_lds,
                         down_func=warp_decode_ext.warp_decode_down_reduce_lds),
                    lambda h, g: wd_bf16_moe_block(
                        h,
                        g,
                        wp,
                        gate_up_func=warp_decode_ext.warp_decode_gate_up_bf16_lds,
                        down_func=warp_decode_ext.warp_decode_down_reduce_lds),
                ),
            }
            if args.include_base:
                path_specs.update({
                    "wd_fp8_base": (
                        bench_wd_fp8,
                        dict(gate_up_func=warp_decode_ext.warp_decode_gate_up_fp8_base,
                             down_func=warp_decode_ext.warp_decode_down_reduce_base),
                        lambda h, g: wd_fp8_moe_block(
                            h,
                            g,
                            wp,
                            gate_up_func=warp_decode_ext.warp_decode_gate_up_fp8_base,
                            down_func=warp_decode_ext.warp_decode_down_reduce_base),
                    ),
                    "wd_bf16_base": (
                        bench_wd_bf16,
                        dict(gate_up_func=warp_decode_ext.warp_decode_gate_up_bf16_base,
                             down_func=warp_decode_ext.warp_decode_down_reduce_base),
                        lambda h, g: wd_bf16_moe_block(
                            h,
                            g,
                            wp,
                            gate_up_func=warp_decode_ext.warp_decode_gate_up_bf16_base,
                            down_func=warp_decode_ext.warp_decode_down_reduce_base),
                    ),
                })
            if args.include_pkf32:
                path_specs.update({
                    "wd_fp8_pkf32": (
                        bench_wd_fp8,
                        dict(gate_up_func=warp_decode_ext.warp_decode_gate_up_fp8_pkf32,
                             down_func=warp_decode_ext.warp_decode_down_reduce_pkf32),
                        lambda h, g: wd_fp8_moe_block(
                            h,
                            g,
                            wp,
                            gate_up_func=warp_decode_ext.warp_decode_gate_up_fp8_pkf32,
                            down_func=warp_decode_ext.warp_decode_down_reduce_pkf32),
                    ),
                    "wd_bf16_pkf32": (
                        bench_wd_bf16,
                        dict(gate_up_func=warp_decode_ext.warp_decode_gate_up_bf16_pkf32,
                             down_func=warp_decode_ext.warp_decode_down_reduce_pkf32),
                        lambda h, g: wd_bf16_moe_block(
                            h,
                            g,
                            wp,
                            gate_up_func=warp_decode_ext.warp_decode_gate_up_bf16_pkf32,
                            down_func=warp_decode_ext.warp_decode_down_reduce_pkf32),
                    ),
                })

            path_results: dict[str, StageTimings] = {}

            path_results["aiter"] = bench_aiter(hidden_states, gating, wp, iters, warmup)
            for name, (bench_func, kwargs, _) in path_specs.items():
                path_results[name] = bench_func(hidden_states, gating, wp, iters, warmup, **kwargs)

            # Correctness: only for the smaller batches (torch ref is O(B*TOPK*E))
            if args.correctness and B in (1, 2, 8):
                ref = torch_moe_blockscale_ref(hidden_states, gating, wp)
                y_a = aiter_moe_block(hidden_states, gating, wp)
                path_results["aiter"].err = _err_metrics(ref, y_a)
                for name, (_, _, path_func) in path_specs.items():
                    path_results[name].err = _err_metrics(ref, path_func(hidden_states, gating))

            # Ratio = slowest total / current total (so 1.0 = slowest)
            slowest = max(p.total_us for p in path_results.values())

            for name, tt in path_results.items():
                bytes_ = path_bytes(B, shape, name)
                gbs = bytes_ / (tt.core_us * 1e3) if tt.core_us > 0 else 0.0
                ratio = slowest / tt.total_us if tt.total_us > 0 else 0.0
                if tt.err is not None:
                    err_str = (
                        f"{tt.err['err_ratio']:.3f}/"
                        f"{tt.err['cosine_sim']:.4f}"
                    )
                else:
                    err_str = "     -"
                print(
                    f"{shape.name:<14} {B:>4} {name:<9} "
                    f"{tt.total_us:>10.1f} {tt.topk_us:>9.1f} {tt.quant_us:>10.1f} "
                    f"{tt.gate_up_us:>12.1f} {tt.down_us:>10.1f} {tt.core_us:>10.1f} "
                    f"{gbs:>8.1f} {ratio:>7.2f} {err_str:>14}"
                )
                rows.append({
                    "shape": shape.name,
                    "HIDDEN": shape.HIDDEN,
                    "INTER": shape.INTER,
                    "E": shape.E,
                    "TOPK": shape.TOPK,
                    "B": B,
                    "path": name,
                    "total_us": tt.total_us,
                    "topk_us": tt.topk_us,
                    "quant_us": tt.quant_us,
                    "gate_up_us": tt.gate_up_us,
                    "down_us": tt.down_us,
                    "core_us": tt.core_us,
                    "gb_per_s": gbs,
                    "ratio_vs_slowest": ratio,
                    "err_ratio": tt.err["err_ratio"] if tt.err else None,
                    "max_abs_err": tt.err["max_abs_err"] if tt.err else None,
                    "cos_sim": tt.err["cosine_sim"] if tt.err else None,
                })
            print()

    if args.csv:
        out = Path(args.csv)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {out}")


def _err_metrics(a: torch.Tensor, b: torch.Tensor, rtol=1e-2, atol=1e-2) -> dict:
    a_f = a.to(F32).flatten()
    b_f = b.to(F32).flatten()
    close = torch.isclose(a_f, b_f, rtol=rtol, atol=atol)
    err_ratio = 1.0 - close.float().mean().item()
    max_abs_err = (a_f - b_f).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    return dict(err_ratio=err_ratio, max_abs_err=max_abs_err, cosine_sim=cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES.keys()),
                    choices=list(SHAPES.keys()))
    ap.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    ap.add_argument("--iters", type=int, default=50,
                    help="perftest iterations per timed call")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--csv", type=str, default=None,
                    help="optional CSV output path")
    ap.add_argument("--include-base", action="store_true",
                    help="also benchmark pre-dot2 baseline warp-decode variants")
    ap.add_argument("--include-pkf32", action="store_true",
                    help="also benchmark packed-FP32 FMA warp-decode variants")
    ap.add_argument("--no-correctness", dest="correctness",
                    action="store_false", default=True,
                    help="skip torch_moe_blockscale correctness check")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU available")

    sweep(args)


if __name__ == "__main__":
    main()
