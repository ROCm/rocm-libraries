#!/usr/bin/env python3
"""Host-side builder for the productized dense flash-attention prefill kernel
(``kernels/gfx950/attention_dense.py``).

Owns the host path: spec construction, kernel-spec generation, compilation, ABI
signature, and runtime launch — plus a torch/SDPA parity check and a benchmark. The
kernel bakes in the winning levers (CK-1 transposed PV, LDS K-padding, exp2_fast,
sched template, diagonal masking, depth-1 cluster, vectorized store); only the KV tile
(`block_n`) and occupancy hint (`waves_per_eu`) are tunable.

Usage:
    python attention_dense_prefill.py                 # parity + bench, default shapes
    python attention_dense_prefill.py --bn 128        # sweep block_n
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from kernels.gfx950.attention_dense import (  # noqa: E402
    AttentionDenseSpec,
    build_attention_dense,
    supports_attention_dense,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.helpers.spec import SignatureBuilder  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig  # noqa: E402

_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _build_request(args, sq: int):
    """Build AttentionRequest from CLI args for dispatch path."""
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
        # Required fields
        batch=1,
        nhead_q=args.hq,
        nhead_k=args.hkv,
        seqlen_q=sq,
        seqlen_k=sq,  # self-attention
        hdim_q=args.d,
        hdim_v=args.d,
        arch="gfx950",
        # Optional fields
        mask_type=args.causal,  # 0 or 1, converted to bool by _dense_spec
        use_sinks=args.use_sinks,
        sliding_window=args.sw,
        dtype=args.dtype,
        algorithm="attention_dense",  # Required for opt-in to dense candidate
        # Dense-specific knobs
        dense_persistent="off" if not args.persistent else "on",
        dense_num_persistent=args.np,
        dense_persist_decode="auto",
    )


def _make_launcher(spec: AttentionDenseSpec):
    """kernel-spec generation + compilation + ABI signature -> cached launcher."""
    ok, why = supports_attention_dense(spec)
    if not ok:
        raise ValueError(f"unsupported spec: {why}")
    art = compile_kernel(
        build_attention_dense(spec),
        arch="gfx950",
        backend="python",
        capture_ir_text=False,
    )
    sb = (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("v_ptr", spec.dtype)
        .ptr("o_ptr", spec.dtype)
        .scalar("scale", "f32")
    )
    if spec.use_sinks:
        sb = sb.ptr("sink_ptr", spec.dtype)
    if spec.varlen:
        sb = sb.ptr("cu_seqlens_q", "i32").ptr("cu_seqlens_kv", "i32")
    sig = sb.build()
    return KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)


def _launch_config(spec: AttentionDenseSpec, stream) -> LaunchConfig:
    if spec.persistent:
        # 1-D grid of long-lived CTAs; each grid-strides over all work items.
        grid = (spec.num_persistent, 1, 1)
    else:
        grid = (spec.seqlen_q // 256, spec.num_query_heads, spec.batch)  # BLOCK_M = 256
    return LaunchConfig(grid=grid, block=(spec.num_waves * 64, 1, 1), stream=stream)


def run(
    spec: AttentionDenseSpec, *, warmup: int = 15, iters: int = 50, check: bool = True
):
    dev = "cuda"
    dt = _TORCH_DT[spec.dtype]
    B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    Hq, Hkv, D = spec.num_query_heads, spec.num_kv_heads, spec.head_size
    torch.manual_seed(0)
    q = (torch.randn(B, Sq, Hq, D, dtype=dt, device=dev) * 0.2).contiguous()
    k = (torch.randn(B, Skv, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
    v = (torch.randn(B, Skv, Hkv, D, dtype=dt, device=dev) * 0.2).contiguous()
    out = torch.zeros(B, Sq, Hq, D, dtype=dt, device=dev)
    scale = 1.0 / math.sqrt(D)

    # Generate sink tensor if needed (per query head, raw attention scores)
    sinks = None
    if spec.use_sinks:
        sinks = torch.randn(Hq, dtype=dt, device=dev).contiguous()

    launcher = _make_launcher(spec)
    stream = torch.cuda.current_stream().cuda_stream
    cfg = _launch_config(spec, stream)
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": scale}
    if spec.use_sinks:
        vals["sink_ptr"] = sinks

    def call():
        launcher(vals, config=cfg)

    call()
    torch.cuda.synchronize()

    err = float("nan")
    if check:
        qh = q.transpose(1, 2).float()  # [B, Hq, Sq, D]
        rep = Hq // Hkv
        kh = k.transpose(1, 2).repeat_interleave(rep, 1).float()  # [B, Hq, Skv, D]
        vh = v.transpose(1, 2).repeat_interleave(rep, 1).float()  # [B, Hq, Skv, D]
        W = spec.sliding_window

        # When sinks are enabled, must use manual implementation (torch SDPA doesn't support sinks)
        if spec.use_sinks:
            # Manual attention computation (matches parity_unified_attention.py pattern)
            attn = torch.einsum("bhqd,bhkd->bhqk", qh, kh) / math.sqrt(D)

            # Apply causal and/or sliding window masks
            if spec.causal or W > 0:
                qi = torch.arange(Sq, device=dev).view(-1, 1)
                ki = torch.arange(Skv, device=dev).view(1, -1)
                mask = torch.zeros(Sq, Skv, dtype=torch.bool, device=dev)
                if spec.causal:
                    mask |= ki > qi  # Causal: mask future tokens
                if W > 0:
                    mask |= ki <= (qi - W)  # Sliding window: mask tokens beyond window
                attn.masked_fill_(mask.view(1, 1, Sq, Skv), float("-inf"))

            # Add sinks to attention scores (append virtual sink token)
            if sinks is not None:
                # Sinks are raw attention scores (same domain as attn)
                sink_scores = sinks.float().view(1, Hq, 1, 1).expand(B, Hq, Sq, 1)
                attn = torch.cat([attn, sink_scores], dim=-1)  # [B, Hq, Sq, Skv+1]

            # Softmax over all scores (including sink)
            attn = torch.softmax(attn, dim=-1)

            # Remove sink from attention weights (sink has no V vector)
            if sinks is not None:
                attn = attn[..., :-1]  # [B, Hq, Sq, Skv]

            ref = torch.einsum("bhqk,bhkd->bhqd", attn, vh).transpose(1, 2)

        else:
            # Original fast path: use PyTorch's optimized SDPA when sinks not needed
            if spec.causal and W > 0:
                # Banded mask: keep k in [q-W+1, q] (causal AND sliding window).
                qi = torch.arange(Sq, device=dev).view(-1, 1)
                ki = torch.arange(Skv, device=dev).view(1, -1)
                allowed = (ki <= qi) & (ki > qi - W)
                ref = torch.nn.functional.scaled_dot_product_attention(
                    qh, kh, vh, attn_mask=allowed
                ).transpose(1, 2)
            else:
                ref = torch.nn.functional.scaled_dot_product_attention(
                    qh, kh, vh, is_causal=spec.causal
                ).transpose(1, 2)
        err = (out.float() - ref).abs().max().item()

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call()
    e.record()
    e.synchronize()
    ms = s.elapsed_time(e) / iters
    W = spec.sliding_window
    if spec.causal and W > 0:
        # banded pair count: sum_q min(q+1, W) = W*Sq - W*(W-1)/2 (Sq>=W)
        pairs = W * Sq - W * (W - 1) // 2 if Sq >= W else Sq * (Sq + 1) // 2
        flops = 4 * B * Hq * D * pairs
    elif spec.causal:
        flops = 4 * B * Hq * D * (Sq * (Sq + 1) // 2)
    else:
        flops = 2 * 2 * B * Hq * D * Sq * Skv
    tf = flops / (ms * 1e-3) / 1e12
    status = "PASS" if (not check or err < 2e-2) else "FAIL"
    print(
        f"[{spec.kernel_name()}] {ms:.4f} ms  {tf:.1f} TFLOPS  max_abs={err:.2e}  {status}"
    )
    return ms, tf, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bn", type=int, default=64, help="block_n (KV tile)")
    ap.add_argument("--wpe", type=int, default=2, help="waves_per_eu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--hq", type=int, default=128)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument(
        "--persistent",
        action="store_true",
        help="grid-stride persistent kernel (amortizes per-CTA launch/setup; "
        "+70%% at Sq=8192 causal)",
    )
    ap.add_argument("--np", type=int, default=256, help="num_persistent CTAs")
    ap.add_argument("--interleave", action="store_true", help="boustrophedon qb order")
    ap.add_argument(
        "--sw", type=int, default=0, help="sliding_window (0=off; multiple of --bn)"
    )
    ap.add_argument("--use-sinks", action="store_true", help="enable attention sinks")
    ap.add_argument(
        "--dispatch",
        action="store_true",
        help="route through dispatch layer instead of direct spec construction",
    )
    args = ap.parse_args()

    # Warn about incompatible flags when using dispatch path
    if args.dispatch:
        if args.bn != 64:
            print(f"Warning: --dispatch uses block_n=64 (ignoring --bn {args.bn})")
        if args.wpe != 2:
            print(
                f"Warning: --dispatch uses waves_per_eu=2 (ignoring --wpe {args.wpe})"
            )
        if args.interleave:
            print("Warning: --dispatch does not support --interleave")

    for sq in (256, 512, 2048, 8192):
        if args.dispatch:
            # Route through dispatch layer
            from dispatch.attention.gfx950 import dense_spec_for_request

            req = _build_request(args, sq)
            spec = dense_spec_for_request(req)
        else:
            # Direct spec construction (original path)
            spec = AttentionDenseSpec(
                batch=1,
                seqlen_q=sq,
                seqlen_kv=sq,
                num_query_heads=args.hq,
                num_kv_heads=args.hkv,
                head_size=args.d,
                causal=bool(args.causal),
                dtype=args.dtype,
                block_n=args.bn,
                waves_per_eu=args.wpe,
                persistent=args.persistent,
                num_persistent=args.np,
                interleave=args.interleave,
                sliding_window=args.sw,
                use_sinks=args.use_sinks,
            )
        run(spec)


if __name__ == "__main__":
    main()
