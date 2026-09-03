#!/usr/bin/env python3
"""Host-side builder for the productized dense flash-attention prefill kernel
(``kernels/gfx950/attention_dense.py``).

Owns the host path: spec construction, kernel-spec generation, compilation, ABI
signature, and runtime launch — plus a torch/SDPA parity check and a benchmark. The
kernel bakes in the winning levers (CK-1 transposed PV, LDS K-padding, exp2_fast,
sched template, diagonal masking, depth-1 cluster, vectorized store). Tile geometry
(`block_m`, `block_n`, `lds_v_row_pad`), occupancy, persistent decode, and the
gfx950 wide-LDS-DMA path are captured explicitly by ``AttentionDenseSpec``.

Usage:
    python attention_dense_prefill.py                 # parity + bench, default shapes
    python attention_dense_prefill.py --bn 128        # sweep block_n
    python attention_dense_prefill.py --exact-shape --sq 8192 --hq 32 --dtype fp16 \\
        --persistent --json-out result.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

from kernels.gfx950.attention_dense import (  # noqa: E402
    AttentionDenseSpec,
    DENSE_TILE_GEOMETRIES,
    attention_dense_block,
    attention_dense_grid,
    build_attention_dense,
    supports_attention_dense,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.helpers.spec import SignatureBuilder  # noqa: E402
from rocke.runtime import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)

_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}
_TOL = 2e-2


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
    return LaunchConfig(
        grid=attention_dense_grid(spec),
        block=attention_dense_block(spec),
        stream=stream,
    )


def _causal_flops(spec: AttentionDenseSpec) -> int:
    B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    Hq, D = spec.num_query_heads, spec.head_size
    W = spec.sliding_window
    if spec.causal and W > 0:
        pairs = W * Sq - W * (W - 1) // 2 if Sq >= W else Sq * (Sq + 1) // 2
        return 4 * B * Hq * D * pairs
    if spec.causal:
        return 4 * B * Hq * D * (Sq * (Sq + 1) // 2)
    return 2 * 2 * B * Hq * D * Sq * Skv


def make_spec_from_shape(shape: dict[str, Any]) -> AttentionDenseSpec:
    """Build a validated ``AttentionDenseSpec`` from a benchmark shape mapping."""
    defaults = DENSE_TILE_GEOMETRIES["default"]
    geometry = {
        "block_m": int(shape.get("block_m", defaults["block_m"])),
        "block_n": int(shape.get("block_n", defaults["block_n"])),
        "lds_v_row_pad": int(
            shape.get("lds_v_row_pad", defaults["lds_v_row_pad"])
        ),
    }
    return AttentionDenseSpec(
        batch=int(shape.get("batch", 1)),
        seqlen_q=int(shape["seqlen_q"]),
        seqlen_kv=int(shape.get("seqlen_kv", shape["seqlen_q"])),
        num_query_heads=int(shape["num_query_heads"]),
        num_kv_heads=int(shape["num_kv_heads"]),
        head_size=int(shape["head_size"]),
        causal=bool(shape.get("causal", True)),
        dtype=str(shape.get("dtype", "fp16")),
        **geometry,
        waves_per_eu=int(shape.get("waves_per_eu", 2)),
        persistent=bool(shape.get("persistent", False)),
        num_persistent=int(shape.get("num_persistent", 256)),
        interleave=bool(shape.get("interleave", False)),
        sliding_window=int(shape.get("sliding_window", 0)),
        use_sinks=bool(shape.get("use_sinks", False)),
        lazy_rescale=bool(shape.get("lazy_rescale", True)),
        persist_decode=str(shape.get("persist_decode", "auto")),
        wide_lds_dma=bool(shape.get("wide_lds_dma", False)),
    )


def run_benchmark(
    spec: AttentionDenseSpec,
    *,
    warmup: int = 15,
    iters: int = 50,
    seed: int = 0,
    check: bool = True,
) -> dict[str, Any]:
    ms, tf, err = run(spec, warmup=warmup, iters=iters, check=check, seed=seed)
    ok = (not check) or (err < _TOL)
    return {
        "kernel_name": spec.kernel_name(),
        "persist_decode": spec.resolved_persist_decode,
        "ms": ms,
        "tflops": tf,
        "max_abs": err,
        "ok": ok,
        "flops": _causal_flops(spec),
        "grid": attention_dense_grid(spec),
        "block": attention_dense_block(spec),
    }


def run(
    spec: AttentionDenseSpec,
    *,
    warmup: int = 15,
    iters: int = 50,
    check: bool = True,
    seed: int = 0,
):
    dev = "cuda"
    dt = _TORCH_DT[spec.dtype]
    B, Sq, Skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    Hq, Hkv, D = spec.num_query_heads, spec.num_kv_heads, spec.head_size
    torch.manual_seed(seed)
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
            # Query-chunked attention to avoid OOM on large seqlens.
            # Computing attn=[B, Hq, Sq, Skv] all at once allocates B*Hq*Sq*Skv*4 bytes
            # (fp32), e.g., 34.4 GB for Sq=8192, Hq=128, then torch.cat and torch.softmax
            # each allocate another copy. Chunking over queries caps memory at ~1 GiB per
            # chunk and is exact (softmax normalizes along keys, so query rows are
            # independent).
            ki = torch.arange(Skv, device=dev).view(1, -1)
            sink_col = sinks.float().view(1, Hq, 1, 1)
            # Cap each chunk at ~1 GiB of scores (B*Hq*q_blk*(Skv+1)*4 bytes)
            q_blk = max(1, min(Sq, (1 << 30) // max(1, B * Hq * (Skv + 1) * 4)))
            ref = torch.empty_like(qh)
            for q0 in range(0, Sq, q_blk):
                q1 = min(q0 + q_blk, Sq)
                qn = q1 - q0
                attn = torch.einsum("bhqd,bhkd->bhqk", qh[:, :, q0:q1], kh) / math.sqrt(
                    D
                )
                if spec.causal or W > 0:
                    # Global query indices: chunk-local arange slides the mask boundary
                    qi = torch.arange(q0, q1, device=dev).view(-1, 1)
                    mask = torch.zeros(qn, Skv, dtype=torch.bool, device=dev)
                    if spec.causal:
                        mask |= ki > qi  # Causal: mask future tokens
                    if W > 0:
                        mask |= ki <= (
                            qi - W
                        )  # Sliding window: mask tokens beyond window
                    attn.masked_fill_(mask.view(1, 1, qn, Skv), float("-inf"))
                attn = torch.cat([attn, sink_col.expand(B, Hq, qn, 1)], dim=-1)
                attn = torch.softmax(attn, dim=-1)[
                    ..., :-1
                ]  # Softmax then drop sink column
                ref[:, :, q0:q1] = torch.einsum("bhqk,bhkd->bhqd", attn, vh)
            ref = ref.transpose(1, 2)

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

    ms = time_launches(call, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    flops = _causal_flops(spec)
    tf = flops / (ms * 1e-3) / 1e12
    status = "PASS" if (not check or err < _TOL) else "FAIL"
    print(
        f"[{spec.kernel_name()}] {ms:.4f} ms  {tf:.1f} TFLOPS  max_abs={err:.2e}  {status}"
    )
    return ms, tf, err


def main():
    ap = argparse.ArgumentParser()
    defaults = DENSE_TILE_GEOMETRIES["default"]
    ap.add_argument(
        "--block-m",
        "--bm",
        dest="block_m",
        type=int,
        default=defaults["block_m"],
        help="block_m (query rows per CTA)",
    )
    ap.add_argument(
        "--bn",
        type=int,
        default=defaults["block_n"],
        help="block_n (KV tile)",
    )
    ap.add_argument(
        "--lds-v-row-pad",
        "--vpad",
        dest="lds_v_row_pad",
        type=int,
        default=defaults["lds_v_row_pad"],
        help="D128 V-row LDS padding in bf16 elements",
    )
    ap.add_argument("--wpe", type=int, default=2, help="waves_per_eu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--hq", type=int, default=128)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--sq", type=int, default=None, help="seqlen (exact-shape mode)")
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
        "--persist-decode",
        default="auto",
        choices=[
            "auto",
            "qb_major",
            "hkv_major",
            "gqa_pair",
            "gqa_pair_2phase",
        ],
    )
    ap.add_argument(
        "--sw", type=int, default=0, help="sliding_window (0=off; multiple of --bn)"
    )
    ap.add_argument("--use-sinks", action="store_true", help="enable attention sinks")
    ap.add_argument(
        "--wide-lds-dma",
        action="store_true",
        help="use gfx950 dwordx4 buffer-to-LDS slab loading",
    )
    ap.add_argument(
        "--exact-shape",
        action="store_true",
        help="run only --sq (default 8192) once; emit JSON with --json-out",
    )
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    if args.exact_shape:
        sq = args.sq if args.sq is not None else 8192
        shape = {
            "batch": 1,
            "seqlen_q": sq,
            "seqlen_kv": sq,
            "num_query_heads": args.hq,
            "num_kv_heads": args.hkv,
            "head_size": args.d,
            "causal": bool(args.causal),
            "dtype": args.dtype,
            "block_m": args.block_m,
            "block_n": args.bn,
            "lds_v_row_pad": args.lds_v_row_pad,
            "waves_per_eu": args.wpe,
            "persistent": args.persistent,
            "num_persistent": args.np,
            "interleave": args.interleave,
            "sliding_window": args.sw,
            "use_sinks": args.use_sinks,
            "persist_decode": args.persist_decode,
            "wide_lds_dma": args.wide_lds_dma,
        }
        spec = make_spec_from_shape(shape)
        result = run_benchmark(
            spec,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            check=not args.no_check,
        )
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
        return 0 if result["ok"] else 2

    for sq in (256, 512, 2048, 8192):
        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=sq,
            seqlen_kv=sq,
            num_query_heads=args.hq,
            num_kv_heads=args.hkv,
            head_size=args.d,
            causal=bool(args.causal),
            dtype=args.dtype,
            block_m=args.block_m,
            block_n=args.bn,
            lds_v_row_pad=args.lds_v_row_pad,
            waves_per_eu=args.wpe,
            persistent=args.persistent,
            num_persistent=args.np,
            interleave=args.interleave,
            sliding_window=args.sw,
            use_sinks=args.use_sinks,
            persist_decode=args.persist_decode,
            wide_lds_dma=args.wide_lds_dma,
        )
        run(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
