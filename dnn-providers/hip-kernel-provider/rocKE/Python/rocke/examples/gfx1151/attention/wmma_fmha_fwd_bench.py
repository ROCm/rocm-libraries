# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark the gfx1151 WMMA FMHA forward on a Strix Halo node.

Builds the RDNA3.5 WMMA attention kernel, verifies it once against a numpy
dense-attention reference, then times the launch with HIP events and reports
latency and achieved attention throughput. Companion to
``wmma_fmha_fwd_verify`` (same kernel and ABI); this one measures the kernel
that stages each K-tile's V rows through LDS instead of gathering V column-wise
from global memory.

Must run on a gfx1151 device.

    PYTHONPATH=Python python3 -m rocke.examples.gfx1151.attention.wmma_fmha_fwd_bench \
        --seqlen-q 512 --seqlen-k 512 --head-size 128 --heads 8 --batch 4
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct

from rocke.helpers import compile_kernel
from rocke.instances.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    build_wmma_fmha_fwd,
    wmma_fmha_fwd_grid,
)
from rocke.runtime.hip_module import Runtime
from rocke.runtime.launcher import time_launches


def _ref_attention(Q, K, V, *, causal: bool):
    """Dense attention reference, Q/K/V shape ``(seqlen, heads, head_size)``."""
    import numpy as np

    d = Q.shape[-1]
    scores = np.einsum("ihd,jhd->ihj", Q.astype(np.float32), K.astype(np.float32))
    scores /= math.sqrt(d)
    if causal:
        q_pos = np.arange(Q.shape[0])[:, None, None]
        k_pos = np.arange(K.shape[0])[None, None, :]
        scores = np.where(k_pos <= q_pos, scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", probs, V.astype(np.float32))
    return out.astype(np.float16)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1151")
    p.add_argument("--seqlen-q", type=int, default=512)
    p.add_argument("--seqlen-k", type=int, default=512)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=0, help="0 -> MHA (== heads)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    import numpy as np

    for d, name in (
        (args.seqlen_q, "seqlen_q"),
        (args.seqlen_k, "seqlen_k"),
        (args.head_size, "head_size"),
    ):
        if d % 16:
            raise SystemExit(f"{name}={d} must be a multiple of 16 (WMMA 16x16 tile)")

    kvh = args.kv_heads or args.heads
    spec = WmmaFmhaFwdSpec(
        head_size=args.head_size,
        num_query_heads=args.heads,
        num_kv_heads=kvh,
        mask_mode="causal" if args.causal else "none",
        name=f"wmma_fmha_bench_{args.arch}",
    )
    art = compile_kernel(build_wmma_fmha_fwd(spec, arch=args.arch), arch=args.arch)
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa})")

    B, Hq, Hk, D = args.batch, args.heads, kvh, args.head_size
    Sq, Sk = args.seqlen_q, args.seqlen_k
    rng = np.random.default_rng(0xA11E)

    Q = (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, Sq, Hq, D), dtype=np.float16)

    stride_q_token = Hq * D
    stride_q_head = D
    stride_k_token = Hk * D
    stride_k_head = D
    stride_v_token = Hk * D
    stride_v_head = D
    stride_o_token = Hq * D
    stride_o_head = D
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    grid = wmma_fmha_fwd_grid(spec, seqlen_q=Sq, batch=B)
    block = (spec.block_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd = rt.alloc(Q.nbytes)
    kd = rt.alloc(K.nbytes)
    vd = rt.alloc(V.nbytes)
    od = rt.alloc(Out.nbytes)
    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(K), K.nbytes)
    rt.memcpy_h2d(vd, u8(V), V.nbytes)
    rt.memset(od, 0, Out.nbytes)

    packed = struct.pack(
        "<QQQQfiiiiiiiiii",
        qd,
        kd,
        vd,
        od,
        scale_log2,
        Sq,
        Sk,
        stride_q_token,
        stride_q_head,
        stride_k_token,
        stride_k_head,
        stride_v_token,
        stride_v_head,
        stride_o_token,
        stride_o_head,
    )

    # ---- Correctness gate (one launch) before timing ----
    rt.launch(fn, grid, block, packed)
    rt.sync()
    rt.memcpy_d2h(u8(Out), od, Out.nbytes)
    ref = np.empty_like(Out)
    for bi in range(B):
        if Hk != Hq:
            rep = Hq // Hk
            Kb = np.repeat(K[bi], rep, axis=1)
            Vb = np.repeat(V[bi], rep, axis=1)
        else:
            Kb, Vb = K[bi], V[bi]
        ref[bi] = _ref_attention(Q[bi], Kb, Vb, causal=args.causal)
    max_abs = float(np.abs(Out.astype(np.float32) - ref.astype(np.float32)).max())
    if max_abs > args.tol:
        print(f"[{args.arch}] correctness FAIL max_abs={max_abs:.3e} > {args.tol:.0e}")
        return 1

    # ---- Timed loop ----
    ms = time_launches(
        lambda: rt.launch(fn, grid, block, packed),
        warmup=args.warmup,
        iters=args.iters,
    )

    # Attention fwd FLOPs: QK^T (2*B*Hq*Sq*Sk*D) + PV (2*B*Hq*Sq*Sk*D).
    flops = 4.0 * B * Hq * Sq * Sk * D
    if args.causal:
        flops *= 0.5  # ~half the (q,k) pairs are masked out
    tflops = flops / (ms * 1e-3) / 1e12

    # HBM traffic lower bound: read Q/K/V once, write O once (fp16).
    bytes_io = Q.nbytes + K.nbytes + V.nbytes + Out.nbytes
    gbps = bytes_io / (ms * 1e-3) / 1e9

    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    module.unload()

    print(
        f"[{args.arch}] WMMA FMHA B={B} Sq={Sq} Sk={Sk} D={D} Hq={Hq} Hk={Hk} "
        f"causal={args.causal}: max_abs={max_abs:.2e} grid={grid} | "
        f"{ms * 1e3:.1f} us/iter  {tflops:.2f} TFLOP/s  {gbps:.1f} GB/s (IO-floor)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
