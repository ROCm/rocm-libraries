# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numeric-verify and time the gfx1151 paged split-K decode kernel.

Runs the torch custom op ``rocke_gfx1151::paged_decode_splitk`` against an fp32
oracle and, optionally, against vLLM's incumbent ``ops.paged_attention_rocm``:

    PYTHONPATH=python python3 -m builders.gfx1151.attention.paged_decode_splitk_verify \\
        --batch 8 --seqlen-k 4096 --num-splits 0 --bench

Two deliberate hostilities in the fake cache, both because the failure mode of
a paged kernel is plausible-looking wrong numbers rather than a crash:

* the block table is **shuffled by default** -- an identity table lets a kernel
  that ignores the indirection entirely still pass;
* every byte the kernel must not read (slack pages, the padded tail of the last
  page of each request) is **poisoned with 7777.0**, so an off-by-one page or
  slot blows past any tolerance instead of quietly attenuating the output.

Must execute on a gfx1151 board.
"""

from __future__ import annotations

import argparse
import math
import time


def _paged_layout(torch, K, V, seq_lens, kv_block_size, *, shuffle, poison=7777.0):
    """Scatter logical K/V into the paged layout vLLM's ROCm backend writes.

    Returns ``(k_cache, v_cache, block_table)`` with::

        k_cache [num_blocks, Hk, D/x, BS, x]   x = 8   (2-byte elements)
        v_cache [num_blocks, Hk, D, BS]                slot FASTEST

    ``K``/``V`` are ``[B, Sk_max, Hk, D]``; only the first ``seq_lens[b]`` rows
    of each request are scattered, so the tail of the last page stays poison.
    """
    B, Sk_max, Hk, D = K.shape
    X = 8
    blocks_per_req = (Sk_max + kv_block_size - 1) // kv_block_size
    needed = B * blocks_per_req
    # Slack pages exist only to be poison, and to push the used ids off the
    # identity mapping.
    num_blocks = needed + max(4, needed // 2)

    g = torch.Generator(device="cpu").manual_seed(0x9A9ED)
    ids = (
        torch.randperm(num_blocks, generator=g)[:needed]
        if shuffle
        else torch.arange(needed)
    )
    block_table = ids.to(torch.int32).reshape(B, blocks_per_req)

    dev, dt = K.device, K.dtype
    k_cache = torch.full(
        (num_blocks, Hk, D // X, kv_block_size, X), poison, dtype=dt, device=dev
    )
    v_cache = torch.full(
        (num_blocks, Hk, D, kv_block_size), poison, dtype=dt, device=dev
    )

    # Page-aligned staging buffers, poison everywhere the request does not
    # reach: past its seq_len, and past Sk_max in the padded tail.
    padded = blocks_per_req * kv_block_size
    kpad = torch.full((B, padded, Hk, D), poison, dtype=dt, device=dev)
    vpad = torch.full((B, padded, Hk, D), poison, dtype=dt, device=dev)
    for b in range(B):
        n = int(seq_lens[b])
        kpad[b, :n] = K[b, :n]
        vpad[b, :n] = V[b, :n]

    ids = block_table.reshape(-1).to(dev).long()
    # [B, blocks, BS, Hk, D] -> the two cache orders.
    ksrc = kpad.reshape(B * blocks_per_req, kv_block_size, Hk, D // X, X)
    k_cache[ids] = ksrc.permute(0, 2, 3, 1, 4).contiguous()
    vsrc = vpad.reshape(B * blocks_per_req, kv_block_size, Hk, D)
    v_cache[ids] = vsrc.permute(0, 2, 3, 1).contiguous()
    return k_cache, v_cache, block_table.to(dev)


def _oracle(torch, Q, K, V, seq_lens, scale):
    """fp32 decode-attention oracle, ported from paged-attention-decode.

    Same shape as ``paged_attention_decode_reference``: per request, gather the
    live keys, expand the kv heads across the GQA group, subtract the row max,
    softmax, weighted-sum. Loop-based on purpose -- this is a correctness
    oracle, not a baseline.
    """
    B, Hq, D = Q.shape
    Hk = K.shape[2]
    rep = Hq // Hk
    out = torch.zeros((B, Hq, D), dtype=torch.float32, device=Q.device)
    for b in range(B):
        n = int(seq_lens[b])
        if n == 0:
            continue
        k = K[b, :n].to(torch.float32).repeat_interleave(rep, dim=1)  # [n, Hq, D]
        v = V[b, :n].to(torch.float32).repeat_interleave(rep, dim=1)
        scores = torch.einsum("hd,shd->hs", Q[b].to(torch.float32), k) * scale
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hs,shd->hd", probs, v)
    return out


def _time_op(torch, fn, warmup, iters):
    """Minimum wall time across ``iters`` reps, in microseconds.

    Minimum, not median: the box is shared, so every sample is contaminated
    upward and only the floor is a property of the kernel.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6


def _run_case(torch, args, B, Sk, num_splits, *, verbose=True):
    from .paged_decode_torch_ops import register_torch_custom_ops

    register_torch_custom_ops()

    Hq, Hk, D = args.heads, args.kv_heads, args.head_size
    dev = torch.device("cuda")
    dt = torch.float16 if args.dtype == "f16" else torch.bfloat16
    scale = 1.0 / math.sqrt(D)

    gen = torch.Generator(device=dev).manual_seed(0xA11E + B * 1000 + Sk)
    Sk_max = max(Sk, 1)
    Q = (
        torch.randn((B, Hq, D), generator=gen, device=dev, dtype=torch.float32) * 0.3
    ).to(dt)
    K = (
        torch.randn((B, Sk_max, Hk, D), generator=gen, device=dev, dtype=torch.float32)
        * 0.3
    ).to(dt)
    V = (
        torch.randn((B, Sk_max, Hk, D), generator=gen, device=dev, dtype=torch.float32)
        * 0.3
    ).to(dt)
    seq_lens = torch.full((B,), Sk, dtype=torch.int32, device="cpu")

    k_cache, v_cache, block_table = _paged_layout(
        torch, K, V, seq_lens, args.kv_block_size, shuffle=bool(args.shuffle_blocks)
    )
    seq_lens_d = seq_lens.to(dev)
    out = torch.full((B, Hq, D), float("nan"), dtype=dt, device=dev)

    def launch():
        torch.ops.rocke_gfx1151.paged_decode_splitk(
            Q,
            k_cache,
            v_cache,
            block_table,
            seq_lens_d,
            out,
            scale,
            num_splits,
            args.d_lanes,
            args.block_n,
            args.unroll,
            args.heads_per_cta,
            args.waves_per_eu,
            args.reduce_unroll,
        )

    launch()
    torch.cuda.synchronize()

    ref = _oracle(torch, Q, K, V, seq_lens, scale)
    got = out.to(torch.float32)
    nan = int(torch.isnan(got).sum())
    diff = (got - ref).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    bad = int((diff > args.tol).sum())
    ok = nan == 0 and max_abs <= args.tol

    us = (
        _time_op(torch, launch, args.warmup, args.iters) if args.bench else float("nan")
    )

    if verbose:
        verdict = "PASS" if ok else "FAIL"
        line = (
            f"B={B:<3} Sk={Sk:<5} splits={num_splits:<3} "
            f"max_abs={max_abs:.3e} bad={bad}/{got.numel()} nan={nan}: {verdict}"
        )
        if args.bench:
            line += f"  {us:8.1f}us"
        print(line)
    return ok, max_abs, us


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--dtype", default="f16", choices=("f16", "bf16"))
    p.add_argument("--kv-block-size", type=int, default=16)
    p.add_argument("--batch", type=int, default=0, help="0 -> sweep")
    p.add_argument("--seqlen-k", type=int, default=0, help="0 -> sweep")
    p.add_argument("--num-splits", type=int, default=-1, help="-1 -> sweep, 0 -> auto")
    p.add_argument(
        "--shuffle-blocks",
        type=int,
        default=1,
        choices=(0, 1),
        help="scatter pages non-contiguously (default on: an identity table "
        "lets a kernel that ignores the block table still PASS)",
    )
    p.add_argument("--d-lanes", type=int, default=0, help="0 -> default (16)")
    p.add_argument("--block-n", type=int, default=0, help="0 -> derived from d-lanes")
    p.add_argument("--unroll", type=int, default=1, help="kv blocks per loop iteration")
    p.add_argument("--heads-per-cta", type=int, default=0, help="0 -> full GQA fusion")
    p.add_argument("--waves-per-eu", type=int, default=0, help="0 -> no hint")
    p.add_argument("--reduce-unroll", type=int, default=0, choices=(0, 1))
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("paged_decode_splitk_verify must run on a gfx1151 board")

    batches = [args.batch] if args.batch else [1, 2, 8, 32]
    # 0 must return zeros (not NaN); 1/15/16/17 are the partial-page and
    # split-boundary cases; 4095 is a non-multiple at scale.
    seqlens = [args.seqlen_k] if args.seqlen_k else [0, 1, 15, 16, 17, 1024, 4095, 8192]
    splits = [args.num_splits] if args.num_splits >= 0 else [1, 8, 16]

    failures = 0
    for B in batches:
        for Sk in seqlens:
            for ns in splits:
                ok, _, _ = _run_case(torch, args, B, Sk, ns)
                failures += 0 if ok else 1

    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
