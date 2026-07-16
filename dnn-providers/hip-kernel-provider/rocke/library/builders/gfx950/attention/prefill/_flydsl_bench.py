#!/usr/bin/env python3
"""Standalone flyDSL flash-attn timer (own process + interpreter).

Uses flyDSL **0.2.4** (pip-installed into /tmp/flydsl024-venv, torch inherited
from atom-venv via a .pth) + the flydsl-main HEAD kernels (also 0.2.4). This is
the DUALWAVE_SWP path: dense/varlen, GQA/MHA, causal. flyDSL prefill has NO
sliding window. Run this with /tmp/flydsl024-venv/bin/python and
PYTHONPATH=/workspace/flydsl-main.

Modes:
  dense  <B> <S> <Hq> <Hkv> <D> <causal 0/1> <warmup> <iters>
  varlen <seqlens_csv> <Hq> <Hkv> <D> <warmup> <iters>   (causal)
Prints: "OK <ms> <tflops> <max_abs>" or "ERR <message>".
"""
import sys

import torch


def _time(call, wu, it):
    for _ in range(wu):
        call()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        call()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / it


def main():
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func

    dev = "cuda"
    DT = torch.bfloat16
    torch.manual_seed(1)
    mode = sys.argv[1]

    if mode == "dense":
        B, S, Hq, Hkv, D = (int(sys.argv[i]) for i in range(2, 7))
        causal = bool(int(sys.argv[7]))
        wu, it = int(sys.argv[8]), int(sys.argv[9])
        q = (torch.randn(B, S, Hq, D, dtype=DT, device=dev) * 0.2).contiguous()
        k = (torch.randn(B, S, Hkv, D, dtype=DT, device=dev) * 0.2).contiguous()
        v = (torch.randn(B, S, Hkv, D, dtype=DT, device=dev) * 0.2).contiguous()

        def call():
            return flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=Hkv)

        o = call()
        torch.cuda.synchronize()
        rep = Hq // Hkv
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).repeat_interleave(rep, 1).float(),
            v.transpose(1, 2).repeat_interleave(rep, 1).float(),
            is_causal=causal,
        ).transpose(1, 2)
        err = (o.float() - ref).abs().max().item()
        ms = _time(call, wu, it)
        pairs = (S * (S + 1) // 2) if causal else S * S
        tf = 4 * B * Hq * D * pairs / (ms * 1e-3) / 1e12
        print(f"OK {ms:.6f} {tf:.3f} {err:.3e}")
        return

    if mode == "varlen":
        seqlens = [int(x) for x in sys.argv[2].split(",")]
        Hq, Hkv, D = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        wu, it = int(sys.argv[6]), int(sys.argv[7])
        B = len(seqlens)
        total = sum(seqlens)
        mx = max(seqlens)
        q = (torch.randn(total, Hq, D, dtype=DT, device=dev) * 0.2).contiguous()
        k = (torch.randn(total, Hkv, D, dtype=DT, device=dev) * 0.2).contiguous()
        v = (torch.randn(total, Hkv, D, dtype=DT, device=dev) * 0.2).contiguous()
        cu = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=dev).cumsum(0)

        def call():
            return flydsl_flash_attn_func(
                q,
                k,
                v,
                causal=True,
                num_kv_heads=Hkv,
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=mx,
                cross_seqlen=False,
            )

        o = call()
        torch.cuda.synchronize()
        rep = Hq // Hkv
        err = 0.0
        for i, s in enumerate(seqlens):
            st = sum(seqlens[:i])
            qh = q[st : st + s].transpose(0, 1).float().unsqueeze(0)
            kh = (
                k[st : st + s]
                .transpose(0, 1)
                .repeat_interleave(rep, 0)
                .float()
                .unsqueeze(0)
            )
            vh = (
                v[st : st + s]
                .transpose(0, 1)
                .repeat_interleave(rep, 0)
                .float()
                .unsqueeze(0)
            )
            ref = (
                torch.nn.functional.scaled_dot_product_attention(
                    qh, kh, vh, is_causal=True
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            err = max(err, (o[st : st + s].float() - ref).abs().max().item())
        ms = _time(call, wu, it)
        pairs = sum(x * (x + 1) // 2 for x in seqlens)
        tf = 4 * Hq * D * pairs / (ms * 1e-3) / 1e12
        print(f"OK {ms:.6f} {tf:.3f} {err:.3e}")
        return

    print(f"ERR unknown mode {mode!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"ERR {exc!r}")
