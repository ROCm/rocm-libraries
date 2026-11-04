#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
llama_block_ab.py -- run a stack of Llama/Mistral-style decoder blocks and A/B the
forward native vs the hipdnn_torch injection layer.

This is the **LLM decoder** sample: each block is the modern decoder recipe --
``F.rms_norm`` (pre-norm) -> RoPE -> GQA **causal** ``F.scaled_dot_product_attention``
(q/k/v/o are ``F.linear``) -> residual -> ``F.rms_norm`` -> SwiGLU MLP
(``down(F.silu(gate(x)) * up(x))``, all ``F.linear``) -> residual. Geometry defaults to
Llama-3.2-1B (dim=2048, 32 q-heads / 8 kv-heads, head_dim=64, mlp=8192).

It is the best 1:1 match to the hipDNN op families (gemm, fmha/sdpa, rmsnorm, silu). Always
run under ``HIPDNN_TORCH_SELECT=default`` with all providers co-loaded (AOT/hipblaslt/
MIOpen): that measures *general hipDNN coverage* -- "did hipDNN route this op at all, by
whatever engine won" -- which is the portable, arch-independent signal. (Do NOT use
``force``: pinning ``AOT_CATALOG_ENGINE`` only reports AOT-specific attribution, a
gfx1151-only story, and turns every non-AOT-served op into a false decline.)

The honest coverage caveat still holds under ``default``: a real decoder runs on 3-D
``[B,T,D]`` activations at true rank (the pure-passthrough contract keeps them 3-D), and a
handful of shapes have NO loaded engine yet (3-D weighted rms_norm, N-D linear+bias, causal
SDPA at some head configs -- see ``tests/test_parity.py`` xfails). Those fall back to native
and show up in the report -- that decline list IS the point (coverage backlog).

Weights use nn.Linear's default (small) init and RMSNorm weight = 1, so bf16 parity is
meaningful across a few stacked blocks. Random weights are used on purpose -- the
checkpoint changes values, not the call sites / shapes / op mix that route to hipDNN.

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all provider .so/.dll> \
        python samples/llama_block_ab.py --ops linear,rmsnorm,sdpa,silu

Knobs: LLAMA_DIM LLAMA_HEADS LLAMA_KV_HEADS LLAMA_MLP LLAMA_SEQ LLAMA_LAYERS LLAMA_B
LLAMA_WARMUP LLAMA_ITERS. NOTE on ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL``: it normally
makes the *native* SDPA baseline a real fused kernel instead of the math strawman -- BUT on
this gfx1151 ROCm build the experimental
aotriton fused SDPA crashes for every tested shape (it enqueues a bad launch that poisons the
HIP context, so the *next* op fails with ``HIP error: invalid argument``). Leave it OFF here;
the math backend gives a correct native baseline. Re-enable once the fused kernel is fixed.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402


def build_llama(torch, dim, heads, kv_heads, mlp, seq, layers, device, dtype):
    """A canonical Llama/Mistral decoder stack (pre-norm, RoPE, GQA, SwiGLU)."""
    nn = torch.nn
    F = torch.nn.functional
    head_dim = dim // heads
    assert heads % kv_heads == 0, "heads must be a multiple of kv_heads (GQA)"
    n_rep = heads // kv_heads

    # ---- RoPE tables (elementwise; not a routed op) ---------------------------
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    cos = emb.cos().to(device=device, dtype=dtype)
    sin = emb.sin().to(device=device, dtype=dtype)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(x):  # x: [b, h, s, d]
        return x * cos[None, None] + rotate_half(x) * sin[None, None]

    class RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return F.rms_norm(x, (dim,), self.w, 1e-5)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = RMSNorm()
            self.q = nn.Linear(dim, heads * head_dim, bias=False)
            self.k = nn.Linear(dim, kv_heads * head_dim, bias=False)
            self.v = nn.Linear(dim, kv_heads * head_dim, bias=False)
            self.o = nn.Linear(heads * head_dim, dim, bias=False)
            self.n2 = RMSNorm()
            self.gate = nn.Linear(dim, mlp, bias=False)
            self.up = nn.Linear(dim, mlp, bias=False)
            self.down = nn.Linear(mlp, dim, bias=False)

        def forward(self, x):
            b, s, _ = x.shape
            h = self.n1(x)
            q = self.q(h).view(b, s, heads, head_dim).transpose(1, 2)
            k = self.k(h).view(b, s, kv_heads, head_dim).transpose(1, 2)
            v = self.v(h).view(b, s, kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rope(q), apply_rope(k)
            # GQA: expand kv heads to match q heads (classic repeat_kv).
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(b, s, heads * head_dim)
            x = x + self.o(a)
            h = self.n2(x)
            x = x + self.down(F.silu(self.gate(h)) * self.up(h))
            return x

    class Llama(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.norm = RMSNorm()

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    return Llama().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops",
        default="linear,rmsnorm,sdpa,silu",
        help="comma-separated overrides to route",
    )
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print(
            "provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr
        )
        return 1

    global torch
    import torch

    dtype = torch.bfloat16
    device = torch.device("cuda")
    dim = int(os.environ.get("LLAMA_DIM", "2048"))
    heads = int(os.environ.get("LLAMA_HEADS", "32"))
    kv_heads = int(os.environ.get("LLAMA_KV_HEADS", "8"))
    mlp = int(os.environ.get("LLAMA_MLP", "8192"))
    seq = int(os.environ.get("LLAMA_SEQ", "512"))
    layers = int(os.environ.get("LLAMA_LAYERS", "4"))
    batch = int(os.environ.get("LLAMA_B", "1"))
    warmup = int(os.environ.get("LLAMA_WARMUP", "3"))
    iters = int(os.environ.get("LLAMA_ITERS", "10"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    model = build_llama(torch, dim, heads, kv_heads, mlp, seq, layers, device, dtype)
    gen = torch.Generator(device="cpu").manual_seed(3)
    x = (
        torch.randn(batch, seq, dim, generator=gen, dtype=torch.float32)
        .to(dtype)
        .to(device)
    )
    print(
        f"config  = Llama dim={dim} heads={heads} kv_heads={kv_heads} "
        f"head_dim={dim // heads} mlp={mlp} layers={layers}  "
        f"input[{batch},{seq},{dim}] dtype={dtype}"
    )
    print(f"routing = {ops}")
    print()

    def fwd():
        with torch.no_grad():
            return model(x)

    # ---- correctness: native vs injected on identical input/weights ----------
    with torch.no_grad():
        hipdnn_torch.uninstall()
        y_native = fwd().float()
        hipdnn_torch.install(ops)
        hipdnn_torch.reset()
        y_over = fwd().float()
        hipdnn_torch.uninstall()

    fin_n = bool(torch.isfinite(y_native).all())
    fin_o = bool(torch.isfinite(y_over).all())
    if fin_n and fin_o:
        max_err = float((y_native - y_over).abs().max().item())
        denom = float(y_native.abs().max().item()) or 1.0
        print(
            f"correctness native-vs-injected: "
            f"{'OK ' if max_err / denom < 8e-2 else 'BAD'} "
            f"max_abs_err={max_err:.5f} rel={max_err/denom:.4f}"
        )
    else:
        print(
            f"correctness: parity not measurable "
            f"(native finite={fin_n}, injected finite={fin_o})"
        )
    print()
    print(hipdnn_torch.report(ops))
    print()

    # ---- A/B forward wall-clock -----------------------------------------------
    def timed():
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3

    hipdnn_torch.uninstall()
    native_ms = timed()
    hipdnn_torch.install(ops)
    over_ms = timed()
    hipdnn_torch.uninstall()
    print(f"A/B forward wall-clock (warmup={warmup} iters={iters}):")
    print(f"  native   = {native_ms:8.3f} ms/fwd")
    print(
        f"  injected = {over_ms:8.3f} ms/fwd   "
        f"speedup={native_ms/over_ms if over_ms else float('nan'):5.3f}x"
    )
    print()

    print("=== device-time census: one forward, by op category (cuda events) ===")
    print(device_time_census(torch, fwd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
