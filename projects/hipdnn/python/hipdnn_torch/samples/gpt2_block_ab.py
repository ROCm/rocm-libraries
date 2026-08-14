#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
gpt2_block_ab.py -- run a stack of GPT-2-style **classic** decoder blocks and A/B the forward
native vs the hipdnn_torch injection layer.

This is the pre-RMSNorm decoder recipe, deliberately complementing ``llama_block_ab.py`` on
the **LayerNorm + GELU** axis (Llama is RMSNorm + SiLU): each block is
``LayerNorm`` (pre-norm) -> causal ``F.scaled_dot_product_attention`` (fused QKV / output are
``F.linear``, multi-head, NO GQA, NO RoPE) -> residual -> ``LayerNorm`` -> MLP
(``proj(F.gelu(fc(x)))``, both ``F.linear``) -> residual. Geometry defaults to GPT-2 base
(dim=768, 12 heads, head_dim=64, mlp=3072).

Self-contained (no ``transformers``) on purpose: it avoids GPT-2's ``Conv1D`` (a transposed
matmul, not ``nn.Linear``) so the routed op mix is unambiguous, and it can't drag a
mismatched torch into the ROCm nightly venv. Random weights are fine for routing/coverage.

Always run under ``HIPDNN_TORCH_SELECT=default`` with all providers co-loaded. Expected
model-scale picture (mirrors llama, on the LN/GELU axis): the 3-D ``[B,T,D]`` ``linear`` and
3-D ``layer_norm`` and the causal ``sdpa`` all decline (no loaded engine yet) = backlog;
``gelu`` is the pointwise op most likely to route (AOT), analogous to llama's SiLU.

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all .dll> \
        python samples/gpt2_block_ab.py --ops linear,layernorm,sdpa,gelu

Knobs: GPT2_DIM GPT2_HEADS GPT2_MLP GPT2_SEQ GPT2_LAYERS GPT2_B GPT2_WARMUP GPT2_ITERS. Set
``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`` -- this block uses SDPA
and the experimental aotriton fused SDPA crashes on this gfx1151 ROCm build (see
``llama_block_ab.py``); the math backend is the correct native baseline.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402


def build_gpt2(torch, dim, heads, mlp, seq, layers, device, dtype):
    """A canonical GPT-2 decoder stack (pre-norm LayerNorm, causal MHA, GELU MLP)."""
    nn = torch.nn
    F = torch.nn.functional
    head_dim = dim // heads
    assert dim % heads == 0, "dim must be divisible by heads"

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.ln2 = nn.LayerNorm(dim)
            self.fc = nn.Linear(dim, mlp)
            self.out = nn.Linear(mlp, dim)

        def forward(self, x):
            b, s, _ = x.shape
            h = self.ln1(x)
            qkv = self.qkv(h).view(b, s, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # each [b, heads, s, head_dim]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(b, s, dim)
            x = x + self.proj(a)
            h = self.ln2(x)
            x = x + self.out(F.gelu(self.fc(h)))
            return x

    class GPT2(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.lnf = nn.LayerNorm(dim)

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return self.lnf(x)

    return GPT2().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ops", default="linear,layernorm,sdpa,gelu", help="overrides to route")
    ap.add_argument(
        "--dtype", default="bf16", choices=["bf16", "f16"], help="model/activation dtype"
    )
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print("provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr)
        return 1

    global torch
    import torch

    dtype = torch.float16 if args.dtype == "f16" else torch.bfloat16
    device = torch.device("cuda")
    dim = int(os.environ.get("GPT2_DIM", "768"))
    heads = int(os.environ.get("GPT2_HEADS", "12"))
    mlp = int(os.environ.get("GPT2_MLP", "3072"))
    seq = int(os.environ.get("GPT2_SEQ", "512"))
    layers = int(os.environ.get("GPT2_LAYERS", "6"))
    batch = int(os.environ.get("GPT2_B", "1"))
    warmup = int(os.environ.get("GPT2_WARMUP", "3"))
    iters = int(os.environ.get("GPT2_ITERS", "10"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    model = build_gpt2(torch, dim, heads, mlp, seq, layers, device, dtype)
    gen = torch.Generator(device="cpu").manual_seed(2)
    x = torch.randn(batch, seq, dim, generator=gen, dtype=torch.float32).to(dtype).to(device)
    print(
        f"config  = GPT-2 dim={dim} heads={heads} head_dim={dim // heads} mlp={mlp} "
        f"layers={layers}  input[{batch},{seq},{dim}] dtype={dtype}"
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
