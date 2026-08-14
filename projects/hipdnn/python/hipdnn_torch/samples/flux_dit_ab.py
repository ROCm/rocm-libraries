#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
flux_dit_ab.py -- run a Flux-style **DiT (diffusion transformer)** block stack and A/B the
forward native vs the hipdnn_torch injection layer. This is the **modern rmsnorm-heavy DiT**
entry: unlike a UNet, Flux carries the diffusion signal through a pure transformer, and its
signature detail is **QK-normalization with RMSNorm** inside attention (``q``/``k`` are
RMSNorm'd over the head_dim before SDPA). So this block stresses RMSNorm at a **4-D
``[B,H,S,D]``** activation -- a different rank than Llama's 3-D ``[B,S,D]`` token RMSNorm --
alongside adaLN-Zero modulation (a ``SiLU``+``Linear`` producing per-block shift/scale/gate),
gemm-heavy attention/MLP, non-causal SDPA, and GELU.

Self-contained (plain ``torch.nn``; no ``diffusers``) so it stays small and can't drag a
mismatched torch into the ROCm nightly venv -- the real FLUX.1 checkpoints are ~24 GB and the
point here is the routed op mix, not the weights. Geometry is a scaled-down single-stream DiT
(dim=1536, 24 heads, head_dim=64, mlp=4x). Random weights are fine for routing/coverage.

Always run under ``HIPDNN_TORCH_SELECT=default`` with all providers co-loaded. Set
``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`` (SDPA + gfx1151 aotriton
crash, see ``llama_block_ab.py``).

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all .dll> \
        python samples/flux_dit_ab.py --ops linear,rmsnorm,sdpa,silu,gelu

Knobs: FLUX_DIM FLUX_HEADS FLUX_SEQ FLUX_LAYERS FLUX_B FLUX_WARMUP FLUX_ITERS.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402


def build_flux_dit(torch, dim, heads, seq, layers, device, dtype):
    """A scaled-down Flux single-stream DiT stack (adaLN-Zero + QK-RMSNorm attention)."""
    nn = torch.nn
    F = torch.nn.functional
    head_dim = dim // heads
    mlp = 4 * dim
    assert dim % heads == 0, "dim must be divisible by heads"

    class RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return F.rms_norm(x, (x.shape[-1],), self.weight, eps=1e-6)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            # adaLN-Zero modulation: SiLU(vec) -> Linear -> 6 chunks
            self.mod = nn.Linear(dim, 6 * dim)
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.q_norm = RMSNorm(head_dim)  # QK-norm (the Flux signature)
            self.k_norm = RMSNorm(head_dim)
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.fc = nn.Linear(dim, mlp)
            self.out = nn.Linear(mlp, dim)

        def forward(self, x, vec):
            b, s, _ = x.shape
            mod = self.mod(F.silu(vec))  # [b, 6*dim]
            sa, sc, sg, fa, fc_, fg = mod.chunk(6, dim=-1)

            h = self.norm1(x) * (1 + sc.unsqueeze(1)) + sa.unsqueeze(1)
            qkv = self.qkv(h).view(b, s, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [b, heads, s, head_dim]
            q = self.q_norm(q)  # RMSNorm over head_dim (4-D input)
            k = self.k_norm(k)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            a = a.transpose(1, 2).reshape(b, s, dim)
            x = x + sg.unsqueeze(1) * self.proj(a)

            h = self.norm2(x) * (1 + fc_.unsqueeze(1)) + fa.unsqueeze(1)
            h = self.out(F.gelu(self.fc(h)))
            x = x + fg.unsqueeze(1) * h
            return x

    class FluxDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        def forward(self, x, vec):
            for blk in self.blocks:
                x = blk(x, vec)
            return self.norm_out(x)

    return FluxDiT().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops", default="linear,rmsnorm,sdpa,silu,gelu", help="overrides to route"
    )
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print("provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr)
        return 1

    global torch
    import torch

    dtype = torch.bfloat16
    device = torch.device("cuda")
    dim = int(os.environ.get("FLUX_DIM", "1536"))
    heads = int(os.environ.get("FLUX_HEADS", "24"))
    seq = int(os.environ.get("FLUX_SEQ", "256"))
    layers = int(os.environ.get("FLUX_LAYERS", "4"))
    batch = int(os.environ.get("FLUX_B", "1"))
    warmup = int(os.environ.get("FLUX_WARMUP", "2"))
    iters = int(os.environ.get("FLUX_ITERS", "5"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    model = build_flux_dit(torch, dim, heads, seq, layers, device, dtype)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(batch, seq, dim, generator=gen, dtype=torch.float32).to(dtype).to(device)
    vec = torch.randn(batch, dim, generator=gen, dtype=torch.float32).to(dtype).to(device)
    print(
        f"config  = Flux-DiT dim={dim} heads={heads} head_dim={dim // heads} mlp={4*dim} "
        f"layers={layers}  tokens[{batch},{seq},{dim}] dtype={dtype} (QK-RMSNorm, adaLN-Zero)"
    )
    print(f"routing = {ops}")
    print()

    def fwd():
        with torch.no_grad():
            return model(x, vec)

    with torch.no_grad():
        hipdnn_torch.uninstall()
        y_native = fwd().float()
        hipdnn_torch.install(ops)
        hipdnn_torch.reset()
        y_over = fwd().float()
        hipdnn_torch.uninstall()

    print(f"output  = {tuple(y_native.shape)}")
    fin_n = bool(torch.isfinite(y_native).all())
    fin_o = bool(torch.isfinite(y_over).all())
    if fin_n and fin_o:
        max_err = float((y_native - y_over).abs().max().item())
        denom = float(y_native.abs().max().item()) or 1.0
        rel = max_err / denom
        print(
            f"correctness native-vs-injected: {'OK ' if rel < 8e-2 else 'BAD'} "
            f"max_abs_err={max_err:.5f} rel={rel:.4f}"
        )
    else:
        print(
            f"correctness: parity not measurable "
            f"(native finite={fin_n}, injected finite={fin_o})"
        )
    print()
    print(hipdnn_torch.report(ops))
    print()

    def timed():
        with torch.no_grad():
            for _ in range(warmup):
                model(x, vec)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x, vec)
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
