#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
vit_ab.py -- run a standard Vision Transformer (ViT-B/16 shape) and A/B its forward
native vs the hipdnn_torch injection layer.

This is the **widest op-mix** sample: a ViT touches every override at once --
``F.conv2d`` (the patch-embed stem, NCHW), then per block ``F.layer_norm``,
``F.linear`` (qkv + proj + MLP), ``F.scaled_dot_product_attention``, and ``F.gelu``.
It is hand-built (no torchvision/timm dependency) but uses the canonical ViT-B/16
geometry, so the routed shapes match a real classifier.

Weights are random; the transformer path is numerically stable in bf16 (no group_norm
cancellation), so BOTH parity and the per-op census are meaningful here.

    HIPDNN_TORCH_PROVIDER_SO=<...>/libhip_kernel_provider.so \
        TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        python samples/vit_ab.py
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402


def build_vit(torch, img, patch, dim, depth, heads, device, dtype):
    """A canonical ViT: conv2d patch stem + pre-norm transformer encoder blocks."""
    nn = torch.nn
    F = torch.nn.functional
    n_tokens = (img // patch) ** 2 + 1  # + cls

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.n2 = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * 4)
            self.fc2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            b, n, d = x.shape
            h = self.n1(x)
            qkv = self.qkv(h).reshape(b, n, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [b, heads, n, d/heads] (permuted views)
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(1, 2).reshape(b, n, d)
            x = x + self.proj(a)
            h = self.n2(x)
            x = x + self.fc2(F.gelu(self.fc1(h)))
            return x

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
            self.cls = nn.Parameter(torch.randn(1, 1, dim))
            self.pos = nn.Parameter(torch.randn(1, n_tokens, dim))
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, 1000)

        def forward(self, x):
            x = self.stem(x).flatten(2).transpose(1, 2)  # [b, tokens, dim]
            cls = self.cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1) + self.pos
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return self.head(x[:, 0])

    return ViT().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops",
        default="conv2d,linear,sdpa,layernorm,gelu,silu,rmsnorm",
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
    batch = int(os.environ.get("VIT_B", "4"))
    img = int(os.environ.get("VIT_IMG", "224"))
    patch = int(os.environ.get("VIT_PATCH", "16"))
    dim = int(os.environ.get("VIT_DIM", "768"))
    depth = int(os.environ.get("VIT_DEPTH", "12"))
    heads = int(os.environ.get("VIT_HEADS", "12"))
    warmup = int(os.environ.get("VIT_WARMUP", "3"))
    iters = int(os.environ.get("VIT_ITERS", "10"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    model = build_vit(torch, img, patch, dim, depth, heads, device, dtype)
    gen = torch.Generator(device="cpu").manual_seed(3)
    x = (
        torch.randn(batch, 3, img, img, generator=gen, dtype=torch.float32)
        .to(dtype)
        .to(device)
    )
    print(
        f"config  = ViT dim={dim} depth={depth} heads={heads} "
        f"img={img}/patch{patch}  input[{batch},3,{img},{img}] dtype={dtype}"
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
