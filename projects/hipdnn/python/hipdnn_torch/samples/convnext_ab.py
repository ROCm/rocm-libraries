#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
convnext_ab.py -- run a ConvNeXt stack and A/B the forward native vs the hipdnn_torch
injection layer. This is the **modern CNN** bridge entry: it deliberately mixes the classic
conv family with the transformer norm/activation families, so it stresses op combinations the
pure-CNN (resnet50) and pure-transformer (gpt2/bert) entries don't.

Each ConvNeXt block is: **depthwise** ``Conv2d(dim,dim,k7,groups=dim)`` -> channels-last
``LayerNorm(dim)`` -> pointwise ``Linear(dim,4*dim)`` -> ``GELU`` -> ``Linear(4*dim,dim)`` ->
residual. Between stages: ``LayerNorm`` + strided ``Conv2d(k2,s2)`` downsample. The stem is a
patchify ``Conv2d(3,dim0,k4,s4)``. Two facts make this a distinctive coverage probe:
  * the convs are **grouped/depthwise** (``groups==dim``), which AOT's conv adapter (rank-4
    NHWC groups==1) cannot serve -- MIOpen has to take them under ``default``;
  * the pointwise MLP ``Linear`` sees a **4-D** ``[B,H,W,C]`` activation (channels-last), a
    different N-D linear shape than the 3-D ``[B,T,D]`` transformer linears.

Self-contained (plain ``torch.nn``; no ``torchvision``); random weights are fine for
routing/coverage. Always run under ``HIPDNN_TORCH_SELECT=default`` with all providers loaded.

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all .dll> \
        python samples/convnext_ab.py --ops conv2d,linear,layernorm,gelu

Knobs: CNX_IMG CNX_B CNX_WARMUP CNX_ITERS. No SDPA -> no aotriton flag needed.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402


def build_convnext(torch, dims, depths, device, dtype):
    """A canonical ConvNeXt stack (depthwise conv + channels-last LN + pointwise MLP)."""
    nn = torch.nn
    F = torch.nn.functional

    class Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.norm = nn.LayerNorm(dim)
            self.pw1 = nn.Linear(dim, 4 * dim)
            self.pw2 = nn.Linear(4 * dim, dim)

        def forward(self, x):
            h = self.dw(x)
            h = h.permute(0, 2, 3, 1)  # NCHW -> NHWC (channels-last for LN/Linear)
            h = self.norm(h)
            h = self.pw2(F.gelu(self.pw1(h)))
            h = h.permute(0, 3, 1, 2)  # back to NCHW
            return x + h

    class Downsample(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.norm = nn.LayerNorm(cin)
            self.conv = nn.Conv2d(cin, cout, kernel_size=2, stride=2)

        def forward(self, x):
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
            return self.conv(x)

    class ConvNeXt(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)
            self.stages = nn.ModuleList()
            self.downs = nn.ModuleList()
            for si, (dim, depth) in enumerate(zip(dims, depths)):
                self.stages.append(nn.Sequential(*[Block(dim) for _ in range(depth)]))
                if si < len(dims) - 1:
                    self.downs.append(Downsample(dim, dims[si + 1]))
            self.head_norm = nn.LayerNorm(dims[-1])
            self.head = nn.Linear(dims[-1], 1000)

        def forward(self, x):
            x = self.stem(x)
            for si, stage in enumerate(self.stages):
                x = stage(x)
                if si < len(self.downs):
                    x = self.downs[si](x)
            x = x.mean(dim=(2, 3))  # global avg pool
            return self.head(self.head_norm(x))

    return ConvNeXt().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ops", default="conv2d,linear,layernorm,gelu", help="overrides to route")
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print("provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr)
        return 1

    global torch
    import torch

    dtype = torch.bfloat16
    device = torch.device("cuda")
    img = int(os.environ.get("CNX_IMG", "224"))
    batch = int(os.environ.get("CNX_B", "8"))
    warmup = int(os.environ.get("CNX_WARMUP", "3"))
    iters = int(os.environ.get("CNX_ITERS", "10"))
    dims = [96, 192, 384]
    depths = [3, 3, 3]

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    model = build_convnext(torch, dims, depths, device, dtype)
    gen = torch.Generator(device="cpu").manual_seed(96)
    x = torch.randn(batch, 3, img, img, generator=gen, dtype=torch.float32).to(dtype).to(device)
    print(
        f"config  = ConvNeXt dims={dims} depths={depths}  input[{batch},3,{img},{img}] "
        f"dtype={dtype} (depthwise conv, channels-last LN/MLP)"
    )
    print(f"routing = {ops}")
    print()

    def fwd():
        with torch.no_grad():
            return model(x)

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
