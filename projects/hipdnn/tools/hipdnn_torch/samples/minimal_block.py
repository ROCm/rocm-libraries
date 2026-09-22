#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
minimal_block.py -- the 5-minute "try it" sample for hipdnn_torch.

Self-contained: NO external model checkout. It builds a tiny transformer block out
of the exact functionals hipdnn_torch intercepts -- ``F.rms_norm`` + ``F.layer_norm``
+ ``nn.Linear`` (``F.linear``) + ``F.scaled_dot_product_attention`` + a SwiGLU-ish
MLP mixing ``F.silu`` (routes) and ``F.gelu`` (default erf variant, which has no
catalog builder yet, so it deliberately shows up as a native fallback in the report)
-- with shapes the shipped gfx1151 kernels can serve (heads=32, head_dim=64,
hidden=2048, seqlen and feature dims all multiples of 16). It then:

  1. runs one forward with plain PyTorch (baseline),
  2. installs hipdnn_torch and runs the same forward,
  3. asserts the two outputs match within a bf16 tolerance, and
  4. prints the intercept report -- per-shape AOT vs native counts and, for every
     native fallback, the reason (the "what hipDNN still needs" list).

Prerequisites (see ../README.md "Environment setup"): a provider-compatible torch,
the WSL librocdxg shim on LD_LIBRARY_PATH (WSL only), and
``HIPDNN_TORCH_PROVIDER_SO`` pointing at the built plugin. Then::

    HIPDNN_TORCH_PROVIDER_SO=<build>/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
        python samples/minimal_block.py
"""

import os
import sys

# Make `import hipdnn_torch` work when run directly from the samples/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402


def build_block(torch):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        """Pre-norm attention + MLP block; hidden = heads * head_dim = 2048."""

        def __init__(self, heads=32, head_dim=64, mlp=8192, dtype=torch.bfloat16):
            super().__init__()
            hidden = heads * head_dim
            self.heads, self.head_dim, self.hidden = heads, head_dim, hidden
            self.eps = 1.0e-6
            self.n1 = nn.Parameter(torch.ones(hidden, dtype=dtype))
            # Second norm is a LayerNorm (scale + bias) to exercise F.layer_norm.
            self.ln_w = nn.Parameter(torch.ones(hidden, dtype=dtype))
            self.ln_b = nn.Parameter(torch.zeros(hidden, dtype=dtype))
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=False, dtype=dtype)
            self.proj = nn.Linear(hidden, hidden, bias=True, dtype=dtype)
            self.fc1 = nn.Linear(hidden, mlp, bias=True, dtype=dtype)
            self.gate = nn.Linear(hidden, mlp, bias=True, dtype=dtype)
            self.fc2 = nn.Linear(mlp, hidden, bias=True, dtype=dtype)

        def forward(self, x):
            b, s, _ = x.shape
            h = F.rms_norm(x, (self.hidden,), self.n1, eps=self.eps)  # rms_norm site
            qkv = self.qkv(h).view(b, s, 3, self.heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B,H,S,D]
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(1, 2).reshape(b, s, self.hidden)
            x = x + self.proj(a)
            # LayerNorm site, then a gated MLP: F.silu routes, F.gelu (erf) falls back.
            h = F.layer_norm(x, (self.hidden,), self.ln_w, self.ln_b, eps=1.0e-5)
            x = x + self.fc2(F.gelu(self.fc1(h)) * F.silu(self.gate(h)))
            return x

    return Block


def main() -> int:
    if not hipdnn_torch.provider_ready():
        print(
            "provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO and run "
            "under a provider-compatible torch (see ../README.md).",
            file=sys.stderr,
        )
        return 1

    import torch

    hipdnn_torch.enable_logging()  # show each native fallback as it happens

    device = torch.device("cuda")
    dtype = torch.bfloat16
    seqlen = int(os.environ.get("MB_SEQ", "128"))  # multiple of 16

    torch.manual_seed(0)
    Block = build_block(torch)
    model = Block(dtype=dtype).to(device).eval()
    # Small init so the bf16 forward stays finite / well-conditioned.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.mul_(0.05)
        x = torch.randn(1, seqlen, model.hidden, dtype=dtype, device=device) * 0.1

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"block   = heads=32 head_dim=64 hidden={model.hidden} seq={seqlen} dtype={dtype}"
    )
    print()

    # 1) baseline (nothing patched)
    with torch.no_grad():
        y_native = model(x).float()

    # 2) route through hipDNN
    hipdnn_torch.install()
    hipdnn_torch.reset()
    with torch.no_grad():
        y_hipdnn = model(x).float()
    hipdnn_torch.uninstall()

    # 3) parity
    max_err = float((y_native - y_hipdnn).abs().max().item())
    denom = float(y_native.abs().max().item()) or 1.0
    rel = max_err / denom
    ok = rel < 5.0e-2
    print(
        f"parity native-vs-hipdnn: {'OK ' if ok else 'BAD'} "
        f"max_abs_err={max_err:.5f} rel={rel:.4f} (out|max|={denom:.4f})"
    )
    print()

    # 4) what routed to hipDNN, and what fell back (and why)
    print(hipdnn_torch.report())
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
