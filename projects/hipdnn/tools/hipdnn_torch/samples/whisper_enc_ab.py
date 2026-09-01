#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
whisper_enc_ab.py -- run a Whisper-style audio **encoder** and A/B the forward native vs the
hipdnn_torch injection layer. This is the **audio-domain transformer** entry: a Conv1d mel
frontend feeding a stack of bidirectional (non-causal) transformer encoder blocks
(pre-norm LayerNorm, GELU MLP, self-attention). Geometry defaults to Whisper-base
(dim=512, 8 heads, head_dim=64, mlp=2048).

Two things make it a distinct coverage probe:
  * the frontend is ``Conv1d`` (two convs, the second stride-2), and **conv1d is NOT a routed
    family** (the injection only intercepts ``F.conv2d``/``F.conv3d``) -- so the frontend
    always stays native and shows up as an "other" bucket in the census, documenting a conv
    rank the catalog doesn't cover;
  * everything after the frontend is the same non-causal transformer op mix as ``bert_block``
    but at a different geometry (dim=512, head_dim=64), a second data point on the SDPA
    head-config gap.

Self-contained (plain ``torch.nn``; no ``transformers``); random weights are fine for
routing/coverage. Set ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0``
(SDPA + gfx1151 aotriton crash, see ``llama_block_ab.py``).

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SOS=<all .dll> \
        python samples/whisper_enc_ab.py --ops linear,layernorm,sdpa,gelu

Knobs: WHIS_DIM WHIS_HEADS WHIS_MLP WHIS_MEL WHIS_LAYERS WHIS_B WHIS_WARMUP WHIS_ITERS.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipdnn_torch  # noqa: E402
from _census import device_time_census  # noqa: E402


def build_whisper_enc(torch, n_mels, dim, heads, mlp, layers, device, dtype):
    """A canonical Whisper encoder (Conv1d frontend + non-causal transformer blocks)."""
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
            q, k, v = qkv[0], qkv[1], qkv[2]  # [b, heads, s, head_dim]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            a = a.transpose(1, 2).reshape(b, s, dim)
            x = x + self.proj(a)
            h = self.ln2(x)
            x = x + self.out(F.gelu(self.fc(h)))
            return x

    class WhisperEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(n_mels, dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.ln_post = nn.LayerNorm(dim)

        def forward(self, mel):
            x = F.gelu(self.conv1(mel))
            x = F.gelu(self.conv2(x))  # [b, dim, t]
            x = x.permute(0, 2, 1)  # [b, t, dim]
            for blk in self.blocks:
                x = blk(x)
            return self.ln_post(x)

    return WhisperEnc().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops", default="linear,layernorm,sdpa,gelu", help="overrides to route"
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
    n_mels = 80
    dim = int(os.environ.get("WHIS_DIM", "512"))
    heads = int(os.environ.get("WHIS_HEADS", "8"))
    mlp = int(os.environ.get("WHIS_MLP", "2048"))
    mel_t = int(
        os.environ.get("WHIS_MEL", "800")
    )  # frames; conv2 stride-2 -> mel_t/2 tokens
    layers = int(os.environ.get("WHIS_LAYERS", "6"))
    batch = int(os.environ.get("WHIS_B", "1"))
    warmup = int(os.environ.get("WHIS_WARMUP", "3"))
    iters = int(os.environ.get("WHIS_ITERS", "10"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(
        f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}"
        f"  engine={os.environ.get('HIPDNN_TORCH_ENGINE', '(none)')}"
    )
    model = build_whisper_enc(torch, n_mels, dim, heads, mlp, layers, device, dtype)
    gen = torch.Generator(device="cpu").manual_seed(320)
    mel = (
        torch.randn(batch, n_mels, mel_t, generator=gen, dtype=torch.float32)
        .to(dtype)
        .to(device)
    )
    print(
        f"config  = Whisper-enc dim={dim} heads={heads} head_dim={dim // heads} mlp={mlp} "
        f"layers={layers}  mel[{batch},{n_mels},{mel_t}] -> {mel_t // 2} tokens  dtype={dtype}"
    )
    print(f"routing = {ops}  (conv1d frontend is NOT a routed family -> stays native)")
    print()

    def fwd():
        with torch.no_grad():
            return model(mel)

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
                model(mel)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model(mel)
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
