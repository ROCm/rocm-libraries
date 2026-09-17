#!/usr/bin/env python3
"""M5 Phase 2 (block-first): A/B a real Mistral-7B decoder block, native SDPA vs the
hipdnn_torch injection that routes SDPA -> hipDNN -> flyDSL.

Geometry is Mistral-7B-v0.1 EXACT: hidden=4096, 32 query heads / 8 kv heads,
head_dim=128, SwiGLU mlp=14336. The attention call mirrors what real HF transformers
(5.x) emits for Mistral PREFILL on ROCm: F.scaled_dot_product_attention with k/v at 8
heads + enable_gqa=True + is_causal=True + default scale (1/sqrt(128)) and NO mask
(verified from transformers/integrations/sdpa_attention.py::use_gqa_in_sdpa, which is
True here: attention_mask is None, head_dim=128<=256). q/k/v are BSHD views (project ->
[B,S,H,D] contiguous -> transpose(1,2)), which is what flyDSL's graph_match requires.

Prefill only, seq <= 4096 (Mistral SWA=4096; plain causal == sliding-window when the
window covers the whole sequence, so the plain-causal flyDSL kernel is correct there).

Random weights on purpose: the checkpoint changes values, not the call sites/shapes/op
mix that route to hipDNN. bf16 parity across one block is meaningful (RMSNorm w=1).

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SO=<...> \
        python mistral_block_ab.py [--ops sdpa] [--seq 512] [--layers 1]

Knobs (env): MISTRAL_SEQ MISTRAL_LAYERS MISTRAL_B MISTRAL_WARMUP MISTRAL_ITERS.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(
    0, "/home/AMD/brpepers/wt/hipdnn-flydsl-poc/projects/hipdnn/tools/hipdnn_torch"
)

import hipdnn_torch  # noqa: E402

# Mistral-7B-v0.1 exact attention/MLP geometry.
DIM = 4096
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
MLP = 14336
ROPE_THETA = 10000.0


def build_mistral(torch, seq, layers, device, dtype):
    nn = torch.nn
    F = torch.nn.functional
    n_rep = HEADS // KV_HEADS

    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
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
            self.w = nn.Parameter(torch.ones(DIM))

        def forward(self, x):
            return F.rms_norm(x, (DIM,), self.w, 1e-5)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = RMSNorm()
            self.q = nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
            self.k = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
            self.v = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
            self.o = nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
            self.n2 = RMSNorm()
            self.gate = nn.Linear(DIM, MLP, bias=False)
            self.up = nn.Linear(DIM, MLP, bias=False)
            self.down = nn.Linear(MLP, DIM, bias=False)

        def forward(self, x):
            b, s, _ = x.shape
            h = self.n1(x)
            # BSHD views: [b,s,H,d] contiguous then transpose -> [b,H,s,d] with BSHD memory.
            q = self.q(h).view(b, s, HEADS, HEAD_DIM).transpose(1, 2)
            k = self.k(h).view(b, s, KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = self.v(h).view(b, s, KV_HEADS, HEAD_DIM).transpose(1, 2)
            q, k = apply_rope(q), apply_rope(k)
            # Mistral prefill on ROCm: GQA stays 8 kv-heads, enable_gqa=True, causal, no mask,
            # default scale (1/sqrt(head_dim)). This is the flyDSL (32,8,128,causal,bf16) shape.
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            a = a.transpose(1, 2).reshape(b, s, HEADS * HEAD_DIM)
            x = x + self.o(a)
            h = self.n2(x)
            x = x + self.down(F.silu(self.gate(h)) * self.up(h))
            return x

    class Mistral(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.norm = RMSNorm()

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    return Mistral().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ops", default="sdpa", help="comma-separated overrides to route")
    ap.add_argument("--seq", type=int, default=int(os.environ.get("MISTRAL_SEQ", "512")))
    ap.add_argument("--layers", type=int, default=int(os.environ.get("MISTRAL_LAYERS", "1")))
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if not hipdnn_torch.provider_ready():
        print("provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr)
        return 1

    import torch

    if args.seq > 4096:
        print(f"WARNING seq={args.seq} > 4096 (Mistral SWA); plain-causal flyDSL kernel "
              "is only correct for prefill <= 4096.", file=sys.stderr)

    dtype = torch.bfloat16
    device = torch.device("cuda")
    batch = int(os.environ.get("MISTRAL_B", "1"))
    warmup = int(os.environ.get("MISTRAL_WARMUP", "3"))
    iters = int(os.environ.get("MISTRAL_ITERS", "10"))

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}")
    model = build_mistral(torch, args.seq, args.layers, device, dtype)
    gen = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(batch, args.seq, DIM, generator=gen, dtype=torch.float32).to(dtype).to(device)
    print(f"config  = Mistral-7B dim={DIM} heads={HEADS} kv_heads={KV_HEADS} "
          f"head_dim={HEAD_DIM} mlp={MLP} layers={args.layers}  "
          f"input[{batch},{args.seq},{DIM}] dtype=bf16")
    print(f"routing = {ops}  (scale={1.0/math.sqrt(HEAD_DIM):.6f})")
    print()

    def fwd():
        with torch.no_grad():
            return model(x)

    # correctness: native vs injected, identical input/weights
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
        ok = max_err / denom < 8e-2
        print(f"correctness native-vs-injected: {'OK ' if ok else 'BAD'} "
              f"max_abs_err={max_err:.5f} rel={max_err/denom:.4f}")
    else:
        print(f"correctness: not measurable (native finite={fin_n}, injected finite={fin_o})")
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
    print(f"  injected = {over_ms:8.3f} ms/fwd   "
          f"speedup={native_ms/over_ms if over_ms else float('nan'):5.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
