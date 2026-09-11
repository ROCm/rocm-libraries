#!/usr/bin/env python3
"""Llama-3-8B decoder block A/B across a LONG-CONTEXT seq sweep: native SDPA vs the
hipdnn_torch injection (SDPA -> hipDNN -> best-of-breed engine).

Why Llama-3-8B for the long-context push:
  * attention geometry is IDENTICAL to Mistral-7B -- hidden=4096, 32 q heads / 8 kv heads,
    head_dim=128, SwiGLU mlp=14336 -- so it reuses the SAME flyDSL (32,8,128,causal,bf16)
    HSACO we already AOT'd. Zero new attention compiles.
  * UNLIKE Mistral (SWA=4096), Llama-3 is PLAIN FULL causal (no sliding window,
    rope_theta=500000). So the plain-causal flyDSL kernel stays numerically CORRECT at
    8K/16K/32K -- which is exactly the regime where attention (O(S^2)) overtakes the
    MLP (O(S)) and the whole-model uplift climbs toward the attention-isolated number.

Per seq we report TWO A/B ratios:
  block_speedup  = native_block_ms / injected_block_ms   (whole decoder block)
  attn_speedup   = native_sdpa_ms  / injected_sdpa_ms    (the isolated SDPA call)
and the attention share = attn_ms / block_ms (native), so you can watch it grow with S.

    HIPDNN_TORCH_SELECT=default bash run_py.sh llama3_block_ab.py [--seqs 512,2048,8192,16384,32768]

Knobs (env): LLAMA_LAYERS LLAMA_B LLAMA_WARMUP LLAMA_ITERS.
"""
import argparse
import math
import os
import statistics
import sys

sys.path.insert(
    0, "/home/AMD/brpepers/wt/hipdnn-flydsl-poc/projects/hipdnn/tools/hipdnn_torch"
)
import hipdnn_torch  # noqa: E402

# Llama-3-8B exact attention/MLP geometry (== Mistral-7B geometry, different rope_theta + no SWA).
DIM = 4096
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
MLP = 14336
ROPE_THETA = 500000.0


def build_block(torch, seq, layers, device, dtype):
    nn = torch.nn
    F = torch.nn.functional

    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(device=device, dtype=dtype)
    sin = emb.sin().to(device=device, dtype=dtype)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(x):
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
            q = self.q(h).view(b, s, HEADS, HEAD_DIM).transpose(1, 2)
            k = self.k(h).view(b, s, KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = self.v(h).view(b, s, KV_HEADS, HEAD_DIM).transpose(1, 2)
            q, k = apply_rope(q), apply_rope(k)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            a = a.transpose(1, 2).reshape(b, s, HEADS * HEAD_DIM)
            x = x + self.o(a)
            h = self.n2(x)
            x = x + self.down(F.silu(self.gate(h)) * self.up(h))
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.norm = RMSNorm()

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    return Model().to(device=device, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ops", default="sdpa")
    ap.add_argument("--seqs", default="512,2048,8192,16384,32768",
                    help="comma-separated prefill lengths to sweep")
    ap.add_argument("--layers", type=int, default=int(os.environ.get("LLAMA_LAYERS", "1")))
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]

    if not hipdnn_torch.provider_ready():
        print("provider/torch not ready -- set HIPDNN_TORCH_PROVIDER_SO.", file=sys.stderr)
        return 1

    import torch
    import torch.nn.functional as F

    dtype = torch.bfloat16
    device = torch.device("cuda")
    batch = int(os.environ.get("LLAMA_B", "1"))
    warmup = int(os.environ.get("LLAMA_WARMUP", "10"))
    iters = int(os.environ.get("LLAMA_ITERS", "50"))
    scale = 1.0 / math.sqrt(HEAD_DIM)

    print(f"device  = {torch.cuda.get_device_name(0)}")
    print(f"select  = {os.environ.get('HIPDNN_TORCH_SELECT', 'default')}  "
          f"tune={os.environ.get('HIPDNN_TORCH_TUNE', '<unset>')}")
    print(f"config  = Llama-3-8B dim={DIM} heads={HEADS}/{KV_HEADS} head_dim={HEAD_DIM} "
          f"mlp={MLP} layers={args.layers} (PLAIN causal, rope_theta={ROPE_THETA:g})")
    print(f"routing = {ops}  scale={scale:.6f}")
    print()

    native_sdpa = F.scaled_dot_product_attention

    def cuda_time(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(iters):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            s.append(a.elapsed_time(b))
        return statistics.median(s)

    def bshd(b, h, sq, d):
        return (torch.randn(b, sq, h, d, dtype=dtype, device=device) * 0.1).transpose(1, 2)

    header = (f"{'seq':>7} {'blk_nat':>9} {'blk_inj':>9} {'blk_up':>7} "
              f"{'attn_nat':>9} {'attn_inj':>9} {'attn_up':>8} {'attn_share':>10} {'winner':>16}")
    rows = []
    NAMES = {"0x45d33a15c6a70e0b": "FlydslAttention", "5031429073904537099": "FlydslAttention"}

    for seq in seqs:
        torch.manual_seed(3)
        model = build_block(torch, seq, args.layers, device, dtype)
        x = (torch.randn(batch, seq, DIM, dtype=torch.float32, device=device) * 1.0).to(dtype)

        def blk():
            with torch.no_grad():
                return model(x)

        # correctness (native vs injected) once per seq
        with torch.no_grad():
            hipdnn_torch.uninstall()
            y_nat = blk().float()
            hipdnn_torch.reset(); hipdnn_torch.install(ops)
            y_inj = blk().float()
            hipdnn_torch.uninstall()
        err = float((y_nat - y_inj).abs().max().item())
        den = float(y_nat.abs().max().item()) or 1.0
        ok = "OK " if err / den < 8e-2 else "BAD"

        # isolated SDPA tensors (BSHD), same geometry the block feeds SDPA
        q = bshd(batch, HEADS, seq, HEAD_DIM)
        k = bshd(batch, KV_HEADS, seq, HEAD_DIM)
        v = bshd(batch, KV_HEADS, seq, HEAD_DIM)

        def sdpa():
            with torch.no_grad():
                return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True, enable_gqa=True)

        # native (injection off)
        hipdnn_torch.uninstall()
        blk_nat = cuda_time(blk)
        attn_nat = cuda_time(sdpa)
        # injected (best-of-breed)
        hipdnn_torch.reset(); hipdnn_torch.install(ops)
        blk_inj = cuda_time(blk)
        attn_inj = cuda_time(sdpa)
        eng = hipdnn_torch.overrides()["sdpa"]._last_engine
        hipdnn_torch.uninstall()

        w = NAMES.get(str(eng), str(eng))
        rows.append((seq, blk_nat, blk_inj, attn_nat, attn_inj, w, ok, err / den))
        print(f"[seq={seq:>6}] corr={ok} rel={err/den:.4f}  "
              f"blk {blk_nat:.3f}->{blk_inj:.3f}ms ({blk_nat/blk_inj:.3f}x)  "
              f"attn {attn_nat:.4f}->{attn_inj:.4f}ms ({attn_nat/attn_inj:.3f}x)  win={w}",
              flush=True)

    print()
    print(header)
    for seq, bn, bi, an, ai, w, ok, rel in rows:
        share = an / bn if bn else float("nan")
        print(f"{seq:>7} {bn:9.3f} {bi:9.3f} {bn/bi:6.3f}x {an:9.4f} {ai:9.4f} "
              f"{an/ai:7.3f}x {share*100:9.1f}% {w:>16}")
    print()
    print("attn_share = native SDPA ms / native block ms (grows with S; that's the ceiling the")
    print("whole-block speedup approaches as attention's O(S^2) overtakes the O(S) MLP).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
