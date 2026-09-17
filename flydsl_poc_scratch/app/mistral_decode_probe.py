#!/usr/bin/env python3
"""M5 decode coverage probe (census-only, NO decode kernel build).

Prefill is proven (flyDSL routed for every 32/8 causal call). This probe answers the
OTHER half of the user's M5 question -- "can hipDNN/flyDSL serve DECODE?" -- by feeding
the injection the exact SDPA shape a Mistral decode step emits and logging what routes
to an engine vs what falls back to native torch. We deliberately do NOT AOT a decode
kernel; the point is an honest census of the coverage gap.

Real HF Mistral decode SDPA (transformers 5.x, ROCm):
  q = [B, 32, 1, 128]   (ONE new query token)
  k = v = [B, 8, Skv, 128]   (8 kv-heads, Skv = cached context length)
  enable_gqa=True, attention_mask=None, is_causal=False   (q_len==1 => causal is a no-op,
    so HF passes is_causal=False and NO mask). Default scale = 1/sqrt(128).
Tensors are BSHD views (project -> [B,S,H,D] contiguous -> transpose(1,2)), same as
prefill. We sweep a few realistic cache depths so the census shows every (1, Skv) row.

    HIPDNN_TORCH_SELECT=default HIPDNN_TORCH_PROVIDER_SO=<...> python mistral_decode_probe.py
"""
import math
import os
import sys

sys.path.insert(
    0, "/home/AMD/brpepers/wt/hipdnn-flydsl-poc/projects/hipdnn/tools/hipdnn_torch"
)
import hipdnn_torch  # noqa: E402

HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128


def main() -> int:
    if not hipdnn_torch.provider_ready():
        print("provider/torch NOT ready -- check HIPDNN_TORCH_PROVIDER_SO / env", file=sys.stderr)
        return 1
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    native_sdpa = F.scaled_dot_product_attention  # capture pre-install

    print(f"device = {torch.cuda.get_device_name(0)}")
    print(f"probe  = Mistral decode SDPA  q=[B,{HEADS},1,{HEAD_DIM}] kv=[B,{KV_HEADS},Skv,{HEAD_DIM}] "
          f"enable_gqa=True is_causal=False scale={scale:.6f}")
    print()

    # (label, B, Skv) -- one new query token attending to Skv cached kv positions.
    cache_depths = [128, 512, 1024, 2048, 4096]
    cases = [(f"decode @ Skv={s}", 1, s) for s in cache_depths]

    def _bshd(b, h, s, d):  # BSHD view, same construction real models use
        return (torch.randn(b, s, h, d, dtype=torch.bfloat16, device=dev) * 0.1).transpose(1, 2)

    refs, tensors = {}, {}
    for label, b, skv in cases:
        q = _bshd(b, HEADS, 1, HEAD_DIM)      # single query token
        k = _bshd(b, KV_HEADS, skv, HEAD_DIM)
        v = _bshd(b, KV_HEADS, skv, HEAD_DIM)
        with torch.no_grad():
            refs[label] = native_sdpa(q, k, v, scale=scale, is_causal=False, enable_gqa=True).float()
        tensors[label] = (q, k, v)

    hipdnn_torch.reset()
    hipdnn_torch.enable_logging()
    hipdnn_torch.install(["sdpa"])
    try:
        for label, b, skv in cases:
            q, k, v = tensors[label]
            with torch.no_grad():
                got = F.scaled_dot_product_attention(
                    q, k, v, scale=scale, is_causal=False, enable_gqa=True
                ).float()
            err = float((got - refs[label]).abs().max().item())
            print(f"[{label}]  maxerr vs native = {err:.3e}")
    finally:
        hipdnn_torch.uninstall(["sdpa"])

    print()
    print(hipdnn_torch.report(["sdpa"]))
    print()
    print("Interpretation: aot>0 => hipDNN/flyDSL served that decode shape; native>0 => fell back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
