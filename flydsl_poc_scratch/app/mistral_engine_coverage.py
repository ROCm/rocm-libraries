#!/usr/bin/env python3
"""Which engines can serve Mistral's SDPA shapes? (rocKE vs flyDSL vs asm_sdpa overlap)

The per-shape census only records the WINNER. To answer "does more than one engine
serve this Mistral shape" we force an exhaustive autotune sweep (HIPDNN_TORCH_TUNE=tune),
which measures EVERY engine that both enumerates AND executes for the shape, and print
tuning_report() -- the full ranked (engine, time) list per shape.

Two Mistral shapes, exact geometry (GQA 32/8, D128, bf16, BSHD views, default scale):
  PREFILL  Sq=Skv=S, is_causal=True   (flyDSL is our AOT'd kernel; asm has no causal)
  DECODE   Sq=1, Skv=S, is_causal=False (asm_sdpa serves; flyDSL has no noncausal 32/8)

Run with the standard injection env + HIPDNN_TORCH_TUNE=tune.
"""
import math
import os
import sys

sys.path.insert(
    0, "/home/AMD/brpepers/wt/hipdnn-flydsl-poc/projects/hipdnn/tools/hipdnn_torch"
)
import hipdnn_torch  # noqa: E402

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128


def main() -> int:
    if not hipdnn_torch.provider_ready():
        print("provider/torch NOT ready", file=sys.stderr)
        return 1
    import torch
    import torch.nn.functional as F

    os.environ.setdefault("HIPDNN_TORCH_TUNE", "tune")  # force exhaustive sweep
    dev = torch.device("cuda")
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    print(f"device = {torch.cuda.get_device_name(0)}   HIPDNN_TORCH_TUNE={os.environ.get('HIPDNN_TORCH_TUNE')}")
    print()

    def _bshd(b, h, s, d):
        return (torch.randn(b, s, h, d, dtype=torch.bfloat16, device=dev) * 0.1).transpose(1, 2)

    # (label, Sq, Skv, causal)
    cases = [
        ("PREFILL causal  S=1024", 1024, 1024, True),
        ("PREFILL causal  S=2048", 2048, 2048, True),
        ("DECODE  noncausal Skv=1024", 1, 1024, False),
        ("DECODE  noncausal Skv=2048", 1, 2048, False),
    ]

    hipdnn_torch.reset()
    hipdnn_torch.enable_logging()
    hipdnn_torch.install(["sdpa"])
    try:
        for label, sq, skv, causal in cases:
            q = _bshd(1, HEADS, sq, HEAD_DIM)
            k = _bshd(1, KV_HEADS, skv, HEAD_DIM)
            v = _bshd(1, KV_HEADS, skv, HEAD_DIM)
            with torch.no_grad():
                F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal, enable_gqa=True)
            print(f"swept: {label}")
    finally:
        hipdnn_torch.uninstall(["sdpa"])

    print()
    print("=== full ranked engine coverage per shape (autotune sweep) ===")
    print(hipdnn_torch.tuning_report(["sdpa"]))
    print()
    print(hipdnn_torch.report(["sdpa"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
