#!/usr/bin/env python3
"""M5 Phase 1 smoke: does Mistral-7B's SDPA shape route through hipDNN → flyDSL?

Mistral-7B-v0.1 attention: GQA num_query_heads=32, num_kv_heads=8, head_dim=128,
causal, bf16. Before Phase 1 this fell back to native torch (no flyDSL kernel for
32/8); after AOT'ing flash_attn_real_h32kv8_d128_causal_bf16_gfx950.hsaco + adding
one kernelDescriptor, it should route aot=1 engine=FlydslAttention.

Runs the census both for the Mistral shape AND a known-good non-causal H8 control.
Native references are computed BEFORE install() (the monkeypatch replaces F.sdpa
GLOBALLY — a sdpa_kernel(MATH) context does NOT bypass it).
"""
import math
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
)  # not used, but keep scratch importable if needed
sys.path.insert(
    0,
    "/home/AMD/brpepers/wt/hipdnn-flydsl-poc/projects/hipdnn/tools/hipdnn_torch",
)

import hipdnn_torch  # noqa: E402


def _sdpa_gqa(F, q, k, v, scale, causal):
    # torch F.sdpa needs enable_gqa=True when num_kv_heads < num_query_heads.
    return F.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=causal, enable_gqa=True
    )


def main() -> int:
    if not hipdnn_torch.provider_ready():
        print("provider/torch NOT ready — check HIPDNN_TORCH_PROVIDER_SO / env", file=sys.stderr)
        return 1

    import torch
    import torch.nn.functional as F

    native_sdpa = F.scaled_dot_product_attention  # capture original before install
    dev = torch.device("cuda")
    torch.manual_seed(0)

    props = torch.cuda.get_device_properties(0)
    print(f"device = {torch.cuda.get_device_name(0)} ({getattr(props,'gcnArchName','?')})")
    print(f"torch  = {torch.__version__}  hip={getattr(torch.version,'hip',None)}")
    print()

    # (label, B, Hq, Hkv, S, D, causal)
    cases = [
        ("mistral-prefill (GQA 32/8 causal)", 1, 32, 8, 256, 128, True),
        ("control non-causal H8", 1, 8, 8, 256, 128, False),
    ]

    def _bshd(b, h, s, d):
        # Build a tensor the way a real model does: project QKV -> reshape to
        # [B,S,H,D] contiguous (token-major / BSHD memory) -> transpose to the
        # [B,H,S,D] logical shape torch.sdpa expects. The result is a VIEW whose
        # strides are BSHD (head stride == D, seq stride == H*D) -- which is what
        # flyDSL's graph_match requires. A plain randn(B,H,S,D) is BHSD-contiguous
        # (head-major) and would be (correctly) declined by flyDSL.
        return (torch.randn(b, s, h, d, dtype=torch.bfloat16, device=dev) * 0.1).transpose(1, 2)

    refs = {}
    tensors = {}
    for label, b, hq, hkv, s, d, causal in cases:
        q = _bshd(b, hq, s, d)
        k = _bshd(b, hkv, s, d)
        v = _bshd(b, hkv, s, d)
        scale = 1.0 / math.sqrt(d)
        with torch.no_grad():
            refs[label] = _sdpa_gqa(F, q, k, v, scale, causal).float()  # native, pre-install
        tensors[label] = (q, k, v, scale, causal)

    hipdnn_torch.reset()
    hipdnn_torch.enable_logging()
    hipdnn_torch.install(["sdpa"])
    try:
        for label, b, hq, hkv, s, d, causal in cases:
            q, k, v, scale, causal = tensors[label]
            with torch.no_grad():
                got = _sdpa_gqa(F, q, k, v, scale, causal).float()
            err = float((got - refs[label]).abs().max().item())
            print(f"[{label}]  maxerr vs native = {err:.3e}")
    finally:
        hipdnn_torch.uninstall(["sdpa"])

    print()
    print(hipdnn_torch.report(["sdpa"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
