#!/usr/bin/env python3
"""Hunt a genuine 3-way engine mix (flyDSL AND rocKE AND asm_sdpa each winning >=1 shape).

Grounded in the actual AOT/pack coverage on this box:
  * asm_sdpa gfx950 forward = NON-CAUSAL hd128/hd192 bf16 ONLY (fmha_fwd.csv has zero causal
    rows) -> asm can only enter for noncausal / decode.
  * flyDSL registered: causal h8/h16/h32kv8 + noncausal h8/h16, all d128 bf16.
  * rocKE (Gfx950AttentionDense) baked instances: noncausal nqh8/nkh8 sq256, and causal
    nqh32/nkh8 sq{512,1024,2048,4096}. rocKE bakes seqlen EXACTLY.

So the ONE shape where all three co-enumerate is noncausal MHA h8 d128 @ S=256 (rocKE's POC
instance). The causal 32/8 prefills are flyDSL-vs-rocKE (asm ineligible). Decode Sq=1 is asm-only.
This probe sweeps each engine's sweet spot and prints the full ranked (engine,time) list per shape
via the exhaustive autotune sweep, so we can see who actually wins each.

Maps to a real VLM/enc-dec workload: noncausal MHA block = a ViT/encoder layer; causal 32/8 =
the LLM prefill; Sq=1 noncausal = LLM decode.

    HIPDNN_TORCH_TUNE=tune bash run_py.sh three_way_probe.py
"""
import math
import os
import sys

sys.path.insert(0, "/home/AMD/brpepers/wt/hipdnn-flydsl-poc/projects/hipdnn/tools/hipdnn_torch")
import hipdnn_torch  # noqa: E402

HEAD_DIM = 128

# (label, num_q_heads, num_kv_heads, Sq, Skv, causal)
CASES = [
    ("noncausal MHA h8   S=256   (asm|flyDSL|rocKE)", 8, 8, 256, 256, False),
    ("noncausal MHA h16  S=256   (asm|flyDSL)",       16, 16, 256, 256, False),
    ("causal    GQA 32/8 S=512   (flyDSL|rocKE)",     32, 8, 512, 512, True),
    ("causal    GQA 32/8 S=1024  (flyDSL|rocKE)",     32, 8, 1024, 1024, True),
    ("causal    GQA 32/8 S=4096  (flyDSL|rocKE)",     32, 8, 4096, 4096, True),
    ("decode    GQA 32/8 Skv=2048 (asm)",             32, 8, 1, 2048, False),
]


def main() -> int:
    if not hipdnn_torch.provider_ready():
        print("provider/torch NOT ready", file=sys.stderr)
        return 1
    import torch
    import torch.nn.functional as F

    os.environ.setdefault("HIPDNN_TORCH_TUNE", "tune")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    print(f"device = {torch.cuda.get_device_name(0)}   TUNE={os.environ.get('HIPDNN_TORCH_TUNE')}")
    print()

    def _bshd(b, h, s, d):
        return (torch.randn(b, s, h, d, dtype=torch.bfloat16, device=dev) * 0.1).transpose(1, 2)

    hipdnn_torch.reset()
    hipdnn_torch.enable_logging()
    hipdnn_torch.install(["sdpa"])
    try:
        for label, nqh, nkh, sq, skv, causal in CASES:
            q = _bshd(1, nqh, sq, HEAD_DIM)
            k = _bshd(1, nkh, skv, HEAD_DIM)
            v = _bshd(1, nkh, skv, HEAD_DIM)
            gqa = nqh != nkh
            with torch.no_grad():
                F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal, enable_gqa=gqa)
            eng = hipdnn_torch.overrides()["sdpa"]._last_engine
            print(f"swept: {label:44s} -> last_engine={eng}")
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
