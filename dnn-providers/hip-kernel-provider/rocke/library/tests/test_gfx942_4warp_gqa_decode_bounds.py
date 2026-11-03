# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU bounds guard for the gfx942 4-warp GQA D128 body on decode.

The D128 body of ``build_gfx942_4warp_gqa`` reads a whole Q block (128 rows, or
32 tokens when folded) and whole KV tiles. Without its ``q_last`` / ``k_last``
clamps, a one-token decode sequence reads up to 127 Q rows past its token -- off
the end of the Q tensor for the last sequences, which faulted the MI325X decode
sweep at this file's shape -- and a KV length that is not a tile multiple reads
block-table entries past the sequence's pages.

The block-table padding here points at a NaN-filled page. An unclamped tile
gathers it, and its NaN V rows reach the output through P = 0 (0 * NaN = NaN), so
the K/V half fails deterministically. The Q over-read can only fault.
"""

from __future__ import annotations

import math

import pytest

_SPEC = "gfx942_u2d_gfx942_4warp_nw1_mw32_t64_llvm"


def _gpu_ready():
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    return "mi300" in name or "mi325" in name or "gfx942" in name


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"
)


@requires_gfx942_gpu
@pytest.mark.parametrize(
    "hq,hk,window", [(64, 4, 0), (32, 8, 128)], ids=["plain", "fold"]
)
def test_decode_reads_stay_in_bounds(hq, hk, window):
    import torch
    from dispatch.attention import AttentionRequest, dispatch_attention

    B, D, BS = 64, 128, 16
    # Ragged KV lengths in (4064, 4096): none is a multiple of the 32-key tile.
    kv_lens = [4095 - (7 * i) % 31 for i in range(B)]
    max_k = max(kv_lens)
    # One spare column, so every over-read of a block-table row lands on padding.
    pages = -(-max_k // BS) + 1
    poison = B * pages
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, hq, D, generator=g, dtype=torch.bfloat16)
    k = torch.randn(poison + 1, BS, hk, D, generator=g, dtype=torch.bfloat16)
    v = torch.randn(poison + 1, BS, hk, D, generator=g, dtype=torch.bfloat16)
    k[poison] = float("nan")
    v[poison] = float("nan")
    table = torch.full((B, pages), poison, dtype=torch.int32)
    for s, n in enumerate(kv_lens):
        used = -(-n // BS)
        table[s, :used] = torch.arange(s * pages, s * pages + used)

    dev = "cuda"
    tensors = {
        "q": q.to(dev),
        "out": torch.empty(B, hq, D, dtype=torch.bfloat16, device=dev),
        "k": k.to(dev),
        "v": v.to(dev),
        "cu_seqlens_q": torch.arange(B + 1, dtype=torch.int32, device=dev),
        "seqused_k": torch.tensor(kv_lens, dtype=torch.int32, device=dev),
        "block_table": table.to(dev),
    }
    request = AttentionRequest(
        batch=B, nhead_q=hq, nhead_k=hk, seqlen_q=1, seqlen_k=max_k,
        hdim_q=D, hdim_v=D, arch="gfx942", mask_type=2, sliding_window=window,
        dtype="bf16", kv_block_size=BS, algorithm="unified_tuning", spec_id=_SPEC,
    )  # fmt: skip
    dispatch_attention(request).bind_torch(tensors).launch()
    torch.cuda.synchronize()
    out = tensors["out"].float().cpu()
    assert torch.isfinite(out).all(), "padding page reached the output"

    # fp32 reference: the single query sees keys [n - window, n) (all if window=0).
    kf = k.float().view(-1, hk, D)
    vf = v.float().view(-1, hk, D)
    worst = 0.0
    for s, n in enumerate(kv_lens):
        keys = torch.arange(max(0, n - window) if window else 0, n)
        phys = table[s, keys // BS].long() * BS + keys % BS
        ks = kf[phys].repeat_interleave(hq // hk, dim=1)
        vs = vf[phys].repeat_interleave(hq // hk, dim=1)
        p = torch.softmax(
            torch.einsum("hd,nhd->hn", q[s].float(), ks) / math.sqrt(D), dim=-1
        )
        ref = torch.einsum("hn,nhd->hd", p, vs)
        worst = max(worst, (out[s] - ref).abs().max().item())
    # 4e-2 is the repo-wide bf16 attention gate (see the fold numeric test).
    assert worst <= 4e-2, f"max_abs {worst:.4f} > 4e-2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
