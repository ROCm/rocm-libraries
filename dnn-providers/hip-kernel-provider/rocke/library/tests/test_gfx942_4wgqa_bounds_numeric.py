# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU guard for the gfx942 4-warp bodies on odd sequence lengths.

The lean natural-QK body (head_size 192 and 256 bf16 causal prefill) walks 64-key
tiles and 128-row Q blocks; the D128 body (sliding-window prefill, and any D128
shape the 4-warp tuning spec is pinned to, decode included) walks 32- or 64-key
tiles and 128-row Q blocks (32 tokens when head-folded). When a sequence length is
not a tile multiple, the last tile covers keys and rows past the sequence. A body
must not read block-table entries, K/V rows or Q rows for them: an
out-of-sequence block-table entry holds an arbitrary page id, and the K/V load built
from it faults. A one-token decode sequence reads 127 Q rows past its token without
the Q clamp.

Each block-table row here carries extra columns filled with a page id whose K/V
offset lies about 2 GiB past the cache, so any read past a sequence's last page
faults deterministically instead of depending on what the allocator left there.
Output is checked against an fp32 reference. The structural CPU twins are
``TestLeanBodyBoundsReads`` and ``TestD128BodyBoundsReads`` in
``test_d256_gfx942_fast_routing.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from kernels.common import attention_unified as au

_4WARP_SPEC = "gfx942_u2d_gfx942_4warp_nw1_mw32_t64_llvm"


def _gpu_ready():
    """Gate on ``gcnArchName`` (the ISA target), not the marketing name, which
    varies across gfx942 parts."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    # gcnArchName exists only on ROCm builds of torch.
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return "gfx942" in torch.cuda.get_device_properties(0).gcnArchName.lower()


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"
)

_PAD_COLS = 8


def _paged_inputs(seqs, hq, hkv, d, bs, dt, seed):
    """Random Q/K/V, a permuted block table whose padding columns hold the
    poison page id, sequence lengths, and a NaN-filled output."""
    import torch

    tdt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dt]
    pages = [-(-k // bs) for _, k in seqs]
    nblocks = sum(pages) + 3
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        sum(s for s, _ in seqs), hq, d, device="cuda", dtype=tdt, generator=g
    )
    kc = torch.randn(nblocks, bs, hkv, d, device="cuda", dtype=tdt, generator=g)
    vc = torch.randn(nblocks, bs, hkv, d, device="cuda", dtype=tdt, generator=g)
    perm = torch.randperm(nblocks, device="cuda", generator=g).to(torch.int32)
    poison = (1 << 30) // (bs * hkv * d)
    bt = torch.full(
        (len(seqs), max(pages) + _PAD_COLS), poison, dtype=torch.int32, device="cuda"
    )
    p0 = 0
    for i, n in enumerate(pages):
        bt[i, :n] = perm[p0 : p0 + n]
        p0 += n
    cu = torch.tensor(
        [0] + [s for s, _ in seqs], dtype=torch.int32, device="cuda"
    ).cumsum(0, dtype=torch.int32)
    kl = torch.tensor([k for _, k in seqs], dtype=torch.int32, device="cuda")
    return SimpleNamespace(
        q=q,
        k=kc,
        v=vc,
        bt=bt,
        cu=cu,
        kl=kl,
        pages=pages,
        nblocks=nblocks,
        out=torch.full_like(q, float("nan")),
    )


def _launch_production(x, seqs, hq, hkv, d, bs, dt, window):
    """Launch through the production route; return the kernel name it builds."""
    import torch
    from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

    problem = au.UnifiedAttentionProblem(
        total_q=sum(s for s, _ in seqs),
        num_seqs=len(seqs),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(s for s, _ in seqs),
        max_seqlen_k=max(k for _, k in seqs),
        dtype=dt,
        sliding_window=window,
        num_cus=torch.cuda.get_device_properties(0).multi_processor_count,
        num_kv_blocks=x.nblocks,
    )
    assert (
        au._gfx942_4warp_route(problem, "gfx942") is not None
    ), "case must exercise a 4-warp body"
    spec = au._tiled_spec_from_problem(problem, "gfx942")
    au.run_unified_attention_torch(
        problem=problem,
        q=x.q,
        k=x.k,
        v=x.v,
        out=x.out,
        cu_seqlens_q=x.cu,
        seqused_k=x.kl,
        softmax_scale=1.0 / math.sqrt(d),
        block_table=x.bt,
        softcap=0.0,
        backend="auto",
    )
    return build_gfx942_4warp_gqa(spec, arch="gfx942").name


def _launch_pinned(x, seqs, hq, hkv, d, bs, dt, window):
    """Launch the 4-warp tuning spec through the dispatcher (the only way to reach
    a 4-warp body on decode); return the kernel name it builds."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    assert len({s for s, _ in seqs}) == 1, "pinned cases use one query length"
    result = dispatch_attention(
        AttentionRequest(
            batch=len(seqs),
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=seqs[0][0],
            seqlen_k=max(k for _, k in seqs),
            hdim_q=d,
            hdim_v=d,
            arch="gfx942",
            mask_type=2,
            sliding_window=window,
            dtype=dt,
            kv_block_size=bs,
            algorithm="unified_tuning",
            spec_id=_4WARP_SPEC,
        )
    )
    tensors = {
        "q": x.q,
        "out": x.out,
        "k": x.k,
        "v": x.v,
        "cu_seqlens_q": x.cu,
        "seqused_k": x.kl,
        "block_table": x.bt,
    }
    result.bind_torch(tensors).launch()
    return result.build().name


def _max_abs_vs_reference(x, seqs, hq, hkv, d, window):
    """max |out - ref| over all sequences against an fp32 SDPA reference with the
    bottom-right causal (and sliding-window) mask; asserts the output is finite."""
    import torch
    import torch.nn.functional as F

    rep = hq // hkv
    st = 0
    worst = 0.0
    for i, (sq, skv) in enumerate(seqs):
        idx = x.bt[i, : x.pages[i]].long()
        k = x.k[idx].reshape(-1, hkv, d)[:skv].float().permute(1, 0, 2)
        v = x.v[idx].reshape(-1, hkv, d)[:skv].float().permute(1, 0, 2)
        k = k.repeat_interleave(rep, 0)
        v = v.repeat_interleave(rep, 0)
        qs = x.q[st : st + sq].float().permute(1, 0, 2)
        ii = torch.arange(sq, device="cuda").view(sq, 1) + (skv - sq)
        jj = torch.arange(skv, device="cuda").view(1, skv)
        keep = jj <= ii
        if window:
            keep &= ii - jj < window
        mask = torch.where(keep, 0.0, float("-inf"))
        ref = F.scaled_dot_product_attention(
            qs, k, v, attn_mask=mask, scale=1.0 / math.sqrt(d)
        ).permute(1, 0, 2)
        o = x.out[st : st + sq].float()
        assert torch.isfinite(o).all(), f"seq {i}: non-finite output"
        worst = max(worst, (o - ref).abs().max().item())
        st += sq
    return worst


def _run(seqs, hq, hkv, d, bs, dt="bf16", window=0, pinned=False, seed=0):
    """Launch one case; return (max_abs vs the fp32 reference, kernel name)."""
    import torch

    x = _paged_inputs(seqs, hq, hkv, d, bs, dt, seed)
    launch = _launch_pinned if pinned else _launch_production
    name = launch(x, seqs, hq, hkv, d, bs, dt, window)
    torch.cuda.synchronize()
    return _max_abs_vs_reference(x, seqs, hq, hkv, d, window), name


@requires_gfx942_gpu
@pytest.mark.parametrize("d", [192, 256])
@pytest.mark.parametrize(
    "seqs,hq,hkv,bs",
    [
        ([(129, 129)], 16, 2, 16),
        ([(129, 129)], 16, 2, 32),
        ([(2, 2)], 12, 2, 16),
        ([(339, 367)], 40, 8, 16),
        ([(513, 513), (77, 2001), (1023, 1023)], 16, 16, 16),
    ],
)
def test_lean_body_odd_lengths(d, seqs, hq, hkv, bs):
    assert _run(seqs, hq, hkv, d, bs)[0] <= 5e-2


@requires_gfx942_gpu
@pytest.mark.parametrize(
    "seqs",
    [[(129, 129)], [(339, 367)], [(513, 513), (77, 2001), (1023, 1023)]],
)
@pytest.mark.parametrize(
    "dt,hq,hkv,bs,suffix",
    [
        ("bf16", 32, 8, 16, "_4wgqa_fold"),
        ("fp16", 32, 8, 16, "_4wgqa"),
        ("bf16", 64, 8, 32, "_4wgqa"),
        ("bf16", 32, 8, 64, "_4wgqa"),
    ],
)
def test_d128_swa_odd_lengths(seqs, dt, hq, hkv, bs, suffix):
    worst, name = _run(seqs, hq, hkv, 128, bs, dt=dt, window=64)
    assert name.endswith(suffix), name
    assert worst <= 5e-2


@requires_gfx942_gpu
@pytest.mark.parametrize(
    "hq,hkv,window,suffix", [(64, 4, 0, "_4wgqa"), (32, 8, 128, "_4wgqa_fold")]
)
def test_d128_decode_pinned(hq, hkv, window, suffix):
    """One query token per sequence, KV lengths in (4064, 4096) that are not tile
    multiples: the gfx942 decode tuning shape that faulted (B=64, Hq=64, Hkv=4)."""
    seqs = [(1, 4095 - (7 * i) % 31) for i in range(64)]
    worst, name = _run(seqs, hq, hkv, 128, 16, window=window, pinned=True)
    assert name.endswith(suffix), name
    assert worst <= 5e-2
