# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiled 2D attention: a row whose first KV tile is fully masked (sliding
window) must still get the right O.

The CTA's tile range starts at the window start of its first row, so later rows
can see a fully masked first tile. The running max must not be pinned at 0 by
that tile: with every visible score negative, exp2(s - 0) rounds P to 0 (fp16
at about -17, bf16 at about -103 natural) and O collapses although the row sees
keys. Each epilogue branch is pinned so all three clamp sites are covered.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from kernels.common.attention_unified import (  # noqa: E402
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)


def _arch():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return next((a for a in ("gfx942", "gfx950") if a in name), None)


requires_cdna3_or_4 = pytest.mark.skipif(
    _arch() is None, reason="needs a gfx942 or gfx950 GPU with ROCm torch"
)

# One dispatcher geometry per epilogue branch of build_unified_attention_2d_tiled.
_BRANCH_VARIANT = {
    "gfx942": {
        "transposed32": "transposed_x8_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32x8_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    "gfx950": {
        "transposed32": "transposed32_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
}

# (dtype, natural-log score of every visible key). fp16 P underflows to 0 at
# about -17 when m is pinned at 0; bf16 needs a far lower score.
_SCORES = [("fp16", -20.0), ("bf16", -200.0)]
_WINDOW = 64
_HQ, _HKV, _D, _BS = 32, 8, 128, 16


def _problem(dtype, q_len, kv_len, num_blocks):
    return UnifiedAttentionProblem(
        total_q=q_len,
        num_seqs=1,
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_D,
        block_size=_BS,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        dtype=dtype,
        sliding_window=_WINDOW,
        num_kv_blocks=num_blocks,
    )


def _branch_spec(problem, branch):
    from dispatch.attention.unified_rules import canonicalize_tuning_spec

    arch = _arch()
    if arch == "gfx942":
        from dispatch.attention.gfx942_unified import (
            GFX942_TUNING_VARIANTS as variants,
        )
    else:
        from dispatch.attention.gfx950_unified import (
            GFX950_TUNING_VARIANTS as variants,
        )
    base = branch.removesuffix("_s1")
    spec_id = f"{arch}_u2d_{_BRANCH_VARIANT[arch][base]}"
    variant = next(v for v in variants if v.spec_id == spec_id)
    knobs = {"use_transposed_scalar_state": True} if branch.endswith("_s1") else {}
    spec, why = canonicalize_tuning_spec(problem, variant, knobs)
    assert spec is not None, f"{spec_id} {knobs}: {why}"
    return spec


def _reference(q, k_cache, v_cache, kv_len, block_table, scale):
    """FP32 O for one sequence, bottom-right causal + sliding window."""
    q_len = q.shape[0]
    nblk = (kv_len + _BS - 1) // _BS
    blocks = block_table[0, :nblk].long()
    k = k_cache[blocks].reshape(-1, _HKV, _D)[:kv_len].float()
    v = v_cache[blocks].reshape(-1, _HKV, _D)[:kv_len].float()
    k = k.transpose(0, 1).repeat_interleave(_HQ // _HKV, dim=0)
    v = v.transpose(0, 1).repeat_interleave(_HQ // _HKV, dim=0)
    s = torch.matmul(q.float().transpose(0, 1), k.transpose(-1, -2)) * scale
    p = torch.arange(q_len, device=q.device).view(-1, 1)
    kk = torch.arange(kv_len, device=q.device).view(1, -1)
    diag = p + (kv_len - q_len)
    visible = (kk <= diag) & (diag - kk < _WINDOW)
    s = s.masked_fill(~visible, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v).transpose(0, 1)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,score", _SCORES)
@pytest.mark.parametrize(
    "branch", ["transposed32", "transposed32_s1", "mfma32", "striped16"]
)
def test_masked_leading_tile_keeps_rows_with_keys(branch, dtype, score):
    q_len = kv_len = 256
    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(0)
    max_blocks = (kv_len + _BS - 1) // _BS
    num_blocks = max_blocks + 3
    scale = 1.0 / math.sqrt(_D)
    # Every visible score equals `score`: q = a, k = b with a * b * D * scale = score.
    a = 1.0 if dtype == "fp16" else 4.0
    b = score / (a * _D * scale)
    q = torch.full((q_len, _HQ, _D), a, device="cuda", dtype=tdt)
    k_cache = torch.full((num_blocks, _BS, _HKV, _D), b, device="cuda", dtype=tdt)
    v_cache = torch.randn(
        num_blocks, _BS, _HKV, _D, device="cuda", dtype=tdt, generator=g
    )
    perm = torch.randperm(num_blocks, device="cuda", generator=g)[:max_blocks]
    block_table = perm.view(1, max_blocks).to(torch.int32).contiguous()
    cu = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    seqused = torch.tensor([kv_len], dtype=torch.int32, device="cuda")

    problem = _problem(dtype, q_len, kv_len, num_blocks)
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=seqused,
        softmax_scale=scale,
        block_table=block_table,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(problem, branch),
    )
    torch.cuda.synchronize()
    o_ref = _reference(q, k_cache, v_cache, kv_len, block_table, scale)

    assert torch.isfinite(out).all(), "non-finite O"
    err = (out.float() - o_ref).abs()
    # Every row sees at least one key, so a collapsed row has |O| ~ 0 while the
    # reference is the mean of V over its window (|mean| ~ 0.1 for randn V).
    worst = err.amax(dim=(1, 2))
    tol = 2e-2 if dtype == "fp16" else 4e-2
    assert worst.max().item() < tol, (
        f"{(worst >= tol).sum().item()} of {q_len} query rows wrong; first bad "
        f"row {int((worst >= tol).nonzero()[0])}"
    )
