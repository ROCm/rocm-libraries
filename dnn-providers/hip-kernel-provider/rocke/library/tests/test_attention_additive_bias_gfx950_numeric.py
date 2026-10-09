# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for additive attention bias on gfx950.

Drives ``run_unified_attention_torch`` with an fp32 additive bias on both the 2D
tiled (``backend="tiled"``) and 3D split-KV (``backend="3d"``) paths and checks
max_abs against an fp32 torch reference of
``softmax(Q @ K^T * scale + bias, causal) @ V``.

Bias addressing (element strides; Sk stride is always 1):
``bias[seq_idx*batch_stride + q_head*head_stride + q_local*sq_stride + k_abs]``.
A stride of 0 broadcasts that dimension. ``q_head`` is the *query* head, so
the GQA cases check that a per-query-head bias is not collapsed onto KV heads.

Every test is marked ``gpu`` and gated with a device skipif, so it is a graceful
skip on a CPU CI box. Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_additive_bias_gfx950_numeric.py
"""

from __future__ import annotations

import pytest

from kernels import UnifiedAttentionProblem, run_unified_attention_torch


torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_ready():
    """True only on a gfx950 box with ROCm torch (gate on ``gcnArchName``)."""
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return "gfx950" in arch


requires_gfx950_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"
)

_TORCH_DT = {"fp16": "float16", "bf16": "bfloat16"}
_TOL = {"fp16": 2e-2, "bf16": 4e-2}

_HEAD_SIZE = 128
_BLOCK_SIZE = 16

# (query_len, kv_len) per sequence. 2D covers mixed prefill with a context
# prefix; 3D covers decode-style split-KV with a tail block.
_SEQ_LENS = {
    "tiled": [(37, 101), (64, 64), (5, 130)],
    "3d": [(1, 300), (3, 517), (1, 64)],
}

# bias layouts: name -> (broadcast batch, broadcast head, extra Sk padding)
_BIAS_MODES = {
    "full": (False, False, 0),
    "full_padded_sq": (False, False, 7),
    "bcast_b": (True, False, 0),
    "bcast_h": (False, True, 0),
    "bcast_bh": (True, True, 0),
}

_HEADS = {"mha": (8, 8), "gqa": (8, 2)}


def _make_inputs(seq_lens, hq, hkv, dtype, seed=0):
    torch.manual_seed(seed)
    tdt = getattr(torch, _TORCH_DT[dtype])
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    max_blocks = (max(kv_lens) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    num_blocks = len(seq_lens) * max_blocks
    q = torch.randn(sum(query_lens), hq, _HEAD_SIZE, dtype=tdt, device="cuda")
    k = torch.randn(num_blocks, _BLOCK_SIZE, hkv, _HEAD_SIZE, dtype=tdt, device="cuda")
    v = torch.randn_like(k)
    # Disjoint shuffled pages so every sequence reads distinct KV.
    block_table = (
        torch.randperm(num_blocks, device="cuda")
        .to(torch.int32)
        .view(len(seq_lens), max_blocks)
        .contiguous()
    )
    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    return q, k, v, block_table, cu_q, seqused_k


def _make_bias(mode, num_seqs, hq, max_q, max_k):
    """Return (storage, batch_stride, head_stride, sq_stride, full_view).

    ``full_view`` is the logical ``[num_seqs, hq, max_q, max_k]`` bias the
    reference adds; ``storage`` is the compact tensor the kernel reads.
    """
    bcast_b, bcast_h, pad = _BIAS_MODES[mode]
    shape = [1 if bcast_b else num_seqs, 1 if bcast_h else hq, max_q, max_k + pad]
    storage = torch.randn(*shape, dtype=torch.float32, device="cuda")
    sq_stride = max_k + pad
    head_stride = 0 if bcast_h else max_q * sq_stride
    batch_stride = 0 if bcast_b else (1 if bcast_h else hq) * max_q * sq_stride
    full_view = storage[..., :max_k].expand(num_seqs, hq, max_q, max_k)
    return storage, batch_stride, head_stride, sq_stride, full_view


def _reference(q, k, v, block_table, seq_lens, scale, bias_full):
    """fp32 causal paged attention with an additive bias (bottom-right causal)."""
    hq, hkv = q.shape[1], k.shape[2]
    outs = []
    start = 0
    for i, (q_len, kv_len) in enumerate(seq_lens):
        n_blk = (kv_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        blocks = block_table[i, :n_blk].long()
        ks = k[blocks].reshape(-1, hkv, _HEAD_SIZE)[:kv_len].float()
        vs = v[blocks].reshape(-1, hkv, _HEAD_SIZE)[:kv_len].float()
        ks = torch.repeat_interleave(ks, hq // hkv, dim=1)
        vs = torch.repeat_interleave(vs, hq // hkv, dim=1)
        qs = q[start : start + q_len].float()
        s = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if bias_full is not None:
            s = s + bias_full[i, :, :q_len, :kv_len]
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=q.device), diagonal=kv_len - q_len + 1
        ).bool()
        s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vs))
        start += q_len
    return torch.cat(outs, dim=0)


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("heads", sorted(_HEADS))
@pytest.mark.parametrize("bias_mode", list(_BIAS_MODES))
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("backend", ["tiled", "3d"])
def test_additive_bias_matches_reference(backend, dtype, bias_mode, heads):
    hq, hkv = _HEADS[heads]
    seq_lens = _SEQ_LENS[backend]
    q, k, v, block_table, cu_q, seqused_k = _make_inputs(seq_lens, hq, hkv, dtype)
    max_q = max(s[0] for s in seq_lens)
    max_k = max(s[1] for s in seq_lens)
    scale = _HEAD_SIZE**-0.5
    storage, b_stride, h_stride, sq_stride, bias_full = _make_bias(
        bias_mode, len(seq_lens), hq, max_q, max_k
    )

    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(seq_lens),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dtype=dtype,
        use_additive_bias=True,
        num_kv_blocks=int(k.shape[0]),
    )
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        softmax_scale=scale,
        block_table=block_table,
        softcap=0.0,
        additive_bias=storage,
        additive_bias_batch_stride=b_stride,
        additive_bias_head_stride=h_stride,
        additive_bias_sq_stride=sq_stride,
        backend=backend,
    )
    torch.cuda.synchronize()

    ref = _reference(q, k, v, block_table, seq_lens, scale, bias_full)
    ref_nobias = _reference(q, k, v, block_table, seq_lens, scale, None)
    tol = _TOL[dtype]
    max_abs = (ref - out.float()).abs().max().item()
    assert (
        max_abs < tol
    ), f"{backend}/{dtype}/{bias_mode}/{heads}: max_abs={max_abs:.3e} >= {tol}"
    # Guard against a silently ignored bias: the bias must move the output.
    shift = (ref - ref_nobias).abs().max().item()
    assert shift > 10 * tol, f"bias too weak to be observable (shift={shift:.3e})"
