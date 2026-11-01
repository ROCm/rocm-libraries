# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric check of the gfx950 tiled-2D QK paths with fp8 K kept in LDS.

With an fp8 KV cache and ``use_fp8_mfma_qk`` the kernel stores K in LDS as raw
fp8 bytes and dequantizes it (``* k_scale``) in registers before the bf16 QK
MFMA. Production dispatch does not pick this knob for every QK path, but tuning
and explicit knob pins can, so each path is launched here through an explicit
tuning spec and compared with an fp32 reference over the dequantized cache.

A ``k_scale`` that is not a power of two makes a missing or wrong scale visible.
The non-transposed 32x32 path once read the fp8 buffer as bf16; its output was
then far off the reference.

Marked ``gpu`` and skipped unless a gfx950 GPU with ROCm torch is present.
"""

from __future__ import annotations

import pytest

from dispatch.attention.common import AttentionTuningSpec
from kernels import UnifiedAttentionProblem, run_unified_attention_torch
from kernels.gfx950.attention_tiled_2d import UnifiedAttention2DTiledSpec

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


def _gpu_ready():
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName.lower()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"),
]

_HEAD_SIZE = 64
_BLOCK_SIZE = 32
_HQ, _HKV = 8, 8
# (query_len, kv_len) per sequence: a partial last KV tile and a short query.
_SEQ_LENS = [(96, 160), (40, 77)]
_K_SCALE = 0.75
_V_SCALE = 1.25
# bf16 output after fp8 K/V; the reference runs over the same dequantized values.
_TOL = 4e-2

# One spec per QK path that can hold fp8 K in LDS.
_QK_PATHS = {
    "16x16": dict(),
    "32x32 non-transposed": dict(use_mfma_32x32=True, block_m_per_warp=32),
    "32x32 transposed": dict(
        use_mfma_32x32=True, use_transposed_qk_32x32=True, block_m_per_warp=32
    ),
}


def _inputs(seed=0):
    torch.manual_seed(seed)
    query_lens = [s[0] for s in _SEQ_LENS]
    kv_lens = [s[1] for s in _SEQ_LENS]
    max_blocks = (max(kv_lens) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    num_blocks = len(_SEQ_LENS) * max_blocks
    q = torch.randn(sum(query_lens), _HQ, _HEAD_SIZE, device="cuda").bfloat16()
    shape = (num_blocks, _BLOCK_SIZE, _HKV, _HEAD_SIZE)
    k = torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)
    block_table = (
        torch.randperm(num_blocks, device="cuda")
        .to(torch.int32)
        .view(len(_SEQ_LENS), max_blocks)
        .contiguous()
    )
    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    return q, k, v, block_table, cu_q, seqused_k


def _reference(q, k, v, block_table, scale):
    """fp32 paged attention with a bottom-right causal mask over the
    dequantized cache (``k * k_scale``, ``v * v_scale``)."""
    outs = []
    start = 0
    for i, (q_len, kv_len) in enumerate(_SEQ_LENS):
        n_blk = (kv_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        blocks = block_table[i, :n_blk].long()
        ks = k[blocks].float().reshape(-1, _HKV, _HEAD_SIZE)[:kv_len] * _K_SCALE
        vs = v[blocks].float().reshape(-1, _HKV, _HEAD_SIZE)[:kv_len] * _V_SCALE
        ks = torch.repeat_interleave(ks, _HQ // _HKV, dim=1)
        vs = torch.repeat_interleave(vs, _HQ // _HKV, dim=1)
        s = torch.einsum("qhd,khd->hqk", q[start : start + q_len].float(), ks) * scale
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=q.device), diagonal=kv_len - q_len + 1
        ).bool()
        p = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vs))
        start += q_len
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize("path", sorted(_QK_PATHS))
def test_fp8_k_in_lds_matches_reference(path):
    q, k, v, block_table, cu_q, seqused_k = _inputs()
    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(_SEQ_LENS),
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=max(s[0] for s in _SEQ_LENS),
        max_seqlen_k=max(s[1] for s in _SEQ_LENS),
        dtype="bf16",
        use_fp8=True,
        num_kv_blocks=int(k.shape[0]),
    )
    kernel_spec = UnifiedAttention2DTiledSpec(
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        tile_size=64,
        kv_storage_dtype="fp8e4m3",
        use_fp8_mfma_qk=True,
        **_QK_PATHS[path],
    )
    tuning_spec = AttentionTuningSpec(
        path="2d",
        arch="gfx950",
        builder_kind="tiled",
        compile_backend="llvm",
        candidate_name="fp8_k_lds_numeric",
        tuning_id="fp8_k_lds_numeric",
        kernel_spec=kernel_spec,
        num_kv_blocks=problem.num_kv_blocks,
    )
    scale = _HEAD_SIZE**-0.5
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
        k_scale=_K_SCALE,
        v_scale=_V_SCALE,
        tuning_spec=tuning_spec,
    )
    torch.cuda.synchronize()
    max_abs = (_reference(q, k, v, block_table, scale) - out.float()).abs().max()
    assert max_abs.item() < _TOL, f"{path}: max_abs={max_abs.item():.3e}"
