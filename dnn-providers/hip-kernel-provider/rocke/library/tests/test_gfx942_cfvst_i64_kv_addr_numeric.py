# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU check: gfx942 cfvst prefill reads a paged KV cache past 2^31 bytes.

The default gfx942 bf16 D64 causal prefill V loader (conflict-free V store,
cfvst) used to form an i32 byte offset from the cache base, so a cache past
2^31 bytes wrapped (wrong data or a GPU fault), and the launch guard then
refused it. Under ``use_i64_kv_addr`` it now forms an i64 offset.

This launches the production route through ``run_unified_attention_torch`` on a
69632-block (2.125 GiB per K and V) cache with every block-table entry on the
top blocks, all past 2^31 bytes, and checks max_abs against the fp32 paged
reference. A wrapped offset reads other blocks' random data and fails the
check. The routing anchor runs without a GPU and pins that this problem
compiles an i64 cfvst kernel with no launch limit, so the numeric check cannot
silently test another kernel.
"""

from __future__ import annotations

from unittest import mock

import pytest

import builders.common.attention_spec_builder as _asb
from kernels import UnifiedAttentionProblem
from kernels.common import attention_unified as au

# 32 KiB blocks (32 tokens x 8 kv heads x D64 x bf16): 69632 blocks is 2.125 GiB,
# and every block from 65536 up starts past 2^31 bytes.
_NUM_BLOCKS = 69632
_FIRST_BLOCK_PAST_2GIB = 65536
_SEQ = 4096
_NUM_SEQS = 2
_BLOCK_SIZE = 32
# K and V caches (2.125 GiB each) plus the reference's fp32 working set.
_GPU_BYTES_NEEDED = 8 << 30


def _problem(num_kv_blocks: int = 0) -> UnifiedAttentionProblem:
    return UnifiedAttentionProblem(
        total_q=_SEQ * _NUM_SEQS,
        num_seqs=_NUM_SEQS,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=64,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=_SEQ,
        max_seqlen_k=_SEQ,
        dtype="bf16",
        num_kv_blocks=num_kv_blocks,
    )


def _gpu_ready() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    if not ("mi300" in name or "gfx942" in name):
        return False
    free, _ = torch.cuda.mem_get_info()
    return free >= _GPU_BYTES_NEEDED


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(),
    reason="needs a gfx942 GPU with ROCm torch and ~8 GiB free",
)


def _fp32_causal_paged_reference(query, key_cache, value_cache, block_tables, scale):
    """fp32 causal paged attention, one sequence and one KV head group at a time.

    Every sequence has ``_SEQ`` queries and keys, so the causal diagonal is the
    plain lower triangle. Chunking by KV head keeps the fp32 score matrix small.
    """
    import torch

    nq, nkv = query.shape[1], key_cache.shape[2]
    group = nq // nkv
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    causal = torch.ones(_SEQ, _SEQ, dtype=torch.bool, device=query.device).triu(1)
    for s in range(_NUM_SEQS):
        rows = slice(s * _SEQ, (s + 1) * _SEQ)
        blocks = block_tables[s].long()
        k = key_cache[blocks].reshape(_SEQ, nkv, -1).float()
        v = value_cache[blocks].reshape(_SEQ, nkv, -1).float()
        for h in range(nkv):
            heads = slice(h * group, (h + 1) * group)
            q = query[rows, heads].float() * scale  # [q, group, d]
            scores = torch.einsum("qgd,kd->gqk", q, k[:, h])
            scores.masked_fill_(causal, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[rows, heads] = torch.einsum("gqk,kd->qgd", probs, v[:, h])
    return out


def test_route_compiles_i64_cfvst_without_limit():
    p = _problem(_NUM_BLOCKS)
    with mock.patch.object(
        au, "_resolve_attention_arch", return_value="gfx942"
    ), mock.patch.dict(au._2D_LAUNCH_META, clear=True), mock.patch.dict(
        au._2D_KV_ADDR_LIMITS, clear=True
    ):
        spec = _asb._tiled_spec_from_problem(p, "gfx942")
        meta = au._get_2d_launch_meta(
            p, au._tiled_cache_key(p, "gfx942"), arch="gfx942"
        )
    assert spec.use_conflict_free_v_store
    assert spec.use_i64_kv_addr
    assert meta.kv_addr_limit is None


@requires_gfx942_gpu
def test_cfvst_prefill_on_top_blocks_past_2gib():
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    blocks_per_seq = _SEQ // _BLOCK_SIZE
    # Random data everywhere, so a wrapped offset reads values that differ.
    key_cache = torch.randn(
        _NUM_BLOCKS, _BLOCK_SIZE, 8, 64, dtype=torch.bfloat16, device=dev
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(_SEQ * _NUM_SEQS, 32, 64, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(query)
    # The topmost blocks in a fixed shuffle: the table includes the last block
    # and nothing below 2^31 bytes.
    used = _NUM_SEQS * blocks_per_seq
    top = torch.arange(_NUM_BLOCKS - used, _NUM_BLOCKS, dtype=torch.int32)
    top = top[torch.randperm(used, generator=torch.Generator().manual_seed(0))]
    assert int(top.min()) >= _FIRST_BLOCK_PAST_2GIB
    block_tables = top.view(_NUM_SEQS, blocks_per_seq).to(dev)
    cu_q = torch.arange(0, _SEQ * (_NUM_SEQS + 1), _SEQ, dtype=torch.int32, device=dev)
    kv_lens = torch.full((_NUM_SEQS,), _SEQ, dtype=torch.int32, device=dev)
    scale = 64**-0.5

    with mock.patch.object(au, "_resolve_attention_arch", return_value="gfx942"):
        # num_kv_blocks stays 0: the launcher fills it from k.shape[0], as in
        # production, which makes the cfvst spec i64.
        au.run_unified_attention_torch(
            problem=_problem(),
            q=query,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=cu_q,
            seqused_k=kv_lens,
            softmax_scale=scale,
            block_table=block_tables,
            softcap=0.0,
            backend="tiled",
        )
    torch.cuda.synchronize()

    ref = _fp32_causal_paged_reference(
        query, key_cache, value_cache, block_tables, scale
    )
    max_abs = float((ref - out.float()).abs().max())
    # 4e-2 is the repo-wide bf16 attention gate (ALGORITHM.md section 10).
    assert max_abs <= 4e-2, f"cfvst past 2 GiB: max_abs {max_abs:.4f} > 4e-2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
