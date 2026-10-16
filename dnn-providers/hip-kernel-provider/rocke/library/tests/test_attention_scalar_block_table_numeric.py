# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scalar fallback launches with block-table rows wider than the batch needs.

Callers usually size block-table rows for the longest sequence they admit, not
the longest one in the current batch. The scalar kernels index rows by the
table's real row stride (the ``block_table_stride`` kernarg), so every sequence
after the first must still find its own blocks. Unused table slots point at a
poison block, so a wrong row stride corrupts the output instead of passing by
luck.

Reference attention is computed independently in FP32 from the logical K/V
after rounding inputs to bf16. Requires a ROCm GPU; skips otherwise.
"""

import itertools
import math

import pytest

from kernels.common import attention_unified as au

_HQ, _HKV, _D, _BS = 8, 2, 64, 16
_ATOL = 1e-2


@pytest.fixture(scope="module")
def torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    return torch


def _case(torch, q_lens, kv_lens, row_width, *, view_of_wider=False, seed=0):
    """Build a paged case whose block-table rows hold ``row_width`` slots."""
    gen = torch.Generator().manual_seed(seed)
    needed = [math.ceil(n / _BS) for n in kv_lens]
    num_blocks = sum(needed) + 1
    poison = num_blocks - 1
    order = torch.randperm(num_blocks - 1, generator=gen).tolist()
    alloc_width = row_width + 5 if view_of_wider else row_width
    table = torch.full((len(kv_lens), alloc_width), poison, dtype=torch.int32)
    for s, n in enumerate(needed):
        table[s, :n] = torch.tensor(order[:n], dtype=torch.int32)
        order = order[n:]

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16)

    q = randn(sum(q_lens), _HQ, _D)
    k = randn(num_blocks, _BS, _HKV, _D)
    v = randn(num_blocks, _BS, _HKV, _D)
    k[poison] = 64.0
    v[poison] = 64.0
    cu = torch.tensor([0, *itertools.accumulate(q_lens)], dtype=torch.int32)
    problem = au.UnifiedAttentionProblem(
        total_q=sum(q_lens),
        num_seqs=len(q_lens),
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_D,
        block_size=_BS,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        dtype="bf16",
    )
    dev = table.cuda()
    if view_of_wider:
        dev = dev[:, :row_width]
    return problem, q, k, v, cu, table, dev


def _reference(torch, q, k, v, cu, table, q_lens, kv_lens):
    out = torch.zeros(q.shape, dtype=torch.float32)
    ratio = _HQ // _HKV
    for s, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        blocks = table[s, : math.ceil(kv_len / _BS)].long()
        keys = k[blocks].reshape(-1, _HKV, _D)[:kv_len].float()
        values = v[blocks].reshape(-1, _HKV, _D)[:kv_len].float()
        context = kv_len - q_len
        for i in range(q_len):
            tok = int(cu[s]) + i
            visible = context + i + 1
            for h in range(_HQ):
                scores = keys[:visible, h // ratio] @ q[tok, h].float()
                probs = (scores / math.sqrt(_D)).softmax(0)
                out[tok, h] = probs @ values[:visible, h // ratio]
    return out


def _launch(torch, problem, q, k, v, cu, kv_lens, block_table):
    out = torch.empty(q.shape, dtype=q.dtype, device="cuda")
    au.run_unified_attention_torch(
        problem=problem,
        q=q.cuda(),
        k=k.cuda(),
        v=v.cuda(),
        out=out,
        cu_seqlens_q=cu.cuda(),
        seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        softmax_scale=1.0 / math.sqrt(_D),
        block_table=block_table,
        softcap=0.0,
        backend="scalar",
    )
    torch.cuda.synchronize()
    return out.float().cpu()


@pytest.mark.parametrize("view_of_wider", (False, True), ids=("wide", "view"))
def test_wide_block_table_rows(torch, view_of_wider):
    q_lens, kv_lens = [5, 1, 9], [37, 20, 61]
    row_width = math.ceil(max(kv_lens) / _BS) + 6
    problem, q, k, v, cu, table, dev = _case(
        torch, q_lens, kv_lens, row_width, view_of_wider=view_of_wider
    )
    assert dev.stride(0) > math.ceil(problem.max_seqlen_k / _BS)
    got = _launch(torch, problem, q, k, v, cu, kv_lens, dev)
    want = _reference(torch, q, k, v, cu, table, q_lens, kv_lens)
    max_abs = (got - want).abs().max().item()
    assert max_abs <= _ATOL, f"max_abs {max_abs:.3e} vs fp32 reference"


def test_one_binary_serves_different_batches(torch):
    """Shapes that differ only in batch geometry share a key; the second
    launch must reuse the first binary and still match the reference."""
    first = ([4, 2], [30, 17], 4)
    second = ([1, 7, 3, 2], [90, 64, 15, 33], 12)
    launchers = None
    for i, (q_lens, kv_lens, width) in enumerate((first, second)):
        problem, q, k, v, cu, table, dev = _case(torch, q_lens, kv_lens, width, seed=i)
        got = _launch(torch, problem, q, k, v, cu, kv_lens, dev)
        if launchers is None:
            launchers = len(au._SCALAR_LAUNCHERS)
        else:
            assert len(au._SCALAR_LAUNCHERS) == launchers
        want = _reference(torch, q, k, v, cu, table, q_lens, kv_lens)
        max_abs = (got - want).abs().max().item()
        assert max_abs <= _ATOL, f"case {i}: max_abs {max_abs:.3e}"
