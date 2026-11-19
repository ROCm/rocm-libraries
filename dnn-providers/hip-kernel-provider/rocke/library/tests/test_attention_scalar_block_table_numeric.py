# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scalar fallback launches with block-table rows wider than the batch needs.

Callers usually size block-table rows for the longest sequence they admit, not
the longest one in the current batch. The scalar kernels index rows by the
table's real row stride (the ``block_table_stride`` kernarg), so every sequence
after the first must still find its own blocks. Unused table slots point at a
poison block, so a wrong row stride corrupts the output instead of passing by
luck.

The launch goes through ``run_unified_attention_torch(backend="scalar")`` with
``rocke.runtime`` device buffers instead of torch tensors, so the test runs in
GPU lanes without torch. Reference attention is computed independently in FP32
with numpy from the logical K/V, after rounding inputs to fp16. Skips without a
GPU, and on devices whose target id the attention builders do not know.
"""

import itertools
import math

import numpy as np
import pytest

from kernels.common import attention_unified as au
from rocke.core.arch.target import known_arches
from rocke.runtime.hip_module import get_device_arch

from ._attention_device_array import DeviceArray as _DeviceArray

_HQ, _HKV, _D, _BS = 8, 2, 64, 16
_ATOL = 1e-2

# The launch compiles for the device's own target id, which the builders must
# know (gfx11 devices are only known through their generic target).
_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH not in known_arches(),
        reason=f"needs a GPU with a known attention target, found {_ARCH}",
    ),
]


@pytest.fixture(scope="module")
def rt():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


def _case(q_lens, kv_lens, row_width, *, seed):
    """A paged case whose block-table rows hold ``row_width`` slots."""
    rng = np.random.default_rng(seed)
    needed = [math.ceil(n / _BS) for n in kv_lens]
    num_blocks = sum(needed) + 1
    poison = num_blocks - 1
    order = iter(rng.permutation(num_blocks - 1))
    table = np.full((len(kv_lens), row_width), poison, dtype=np.int32)
    for s, n in enumerate(needed):
        table[s, :n] = [next(order) for _ in range(n)]

    def randn(*shape):
        return rng.standard_normal(shape).astype(np.float16)

    q = randn(sum(q_lens), _HQ, _D)
    k = randn(num_blocks, _BS, _HKV, _D)
    v = randn(num_blocks, _BS, _HKV, _D)
    k[poison] = 64.0
    v[poison] = 64.0
    cu = np.array([0, *itertools.accumulate(q_lens)], dtype=np.int32)
    problem = au.UnifiedAttentionProblem(
        total_q=sum(q_lens),
        num_seqs=len(q_lens),
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_D,
        block_size=_BS,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        dtype="fp16",
    )
    return problem, q, k, v, cu, table


def _reference(q, k, v, cu, table, q_lens, kv_lens):
    out = np.zeros(q.shape, dtype=np.float32)
    ratio = _HQ // _HKV
    for s, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        blocks = table[s, : math.ceil(kv_len / _BS)]
        keys = k[blocks].reshape(-1, _HKV, _D)[:kv_len].astype(np.float32)
        values = v[blocks].reshape(-1, _HKV, _D)[:kv_len].astype(np.float32)
        context = kv_len - q_len
        for i in range(q_len):
            tok = int(cu[s]) + i
            visible = context + i + 1
            for h in range(_HQ):
                scores = keys[:visible, h // ratio] @ q[tok, h].astype(np.float32)
                scores /= math.sqrt(_D)
                probs = np.exp(scores - scores.max())
                out[tok, h] = (probs / probs.sum()) @ values[:visible, h // ratio]
    return out


def _launch(rt, problem, q, k, v, cu, kv_lens, block_table):
    out = _DeviceArray(rt, np.zeros_like(q))
    au.run_unified_attention_torch(
        problem=problem,
        q=_DeviceArray(rt, q),
        k=_DeviceArray(rt, k),
        v=_DeviceArray(rt, v),
        out=out,
        cu_seqlens_q=_DeviceArray(rt, cu),
        seqused_k=_DeviceArray(rt, np.array(kv_lens, dtype=np.int32)),
        softmax_scale=1.0 / math.sqrt(_D),
        block_table=block_table,
        softcap=0.0,
        backend="scalar",
    )
    return out.read().astype(np.float32)


@pytest.mark.parametrize("view_of_wider", (False, True), ids=("wide", "view"))
def test_wide_block_table_rows(rt, view_of_wider):
    q_lens, kv_lens = [5, 1, 9], [37, 20, 61]
    row_width = math.ceil(max(kv_lens) / _BS) + 6
    alloc_width = row_width + 5 if view_of_wider else row_width
    problem, q, k, v, cu, table = _case(q_lens, kv_lens, alloc_width, seed=0)
    dev = _DeviceArray(rt, table, shape=(len(q_lens), row_width))
    assert dev.stride(0) > math.ceil(problem.max_seqlen_k / _BS)
    got = _launch(rt, problem, q, k, v, cu, kv_lens, dev)
    want = _reference(q, k, v, cu, table, q_lens, kv_lens)
    max_abs = float(np.abs(got - want).max())
    assert max_abs <= _ATOL, f"max_abs {max_abs:.3e} vs fp32 reference"


def test_one_binary_serves_different_batches(rt):
    """Shapes that differ only in batch geometry share a key; the second
    launch must reuse the first binary and still match the reference."""
    first = ([4, 2], [30, 17], 4)
    second = ([1, 7, 3, 2], [90, 64, 15, 33], 12)
    launchers = None
    for i, (q_lens, kv_lens, width) in enumerate((first, second)):
        problem, q, k, v, cu, table = _case(q_lens, kv_lens, width, seed=i)
        got = _launch(rt, problem, q, k, v, cu, kv_lens, _DeviceArray(rt, table))
        if launchers is None:
            launchers = len(au._SCALAR_LAUNCHERS)
        else:
            assert len(au._SCALAR_LAUNCHERS) == launchers
        want = _reference(q, k, v, cu, table, q_lens, kv_lens)
        max_abs = float(np.abs(got - want).max())
        assert max_abs <= _ATOL, f"case {i}: max_abs {max_abs:.3e}"
