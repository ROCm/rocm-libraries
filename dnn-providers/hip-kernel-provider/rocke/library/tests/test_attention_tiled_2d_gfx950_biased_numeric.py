# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 tiled 2D attention with score modifiers on the transposed 32x32 path.

The gfx950 selector keeps ALiBi, QQ-bias, softcap and sink problems on the
transposed 32x32 path: the multi-batch combo, the single-batch combo and the
multi-batch transposed cohort. Each case here launches the 2D kernel the
selector picks for such a problem and compares O with an FP32 reference over
the same fp16/bf16-rounded inputs. The KV cache is paged through a shuffled
block table whose spare slots point at a poison page, and the sinks are
uploaded in the working dtype, which is how the kernel reads them. Each case
first checks on the host that dropping any one of its modifiers from the
reference moves O by several tolerances, so a kernel that skipped one would
fail.

``test_each_modifier_on_each_route`` takes every modifier once and every route
once; ``test_modifier_mixes`` (exhaustive) runs the mixes and the other head
pairings. The kernels run on ``rocke.runtime`` device buffers filled from numpy,
so the tests need a gfx950 GPU but not torch.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest
from dispatch.attention.common import AttentionTuningSpec
from kernels.common import attention_unified as au
from kernels.common.attention_unified import UnifiedAttentionProblem
from rocke.runtime.hip_module import get_device_arch

from ._attention_paged_reference import (
    alibi_score_bias,
    paged_attention_reference,
    qq_score_bias,
    round_to_half,
)
from ._attention_tiled_launch import launch_tiled

ARCH = "gfx950"
_DEVICE_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _DEVICE_ARCH != ARCH, reason=f"needs a {ARCH} GPU, found {_DEVICE_ARCH}"
    ),
]

# max |O - reference| over every element; the 2D numeric tests' bounds.
TOLERANCE = {"fp16": 2e-2, "bf16": 4e-2}

# Every shape keeps max_seqlen_q above 256, the combo threshold, and has a
# sequence with q_len == kv_len, whose first query rows see only a few keys:
# there a sink or a capped score moves O far past the tolerance.
MULTI_Q_LENS, MULTI_KV_LENS = (300, 520), (600, 520)
SINGLE_Q_LENS, SINGLE_KV_LENS = (512,), (512,)

# Small enough to change the scores of unit-variance Q and K.
SOFTCAP = 4.0

# The modifiers each case must show the kernel applies: dropping any one of them
# from the reference moves O by more than this many tolerances.
MODIFIER_EFFECT_IN_TOLERANCES = 4

# The route each case must take, as the selector predicate that admits it.
COMBO = "combo"
SINGLE_BATCH_COMBO = "single-batch combo"
TRANSPOSED = "transposed 32x32"


class Case(NamedTuple):
    route: str
    head_size: int
    block_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    sliding_window: int = 0
    softcap: float = 0.0
    use_alibi: bool = False
    use_qq_bias: bool = False
    use_sinks: bool = False


def _combo(**changes):
    """The combo: bf16, d64, block size 32, GQA-8, max_seqlen_q above 256."""
    base = Case(COMBO, 64, 32, 64, 8, "bf16", MULTI_Q_LENS, MULTI_KV_LENS)
    return base._replace(**changes)


def _single_batch_combo(
    head_size, block_size, num_query_heads, num_kv_heads, **changes
):
    """One sequence; GQA-8 at d64 would take the combo instead."""
    base = Case(
        SINGLE_BATCH_COMBO,
        head_size,
        block_size,
        num_query_heads,
        num_kv_heads,
        "bf16",
        SINGLE_Q_LENS,
        SINGLE_KV_LENS,
    )
    return base._replace(**changes)


def _multi_batch_transposed(**changes):
    """The combo geometry at block size 16, which the combo does not take."""
    return _combo(route=TRANSPOSED, block_size=16, **changes)


EACH_MODIFIER_ON_EACH_ROUTE = {
    "combo-alibi": _combo(use_alibi=True),
    "combo-qq": _combo(use_qq_bias=True),
    "combo-softcap": _combo(softcap=SOFTCAP),
    "combo-window-alibi": _combo(sliding_window=128, use_alibi=True),
    "combo-fp16-sinks": _combo(dtype="fp16", use_sinks=True),
    "single-batch-d128-alibi": _single_batch_combo(128, 16, 64, 8, use_alibi=True),
    "multi-batch-d64-block16-qq": _multi_batch_transposed(use_qq_bias=True),
}

MODIFIER_MIXES = {
    "combo-alibi-qq": _combo(use_alibi=True, use_qq_bias=True),
    "combo-alibi-qq-softcap": _combo(use_alibi=True, use_qq_bias=True, softcap=SOFTCAP),
    "combo-heads32x4-alibi": _combo(num_query_heads=32, num_kv_heads=4, use_alibi=True),
    "combo-window-softcap": _combo(sliding_window=128, softcap=SOFTCAP),
    "combo-fp16-sinks-alibi": _combo(dtype="fp16", use_sinks=True, use_alibi=True),
    "combo-one-sequence-sinks-softcap": _combo(
        q_lens=SINGLE_Q_LENS, kv_lens=SINGLE_KV_LENS, use_sinks=True, softcap=SOFTCAP
    ),
    "single-batch-d128-qq": _single_batch_combo(128, 16, 64, 8, use_qq_bias=True),
    "single-batch-d128-fp16-sinks-alibi": _single_batch_combo(
        128, 16, 64, 8, dtype="fp16", use_sinks=True, use_alibi=True
    ),
    "single-batch-d64-alibi": _single_batch_combo(64, 32, 32, 32, use_alibi=True),
    "single-batch-d64-sinks": _single_batch_combo(64, 32, 32, 32, use_sinks=True),
    "multi-batch-d64-block16-alibi": _multi_batch_transposed(use_alibi=True),
}


def _problem(case: Case, num_kv_blocks: int) -> UnifiedAttentionProblem:
    return UnifiedAttentionProblem(
        total_q=sum(case.q_lens),
        num_seqs=len(case.q_lens),
        num_query_heads=case.num_query_heads,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        block_size=case.block_size,
        max_seqlen_q=max(case.q_lens),
        max_seqlen_k=max(case.kv_lens),
        dtype=case.dtype,
        sliding_window=case.sliding_window,
        softcap=case.softcap,
        use_sinks=case.use_sinks,
        use_alibi=case.use_alibi,
        use_qq_bias=case.use_qq_bias,
        num_kv_blocks=num_kv_blocks,
    )


def _takes_route(problem: UnifiedAttentionProblem, route: str) -> bool:
    """Whether the selector puts ``problem`` on ``route``; the combo takes
    precedence where its cohort overlaps the single-batch combo's."""
    combo = au._enable_combo_2d(problem, ARCH)
    single_batch = au._enable_single_batch_combo(problem, ARCH) and not combo
    if route == COMBO:
        return combo
    if route == SINGLE_BATCH_COMBO:
        return single_batch
    return au._enable_transposed_qk_32x32(problem, ARCH) and not (combo or single_batch)


def _inputs(case: Case, seed: int = 0):
    """float32 Q, paged K/V, block table and modifier tensors, every value the
    kernel reads in the working dtype already rounded to it."""
    rng = np.random.default_rng(seed)

    def randn(*shape):
        return round_to_half(rng.standard_normal(shape, np.float32), case.dtype)

    pages = [math.ceil(kv_len / case.block_size) for kv_len in case.kv_lens]
    num_blocks = sum(pages) + 1
    poison = num_blocks - 1
    order = iter(rng.permutation(poison))
    block_table = np.full((len(pages), max(pages) + 3), poison, np.int32)
    for seq, count in enumerate(pages):
        block_table[seq, :count] = [next(order) for _ in range(count)]
    cache_shape = (num_blocks, case.block_size, case.num_kv_heads, case.head_size)
    k_cache, v_cache = randn(*cache_shape), randn(*cache_shape)
    # A page the kernel must never read: any leak shows up as a large error.
    k_cache[poison] = 64.0
    v_cache[poison] = 64.0
    heads = np.arange(case.num_query_heads, dtype=np.float32)
    modifiers = {
        "alibi_slopes": (
            (2.0 ** (-8.0 * (heads + 1) / case.num_query_heads)).astype(np.float32)
            if case.use_alibi
            else None
        ),
        "qq_bias": (
            (rng.standard_normal((max(case.q_lens),) * 2, np.float32) * 2.0)
            if case.use_qq_bias
            else None
        ),
        "sinks": randn(case.num_query_heads) if case.use_sinks else None,
    }
    q = randn(sum(case.q_lens), case.num_query_heads, case.head_size)
    return q, k_cache, v_cache, block_table, modifiers


def _reference(case, q, k_cache, v_cache, block_table, modifiers, scale):
    out = np.zeros(q.shape, np.float32)
    start = 0
    for seq, (q_len, kv_len) in enumerate(zip(case.q_lens, case.kv_lens)):
        score_bias = np.zeros((1, q_len, kv_len))
        if modifiers["alibi_slopes"] is not None:
            score_bias = score_bias + alibi_score_bias(
                modifiers["alibi_slopes"], q_len, kv_len
            )
        if modifiers["qq_bias"] is not None:
            score_bias = score_bias + qq_score_bias(modifiers["qq_bias"], q_len, kv_len)
        out[start : start + q_len] = paged_attention_reference(
            q[start : start + q_len],
            k_cache,
            v_cache,
            kv_len,
            block_table[seq],
            scale,
            sliding_window=case.sliding_window,
            score_bias=score_bias,
            softcap=case.softcap,
            sinks=modifiers["sinks"],
        )
        start += q_len
    return out


def _each_modifier_dropped(case: Case, modifiers: dict):
    """``(name, case, modifiers)`` with one of the case's modifiers removed."""
    if case.softcap:
        yield "softcap", case._replace(softcap=0.0), modifiers
    for name, tensor in modifiers.items():
        if tensor is not None:
            yield name, case, {**modifiers, name: None}


def _check(case: Case):
    q, k_cache, v_cache, block_table, modifiers = _inputs(case)
    problem = _problem(case, num_kv_blocks=k_cache.shape[0])
    assert _takes_route(problem, case.route), f"no longer on the {case.route}"
    scale = 1.0 / math.sqrt(case.head_size)
    want = _reference(case, q, k_cache, v_cache, block_table, modifiers, scale)
    # A kernel that skipped a modifier must fail: each one moves O well past
    # the tolerance on these inputs.
    for name, reduced_case, reduced_modifiers in _each_modifier_dropped(
        case, modifiers
    ):
        without = _reference(
            reduced_case, q, k_cache, v_cache, block_table, reduced_modifiers, scale
        )
        effect = float(np.abs(want - without).max())
        assert (
            effect > MODIFIER_EFFECT_IN_TOLERANCES * TOLERANCE[case.dtype]
        ), f"dropping {name} moves O by only {effect:.3e}"
    kernel_spec = au._tiled_spec_from_problem(problem, ARCH)
    spec = AttentionTuningSpec(
        path="2d",
        arch=ARCH,
        builder_kind="tiled",
        compile_backend="llvm",
        candidate_name="biased_numeric",
        tuning_id="biased_numeric",
        kernel_spec=kernel_spec,
        num_kv_blocks=problem.num_kv_blocks,
    )
    out = launch_tiled(
        problem,
        spec,
        q,
        k_cache,
        v_cache,
        block_table,
        scale,
        q_lens=list(case.q_lens),
        kv_lens=list(case.kv_lens),
        softcap=case.softcap,
        **modifiers,
    )
    error = float(np.abs(out - want).max())
    assert error <= TOLERANCE[case.dtype], f"max |O - reference| = {error:.3e}"


@pytest.mark.parametrize(
    "case", EACH_MODIFIER_ON_EACH_ROUTE.values(), ids=EACH_MODIFIER_ON_EACH_ROUTE
)
def test_each_modifier_on_each_route(case):
    _check(case)


@pytest.mark.exhaustive
@pytest.mark.parametrize("case", MODIFIER_MIXES.values(), ids=MODIFIER_MIXES)
def test_modifier_mixes(case):
    _check(case)
