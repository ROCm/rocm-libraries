# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free inputs, reference and launches for the paged unified-attention
cases of ``builders.gfx950.attention.prefill.unified_attention_cases``.

A case's sequences read ``ceil(kv_len / block_size)`` pages each. Their values
live in :attr:`PagedInputs.key_pages` / ``value_pages``, one row per page read,
at the cache slots :attr:`PagedInputs.page_ids`; every other slot of the
``num_blocks`` cache holds cheap random filler (see
:func:`._attention_unified_support.sparse_paged_cache`), so a cache of GiBs costs
only the pages a test reads, and a kernel that reads a wrong page is caught.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np
from builders.gfx950.attention.prefill.unified_attention_cases import Scenario
from kernels.common import attention_unified as au
from rocke.runtime.device_array import DeviceArray

from ._attention_paged_reference import (
    alibi_score_bias,
    decode_half,
    encode_half,
    paged_attention_reference,
    qq_score_bias,
)
from ._attention_unified_support import random_half, sparse_paged_cache

__all__ = ["PagedInputs", "case_problem", "case_reference", "make_inputs", "run_case"]

# The arch the gfx950 cases target, and the CU count the parity harness fixes
# its problems at, so a case selects the kernels the harness measures.
_ARCH = "gfx950"
_HARNESS_NUM_CUS = 120
# The compile route of every launch: COMGR, which every test environment has.
# The runner's automatic choice compiles some long prefills with hipcc, which
# the CI test environment does not have; these cases check the kernels, not
# the route, whose hipcc side the additive-bias lane covers.
_COMPILE_BACKEND = "llvm"


@dataclass
class PagedInputs:
    """One case's host inputs; see the module docstring for the page layout."""

    query: np.ndarray
    key_pages: np.ndarray
    value_pages: np.ndarray
    page_ids: np.ndarray
    block_tables: np.ndarray
    sinks: np.ndarray | None
    alibi_slopes: np.ndarray | None
    qq_bias: np.ndarray | None
    seed: int

    def pages_of(self, case: Scenario, seq: int) -> np.ndarray:
        """Rows of ``key_pages`` / ``value_pages`` sequence ``seq`` reads, in order."""
        counts = _page_counts(case)
        start = sum(counts[:seq])
        return np.arange(start, start + counts[seq])


def _page_counts(case: Scenario) -> list[int]:
    """The pages each sequence of ``case`` reads."""
    return [-(-kv // case.block_size) for _, kv in case.seq_lens]


def make_inputs(case: Scenario, seed: int = 0) -> PagedInputs:
    """Random inputs for ``case``: standard-normal Q/K/V and sinks rounded to the
    case dtype, distinct random cache slots for the pages the sequences read,
    and the parity harness's ALiBi slopes and query-to-query bias."""
    rng = np.random.default_rng(seed)
    query_lens = [q for q, _ in case.seq_lens]
    counts = _page_counts(case)
    page_shape = (case.block_size, case.num_kv_heads, case.head_size)
    page_ids = rng.choice(case.num_blocks, size=sum(counts), replace=False)
    # Columns past a sequence's pages hold valid but unread slots.
    block_tables = rng.integers(
        0, case.num_blocks, (len(counts), max(counts)), dtype=np.int32
    )
    for row, start, count in zip(
        block_tables, itertools.accumulate([0, *counts]), counts
    ):
        row[:count] = page_ids[start : start + count]
    qq_bias = None
    if case.use_qq_bias:
        rows = max(max(query_lens), case.qq_bias_stride_0)
        qq_bias = (
            rng.standard_normal((rows, case.qq_bias_stride_0), np.float32) * 0.1
        ).astype(np.float32)
    return PagedInputs(
        query=random_half(
            rng, (sum(query_lens), case.num_query_heads, case.head_size), case.dtype
        ),
        key_pages=random_half(rng, (sum(counts), *page_shape), case.dtype),
        value_pages=random_half(rng, (sum(counts), *page_shape), case.dtype),
        page_ids=page_ids.astype(np.int32),
        block_tables=block_tables,
        sinks=(
            random_half(rng, (case.num_query_heads,), case.dtype)
            if case.use_sinks
            else None
        ),
        alibi_slopes=(
            -np.linspace(0.05, 0.5, case.num_query_heads, dtype=np.float32)
            if case.use_alibi
            else None
        ),
        qq_bias=qq_bias,
        seed=seed,
    )


def case_reference(case: Scenario, inputs: PagedInputs) -> np.ndarray:
    """FP64 output of every sequence, ``[total_q, num_query_heads, head_size]``
    float32, computed one KV head group at a time to keep each score matrix
    small."""
    group = case.num_query_heads // case.num_kv_heads
    scale = case.head_size**-0.5
    out = np.empty(inputs.query.shape, np.float32)
    starts = itertools.accumulate([0, *(q for q, _ in case.seq_lens)])
    for seq, ((q_len, kv_len), start) in enumerate(zip(case.seq_lens, starts)):
        pages = inputs.pages_of(case, seq)
        # [num_query_heads, q_len, kv_len], or None without a score bias.
        bias = None
        if inputs.alibi_slopes is not None:
            bias = alibi_score_bias(
                inputs.alibi_slopes, q_len, kv_len, top_left=case.causal_top_left
            )
        if inputs.qq_bias is not None:
            qq = qq_score_bias(
                inputs.qq_bias, q_len, kv_len, top_left=case.causal_top_left
            )
            bias = np.broadcast_to(qq, (case.num_query_heads, q_len, kv_len)) + (
                0.0 if bias is None else bias
            )
        for kv_head in range(case.num_kv_heads):
            heads = slice(kv_head * group, (kv_head + 1) * group)
            out[start : start + q_len, heads] = paged_attention_reference(
                inputs.query[start : start + q_len, heads],
                inputs.key_pages[:, :, kv_head : kv_head + 1],
                inputs.value_pages[:, :, kv_head : kv_head + 1],
                kv_len,
                pages,
                scale,
                sliding_window=case.sliding_window or 0,
                top_left=case.causal_top_left,
                right_bound=case.right_bound,
                score_bias=None if bias is None else bias[heads],
                softcap=case.softcap,
                sinks=None if inputs.sinks is None else inputs.sinks[heads],
            )
    return out


def case_problem(case: Scenario, inputs: PagedInputs) -> au.UnifiedAttentionProblem:
    """The problem the parity harness launches ``case`` with."""
    return au.UnifiedAttentionProblem(
        total_q=inputs.query.shape[0],
        num_seqs=len(case.seq_lens),
        num_query_heads=case.num_query_heads,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        block_size=case.block_size,
        max_seqlen_q=max(q for q, _ in case.seq_lens),
        max_seqlen_k=max(kv for _, kv in case.seq_lens),
        dtype=case.dtype,
        sliding_window=case.sliding_window or 0,
        softcap=float(case.softcap),
        use_sinks=inputs.sinks is not None,
        use_alibi=inputs.alibi_slopes is not None,
        use_qq_bias=inputs.qq_bias is not None,
        num_cus=_HARNESS_NUM_CUS,
        num_kv_blocks=case.num_blocks,
        causal_top_left=case.causal_top_left,
        right_bound=case.right_bound,
    )


def run_case(
    case: Scenario,
    inputs: PagedInputs,
    path: str,
    *,
    problem: au.UnifiedAttentionProblem | None = None,
) -> np.ndarray:
    """Launch ``case`` on the current GPU and return O as float32.

    ``path`` ``"2d"`` builds the production tiled 2D spec for the problem and
    launches it directly, as the parity harness's 2D lane does (so a patched
    ``_tiled_spec_from_problem`` takes effect on every call); any other value is
    the ``backend`` of ``run_unified_attention_torch``. ``problem`` replaces
    :func:`case_problem`'s. Either way the launch compiles through COMGR: the
    problem's ``compile_backend`` is set to ``"llvm"``. The output starts as
    NaN, so an element no kernel writes fails the caller's checks.
    """
    problem = replace(
        problem or case_problem(case, inputs), compile_backend=_COMPILE_BACKEND
    )
    rng = np.random.default_rng(inputs.seed + 1)
    cache_shape = (case.num_blocks, case.block_size, case.num_kv_heads, case.head_size)
    out = DeviceArray(
        encode_half(np.full(inputs.query.shape, np.nan), case.dtype), case.dtype
    )
    arrays = {
        "q": DeviceArray(encode_half(inputs.query, case.dtype), case.dtype),
        "k": sparse_paged_cache(
            rng, cache_shape, inputs.page_ids, inputs.key_pages, case.dtype
        ),
        "v": sparse_paged_cache(
            rng, cache_shape, inputs.page_ids, inputs.value_pages, case.dtype
        ),
        "out": out,
        "cu_seqlens_q": DeviceArray(
            np.array([0, *itertools.accumulate(q for q, _ in case.seq_lens)], np.int32)
        ),
        "seqused_k": DeviceArray(np.array([kv for _, kv in case.seq_lens], np.int32)),
        "block_table": DeviceArray(inputs.block_tables),
        "sinks": (
            None
            if inputs.sinks is None
            else DeviceArray(encode_half(inputs.sinks, case.dtype), case.dtype)
        ),
        "alibi_slopes": (
            None if inputs.alibi_slopes is None else DeviceArray(inputs.alibi_slopes)
        ),
        "qq_bias": None if inputs.qq_bias is None else DeviceArray(inputs.qq_bias),
    }
    qq_stride = 0 if inputs.qq_bias is None else inputs.qq_bias.shape[1]
    if path == "2d":
        _launch_production_2d(problem, arrays, qq_stride, case.softcap)
    else:
        au.run_unified_attention_torch(
            problem=problem,
            **arrays,
            softmax_scale=case.head_size**-0.5,
            softcap=float(case.softcap),
            qq_bias_stride_0=qq_stride,
            backend=path,
        )
    return decode_half(out.values(), case.dtype)


def _launch_production_2d(problem, arrays, qq_stride: int, softcap: float) -> None:
    """Build, compile (through COMGR) and launch the production tiled 2D spec
    of ``problem``, outside the runner's launcher caches."""
    from kernels import build_unified_attention_2d_tiled
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    spec = au._tiled_spec_from_problem(problem, _ARCH)
    kernel = build_unified_attention_2d_tiled(spec, arch=_ARCH)
    artifact = compile_kernel(kernel, arch=_ARCH, capture_ir_text=False)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=au.tiled_2d_signature(_ARCH, problem.dtype),
    )
    values = au._attn_values(
        problem=problem,
        q=arrays["q"],
        k=arrays["k"],
        v=arrays["v"],
        out=arrays["out"],
        cu_seqlens_q=arrays["cu_seqlens_q"],
        seqused_k=arrays["seqused_k"],
        softmax_scale=problem.head_size**-0.5,
        block_table=arrays["block_table"],
        softcap=float(softcap),
        sinks=arrays["sinks"],
        bt_stride=arrays["block_table"].stride(0),
        include_bt_stride=True,
        alibi_slopes=arrays["alibi_slopes"],
        qq_bias=arrays["qq_bias"],
        qq_bias_stride_0=qq_stride,
        include_qq_bias_stride=True,
        kv_addr_limit=au._spec_kv_addr_limit(problem, spec),
    )
    num_q_blocks = problem.total_q // spec.block_q + problem.num_seqs
    launcher(
        values,
        config=LaunchConfig(
            grid=(problem.num_kv_heads, num_q_blocks, 1),
            block=(64 * spec.num_warps, 1, 1),
        ),
    )
