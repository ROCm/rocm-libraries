# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sliding-window and sink prefill on the gfx950 single-batch transposed combo.

Single-sequence D64/D128 prefill with a sliding window and/or attention sinks
routes to the transposed 32x32 combo. Sinks seed the per-row running max, which
only the scalar-state layout maps to the right rows, so every sink request takes
it.

The routing assertions run on CPU; the numeric ones need a gfx950 GPU and compare
against the float64 NumPy paged attention in ``_unified_paged_numpy``.
"""

from __future__ import annotations

import pytest
from builders.common.attention_spec_builder import _tiled_spec_from_problem
from kernels.common.attention_unified import UnifiedAttentionProblem
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from . import _unified_paged_numpy as U

_ARCH = get_device_arch()

requires_gfx950_gpu = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"needs a gfx950 GPU, found {_ARCH}"
)


@pytest.fixture(scope="module")
def rt():
    return an.runtime()


# (name, head_size, seq_len, num_query_heads, num_kv_heads, block_size, dtype,
#  window, sinks)
_CASES = [
    ("d128_fp16_w128_sinks", 128, 1024, 64, 8, 64, "fp16", 128, True),
    ("d128_bf16_w256_sinks", 128, 1024, 64, 8, 64, "bf16", 256, True),
    ("d128_bf16_w512_bs16", 128, 1024, 32, 8, 16, "bf16", 512, False),
    ("d128_fp16_w512_bs32", 128, 1024, 32, 8, 32, "fp16", 512, False),
    ("d128_bf16_sinks", 128, 512, 64, 8, 64, "bf16", 0, True),
    ("d64_bf16_w128_sinks", 64, 2048, 64, 8, 64, "bf16", 128, True),
    ("d64_bf16_w128_sinks_short", 64, 512, 64, 8, 64, "bf16", 128, True),
    ("d64_fp16_w256", 64, 512, 64, 8, 64, "fp16", 256, False),
    ("d64_bf16_sinks", 64, 2048, 64, 8, 64, "bf16", 0, True),
]


def _problem(d, seq, hq, hkv, bs, dtype, window, sinks) -> UnifiedAttentionProblem:
    return UnifiedAttentionProblem(
        total_q=seq,
        num_seqs=1,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        dtype=dtype,
        sliding_window=window,
        use_sinks=sinks,
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c[0])
def test_routes_to_transposed_combo(case):
    _, d, seq, hq, hkv, bs, dtype, window, sinks = case
    spec = _tiled_spec_from_problem(
        _problem(d, seq, hq, hkv, bs, dtype, window, sinks), "gfx950"
    )
    assert spec.use_mfma_32x32 and spec.use_transposed_qk_32x32
    assert spec.block_m_per_warp == 32
    if window and d == 128:
        assert spec.tile_size == 64
    if sinks:
        assert spec.use_transposed_scalar_state


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("path", ["2d", "auto"])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c[0])
def test_matches_reference(rt, case, path):
    name, d, seq, hq, hkv, bs, dtype, window, sinks = case
    scenario = U.Scenario(
        name=name,
        seq_lens=[(seq, seq)],
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        dtype=dtype,
        sliding_window=window or None,
        use_sinks=sinks,
        num_blocks=seq // bs + 8,
    )
    data, ref = U.cached_case(scenario)
    out = U.launch(rt, scenario, data, path)
    an.assert_close(out, ref, dtype, f"{name}/{path}")
