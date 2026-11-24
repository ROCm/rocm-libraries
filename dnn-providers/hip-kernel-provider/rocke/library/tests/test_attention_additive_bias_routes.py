# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where a biased gfx950 problem routes, pinned without a GPU.

The numeric lane (``test_attention_additive_bias_gfx950_numeric.py``) is GPU
only, so a selector change that moved one of its cases onto another score path
or compile backend would go unnoticed in CI. The route is a pure function of
the problem, so this file pins it for every 2D case of that lane, and the knobs
the bias must set on the two transposed 32x32 cohorts. It also pins that the
builders with no bias score site (fastKV register-P, the scalar 2D and 3D
kernels) refuse a biased spec, since nothing routes a biased problem to them.
"""

from dataclasses import replace

import pytest

from kernels import UnifiedAttentionProblem
from kernels.common.attention_unified import (
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    _select_2d_compile_backend,
    _tiled_spec_from_problem,
    build_unified_attention_2d,
    build_unified_attention_3d,
    supports_native_unified_attention,
)

from .test_attention_additive_bias_gfx950_numeric import (
    _BLOCK_SIZE,
    _CASES,
    _COMBO_HEADS,
    _COMBO_SEQ_LENS,
    _EXTRA_CASES,
    _HEAD_SIZE,
    _HEADS,
)

# case -> {dtype: (score path, compile backend)}.
_ROUTES = {
    "tiled": {"fp16": ("16x16", "llvm"), "bf16": ("16x16", "llvm")},
    "tiled_long": {"fp16": ("transposed32", "llvm"), "bf16": ("transposed32", "llvm")},
    "tiled_long_1seq": {
        "fp16": ("transposed32", "llvm"),
        "bf16": ("transposed32", "llvm"),
    },
    "tiled_long_hipcc": {
        "fp16": ("transposed32", "hipcc"),
        "bf16": ("transposed32", "hipcc"),
    },
    "tiled_softcap": {"fp16": ("16x16", "llvm"), "bf16": ("16x16", "llvm")},
    "tiled_window": {"fp16": ("16x16", "llvm"), "bf16": ("16x16", "llvm")},
    "tiled_window_1seq": {
        "fp16": ("transposed32", "llvm"),
        "bf16": ("transposed32", "llvm"),
    },
}

# With bias, the transposed 32x32 combo cohorts keep mask-once and mask-limit off.
_BIAS_KNOBS = {"use_transposed_mask_once": False, "use_transposed_mask_limit": False}


def _problem(seq_lens, hq, hkv, dtype, *, head_size, block_size, **fields):
    """The problem the numeric lane builds for these sequences."""
    max_q = max(q for q, _ in seq_lens)
    max_k = max(k for _, k in seq_lens)
    return UnifiedAttentionProblem(
        total_q=sum(q for q, _ in seq_lens),
        num_seqs=len(seq_lens),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dtype=dtype,
        use_additive_bias=True,
        num_kv_blocks=len(seq_lens) * -(-max_k // block_size),
        **fields,
    )


def _route(problem):
    spec = _tiled_spec_from_problem(problem, "gfx950")
    assert spec.use_additive_bias
    if spec.use_mfma_32x32 and spec.use_transposed_qk_32x32:
        assert spec.block_m_per_warp == 32
        path = "transposed32"
    elif spec.use_mfma_32x32:
        path = "32x32"
    else:
        path = "16x16"
    return (path, _select_2d_compile_backend(problem, "gfx950")), spec


def _seq_lens_and_features(case):
    if case in _CASES:
        return _CASES[case][1], {}
    return _EXTRA_CASES[case][1], _EXTRA_CASES[case][2]


def test_every_2d_numeric_case_has_a_pinned_route():
    cases = {**_CASES, **_EXTRA_CASES}
    assert {c for c, spec in cases.items() if spec[0] == "tiled"} == set(_ROUTES)


@pytest.mark.parametrize("heads", sorted(_HEADS))
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", sorted(_ROUTES))
def test_numeric_case_route(case, dtype, heads):
    seq_lens, features = _seq_lens_and_features(case)
    hq, hkv = _HEADS[heads]
    problem = _problem(
        seq_lens,
        hq,
        hkv,
        dtype,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        softcap=float(features.get("softcap", 0.0)),
        sliding_window=int(features.get("sliding_window", 0)),
    )
    route, _ = _route(problem)
    assert route == _ROUTES[case][dtype]


@pytest.mark.parametrize("heads", sorted(_COMBO_HEADS))
def test_combo_cohort_route_and_knobs(heads):
    # d64 / block 32 / GQA-8 bf16 prefill: the transposed 32x32 combo stack.
    hq, hkv = _COMBO_HEADS[heads]
    problem = _problem(_COMBO_SEQ_LENS, hq, hkv, "bf16", head_size=64, block_size=32)
    route, spec = _route(problem)
    assert route == ("transposed32", "llvm")
    assert spec.use_transposed_scalar_state
    assert spec.use_mfma32_skip_legacy_qreg
    for knob, want in _BIAS_KNOBS.items():
        assert getattr(spec, knob) == want, knob


@pytest.mark.parametrize("head_size", [64, 128])
def test_fp8_single_batch_cohort_route_and_knobs(head_size):
    # fp8 K/V single batch (block 64, 64/8 heads): the transposed 32x32 combo,
    # with the bias turning its mask-once / mask-limit opts off.
    problem = _problem(
        [(384, 640)], 64, 8, "bf16", head_size=head_size, block_size=64, use_fp8=True
    )
    route, spec = _route(problem)
    assert route[0] == "transposed32"
    assert spec.kv_storage_dtype == "fp8e4m3"
    for knob, want in _BIAS_KNOBS.items():
        assert getattr(spec, knob) == want, knob


def test_fastkv_register_p_rejects_additive_bias():
    """The fastKV register-P wrapper has no bias score site, so the wrapper
    itself refuses a biased spec; the same spec without bias builds."""
    from kernels import UnifiedAttention2DTiledSpec
    from kernels.gfx950.attention_tiled_2d_fastkv_regp import (
        build_unified_attention_2d_fastkv_register_p,
    )

    # The fastKV parity family's base, with mask-once off: the spec refuses
    # bias with mask-once before the wrapper is reached.
    spec = UnifiedAttention2DTiledSpec(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        waves_per_eu=2,
        tile_size=64,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_fast_paged_kv_desc=True,
        use_additive_bias=True,
    )
    with pytest.raises(
        ValueError, match="fastKV register-P experiment does not support additive bias"
    ):
        build_unified_attention_2d_fastkv_register_p(spec, arch="gfx950")
    build_unified_attention_2d_fastkv_register_p(
        replace(spec, use_additive_bias=False), arch="gfx950"
    )


@pytest.mark.parametrize(
    ("entry", "config", "reason"),
    [
        (
            "gfx950_attention_tiled_2d_fastkv_regp_lower_llvm",
            {"use_transposed_mask_once": False},
            "fastKV register-P experiment does not support additive bias",
        ),
        (
            "attention_unified_lower_llvm",
            {
                "head_size": 128,
                "block_size": 16,
                "num_query_heads": 8,
                "num_kv_heads": 2,
                "total_q": 64,
                "max_seqlen_q": 64,
                "max_seqlen_k": 64,
                "num_seqs": 1,
                "dtype": "bf16",
            },
            "scalar 2D kernel: additive bias runs on the gfx950 tiled kernels only",
        ),
    ],
)
def test_native_builders_refuse_additive_bias(entry, config, reason):
    # The C++ fastKV and scalar builders take no bias. Through the rocke_engine
    # binding (built with ROCKE_BUILD_PYBIND=ON; CI does not build it) a biased
    # config must hit their reject, not build a kernel that drops the bias.
    engine = pytest.importorskip("rocke_engine")
    lower = getattr(engine, entry)
    assert lower(config, "gfx950")
    with pytest.raises(RuntimeError, match=reason):
        lower({**config, "use_additive_bias": True}, "gfx950")


@pytest.mark.parametrize("kind", ["2D", "3D"])
def test_scalar_builders_refuse_additive_bias(kind):
    # The scalar kernels have no bias score site. Their public builders must
    # refuse a biased problem, as the C++ ones do, not emit a kernel that
    # silently drops the bias.
    def problem(bias):
        return UnifiedAttentionProblem(
            total_q=64,
            num_seqs=1,
            num_query_heads=8,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            max_seqlen_q=64,
            max_seqlen_k=64,
            dtype="bf16",
            use_additive_bias=bias,
        )

    def build(bias):
        if kind == "2D":
            return build_unified_attention_2d(
                UnifiedAttention2DSpec(problem=problem(bias))
            )
        return build_unified_attention_3d(
            UnifiedAttention3DSpec(problem=problem(bias), num_segments=8)
        )

    assert build(False).name
    reason = (
        f"scalar {kind} kernel: additive bias runs on the gfx950 tiled kernels only"
    )
    with pytest.raises(ValueError, match=reason):
        build(True)
    if kind == "2D":
        assert supports_native_unified_attention(problem(True), "gfx950") == (
            False,
            reason,
        )
