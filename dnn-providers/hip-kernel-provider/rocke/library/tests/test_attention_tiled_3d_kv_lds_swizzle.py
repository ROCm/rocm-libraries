# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Swizzled K/V LDS tiles (``use_kv_lds_swizzle``) of the gfx950 and gfx942 3D
segment kernels.

The numeric checks launch on ``rocke.runtime`` device arrays and compare with an
FP64 numpy decode reference; they need a gfx942 or gfx950 GPU.
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import numpy as np
import pytest
from kernels.gfx950.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
)
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.lanes import BINDING_LANE, import_or_skip_until

from ._attention_tiled_launch import launch_unified
from ._attention_unified_support import (
    decode_reference,
    paged_cache,
    random_half,
    strided_bshd,
)
from ._tiled_3d_binding_ir import lower_like_tiled_3d_binding


def _spec(**kw) -> UnifiedAttention3DTiledSpec:
    base: dict[str, object] = {
        "head_size": 128,
        "block_size": 16,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "dtype": "bf16",
        "use_sinks": False,
        "sliding_window": 0,
        "has_softcap": False,
        "num_segments": 16,
        "use_kv_lds_swizzle": True,
    }
    base.update(kw)
    return UnifiedAttention3DTiledSpec(**base)  # type: ignore[arg-type]


def test_needs_fp16_bf16_kv():
    with pytest.raises(ValueError, match="needs fp16/bf16 KV"):
        _spec(kv_storage_dtype="fp8e4m3")


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
@pytest.mark.parametrize("kv_layout", ["paged", "strided"])
def test_needs_power_of_two_chunks_per_row(arch, kv_layout):
    # head_size 192 has 24 chunks per row: chunk ^ (row % 24) leaves the row.
    make = _spec if arch == "gfx950" else _gfx942_spec
    extra = {} if arch == "gfx950" else {"use_wide_kv_load": False}
    with pytest.raises(ValueError, match="head_size / 8 a power of two"):
        make(head_size=192, kv_layout=kv_layout, **extra)
    make(head_size=192, kv_layout=kv_layout, use_kv_lds_swizzle=False, **extra)
    engine = import_or_skip_until("rocke_engine", BINDING_LANE)
    fields = asdict(
        make(head_size=192, kv_layout=kv_layout, use_kv_lds_swizzle=False, **extra)
    )
    fields["use_kv_lds_swizzle"] = True
    lower = getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")
    # gfx950 paged also trips the partial-row copy check first.
    with pytest.raises(Exception, match="use_kv_lds_swizzle"):
        lower(fields, arch)


def test_kernel_name_token():
    assert _spec().kernel_name().endswith("_kvswz")
    assert "kvswz" not in _spec(use_kv_lds_swizzle=False).kernel_name()


@pytest.mark.parametrize(
    "head_size, tile, kv_layout",
    [
        (64, 32, "paged"),
        (128, 64, "paged"),
        (128, 48, "paged"),
        (256, None, "paged"),
        (64, None, "strided"),
        (128, None, "strided"),
    ],
)
def test_native_builder_matches_python(head_size, tile, kv_layout):
    engine = import_or_skip_until("rocke_engine", BINDING_LANE)
    spec = _spec(
        head_size=head_size,
        num_query_heads=32,
        num_kv_heads=4,
        num_segments=8,
        num_seqs=3,
        tile_size_override=tile,
        use_decode_grid=True,
        kv_cache_policy="stream",
        kv_layout=kv_layout,
    )
    expected = lower_like_tiled_3d_binding(spec, "gfx950")
    actual = engine.gfx950_attention_tiled_3d_lower_llvm(asdict(spec), "gfx950")
    assert actual == expected


def _gfx942_spec(**kw):
    from kernels.gfx942.attention_tiled_3d import (
        UnifiedAttention3DTiledSpec as Gfx942Spec,
    )

    base: dict[str, object] = {
        "head_size": 128,
        "block_size": 16,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "dtype": "bf16",
        "use_sinks": False,
        "sliding_window": 0,
        "has_softcap": False,
        "num_segments": 16,
        "use_wide_kv_load": True,
        "use_kv_lds_swizzle": True,
    }
    base.update(kw)
    return Gfx942Spec(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("wide", [True, False])
def test_gfx942_spec_contract(wide):
    assert _gfx942_spec(use_wide_kv_load=wide).kernel_name().endswith("_kvswz")
    with pytest.raises(ValueError, match="needs fp16/bf16 KV"):
        _gfx942_spec(use_wide_kv_load=wide, kv_storage_dtype="fp8e4m3")


@pytest.mark.parametrize(
    "kw",
    [
        {"use_decode_grid": True, "kv_cache_policy": "stream"},
        {"head_size": 64, "dtype": "fp16", "use_sinks": True, "has_softcap": True},
        {"use_wide_kv_load": False, "head_size": 256, "num_query_heads": 64},
        {"use_wide_kv_load": False, "block_size": 32, "tile_size_override": 16},
        {"use_wide_kv_load": False, "kv_layout": "strided"},
        {"use_wide_kv_load": False, "kv_layout": "strided", "head_size": 256},
    ],
)
def test_gfx942_native_builder_matches_python(kw):
    engine = import_or_skip_until("rocke_engine", BINDING_LANE)
    spec = _gfx942_spec(num_seqs=3, **kw)
    expected = lower_like_tiled_3d_binding(spec, "gfx942")
    assert engine.gfx942_attention_tiled_3d_lower_llvm(asdict(spec), "gfx942") == (
        expected
    )


@pytest.mark.parametrize(
    "kw, arch, expected",
    [
        ({}, "gfx942", True),
        ({"head_size": 64}, "gfx942", True),
        ({"head_size": 256}, "gfx942", True),
        ({"head_size": 256, "block_size": 32}, "gfx942", False),
        ({"head_size": 256, "use_fp8": True}, "gfx942", False),
        ({"total_q": 8, "max_seqlen_q": 2}, "gfx942", False),
        ({"use_fp8": True}, "gfx942", False),
        ({}, "gfx1250", False),
    ],
)
def test_gfx942_dispatch_scope(kw, arch, expected):
    from kernels.common import attention_unified as au

    base: dict[str, object] = {
        "total_q": 4,
        "num_seqs": 4,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_seqlen_q": 1,
        "max_seqlen_k": 8192,
        "dtype": "bf16",
    }
    base.update(kw)
    problem = au.UnifiedAttentionProblem(**base)  # type: ignore[arg-type]
    assert au._kv_lds_swizzle_3d(problem, arch) is expected


def _decode_case(seed, dtype, head_size):
    """Q, logical K/V and ragged lengths: a partial last tile, a sequence
    shorter than one tile, and an empty one."""
    batch, q_heads, kv_heads, capacity = 3, 16, 4, 203
    rng = np.random.default_rng(seed)
    q = random_half(rng, (batch, q_heads, head_size), dtype)
    k = random_half(rng, (batch, kv_heads, capacity, head_size), dtype)
    v = random_half(rng, (batch, kv_heads, capacity, head_size), dtype)
    return q, k, v, [capacity, 5, 0]


def _assert_matches_reference(out, q, k, v, lengths, dtype):
    expected = decode_reference(q, k, v, lengths, 1 / math.sqrt(q.shape[2]))
    atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
    np.testing.assert_allclose(out, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("head_size, tile", [(64, 32), (128, 16), (128, 64), (256, 16)])
def test_swizzled_decode_matches_reference(dtype, head_size, tile):
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common import attention_unified as au

    current_device_arch_in(("gfx950",), "the gfx950 swizzled decode")
    block_size = 16
    q, k_logical, v_logical, lengths = _decode_case(
        29 + head_size + tile, dtype, head_size
    )
    batch, q_heads, _ = q.shape
    kv_heads, capacity = k_logical.shape[1:3]
    k, table = paged_cache(k_logical, block_size, dtype)
    v, _ = paged_cache(v_logical, block_size, dtype)
    problem = au.UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        dtype=dtype,
        num_kv_blocks=k.shape[0],
    )
    num_segments = 4
    segment = replace(
        au._tiled_3d_spec_from_problem(problem, "gfx950"),
        num_segments=num_segments,
        tile_size_override=None if tile == block_size else tile,
        use_kv_lds_swizzle=True,
    )
    reduce = UnifiedAttentionReduceTiledSpec(
        head_size=head_size,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        dtype=dtype,
        num_segments=num_segments,
    )
    spec = AttentionTuningSpec(
        path="3d",
        arch="gfx950",
        builder_kind="tiled_3d",
        compile_backend="llvm",
        candidate_name="kv_lds_swizzle_test",
        tuning_id="kv_lds_swizzle_test",
        kernel_spec=segment,
        reduce_spec=reduce,
        num_kv_blocks=k.shape[0],
    )
    out = launch_unified(
        problem,
        q,
        k,
        v,
        table,
        1 / math.sqrt(head_size),
        q_lens=[1] * batch,
        kv_lens=lengths,
        tuning_spec=spec,
    )
    _assert_matches_reference(out, q, k_logical, v_logical, lengths, dtype)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "head_size, wide",
    [(64, True), (128, True), (64, False), (128, False), (256, False)],
)
def test_gfx942_swizzled_decode_matches_reference(dtype, head_size, wide):
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common import attention_unified as au
    from kernels.gfx942.attention_tiled_3d import (
        UnifiedAttentionReduceTiledSpec as Gfx942Reduce,
    )

    current_device_arch_in(("gfx942",), "the gfx942 swizzled decode")
    block_size = 16
    q, k_logical, v_logical, lengths = _decode_case(31 + head_size, dtype, head_size)
    batch, q_heads, _ = q.shape
    kv_heads, capacity = k_logical.shape[1:3]
    k, table = paged_cache(k_logical, block_size, dtype)
    v, _ = paged_cache(v_logical, block_size, dtype)
    problem = au.UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        dtype=dtype,
        num_kv_blocks=k.shape[0],
    )
    num_segments = 4
    segment = replace(
        au._tiled_3d_spec_from_problem(problem, "gfx942"),
        num_segments=num_segments,
        use_wide_kv_load=wide,
        use_kv_lds_swizzle=True,
        use_wave_page_id_load=False,
    )
    reduce = Gfx942Reduce(
        head_size=head_size,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        dtype=dtype,
        num_segments=num_segments,
    )
    spec = AttentionTuningSpec(
        path="3d",
        arch="gfx942",
        builder_kind="tiled_3d",
        compile_backend="llvm",
        candidate_name="kv_lds_swizzle_test",
        tuning_id="kv_lds_swizzle_test",
        kernel_spec=segment,
        reduce_spec=reduce,
        num_kv_blocks=k.shape[0],
    )
    out = launch_unified(
        problem,
        q,
        k,
        v,
        table,
        1 / math.sqrt(head_size),
        q_lens=[1] * batch,
        kv_lens=lengths,
        tuning_spec=spec,
    )
    _assert_matches_reference(out, q, k_logical, v_logical, lengths, dtype)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_strided_swizzled_decode_matches_reference(dtype, head_size):
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common import attention_unified as au

    arch = current_device_arch_in(("gfx942", "gfx950"), "strided swizzled decode")
    q, k_logical, v_logical, lengths = _decode_case(53 + head_size, dtype, head_size)
    batch, q_heads, _ = q.shape
    kv_heads, capacity = k_logical.shape[1:3]
    problem = au.UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=head_size,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        dtype=dtype,
    )
    segment, reduce = au._strided_3d_specs_from_problem(problem, arch=arch)
    spec = AttentionTuningSpec(
        path="3d",
        arch=arch,
        builder_kind="tiled_3d",
        compile_backend="llvm",
        candidate_name="kv_lds_swizzle_test",
        tuning_id="kv_lds_swizzle_test",
        kernel_spec=replace(segment, use_kv_lds_swizzle=True),
        reduce_spec=reduce,
    )
    out = launch_unified(
        problem,
        q,
        # [batch, token, head, dim] storage seen as [batch, head, token, dim].
        strided_bshd(k_logical, dtype),
        strided_bshd(v_logical, dtype),
        None,
        1 / math.sqrt(head_size),
        q_lens=[1] * batch,
        kv_lens=lengths,
        tuning_spec=spec,
        kv_layout="strided",
    )
    _assert_matches_reference(out, q, k_logical, v_logical, lengths, dtype)
