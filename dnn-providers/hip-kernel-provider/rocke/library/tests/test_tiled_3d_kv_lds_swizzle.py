# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Swizzled K/V LDS tiles (``use_kv_lds_swizzle``) of the gfx950 and gfx942 3D
segment kernels."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import pytest

from kernels.gfx950.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
)


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


@pytest.mark.parametrize(
    "kw", [{"kv_layout": "strided"}, {"kv_storage_dtype": "fp8e4m3"}]
)
def test_needs_paged_fp16_bf16_kv(kw):
    with pytest.raises(ValueError, match="needs paged fp16/bf16 KV"):
        _spec(**kw)


def test_kernel_name_token():
    assert _spec().kernel_name().endswith("_kvswz")
    assert "kvswz" not in _spec(use_kv_lds_swizzle=False).kernel_name()


@pytest.mark.parametrize("head_size, tile", [(64, 32), (128, 64), (256, None)])
def test_native_builder_matches_python(head_size, tile):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    engine = pytest.importorskip("rocke_engine")
    spec = _spec(
        head_size=head_size,
        num_query_heads=32,
        num_kv_heads=4,
        num_segments=8,
        num_seqs=3,
        tile_size_override=tile,
        use_decode_grid=True,
        kv_cache_policy="stream",
    )
    reduce = UnifiedAttentionReduceTiledSpec(
        head_size=head_size,
        num_query_heads=32,
        num_kv_heads=4,
        dtype="bf16",
        num_segments=8,
    )
    expected = _lower_kernel_to_llvm_python(
        build_unified_attention_3d_tiled(spec, arch="gfx950"), arch="gfx950"
    ) + _lower_kernel_to_llvm_python(
        build_unified_attention_reduce_tiled(reduce, arch="gfx950"), arch="gfx950"
    )
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


def test_gfx942_needs_the_wide_loader():
    with pytest.raises(ValueError, match="needs the wide K/V loader"):
        _gfx942_spec(use_wide_kv_load=False)
    with pytest.raises(ValueError, match="needs paged fp16/bf16 KV"):
        _gfx942_spec(kv_storage_dtype="fp8e4m3")
    assert _gfx942_spec().kernel_name().endswith("_kvswz")
    engine = pytest.importorskip("rocke_engine")
    spec = asdict(_gfx942_spec(use_kv_lds_swizzle=False, use_wide_kv_load=False))
    spec["use_kv_lds_swizzle"] = True
    with pytest.raises(Exception, match="needs the wide K/V loader"):
        engine.gfx942_attention_tiled_3d_lower_llvm(spec, "gfx942")


@pytest.mark.parametrize(
    "kw",
    [
        {"use_decode_grid": True, "kv_cache_policy": "stream"},
        {"head_size": 64, "dtype": "fp16", "use_sinks": True, "has_softcap": True},
    ],
)
def test_gfx942_native_builder_matches_python(kw):
    from kernels.gfx942.attention_tiled_3d import (
        build_unified_attention_3d_tiled as build_gfx942,
    )
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    engine = pytest.importorskip("rocke_engine")
    spec = _gfx942_spec(num_seqs=3, **kw)
    # The gfx942 binding lowers the segment kernel only.
    expected = _lower_kernel_to_llvm_python(
        build_gfx942(spec, arch="gfx942"), arch="gfx942"
    )
    assert engine.gfx942_attention_tiled_3d_lower_llvm(asdict(spec), "gfx942") == (
        expected
    )


@pytest.mark.parametrize(
    "kw, arch, expected",
    [
        ({}, "gfx942", True),
        ({"head_size": 64}, "gfx942", False),
        ({"head_size": 256}, "gfx942", False),
        ({"total_q": 8, "max_seqlen_q": 2}, "gfx942", False),
        ({"use_fp8": True}, "gfx942", False),
        ({}, "gfx1250", False),
    ],
)
def test_gfx942_dispatch_scope(monkeypatch, kw, arch, expected):
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
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)
    monkeypatch.delenv("HIPDNN_GFX942_3D_WKV", raising=False)
    problem = au.UnifiedAttentionProblem(**base)  # type: ignore[arg-type]
    assert au._kv_lds_swizzle_3d(problem) is expected


def _reference(torch, q, k, v, lengths):
    output = torch.zeros_like(q, dtype=torch.float32)
    ratio = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths):
        if length == 0:
            continue
        for head in range(q.shape[1]):
            keys = k[batch, head // ratio, :length].float()
            values = v[batch, head // ratio, :length].float()
            scores = (keys @ q[batch, head].float()) / math.sqrt(q.shape[2])
            output[batch, head] = scores.softmax(0) @ values
    return output


def _paged_cache(torch, logical, block_size):
    # Pages in reverse order, so neighbouring table entries are not
    # neighbouring pages.
    batch, heads, capacity, dim = logical.shape
    blocks = (capacity + block_size - 1) // block_size
    backing = torch.zeros(batch, blocks * block_size, heads, dim, dtype=logical.dtype)
    backing[:, :capacity].copy_(logical.permute(0, 2, 1, 3))
    pages = backing.reshape(batch * blocks, block_size, heads, dim).flip(0).contiguous()
    table = torch.arange(batch * blocks - 1, -1, -1, dtype=torch.int32).view(
        batch, blocks
    )
    return pages.cuda(), table.cuda()


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("head_size, tile", [(64, 32), (128, 16), (128, 64), (256, 16)])
def test_swizzled_decode_matches_reference(dtype, head_size, tile):
    torch = pytest.importorskip("torch")
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common import attention_unified as au
    from rocke.runtime.hip_module import get_device_arch

    if not torch.cuda.is_available() or get_device_arch() != "gfx950":
        pytest.skip("requires a gfx950 GPU")
    batch, q_heads, kv_heads, capacity, block_size = 3, 16, 4, 203, 16
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(29 + head_size + tile)
    q_host = torch.randn(batch, q_heads, head_size, generator=rng).to(tdtype)
    k_host = torch.randn(batch, kv_heads, capacity, head_size, generator=rng).to(tdtype)
    v_host = torch.randn(batch, kv_heads, capacity, head_size, generator=rng).to(tdtype)
    k, table = _paged_cache(torch, k_host, block_size)
    v, _ = _paged_cache(torch, v_host, block_size)
    # Ragged: a partial last tile, a sequence shorter than one tile, and an
    # empty one.
    lengths_host = [capacity, 5, 0]
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
        clamp_arch="gfx950",
        num_kv_blocks=k.shape[0],
    )
    num_segments = 4
    segment = replace(
        au._tiled_3d_spec_from_problem(problem),
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
    out = torch.empty_like(q_host).cuda()
    au.run_unified_attention_torch(
        problem=problem,
        q=q_host.cuda(),
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32).cuda(),
        seqused_k=torch.tensor(lengths_host, dtype=torch.int32).cuda(),
        softmax_scale=1 / math.sqrt(head_size),
        block_table=table,
        softcap=0.0,
        tuning_spec=spec,
    )
    torch.cuda.synchronize()
    expected = _reference(torch, q_host, k_host, v_host, lengths_host)
    atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
    torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("head_size", [64, 128])
def test_gfx942_swizzled_decode_matches_reference(dtype, head_size):
    torch = pytest.importorskip("torch")
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common import attention_unified as au
    from kernels.gfx942.attention_tiled_3d import (
        UnifiedAttentionReduceTiledSpec as Gfx942Reduce,
    )
    from rocke.runtime.hip_module import get_device_arch

    if not torch.cuda.is_available() or get_device_arch() != "gfx942":
        pytest.skip("requires a gfx942 GPU")
    batch, q_heads, kv_heads, capacity, block_size = 3, 16, 4, 203, 16
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(31 + head_size)
    q_host = torch.randn(batch, q_heads, head_size, generator=rng).to(tdtype)
    k_host = torch.randn(batch, kv_heads, capacity, head_size, generator=rng).to(tdtype)
    v_host = torch.randn(batch, kv_heads, capacity, head_size, generator=rng).to(tdtype)
    k, table = _paged_cache(torch, k_host, block_size)
    v, _ = _paged_cache(torch, v_host, block_size)
    # Ragged: a partial last tile, a sequence shorter than one tile, and an
    # empty one.
    lengths_host = [capacity, 5, 0]
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
        clamp_arch="gfx942",
        num_kv_blocks=k.shape[0],
    )
    num_segments = 4
    segment = replace(
        au._tiled_3d_spec_from_problem(problem),
        num_segments=num_segments,
        use_wide_kv_load=True,
        use_kv_lds_swizzle=True,
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
    out = torch.empty_like(q_host).cuda()
    au.run_unified_attention_torch(
        problem=problem,
        q=q_host.cuda(),
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32).cuda(),
        seqused_k=torch.tensor(lengths_host, dtype=torch.int32).cuda(),
        softmax_scale=1 / math.sqrt(head_size),
        block_table=table,
        softcap=0.0,
        tuning_spec=spec,
    )
    torch.cuda.synchronize()
    expected = _reference(torch, q_host, k_host, v_host, lengths_host)
    atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
    torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)
