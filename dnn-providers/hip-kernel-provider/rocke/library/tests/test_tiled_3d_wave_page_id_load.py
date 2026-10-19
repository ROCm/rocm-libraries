# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wave page-id load (``use_wave_page_id_load``) of the gfx950 and gfx942 3D
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
        "use_wave_page_id_load": True,
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
    assert _spec().kernel_name().endswith("_wpid")
    assert "wpid" not in _spec(use_wave_page_id_load=False).kernel_name()


@pytest.mark.parametrize("tile, swizzle", [(None, False), (32, False), (64, True)])
def test_native_builder_matches_python(tile, swizzle):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    engine = pytest.importorskip("rocke_engine")
    spec = _spec(
        num_query_heads=32,
        num_kv_heads=4,
        num_segments=8,
        num_seqs=3,
        tile_size_override=tile,
        use_decode_grid=True,
        kv_cache_policy="stream",
        use_kv_lds_swizzle=swizzle,
    )
    reduce = UnifiedAttentionReduceTiledSpec(
        head_size=128,
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
        "use_wave_page_id_load": True,
    }
    base.update(kw)
    return Gfx942Spec(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kw",
    [{}, {"use_wide_kv_load": False}, {"block_size": 32, "tile_size_override": 16}],
)
def test_gfx942_spec_contract(kw):
    assert _gfx942_spec(**kw).kernel_name().endswith("_wpid")
    with pytest.raises(ValueError, match="needs paged fp16/bf16 KV"):
        _gfx942_spec(kv_storage_dtype="fp8e4m3", **kw)


@pytest.mark.parametrize(
    "kw",
    [
        {
            "use_decode_grid": True,
            "kv_cache_policy": "stream",
            "use_kv_lds_swizzle": True,
        },
        {"head_size": 64, "block_size": 32, "dtype": "fp16", "sliding_window": 64},
        {"use_wide_kv_load": False, "head_size": 256, "use_kv_lds_swizzle": True},
        {"use_wide_kv_load": False, "dtype": "fp16", "use_sinks": True},
        {"block_size": 32, "tile_size_override": 16, "use_kv_lds_swizzle": True},
        {"use_wide_kv_load": False, "block_size": 32, "tile_size_override": 16},
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
        ({"head_size": 64}, "gfx942", True),
        ({"head_size": 256}, "gfx942", True),
        ({"head_size": 256, "block_size": 32}, "gfx942", False),
        ({"head_size": 256, "use_fp8": True}, "gfx942", False),
        ({"block_size": 32}, "gfx942", False),
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
    assert au._wave_page_id_load_3d(problem, arch) is expected


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
@pytest.mark.parametrize(
    "tile, num_segments, swizzle", [(16, 1, False), (16, 4, False), (64, 2, True)]
)
def test_wave_page_id_load_matches_reference(dtype, tile, num_segments, swizzle):
    torch = pytest.importorskip("torch")
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common import attention_unified as au
    from rocke.runtime.hip_module import get_device_arch

    if not torch.cuda.is_available() or get_device_arch() != "gfx950":
        pytest.skip("requires a gfx950 GPU")
    # 2203 tokens = 138 pages: a one-segment split spans three 64-page windows.
    batch, q_heads, kv_heads, capacity, block_size = 3, 16, 4, 2203, 16
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(17 + tile + num_segments)
    q_host = torch.randn(batch, q_heads, 128, generator=rng).to(tdtype)
    k_host = torch.randn(batch, kv_heads, capacity, 128, generator=rng).to(tdtype)
    v_host = torch.randn(batch, kv_heads, capacity, 128, generator=rng).to(tdtype)
    k, table = _paged_cache(torch, k_host, block_size)
    v, _ = _paged_cache(torch, v_host, block_size)
    # Ragged: a partial last tile, a sequence shorter than one tile, and an
    # empty one.
    lengths_host = [capacity, 5, 0]
    # Poison: table entries past a sequence's last page point at an extra
    # all-NaN page; a kernel that reads them returns NaN.
    nan_page = torch.full_like(k[:1], float("nan"))
    k = torch.cat([k, nan_page])
    v = torch.cat([v, nan_page])
    for i, length in enumerate(lengths_host):
        table[i, (length + block_size - 1) // block_size :] = k.shape[0] - 1
    problem = au.UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=128,
        block_size=block_size,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        dtype=dtype,
        num_kv_blocks=k.shape[0],
    )
    segment = replace(
        au._tiled_3d_spec_from_problem(problem, "gfx950"),
        num_segments=num_segments,
        tile_size_override=None if tile == block_size else tile,
        use_kv_lds_swizzle=swizzle,
        use_wave_page_id_load=True,
    )
    reduce = UnifiedAttentionReduceTiledSpec(
        head_size=128,
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
        candidate_name="wave_page_id_load_test",
        tuning_id="wave_page_id_load_test",
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
        softmax_scale=1 / math.sqrt(128),
        block_table=table,
        softcap=0.0,
        tuning_spec=spec,
    )
    torch.cuda.synchronize()
    expected = _reference(torch, q_host, k_host, v_host, lengths_host)
    atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
    torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "head_size, block_size, tile, wide, num_segments, swizzle",
    [
        (128, 16, 16, True, 1, False),
        (128, 16, 16, True, 4, True),
        (128, 16, 16, False, 1, False),
        (128, 16, 16, False, 4, True),
        (256, 16, 16, False, 2, True),
        (128, 32, 16, True, 1, True),
        (128, 32, 16, False, 1, False),
        (128, 32, 16, False, 4, True),
    ],
)
def test_gfx942_wave_page_id_load_matches_reference(
    dtype, head_size, block_size, tile, wide, num_segments, swizzle
):
    torch = pytest.importorskip("torch")
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common import attention_unified as au
    from kernels.gfx942.attention_tiled_3d import (
        UnifiedAttentionReduceTiledSpec as Gfx942Reduce,
    )
    from rocke.runtime.hip_module import get_device_arch

    if not torch.cuda.is_available() or get_device_arch() != "gfx942":
        pytest.skip("requires a gfx942 GPU")
    # 2203 tokens: a one-segment split spans several 64-page windows.
    batch, q_heads, kv_heads, capacity = 3, 16, 4, 2203
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(41 + num_segments + head_size + block_size)
    q_host = torch.randn(batch, q_heads, head_size, generator=rng).to(tdtype)
    k_host = torch.randn(batch, kv_heads, capacity, head_size, generator=rng)
    v_host = torch.randn(batch, kv_heads, capacity, head_size, generator=rng)
    k_host, v_host = k_host.to(tdtype), v_host.to(tdtype)
    k, table = _paged_cache(torch, k_host, block_size)
    v, _ = _paged_cache(torch, v_host, block_size)
    # Ragged: a partial last tile, a sequence shorter than one tile, and an
    # empty one.
    lengths_host = [capacity, 5, 0]
    # Poison: table entries past a sequence's last page point at an extra
    # all-NaN page; a kernel that reads them returns NaN.
    nan_page = torch.full_like(k[:1], float("nan"))
    k = torch.cat([k, nan_page])
    v = torch.cat([v, nan_page])
    for i, length in enumerate(lengths_host):
        table[i, (length + block_size - 1) // block_size :] = k.shape[0] - 1
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
    segment = replace(
        au._tiled_3d_spec_from_problem(problem, "gfx942"),
        num_segments=num_segments,
        tile_size_override=None if tile == block_size else tile,
        use_wide_kv_load=wide,
        use_kv_lds_swizzle=swizzle,
        use_wave_page_id_load=True,
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
        candidate_name="wave_page_id_load_test",
        tuning_id="wave_page_id_load_test",
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
