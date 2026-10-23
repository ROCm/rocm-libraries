# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Loader boundary agreement between concrete specs, admission and engines."""

from dataclasses import asdict
from importlib import import_module
from types import SimpleNamespace

import pytest
from rocke.core.lower_llvm import LLVM_FLAVORS

from kernels.common._attention_kv_loader import (
    AttentionKVLoaderKind,
    resolve_attention_kv_loader,
)
from kernels.common.attention_paged_decode import validate_paged_loader_geometry
from kernels.common.attention_unified import UnifiedAttentionProblem


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "page,tile", [(1, 32), (16, 16), (16, 32), (32, 32), (64, 32), (64, 64)]
)
@pytest.mark.parametrize("layout", ["paged", "strided"])
@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_loader_boundary_engine_agreement(
    arch, page, tile, layout, flavor, monkeypatch
):
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    segment = module.UnifiedAttention3DTiledSpec(
        head_size=64,
        block_size=page,
        tile_size_override=tile,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=8,
        num_seqs=3,
        kv_layout=layout,
    )
    gather = layout == "paged" and (
        (page, tile) in ((1, 32), (16, 32))
        or (arch == "gfx950" and (page, tile) == (64, 32))
    )
    expected_kind = (
        AttentionKVLoaderKind.STRIDED
        if layout == "strided"
        else (
            AttentionKVLoaderKind.PAGED_TOKEN_GATHER
            if gather
            else AttentionKVLoaderKind.PAGED_DESCRIPTOR
        )
    )
    assert resolve_attention_kv_loader(arch, layout, page, tile) is expected_kind
    assert segment.uses_paged_gather is gather
    reduce = module.UnifiedAttentionReduceTiledSpec(
        head_size=64,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="fp16",
        num_segments=8,
    )
    if layout == "paged":
        problem = UnifiedAttentionProblem(
            total_q=3,
            num_seqs=3,
            num_query_heads=8,
            num_kv_heads=2,
            head_size=64,
            block_size=page,
            max_seqlen_q=1,
            max_seqlen_k=97,
            dtype="fp16",
            clamp_arch=arch,
        )
        # This structural protocol has no internal loader-kind field.
        snapshot = SimpleNamespace(
            **asdict(segment), tile_size=tile, uses_paged_gather=gather
        )
        tuning = SimpleNamespace(
            arch=arch, path="3d", kernel_spec=snapshot, reduce_spec=reduce
        )
        if gather:
            # Raw builders retain the caller-validity contract tested below;
            # an unmarked pair cannot enter the runtime gather launcher.
            with pytest.raises(ValueError, match="metadata containment"):
                validate_paged_loader_geometry(problem, tuning)
        else:
            assert validate_paged_loader_geometry(problem, tuning) == (
                page,
                tile,
                False,
            )

    expected = _lower_kernel_to_llvm_python(
        module.build_unified_attention_3d_tiled(segment, arch=arch), arch=arch
    )
    if arch == "gfx950":
        expected += _lower_kernel_to_llvm_python(
            module.build_unified_attention_reduce_tiled(reduce, arch=arch), arch=arch
        )
    assert (
        getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(asdict(segment), arch)
        == expected
    )


@pytest.mark.parametrize(
    "arch,page,tile",
    [("gfx942", 1, 16), ("gfx950", 1, 16), ("gfx942", 16, 64), ("gfx942", 32, 64)],
)
def test_loader_rejects_unimplemented_gather_geometry_in_both_engines(arch, page, tile):
    engine = pytest.importorskip("rocke_engine")
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    args = dict(
        head_size=64,
        block_size=page,
        tile_size_override=tile,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=8,
        num_seqs=3,
    )
    with pytest.raises(ValueError, match="paged gather"):
        module.UnifiedAttention3DTiledSpec(**args)
    with pytest.raises((ValueError, RuntimeError), match="paged gather"):
        getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(args, arch)


@pytest.mark.parametrize(
    "arch,transport",
    [("gfx942", "fp8"), ("gfx942", "wide"), ("gfx950", "fp8"), ("gfx950", "i64")],
)
@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_descriptor_transport_native_parity(arch, transport, flavor, monkeypatch):
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    segment = module.UnifiedAttention3DTiledSpec(
        head_size=128,
        block_size=32,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="bf16",
        use_sinks=True,
        sliding_window=33,
        has_softcap=True,
        use_alibi=True,
        use_qq_bias=True,
        num_segments=8,
        num_seqs=3,
        kv_storage_dtype="fp8e4m3" if transport == "fp8" else None,
        use_wide_kv_load=transport == "wide",
        use_i64_kv_addr=transport == "i64",
    )
    assert not segment.uses_paged_gather
    expected = _lower_kernel_to_llvm_python(
        module.build_unified_attention_3d_tiled(segment, arch=arch), arch=arch
    )
    if arch == "gfx950":
        reduce = module.UnifiedAttentionReduceTiledSpec(
            head_size=128,
            num_query_heads=8,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=8,
        )
        expected += _lower_kernel_to_llvm_python(
            module.build_unified_attention_reduce_tiled(reduce, arch=arch), arch=arch
        )
    assert (
        getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(asdict(segment), arch)
        == expected
    )


@pytest.mark.parametrize(
    "page,tile,transport",
    [(16, 32, "i64")]
    + [
        (page, tile, transport)
        for page, tile in [(16, 64), (32, 64), (16, 128), (64, 128)]
        for transport in ["plain", "i64", "swizzle_page_ids"]
    ],
)
@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_gfx950_whole_page_loader_native_parity(
    page, tile, transport, flavor, monkeypatch
):
    # P16/T32 keeps token gather for bounded caches, while i64 needs the
    # incoming whole-page loader. Larger tiles exercise multiple physical pages.
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    module = import_module("kernels.gfx950.attention_tiled_3d")
    segment = module.UnifiedAttention3DTiledSpec(
        head_size=64,
        block_size=page,
        tile_size_override=tile,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="bf16",
        num_seqs=3,
        num_segments=8,
        use_sinks=True,
        sliding_window=33,
        has_softcap=True,
        use_i64_kv_addr=transport == "i64",
        use_kv_lds_swizzle=transport == "swizzle_page_ids",
        use_wave_page_id_load=transport == "swizzle_page_ids",
    )
    assert not segment.uses_paged_gather
    reduce = module.UnifiedAttentionReduceTiledSpec(
        head_size=64, num_query_heads=8, num_kv_heads=2, dtype="bf16", num_segments=8
    )
    snapshot = SimpleNamespace(
        **asdict(segment), tile_size=tile, uses_paged_gather=False
    )
    problem = UnifiedAttentionProblem(
        total_q=3,
        num_seqs=3,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=64,
        block_size=page,
        max_seqlen_q=1,
        max_seqlen_k=97,
        dtype="bf16",
        clamp_arch="gfx950",
    )
    tuning = SimpleNamespace(
        arch="gfx950", path="3d", kernel_spec=snapshot, reduce_spec=reduce
    )
    assert validate_paged_loader_geometry(problem, tuning) == (page, tile, False)
    expected = _lower_kernel_to_llvm_python(
        module.build_unified_attention_3d_tiled(segment, arch="gfx950"), arch="gfx950"
    ) + _lower_kernel_to_llvm_python(
        module.build_unified_attention_reduce_tiled(reduce, arch="gfx950"),
        arch="gfx950",
    )
    assert (
        engine.gfx950_attention_tiled_3d_lower_llvm(asdict(segment), "gfx950")
        == expected
    )
