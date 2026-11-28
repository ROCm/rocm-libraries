# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fixed paged metadata geometry must not alias runtime tensor-stride kernels."""

from dataclasses import replace
from importlib import import_module

import pytest

from kernels.common import attention_unified as au
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _problem(page):
    return au.UnifiedAttentionProblem(
        total_q=3,
        num_seqs=3,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=64,
        block_size=page,
        max_seqlen_q=1,
        max_seqlen_k=65,
        dtype="fp16",
        sliding_window=33,
        num_cus=32,
        target_ctas=128,
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page,bounded", [(1, True), (16, False)])
def test_default_metadata_identity_keeps_required_specialization(
    arch, page, bounded, monkeypatch
):
    # Hold policy outputs fixed so this tests metadata identity, not a change
    # in segmentation, residency, or occupancy selected for a larger shape.
    monkeypatch.setattr(au, "_num_segments", lambda *args: 8)
    monkeypatch.setattr(au, "_kv_cache_policy_3d", lambda *args: "all")
    monkeypatch.setattr(au, "_select_3d_waves_per_eu", lambda *args: None)
    problem = _problem(page)
    key = au._tiled_3d_cache_key(problem, arch)
    for other in (
        replace(problem, num_query_heads=16, num_kv_heads=4),
        replace(problem, sliding_window=63),
    ):
        assert (key != au._tiled_3d_cache_key(other, arch)) is bounded


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("bounded", [False, True])
def test_bounded_names_cover_fixed_heads_and_window(arch, bounded, monkeypatch):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm23")
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    spec = module.UnifiedAttention3DTiledSpec(
        head_size=64,
        block_size=32,
        tile_size_override=32,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="fp16",
        use_sinks=False,
        sliding_window=33,
        has_softcap=False,
        num_segments=8,
        num_seqs=3,
        use_decode_grid=True,
        metadata_containment=bounded,
    )
    text = _lower_kernel_to_llvm_python(
        module.build_unified_attention_3d_tiled(spec, arch=arch), arch=arch
    )
    for other in (
        replace(spec, num_query_heads=16, num_kv_heads=4),
        replace(spec, sliding_window=63),
    ):
        assert (spec.kernel_name() != other.kernel_name()) is bounded
        other_text = _lower_kernel_to_llvm_python(
            module.build_unified_attention_3d_tiled(other, arch=arch), arch=arch
        )
        assert (text != other_text) is bounded


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_raw_gather_names_cover_fixed_pool_heads(arch):
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    spec = module.UnifiedAttention3DTiledSpec(
        head_size=64,
        block_size=16 if arch == "gfx942" else 64,
        tile_size_override=32,
        num_query_heads=8,
        num_kv_heads=2,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=8,
    )
    assert spec.uses_paged_gather and not spec.metadata_containment
    other = replace(spec, num_query_heads=16, num_kv_heads=4)
    assert spec.kernel_name() != other.kernel_name()
