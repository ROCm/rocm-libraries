# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke_engine attention bindings take ``asdict(spec)`` and must not drop
any field of it: a field the C struct carries reaches the builder, and a field
it has no slot for is rejected unless it holds its default."""

from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx942Spec2D,
)
from kernels.gfx942.attention_tiled_2d import (
    build_unified_attention_2d_tiled as build_gfx942_2d,
)
from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx950Spec2D,
)
from kernels.gfx950.attention_tiled_2d import (
    build_unified_attention_2d_tiled as build_gfx950_2d,
)
from kernels.gfx950.attention_tiled_3d import UnifiedAttention3DTiledSpec

# Python spec fields the shared C tiled-2D struct has no slot for.
_UNPORTED_2D = {
    "use_v_hbm_direct",
    "use_k_hbm_direct",
    "use_causal_mask_phase_split",
    "use_additive_bias",
    "use_kq_lds_pad",
    "kq_lds_pad_halves",
    "use_mask_phase_split",
    "causal_top_left",
    "right_bound",
}

_REQUIRED_2D = {
    "head_size": 64,
    "block_size": 32,
    "num_query_heads": 32,
    "num_kv_heads": 32,
    "use_sinks": False,
    "sliding_window": 0,
    "has_softcap": False,
}


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
    return pytest.importorskip("rocke_engine")


def _lower_python(kernel, arch):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(kernel, arch=arch)


def test_gfx942_2d_binding_keeps_the_x8_path(engine):
    spec = Gfx942Spec2D(
        **_REQUIRED_2D,
        dtype="fp16",
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
    )
    expected = _lower_python(build_gfx942_2d(spec, arch="gfx942"), "gfx942")
    assert "32x32x8" in expected
    actual = engine.gfx942_attention_tiled_2d_lower_llvm(asdict(spec), "gfx942")
    assert actual == expected


def test_gfx950_2d_default_spec_round_trips(engine):
    spec = Gfx950Spec2D(**_REQUIRED_2D, dtype="bf16")
    expected = _lower_python(build_gfx950_2d(spec, arch="gfx950"), "gfx950")
    assert (
        engine.gfx950_attention_tiled_2d_lower_llvm(asdict(spec), "gfx950") == expected
    )


@pytest.mark.parametrize(
    "field, value",
    [("use_additive_bias", True), ("kq_lds_pad_halves", 16), ("right_bound", -1)],
)
def test_2d_binding_rejects_unported_fields(engine, field, value):
    d = asdict(Gfx950Spec2D(**_REQUIRED_2D, dtype="bf16"))
    d[field] = value
    with pytest.raises(NotImplementedError, match=field):
        engine.gfx950_attention_tiled_2d_lower_llvm(d, "gfx950")


def test_3d_binding_rejects_unported_fields(engine):
    spec = UnifiedAttention3DTiledSpec(
        head_size=64,
        block_size=16,
        num_query_heads=8,
        num_kv_heads=8,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=8,
    )
    d = asdict(spec)
    d["right_bound"] = 5
    with pytest.raises(NotImplementedError, match="right_bound"):
        engine.gfx950_attention_tiled_3d_lower_llvm(d, "gfx950")


def test_scalar_binding_rejects_fp8_kv(engine):
    problem = {
        "head_size": 64,
        "block_size": 16,
        "num_query_heads": 4,
        "num_kv_heads": 2,
        "dtype": "fp16",
        "total_q": 16,
        "max_seqlen_q": 16,
        "max_seqlen_k": 64,
        "num_seqs": 1,
    }
    engine.attention_unified_lower_llvm(problem, "gfx950")
    with pytest.raises(NotImplementedError, match="use_fp8"):
        engine.attention_unified_lower_llvm(dict(problem, use_fp8=True), "gfx950")


_PROBLEM = {
    "total_q": 512,
    "num_seqs": 1,
    "num_query_heads": 32,
    "num_kv_heads": 32,
    "head_size": 64,
    "block_size": 32,
    "max_seqlen_q": 512,
    "max_seqlen_k": 4096,
    "dtype": "bf16",
}


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_selector_hook_returns_every_spec_field(engine, arch):
    hook = getattr(engine, f"{arch}_attn_tiled_spec_from_problem")
    spec = hook(_PROBLEM, arch)
    python_fields = {f.name for f in fields(Gfx942Spec2D)} | {
        f.name for f in fields(Gfx950Spec2D)
    }
    assert set(spec) == python_fields - _UNPORTED_2D


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_3d_selector_hook_returns_every_spec_field(engine, arch):
    from kernels.gfx942.attention_tiled_3d import (
        UnifiedAttention3DTiledSpec as Gfx942Spec3D,
    )

    hook = getattr(engine, f"{arch}_attn_tiled_3d_spec_from_problem")
    spec = hook(_PROBLEM, arch)
    python_fields = {f.name for f in fields(Gfx942Spec3D)} | {
        f.name for f in fields(UnifiedAttention3DTiledSpec)
    }
    # kv_layout selects the strided-KV builder; the problem always builds paged.
    unported = {"use_additive_bias", "causal_top_left", "right_bound", "kv_layout"}
    assert set(spec) == python_fields - unported


def test_selector_hook_rejects_unported_arch_and_fields(engine):
    with pytest.raises(NotImplementedError, match="gfx950"):
        engine.gfx950_attn_tiled_spec_from_problem(_PROBLEM, "gfx942")
    with pytest.raises(NotImplementedError, match="target_ctas"):
        engine.gfx950_attn_tiled_spec_from_problem(
            dict(_PROBLEM, target_ctas=256), "gfx950"
        )
