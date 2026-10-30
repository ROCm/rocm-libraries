# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke_engine attention bindings take ``asdict(spec)`` and must not drop
any field of it: a field the C struct carries reaches the builder, and a field
it has no slot for is rejected unless it holds its default."""

from __future__ import annotations

import re
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

# Python spec fields the tiled-2D spec-from-problem hooks do not return.
_UNPORTED_2D = {"use_additive_bias"}

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


@pytest.mark.parametrize(
    "field", ["use_v_hbm_direct", "use_k_hbm_direct", "use_causal_mask_phase_split"]
)
def test_gfx942_2d_binding_reads_gfx942_only_fields(engine, field):
    spec = Gfx942Spec2D(
        **_REQUIRED_2D,
        dtype="fp16",
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
        **{field: True},
    )
    expected = _lower_python(build_gfx942_2d(spec, arch="gfx942"), "gfx942")
    actual = engine.gfx942_attention_tiled_2d_lower_llvm(asdict(spec), "gfx942")
    assert actual == expected


def test_gfx950_2d_default_spec_round_trips(engine):
    spec = Gfx950Spec2D(**_REQUIRED_2D, dtype="bf16")
    expected = _lower_python(build_gfx950_2d(spec, arch="gfx950"), "gfx950")
    assert (
        engine.gfx950_attention_tiled_2d_lower_llvm(asdict(spec), "gfx950") == expected
    )


def test_2d_binding_rejects_unported_fields(engine):
    d = asdict(Gfx950Spec2D(**_REQUIRED_2D, dtype="bf16"))
    d["use_additive_bias"] = True
    with pytest.raises(NotImplementedError, match="use_additive_bias"):
        engine.gfx950_attention_tiled_2d_lower_llvm(d, "gfx950")


@pytest.mark.parametrize(
    "band",
    [
        {"causal_top_left": True},
        {"right_bound": 5, "sliding_window": 64},
        {"causal_top_left": True, "right_bound": -1},
    ],
)
def test_gfx950_2d_binding_reads_mask_band(engine, band):
    spec = Gfx950Spec2D(**dict(_REQUIRED_2D, dtype="bf16", **band))
    expected = _lower_python(build_gfx950_2d(spec, arch="gfx950"), "gfx950")
    assert (
        engine.gfx950_attention_tiled_2d_lower_llvm(asdict(spec), "gfx950") == expected
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"right_bound": -2},
        {"right_bound": 3, "use_alibi": True},
        {"right_bound": -1, "use_qq_bias": True},
    ],
)
def test_gfx950_2d_mask_band_rejects_match_python(engine, changes):
    d = dict(_REQUIRED_2D, dtype="bf16", **changes)
    with pytest.raises(ValueError) as py_err:
        Gfx950Spec2D(**d)
    full = asdict(Gfx950Spec2D(**_REQUIRED_2D, dtype="bf16"))
    full.update(d)
    with pytest.raises((ValueError, RuntimeError), match=re.escape(str(py_err.value))):
        engine.gfx950_attention_tiled_2d_lower_llvm(full, "gfx950")


@pytest.mark.parametrize(
    "band, label",
    [
        ({"causal_top_left": True}, "top-left/causal"),
        ({"right_bound": -1}, "bottom-right/unbounded"),
        ({"right_bound": 4}, "bottom-right/right_bound=4"),
    ],
)
def test_gfx942_2d_binding_rejects_mask_band(engine, band, label):
    # The gfx942 Python spec has no band fields; only a hand-built dict reaches
    # the C++ validator with one.
    d = asdict(Gfx942Spec2D(**_REQUIRED_2D, dtype="fp16"))
    d.update(band)
    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape(f"gfx942 tiled attention does not implement {label} masks"),
    ):
        engine.gfx942_attention_tiled_2d_lower_llvm(d, "gfx942")


_TRANSPOSED_32X32 = {
    "use_mfma_32x32": True,
    "use_transposed_qk_32x32": True,
    "block_m_per_warp": 32,
    "tile_size": 64,
}


@pytest.mark.parametrize(
    "changes",
    [
        {"use_q_reread": True, "use_transposed_qk_32x32": False},
        {"use_q_reread": True, "kv_storage_dtype": "fp8e4m3"},
        {"kv_ring_depth": 4},
        {"kv_ring_depth": 3, "kv_storage_dtype": "fp8e4m3"},
        {"kv_ring_depth": 3, "use_grouped_kv2_softmax": True},
        {"kv_ring_depth": 3, "use_v_double_buffer": True},
        {"kv_ring_depth": 3, "use_early_v_schedule": True},
        {
            "kv_ring_depth": 3,
            "use_v_double_buffer": True,
            "use_staggered_iter_wait": True,
        },
        {"kv_ring_depth": 3, "use_transposed_qk_32x32": False},
        {"kv_ring_depth": 3, "use_k_single_buffer": True},
    ],
)
def test_gfx950_2d_ring_and_q_reread_rejects_match_python(engine, changes):
    d = dict(_REQUIRED_2D, dtype="bf16", **dict(_TRANSPOSED_32X32, **changes))
    with pytest.raises((ValueError, NotImplementedError)) as py_err:
        Gfx950Spec2D(**d)
    full = asdict(Gfx950Spec2D(**_REQUIRED_2D, dtype="bf16"))
    full.update(d)
    with pytest.raises(
        (ValueError, NotImplementedError, RuntimeError),
        match=re.escape(str(py_err.value)),
    ):
        engine.gfx950_attention_tiled_2d_lower_llvm(full, "gfx950")


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
    d["use_additive_bias"] = True
    with pytest.raises(NotImplementedError, match="use_additive_bias"):
        engine.gfx950_attention_tiled_3d_lower_llvm(d, "gfx950")


_REQUIRED_3D = {
    "head_size": 64,
    "block_size": 16,
    "num_query_heads": 8,
    "num_kv_heads": 8,
    "dtype": "fp16",
    "use_sinks": False,
    "sliding_window": 0,
    "has_softcap": False,
    "num_segments": 8,
}


@pytest.mark.parametrize(
    "band",
    [
        {"causal_top_left": True},
        {"right_bound": 6, "sliding_window": 64},
        {"causal_top_left": True, "right_bound": -1},
    ],
)
def test_gfx950_3d_binding_reads_mask_band(engine, band):
    from kernels.gfx950.attention_tiled_3d import (
        UnifiedAttentionReduceTiledSpec,
        build_unified_attention_3d_tiled,
        build_unified_attention_reduce_tiled,
    )

    spec = UnifiedAttention3DTiledSpec(**dict(_REQUIRED_3D, **band))
    reduce_spec = UnifiedAttentionReduceTiledSpec(
        head_size=spec.head_size,
        num_query_heads=spec.num_query_heads,
        num_kv_heads=spec.num_kv_heads,
        dtype=spec.dtype,
        num_segments=spec.num_segments,
    )
    expected = _lower_python(
        build_unified_attention_3d_tiled(spec, arch="gfx950"), "gfx950"
    ) + _lower_python(
        build_unified_attention_reduce_tiled(reduce_spec, arch="gfx950"), "gfx950"
    )
    assert (
        engine.gfx950_attention_tiled_3d_lower_llvm(asdict(spec), "gfx950") == expected
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"right_bound": -2},
        {"right_bound": 3, "use_alibi": True},
        {"right_bound": -1, "use_qq_bias": True},
    ],
)
def test_gfx950_3d_mask_band_rejects_match_python(engine, changes):
    d = dict(_REQUIRED_3D, **changes)
    with pytest.raises(ValueError) as py_err:
        UnifiedAttention3DTiledSpec(**d)
    full = asdict(UnifiedAttention3DTiledSpec(**_REQUIRED_3D))
    full.update(d)
    with pytest.raises((ValueError, RuntimeError), match=re.escape(str(py_err.value))):
        engine.gfx950_attention_tiled_3d_lower_llvm(full, "gfx950")


@pytest.mark.parametrize(
    "band, label",
    [
        ({"causal_top_left": True}, "top-left/causal"),
        ({"right_bound": -1}, "bottom-right/unbounded"),
        ({"right_bound": 4}, "bottom-right/right_bound=4"),
    ],
)
def test_gfx942_3d_binding_rejects_mask_band(engine, band, label):
    d = dict(_REQUIRED_3D, **band)
    with pytest.raises(
        (ValueError, RuntimeError),
        match=re.escape(f"gfx942 tiled attention does not implement {label} masks"),
    ):
        engine.gfx942_attention_tiled_3d_lower_llvm(d, "gfx942")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "changes, message",
    [
        ({"kv_layout": "dense"}, "kv_layout must be 'paged' or 'strided'"),
        (
            {"kv_layout": "strided", "kv_storage_dtype": "fp8e4m3"},
            "strided KV requires",
        ),
        ({"kv_layout": "strided", "use_wave_page_id_load": True}, "needs paged"),
    ],
)
def test_3d_binding_validates_kv_layout(engine, arch, changes, message):
    d = dict(
        head_size=64,
        block_size=16,
        num_query_heads=8,
        num_kv_heads=8,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=8,
        **changes,
    )
    with pytest.raises((ValueError, RuntimeError), match=message):
        getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(d, arch)


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


def test_scalar_binding_reads_mask_band(engine):
    import kernels.common.attention_unified as au

    problem = au.UnifiedAttentionProblem(
        total_q=16,
        num_seqs=1,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        block_size=16,
        max_seqlen_q=16,
        max_seqlen_k=64,
        dtype="fp16",
        causal_top_left=True,
        right_bound=5,
    )
    kernel = au.build_unified_attention_2d(au.UnifiedAttention2DSpec(problem))
    expected = _lower_python(kernel, "gfx950")
    assert "tl_rb5" in expected
    assert engine.attention_unified_lower_llvm(asdict(problem), "gfx950") == expected


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
    assert set(spec) == python_fields - {"use_additive_bias"}
    assert spec["kv_layout"] == "paged"


def test_selector_hook_rejects_unported_arch_and_fields(engine):
    with pytest.raises(NotImplementedError, match="gfx950"):
        engine.gfx950_attn_tiled_spec_from_problem(_PROBLEM, "gfx942")
    with pytest.raises(NotImplementedError, match="target_ctas"):
        engine.gfx950_attn_tiled_spec_from_problem(
            dict(_PROBLEM, target_ctas=256), "gfx950"
        )
