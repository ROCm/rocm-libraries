# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the gfx942 3D (split-KV) unified attention module.

Covers ``kernels.gfx942.attention_tiled_3d``'s spec-first support predicates and
the segment+reduce ``KernelModule`` builder; no GPU required (the one compile
test needs comgr and skips without it).
"""

from dataclasses import replace

import pytest

from kernels.common.attention_arch import attention_lds_capacity_bytes
from kernels.gfx942 import attention_tiled_3d as mod
from kernels.gfx942.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
)
from rocke.helpers import KernelModule


def _spec(head_size=128, hq=32, hkv=8, dtype="fp16", num_segments=16):
    return UnifiedAttentionReduceTiledSpec(
        head_size=head_size,
        num_query_heads=hq,
        num_kv_heads=hkv,
        dtype=dtype,
        num_segments=num_segments,
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_accepts_supported_cells(head_size, dtype):
    ok, reason = mod.supports_tiled_reduce_spec(
        _spec(head_size=head_size, dtype=dtype), arch="gfx942"
    )
    assert ok, reason


@pytest.mark.parametrize(
    "kwargs, field",
    [
        (dict(head_size=96), "head_size"),
        (dict(head_size=0), "head_size"),
        (dict(dtype="fp32"), "fp16/bf16"),
        (dict(num_segments=1 << 20), "num_segments"),
    ],
)
def test_rejects_unsupported_field(kwargs, field):
    ok, reason = mod.supports_tiled_reduce_spec(_spec(**kwargs))
    assert not ok
    assert field in reason


def test_rejects_other_arch():
    ok, reason = mod.supports_tiled_reduce_spec(_spec(), arch="gfx90a")
    assert not ok
    assert reason


def test_lds_edge_uses_real_cap():
    cap = attention_lds_capacity_bytes("gfx942")
    ok, reason = mod.supports_tiled_reduce_spec(_spec(num_segments=cap // 4))
    assert ok, reason
    ok, reason = mod.supports_tiled_reduce_spec(_spec(num_segments=cap // 4 + 1))
    assert not ok
    assert "num_segments" in reason


# --- segment + reduce module -------------------------------------------------


def _segment(**kw):
    """A strided decode segment spec as the strided candidate builds it."""
    base = dict(
        head_size=128,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=16,
        num_seqs=1,
        kv_layout="strided",
        use_decode_grid=True,
        kv_cache_policy="stream",
    )
    base.update(kw)
    return UnifiedAttention3DTiledSpec(**base)


def _strided_request(head_size, block_size, batch, dtype="fp16", nhead_q=32):
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
        batch=batch,
        nhead_q=nhead_q,
        nhead_k=8,
        seqlen_q=1,
        seqlen_k=4096,
        hdim_q=head_size,
        hdim_v=head_size,
        arch="gfx942",
        dtype=dtype,
        kv_layout="strided",
        kv_block_size=block_size,
    )


def _runtime_specs(head_size, block_size, batch, dtype="fp16", nhead_q=32):
    """``(segment, reduce)`` that rocKE's strided decode candidate selects."""
    from dispatch.attention.strided_decode import make_candidate

    selected = make_candidate().select_spec(
        _strided_request(head_size, block_size, batch, dtype, nhead_q)
    )
    return selected.kernel_spec, selected.reduce_spec


_RUNTIME_CELLS = [
    (d, b, batch) for d in (64, 128, 256) for b in (16, 32, 64) for batch in (1, 4, 64)
]


def test_reduce_spec_follows_segment():
    red = mod.reduce_spec_for(
        _segment(head_size=256, num_segments=24, waves_per_eu=2, dtype="bf16")
    )
    assert red == UnifiedAttentionReduceTiledSpec(
        head_size=256,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=24,
        waves_per_eu=2,
    )


@pytest.mark.parametrize("head_size, block_size, batch", _RUNTIME_CELLS)
def test_reduce_spec_matches_runtime_selection(head_size, block_size, batch):
    segment, reduce = _runtime_specs(head_size, block_size, batch)
    assert mod.reduce_spec_for(segment) == reduce


@pytest.mark.parametrize("head_size, block_size, batch", _RUNTIME_CELLS)
def test_module_predicate_accepts_runtime_selection(head_size, block_size, batch):
    segment, _ = _runtime_specs(head_size, block_size, batch)
    ok, reason = mod.supports_unified_attention_3d_tiled_module(segment, arch="gfx942")
    assert ok, reason


def test_module_has_segment_primary_and_reduce_companion():
    seg = _segment()
    module = mod.build_unified_attention_3d_tiled_module(seg, arch="gfx942")
    assert isinstance(module, KernelModule)
    assert module.primary.name == seg.kernel_name()
    assert {r: k.name for r, k in module.companions.items()} == {
        "reduce": mod.reduce_spec_for(seg).kernel_name()
    }


def test_module_predicate_rejects_segment_lds_overflow():
    # D256 block64 without the runtime's tile override stages a 64-row K/V
    # tile: (16*256 + 3*64*256) * 2 B = 106,496 B > 65,536 B on gfx942.
    seg = _segment(head_size=256, block_size=64, tile_size_override=None)
    from kernels.common.attention_unified import _tiled_3d_segment_lds_bytes

    assert _tiled_3d_segment_lds_bytes(seg) > attention_lds_capacity_bytes("gfx942")
    ok, reason = mod.supports_unified_attention_3d_tiled_module(seg, arch="gfx942")
    assert not ok
    assert "LDS" in reason
    ok, reason = mod.supports_unified_attention_3d_tiled_module(
        replace(seg, tile_size_override=32), arch="gfx942"
    )
    assert ok, reason


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        (dict(head_size=96), "head_size"),
        (dict(block_size=128), "block_size"),
        # Ratios 1-16 build (a ratio that does not divide the 16-row tile masks its
        # padding rows); above 16 is refused.
        (dict(num_query_heads=32, num_kv_heads=1), "num_queries_per_kv"),
        (dict(num_segments=(1 << 14) + 1), "num_segments"),
    ],
)
def test_module_predicate_rejects_unsupported(kwargs, needle):
    ok, reason = mod.supports_unified_attention_3d_tiled_module(_segment(**kwargs))
    assert not ok
    assert needle in reason


def test_module_predicate_rejects_other_arch():
    ok, reason = mod.supports_unified_attention_3d_tiled_module(
        _segment(), arch="gfx90a"
    )
    assert not ok
    assert reason


@pytest.mark.parametrize("block_size, tile", [(32, 24), (32, 64), (64, 8), (16, 0)])
def test_module_predicate_refuses_a_tile_the_builder_cannot_walk(block_size, tile):
    # 24 and 64 do not divide 32 (the builder asserts BS % T == 0); 8 divides 64 but
    # is not a whole 16-wide MFMA step (T // MFMA_N would be 0); 0 is no tile.
    seg = _segment(block_size=block_size, tile_size_override=tile)
    ok, reason = mod.supports_unified_attention_3d_tiled_module(seg, arch="gfx942")
    assert not ok
    assert "KV tile" in reason


@pytest.mark.parametrize("block_size, tile", [(16, None), (32, 16), (64, 32), (64, 16)])
def test_module_predicate_accepts_only_tiles_the_builder_builds(block_size, tile):
    seg = _segment(block_size=block_size, tile_size_override=tile)
    ok, reason = mod.supports_unified_attention_3d_tiled_module(seg, arch="gfx942")
    assert ok, reason
    module = mod.build_unified_attention_3d_tiled_module(seg, arch="gfx942")
    assert module.primary.name == seg.kernel_name()


def test_i64_kv_addr_spec_cannot_be_built():
    # The gfx942 segment spec refuses >2 GiB addressing at construction, so the
    # packer can never hand such a spec to the predicate or the builder.
    with pytest.raises(NotImplementedError, match="use_i64_kv_addr"):
        _segment(kv_layout="paged", use_i64_kv_addr=True)


def _lower(kernel):
    from rocke.helpers.compile import lower_kernel_for_comgr

    lowered = lower_kernel_for_comgr(kernel, arch="gfx942", backend="python")
    return lowered.llvm_text, lowered.options


@pytest.mark.parametrize("gqa", [1, 4, 16])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_num_seqs_is_ir_neutral_on_the_decode_grid(head_size, block_size, dtype, gqa):
    """The catalog drops ``num_seqs`` (batch) from the kernel axes: the segment IR
    must not depend on it. Covers every catalog D x block x dtype, with the tile the
    runtime picks for (D, block), at GQA ratios 1/4/16 (8 KV heads). The reduce
    comparison is structural only -- the reduce spec has no ``num_seqs``. The
    control, a different ``num_segments``, shows the comparison can see a change
    (``num_seqs`` sizes nothing in the IR since the sequence search reads it as a
    kernarg, ``RUNTIME_SEARCH_ITERS``)."""
    seg_1, _ = _runtime_specs(head_size, block_size, 1, dtype, nhead_q=8 * gqa)
    expected_tile = block_size // 2 if head_size >= 128 and block_size >= 32 else None
    assert seg_1.tile_size_override == expected_tile
    assert seg_1.num_queries_per_kv == gqa and seg_1.dtype == dtype
    # Pin num_segments: the runtime picks it from batch, which this test varies.
    seg_1 = replace(seg_1, num_segments=16)
    seg_64 = replace(seg_1, num_seqs=64)
    assert seg_1.use_decode_grid and seg_64.num_seqs == 64
    mod_1 = mod.build_unified_attention_3d_tiled_module(seg_1, arch="gfx942")
    mod_64 = mod.build_unified_attention_3d_tiled_module(seg_64, arch="gfx942")
    for k_1, k_64 in zip(mod_1.kernels, mod_64.kernels):
        assert k_1.name == k_64.name
        assert _lower(k_1) == _lower(k_64)
    other = mod.build_unified_attention_3d_tiled(
        replace(seg_1, num_segments=32), arch="gfx942"
    )
    assert _lower(other) != _lower(mod_1.primary)


@pytest.mark.parametrize("gqa", [1, 4, 16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_head_counts_are_ir_neutral_at_a_fixed_gqa_ratio(head_size, gqa):
    """The catalog keeps one kernel per GQA ratio, not per head pair: the segment
    and reduce read ``num_query_heads`` / ``num_kv_heads`` as kernargs
    (``RUNTIME_KERNARG_FIELDS``), so head pairs sharing a ratio lower to one IR.
    The control, the next ratio up, shows the comparison can see a change."""
    seg, _ = _runtime_specs(head_size, 16, 1, "bf16", nhead_q=8 * gqa)
    seg = replace(seg, num_segments=16)
    pairs = [(gqa * h, h) for h in (1, 2, 8)]
    lowered = []
    for hq, hkv in pairs:
        module = mod.build_unified_attention_3d_tiled_module(
            replace(seg, num_query_heads=hq, num_kv_heads=hkv), arch="gfx942"
        )
        lowered.append([(k.name, _lower(k)) for k in module.kernels])
    assert all(x == lowered[0] for x in lowered[1:]), pairs
    other = mod.build_unified_attention_3d_tiled(
        replace(seg, num_query_heads=2 * gqa * 8, num_kv_heads=8), arch="gfx942"
    )
    assert _lower(other) != lowered[0][0][1]
    assert {
        replace(seg, num_query_heads=hq, num_kv_heads=hkv).resolved_num_queries_per_kv()
        for hq, hkv in pairs
    } == {gqa}


@pytest.mark.parametrize("tile", [None, 16])
def test_resolved_tile_size_override_is_the_tile_the_kernel_runs(tile):
    """A catalog records the knob through this accessor; ``None`` is one block."""
    spec = _segment(block_size=32, tile_size_override=tile)
    assert spec.resolved_tile_size_override() == spec.tile_size == (tile or 32)


def test_module_compiles_to_one_code_object():
    """Both kernels link into one gfx942 code object (comgr, no GPU). Only a
    missing comgr, or one without a gfx942 target, skips; a compile error fails."""
    try:
        from rocke.runtime.comgr import supported_isas

        isas = supported_isas()
    except Exception as exc:  # noqa: BLE001 - env-dependent
        pytest.skip(f"comgr toolchain unavailable: {exc}")
    if "amdgcn-amd-amdhsa--gfx942" not in isas:
        pytest.skip("the loaded comgr has no gfx942 target")
    from rocke.helpers import compile_module

    seg, reduce = _runtime_specs(128, 16, batch=4)
    artifact = compile_module(
        mod.build_unified_attention_3d_tiled_module(seg, arch="gfx942"),
        arch="gfx942",
        backend="python",
        capture_ir_text=False,
    )
    assert artifact.kernel_name == seg.kernel_name()
    assert artifact.companion_names == {"reduce": reduce.kernel_name()}
    assert len(artifact.hsaco) > 0
