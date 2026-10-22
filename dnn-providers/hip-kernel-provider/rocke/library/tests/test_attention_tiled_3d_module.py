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
    ok, reason = mod.supports_unified_attention_reduce_tiled(
        _spec(head_size=head_size, dtype=dtype), arch="gfx942"
    )
    assert ok, reason


@pytest.mark.parametrize(
    "kwargs, field",
    [
        (dict(head_size=96), "head_size"),
        (dict(head_size=0), "head_size"),
        (dict(dtype="fp32"), "dtype"),
        (dict(num_segments=1 << 20), "num_segments"),
    ],
)
def test_rejects_unsupported_field(kwargs, field):
    ok, reason = mod.supports_unified_attention_reduce_tiled(_spec(**kwargs))
    assert not ok
    assert field in reason


def test_rejects_other_arch():
    ok, reason = mod.supports_unified_attention_reduce_tiled(_spec(), arch="gfx90a")
    assert not ok
    assert reason


def test_lds_edge_uses_real_cap():
    cap = attention_lds_capacity_bytes("gfx942")
    ok, reason = mod.supports_unified_attention_reduce_tiled(
        _spec(num_segments=cap // 4)
    )
    assert ok, reason
    ok, reason = mod.supports_unified_attention_reduce_tiled(
        _spec(num_segments=cap // 4 + 1)
    )
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


def _strided_request(head_size, block_size, batch, dtype="fp16"):
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
        batch=batch,
        nhead_q=32,
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


def _runtime_specs(head_size, block_size, batch, dtype="fp16"):
    """``(segment, reduce)`` that rocKE's strided decode candidate selects."""
    from dispatch.attention.strided_decode import make_candidate

    selected = make_candidate().select_spec(
        _strided_request(head_size, block_size, batch, dtype)
    )
    return selected.kernel_spec, selected.reduce_spec


_RUNTIME_CELLS = [
    (d, b, batch) for d in (64, 128, 256) for b in (16, 32, 64) for batch in (1, 4, 64)
]


def test_reduce_spec_follows_segment():
    red = mod.reduce_spec_for_segment(
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
    assert mod.reduce_spec_for_segment(segment) == reduce


@pytest.mark.parametrize("head_size, block_size, batch", _RUNTIME_CELLS)
def test_module_predicate_accepts_runtime_selection(head_size, block_size, batch):
    segment, _ = _runtime_specs(head_size, block_size, batch)
    ok, reason = mod.supports_unified_attention_3d_module(segment, arch="gfx942")
    assert ok, reason


def test_module_has_segment_primary_and_reduce_companion():
    seg = _segment()
    module = mod.build_unified_attention_3d_module(seg, arch="gfx942")
    assert isinstance(module, KernelModule)
    assert module.primary.name == seg.kernel_name()
    assert {r: k.name for r, k in module.companions.items()} == {
        "reduce": mod.reduce_spec_for_segment(seg).kernel_name()
    }


def test_module_predicate_rejects_segment_lds_overflow():
    # D256 block64 without the runtime's tile override stages a 64-row K/V
    # tile: (16*256 + 3*64*256) * 2 B = 106,496 B > 65,536 B on gfx942.
    seg = _segment(head_size=256, block_size=64, tile_size_override=None)
    from kernels.common.attention_unified import _tiled_3d_segment_lds_bytes

    assert _tiled_3d_segment_lds_bytes(seg) > attention_lds_capacity_bytes("gfx942")
    ok, reason = mod.supports_unified_attention_3d_module(seg, arch="gfx942")
    assert not ok
    assert "LDS" in reason
    ok, reason = mod.supports_unified_attention_3d_module(
        replace(seg, tile_size_override=32), arch="gfx942"
    )
    assert ok, reason


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        (dict(head_size=96), "head_size"),
        (dict(block_size=128), "block_size"),
        (dict(num_query_heads=24, num_kv_heads=8), "num_queries_per_kv"),
        (dict(num_segments=(1 << 14) + 1), "num_segments"),
    ],
)
def test_module_predicate_rejects_unsupported(kwargs, needle):
    ok, reason = mod.supports_unified_attention_3d_module(_segment(**kwargs))
    assert not ok
    assert needle in reason


def test_module_predicate_rejects_other_arch():
    ok, reason = mod.supports_unified_attention_3d_module(_segment(), arch="gfx90a")
    assert not ok
    assert reason


def test_i64_kv_addr_spec_cannot_be_built():
    # The gfx942 segment spec refuses >2 GiB addressing at construction, so the
    # packer can never hand such a spec to the predicate or the builder.
    with pytest.raises(NotImplementedError, match="use_i64_kv_addr"):
        _segment(kv_layout="paged", use_i64_kv_addr=True)


def _lower(kernel):
    from rocke.helpers.compile import lower_kernel_for_comgr

    lowered = lower_kernel_for_comgr(kernel, arch="gfx942", backend="python")
    return lowered.llvm_text, lowered.options


@pytest.mark.parametrize("head_size, block_size", [(64, 16), (128, 32), (256, 64)])
def test_num_seqs_is_ir_neutral_on_the_decode_grid(head_size, block_size):
    """The catalog drops ``num_seqs`` (batch) from the kernel axes: on the decode
    grid the segment and reduce IR must not depend on it. Off the decode grid
    ``num_seqs`` sizes the sequence binary search, so the segment IR differs
    there -- the control that shows this comparison can see a change."""
    seg_1, _ = _runtime_specs(head_size, block_size, batch=1)
    seg_1 = replace(seg_1, num_segments=16)
    seg_64 = replace(seg_1, num_seqs=64)
    assert seg_1.use_decode_grid and seg_64.num_seqs == 64
    mod_1 = mod.build_unified_attention_3d_module(seg_1, arch="gfx942")
    mod_64 = mod.build_unified_attention_3d_module(seg_64, arch="gfx942")
    for k_1, k_64 in zip(mod_1.kernels, mod_64.kernels):
        assert k_1.name == k_64.name
        assert _lower(k_1) == _lower(k_64)
    off_grid = [
        mod.build_unified_attention_3d_tiled(
            replace(s, use_decode_grid=False), arch="gfx942"
        )
        for s in (seg_1, seg_64)
    ]
    assert _lower(off_grid[0]) != _lower(off_grid[1])


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
        mod.build_unified_attention_3d_module(seg, arch="gfx942"),
        arch="gfx942",
        backend="python",
        capture_ir_text=False,
    )
    assert artifact.kernel_name == seg.kernel_name()
    assert artifact.companion_names == {"reduce": reduce.kernel_name()}
    assert len(artifact.hsaco) > 0
