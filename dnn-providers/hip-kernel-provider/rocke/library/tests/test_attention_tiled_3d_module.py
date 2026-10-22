# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the gfx942 3D (split-KV) unified attention module.

Covers ``kernels.gfx942.attention_tiled_3d``'s spec-first support predicates and
module-level contracts; no GPU required.
"""

import pytest

from kernels.common.attention_arch import attention_lds_capacity_bytes
from kernels.gfx942 import attention_tiled_3d as mod
from kernels.gfx942.attention_tiled_3d import UnifiedAttentionReduceTiledSpec


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
