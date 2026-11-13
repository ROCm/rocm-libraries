# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 unified-attention 2D-tiled kernels compile with the direct backend.

Each kernel starts with ``if (q_block >= this sequence's blocks) return``, a
``cf.return`` under a condition computed by a sequence-lookup loop. The direct
compiler must take it (no COMGR fallback, ``ROCKE_DIRECT_STRICT=1``). This
builder's loop has a constant trip count; the serialized IR of develop's builder
derives the count from ``num_seqs`` through ``arith.sitofp_f32``, which
``tests/runtime/test_direct_early_return.py`` covers.

CPU only (no kernel launch); skips without the direct library.
"""

from __future__ import annotations

import pytest

from kernels import UnifiedAttention2DTiledSpec, build_unified_attention_2d_tiled
from rocke.helpers.compile import compile_kernel
from rocke.runtime import code_object

_SPECS = [
    dict(
        head_size=128, block_size=16, num_query_heads=16, num_kv_heads=2, dtype="fp16"
    ),
    dict(
        head_size=128, block_size=16, num_query_heads=16, num_kv_heads=2, dtype="bf16"
    ),
    dict(head_size=64, block_size=16, num_query_heads=16, num_kv_heads=2, dtype="bf16"),
    dict(
        head_size=128,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=2,
        dtype="fp16",
        sliding_window=64,
    ),
]


def test_attention_builders_request_their_direct_levers() -> None:
    """``attrs["direct_levers"]`` (not IR) names the direct-compiler levers each gfx950 unified-attention
    kernel wants: the 2D kernel's K/V tile loads ahead of its QK MFMAs and its block-table selects together,
    the reduce kernel's wider scheduling window."""
    from kernels import (
        UnifiedAttentionReduceTiledSpec,
        build_unified_attention_reduce_tiled,
    )

    spec2d = UnifiedAttention2DTiledSpec(
        use_sinks=False,
        has_softcap=False,
        sliding_window=0,
        **_SPECS[0],
    )
    assert (
        build_unified_attention_2d_tiled(spec2d, arch="gfx950").attrs["direct_levers"]
        == "dmafree,smemsink"
    )
    reduce_spec = UnifiedAttentionReduceTiledSpec(
        head_size=128, num_query_heads=16, num_kv_heads=2, dtype="fp16", num_segments=64
    )
    assert (
        build_unified_attention_reduce_tiled(reduce_spec, arch="gfx950").attrs[
            "direct_levers"
        ]
        == "widewin"
    )


@pytest.mark.parametrize(
    "fields",
    _SPECS,
    ids=lambda f: f"{f['dtype']}_d{f['head_size']}_w{f.get('sliding_window', 0)}",
)
def test_unified_2d_tiled_compiles_direct_strict(fields, monkeypatch) -> None:
    if code_object._direct_library_path() is None:
        pytest.skip("build the Rust cdylib or set ROCKE_CODEOBJ_DIRECT_LIB")
    if "gfx950" not in code_object.direct_supported_targets():
        pytest.skip("the direct library predates gfx950")
    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    spec = UnifiedAttention2DTiledSpec(
        use_sinks=False, has_softcap=False, **{"sliding_window": 0, **fields}
    )
    art = compile_kernel(
        build_unified_attention_2d_tiled(spec, arch="gfx950"),
        arch="gfx950",
        code_object_backend="direct",
    )
    assert (art.code_object_backend, art.direct_fallback) == ("direct", "")
    assert art.hsaco_bytes > 0
