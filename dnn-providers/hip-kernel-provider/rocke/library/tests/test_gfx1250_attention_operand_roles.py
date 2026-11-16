# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU emission-role checks, not qualification of asymmetric WMMA pipelines."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rocke.core.arch import ArchTarget, MmaCatalog
from rocke.core.ir import F32, VectorType

from kernels.gfx1250.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from kernels.gfx1250.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    build_unified_attention_3d_tiled,
)


def _walk(region):
    for op in region.ops:
        yield op
        for child in op.regions:
            yield from _walk(child)


@pytest.mark.parametrize("path", ["2d", "2d_register_p", "3d", "3d_multi_wave"])
def test_tiled_initializers_distinguish_source_and_result(monkeypatch, path):
    target = ArchTarget.from_gfx("gfx1250")
    atom = target.mma.by_op_id("wmma_gfx1250_f32_16x16x32_bf16")
    source_width = atom.srcs[2].frag_len + 1
    custom = replace(
        atom, srcs=(*atom.srcs[:2], replace(atom.srcs[2], frag_len=source_width))
    )
    target = replace(target, mma=MmaCatalog([custom]))
    monkeypatch.setattr(ArchTarget, "from_gfx", staticmethod(lambda arch: target))
    common = dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=4,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=2,
        kv_storage_dtype="fp8e4m3",
    )
    if path.startswith("2d"):
        kernel = build_unified_attention_2d_tiled(
            UnifiedAttention2DTiledSpec(
                **common, tile_size=32, use_register_p=path == "2d_register_p"
            ),
            arch="gfx1250",
        )
    else:
        kernel = build_unified_attention_3d_tiled(
            UnifiedAttention3DTiledSpec(
                **common,
                num_segments=4,
                num_waves=2 if path == "3d_multi_wave" else 1,
                use_wide_lds_reads=path != "3d_multi_wave",
            ),
            arch="gfx1250",
        )
    ops = list(_walk(kernel.body))
    definitions = {value.name: op for op in ops for value in op.results}
    mmas = [op for op in ops if op.name == "tile.mma"]
    assert mmas and all(
        op.result.type == VectorType(F32, atom.dst.frag_len) for op in mmas
    )
    # QK starts each N-subtile with a fresh zero, then carries instruction results.
    qk_zeros = [
        definitions[op.operands[2].name]
        for op in mmas
        if definitions.get(op.operands[2].name) is not None
        and definitions[op.operands[2].name].name == "arith.constant_vec"
    ]
    assert len(qk_zeros) == 2
    assert all(op.result.type == VectorType(F32, source_width) for op in qk_zeros)
    # PV's initial zeros seed the loop, not the output-partial buffers.
    pv_zeros = [
        definitions[value.name]
        for op in ops
        if op.name == "scf.for"
        for value in op.operands[3:]
        if value.name in definitions
        and definitions[value.name].name == "arith.constant_vec"
        and isinstance(value.type, VectorType)
        and value.type.elem == F32
    ]
    assert len(pv_zeros) == common["head_size"] // 16
    assert all(op.result.type == VectorType(F32, source_width) for op in pv_zeros)
    source_zeros = {id(op) for op in qk_zeros + pv_zeros}
    result_zeros = [
        op
        for op in ops
        if op.name == "arith.constant_vec"
        and op.result.type.elem == F32
        and id(op) not in source_zeros
    ]
    # Neutral/combined output partials remain result-sized, including multi-wave.
    if path.startswith("3d"):
        assert result_zeros
    assert all(
        op.result.type == VectorType(F32, atom.dst.frag_len) for op in result_zeros
    )
