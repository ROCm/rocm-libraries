# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Accumulator initialization and result slots have separate metadata roles."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from rocke.core.arch import ArchTarget, MmaCatalog
from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType, VectorType
from rocke.helpers.wmma_attention import (
    _wmma_attention_fwd_inner_body,
    _wmma_attn_op_id,
)


def _walk(region):
    for op in region.ops:
        yield op
        for child in op.regions:
            yield from _walk(child)


@pytest.mark.parametrize(
    "arch", ["gfx1151", "gfx11-generic", "gfx1201", "gfx12-generic"]
)
@pytest.mark.parametrize("dtype,element", [("fp16", F16), ("bf16", BF16)])
def test_wmma_attention_zeros_use_source_width_without_changing_result(
    arch, dtype, element
):
    target = ArchTarget.from_gfx(arch)
    atom = target.mma.by_op_id(_wmma_attn_op_id(target, dtype))
    source_width = atom.srcs[2].frag_len + 1

    # This asymmetric fixture checks role selection, not instruction legality.
    def source_map_must_not_drive_result_scatter(*args):
        raise AssertionError("result scatter used the accumulator input map")

    custom = replace(
        atom,
        srcs=(
            *atom.srcs[:2],
            replace(
                atom.srcs[2],
                frag_len=source_width,
                layout=replace(
                    atom.srcs[2].layout,
                    frag_len=source_width,
                    fn=source_map_must_not_drive_result_scatter,
                ),
            ),
        ),
    )
    target = replace(target, mma=MmaCatalog([custom]))
    b = IRBuilder("wmma_source_result_roles")
    zero, stride = b.const_i32(0), b.const_i32(16)
    _wmma_attention_fwd_inner_body(
        b,
        Q=b.param("Q", PtrType(element, "global")),
        K=b.param("K", PtrType(element, "global")),
        V=b.param("V", PtrType(element, "global")),
        O=b.param("O", PtrType(element, "global")),
        head_size=16,
        seqlen_k=b.param("n", I32),
        q_tile_base=zero,
        head_idx=zero,
        kv_head_idx=zero,
        q_pos_base=None,
        stride_q_token=stride,
        stride_q_head=stride,
        stride_k_token=stride,
        stride_k_head=stride,
        stride_v_token=stride,
        stride_v_head=stride,
        stride_o_token=stride,
        stride_o_head=stride,
        scale_log2=b.param("scale", F32),
        dtype=dtype,
        mask_mode="none",
        sliding_window=0,
        causal_ctx_offset=None,
        k_token_offset_elems=None,
        v_token_offset_elems=None,
        k_row_base_fn=None,
        v_row_base_fn=None,
        k_tile_start=None,
        k_tile_stop=None,
        extra_score_transform=None,
        extra_mask_predicate=None,
        extra_skip_predicate=None,
        k_block_iter_fn=None,
        v_scale=None,
        arch=arch,
        target=target,
    )
    ops = list(_walk(b.kernel.body))
    zeros = [
        op
        for op in ops
        if op.name == "arith.constant_vec" and op.attrs["elem"] == "f32"
    ]
    assert len(zeros) == 2
    assert all(op.result.type == VectorType(F32, source_width) for op in zeros)
    mmas = [op for op in ops if op.name == "tile.mma"]
    assert len(mmas) == 2
    assert all(op.operands[2].type == VectorType(F32, source_width) for op in mmas)
    assert all(op.result.type == VectorType(F32, atom.dst.frag_len) for op in mmas)


def test_wmma_output_tensor_and_store_use_the_destination_map():
    from rocke.helpers.distribution import WmmaTensor, store_wmma_acc

    atom = ArchTarget.from_gfx("gfx1151").mma.by_op_id("wmma_f32_16x16x16_f16")

    def source_map(*args):
        raise AssertionError("output operation used the source2 layout")

    custom = replace(
        atom,
        srcs=(
            *atom.srcs[:2],
            replace(
                atom.srcs[2],
                frag_len=3,
                layout=replace(atom.srcs[2].layout, frag_len=3, fn=source_map),
            ),
        ),
    )
    wrapper = SimpleNamespace(
        c_per_lane=custom.dst.frag_len,
        mma_op=lambda arch: custom,
        c_layout=lambda arch: custom.c_layout(),
    )
    b = IRBuilder("output_layout_roles")
    lane = b.thread_id_x()
    acc = b.param("acc", VectorType(F32, custom.dst.frag_len))
    tensor = WmmaTensor(atom=wrapper, role="c", value=acc)
    assert tensor._layout() is custom.dst.layout
    assert custom.c_frag_len == 3
    stores = []
    window = SimpleNamespace(
        dtype=F32, store_scalar=lambda *args, **kwargs: stores.append((args, kwargs))
    )
    store_wmma_acc(b, window, wrapper, lane, acc)
    assert len(stores) == custom.dst.frag_len
