# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Zero-initialized FMHA harnesses scatter MMA results with the result map."""

from dataclasses import replace
from importlib import import_module

import pytest

from rocke.core.ir_serialize import serialize
from rocke.helpers.atoms import WmmaAtom


@pytest.mark.parametrize(
    "module,config,builder",
    [
        ("fmha_blockn", "BlockNCfg", "build_wmma_fmha_blockn"),
        ("fmha_pipelined", "PipelinedCfg", "build_wmma_fmha_pipelined"),
        ("fmha_regblocked", "RegBlockedCfg", "build_wmma_fmha_regblocked"),
        ("fmha_multiwave", "MultiWaveCfg", "build_wmma_fmha_multiwave"),
        ("fmha_singlewave", "SingleWaveCfg", "build_wmma_fmha_singlewave"),
    ],
)
def test_result_coordinates_do_not_follow_accumulator_input(
    monkeypatch, module, config, builder
):
    harness = import_module(f"builders.gfx1151.attention.{module}")
    spec = getattr(harness, config)(head_size=64, num_query_heads=1)
    build = getattr(harness, builder)
    expected = serialize(build(spec, arch="gfx1151"))
    resolve = WmmaAtom.mma_op
    selected = []

    def shifted(b, lane, slot):
        row, col = resolve(WmmaAtom.f16_16x16x16(), "gfx1151").dst.layout.fn(
            b, lane, slot
        )
        return b.add(row, b.const_i32(1)), col

    def source_map_must_not_be_read(*_args):
        raise AssertionError("result coordinates read the accumulator-input map")

    def changed_source(self, arch="gfx1151"):
        op = resolve(self, arch)
        selected.append(op.op_id)
        srcs = list(op.srcs)
        srcs[2] = replace(
            srcs[2], layout=replace(srcs[2].layout, fn=source_map_must_not_be_read)
        )
        return replace(op, srcs=tuple(srcs))

    monkeypatch.setattr(WmmaAtom, "mma_op", changed_source)
    assert serialize(build(spec, arch="gfx1151")) == expected
    assert selected and set(selected) == {"wmma_f32_16x16x16_f16"}

    def changed_result(self, arch="gfx1151"):
        op = resolve(self, arch)
        return replace(
            op, dst=replace(op.dst, layout=replace(op.dst.layout, fn=shifted))
        )

    monkeypatch.setattr(WmmaAtom, "mma_op", changed_result)
    # Negative control: the same map mutation must affect the result consumer.
    assert serialize(build(spec, arch="gfx1151")) != expected
