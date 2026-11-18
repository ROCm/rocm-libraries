# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 consumers retain native/legacy operand and scale-coordinate contracts."""

import ast
import inspect
from itertools import product

import pytest

from rocke.core.arch import ArchTarget
from rocke.core.ir import I32, VectorType
from rocke.core.verify import verify
from rocke.helpers.tiling import make_tile_desc
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)

_FORMATS = ("fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1")


@pytest.mark.parametrize(
    "a,b,block,output", list(product(_FORMATS, _FORMATS, (16, 32), ("bf16", "fp16")))
)
def test_native_scaled_consumer_verifies_with_instruction_sized_operands(
    a, b, block, output
):
    spec = BlockScaledGemmSpec(
        "native_io",
        32,
        48,
        256,
        dtype_a=a,
        dtype_b=b,
        dtype_c=output,
        matrix_path="wmma_scale16" if block == 16 else "wmma_scale",
        scale_dtype="e8m0",
        block_k=block,
    )
    kernel = build_block_scaled_gemm(spec)
    assert not verify(kernel)
    calls = [op for op in kernel.body.ops if op.name == "tile.mma"]
    assert len(calls) == 2
    for op in calls:
        assert op.operands[0].type == op.operands[1].type == VectorType(I32, 16)


@pytest.mark.parametrize(
    "a,b,scale,block",
    list(product(_FORMATS[:2], _FORMATS[:2], ("fp16", "fp32"), (64, 128))),
)
def test_legacy_consumer_keeps_contiguous_half_wave_payload(a, b, scale, block):
    spec = BlockScaledGemmSpec(
        "legacy_io",
        16,
        32,
        128,
        dtype_a=a,
        dtype_b=b,
        matrix_path="wmma",
        scale_dtype=scale,
        block_k=block,
    )
    kernel = build_block_scaled_gemm(spec)
    assert not verify(kernel)
    calls = [op for op in kernel.body.ops if op.name == "tile.mma"]
    assert len(calls) == 2
    assert all(
        op.operands[0].type == op.operands[1].type == VectorType(I32, 8) for op in calls
    )


class _Math:
    const_i32 = staticmethod(lambda value: value)
    add = staticmethod(lambda a, b: a + b)
    mul = staticmethod(lambda a, b: a * b)
    div = staticmethod(lambda a, b: a // b)
    mod = staticmethod(lambda a, b: a % b)


@pytest.mark.parametrize("block", (16, 32))
def test_native_scale_distribution_matches_actual_atom_layouts(block):
    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype="fp4",
        b_dtype="fp4",
        c_dtype="fp32",
        scales=("e8m0", "e8m0", block),
        m=16,
        n=16,
        k=128,
    )
    count = atom.a_scale_frag_len
    desc = make_tile_desc(
        shape=(16, count),
        thread_dist=(16, 1),
        thread_tile=(1, count),
        thread_broadcast=2,
        wave_size=32,
    )
    mapper = RegisterMapper(desc.layout)
    for lane in range(32):
        for index in range(count):
            row, group = atom.a_scale_layout().coord(_Math(), lane, index)
            b_group, column = atom.b_scale_layout().coord(_Math(), lane, index)
            assert (
                tuple(mapper.matrix_coordinates(lane, index))
                == (row, group)
                == (column, b_group)
            )


def test_consumer_does_not_bypass_fragment_loads_or_stores():
    tree = ast.parse(inspect.getsource(build_block_scaled_gemm))
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call, ast.Attribute)
        and call.attr in ("global_load", "global_load_vN", "global_store")
        for call in calls
    )
    assert not any(
        isinstance(call, ast.Name)
        and call.id in ("load_matrix_fragment", "pack_fragment_bits")
        for call in calls
    )
