# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 consumers retain native/legacy operand and scale-coordinate contracts."""

import ast
import importlib
import inspect
import re
from dataclasses import replace
from itertools import product

import pytest

from rocke.core.arch import ArchTarget, MmaCatalog
from rocke.core.ir import I32, VectorType
from rocke.core.verify import verify
from rocke.helpers.tiling import make_mma_scale_desc
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)
from rocke.tests.lane_maps import lane_placements

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


def _scale_atom(target, block):
    mfma = target == "gfx950"
    m = 32 if mfma else 16
    return ArchTarget.from_gfx(target).mma.op_for_shape(
        family="mfma_scaled" if mfma else "wmma_scaled",
        a_dtype="fp4",
        b_dtype="fp8",
        c_dtype="fp32",
        scales=("e8m0", "e8m0", block),
        m=m,
        n=m,
        k=64 if mfma else 128,
    )


@pytest.mark.parametrize(
    "target,block", [("gfx950", 32), ("gfx1250", 16), ("gfx1250", 32)]
)
def test_native_scale_distribution_matches_actual_atom_layouts(target, block):
    atom = _scale_atom(target, block)
    desc = make_mma_scale_desc(atom, target=target)
    count = atom.a_scale_frag_len
    assert desc.register_count == count
    mapper = RegisterMapper(desc.layout)
    actual = [
        mapper.matrix_coordinates(lane, index)
        for lane in range(atom.wave_size)
        for index in range(count)
    ]
    assert actual == lane_placements(atom.a_scale_layout(), atom.wave_size)
    assert actual == [
        (column, group)
        for group, column in lane_placements(atom.b_scale_layout(), atom.wave_size)
    ]


def test_scale_descriptor_refuses_non_atom():
    with pytest.raises(TypeError, match=r"^scale descriptor requires an MmaOp$"):
        make_mma_scale_desc(None, target="gfx950")


def test_scale_descriptor_refuses_unmodelled_target():
    target = ArchTarget.from_gfx("gfx942")
    atom = target.mma.op_for_shape(
        family="mma", a_dtype="tf32", b_dtype="tf32", c_dtype="fp32", m=16, n=16, k=8
    )
    assert atom is not None
    with pytest.raises(
        NotImplementedError, match=r"^unmodelled scale layout on 'gfx942'$"
    ):
        make_mma_scale_desc(atom, target=target.gfx)


@pytest.mark.parametrize("case", ("shape", "map", "missing-profile"))
def test_scale_descriptor_refuses_unmodelled_catalog_entry(monkeypatch, case):
    target = ArchTarget.from_gfx("gfx1250")
    atom = _scale_atom(target.gfx, 32)
    if case == "shape":
        atom = replace(atom, m=32)
        message = f"unmodelled scale shape for '{atom.op_id}'"
    elif case == "map":
        src = atom.srcs[1]
        # Leave slot count, wave and role intact but give B a different map.
        layout = replace(atom.b_scale_layout(), fn=atom.a_scale_layout().fn)
        src = replace(src, scale=replace(src.scale, layout=layout))
        atom = replace(atom, srcs=(atom.srcs[0], src, atom.srcs[2]))
        message = f"unmodelled scale_src1 map for '{atom.op_id}'"
    else:
        message = f"unmodelled scale_src0 map for '{atom.op_id}'"
    ops = [atom]
    if case != "missing-profile":
        ops.extend(op for op in target.mma.ops if op.op_id != atom.op_id)
    # The row itself is registered; these tests must reach the profile guards
    # rather than simply reject supplied metadata that differs from the catalog.
    catalog = MmaCatalog(ops)
    assert catalog.by_op_id(atom.op_id) is atom
    monkeypatch.setattr(ArchTarget, "from_gfx", lambda _: replace(target, mma=catalog))
    with pytest.raises(NotImplementedError, match=f"^{re.escape(message)}$"):
        make_mma_scale_desc(atom, target=target.gfx)


def test_scale_descriptor_refuses_conflicting_atom_metadata():
    atom = _scale_atom("gfx1250", 32)
    with pytest.raises(ValueError, match="registered atom metadata"):
        make_mma_scale_desc(replace(atom, m=32), target="gfx1250")


def test_legacy_consumer_refuses_a_missing_descriptor_atom(monkeypatch):
    instance = importlib.import_module("rocke.instances.gfx1250.block_scaled_gemm")
    spec = BlockScaledGemmSpec(
        "missing_atom",
        16,
        16,
        64,
        dtype_a="fp8",
        dtype_b="bf8",
        matrix_path="wmma",
        block_k=64,
        scale_dtype="fp32",
    )
    assert instance.is_valid_spec(spec, arch="gfx1250")[0]
    # Simulate a stale legacy op-ID mapping after shape/format admission.
    monkeypatch.setattr(instance, "_wmma_op_id", lambda *_: "missing_legacy_wmma_op")
    with pytest.raises(
        ValueError, match="missing matrix operand contract: missing_legacy_wmma_op"
    ):
        build_block_scaled_gemm(spec)


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
