# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Instruction ABI padding is separate from logical fragment transport."""

from dataclasses import replace
from itertools import product
import os
from pathlib import Path
import subprocess

import pytest

from rocke.core.arch import ArchTarget
from rocke.core.arch.mfma_scale import MATRIX_FORMATS
from rocke.core.ir import F32, I32, IRBuilder, VectorType, dtype_to_ir_type
from rocke.core.ir_serialize import serialize
from rocke.helpers.mma_operand import prepare_scaled_mma_operand
from rocke.helpers.tiling import TileDesc, fill_fragment, make_fragment
from rocke.helpers.tiling.mma.warp_encoding import canonical_layouts
from rocke.helpers.tiling.traits import load_mma_traits


def _contract(target, a, b, role):
    shape = (32, 32, 64) if target == "gfx950" else (16, 16, 128)
    family = "mfma_scaled" if target == "gfx950" else "wmma_scaled"
    atom = ArchTarget.from_gfx(target).mma.op_for_shape(
        family=family,
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        scales=("e8m0", "e8m0", 32),
        m=shape[0],
        n=shape[1],
        k=shape[2],
    )
    traits = load_mma_traits().select(
        target=target,
        a=dtype_to_ir_type(a),
        b=dtype_to_ir_type(b),
        c=F32,
        m=shape[0],
        n=shape[1],
        k=shape[2],
        family="scaled",
        scale_block_k=32,
    )
    index = 0 if role == "a" else 1
    encoding = canonical_layouts(traits, (1, 1, 1))[index]
    desc = TileDesc((shape[index], shape[2]), encoding)
    return atom, desc, dtype_to_ir_type(a if index == 0 else b)


@pytest.mark.parametrize(
    "target,a,b,role",
    list(product(("gfx950", "gfx1250"), MATRIX_FORMATS, MATRIX_FORMATS, ("a", "b"))),
)
def test_scaled_operand_adaptation_preserves_live_fragment(target, a, b, role):
    atom, desc, dtype = _contract(target, a, b, role)
    builder = IRBuilder("scaled_operand")
    fragment = make_fragment(desc, dtype)
    fill_fragment(builder, fragment, 0)
    live = fragment.value
    before = len(builder.kernel.body.ops)
    adapted = prepare_scaled_mma_operand(builder, atom, role, fragment)
    assert fragment.value is live
    assert adapted.type == VectorType(I32, 8 if target == "gfx950" else 16)
    assert all(
        op.name in ("arith.bitcast", "arith.constant", "vector.splat", "vector.concat")
        for op in builder.kernel.body.ops[before:]
    )
    if dtype.name in ("fp6e2m3", "fp6e3m2", "fp4e2m1"):
        assert adapted.op.name == "vector.concat"
        assert adapted.op.operands[1].op.name == "vector.splat"
        assert adapted.op.operands[1].op.operands[0].op.attrs["value"] == 0


@pytest.mark.parametrize(
    "target,a,b,role",
    list(product(("gfx950", "gfx1250"), MATRIX_FORMATS, MATRIX_FORMATS, ("a", "b"))),
)
def test_native_scaled_operand_preparation_emits_identical_ir(target, a, b, role):
    executable = os.environ.get("ROCKE_FRAGMENT_IO_TEST")
    if not executable:
        pytest.skip("set ROCKE_FRAGMENT_IO_TEST to the built native fragment emitter")
    assert Path(executable).is_file()
    atom, desc, dtype = _contract(target, a, b, role)
    builder = IRBuilder("scaled_operand")
    fragment = make_fragment(desc, dtype)
    fill_fragment(builder, fragment, 0)
    prepare_scaled_mma_operand(builder, atom, role, fragment)
    result = subprocess.run(
        [executable, "--prepare", target, a, b, role],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout == serialize(builder.kernel)


def test_non_fragment_operand_is_refused_before_emission():
    atom, _, _ = _contract("gfx950", "fp4e2m1", "fp4e2m1", "a")
    builder = IRBuilder("invalid_fragment")
    before = serialize(builder.kernel)
    with pytest.raises(TypeError, match="requires a Fragment"):
        prepare_scaled_mma_operand(builder, atom, "a", None)
    assert serialize(builder.kernel) == before


def test_missing_atom_is_refused_before_emission():
    _, desc, dtype = _contract("gfx950", "fp4e2m1", "fp4e2m1", "a")
    builder = IRBuilder("missing_atom")
    fragment = make_fragment(desc, dtype)
    before = serialize(builder.kernel)
    with pytest.raises(TypeError, match="requires an MmaOp"):
        prepare_scaled_mma_operand(builder, None, "a", fragment)
    assert serialize(builder.kernel) == before


@pytest.mark.parametrize("role", ("c", "", None))
def test_invalid_operand_role_is_refused_before_emission(role):
    atom, desc, dtype = _contract("gfx950", "fp4e2m1", "fp4e2m1", "a")
    builder = IRBuilder("invalid_role")
    fragment = make_fragment(desc, dtype)
    before = serialize(builder.kernel)
    with pytest.raises(ValueError, match="expected 'a' or 'b'"):
        prepare_scaled_mma_operand(builder, atom, role, fragment)
    assert serialize(builder.kernel) == before


@pytest.mark.parametrize("target", ("gfx950", "gfx1250"))
@pytest.mark.parametrize("conflict", ("a", "b"))
@pytest.mark.parametrize("role", ("a", "b"))
def test_conflicting_atom_format_is_refused_before_emission(target, conflict, role):
    fp8, bf8 = "fp8e4m3", "bf8e5m2"
    a, b = (bf8, fp8) if conflict == "a" else (fp8, bf8)
    mixed, desc, dtype = _contract(target, a, b, role)
    fp8_atom, _, _ = _contract(target, fp8, fp8, role)
    stale = replace(mixed, op_id=fp8_atom.op_id)
    builder = IRBuilder("conflicting_atom_format")
    fragment = make_fragment(desc, dtype)
    fill_fragment(builder, fragment, 0)
    before = serialize(builder.kernel)
    with pytest.raises(
        ValueError,
        match=f"atom operand {conflict} dtype='bf8e5m2' conflicts with catalog dtype='fp8e4m3'",
    ):
        prepare_scaled_mma_operand(builder, stale, role, fragment)
    assert serialize(builder.kernel) == before


def test_operand_format_mismatch_is_refused_before_emission():
    atom, desc, _ = _contract("gfx950", "fp4e2m1", "fp4e2m1", "a")
    builder = IRBuilder("invalid_format")
    fragment = make_fragment(desc, dtype_to_ir_type("fp6e2m3"))
    before = serialize(builder.kernel)
    with pytest.raises(ValueError, match="expected 'fp4e2m1'"):
        prepare_scaled_mma_operand(builder, atom, "a", fragment)
    assert serialize(builder.kernel) == before
