# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Accumulator descriptors match the selected instruction's actual C/D maps."""

from dataclasses import replace
from itertools import product

import pytest

from rocke.core.arch import ArchTarget
from rocke.core.arch.mfma_scale import MATRIX_FORMATS
from rocke.helpers.tiling import make_mma_accumulator_desc
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.tests.lane_maps import lane_placements


_CASES = []
for target, a, b in product(("gfx950", "gfx1250"), MATRIX_FORMATS, MATRIX_FORMATS):
    for block_k in (32,) if target == "gfx950" else (16, 32):
        for result in (False, True):
            _CASES.append(
                (target, a, b, block_k, 32 if target == "gfx950" else 16, result)
            )
for m, result in product((16, 32), (False, True)):
    _CASES.append(("gfx942", "tf32", "tf32", 0, m, result))


@pytest.mark.parametrize("target,a,b,block_k,m,result", _CASES)
def test_accumulator_descriptor_matches_every_registered_coordinate(
    target, a, b, block_k, m, result
):
    family = (
        "mma"
        if target == "gfx942"
        else "mfma_scaled" if target == "gfx950" else "wmma_scaled"
    )
    k = 128 // m if target == "gfx942" else 64 if target == "gfx950" else 128
    atom = ArchTarget.from_gfx(target).mma.op_for_shape(
        family=family,
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        m=m,
        n=m,
        k=k,
        scales=("e8m0", "e8m0", block_k) if block_k else None,
    )
    desc = make_mma_accumulator_desc(atom, target=target, result=result)
    mapper = RegisterMapper(desc.layout)
    metadata = atom.dst if result else atom.srcs[2]
    layout = atom.dst_layout() if result else atom.src_layout(2)
    assert desc.register_count == metadata.frag_len
    assert [
        mapper.matrix_coordinates(lane, slot)
        for lane in range(atom.wave_size)
        for slot in range(metadata.frag_len)
    ] == lane_placements(layout, atom.wave_size)


def test_accumulator_descriptor_refuses_non_atom():
    with pytest.raises(TypeError, match="accumulator descriptor requires an MmaOp"):
        make_mma_accumulator_desc(None, target="gfx950")


def test_gfx1250_profile_is_not_borrowed_for_other_wmma_targets():
    atom = ArchTarget.from_gfx("gfx1151").mma.op_for_shape(
        family="wmma", a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=16, n=16, k=16
    )
    with pytest.raises(NotImplementedError, match="unmodelled accumulator layout"):
        make_mma_accumulator_desc(atom, target="gfx1151")


def test_mutated_atom_metadata_is_refused():
    atom = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        family="mma", a_dtype="tf32", b_dtype="tf32", c_dtype="fp32", m=16, n=16, k=8
    )
    with pytest.raises(ValueError, match="registered atom metadata"):
        make_mma_accumulator_desc(replace(atom, m=17), target="gfx942")
