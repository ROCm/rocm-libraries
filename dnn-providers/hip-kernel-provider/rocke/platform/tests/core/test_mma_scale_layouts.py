# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent lane-coordinate and scale-layout metadata checks."""

from collections import Counter
from dataclasses import replace
import operator

import pytest

from rocke.core.arch import ArchTarget, known_arches
from rocke.core.ir import IRBuilder


def _atom(dtype, block_k):
    return ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=("e8m0", "e8m0", block_k),
    )


def _evaluate_coordinate(value, *, lane):
    op = value.op
    if op.name == "arith.constant":
        return op.attrs["value"]
    if op.name == "gpu.thread_id":
        return lane
    operations = {
        "arith.add": operator.add,
        "arith.mul": operator.mul,
        "arith.div": operator.floordiv,
        "arith.mod": operator.mod,
    }
    assert op.name in operations, op.name
    return operations[op.name](
        *(_evaluate_coordinate(v, lane=lane) for v in op.operands)
    )


@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp4"])
@pytest.mark.parametrize("block_k", [16, 32])
@pytest.mark.parametrize("role", ["a_scale", "b_scale"])
def test_scale_coordinates(dtype, block_k, role):
    atom = _atom(dtype, block_k)
    layout = getattr(atom, f"{role}_layout")()
    count = 128 // block_k
    assert getattr(atom, f"{role}_frag_len") == count
    assert (layout.role, layout.wave_size, layout.frag_len) == (
        "scale_src0" if role == "a_scale" else "scale_src1",
        32,
        count,
    )
    b = IRBuilder("scale_coordinates")
    lane = b.thread_id_x()
    coords = [layout.coord(b, lane, slot) for slot in range(count)]
    seen = Counter()
    for l in range(32):
        for j, pair in enumerate(coords):
            actual = tuple(_evaluate_coordinate(v, lane=l) for v in pair)
            expected = (l % 16, j) if role == "a_scale" else (j, l % 16)
            assert actual == expected
            seen[actual] += 1
    assert len(seen) == 16 * count
    assert set(seen.values()) == {2}
    for slot in (-1, count):
        with pytest.raises(ValueError, match="fragment slot"):
            layout.coord(b, lane, slot)


def test_every_scaled_catalog_coordinates_match_independent_lane_oracle():
    atoms = [
        op
        for op in ArchTarget.from_gfx("gfx1250").mma.ops
        if op.family == "wmma_scaled"
    ]
    assert len(atoms) == 86
    for atom in atoms:
        for role in ("a_scale", "b_scale"):
            layout = getattr(atom, f"{role}_layout")()
            b = IRBuilder("all_scale_coordinates")
            lane = b.thread_id_x()
            count = 128 // atom.scale_block_k
            assert layout.frag_len == count
            for slot in range(count):
                coords = layout.coord(b, lane, slot)
                for l in range(32):
                    actual = tuple(_evaluate_coordinate(v, lane=l) for v in coords)
                    assert actual == (
                        (l % 16, slot) if role == "a_scale" else (slot, l % 16)
                    )


def test_scale_defaults_and_unavailable_maps():
    for gfx in known_arches():
        for atom in ArchTarget.from_gfx(gfx).mma.ops:
            if atom.a_scale_dtype is None:
                assert atom.a_scale_frag_len == atom.b_scale_frag_len == 0
                for accessor in (atom.a_scale_layout, atom.b_scale_layout):
                    with pytest.raises(NotImplementedError, match="no verified"):
                        accessor()
    atom = _atom("fp8", 32)
    unknown = replace(
        atom,
        srcs=tuple(
            replace(src, scale=replace(src.scale, layout=None)) if src.scale else src
            for src in atom.srcs
        ),
    )
    with pytest.raises(NotImplementedError, match="scale_src0"):
        unknown.a_scale_layout()
    with pytest.raises(NotImplementedError, match="scale_src1"):
        unknown.b_scale_layout()
    # Adding scale maps does not invent unverified matrix operand maps.
    for accessor in (atom.a_layout, atom.b_layout):
        with pytest.raises(NotImplementedError):
            accessor()
    assert (
        replace(
            atom,
            srcs=(
                replace(
                    atom.srcs[0],
                    scale=replace(
                        atom.srcs[0].scale,
                        layout=replace(atom.a_scale_layout(), fn=lambda *_: None),
                    ),
                ),
                *atom.srcs[1:],
            ),
        )
        == atom
    )


@pytest.mark.parametrize("role", ["a_scale", "b_scale"])
def test_scale_metadata_rejects_inconsistent_maps(role):
    atom = _atom("fp8", 32)
    layout = getattr(atom, f"{role}_layout")()
    index = 0 if role == "a_scale" else 1

    def update_scale(**kwargs):
        srcs = list(atom.srcs)
        srcs[index] = replace(srcs[index], scale=replace(srcs[index].scale, **kwargs))
        return replace(atom, srcs=tuple(srcs))

    for change in ({"role": "a"}, {"frag_len": 8}, {"wave_size": 64}):
        with pytest.raises(ValueError, match="layout does not match"):
            update_scale(layout=replace(layout, **change))
    for count in (-1, True, 1.5):
        with pytest.raises(ValueError, match="fragment length"):
            update_scale(frag_len=count)
