# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent lane-coordinate, tensor-address and packed-scale oracles."""

from collections import Counter
from dataclasses import replace
import operator

import pytest

from rocke.core.arch import ArchTarget, known_arches
from rocke.core.ir import IRBuilder
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)


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


def _evaluate(value, *, lane=0, tile=(0, 0), loads=None):
    """Evaluate only the integer SSA dependency graph, never the MMA itself."""
    op = value.op
    if op.name == "arith.constant":
        return op.attrs["value"]
    if op.name == "gpu.thread_id":
        return lane
    if op.name == "gpu.block_id":
        return tile[0 if op.attrs["axis"] == "x" else 1]

    def evaluate(v):
        return _evaluate(v, lane=lane, tile=tile, loads=loads)

    if op.name == "memref.global_load_typed":
        ptr, index = op.operands
        offset = evaluate(index)
        role = ptr.name.lstrip("%")
        assert role in ("A_scale", "B_scale")
        loads.append((role, offset))
        # Distinct address-sensitive bytes expose stride, group and order bugs.
        return (offset * 37 + (11 if role == "A_scale" else 73)) % 256
    if op.name == "arith.zext":
        return evaluate(op.operands[0]) & 255
    operations = {
        "arith.add": operator.add,
        "arith.mul": operator.mul,
        "arith.div": operator.floordiv,
        "arith.mod": operator.mod,
        "arith.shl": operator.lshift,
        "arith.or": operator.or_,
    }
    assert op.name in operations, op.name
    return operations[op.name](*(evaluate(v) for v in op.operands))


@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp4"])
@pytest.mark.parametrize("block_k", [16, 32])
@pytest.mark.parametrize("role", ["a_scale", "b_scale"])
def test_scale_coordinates(dtype, block_k, role):
    atom = _atom(dtype, block_k)
    layout = getattr(atom, f"{role}_layout")()
    count = 128 // block_k
    assert getattr(atom, f"{role}_frag_len") == count
    assert (layout.role, layout.wave_size, layout.frag_len) == (role, 32, count)
    b = IRBuilder("scale_coordinates")
    lane = b.thread_id_x()
    coords = [layout.coord(b, lane, slot) for slot in range(count)]
    seen = Counter()
    for l in range(32):
        for j, pair in enumerate(coords):
            actual = tuple(_evaluate(v, lane=l) for v in pair)
            expected = (l % 16, j) if role == "a_scale" else (j, l % 16)
            assert actual == expected
            seen[actual] += 1
    assert len(seen) == 16 * count
    assert set(seen.values()) == {2}
    for slot in (-1, count):
        with pytest.raises(ValueError, match="fragment slot"):
            layout.coord(b, lane, slot)


def test_scale_defaults_and_unavailable_maps():
    for gfx in known_arches():
        for atom in ArchTarget.from_gfx(gfx).mma.ops:
            if atom.a_scale_dtype is None:
                assert atom.a_scale_frag_len == atom.b_scale_frag_len == 0
                for accessor in (atom.a_scale_layout, atom.b_scale_layout):
                    with pytest.raises(NotImplementedError, match="no verified"):
                        accessor()
    atom = _atom("fp8", 32)
    unknown = replace(atom, _a_scale_layout=None, _b_scale_layout=None)
    with pytest.raises(NotImplementedError, match="a_scale"):
        unknown.a_scale_layout()
    with pytest.raises(NotImplementedError, match="b_scale"):
        unknown.b_scale_layout()
    # Adding scale maps does not invent unverified matrix operand maps.
    for accessor in (atom.a_layout, atom.b_layout):
        with pytest.raises(NotImplementedError):
            accessor()
    assert (
        replace(
            atom, _a_scale_layout=replace(atom.a_scale_layout(), fn=lambda *_: None)
        )
        == atom
    )


@pytest.mark.parametrize("role", ["a_scale", "b_scale"])
def test_scale_metadata_rejects_inconsistent_maps(role):
    atom = _atom("fp8", 32)
    layout = getattr(atom, f"{role}_layout")()
    for change in ({"role": "a"}, {"frag_len": 8}, {"wave_size": 64}):
        with pytest.raises(ValueError, match="layout does not match"):
            replace(atom, **{f"_{role}_layout": replace(layout, **change)})
    for count in (-1, True, 1.5):
        with pytest.raises(ValueError, match="fragment length"):
            replace(atom, **{f"{role}_frag_len": count})


@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp4"])
@pytest.mark.parametrize("block_k", [16, 32])
def test_instance_scale_addresses_and_packed_words(dtype, block_k):
    spec = BlockScaledGemmSpec(
        "scale_addresses",
        M=32,
        N=48,
        K=256,
        dtype_a=dtype,
        dtype_b=dtype,
        scale_dtype="e8m0",
        block_k=block_k,
        matrix_path="wmma_scale16" if block_k == 16 else "wmma_scale",
    )
    kernel = build_block_scaled_gemm(spec, arch="gfx1250")
    mmas = [op for op in kernel.body.ops if op.name == "tile.mma"]
    assert len(mmas) == 2
    count = 128 // block_k
    for x, y in ((0, 0), (2, 1)):
        for lane in range(32):
            for step, mma in enumerate(mmas):
                for source, value in enumerate(mma.operands[3:]):
                    offsets = [
                        (
                            (16 * y + lane % 16) * (256 // block_k) + step * count + j
                            if source == 0
                            else (step * count + j) * 48 + 16 * x + lane % 16
                        )
                        for j in range(count)
                    ]
                    role = "A_scale" if source == 0 else "B_scale"
                    expected = int.from_bytes(
                        bytes(
                            (offset * 37 + (11 if source == 0 else 73)) % 256
                            for offset in offsets
                        ),
                        "little",
                    )
                    loads = []
                    assert (
                        _evaluate(value, lane=lane, tile=(x, y), loads=loads)
                        == expected
                    )
                    assert loads == [(role, offset) for offset in offsets]
