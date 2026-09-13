# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Structural invariants for every printable MMA fragment layout."""

from __future__ import annotations

from collections import Counter

from rocke.core import evaluate_layout
from rocke.core.arch import ArchTarget, known_arches

_AUDITED_ARCHES = (
    "gfx11-generic",
    "gfx1151",
    "gfx1201",
    "gfx1250",
    "gfx90a",
    "gfx942",
    "gfx950",
)

_GFX1250_UNMAPPED_OPERANDS = {
    "wmma_gfx1250_f32_16x16x64_fp8_fp8",
    "wmma_gfx1250_f32_16x16x64_fp8_bf8",
    "wmma_gfx1250_f32_16x16x64_bf8_fp8",
    "wmma_gfx1250_f32_16x16x64_bf8_bf8",
}
_EXPECTED_UNMAPPED = {
    *(
        ("gfx1250", op_id, role)
        for op_id in _GFX1250_UNMAPPED_OPERANDS
        for role in ("a", "b")
    ),
    *(
        ("gfx950", op_id, role)
        for op_id in (
            "mfma_f32_16x16x128_fp4",
            "mfma_f32_16x16x96_fp6",
        )
        for role in ("a", "b", "acc")
    ),
}
_GFX11_OPS = {
    "wmma_f32_16x16x16_f16",
    "wmma_f32_16x16x16_bf16",
    "wmma_i32_16x16x16_iu8",
    "wmma_i32_16x16x16_iu4",
}
_EXPECTED_REPLICATED = {
    (arch, op_id, role)
    for arch in ("gfx11-generic", "gfx1151")
    for op_id in _GFX11_OPS
    for role in ("a", "b")
}
_EXPECTED_PACKED = {
    (arch, op_id, role)
    for arch in ("gfx11-generic", "gfx1151")
    for op_id in ("wmma_i32_16x16x16_iu8", "wmma_i32_16x16x16_iu4")
    for role in ("a", "b")
}


def _role_layout_and_shape(op, role):
    if role == "a":
        return op.a_layout(), (op.m, op.k)
    if role == "b":
        return op.b_layout(), (op.k, op.n)
    return op.acc_layout(), (op.m, op.n)


def _expanded_indices(layout, index):
    if layout.packing is None:
        return [tuple(index)]
    expanded = []
    for offset in range(layout.packing.elements_per_slot):
        logical = list(index)
        logical[layout.packing.axis] += offset
        expanded.append(tuple(logical))
    return expanded


def test_every_registered_layout_has_complete_declared_relation():
    missing = set()
    replicated = set()
    packed = set()
    for arch in _AUDITED_ARCHES:
        for op in ArchTarget.from_gfx(arch).mma.ops:
            for role in ("a", "b", "acc"):
                identity = (arch, op.op_id, role)
                try:
                    layout, shape = _role_layout_and_shape(op, role)
                except NotImplementedError:
                    missing.add(identity)
                    continue
                if layout.replication_factor != 1:
                    replicated.add(identity)
                if layout.packing is not None:
                    packed.add(identity)

                coordinates = evaluate_layout(layout)
                physical = {(entry["lane"], entry["slot"]) for entry in coordinates}
                assert len(coordinates) == layout.wave_size * layout.frag_len, identity
                assert physical == {
                    (lane, slot)
                    for lane in range(layout.wave_size)
                    for slot in range(layout.frag_len)
                }, identity

                source_counts = Counter()
                for entry in coordinates:
                    for index in _expanded_indices(layout, entry["index"]):
                        assert 0 <= index[0] < shape[0], (identity, index, shape)
                        assert 0 <= index[1] < shape[1], (identity, index, shape)
                        source_counts[index] += 1

                expected_indices = {
                    (axis0, axis1)
                    for axis0 in range(shape[0])
                    for axis1 in range(shape[1])
                }
                assert set(source_counts) == expected_indices, identity
                assert set(source_counts.values()) == {
                    layout.replication_factor
                }, identity

    assert missing == _EXPECTED_UNMAPPED
    assert replicated == _EXPECTED_REPLICATED
    assert packed == _EXPECTED_PACKED


def test_gfx11_operand_replication_and_packing_are_explicit():
    for arch in ("gfx11-generic", "gfx1151"):
        target = ArchTarget.from_gfx(arch)
        for op_id, packed_width in (
            ("wmma_f32_16x16x16_f16", 1),
            ("wmma_f32_16x16x16_bf16", 1),
            ("wmma_i32_16x16x16_iu8", 4),
            ("wmma_i32_16x16x16_iu4", 8),
        ):
            op = target.mma.by_op_id(op_id)
            assert op is not None
            for layout in (op.a_layout(), op.b_layout()):
                assert layout.replication_factor == 2
                actual_width = (
                    1 if layout.packing is None else layout.packing.elements_per_slot
                )
                assert actual_width == packed_width

        fp16 = target.mma.by_op_id("wmma_f32_16x16x16_f16")
        assert fp16 is not None
        coordinates = evaluate_layout(fp16.a_layout())
        sources = [
            (entry["lane"], entry["slot"])
            for entry in coordinates
            if entry["index"] == [0, 0]
        ]
        assert sources == [(0, 0), (16, 0)]
        assert fp16.acc_layout().replication_factor == 1


def test_gfx1101_has_no_distinct_catalog_entry():
    # gfx11-generic is the family-level target used for otherwise unnamed
    # gfx11xx/gfx115x devices; this is not hardware-specific gfx1101 evidence.
    assert "gfx1101" not in known_arches()
    assert "gfx11-generic" in known_arches()
