# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the register mapper.

Structural checks (num_lanes/num_vector_items/num_repeat) plus the direct numeric oracle:
the C-tile forward map must reproduce rocke's ``MfmaAtom.lane_to_output`` for every
(lane, register) -- the builder-free counterpart to the encoding-field oracle.
"""

from __future__ import annotations

import pytest

from rocke.helpers.tiling.mma.warp_encoding import a_warp_encoding, c_warp_encoding
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.helpers.tiling.traits import load_mma_traits


def _c_mapper(op_id: str = "mfma_f32_16x16x16f16") -> RegisterMapper:
    return RegisterMapper(c_warp_encoding(load_mma_traits().get(op_id)))


def test_c_mapper_counts_match_wave_and_fragment() -> None:
    mapper = _c_mapper()
    assert mapper.num_lanes == 64  # wave64
    assert mapper.num_vector_items == 4  # c_per_lane for 16x16x16 f16
    assert mapper.num_repeat == 1
    assert mapper.matrix_major_size == 16  # rows (M)
    assert mapper.matrix_minor_size == 16  # cols (N)


def test_c_mapper_known_coordinate() -> None:
    mapper = _c_mapper()
    # Derivation: row = (lane // 16) * 4 + register, col = lane % 16.
    assert mapper.matrix_coordinates(lane=0, register=2) == (2, 0)
    assert mapper.matrix_coordinates(lane=17, register=1) == (5, 1)


def test_c_mapper_covers_every_element_once() -> None:
    mapper = _c_mapper()
    inverse = mapper.inverse_map()
    assert len(inverse) == mapper.matrix_major_size * mapper.matrix_minor_size


def test_a_mapper_counts() -> None:
    mapper = RegisterMapper(a_warp_encoding(load_mma_traits().get("mfma_f32_16x16x16f16")))
    assert mapper.num_lanes == 64
    assert mapper.num_vector_items == 4  # a_per_lane


def test_out_of_range_lane_fails_fast() -> None:
    mapper = _c_mapper()
    with pytest.raises(ValueError) as excinfo:
        mapper.matrix_coordinates(lane=64, register=0)
    assert "lane out of range" in str(excinfo.value)


# --- direct numeric oracle vs rocke MfmaAtom.lane_to_output ---
rocke_atoms = pytest.importorskip(
    "rocke.helpers.atoms",
    reason="rocke substrate not importable here; runs on the gfx90a host",
)


class _IntEvalBuilder:
    """Minimal int-evaluating stand-in for rocke's IRBuilder.

    ``MfmaAtom.lane_to_output(b, lane, i)`` emits IR; it only uses const_i32/mod/div/mul/
    add. Feeding it this stub runs the *real* rocke arithmetic on Python ints, giving a
    rigorous (non-transcribed) oracle for the forward map.
    """

    @staticmethod
    def const_i32(value: int) -> int:
        return int(value)

    @staticmethod
    def mod(a: int, c: int) -> int:
        return int(a) % int(c)

    @staticmethod
    def div(a: int, c: int) -> int:
        return int(a) // int(c)

    @staticmethod
    def mul(a: int, c: int) -> int:
        return int(a) * int(c)

    @staticmethod
    def add(a: int, c: int) -> int:
        return int(a) + int(c)


@pytest.mark.parametrize(
    "factory_name, op_id",
    [("f16_16x16x16", "mfma_f32_16x16x16f16"), ("f16_32x32x8", "mfma_f32_32x32x8f16")],
)
def test_c_forward_map_reproduces_lane_to_output(factory_name: str, op_id: str) -> None:
    factory = getattr(rocke_atoms.MfmaAtom, factory_name, None)
    if factory is None:
        pytest.skip(f"rocke MfmaAtom has no {factory_name} factory in this build")
    atom = factory()
    builder = _IntEvalBuilder()
    mapper = _c_mapper(op_id)
    assert mapper.num_lanes == 64
    for lane in range(mapper.num_lanes):
        for register in range(mapper.num_vector_items):
            ours = mapper.matrix_coordinates(lane, register)
            expected = tuple(atom.lane_to_output(builder, lane, register))
            assert ours == expected, (
                f"{op_id}: lane={lane} reg={register} ours={ours} oracle={expected}"
            )
