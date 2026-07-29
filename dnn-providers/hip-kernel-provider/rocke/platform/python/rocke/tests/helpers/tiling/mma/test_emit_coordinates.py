# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline test: the IR-emitting calculate_x matches the pure-int register mapper.

Runs ``emit_tensor_coordinates`` with an int-evaluating IRBuilder stub (so the emitted
arithmetic executes on Python ints) and checks it equals ``RegisterMapper`` across the full
grid for A/B/C encodings. Since the mapper reproduces ``lane_to_output`` (oracle), this ties
the IR-emitting addressing to the hardware layout -- without a GPU.
"""

from __future__ import annotations

import pytest

from rocke.helpers.tiling.emit import emit_tensor_coordinates
from rocke.helpers.tiling.mma.warp_encoding import (
    a_warp_encoding,
    b_warp_encoding,
    c_warp_encoding,
)
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.helpers.tiling.traits import load_mma_traits


class _IntEvalBuilder:
    """IRBuilder stand-in that evaluates the emitted arithmetic on Python ints."""

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


@pytest.mark.parametrize("encoding_fn", [a_warp_encoding, b_warp_encoding, c_warp_encoding])
def test_emit_coordinates_match_register_mapper(encoding_fn) -> None:
    encoding = encoding_fn(load_mma_traits().get("mfma_f32_16x16x16f16"))
    mapper = RegisterMapper(encoding)
    builder = _IntEvalBuilder()
    for lane in range(mapper.num_lanes):
        for register in range(mapper.num_vector_items):
            emitted = emit_tensor_coordinates(builder, encoding, lane, register)
            assert emitted == mapper.matrix_coordinates(lane, register), (lane, register)
