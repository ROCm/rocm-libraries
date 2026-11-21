# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the i32 kernarg wrap boundary in ``runtime.packing``."""

from __future__ import annotations

import struct

import pytest

from rocke.runtime.packing import compile_packer, pack_args

_SIG = [{"name": "x", "type": "i32"}]


def test_a_value_at_the_top_of_the_signed_range_packs_unchanged():
    assert pack_args(_SIG, {"x": 2**31 - 1}) == struct.pack("<i", 2**31 - 1)


def test_a_value_at_the_unsigned_wrap_boundary_packs_as_the_most_negative_i32():
    assert pack_args(_SIG, {"x": 1 << 31}) == struct.pack("<i", -(2**31))


def test_a_value_at_the_top_of_the_unsigned_range_packs_as_minus_one():
    assert pack_args(_SIG, {"x": 2**32 - 1}) == struct.pack("<i", -1)


def test_a_value_past_32_bits_raises_struct_error_not_value_error():
    with pytest.raises(struct.error):
        pack_args(_SIG, {"x": 1 << 32})


def test_compile_packer_agrees_with_pack_args_at_every_boundary():
    packer = compile_packer(_SIG)
    for v in (0, 2**31 - 1, 1 << 31, 2**32 - 1):
        assert packer({"x": v}) == pack_args(_SIG, {"x": v})
    with pytest.raises(struct.error):
        packer({"x": 1 << 32})
