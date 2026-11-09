# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for expanding a configuration string into numeric slots and a word shape."""

from __future__ import annotations

from uhd_gen.dataset.config_features import (
    ABSENT,
    expand,
    numeric_slots,
    required_slots,
    word_key,
)

# Three unrelated layouts from one MIOpen corpus: tuning tuple, C++ template, index.
TUPLE = "fwd,nhwc,bf16,0,0,32,64,32,16,64,4,1,1,1,1"
TEMPLATE = (
    "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 8>"
)
INDEX = "205"


def test_numbers_are_taken_in_order():
    assert numeric_slots(TUPLE, 6) == [0, 0, 32, 64, 32, 16]
    assert numeric_slots(INDEX, 3) == [205, ABSENT, ABSENT]


def test_digits_inside_a_name_are_not_knobs():
    """Digits in `bf16` or `Filter1x1Pad0` would shift every later slot."""
    assert numeric_slots("fwd,nhwc,bf16,0,32", 3) == [0, 32, ABSENT]
    assert numeric_slots(TEMPLATE, 6) == [64, 64, 32, 32, 8, ABSENT]


def test_an_absent_slot_is_a_value_not_a_hole():
    """-1, never NaN: NaN is indistinguishable from a field that went unrecorded."""
    slots = numeric_slots(INDEX, 4)
    assert slots[1:] == [ABSENT, ABSENT, ABSENT]
    assert not any(value != value for value in slots)  # no NaN


def test_slot_count_is_taken_from_the_widest_descriptor():
    """Sized from the data, so a kernel with more knobs is not truncated."""
    # TUPLE has 12 standalone numbers, TEMPLATE 5, INDEX 1.
    assert required_slots([INDEX, TEMPLATE]) == 5
    assert required_slots([INDEX, TUPLE, TEMPLATE]) == 12


def test_the_word_shape_is_the_combination_not_the_words():
    """`Default` beside `OddC` is not the same kernel as `Default` alone."""
    assert word_key(INDEX) == ""
    assert "Filter1x1Pad0" in word_key(TEMPLATE)
    assert word_key("a<Default>") != word_key("a<Default,OddC>")


def test_the_word_shape_is_returned_as_text_not_a_code():
    """The training tool numbers shapes, under features_hash (RFC 0019 §6.5)."""
    _, shapes, _ = expand([TUPLE, TEMPLATE, INDEX])
    assert all(isinstance(shape, str) for shape in shapes)
    assert shapes[2] == "", "a descriptor of pure digits has no word shape"
    assert "Filter1x1Pad0" in shapes[1]


def test_the_shape_is_stable_for_the_same_descriptor():
    """The same string must always give the same shape, whatever else the corpus holds."""
    alone = expand([TEMPLATE])[1]
    crowded = expand([INDEX, TEMPLATE, TUPLE])[1]
    assert alone[0] == crowded[1]


def test_configurations_of_one_kernel_become_distinguishable():
    """Unexpanded, two configurations of one solver give identical feature rows."""
    a = "fwd,nhwc,bf16,0,0,32,64"
    b = "fwd,nhwc,bf16,0,0,128,64"
    rows, shapes, _ = expand([a, b])
    assert rows[0] != rows[1], "two configurations still look identical to a model"
    assert (
        shapes[0] == shapes[1]
    ), "same word shape, so only the numbers should separate them"
