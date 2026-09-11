# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Expanding a configuration string, and the ways a wrong encoding stays invisible.

Nothing here fails loudly when it breaks, which is why each property is pinned:

- an unexpanded descriptor makes every configuration of a kernel identical to a model, so a
  second layer ranks but cannot prefer -- it does not error, it just never chooses;
- a code assigned from row order changes when shards merge in a different sequence, silently
  re-wiring every split a model learned on it;
- a code invented for an unseen shape puts the row at an arbitrary point in a split learned
  from other values, and nothing downstream attributes that back to here;
- NaN for an absent slot is routed by the learner's default direction, which cannot be
  distinguished from a field that exists but was not recorded.

Standard library and `re` only, like the derive suite beside it, so these keep running where
pandas is absent.
"""

from __future__ import annotations

import pytest

from results_import.descriptor import (
    ABSENT,
    MissingVocabularyEntry,
    build_vocabulary,
    expand,
    numeric_slots,
    required_slots,
    word_key,
)

# The three shapes really seen in one MIOpen corpus: a tuning tuple, a C++ template, and a
# bare index. They share no layout, which is the situation the positional scheme has to survive.
TUPLE = "fwd,nhwc,bf16,0,0,32,64,32,16,64,4,1,1,1,1"
TEMPLATE = "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 32, 32, Filter1x1Pad0, 8>"
INDEX = "205"


def test_numbers_are_taken_in_order():
    assert numeric_slots(TUPLE, 6) == [0, 0, 32, 64, 32, 16]
    assert numeric_slots(INDEX, 3) == [205, ABSENT, ABSENT]


def test_digits_inside_a_name_are_not_knobs():
    """`bf16` is a dtype and `Filter1x1Pad0` is one variant name.

    Harvesting their digits shifts every slot after them, so changing a dtype would appear to
    the model as a different tile size -- and it encodes, as a number, what the word key
    already carries.
    """
    assert numeric_slots("fwd,nhwc,bf16,0,32", 3) == [0, 32, ABSENT]
    assert numeric_slots(TEMPLATE, 6) == [64, 64, 32, 32, 8, ABSENT]


def test_an_absent_slot_is_a_value_not_a_hole():
    """-1, never NaN.

    "This kernel has no such field" is a state a tree can split on. NaN is routed by the
    learner's own default direction and is indistinguishable from a field that exists but went
    unrecorded, so the two would train the same and mean different things.
    """
    slots = numeric_slots(INDEX, 4)
    assert slots[1:] == [ABSENT, ABSENT, ABSENT]
    assert not any(value != value for value in slots)  # no NaN


def test_slot_count_is_taken_from_the_widest_descriptor():
    """Sizing from the data, so a kernel with more knobs than today's widest is not truncated.

    A constant chosen when the tool was written would silently drop the trailing knobs of the
    first kernel to exceed it, and those are the ones a tuner varies last.
    """
    # TUPLE is the widest: 12 standalone numbers, against TEMPLATE's 5 and INDEX's 1.
    assert required_slots([INDEX, TEMPLATE]) == 5
    assert required_slots([INDEX, TUPLE, TEMPLATE]) == 12


def test_the_word_shape_is_the_combination_not_the_words():
    """`Default` beside `OddC` is not the same kernel as `Default` alone."""
    assert word_key(INDEX) == ""
    assert "Filter1x1Pad0" in word_key(TEMPLATE)
    assert word_key("a<Default>") != word_key("a<Default,OddC>")


def test_codes_do_not_depend_on_row_order():
    """A vocabulary built first-seen would renumber when shards merged in another sequence.

    Nothing would error: the model would simply have learned splits against codes that now mean
    something else.
    """
    assert build_vocabulary([TUPLE, TEMPLATE, INDEX]) == build_vocabulary([INDEX, TEMPLATE, TUPLE])


def test_a_shape_outside_a_supplied_vocabulary_is_refused():
    """The check that makes two corpora comparable.

    The vocabulary is closed by construction -- the token sequence is fixed by the kernel -- so
    an unknown shape means this corpus was not measured against the same kernels. Extending it
    here would assign a code training never saw.
    """
    _, _, vocabulary, _ = expand([TUPLE, INDEX])
    with pytest.raises(MissingVocabularyEntry, match="not in the supplied vocabulary"):
        expand([TEMPLATE], vocabulary=vocabulary)


def test_a_reused_vocabulary_encodes_identically():
    """The round trip a validation corpus depends on."""
    _, first, vocabulary, _ = expand([TUPLE, TEMPLATE, INDEX])
    _, second, _, _ = expand([TUPLE, TEMPLATE, INDEX], vocabulary=vocabulary)
    assert first == second


def test_configurations_of_one_kernel_become_distinguishable():
    """The whole point, stated as the property that was previously false.

    Two configurations of the same solver differ only inside the descriptor. Unexpanded they
    produce identical feature rows, so a grouped model's second layer scores them the same and
    cannot prefer either -- measured on a real corpus as zero pairs receiving more than one
    distinct score.
    """
    a = "fwd,nhwc,bf16,0,0,32,64"
    b = "fwd,nhwc,bf16,0,0,128,64"
    rows, codes, _, _ = expand([a, b])
    assert rows[0] != rows[1], "two configurations still look identical to a model"
    assert codes[0] == codes[1], "same word shape, so only the numbers should separate them"
