# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a knob ordinal means, and when the tool must refuse to guess."""
import json

import pytest

from uhd_gen import addressing


def candidate(knobs, features):
    return {"knob_settings": knobs, "kernel_features": {"kernel." + k: v for k, v in features.items()}}


def test_an_ordinal_learns_its_value_from_the_candidate_it_addressed():
    # The pair the engine hands over on every enumerated candidate: the pin that selects
    # this kernel, and what the kernel is. No second derivation, so nothing to disagree.
    table = addressing.observe([
        candidate({"dtype": 0, "block_m": 256}, {"dtype": "BF16", "block_m": 256}),
        candidate({"dtype": 1, "block_m": 128}, {"dtype": "FP16", "block_m": 128}),
    ])
    assert addressing.decode(table, "dtype", 1) == "FP16"
    assert addressing.is_ordinal(table, "dtype")


def test_an_integer_knob_pins_its_own_value_and_is_not_an_ordinal():
    """`block_m=256` means 256, not "the value at index 256" -- a reader must be able to
    tell the two apart, because only one of them is meaningful without this table."""
    table = addressing.observe([candidate({"block_m": 256}, {"block_m": 256})])
    assert not addressing.is_ordinal(table, "block_m")
    assert addressing.decode(table, "block_m", 256) == 256


@pytest.mark.parametrize("value", ["BF16", True, 2.5, [1, 2]])
def test_every_non_integer_kmd_type_is_addressable(value):
    """bool, float, string and int_list all pin through an index; before this they were
    dropped from the tuple and their kernels were unreachable."""
    table = addressing.observe([candidate({"field": 0}, {"field": value})])
    expected = tuple(value) if isinstance(value, list) else value
    assert addressing.decode(table, "field", 0) == expected


def test_two_kernels_differing_only_in_a_string_field_get_distinct_pins():
    """The collision that aborted a real run: identical exposed tuples for two kernels."""
    table = addressing.observe([
        candidate({"block_m": 64, "dtype": 0}, {"block_m": 64, "dtype": "BF16"}),
        candidate({"block_m": 64, "dtype": 1}, {"block_m": 64, "dtype": "FP16"}),
    ])
    assert addressing.decode(table, "dtype", 0) != addressing.decode(table, "dtype", 1)


def test_a_numbering_that_changes_mid_corpus_is_an_error_not_an_overwrite():
    """The failure this design exists to prevent. If ordinal 1 meant FP16 for the first
    half of a collection and BF16 for the second, every row from the first half replays
    against a different kernel -- silently, because both are valid integers."""
    table = addressing.observe([candidate({"dtype": 1}, {"dtype": "FP16"})])
    with pytest.raises(ValueError, match="numbering changed during collection"):
        addressing.observe([candidate({"dtype": 1}, {"dtype": "BF16"})], table)


def test_an_unobserved_ordinal_refuses_rather_than_returning_a_neighbour():
    table = addressing.observe([candidate({"dtype": 0}, {"dtype": "BF16"})])
    with pytest.raises(ValueError, match="no observed value for ordinal"):
        addressing.decode(table, "dtype", 7)


def test_a_knob_no_candidate_pinned_is_reported():
    """An advertised knob that addresses nothing is a pack defect whose collision surfaces
    far from this cause."""
    table = addressing.observe([candidate({"dtype": 0}, {"dtype": "BF16"})])
    assert addressing.unaddressable(["dtype", "never_used"], table) == ["never_used"]


def test_the_manifest_form_survives_json_and_says_which_knobs_are_indices():
    table = addressing.observe([
        candidate({"dtype": 0, "tile": 1, "block_m": 64},
                  {"dtype": "BF16", "tile": [4, 4], "block_m": 64}),
    ])
    restored = json.loads(json.dumps(addressing.as_manifest(table)))
    assert restored["dtype"]["ordinal"] is True
    assert restored["block_m"]["ordinal"] is False
    assert restored["tile"]["values"] == [{"pin": 1, "value": [4, 4]}]
