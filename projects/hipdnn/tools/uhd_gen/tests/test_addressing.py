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


def test_shards_of_one_numbering_merge_into_its_union():
    """Each shard of a catalog collection observes only what its own graphs enumerated."""
    first = addressing.as_manifest(addressing.observe([
        candidate({"dtype": 0, "batch": 1}, {"dtype": "BF16", "batch": 1})]))
    second = addressing.as_manifest(addressing.observe([
        candidate({"dtype": 1, "batch": 128}, {"dtype": "FP16", "batch": 128}),
        candidate({"dtype": 0, "batch": 4}, {"dtype": "BF16", "batch": 4})]))
    merged = addressing.merge_manifests([first, second])
    assert merged["dtype"] == {"ordinal": True, "values": [{"pin": 0, "value": "BF16"},
                                                           {"pin": 1, "value": "FP16"}]}
    assert merged["batch"]["ordinal"] is False
    assert [v["pin"] for v in merged["batch"]["values"]] == [1, 4, 128]


def test_shards_that_read_one_pin_differently_are_refused():
    first = addressing.as_manifest(addressing.observe([candidate({"dtype": 0}, {"dtype": "BF16"})]))
    second = addressing.as_manifest(addressing.observe([candidate({"dtype": 0}, {"dtype": "FP16"})]))
    with pytest.raises(ValueError, match="numbered them differently"):
        addressing.merge_manifests([first, second])


def test_load_collections_trains_shards_whose_tables_differ_only_by_what_they_saw(tmp_path):
    from uhd_gen.generate import COLLECTION_MANIFEST, COLLECTION_SCHEMA, load_collections

    def collection(name, at, table):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "corpus.json").write_text("[]")
        (directory / COLLECTION_MANIFEST).write_text(json.dumps({
            "schema": COLLECTION_SCHEMA, "collected_at": at, "role": "sort_kernel_catalog",
            "engine_name": "e", "engine_id": "1", "selector_revision": None,
            "trained_against": {"ued": {"id": "u"}}, "collection_knobs": ["dtype"],
            "knob_encodings": table, "shipping_knobs": ["dtype"], "sources": [None],
            "published": [], "kernel_fields": ["kernel.dtype"], "graphs": [], "commands": [], "devices": [], "row_counts": {}}))
        return directory

    shards = [collection("s0", "2026-10-01T00:00:00", addressing.as_manifest(addressing.observe(
                  [candidate({"dtype": 0}, {"dtype": "BF16"})]))),
              collection("s1", "2026-10-01T00:00:01", addressing.as_manifest(addressing.observe(
                  [candidate({"dtype": 1}, {"dtype": "FP16"})])))]
    merged = load_collections(shards, role="sort_kernel_catalog", sources=[None])
    assert [v["value"] for v in merged["knob_encodings"]["dtype"]["values"]] == ["BF16", "FP16"]


def test_a_catalog_ranker_may_rank_only_on_what_the_shipping_ued_exposes():
    """The runtime drops a model ranking on a non-knob field; training must not produce one."""
    from uhd_gen.features import build_features_signature
    from uhd_gen.generate import exposed_axes, refuse_unexposed_axes

    kernel_fields = {"kernel.dtype", "kernel.block_m"}
    assert exposed_axes(["block_m"]) == {"kernel.block_m"}
    refuse_unexposed_axes(build_features_signature(["kernel.block_m", "q.seqlen"]),
                          kernel_fields, ["block_m"])
    with pytest.raises(ValueError, match=r"\['kernel.dtype'\].*does not expose"):
        refuse_unexposed_axes(build_features_signature(["kernel.dtype", "kernel.block_m"]),
                              kernel_fields, ["block_m"])
