# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Superset template grouping of SDPA forward rows (topology.superset_map).

Each test builds a few rows as dictionaries, takes their signatures through
select_rows.Model with a small class-to-value table, and checks which superset template
each signature joins.
"""
import json
import random
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import select_rows  # noqa: E402
import topology  # noqa: E402

CLASS_VALUES = {
    "ragged_tensors": {
        "padded": [],
        "ragged": ["Q", "K", "V", "O"],
        "ragged_q_side": ["Q", "O"],
        "ragged_kv_side": ["K", "V"],
    },
    "ragged_stats": {"dense": [], "packed_with_q": ["Stats"]},
    "paged_sides": {"present": ["K", "V"], "k_only": ["K"], "v_only": ["V"]},
    "page_tables": {"separate": 2, "shared": 1},
    "packed_page_table": {"dense": False, "packed": True},
    "alignment": {"wide_aligned": {"start_offset": 0, "declared_alignment": 16}},
    "head_dim": {"d64": 64, "d128": 128},
}

BASE_ROW = {
    "level": "standard",
    "batch_packing": "padded",
    "ragged_offset_form": "-",
    "o_packing": "-",
    "v_packing": "-",
    "stats_packing": "-",
    "per_seq_len_q": "-",
    "paged_kv": "absent",
    "page_table_storage": "-",
    "page_table_sharing": "-",
    "sequence_lengths": "uniform",
    "block_mask": "absent",
    "bias": "absent",
    "attention_sinks": "absent",
    "scale_source": "attribute_value",
    "dropout": "absent",
    "dropout_rng_state": "-",
    "dropout_pattern_output": "-",
    "stats_output": "not_requested",
    "max_sumexp_output": "not_requested",
    "amax_s": "not_requested",
    "amax_o": "not_requested",
    "descale_q": "absent",
    "descale_k": "absent",
    "descale_v": "absent",
    "s_quantization": "absent",
    "scale_o": "absent",
}


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "class-to-value.json"
    path.write_text(json.dumps(CLASS_VALUES))
    return select_rows.Model(path)


def row(**changes):
    out = dict(BASE_ROW)
    out.update(changes)
    return out


def signatures(model, rows):
    return [model.tensors(r, True) for r in rows]


def superset_of(model, rows):
    """{signature: superset signature} over the rows, through the name check."""
    mapping = topology.superset_map(signatures(model, rows))
    topology.check_superset_map(mapping)
    return mapping


def test_a_signature_joins_only_a_template_that_holds_all_of_its_tensors(model):
    full, no_bias, other = signatures(
        model,
        [
            row(bias="present", stats_output="requested", attention_sinks="present"),
            row(stats_output="requested", attention_sinks="present"),
            row(bias="present", block_mask="present"),
        ],
    )
    mapping = topology.superset_map([full, no_bias, other])
    assert mapping[no_bias] == full
    assert mapping[full] == full
    assert mapping[other] == other
    assert not set(other) <= set(full)


def test_a_signature_joins_a_larger_template_through_a_smaller_one(model):
    largest, middle, smallest = signatures(
        model,
        [
            row(bias="present", stats_output="requested", attention_sinks="present"),
            row(bias="present", stats_output="requested"),
            row(bias="present"),
        ],
    )
    mapping = topology.superset_map([smallest, middle, largest])
    assert mapping == {largest: largest, middle: largest, smallest: largest}


@pytest.mark.parametrize(
    "larger, smaller",
    [
        (
            {"scale_source": "by_value_constant_tensor", "bias": "present"},
            {"scale_source": "attribute_value"},
        ),
        (
            {"scale_source": "runtime_parameter", "bias": "present"},
            {"scale_source": "by_value_constant_tensor"},
        ),
        (
            {"scale_source": "device_tensor", "bias": "present"},
            {"scale_source": "runtime_parameter"},
        ),
        (
            {"paged_kv": "present", "page_table_storage": "dense", "bias": "present"},
            {"sequence_lengths": "variable"},
        ),
        (
            {
                "paged_kv": "present",
                "page_table_storage": "dense",
                "page_table_sharing": "separate",
                "bias": "present",
            },
            {
                "paged_kv": "k_only",
                "page_table_storage": "dense",
                "page_table_sharing": "separate",
            },
        ),
        (
            {
                "paged_kv": "present",
                "page_table_storage": "packed",
                "page_table_sharing": "separate",
                "bias": "present",
            },
            {
                "paged_kv": "present",
                "page_table_storage": "dense",
                "page_table_sharing": "separate",
            },
        ),
        (
            {
                "dropout": "random",
                "dropout_rng_state": "caller_supplied",
                "dropout_pattern_output": "requested",
                "bias": "present",
            },
            {"dropout": "absent"},
        ),
        (
            {
                "dropout": "caller_mask",
                "dropout_pattern_output": "requested",
                "bias": "present",
            },
            {"dropout": "absent"},
        ),
        (
            {
                "dropout": "random",
                "dropout_rng_state": "caller_supplied_device_memory",
                "dropout_pattern_output": "requested",
                "bias": "present",
            },
            {
                "dropout": "random",
                "dropout_rng_state": "not_supplied",
                "dropout_pattern_output": "requested",
            },
        ),
        (
            {
                "dropout": "random",
                "dropout_rng_state": "caller_supplied_device_memory",
                "dropout_pattern_output": "requested",
                "bias": "present",
            },
            {
                "dropout": "random",
                "dropout_rng_state": "caller_supplied",
                "dropout_pattern_output": "requested",
            },
        ),
        (
            {
                "dropout": "random",
                "dropout_rng_state": "caller_supplied",
                "dropout_pattern_output": "requested",
                "bias": "present",
            },
            {
                "dropout": "random",
                "dropout_rng_state": "caller_supplied_device_memory",
                "dropout_pattern_output": "requested",
            },
        ),
        (
            {"batch_packing": "ragged", "bias": "present"},
            {"batch_packing": "ragged_q_side"},
        ),
        (
            {"batch_packing": "ragged_q_side", "bias": "present"},
            {"batch_packing": "padded"},
        ),
    ],
)
def test_alternative_forms_never_share_a_superset(model, larger, smaller):
    big, small = signatures(model, [row(**larger), row(**smaller)])
    mapping = superset_of(model, [row(**larger), row(**smaller)])
    assert mapping[small] == small, (small, big)
    assert mapping[big] == big


def test_device_memory_seed_and_offset_are_their_own_roles(model):
    (sig,) = signatures(
        model,
        [
            row(
                dropout="random",
                dropout_rng_state="caller_supplied_device_memory",
                dropout_pattern_output="requested",
            )
        ],
    )
    assert {"dropout_seed", "dropout_offset", "rng_dump"} <= set(sig)
    assert topology.topology_name(sig) == "DSeed_DOff_Rng"


@pytest.mark.parametrize("side", ["descale_q", "descale_k", "descale_v"])
def test_descale_forms_never_share_a_superset(model, side):
    by_value = row(**{side: "present", "bias": "present"})
    per_head = row(**{side: "per_head"})
    mapping = superset_of(model, [by_value, per_head])
    for sig in signatures(model, [by_value, per_head]):
        assert mapping[sig] == sig
    assert not topology.can_hold(
        model.tensors(row(**{side: "per_head", "bias": "present"}), True),
        model.tensors(row(**{side: "present"}), True),
    )


def test_an_absent_descale_may_join_a_template_with_one(model):
    with_descales = row(descale_q="present", descale_k="present", descale_v="present")
    without = row()
    big, small = signatures(model, [with_descales, without])
    assert superset_of(model, [with_descales, without])[small] == big


def test_paged_rows_keep_their_sequence_lengths(model):
    paged = row(paged_kv="present", page_table_storage="dense", bias="present")
    paged_small = row(paged_kv="present", page_table_storage="dense")
    big, small = signatures(model, [paged, paged_small])
    assert {"seq_len_q", "seq_len_kv"} <= set(small)
    assert superset_of(model, [paged, paged_small])[small] == big


def test_each_signature_has_one_superset_that_holds_it(model):
    rows = [
        row(bias="present", stats_output="requested"),
        row(bias="present"),
        row(stats_output="requested"),
        row(),
        row(scale_source="runtime_parameter", stats_output="requested"),
        row(scale_source="runtime_parameter"),
        row(attention_sinks="present", block_mask="present"),
        row(block_mask="present"),
    ]
    mapping = superset_of(model, rows)
    assert set(mapping) == set(signatures(model, rows))
    for sig, sup in mapping.items():
        assert topology.can_hold(sup, sig)
        assert mapping[sup] == sup
    assert len(set(mapping.values())) == 3


def test_grouping_does_not_depend_on_the_input_order(model):
    rows = [
        row(bias="present", stats_output="requested"),
        row(bias="present", attention_sinks="present"),
        row(bias="present"),
        row(stats_output="requested"),
        row(attention_sinks="present"),
        row(),
        row(block_mask="present", stats_output="requested"),
        row(block_mask="present"),
    ]
    sigs = signatures(model, rows)
    results = []
    for seed in (1, 2, 3):
        shuffled = list(sigs)
        random.Random(seed).shuffle(shuffled)
        mapping = topology.superset_map(shuffled)
        names = topology.check_superset_map(mapping)
        results.append((mapping, names))
    assert all(r == results[0] for r in results)


def test_ties_are_broken_by_the_sorted_roles(model):
    with_bias = row(bias="present", stats_output="requested")
    with_sink = row(attention_sinks="present", stats_output="requested")
    stats_only = row(stats_output="requested")
    bias_sig, sink_sig, stats_sig = signatures(
        model, [with_bias, with_sink, stats_only]
    )
    assert len(bias_sig) == len(sink_sig)
    mapping = topology.superset_map([sink_sig, stats_sig, bias_sig])
    assert mapping[stats_sig] == min(bias_sig, sink_sig)
    assert mapping[bias_sig] == bias_sig and mapping[sink_sig] == sink_sig


def test_the_smallest_holding_superset_wins(model):
    largest = row(bias="present", attention_sinks="present", stats_output="requested")
    other = row(block_mask="present", bias="present")
    smallest = row(bias="present")
    largest_sig, other_sig, smallest_sig = signatures(model, [largest, other, smallest])
    assert len(other_sig) < len(largest_sig)
    mapping = topology.superset_map([largest_sig, other_sig, smallest_sig])
    assert mapping[other_sig] == other_sig
    assert mapping[smallest_sig] == other_sig


def test_the_superset_name_is_the_name_of_its_largest_tensor_set(model):
    full = row(bias="present", stats_output="requested")
    part = row(stats_output="requested")
    full_sig, part_sig = signatures(model, [full, part])
    mapping = topology.superset_map([full_sig, part_sig])
    names = topology.check_superset_map(mapping)
    assert names[mapping[part_sig]] == topology.topology_name(full_sig) == "Bias_Stats"
    assert topology.superset_topologies(
        [("standard", full_sig), ("standard", part_sig)]
    ) == {("standard", full_sig): "Bias_Stats", ("standard", part_sig): "Bias_Stats"}


def test_each_tier_has_its_own_supersets(model):
    full_sig, part_sig = signatures(
        model, [row(bias="present", stats_output="requested"), row(bias="present")]
    )
    names = topology.superset_topologies(
        [("quick", part_sig), ("full", full_sig), ("full", part_sig)]
    )
    assert names[("quick", part_sig)] == "Bias"
    assert names[("full", part_sig)] == "Bias_Stats"


def test_grouping_summary_counts_sweeps_without_and_with_supersets(model):
    full_sig, part_sig, other_sig = signatures(
        model,
        [
            row(bias="present", stats_output="requested"),
            row(bias="present"),
            row(block_mask="present"),
        ],
    )
    summary = topology.grouping_summary(
        [
            ("standard", full_sig),
            ("standard", part_sig),
            ("standard", part_sig),
            ("standard", other_sig),
        ]
    )
    assert summary["standard"] == {
        "rows": 4,
        "sweeps_exact": 3,
        "one_case_sweeps_exact": 2,
        "sweeps_superset": 2,
        "one_case_sweeps_superset": 1,
        "cases_with_absent_tensors": 2,
        "max_absent_tensors": 1,
    }


def test_larger_signatures_are_placed_first_whatever_their_roles_sort_as():
    small = ("k", "o", "q", "v")
    large = ("k", "o", "q", "v", "w_extra")
    assert small < large
    assert topology.superset_map([small, large]) == {small: large, large: large}


def test_every_ragged_offset_role_has_a_token():
    sig = (
        "k",
        "o",
        "q",
        "ragged_offset_kv_side",
        "ragged_offset_q_side",
        "v",
    )
    assert topology.topology_name(sig) == "RgQs_RgKvs"
    sig = (
        "k",
        "o",
        "q",
        "ragged_offset_q",
        "ragged_offset_stats",
        "stats[ragged]",
        "v",
    )
    assert topology.topology_name(sig) == "RgQ_RgSt_StatsRg"


SHARED_TOKENS_STATS = {
    "batch_packing": "ragged",
    "ragged_offset_form": "shared_tokens",
    "o_packing": "ragged",
    "v_packing": "ragged",
    "stats_output": "requested",
}


def test_ragged_stats_is_its_own_role_when_the_offsets_are_shared(model):
    ragged, dense, no_stats = signatures(
        model,
        [
            row(**SHARED_TOKENS_STATS, stats_packing="packed_with_q"),
            row(**SHARED_TOKENS_STATS, stats_packing="dense"),
            row(**dict(SHARED_TOKENS_STATS, stats_output="not_requested")),
        ],
    )
    assert "stats[ragged]" in ragged and "stats" not in ragged
    assert "stats" in dense and "stats[ragged]" not in dense
    assert topology.topology_name(ragged) == "RgQs_RgKvs_StatsRg"
    assert topology.topology_name(dense) == "RgQs_RgKvs_Stats"
    mapping = topology.superset_map([ragged, dense, no_stats])
    topology.check_superset_map(mapping)
    assert mapping[ragged] == ragged
    assert mapping[dense] == dense
    assert mapping[no_stats] in (ragged, dense)


def test_the_loose_signature_keeps_one_stats_role(model):
    loose = model.tensors(
        row(**SHARED_TOKENS_STATS, stats_packing="packed_with_q"), False
    )
    assert "stats" in loose and "stats[ragged]" not in loose
