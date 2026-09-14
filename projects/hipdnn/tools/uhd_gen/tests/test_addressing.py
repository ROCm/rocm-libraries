# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Ordinal knob addressing: what a pin means, and when it must refuse."""
import json

import pytest

from uhd_gen import addressing

KMD = {"fields": [
    {"name": "block_m", "type": "int"},
    {"name": "dtype", "type": "string"},
    {"name": "causal", "type": "bool"},
    {"name": "scale", "type": "float"},
    {"name": "tile", "type": "int_list"},
    {"name": "never_set", "type": "string"},
]}


def kernel(**metadata):
    return {"metadata": metadata}


def test_an_int_field_addresses_itself_and_gets_no_table():
    tables = addressing.encodings(KMD, [kernel(block_m=128), kernel(block_m=64)])
    assert "block_m" not in tables
    assert addressing.pin("int", 128, tables.get("block_m")) == 128


def test_kernels_differing_only_by_a_string_field_get_distinct_pins():
    # The collision that aborted flyDSL's run: identical exposed tuples for two real kernels.
    kernels = [kernel(block_m=64, dtype="BF16"), kernel(block_m=64, dtype="FP16")]
    tables = addressing.encodings(KMD, kernels)
    pins = {k["metadata"]["dtype"]: addressing.pin("string", k["metadata"]["dtype"], tables["dtype"])
            for k in kernels}
    assert pins["BF16"] != pins["FP16"]
    assert addressing.decode("string", pins["FP16"], tables["dtype"]) == "FP16"


@pytest.mark.parametrize("field_type,values,expected", [
    ("bool", [True, False], [False, True]),
    ("float", [2.5, 0.5, 1.0], [0.5, 1.0, 2.5]),
    ("string", ["b", "A", "a"], ["A", "a", "b"]),
    ("int_list", [[2, 1], [1, 9], [1, 2]], [(1, 2), (1, 9), (2, 1)]),
])
def test_index_order_is_the_specified_total_order(field_type, values, expected):
    # Both sides derive indices independently, so the order is part of the contract: a
    # different sort on one side addresses a different kernel with the same integer.
    name = {"bool": "causal", "float": "scale", "string": "dtype", "int_list": "tile"}[field_type]
    tables = addressing.encodings(KMD, [kernel(**{name: value}) for value in values])
    assert tables[name] == expected


def test_a_kmd_default_is_in_the_value_set_because_the_runtime_completes_metadata():
    # KernelIngestorStateManager::completeMetadata fills the omitted field before the catalog
    # is built, so a table without the default would number every later value one lower than
    # the runtime does -- and every pin would address its neighbour.
    kmd = {"fields": [{"name": "dtype", "type": "string", "default": "FP32"}]}
    tables = addressing.encodings(kmd, [kernel(dtype="BF16"), kernel()])
    assert tables["dtype"] == ["BF16", "FP32"]


def test_a_field_no_kernel_carries_addresses_nothing_and_is_reported():
    tables = addressing.encodings(KMD, [kernel(block_m=64, dtype="BF16")])
    assert "never_set" not in tables
    assert "never_set" in addressing.unaddressable(KMD, tables)
    assert "never_set" not in addressing.exposable(KMD, tables)


def test_exposure_covers_every_addressable_field():
    tables = addressing.encodings(KMD, [kernel(block_m=64, dtype="BF16", causal=True, scale=1.0, tile=[1, 2])])
    assert addressing.exposable(KMD, tables) == ["block_m", "dtype", "causal", "scale", "tile"]


def test_a_value_outside_the_engines_set_refuses_rather_than_addressing_a_neighbour():
    tables = addressing.encodings(KMD, [kernel(dtype="BF16")])
    with pytest.raises(ValueError, match="disagree about the kernel inventory"):
        addressing.pin("string", "FP8", tables["dtype"])


def test_an_ordinal_outside_the_table_refuses_rather_than_decoding_silently():
    tables = addressing.encodings(KMD, [kernel(dtype="BF16")])
    with pytest.raises(ValueError, match="outside this field's value set"):
        addressing.decode("string", 7, tables["dtype"])


def test_pins_round_trip_through_json_so_a_recorded_row_stays_readable():
    kernels = [kernel(tile=[1, 2]), kernel(tile=[4, 4])]
    tables = addressing.encodings(KMD, kernels)
    recorded = json.loads(json.dumps({"tile": tables["tile"]}))
    assert addressing.decode("int_list", addressing.pin("int_list", [4, 4], tables["tile"]),
                             [tuple(value) for value in recorded["tile"]]) == (4, 4)
