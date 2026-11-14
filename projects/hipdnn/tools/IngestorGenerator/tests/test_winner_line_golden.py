# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The Python half of the winner-line guard: ``variant_reduce`` reads the committed
golden shard the SDK test ``TestWinnerLineGolden`` decodes with the SDK reader, and both
assert the same field values. A change to the SDK's line versions or shapes fails one
side until the other follows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import variant_reduce  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "winner_lines_golden.jsonl"

# The values `TestWinnerLineGolden.cpp` expects, field for field.
DEVICE = {
    "gcn_arch_name": "gfx950",
    "warp_size": 64,
    "multi_processor_count": 256,
    "total_global_mem": 309220868096,
    "memory_bus_width": 8192,
    "memory_clock_rate": 1900000,
    "shared_mem_per_block": 163840,
}


def entry(kernel, pack, dispatch, time_ms):
    return {
        "kernel_id": f"c{kernel}000000-0000-4000-8000-00000000000{kernel}",
        "pack_id": f"a{pack}000000-0000-4000-8000-00000000000{pack}",
        "dispatch_id": f"b{pack}000000-0000-4000-8000-00000000000{dispatch}",
        "time_ms": time_ms,
    }


def test_the_reducer_reads_the_golden_lines_as_the_sdk_writes_them():
    v2, dkdv, dq, program = variant_reduce.read_lines(GOLDEN)

    assert (v2.version, v2.slot, v2.picks) == (2, None, None)
    assert v2.entries == [entry(1, 1, 1, 1.5), entry(2, 1, 1, 2.25)]
    assert (dkdv.version, dkdv.slot) == (3, "dkdv")
    assert dkdv.entries == [entry(3, 2, 2, 0.5), entry(4, 2, 2, 0.75)]
    assert (dq.version, dq.slot) == (3, "dq")
    assert dq.entries == [entry(5, 3, 3, 1.25)]
    assert program.version == 3
    assert program.picks == (
        ("dkdv", "c3000000-0000-4000-8000-000000000003"),
        ("dq", "c5000000-0000-4000-8000-000000000005"),
    )
    assert program.time_ms == 4.0
    for line in (v2, dkdv, dq, program):
        assert line.device == DEVICE
    # One graph: its Graph.id differs from the zeroed identity, yet every line keys on it.
    assert len({line.graph for line in (v2, dkdv, dq, program)}) == 1


def test_only_the_golden_rankings_are_ranked_and_each_slot_is_its_own_key():
    records = variant_reduce.read_shard(GOLDEN)
    slots = [variant_reduce._slot_of(key) for key in records]
    assert slots == [None, "dkdv", "dq"]
    assert [len(entries) for entries in records.values()] == [2, 2, 1]
