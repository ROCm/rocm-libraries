# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""completion.py: which shards of a finished run still have work for a resumed link."""

import completion
import pytest
from test_reconcile import row, write_pass

B = [f"v/test_backward.py::b{i}" for i in (1, 2, 3)]
SWEEP = {"kind": "sweep", "direction": "bwd", "pin_bwd": "X"}


def pass_one(shard, rows=None, collected=B):
    """Pass 1: the heuristic served Y, with X also applicable, for every case by default."""
    rows = (
        [row(n, bwd="hipdnn:Y", bwd_engines="X;Y") for n in B] if rows is None else rows
    )
    return write_pass(shard, "", {}, rows, collected)


def sweep_x(shard, ids=B, listing=None):
    write_pass(
        shard,
        "all-engines/bwd/X",
        SWEEP,
        [row(n, bwd="hipdnn:X") for n in ids],
        listing or B,
    )


def test_a_shard_whose_every_pass_is_complete_has_no_gaps(tmp_path):
    pass_one(tmp_path)
    sweep_x(tmp_path)
    assert completion.shard_gaps(tmp_path) == []


def test_an_unfinished_pass_one_is_reported_with_its_missing_count(tmp_path):
    pass_one(tmp_path, [row(n, bwd="hipdnn:Y", bwd_engines="Y") for n in B[:1]])
    assert completion.shard_gaps(tmp_path) == ["main: 2 of 3 cases have no row"]


def test_a_truncated_last_row_counts_as_missing(tmp_path):
    d = pass_one(tmp_path, [row(n, bwd="hipdnn:Y", bwd_engines="Y") for n in B])
    *whole, last = (d / "results.csv").read_text().splitlines(keepends=True)
    (d / "results.csv").write_text("".join(whole) + last[:25])  # a torn trailing write
    assert completion.shard_gaps(tmp_path) == ["main: 1 of 3 cases have no row"]


@pytest.mark.parametrize("text", [None, ""])
def test_a_shard_without_a_collected_list_goes_back_whole(tmp_path, text):
    if text is not None:
        (tmp_path / "collected.txt").write_text(text)
    assert completion.shard_gaps(tmp_path) == ["collected.txt is missing or empty"]


def test_a_pass_the_first_pass_calls_for_but_that_never_started_is_a_gap(tmp_path):
    pass_one(
        tmp_path
    )  # X applicable to all three: a link that died before the sweep of X
    assert completion.shard_gaps(tmp_path) == ["bwd/X: not started (3 cases)"]


def test_a_sweep_cut_short_is_a_gap_even_when_its_node_id_listing_was_complete(
    tmp_path,
):
    pass_one(tmp_path)
    sweep_x(tmp_path, ids=B[:1])
    assert completion.shard_gaps(tmp_path) == ["bwd/X: 2 of 3 cases have no row"]


def test_a_sweep_listing_longer_than_the_call_for_is_still_held_to_its_rows(tmp_path):
    pass_one(tmp_path, [row(B[0], bwd="hipdnn:Y", bwd_engines="X;Y")], collected=B[:1])
    sweep_x(tmp_path, ids=B[:1], listing=B)  # written when pass 1 was longer
    assert completion.shard_gaps(tmp_path) == ["bwd/X: 2 of 3 cases have no row"]


def test_reference_forward_passes_follow_a_failed_forward(tmp_path):
    failed = row(B[0], "fail", fwd="hipdnn:A", bwd_engines="X", fail_side="fwd")
    pass_one(
        tmp_path,
        [
            failed,
            row(B[1], bwd="hipdnn:X", bwd_engines="X"),
            row(B[2], bwd="hipdnn:X", bwd_engines="X"),
        ],
    )
    sweep_x(tmp_path, ids=B[:1], listing=B[:1])
    assert completion.shard_gaps(tmp_path) == ["refwd/discovery: not started (1 cases)"]
    refwd = {"kind": "discovery", "direction": "bwd", "pin_fwd": "reference"}
    write_pass(
        tmp_path, "all-engines/refwd/discovery", refwd,
        [row(B[0], bwd="hipdnn:Y", bwd_engines="X;Y")], B[:1],
    )  # fmt: skip
    assert completion.shard_gaps(tmp_path) == ["refwd/X: not started (1 cases)"]
    write_pass(
        tmp_path, "all-engines/refwd/X",
        {"kind": "refwd", "direction": "bwd", "pin_fwd": "reference", "pin_bwd": "X"},
        [row(B[0], bwd="hipdnn:X")], B[:1],
    )  # fmt: skip
    assert completion.shard_gaps(tmp_path) == []


def test_an_existing_pass_directory_outside_the_call_for_is_still_checked(tmp_path):
    pass_one(tmp_path, [row(n, bwd="hipdnn:Y", bwd_engines="Y") for n in B])
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X"}
    write_pass(tmp_path, "all-engines/pair/A__X", pair, [row(B[0])], B)
    assert completion.shard_gaps(tmp_path) == ["pair/A__X: 2 of 3 cases have no row"]


def test_an_unreadable_results_file_is_reported_not_raised(tmp_path):
    d = pass_one(tmp_path)
    with open(d / "results.csv", "a") as f:
        f.write("garbage\n" + B[0] + ",pass\n")
    (gap,) = completion.shard_gaps(tmp_path)
    assert gap.startswith("results unreadable")


def test_cli_lists_incomplete_and_missing_shards_in_submit_format(tmp_path, capsys):
    run = tmp_path / "bwd2"
    for k in (1, 2):
        shard = run / f"default-{k}of3"
        pass_one(shard)
        sweep_x(shard)
    pass_one(run / "default-2of3", [row(B[0])])
    shards = tmp_path / "shards.txt"
    shards.write_text("# bwd\ndefault 1 3\ndefault 2 3\ndefault 3 3\n")
    assert completion.main([str(run), "--shards-file", str(shards)]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["default 2 3", "default 3 3"]
    assert "default-2of3: main: 2 of 3 cases have no row" in captured.err
    assert "default-3of3: no shard directory" in captured.err
    # without a shards file only the directories that exist are looked at
    assert completion.main([str(run)]) == 0
    assert capsys.readouterr().out.splitlines() == ["default 2 3"]


def test_cli_with_a_missing_results_dir_is_a_usage_error(tmp_path, capsys):
    assert completion.main([str(tmp_path / "nope")]) == 2
    assert "no such results dir" in capsys.readouterr().err
