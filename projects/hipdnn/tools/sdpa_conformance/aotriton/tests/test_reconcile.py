# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""reconcile.py: union over shards, truncated trailing rows, not-run rows."""

import pytest

import reconcile
from reconcile import FIELDS, format_row, read_rows


def row(nodeid, outcome="pass", **kw):
    return {**dict.fromkeys(FIELDS, ""), "nodeid": nodeid, "outcome": outcome, **kw}


def shard(root, name, rows, collected, tail=""):
    d = root / name
    d.mkdir(parents=True)
    (d / "results.csv").write_text(
        ",".join(FIELDS) + "\n" + "".join(format_row(r) for r in rows) + tail
    )
    (d / "collected.txt").write_text("".join(f"{c}\n" for c in collected))
    return d


def by_id(rows):
    return {r["nodeid"]: r for r in rows}


def test_union_over_shards_adds_not_run_for_every_collected_id_without_a_row(tmp_path):
    shard(
        tmp_path,
        "shard-1of2",
        [row("a::t1"), row("a::t3", "fail", reason="x")],
        ["a::t1", "a::t3", "a::t5"],
    )
    shard(
        tmp_path,
        "shard-2of2",
        [row("a::t2", "declined", reason="bwd none")],
        ["a::t2", "a::t4"],
    )
    rows = by_id(reconcile.reconcile(tmp_path))
    assert {k: v["outcome"] for k, v in rows.items()} == {
        "a::t1": "pass",
        "a::t3": "fail",
        "a::t2": "declined",
        "a::t5": "not-run",
        "a::t4": "not-run",
    }


@pytest.mark.parametrize(
    "tail",
    [
        "a::t2,pass,hipdnn,hipdnn:e,",  # cut after a field, no newline
        "a::t2,pass,hip",  # cut inside a field
        "a::t2,pass,hipdnn,hipdnn:e,,,\n",  # complete line, too many fields
    ],
)
def test_truncated_trailing_row_becomes_not_run(tmp_path, tail):
    shard(tmp_path, "shard-1of1", [row("a::t1")], ["a::t1", "a::t2"], tail=tail)
    rows = by_id(reconcile.reconcile(tmp_path))
    assert rows["a::t1"]["outcome"] == "pass" and rows["a::t2"]["outcome"] == "not-run"


def test_malformed_row_in_the_middle_is_an_error(tmp_path):
    d = shard(tmp_path, "shard-1of1", [row("a::t1")], ["a::t1"])
    with open(d / "results.csv", "a") as f:
        f.write("garbage\n" + format_row(row("a::t2")))
    with pytest.raises(ValueError, match="malformed"):
        reconcile.reconcile(tmp_path)


def test_a_case_recorded_twice_keeps_its_first_row(tmp_path):
    shard(tmp_path, "shard-1of2", [row("a::t1", "pass")], ["a::t1"])
    shard(tmp_path, "shard-2of2", [row("a::t1", "fail")], ["a::t1"])
    assert by_id(reconcile.reconcile(tmp_path))["a::t1"]["outcome"] == "pass"


def test_rows_round_trip_with_commas_quotes_and_newlines_in_reasons(tmp_path):
    awkward = row("a::t[1,2]", "fail", reason='bad "quote", and\nnewline')
    shard(tmp_path, "shard-1of1", [awkward], ["a::t[1,2]"])
    got = reconcile.reconcile(tmp_path)[0]
    assert got["nodeid"] == "a::t[1,2]" and got["reason"] == 'bad "quote", and newline'


def test_summary_and_cli_output(tmp_path, capsys):
    shard(
        tmp_path,
        "shard-1of1",
        [
            row("a::t1", mode="hipdnn"),
            row("a::t2", "fail", reason="Forward pass is_allclose=False"),
            row("a::t3", "declined", reason="bwd none"),
            row("a::t4", "declined", reason="bwd none"),
            row("a::t5", "native-skip", reason="hdim"),
            row("a::t7", "no-hipdnn"),
        ],
        ["a::t1", "a::t2", "a::t3", "a::t4", "a::t5", "a::t6", "a::t7"],
    )
    out = tmp_path / "reconciled.csv"
    assert reconcile.main([str(tmp_path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "7 cases" in text and "not-run" in text and "no-hipdnn" in text
    assert "2 declined: bwd none" in " ".join(text.split())
    assert "fail a::t2  Forward pass is_allclose=False" in text
    assert len(read_rows(out)) == 7
