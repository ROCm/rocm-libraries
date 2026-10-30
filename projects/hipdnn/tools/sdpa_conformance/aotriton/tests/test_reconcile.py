# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""reconcile.py: union over shards, truncated trailing rows, not-run rows."""

import csv

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


def test_csv_without_the_engine_columns_still_reads(tmp_path):
    (tmp_path / "results.csv").write_text(
        ",".join(reconcile.LEGACY_FIELDS) + "\na::t1,pass,hipdnn,hipdnn:e,,\n"
    )
    (only,) = read_rows(tmp_path / "results.csv")
    assert (only["outcome"], only["fwd_engines"], only["bwd_engines"]) == (
        "pass",
        "",
        "",
    )


def sweep(out, direction, engine, rows):
    d = out / "all-engines" / direction / engine.lower()
    d.mkdir(parents=True)
    (d / "engine.txt").write_text(engine + "\n")
    (d / "results.csv").write_text(
        ",".join(FIELDS) + "\n" + "".join(format_row(r) for r in rows)
    )


def test_engines_csv_lists_heuristic_and_pinned_rows_and_never_counts_a_mismatch_as_a_pass(
    tmp_path,
):
    fwd, bwd = "v/test_forward.py::f1", "v/test_backward.py::b1"
    (tmp_path / "results.csv").write_text(
        ",".join(FIELDS)
        + "\n"
        + format_row(row(fwd, fwd="hipdnn:A", fwd_engines="A;B"))
        + format_row(row(bwd, fwd="ref:why", bwd="hipdnn:A", bwd_engines="A;B"))
    )
    sweep(tmp_path, "fwd", "B", [row(fwd, fwd="hipdnn:B")])
    sweep(tmp_path, "bwd", "B", [row(bwd, bwd="hipdnn:A")])  # served A despite the pin
    counts = reconcile.write_engines(tmp_path)
    with open(tmp_path / "engines.csv", newline="") as f:
        table = list(csv.reader(f))
    assert table[0] == list(reconcile.ENGINES_FIELDS)
    assert sorted(table[1:]) == sorted(
        [
            [fwd, "fwd", "A", "heuristic", "pass", ""],
            [bwd, "bwd", "A", "heuristic", "pass", ""],
            [fwd, "fwd", "B", "pinned", "pass", ""],
            [bwd, "bwd", "B", "pinned", "declined", "served A, not the pinned B"],
        ]
    )
    assert counts.splitlines() == [
        "bwd A: pass=1 fail=0 declined=0",
        "bwd B: pass=0 fail=0 declined=1",
        "fwd A: pass=1 fail=0 declined=0",
        "fwd B: pass=1 fail=0 declined=0",
    ]
