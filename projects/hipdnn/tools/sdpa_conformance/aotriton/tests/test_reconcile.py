# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""reconcile.py: union over shards, truncated trailing rows, not-run rows."""

import csv
import json

import pytest

import reconcile
from conformance_contract import repro_case as conformance_repro_case
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
    assert sorted(r[:6] for r in table[1:]) == sorted(
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


FWD_T, BWD_T = "v/test_forward.py::f1", "v/test_backward.py::b1"


def write_pass(out, rel, meta, rows, listing, tail=""):
    """A pass directory under ``out`` (``rel`` = "" for pass 1) with its listing and CSV."""
    d = out / rel if rel else out
    d.mkdir(parents=True, exist_ok=True)
    if rel:
        (d / "pass.json").write_text(json.dumps(meta))
    (d / ("nodeids.txt" if rel else "collected.txt")).write_text(
        "".join(f"{n}\n" for n in listing)
    )
    (d / "results.csv").write_text(
        ",".join(FIELDS) + "\n" + "".join(format_row(r) for r in rows) + tail
    )
    return d


def table(out):
    reconcile.write_engines(out)
    with open(out / "engines.csv", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def test_extra_columns_round_trip_and_an_older_csv_reads_them_empty(tmp_path):
    full = row(
        "a::t1", "fail", fwd_kernel="k1", bwd_kernel="k2", fail_side="bwd",
        stage="execute", repro="repro/x",
    )  # fmt: skip
    (tmp_path / "results.csv").write_text(",".join(FIELDS) + "\n" + format_row(full))
    (got,) = read_rows(tmp_path / "results.csv")
    assert [got[k] for k in reconcile.RESULTS_EXTRA_FIELDS] == [
        "k1", "k2", "bwd", "execute", "repro/x",
    ]  # fmt: skip
    brian = ",".join(FIELDS[:8]) + "\na::t2,pass,hipdnn,hipdnn:e,,,A;B,C\n"
    (tmp_path / "old.csv").write_text(brian)
    (old,) = read_rows(tmp_path / "old.csv")
    assert (old["bwd_engines"], old["fwd_kernel"], old["fail_side"], old["repro"]) == (
        "C",
        "",
        "",
        "",
    )


def test_engines_csv_names_the_engine_of_a_crash_with_its_kernel_and_stage(tmp_path):
    write_pass(
        tmp_path, "", {},
        [row(FWD_T, "crash", fwd="hipdnn:E", fail_side="fwd", stage="execute", fwd_kernel="k9", reason="x")],
        [FWD_T],
    )  # fmt: skip
    (r,) = table(tmp_path)
    assert (r["direction"], r["engine"], r["source"], r["outcome"]) == (
        "fwd",
        "E",
        "heuristic",
        "crash",
    )
    assert (r["kernel"], r["stage"], r["pin_fwd"], r["pin_bwd"]) == (
        "k9",
        "execute",
        "",
        "",
    )


def test_pair_and_refwd_passes_get_engine_rows_and_a_mismatch_is_declined(tmp_path):
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X"}
    write_pass(
        tmp_path, "all-engines/pair/A__X", pair,
        [row(BWD_T, bwd="hipdnn:X", fwd="hipdnn:A", bwd_kernel="kx"),
         row("v/test_backward.py::b2", fwd="hipdnn:B", bwd="hipdnn:X"),
         row("v/test_backward.py::b3", "fail", fwd="hipdnn:A", bwd="hipdnn:X", fail_side="fwd")],
        [BWD_T, "v/test_backward.py::b2", "v/test_backward.py::b3"],
    )  # fmt: skip
    refwd = {
        "kind": "refwd",
        "direction": "bwd",
        "pin_fwd": "reference",
        "pin_bwd": "Y",
    }
    write_pass(
        tmp_path, "all-engines/refwd/Y", refwd,
        [row(BWD_T, fwd="ref:pinned reference", bwd="hipdnn:Y")], [BWD_T],
    )  # fmt: skip
    got = {(r["source"], r["nodeid"].split("::")[1]): r for r in table(tmp_path)}
    ok = got[("pair", "b1")]
    assert (ok["direction"], ok["engine"], ok["outcome"], ok["kernel"]) == (
        "bwd",
        "X",
        "pass",
        "kx",
    )
    assert (ok["pin_fwd"], ok["pin_bwd"]) == ("A", "X")
    bad = got[("pair", "b2")]
    assert bad["outcome"] == "declined" and "not the pinned" in bad["reason"]
    side = got[("pair", "b3")]
    assert (side["direction"], side["engine"]) == (
        "fwd",
        "A",
    )  # the failing side's engine
    assert (got[("refwd", "b1")]["engine"], got[("refwd", "b1")]["pin_fwd"]) == (
        "Y",
        "reference",
    )


def test_a_truncated_pass_csv_yields_not_run_rows_and_capture_dirs_are_ignored(
    tmp_path,
):
    ids = [f"v/test_backward.py::b{i}" for i in (1, 2, 3)]
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X"}
    d = write_pass(
        tmp_path, "all-engines/pair/A__X", pair,
        [row(ids[0], bwd="hipdnn:X", fwd="hipdnn:A")], ids, tail="v/test_backward.py::b2,pa",
    )  # fmt: skip
    write_pass(tmp_path, "", {}, [row(ids[0], bwd="hipdnn:Z")], ids[:2])
    capture = d / "capture"
    capture.mkdir()
    (capture / "results.csv").write_text(
        ",".join(FIELDS) + "\n" + format_row(row(ids[1], "crash"))
    )
    (capture / "nodeids.txt").write_text("not-an-id\n")
    not_run = sorted(
        (r["source"], r["nodeid"], r["engine"])
        for r in table(tmp_path)
        if r["outcome"] == "not-run"
    )
    assert not_run == [
        ("heuristic", ids[1], ""),
        ("pair", ids[1], "X"),
        ("pair", ids[2], "X"),
    ]
    assert not any("capture" in str(p) for p in reconcile.pass_dirs(tmp_path))


def test_pass_accounting_prints_expected_against_recorded_per_pass(tmp_path):
    ids = [f"v/test_backward.py::b{i}" for i in (1, 2, 3)]
    write_pass(tmp_path, "", {}, [row(i) for i in ids], ids)
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X"}
    write_pass(tmp_path, "all-engines/pair/A__X", pair, [row(ids[0])], ids)
    lines = reconcile.pass_accounting(tmp_path).splitlines()
    assert lines == [
        "main: expected 3, recorded 3",
        "pair/A__X: expected 3, recorded 1 (2 not-run)",
    ]


def test_inventory_lists_a_zero_engine_with_its_decline_reasons(tmp_path):
    (tmp_path / "engine-inventory.txt").write_text(
        "hipkernel:alpha\nhipkernel:beta\nASM_SDPA_ENGINE\n"
    )
    write_pass(
        tmp_path,
        "",
        {},
        [row(BWD_T, bwd="hipdnn:hipkernel:alpha", fwd="ref:r")],
        [BWD_T],
    )
    (tmp_path / "declines").mkdir()
    (tmp_path / "declines" / "gw0.jsonl").write_text(
        json.dumps(
            {
                "engine": "hipkernel:beta",
                "line": "[isApplicable] head_dim 192 unsupported",
                "nodeid": BWD_T,
            }
        )
        + "\n"
    )
    text = reconcile.inventory_summary(tmp_path, reconcile.engine_rows(tmp_path))
    alpha = next(l for l in text.splitlines() if l.startswith("hipkernel:alpha"))
    assert "pass=1" in alpha
    beta = text[text.index("hipkernel:beta") :]
    assert "pass=0" in beta.splitlines()[0] and "head_dim 192 unsupported" in beta
    asm = text[text.index("ASM_SDPA_ENGINE") :]
    assert "pass=0" in asm.splitlines()[0]


def test_a_watchdog_kill_turns_the_matching_crash_row_into_hang_and_keeps_the_stack(
    tmp_path,
):
    crash = row(FWD_T, "crash", fail_side="fwd", reason="worker crashed (pid 4242)")
    other = row("v/test_forward.py::f2", "crash", reason="worker crashed (pid 77)")
    d = write_pass(tmp_path, "", {}, [crash, other], [FWD_T])
    (d / "watchdog.log").write_text(
        "pytest_gpu_lease.watchdog: sent SIGTERM to pid 4242 (silent 31s)\n"
        "--- pytest_gpu_lease.watchdog: stack of pid 4242 on GPU 0 ---\nframe A\n--- end of stack ---\n"
        "pytest_gpu_lease.watchdog: sent SIGKILL to pid 4242 (silent 40s)\n"
    )
    assert reconcile.mark_hangs(d, tmp_path, "main") == 1
    rows = {r["nodeid"]: r for r in read_rows(d / "results.csv")}
    assert rows[FWD_T]["reason"].startswith("hang") and "4242" in rows[FWD_T]["reason"]
    assert rows["v/test_forward.py::f2"]["reason"] == "worker crashed (pid 77)"
    stack = (
        tmp_path
        / "repro"
        / conformance_repro_case(FWD_T, "main")
        / "watchdog-stack.txt"
    )
    assert "frame A" in stack.read_text()
    assert reconcile.mark_hangs(d, tmp_path, "main") == 0  # idempotent
