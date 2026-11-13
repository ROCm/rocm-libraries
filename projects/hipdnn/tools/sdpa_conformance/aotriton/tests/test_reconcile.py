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


def test_engines_csv_marks_listed_cases_a_cut_short_sweep_never_recorded_as_not_run(
    tmp_path,
):
    f1, f2 = "v/test_forward.py::f1", "v/test_forward.py::f2"
    (tmp_path / "results.csv").write_text(
        ",".join(FIELDS)
        + "\n"
        + format_row(row(f1, fwd="hipdnn:A", fwd_engines="A;B"))
        + format_row(row(f2, fwd="hipdnn:A", fwd_engines="A;B"))
    )
    sweep(tmp_path, "fwd", "B", [row(f1, fwd="hipdnn:B")])  # the sweep died before f2
    (tmp_path / "all-engines" / "fwd" / "b" / "nodeids.txt").write_text(f"{f1}\n{f2}\n")
    counts = reconcile.write_engines(tmp_path)
    assert "fwd B: pass=1 fail=0 declined=0 not-run=1" in counts.splitlines()


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
    legacy_csv = ",".join(FIELDS[:8]) + "\na::t2,pass,hipdnn,hipdnn:e,,,A;B,C\n"
    (tmp_path / "old.csv").write_text(legacy_csv)
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


def test_a_forward_side_fail_of_a_backward_case_is_counted_and_listed_against_its_forward(
    tmp_path,
):
    # The forward failed AOTriton's check, so the backward never ran and its column is empty.
    write_pass(
        tmp_path, "", {},
        [row(BWD_T, "fail", fwd="hipdnn:F", fail_side="fwd", stage="execute", reason="NaN")],
        [BWD_T],
    )  # fmt: skip
    (r,) = table(tmp_path)
    assert (r["direction"], r["engine"], r["source"], r["outcome"]) == (
        "fwd",
        "F",
        "heuristic",
        "fail",
    )
    (f,) = reconcile.failure_rows(tmp_path)
    assert (f["nodeid"], f["direction"], f["engine"], f["fail_side"]) == (
        BWD_T,
        "fwd",
        "F",
        "fwd",
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


def test_a_passing_pair_case_with_an_empty_side_is_declined_not_passed(tmp_path):
    # Every side of a passing case ran; only a failing case may have a side that never ran.
    ids = [f"v/test_backward.py::b{i}" for i in (1, 2)]
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X"}
    write_pass(
        tmp_path, "all-engines/pair/A__X", pair,
        [row(ids[0], "pass", bwd="hipdnn:X"),
         row(ids[1], "fail", bwd="hipdnn:X", fail_side="bwd")],
        ids,
    )  # fmt: skip
    got = {r["nodeid"]: r for r in table(tmp_path)}
    assert (got[ids[0]]["outcome"], got[ids[0]]["reason"]) == (
        "declined",
        "served no engine, not the pinned A",
    )
    assert got[ids[1]]["outcome"] == "fail"


def test_a_fail_whose_failing_side_served_its_pin_stays_a_fail_in_pair_passes(tmp_path):
    # The forward failed its comparison, so the backward never ran and its column is empty.
    ids = [f"v/test_backward.py::b{i}" for i in (1, 2, 3)]
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X"}
    write_pass(
        tmp_path, "all-engines/pair/A__X", pair,
        [row(ids[0], "fail", fwd="hipdnn:A", fail_side="fwd", stage="execute", reason="NaN"),
         row(ids[1], "fail", fwd="hipdnn:B", fail_side="fwd", reason="NaN"),
         row(ids[2], "fail", fwd="hipdnn:A", bwd="hipdnn:Z", fail_side="bwd")],
        ids,
    )  # fmt: skip
    got = {r["nodeid"]: r for r in table(tmp_path)}
    kept = got[ids[0]]
    assert (kept["outcome"], kept["direction"], kept["engine"]) == ("fail", "fwd", "A")
    assert (
        got[ids[1]]["outcome"] == "declined"
    )  # the failing forward is not the pinned A
    assert got[ids[2]]["outcome"] == "declined"  # the backward served Z, not X
    # The failure list keeps every fail row, pinned engine or not.
    assert [f["nodeid"] for f in reconcile.failure_rows(tmp_path)] == ids


def test_a_passing_pair_row_credits_both_pinned_engines_and_a_failure_only_its_side(
    tmp_path,
):
    ids = [f"v/test_backward.py::b{i}" for i in (1, 2)]
    write_pass(tmp_path, "", {}, [], [])
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X"}
    write_pass(
        tmp_path, "all-engines/pair/A__X", pair,
        [row(ids[0], "pass", mode="hipdnn", fwd="hipdnn:A", bwd="hipdnn:X"),
         row(ids[1], "fail", mode="hipdnn", fwd="hipdnn:A", bwd="hipdnn:X", fail_side="bwd")],
        ids,
    )  # fmt: skip
    refwd = {
        "kind": "refwd",
        "direction": "bwd",
        "pin_fwd": "reference",
        "pin_bwd": "Y",
    }
    write_pass(
        tmp_path, "all-engines/refwd/Y", refwd,
        [row(ids[0], "pass", mode="ref-fwd", fwd="ref:pinned reference", bwd="hipdnn:Y")],
        ids[:1],
    )  # fmt: skip
    # The pair's passing row credits its forward A as well as its backward X; its failing row
    # charges only the failing backward; the reference-forward pass has no forward engine.
    assert reconcile.engine_counts(reconcile.engine_rows(tmp_path)).splitlines() == [
        "pair bwd X (fwd A): pass=1 fail=1 declined=0",
        "pair fwd A (bwd X): pass=1 fail=0 declined=0",
        "refwd bwd Y: pass=1 fail=0 declined=0",
    ]


def test_a_state_of_an_erroring_call_names_its_engine_like_a_served_one(tmp_path):
    assert reconcile.served_engines("error:X") == {"X"}
    assert reconcile.served_engines("error:") == set()
    sweep(
        tmp_path,
        "bwd",
        "X",
        [row(BWD_T, "fail", bwd="error:X", fail_side="bwd", reason="boom")],
    )
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X"}
    write_pass(
        tmp_path, "all-engines/pair/A__X", pair,
        [row(BWD_T, "fail", fwd="hipdnn:A", bwd="error:X", fail_side="bwd")], [BWD_T],
    )  # fmt: skip
    assert sorted(
        (r["source"], r["engine"], r["outcome"]) for r in table(tmp_path)
    ) == [
        ("pair", "X", "fail"),
        ("pinned", "X", "fail"),
    ]
    assert len(reconcile.failure_rows(tmp_path)) == 2


def test_a_pinned_pass_that_served_a_composite_state_is_declined(tmp_path):
    fwd_id, bwd_id = FWD_T, BWD_T
    sweep(tmp_path, "fwd", "A", [row(fwd_id, fwd="hipdnn:A+B")])
    sweep(tmp_path, "bwd", "A", [row(bwd_id, bwd="hipdnn:A+B")])
    pair = {"kind": "pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "A"}
    write_pass(
        tmp_path, "all-engines/pair/A__A", pair,
        [row(bwd_id, fwd="hipdnn:A", bwd="hipdnn:A+B")], [bwd_id],
    )  # fmt: skip
    refwd = {
        "kind": "refwd",
        "direction": "bwd",
        "pin_fwd": "reference",
        "pin_bwd": "A",
    }
    write_pass(
        tmp_path, "all-engines/refwd/A", refwd,
        [row(bwd_id, fwd="ref:why", bwd="hipdnn:A+B")], [bwd_id],
    )  # fmt: skip
    rows = table(tmp_path)
    assert len(rows) == 4
    for r in rows:
        assert (r["outcome"], r["reason"]) == (
            "declined",
            "served A+B, not the pinned A",
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


def test_inventory_matches_asm_builders_by_class_name_and_never_a_longer_engine_name(
    tmp_path,
):
    (tmp_path / "engine-inventory.txt").write_text("hipkernel:alpha\nASM_SDPA_ENGINE\n")
    write_pass(tmp_path, "", {}, [], [])
    (tmp_path / "declines").mkdir()
    declines = [
        ("SdpaFwdPlanBuilder", "fwd: head_dim 96"),
        ("SdpaBwdPlanBuilder", "bwd: never on gfx950"),
        ("ConvFwdPlanBuilder", "conv: unrelated"),
        ("hipkernel:alpha_v2", "alpha_v2: other engine"),
    ]
    (tmp_path / "declines" / "gw0.jsonl").write_text(
        "".join(
            json.dumps({"engine": e, "line": l, "nodeid": "n"}) + "\n"
            for e, l in declines
        )
    )
    text = reconcile.inventory_summary(tmp_path, [])
    alpha, asm = (
        text[: text.index("ASM_SDPA_ENGINE")],
        text[text.index("ASM_SDPA_ENGINE") :],
    )
    assert "alpha_v2" not in alpha and "no decline reason logged" in alpha
    assert "head_dim 96" in asm and "never on gfx950" in asm and "unrelated" not in asm


# ------------------------------------------------------------------ candidate passes

ENGINE_E = "hipkernel:flydsl_sdpa_cdna"
ENGINE_SLUG = "hipkernel_flydsl_sdpa_cdna"


def candidate_meta(direction, engine, k):
    return {
        "kind": "candidate",
        "direction": direction,
        f"pin_{direction}": engine,
        f"{direction}_candidate": str(k),
    }


def test_the_candidate_columns_round_trip_and_a_csv_without_them_reads_them_empty(
    tmp_path,
):
    assert reconcile.FIELDS[-2:] == ("fwd_candidate", "bwd_candidate")
    full = row("a::t1", fwd_candidate="2", bwd_candidate='{"E": 3}')
    (tmp_path / "results.csv").write_text(",".join(FIELDS) + "\n" + format_row(full))
    (got,) = read_rows(tmp_path / "results.csv")
    assert (got["fwd_candidate"], got["bwd_candidate"]) == ("2", '{"E": 3}')
    before = FIELDS[:-2]  # a results.csv written before the candidate columns
    old = ",".join(before) + "\n" + ",".join(["a::t2", "pass"] + [""] * 11) + "\n"
    (tmp_path / "old.csv").write_text(old)
    (got,) = read_rows(tmp_path / "old.csv")
    assert (got["nodeid"], got["fwd_candidate"], got["bwd_candidate"]) == (
        "a::t2",
        "",
        "",
    )


def test_each_candidate_pass_gets_its_own_engine_rows_naming_the_candidate(tmp_path):
    ids = [FWD_T, "v/test_forward.py::f2"]
    write_pass(tmp_path, "", {}, [row(i, fwd=f"hipdnn:{ENGINE_E}") for i in ids], ids)
    for k in range(3):
        write_pass(
            tmp_path,
            f"cand/fwd/{ENGINE_SLUG}/{k}",
            candidate_meta("fwd", ENGINE_E, k),
            [row(i, fwd=f"hipdnn:{ENGINE_E}", fwd_kernel=f"kernel-{k}") for i in ids],
            ids,
        )
    rows = [r for r in table(tmp_path) if r["source"] == "candidate"]
    assert len(rows) == 6
    assert sorted({(r["engine"], r["fwd_candidate"], r["kernel"]) for r in rows}) == [
        (ENGINE_E, str(k), f"kernel-{k}") for k in range(3)
    ]
    assert {r["pin_fwd"] for r in rows} == {ENGINE_E}
    counts = reconcile.engine_counts(rows)
    assert counts.splitlines() == [
        f"candidate fwd {ENGINE_E} #{k}: pass=2 fail=0 declined=0" for k in range(3)
    ]


def test_a_candidate_pass_that_served_another_engine_is_declined_and_counted_not_honored(
    tmp_path,
):
    ids = [FWD_T, "v/test_forward.py::f2"]
    write_pass(tmp_path, "", {}, [row(i) for i in ids], ids)
    write_pass(
        tmp_path,
        f"cand/fwd/{ENGINE_SLUG}/1",
        candidate_meta("fwd", ENGINE_E, 1),
        [
            row(ids[0], fwd="hipdnn:other"),
            row(
                ids[1],
                "declined",
                reason="fwd candidate_not_honored: requested kernel a, built b",
            ),
        ],
        ids,
    )
    rows = {r["nodeid"]: r for r in table(tmp_path) if r["source"] == "candidate"}
    assert rows[ids[0]]["outcome"] == "declined"
    assert "not the pinned" in rows[ids[0]]["reason"]
    assert rows[ids[1]]["outcome"] == "declined"
    assert "not_honored=1" in reconcile.engine_counts(list(rows.values()))


def test_not_enumerable_rows_carry_the_message_and_an_unreached_case_is_not_run(
    tmp_path,
):
    ids = [FWD_T, "v/test_forward.py::f2"]
    write_pass(tmp_path, "", {}, [row(i) for i in ids], ids)
    message = "Engine does not support matched-catalog enumeration"
    write_pass(
        tmp_path,
        f"cand/fwd/{ENGINE_SLUG}/not-enumerable",
        {"kind": "candidate", "direction": "fwd", "pin_fwd": ENGINE_E, "fwd_candidate": ""},
        [row(ids[0], "not-enumerable", reason=message)],
        ids,
    )  # fmt: skip
    rows = {r["nodeid"]: r for r in table(tmp_path) if r["source"] == "candidate"}
    assert (rows[ids[0]]["outcome"], rows[ids[0]]["reason"]) == (
        "not-enumerable",
        message,
    )
    assert (rows[ids[0]]["engine"], rows[ids[0]]["fwd_candidate"]) == (ENGINE_E, "")
    assert rows[ids[1]]["outcome"] == "not-run"
    assert "not-enumerable" in reconcile.OUTCOMES
    assert "not-enumerable" in reconcile.summarize([row(ids[0], "not-enumerable")])


def test_a_candidate_pair_pass_pins_both_directions_and_both_candidates(tmp_path):
    write_pass(tmp_path, "", {}, [row(BWD_T)], [BWD_T])
    meta = {
        "kind": "candidate-pair", "direction": "bwd", "pin_fwd": "A", "pin_bwd": "X",
        "fwd_candidate": "1", "bwd_candidate": "2",
    }  # fmt: skip
    write_pass(
        tmp_path,
        "cand/pair/A__X/1x2",
        meta,
        [row(BWD_T, fwd="hipdnn:A", bwd="hipdnn:X", bwd_kernel="kx")],
        [BWD_T],
    )
    # a passing candidate-pair row credits both pinned engines, each with its own kernel column
    got = {
        r["direction"]: r for r in table(tmp_path) if r["source"] == "candidate-pair"
    }
    assert sorted(got) == ["bwd", "fwd"]
    assert [(got[d]["engine"], got[d]["kernel"]) for d in ("fwd", "bwd")] == [
        ("A", ""),
        ("X", "kx"),
    ]
    for d in ("fwd", "bwd"):
        assert (got[d]["pin_fwd"], got[d]["pin_bwd"]) == ("A", "X")
        assert (got[d]["fwd_candidate"], got[d]["bwd_candidate"]) == ("1", "2")


def test_candidate_passes_have_their_own_slug_and_accounting_lines(tmp_path):
    ids = [FWD_T, "v/test_forward.py::f2"]
    write_pass(tmp_path, "", {}, [row(i) for i in ids], ids)
    rel = f"cand/fwd/{ENGINE_SLUG}/0"
    d = write_pass(
        tmp_path, rel, candidate_meta("fwd", ENGINE_E, 0), [row(ids[0])], ids
    )
    assert reconcile.pass_slug(tmp_path, d) == f"cand-fwd-{ENGINE_SLUG}-0"
    assert reconcile.pass_accounting(tmp_path).splitlines() == [
        "main: expected 2, recorded 2",
        f"{rel}: expected 2, recorded 1 (1 not-run)",
    ]


# ------------------------------------------------------------------ run directories and the report


def real_shard(root, name, rows, collected):
    """A shard directory as `run_aotriton.py --out` leaves it under ``results/<run>/<suite>-<k>of<N>``."""
    return write_pass(root / name, "", {}, rows, collected)


@pytest.mark.parametrize("names", [["default-1of2", "forward-2of2"], ["shard-1of2"]])
def test_reconcile_reads_every_shard_directory_naming(tmp_path, names):
    for i, name in enumerate(names):
        real_shard(tmp_path, name, [row(f"a::t{i}")], [f"a::t{i}", f"a::x{i}"])
    (tmp_path / "logs").mkdir()  # not a shard
    got = by_id(reconcile.reconcile(tmp_path))
    assert {v["outcome"] for v in got.values()} == {"pass", "not-run"}
    assert len(got) == 2 * len(names)


def test_shards_sort_by_number_not_by_text(tmp_path):
    for k in (10, 2, 1):
        (tmp_path / f"default-{k}of16").mkdir()
    assert [d.name for d in reconcile.shard_dirs(tmp_path)] == [
        "default-1of16",
        "default-2of16",
        "default-10of16",
    ]


def test_a_run_directory_without_shards_is_an_error_not_an_empty_result(
    tmp_path, capsys
):
    (tmp_path / "logs").mkdir()
    with pytest.raises(FileNotFoundError, match="no shard directories"):
        reconcile.reconcile(tmp_path)
    assert reconcile.main([str(tmp_path)]) == 2
    assert "no shard directories" in capsys.readouterr().err


def test_a_shards_file_names_the_expected_shards(tmp_path):
    f = tmp_path / "shards.txt"
    f.write_text("# bwd\ndefault 1 2\n\nforward 2 2 \n")
    assert reconcile.read_shards_file(f) == ["default-1of2", "forward-2of2"]


BWD = [f"v/test_backward.py::b{i}" for i in range(1, 6)]


@pytest.fixture
def two_shard_run(tmp_path):
    """default-1of3: b1 pass, b2 fail (repro), b3 never ran, plus a bwd sweep of X that lost b2;
    default-2of3: b4 crash, b5 pass; default-3of3 has no directory."""
    run = tmp_path / "bwd2"
    one = real_shard(
        run, "default-1of3",
        [row(BWD[0], bwd="hipdnn:X", bwd_engines="X;Y"),
         row(BWD[1], "fail", bwd="hipdnn:Y", bwd_engines="X;Y", fail_side="bwd",
             stage="execute", repro="repro/abc-main", reason="Backward dq is_allclose=False")],
        BWD[:3],
    )  # fmt: skip
    write_pass(
        one, "all-engines/bwd/X", {"kind": "sweep", "direction": "bwd", "pin_bwd": "X"},
        [row(BWD[1], bwd="hipdnn:X")], [BWD[1]],
    )  # fmt: skip
    (one / "engine-inventory.txt").write_text("X\nY\nZ\n")
    real_shard(
        run, "default-2of3",
        [row(BWD[3], "crash", bwd="hipdnn:X", fail_side="bwd", stage="execute",
             reason="worker crashed (pid 9)"),
         row(BWD[4], bwd="hipdnn:Y")],
        BWD[3:],
    )  # fmt: skip
    (run / "default-2of3" / "declines").mkdir()
    (run / "default-2of3" / "declines" / "gw0.jsonl").write_text(
        json.dumps({"engine": "Z", "line": "Z: head_dim 192", "nodeid": BWD[3]}) + "\n"
    )
    (tmp_path / "shards.txt").write_text("default 1 3\ndefault 2 3\ndefault 3 3\n")
    (tmp_path / "baseline.txt").write_text(
        f"# baseline\n{BWD[1]}\nv/test_backward.py::gone\n"
    )
    return run


def test_report_totals_per_shard_accounting_and_missing_shards(two_shard_run):
    text = reconcile.run_report(
        two_shard_run,
        expected=reconcile.read_shards_file(two_shard_run.parent / "shards.txt"),
    )
    lines = [" ".join(l.split()) for l in text.splitlines()]
    assert "default-1of3: collected 3, recorded 2, not-run 1" in lines
    assert "default-2of3: collected 2, recorded 2, not-run 0" in lines
    assert "default-3of3: MISSING (no shard directory)" in lines
    assert "5 cases" in lines and "pass 2" in lines and "not-run 1" in lines
    assert "main: expected 5, recorded 4 (1 not-run)" in lines  # summed over shards
    assert "bwd/X: expected 1, recorded 1" in lines


def test_report_counts_per_engine_across_shards_and_flags_zero_coverage_with_reasons(
    two_shard_run,
):
    text = reconcile.run_report(two_shard_run)
    lines = [" ".join(l.split()) for l in text.splitlines()]
    assert (
        "bwd X: pass=2 fail=0 declined=0 crash=1" in lines
    )  # b1 + sweep b2 pass, b4 crash
    assert "bwd Y: pass=1 fail=1 declined=0" in lines
    assert "bwd -: pass=0 fail=0 declined=0 not-run=1" in lines
    zero = text[text.index("engines with no pass:") :].splitlines()[0]
    assert "Z" in zero and "X" not in zero
    assert (
        "decline: Z: head_dim 192" in text
    )  # logged by another shard than the inventory's


def test_report_lists_every_failure_with_engine_side_stage_and_repro(two_shard_run):
    text = reconcile.run_report(two_shard_run)
    failures = text[text.index("== failures") :]
    fail = next(l for l in failures.splitlines() if BWD[1] in l)
    for part in ("fail", "default-1of3", "main", "bwd Y", "side=bwd", "stage=execute",
                 "repro=default-1of3/repro/abc-main", "is_allclose=False"):  # fmt: skip
        assert part in fail
    crash = next(l for l in failures.splitlines() if BWD[3] in l)
    assert (
        "crash" in crash
        and "default-2of3" in crash
        and "bwd X" in crash
        and "repro=-" in crash
    )


def test_report_diffs_failing_node_ids_against_a_baseline(two_shard_run):
    baseline = two_shard_run.parent / "baseline.txt"
    text = reconcile.run_report(two_shard_run, baseline=baseline)
    diff = text[text.index("== baseline") :]
    assert "baseline 2 ids, failing here 2, still failing 1" in " ".join(diff.split())
    new = diff[diff.index("new failures") :].split("\n\n")[0]
    assert BWD[3] in new and BWD[1] not in new
    gone = diff[diff.index("no longer failing") :]
    assert "v/test_backward.py::gone: not in this run" in gone


def test_report_cli_prints_the_report_and_writes_the_reconciled_csv(
    two_shard_run, tmp_path, capsys
):
    out = tmp_path / "all.csv"
    rc = reconcile.main(
        [str(two_shard_run), "--report", "--shards-file", str(tmp_path / "shards.txt"),
         "--baseline", str(tmp_path / "baseline.txt"), "--out", str(out)]
    )  # fmt: skip
    assert rc == 0
    text = capsys.readouterr().out
    assert "== shards" in text and "== failures" in text and "== baseline" in text
    assert len(read_rows(out)) == 5


def test_report_splits_baseline_ids_that_no_longer_fail_from_ones_that_did_not_run(
    two_shard_run, tmp_path
):
    baseline = tmp_path / "b2.txt"
    baseline.write_text(f"{BWD[0]}\n{BWD[2]}\nv/test_backward.py::gone\n")
    diff = reconcile.run_report(two_shard_run, baseline=baseline)
    diff = diff[diff.index("== baseline") :]
    fixed, unconfirmed = diff[diff.index("no longer failing") :].split(
        "not confirmed fixed"
    )
    assert "no longer failing" in fixed and f"{BWD[0]}: pass" in fixed
    assert BWD[2] not in fixed and "gone" not in fixed
    assert f"{BWD[2]}: not-run" in unconfirmed
    assert "v/test_backward.py::gone: not in this run" in unconfirmed
    assert "(in baseline, outcome" in fixed and "): 1" in fixed
    assert "): 2" in unconfirmed.splitlines()[0]


def test_report_lists_a_node_id_recorded_by_two_shard_directories(tmp_path):
    real_shard(
        tmp_path,
        "default-1of2",
        [row(BWD_T, "pass"), row("v/test_backward.py::b2")],
        [BWD_T],
    )
    real_shard(tmp_path, "default-2of2", [row(BWD_T, "fail")], [BWD_T])
    text = reconcile.run_report(tmp_path)
    section = text[text.index("== duplicate node ids") :].split("\n\n")[0].splitlines()
    assert section == ["== duplicate node ids", f"{BWD_T}: default-1of2, default-2of2"]
    assert by_id(reconcile.reconcile(tmp_path))[BWD_T]["outcome"] == "pass"
    clean = tmp_path / "clean"
    real_shard(clean, "default-1of1", [row(BWD_T)], [BWD_T])
    assert reconcile.run_report(clean).splitlines()[3:5] == [
        "== duplicate node ids",
        "none",
    ]


# ------------------------------------------------------- failures without an engine on the failing side


def test_a_fail_charged_to_a_reference_forward_is_listed_against_the_hipdnn_backward(
    tmp_path,
):
    # AOTriton's backward test checks the forward output after the backward ran, so a hipDNN
    # backward that corrupts O fails on the side of the reference forward, which has no engine.
    real_shard(
        tmp_path, "default-1of1",
        [row(BWD[0], "fail", mode="ref-fwd", fwd="ref:why", bwd="hipdnn:X", fail_side="fwd",
             reason="Forward pass O mismatch"),
         row(BWD[1], "fail", fail_side="bwd", reason="boom")],
        BWD[:2],
    )  # fmt: skip
    shard_dir = tmp_path / "default-1of1"
    got = {f["nodeid"]: f for f in reconcile.failure_rows(shard_dir)}
    assert (got[BWD[0]]["direction"], got[BWD[0]]["engine"]) == ("fwd", "X")
    assert got[BWD[1]]["engine"] == "-"  # no side served an engine: still listed
    text = reconcile.run_report(tmp_path)
    failures = text[text.index("== failures") :]
    assert "== failures (2)" in failures and BWD[0] in failures and BWD[1] in failures
    baseline = tmp_path / "base.txt"
    baseline.write_text(f"{BWD[0]}\n")
    diff = reconcile.run_report(tmp_path, baseline=baseline)
    assert "failing here 2, still failing 1" in " ".join(diff.split())
    assert f"  {BWD[1]}  [fail bwd -]" in diff


# ------------------------------------------------------------------------------- baseline outcomes


def baseline_text(run, ids, tmp_path):
    baseline = tmp_path / "b.txt"
    baseline.write_text("".join(f"{i}\n" for i in ids))
    text = reconcile.run_report(run, baseline=baseline)
    diff = text[text.index("== baseline") :]
    fixed, unconfirmed = diff[diff.index("no longer failing") :].split(
        "not confirmed fixed"
    )
    return fixed, unconfirmed


def test_baseline_counts_only_a_pass_as_fixed_and_lists_every_other_outcome(tmp_path):
    outcomes = ["pass", "declined", "native-skip", "no-hipdnn", "not-run"]
    ids = [f"v/test_backward.py::{o}" for o in outcomes]
    run = tmp_path / "run"
    real_shard(
        run, "default-1of1",
        [row(i, o) for i, o in zip(ids[:4], outcomes)],
        ids,
    )  # fmt: skip
    fixed, unconfirmed = baseline_text(
        run, [*ids, "v/test_backward.py::gone"], tmp_path
    )
    assert f"{ids[0]}: pass" in fixed and "): 1" in fixed.splitlines()[0]
    for i, o in zip(ids[1:], outcomes[1:]):
        assert i not in fixed
        assert f"{i}: {o}" in unconfirmed
    assert "v/test_backward.py::gone: not in this run" in unconfirmed
    assert "): 5" in unconfirmed.splitlines()[0]


# ------------------------------------------------------------------------------- --compare


def compare_runs(tmp_path):
    ids = [f"v/test_backward.py::c{i}" for i in range(1, 8)]
    old, new = tmp_path / "old", tmp_path / "new"
    real_shard(
        old, "default-1of1",
        [row(ids[0], "pass", bwd="hipdnn:X"), row(ids[1], "fail", bwd="hipdnn:X", fail_side="bwd"),
         row(ids[2], "pass", bwd="hipdnn:X"), row(ids[3], "pass", fwd="hipdnn:F", bwd="hipdnn:X"),
         row(ids[4], "pass", bwd="hipdnn:X"), row(ids[5], "pass", bwd="hipdnn:X")],
        ids[:6],
    )  # fmt: skip
    real_shard(
        new, "default-1of1",
        [row(ids[0], "declined"), row(ids[1], "declined"),
         row(ids[2], "pass", bwd="hipdnn:X"), row(ids[3], "pass", fwd="hipdnn:G", bwd="hipdnn:X"),
         row(ids[4], "declined"), row(ids[6], "pass", bwd="hipdnn:X")],
        [ids[0], ids[1], ids[2], ids[3], ids[4], ids[6]],
    )  # fmt: skip
    return ids, old, new


def test_main_prints_the_engine_splits_with_an_unrecorded_mode_and_compares_identical_runs(
    tmp_path, capsys
):
    ids = [f"v/test_backward.py::b{i}" for i in (1, 2, 3)]
    run = tmp_path / "run"
    shard_dir = real_shard(
        run, "default-1of1",
        [row(ids[0], "pass", mode="hipdnn", bwd="hipdnn:X"),
         row(ids[1], "pass", mode="", bwd="hipdnn:X"),
         row(ids[2], "pass", mode="ref-fwd", fwd="ref:why", bwd="hipdnn:X")],
        ids,
    )  # fmt: skip
    assert (
        reconcile.main([str(run), "--engines", str(shard_dir), "--compare", str(run)])
        == 0
    )
    out = capsys.readouterr().out
    assert "outcome transitions: 0 cases changed outcome" in out
    assert "served engine changed, same outcome: 0" in out
    splits = out.split("bwd X\n", 1)[1].splitlines()[:4]
    assert splits == [
        "  file test_backward.py: pass=3 fail=0 declined=0",
        "  mode hipdnn: pass=1 fail=0 declined=0",
        "  mode ref-fwd: pass=1 fail=0 declined=0",
        "  mode unrecorded: pass=1 fail=0 declined=0",
    ]


def test_compare_lists_every_outcome_transition_with_counts_and_examples(tmp_path):
    ids, old, new = compare_runs(tmp_path)
    text = reconcile.compare_runs(new, old)
    lines = [l.strip() for l in text.splitlines()]
    assert "pass -> declined: 2" in lines  # c1, c5: lost coverage
    assert "fail -> declined: 1" in lines  # read as a fix by a failing-ids diff
    assert "pass -> not in run: 1" in lines and "not in run -> pass: 1" in lines
    at = lines.index("pass -> declined: 2")
    assert lines[at + 1 : at + 3] == [ids[0], ids[4]]  # examples, by node id
    assert "5 cases changed outcome" in text
    assert "only in old" in text and "only in new" in text


def test_compare_lists_a_served_engine_change_with_the_same_outcome(tmp_path):
    ids, old, new = compare_runs(tmp_path)
    text = reconcile.compare_runs(new, old)
    tail = text[text.index("served engine changed") :]
    assert "served engine changed, same outcome: 1" in tail
    assert f"{ids[3]} [pass]: fwd F, bwd X -> fwd G, bwd X" in tail
    assert ids[2] not in tail  # same outcome, same engines


def test_compare_option_prints_the_comparison_and_a_missing_run_exits_2(
    tmp_path, capsys
):
    ids, old, new = compare_runs(tmp_path)
    assert reconcile.main([str(new), "--compare", str(old)]) == 0
    out = capsys.readouterr().out
    assert "== compare old -> new" in out and "pass -> declined: 2" in out
    assert reconcile.main([str(new), "--compare", str(tmp_path / "missing")]) == 2
    assert "missing" in capsys.readouterr().err


# ------------------------------------------------------------------------- engine credit and split


def test_a_backward_case_credits_its_forward_engine_and_a_failure_only_the_failing_side(
    tmp_path,
):
    write_pass(
        tmp_path, "", {},
        [row(BWD_T, "pass", mode="hipdnn", fwd="hipdnn:F", bwd="hipdnn:B"),
         row("v/test_backward.py::b2", "fail", mode="hipdnn", fwd="hipdnn:F", bwd="hipdnn:B",
             fail_side="bwd")],
        [BWD_T, "v/test_backward.py::b2"],
    )  # fmt: skip
    got = sorted((r["direction"], r["engine"], r["outcome"]) for r in table(tmp_path))
    assert got == [
        ("bwd", "B", "fail"),
        ("bwd", "B", "pass"),
        ("fwd", "F", "pass"),
    ]
    assert reconcile.engine_counts(reconcile.engine_rows(tmp_path)).splitlines() == [
        "bwd B: pass=1 fail=1 declined=0",
        "fwd F: pass=1 fail=0 declined=0",
    ]


def test_the_engine_counts_split_by_mode_and_by_test_file(tmp_path):
    ids = ["v/test_backward.py::b1", "v/test_backward.py::b2", "v/test_forward.py::f1"]
    real_shard(
        tmp_path, "default-1of1",
        [row(ids[0], mode="hipdnn", fwd="hipdnn:F", bwd="hipdnn:B"),
         row(ids[1], mode="ref-fwd", fwd="ref:why", bwd="hipdnn:B"),
         row(ids[2], mode="hipdnn", fwd="hipdnn:F")],
        ids,
    )  # fmt: skip
    text = reconcile.run_report(tmp_path)
    section = text[text.index("== engines by mode and test file") :].split("\n\n")[0]
    block = section.split("\nbwd B\n")[1].split("\nfwd")[0]
    assert block.splitlines() == [
        "  file test_backward.py: pass=2 fail=0 declined=0",
        "  mode hipdnn: pass=1 fail=0 declined=0",
        "  mode ref-fwd: pass=1 fail=0 declined=0",
    ]
    fwd = section.split("\nfwd F\n")[1]
    assert fwd.splitlines() == [
        "  file test_backward.py: pass=1 fail=0 declined=0",
        "  file test_forward.py: pass=1 fail=0 declined=0",
        "  mode hipdnn: pass=2 fail=0 declined=0",
    ]
    assert "bwd B: pass=2 fail=0 declined=0" in text  # the first table keeps its lines


# --------------------------------------------------------------------------------- skip classes


@pytest.mark.parametrize(
    "reason, klass",
    [
        ("hdim > 192 AITER ASM kernel does not exist.", "kernel-limit"),
        ("hdim < 64 AITER ASM kernel does not exist.", "kernel-limit"),
        ("Skip hdim > 256 on gfx11 arch due to register pressure.", "kernel-limit"),
        ("Bias is not supported in AITER ASM backend", "kernel-limit"),
        ("GQA is not exposed in AITER ASM backend.", "kernel-limit"),
        ("hdim_qk != hdim_vo is not exposed by AITER ASM backend.", "kernel-limit"),
        ("Dropout unsupported in AITER ASM backend for now. Need adjust FWD PRNG function", "kernel-limit"),
        ("AITER ASM does not support matrix bias", "kernel-limit"),
        ("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True", "api-rule"),
        ("Skip large UT with bias to avoid OOM", "resource"),
        ("Skip large tests (qkd > 4096 * 8192 * 256) due to low VRAM.", "resource"),
        ("[Adiffs] Skip due to known OOM.", "resource"),
        ("SKIP_BWD=1 excludes backward checks", "other"),
        ("something unforeseen", "other"),
    ],
)  # fmt: skip
def test_a_native_skip_reason_is_classed_from_the_vendored_skip_messages(reason, klass):
    assert reconcile.skip_class(reason) == klass


def test_every_vendored_skip_message_is_classed_and_the_summary_breaks_native_skips_down():
    import re
    from pathlib import Path

    vendor = Path(reconcile.__file__).parent / "vendor"
    messages = {
        m
        for p in vendor.rglob("*.py")
        for m in re.findall(r"pytest\.skip\(\s*[\"']([^\"']+)[\"']", p.read_text())
    }
    assert messages  # the vendored tests carry skip guards
    other = {m for m in messages if reconcile.skip_class(m) == "other"}
    assert other == {"SKIP_BWD=1 excludes backward checks"}
    rows = [
        row(
            "a::1", "native-skip", reason="hdim > 192 AITER ASM kernel does not exist."
        ),
        row("a::2", "native-skip", reason="Skip large UT with bias to avoid OOM"),
        row("a::3", "native-skip", reason="Skip large UT with bias to avoid OOM"),
        row("a::4", "pass"),
    ]
    lines = reconcile.summarize(rows).splitlines()
    at = lines.index("native-skip by class:")
    assert lines[at + 1 : at + 3] == ["  kernel-limit 1", "  resource     2"]
