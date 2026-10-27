# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the hipdnn-deep-review skill's ledger.py and reviewlib.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parent.parent / "skills" / "hipdnn-deep-review" / "scripts"
)


def _load(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            f"deep_review_{name}", SCRIPTS / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def ledger():
    return _load("ledger")


@pytest.fixture(scope="module")
def rl():
    return _load("reviewlib")


def finding(
    fid,
    tier="IMPORTANT",
    where="a.cpp:10",
    actual="It returns the wrong value.",
    kind="defect",
    extra="",
):
    return (
        f"ID:        {fid}\nKind:      {kind}\nTier:      {tier}\nWhere:     {where}\n"
        f"Expected:  header says non-negative\nActual:    {actual}\n"
        f"Proof:     ran t, exit 1, $EVIDENCE/r1/lanes/x.log {extra}\nFix:       clamp it\n"
    )


def report(lane, findings=(), coverage=(), mlines=()):
    out = [f"# {lane} (round 1, abcdef123456)", "", "## Findings", ""]
    out += list(findings) or ["(none)"]
    out += ["", "## Coverage", "", "| File | Read depth | Tests |", "|---|---|---|"]
    out += [f"| {p} | {d} | {t} |" for p, d, t in coverage]
    out += ["", "## Checked, no finding", ""] + list(mlines)
    out += ["", "## Handoffs", "", "## Approval requests", ""]
    return "\n".join(out) + "\n"


def verify_file(fid, verdict, tier, claim="", reason="r"):
    return (
        f"ID:        {fid}\nVerdict:   {verdict}\nTier:      {tier}\nClaim:     {claim}\n"
        f"Reason:    {reason}\nProof:     cmd, exit 0, $EVIDENCE/x\nNot established: nothing\n"
    )


@pytest.fixture
def ev(tmp_path):
    """Evidence dir with a seeded round 1: two changed files, CORR1 and TESTQ packets."""
    r1 = tmp_path / "r1"
    for sub in ("lanes", "verify", "packets"):
        (r1 / sub).mkdir(parents=True)
    (r1 / "PIN.md").write_text("Round: 1\n")
    (r1 / "owners.tsv").write_text("CORR1\ta.cpp\nCORR1\tb.bin\n")
    (r1 / "numstat.txt").write_text("3\t2\ta.cpp\n-\t-\tb.bin\n")
    (r1 / "packets" / "CORR1.md").write_text("p")
    (r1 / "packets" / "TESTQ.md").write_text("p")
    return tmp_path


def run(ledger, sub, ev, rnd=1, *extra):
    return ledger.main([sub, str(ev), str(rnd), *extra])


def table(rl, ev, title):
    return rl.find_table(rl.parse_tables((ev / "LEDGER.md").read_text()), title)


def seeded(ledger, ev):
    assert run(ledger, "seed", ev) == 0
    return ev


def with_reports(ledger, ev, **reports):
    for lane, text in reports.items():
        (ev / "r1" / "lanes" / f"{lane.replace('_', '-')}.md").write_text(text)


def test_finding_block_parses_multiline_fenced_actual(rl):
    text = report(
        "CORR1",
        [
            finding(
                "CORR1-p1", actual="First line.\n```\ncode\n\nmore code\n```\nafter"
            )
            + "\n"
            + finding("CORR1-p2")
        ],
    )
    blocks = rl.parse_findings(text)
    assert [b["ID"] for b in blocks] == ["CORR1-p1", "CORR1-p2"]
    assert blocks[0]["Actual"] == "First line.\n```\ncode\n\nmore code\n```\nafter"
    assert blocks[0]["Fix"] == "clamp it" and blocks[1]["Tier"] == "IMPORTANT"


def test_tables_round_trip_with_pipes_and_reject_foreign_rows(rl):
    t = rl.Table("Findings", ["ID", "Claim"], [{"ID": "X-1", "Claim": "a | b"}])
    parsed = rl.parse_tables(rl.render_tables([t]))
    assert parsed[0].rows == [{"ID": "X-1", "Claim": "a | b"}]
    with pytest.raises(rl.Fail):
        rl.parse_tables("## T\n\nfree text\n")


def test_seed_rows_and_round_two_appends_keeping_rows(ledger, rl, ev, capsys):
    seeded(ledger, ev)
    cov = table(rl, ev, "Coverage (round 1)").rows
    assert [(r["File"], r["Lines"], r["Owner lane"], r["Read depth"]) for r in cov] == [
        ("a.cpp", "5", "CORR1", "pending"),
        ("b.bin", "0", "CORR1", "pending"),
    ]
    assert [r["Lane"] for r in table(rl, ev, "Lanes (round 1)").rows] == [
        "CORR1",
        "TESTQ",
    ]
    assert run(ledger, "seed", ev) == 1  # round 1 already seeded
    r2 = ev / "r2"
    (r2 / "packets").mkdir(parents=True)
    (r2 / "PIN.md").write_text("Round: 2\n")
    (r2 / "owners.tsv").write_text("CORR1\ta.cpp\n")
    (r2 / "numstat.txt").write_text("1\t1\ta.cpp\n")
    (ev / "r1" / "lanes" / "CORR1.md").write_text(
        report("CORR1", [finding("CORR1-p1")])
    )
    run(ledger, "ids", ev)
    assert run(ledger, "seed", ev, 2, "--lane", "BUILD=N/A: no cmake") == 0
    assert [r["ID"] for r in table(rl, ev, "Findings").rows] == ["CORR-1"]
    assert [r["File"] for r in table(rl, ev, "Coverage (round 1)").rows] == [
        "a.cpp",
        "b.bin",
    ]
    assert [
        (r["Lane"], r["Status"]) for r in table(rl, ev, "Lanes (round 2)").rows
    ] == [("BUILD", "N/A: no cmake")]


def test_final_ids_are_stable_and_shared_across_correctness_lanes(ledger, rl, ev):
    seeded(ledger, ev)
    with_reports(
        ledger,
        ev,
        CORR1=report("CORR1", [finding("CORR1-p1"), finding("CORR1-p2")]),
        CORR2=report("CORR2", [finding("CORR2-p1")]),
        TESTQ=report("TESTQ", [finding("TESTQ-p1", kind="test-gap")]),
    )
    run(ledger, "ids", ev)
    first = (ev / "r1" / "ID-MAP.md").read_text()
    assert [r["Final ID"] for r in rl.parse_tables(first)[0].rows] == [
        "CORR-1",
        "CORR-2",
        "CORR-3",
        "TESTQ-1",
    ]
    run(ledger, "ids", ev)
    assert (ev / "r1" / "ID-MAP.md").read_text() == first
    # A follow-up and a later report continue the counters without renumbering.
    with_reports(ledger, ev, CORR1_f1=report("CORR1-f1", [finding("CORR1-p3")]))
    run(ledger, "ids", ev)
    rows = {
        r["Lane / provisional"].split("/")[1]: r["Final ID"]
        for r in rl.parse_tables((ev / "r1" / "ID-MAP.md").read_text())[0].rows
    }
    assert (
        rows["CORR1-p1"] == "CORR-1"
        and rows["CORR2-p1"] == "CORR-3"
        and rows["CORR1-p3"] == "CORR-4"
    )
    assert len(table(rl, ev, "Findings").rows) == 5


def test_understated_known_item_and_supersede_flow_into_ledger(ledger, rl, ev):
    seeded(ledger, ev)
    with_reports(
        ledger,
        ev,
        CORR1=report(
            "CORR1",
            [
                finding("CORR1-p1", extra="understated: known-open-item K2"),
                finding("CORR1-p2"),
            ],
        ),
    )
    run(ledger, "ids", ev)
    mp = (ev / "r1" / "ID-MAP.md").read_text()
    (ev / "r1" / "ID-MAP.md").write_text(
        mp.replace("| CORR-2 |  |", "| CORR-2 | CORR-1 |")
    )
    run(ledger, "ids", ev)
    rows = {r["ID"]: r for r in table(rl, ev, "Findings").rows}
    assert rows["CORR-1"]["known-open-item"] == "K2"
    assert rows["CORR-2"]["Status"] == "SUPERSEDED-BY:CORR-1"
    assert (
        rows["CORR-1"]["Claim"] == "It returns the wrong value."
        and rows["CORR-1"]["Tier"] == "IMPORTANT -> IMPORTANT"
    )


def test_seed_skips_blank_numstat_lines(ledger, rl, ev):
    (ev / "r1" / "numstat.txt").write_text("3\t2\ta.cpp\n\n-\t-\tb.bin\n\n")
    assert run(ledger, "seed", ev) == 0
    assert len(table(rl, ev, "Coverage (round 1)").rows) == 2


@pytest.mark.parametrize("tier", ["Blocking", ""])
def test_ids_rejects_a_missing_or_unknown_tier(ledger, ev, capsys, tier):
    seeded(ledger, ev)
    with_reports(ledger, ev, CORR1=report("CORR1", [finding("CORR1-p1", tier=tier)]))
    capsys.readouterr()
    assert run(ledger, "ids", ev) == 1
    err = capsys.readouterr().err
    assert "CORR1.md: CORR1-p1: Tier" in err


def test_claim_is_the_first_sentence_of_the_whole_actual(ledger, rl, ev):
    seeded(ledger, ev)
    long = "word " * 80
    with_reports(
        ledger,
        ev,
        CORR1=report(
            "CORR1",
            [
                finding(
                    "CORR1-p1",
                    actual="The loop never exits when\nn is zero and the buffer is empty."
                    " Later text.",
                ),
                finding("CORR1-p2", actual=long),
            ],
        ),
    )
    assert run(ledger, "ids", ev) == 0
    rows = {r["ID"]: r for r in table(rl, ev, "Findings").rows}
    assert rows["CORR-1"]["Claim"] == (
        "The loop never exits when n is zero and the buffer is empty."
    )
    claim = rows["CORR-2"]["Claim"]
    assert claim.endswith("word…") and len(claim) <= 201


def test_coverage_copies_owner_lane_depth_and_tests(ledger, rl, ev, capsys):
    seeded(ledger, ev)
    with_reports(
        ledger,
        ev,
        CORR1=report(
            "CORR1",
            coverage=[("a.cpp", "full", "t1 | t2"), ("unchanged.h", "full", "-")],
        ),
        TESTQ=report("TESTQ", coverage=[("b.bin", "sampled", "-")]),
    )
    capsys.readouterr()
    assert run(ledger, "coverage", ev) == 0
    rows = {r["File"]: r for r in table(rl, ev, "Coverage (round 1)").rows}
    assert (rows["a.cpp"]["Read depth"], rows["a.cpp"]["Tests reaching it"]) == (
        "full",
        "t1 | t2",
    )
    assert rows["b.bin"]["Read depth"] == "pending"  # TESTQ does not own it
    assert "pending: b.bin" in capsys.readouterr().out


def make_candidates(ledger, ev, *tiers):
    seeded(ledger, ev)
    with_reports(
        ledger,
        ev,
        CORR1=report(
            "CORR1",
            [
                finding(f"CORR1-p{i}", tier=t, actual=f"claim {i}.")
                for i, t in enumerate(tiers, 1)
            ],
        ),
    )
    run(ledger, "ids", ev)


def write_verify(ev, fid, *args, second=False, **kw):
    name = f"{fid}-2" if second else fid
    (ev / "r1" / "verify" / f"{name}.md").write_text(verify_file(fid, *args, **kw))


def test_verdicts_status_tier_and_narrowed_claim(ledger, rl, ev, capsys):
    make_candidates(ledger, ev, "IMPORTANT", "IMPORTANT", "BLOCKING", "SUGGESTION")
    write_verify(ev, "CORR-1", "PARTIAL", "IMPORTANT", claim="Only wrong for n=0.")
    write_verify(ev, "CORR-2", "REFUTED", "IMPORTANT")
    write_verify(ev, "CORR-3", "DOWNGRADED", "SUGGESTION")
    capsys.readouterr()
    assert run(ledger, "verdicts", ev) == 1
    out = capsys.readouterr().out
    rows = {r["ID"]: r for r in table(rl, ev, "Findings").rows}
    assert (rows["CORR-1"]["Status"], rows["CORR-1"]["Claim"]) == (
        "OPEN",
        "Only wrong for n=0.",
    )
    assert (
        rows["CORR-2"]["Status"] == "REFUTED"
        and rows["CORR-3"]["Tier"] == "BLOCKING -> SUGGESTION"
    )
    assert rows["CORR-4"]["Status"] == "" and "missing: CORR-4" in out
    # PARTIAL that keeps its side of the IMPORTANT/SUGGESTION line needs no second verifier.
    assert "CORR-1" not in out
    assert "second verifier needed: CORR-2 (REFUTED IMPORTANT)" in out
    assert "second verifier needed: CORR-3 (tier BLOCKING -> SUGGESTION)" in out


def test_verdicts_exit_1_only_while_a_finding_is_pending(ledger, ev, capsys):
    make_candidates(ledger, ev, "IMPORTANT", "IMPORTANT")
    write_verify(ev, "CORR-1", "CONFIRMED", "IMPORTANT")
    capsys.readouterr()
    assert run(ledger, "verdicts", ev) == 1  # CORR-2 has no verify file
    assert "missing: CORR-2" in capsys.readouterr().out
    write_verify(ev, "CORR-2", "REFUTED", "IMPORTANT")
    assert (
        run(ledger, "verdicts", ev) == 1
    )  # a REFUTED IMPORTANT needs a second verifier
    assert "second verifier needed: CORR-2" in capsys.readouterr().out
    write_verify(ev, "CORR-2", "REFUTED", "IMPORTANT", second=True)
    assert run(ledger, "verdicts", ev) == 0


def test_second_verifier_replaces_differing_verdict_and_is_not_requested_again(
    ledger, rl, ev, capsys
):
    make_candidates(ledger, ev, "IMPORTANT", "SUGGESTION")
    write_verify(ev, "CORR-1", "REFUTED", "IMPORTANT")
    write_verify(ev, "CORR-1", "CONFIRMED", "IMPORTANT", second=True)
    write_verify(ev, "CORR-2", "UPGRADED", "IMPORTANT")
    write_verify(ev, "CORR-2", "UPGRADED", "IMPORTANT", second=True)
    capsys.readouterr()
    run(ledger, "verdicts", ev)
    rows = {r["ID"]: r for r in table(rl, ev, "Findings").rows}
    assert (rows["CORR-1"]["Verdict"], rows["CORR-1"]["Status"]) == (
        "CONFIRMED",
        "OPEN",
    )
    assert rows["CORR-2"]["Tier"] == "SUGGESTION -> IMPORTANT"
    assert "second verifier" not in capsys.readouterr().out
    assert run(ledger, "verdicts", ev) == 0  # idempotent


def test_verdicts_reject_nonconforming_verify_file(ledger, ev, capsys):
    make_candidates(ledger, ev, "IMPORTANT")
    write_verify(ev, "CORR-1", "PARTIAL", "IMPORTANT")  # no narrowed claim
    assert run(ledger, "verdicts", ev) == 1
    assert "PARTIAL without Claim" in capsys.readouterr().out


def test_mlines_collect_fail_missing_and_vacuous_pass(ledger, ev, capsys):
    seeded(ledger, ev)
    with_reports(
        ledger,
        ev,
        TESTQ=report(
            "TESTQ",
            mlines=[
                "M1: FAIL no regression test for the fix",
                "M3: WAIVED(W-HOTFIX) tracked",
            ],
        ),
        TESTB="# TESTB (round 1, abc)\nN/A: no product code\n",
        AUTH=report("AUTH", mlines=["- M4: UNVERIFIED tracker unreachable"]),
    )
    assert run(ledger, "mlines", ev) == 1
    out = capsys.readouterr().out
    assert "file ORCH process BLOCKING: M1 FAIL: no regression test for the fix" in out
    assert "M3 WAIVED(W-HOTFIX)" in out and "M2 PASS (TESTB N/A, vacuous)" in out
    assert "file ORCH process (tier per policy M-rules): M4 UNVERIFIED" in out
    assert "missing: M5 (AUTH: no M-line)" in out


def test_mline_status_may_be_followed_by_punctuation(rl):
    text = report(
        "TESTQ",
        mlines=["M1: PASS. ran the suite", "M2: FAIL; no test", "M3: N/A."],
    )
    assert rl.parse_mlines(text) == [
        (1, "PASS", "ran the suite"),
        (2, "FAIL", "no test"),
        (3, "N/A", ""),
    ]


def test_mlines_complete_report_succeeds(ledger, ev):
    seeded(ledger, ev)
    with_reports(
        ledger,
        ev,
        TESTQ=report("TESTQ", mlines=["M1: PASS ok", "M3: PASS ok"]),
        TESTB=report("TESTB", mlines=["M2: PASS ok"]),
        AUTH=report("AUTH", mlines=["M4: N/A (no PR)", "M5: N/A (no PR)"]),
    )
    assert run(ledger, "mlines", ev) == 0


def test_counts_waits_on_open_questions_and_undecided_decisions(ledger, rl, ev, capsys):
    seeded(ledger, ev)
    tables = rl.parse_tables((ev / "LEDGER.md").read_text())
    rl.find_table(tables, "Decisions").add(
        Round="1", Question="Ship it?", Decision="", **{"Findings affected": ""}
    )
    rl.find_table(tables, "Decisions").add(
        Round="1", Question="Rename it?", Decision="yes", **{"Findings affected": ""}
    )
    rl.find_table(tables, "Open questions").add(ID="Q1", Question="Which API?")
    (ev / "LEDGER.md").write_text(rl.render_tables(tables))
    capsys.readouterr()
    run(ledger, "counts", ev)
    closure = capsys.readouterr().out.splitlines()[-1]
    assert closure == "Closure: OPEN; waiting: question Q1, decision Ship it?"


def test_counts_closure_and_known_open_item(ledger, rl, ev, capsys):
    make_candidates(ledger, ev, "BLOCKING", "IMPORTANT", "SUGGESTION")
    capsys.readouterr()
    run(ledger, "counts", ev)
    out = capsys.readouterr().out
    assert (
        "BLOCKING: OPEN 0 UNVERIFIED 0 candidate 1" in out
        and "Closure: OPEN; waiting: CORR-1" in out
    )
    write_verify(ev, "CORR-1", "UNVERIFIED", "BLOCKING")
    write_verify(ev, "CORR-2", "REFUTED", "IMPORTANT")
    write_verify(ev, "CORR-3", "CONFIRMED", "SUGGESTION")
    run(ledger, "verdicts", ev)
    capsys.readouterr()
    run(ledger, "counts", ev)
    out = capsys.readouterr().out
    assert (
        "BLOCKING: OPEN 0 UNVERIFIED 1" in out
        and "Closure: OPEN; waiting: CORR-1 (BLOCKING UNVERIFIED)" in out
    )
    ft = table(rl, ev, "Findings")
    ft.rows[0]["Status"] = "ACCEPTED-RISK"
    ft.rows[0]["known-open-item"] = "K1"
    tables = rl.parse_tables((ev / "LEDGER.md").read_text())
    (ev / "LEDGER.md").write_text(rl.render_tables(tables[:1] + [ft] + tables[2:]))
    capsys.readouterr()
    run(ledger, "counts", ev)
    out = capsys.readouterr().out
    assert (
        "known-open-item: CORR-1 K1 (ACCEPTED-RISK)" in out and "Closure: CLOSED" in out
    )


def test_failed_lane_keeps_closure_open(ledger, ev, capsys):
    assert run(ledger, "seed", ev, 1, "--lane", "RUN=FAILED (timed out)") == 0
    capsys.readouterr()
    run(ledger, "counts", ev)
    assert (
        "Closure: OPEN; waiting: lane RUN (FAILED (timed out))"
        in capsys.readouterr().out
    )


def test_set_closes_accepts_and_reopens_and_verdicts_skip_closed_rows(
    ledger, rl, ev, capsys
):
    make_candidates(ledger, ev, "IMPORTANT", "BLOCKING")
    write_verify(ev, "CORR-1", "CONFIRMED", "IMPORTANT")
    write_verify(ev, "CORR-2", "CONFIRMED", "BLOCKING")
    run(ledger, "verdicts", ev)
    assert (
        run(
            ledger,
            "set",
            ev,
            2,
            "CORR-1",
            "FIXED@abcdef1",
            "--class",
            "rerun",
            "--note",
            "passes at head",
        )
        == 0
    )
    assert (
        run(
            ledger,
            "set",
            ev,
            2,
            "CORR-2",
            "ACCEPTED-RISK",
            "--note",
            "owner accepts: GPU run not available",
        )
        == 0
    )
    rows = {r["ID"]: r for r in table(rl, ev, "Findings").rows}
    assert (rows["CORR-1"]["Status"], rows["CORR-1"]["Round closed"]) == (
        "FIXED@abcdef1",
        "2",
    )
    assert rows["CORR-1"]["Evidence"] == "rerun: passes at head"
    decision = table(rl, ev, "Decisions").rows[0]
    assert (
        decision["Findings affected"] == "CORR-2" and "GPU run" in decision["Decision"]
    )
    run(ledger, "verdicts", ev)
    rows = {r["ID"]: r for r in table(rl, ev, "Findings").rows}
    assert (
        rows["CORR-1"]["Status"] == "FIXED@abcdef1"
        and rows["CORR-2"]["Status"] == "ACCEPTED-RISK"
    )
    capsys.readouterr()
    run(ledger, "counts", ev)
    assert "Closure: CLOSED" in capsys.readouterr().out
    run(ledger, "set", ev, 3, "CORR-1", "OPEN")
    rows = {r["ID"]: r for r in table(rl, ev, "Findings").rows}
    assert (rows["CORR-1"]["Status"], rows["CORR-1"]["Round closed"]) == ("OPEN", "")
    assert rows["CORR-1"]["Verdict"] == ""
    (ev / "r1" / "verify" / "CORR-1.md").unlink()
    capsys.readouterr()
    run(ledger, "verdicts", ev)
    assert "missing: CORR-1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ("CORR-1", "FIXED@abcdef1"),
        ("CORR-1", "ACCEPTED-RISK"),
        ("CORR-1", "DONE"),
        ("CORR-9", "OPEN"),
    ],
)
def test_set_rejects_incomplete_or_unknown(ledger, ev, args, capsys):
    make_candidates(ledger, ev, "IMPORTANT")
    capsys.readouterr()
    assert run(ledger, "set", ev, 1, *args) == 1


def test_mlines_newest_follow_up_wins(ledger, ev, capsys):
    seeded(ledger, ev)
    lanes = ev / "r1" / "lanes"
    (lanes / "TESTQ.md").write_text(
        report("TESTQ", mlines=["M1: UNVERIFIED not run", "M3: PASS ok"])
    )
    (lanes / "TESTQ-f1.md").write_text(
        report("TESTQ-f1", mlines=["M1: PASS fails at base"])
    )
    (lanes / "TESTB.md").write_text(report("TESTB", mlines=["M2: PASS covered"]))
    (lanes / "AUTH.md").write_text(report("AUTH", mlines=["M4: PASS t", "M5: PASS t"]))
    capsys.readouterr()
    assert run(ledger, "mlines", ev) == 0
    out = capsys.readouterr().out
    assert "M1 PASS (TESTQ-f1)" in out and "UNVERIFIED" not in out


def test_ids_maps_verifier_batch_file_next_to_a_final_id_verdict(ledger, rl, ev):
    make_candidates(ledger, ev, "IMPORTANT")
    write_verify(ev, "CORR-1", "CONFIRMED", "IMPORTANT")
    (ev / "r1" / "verify" / "VER-1.md").write_text(
        verify_file("VER-1", "CONFIRMED", "IMPORTANT")
    )
    (ev / "r1" / "verify" / "VER1.md").write_text(
        report("VER1", [finding("VER1-p1", where="b.bin:1")])
    )
    assert run(ledger, "ids", ev) == 0
    ids = [r["ID"] for r in table(rl, ev, "Findings").rows]
    assert ids == ["CORR-1", "VER-1"]
    assert run(ledger, "verdicts", ev) == 0


FENCED = "```\n{}```\n"


def test_fenced_finding_blocks_are_mapped_and_nested_code_survives(rl):
    first = finding("CORR1-p1", actual="Breaks:\n```cpp\nint x;\n```\nafter")
    text = report(
        "CORR1",
        [FENCED.format(first + "\n" + finding("CORR1-p2")), finding("CORR1-p3")],
    )
    blocks = rl.parse_findings(text)
    assert [b["ID"] for b in blocks] == ["CORR1-p1", "CORR1-p2", "CORR1-p3"]
    assert blocks[0]["Actual"] == "Breaks:\n```cpp\nint x;\n```\nafter"
    assert "```" not in blocks[1]["Fix"] and blocks[1]["Fix"] == "clamp it"
    assert rl.unparsed_ids(text) == 0


def test_ids_fails_on_id_lines_it_cannot_parse(ledger, ev, capsys):
    seeded(ledger, ev)
    odd = "```text\nThe finding follows.\n" + finding("CORR1-p1") + "```\n"
    with_reports(ledger, ev, CORR1=report("CORR1", [odd]))
    assert run(ledger, "ids", ev) == 1
    assert "CORR1.md" in capsys.readouterr().err


def test_coverage_fails_on_annotated_depth_of_an_owned_changed_file(
    ledger, rl, ev, capsys
):
    seeded(ledger, ev)
    with_reports(
        ledger,
        ev,
        CORR1=report(
            "CORR1",
            coverage=[("a.cpp", "full (skimmed)", "-"), ("unchanged.h", "weird", "-")],
        ),
    )
    assert run(ledger, "coverage", ev) == 1
    captured = capsys.readouterr()
    assert "non-conforming: CORR1: a.cpp" in captured.out
    assert "unchanged.h" not in captured.out


@pytest.mark.parametrize("sub", ["ids", "coverage"])
def test_gates_fail_on_a_report_below_a_nested_round_directory(ledger, ev, sub, capsys):
    seeded(ledger, ev)
    nested = ev / "r1" / "r1" / "lanes"
    nested.mkdir(parents=True)
    (nested / "CORR1.md").write_text(report("CORR1"))
    assert run(ledger, sub, ev) == 1
    assert "r1/r1/lanes/CORR1.md" in capsys.readouterr().err.replace("\\", "/")
