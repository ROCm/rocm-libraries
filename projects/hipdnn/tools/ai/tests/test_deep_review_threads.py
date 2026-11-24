# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hipdnn-deep-review PR thread tracking: ledger.py link, pin.py thread intake and
freshness, comments.py universe and the open-thread section of context packets.

GitHub is replaced by a stub of `pin.gh` serving synthetic GraphQL pages."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parent.parent / "skills" / "hipdnn-deep-review" / "scripts"
)
URL = "https://github.com/example/repo/pull/7#discussion_r{}"


def _load(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            f"deep_review_threads_{name}", SCRIPTS / f"{name}.py"
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


@pytest.fixture(scope="module")
def comments():
    return _load("comments")


@pytest.fixture(scope="module")
def part():
    return _load("partition")


@pytest.fixture
def pin():
    return _load("pin")


def comment(n, login, when, body="text"):
    return {
        "author": {"login": login},
        "createdAt": when,
        "updatedAt": when,
        "originalCommit": {"oid": "a" * 40},
        "url": URL.format(n),
        "body": body,
    }


def thread(n, resolved=False, replies=(), root="reviewer", when="2026-01-01T10:00:00Z"):
    """A thread rooted at comment n; replies are (login, time) pairs."""
    nodes = [comment(n, root, when, "first line\nsecond line")]
    nodes += [comment(n * 100 + i, login, t) for i, (login, t) in enumerate(replies)]
    return {
        "isResolved": resolved,
        "isOutdated": False,
        "path": "a.cpp",
        "originalLine": 10 + n,
        "originalStartLine": None,
        "resolvedBy": {"login": "reviewer"} if resolved else None,
        "comments": {"pageInfo": {"hasNextPage": False}, "nodes": nodes},
    }


class FakeGh:
    """Stub for pin.gh: serves threads one per GraphQL page, the PR's base and head, and
    the authenticated login."""

    def __init__(self, threads=(), base="develop", head="a" * 40, errors=None):
        self.threads, self.base, self.head, self.errors = (
            list(threads),
            base,
            head,
            errors,
        )
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("api", "user"):
            return "reviewer\n"
        if args[:2] == ("pr", "view"):
            return json.dumps({"baseRefName": self.base, "headRefOid": self.head})
        assert args[:2] == ("api", "graphql"), args
        if self.errors:
            return json.dumps({"errors": [{"message": self.errors}]})
        cursor = next((a[7:] for a in args if a.startswith("cursor=")), None)
        i = int(cursor) if cursor else 0
        more = i + 1 < len(self.threads)
        return json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "author": {"login": "author"},
                            "reviewThreads": {
                                "pageInfo": {
                                    "hasNextPage": more,
                                    "endCursor": str(i + 1),
                                },
                                "nodes": self.threads[i : i + 1],
                            },
                        }
                    }
                }
            }
        )


FINDING_ROWS = [
    # id, tier, kind, status, claim, where
    ("CORR-1", "IMPORTANT", "defect", "OPEN", "claim one", "a.cpp:11"),
    ("CORR-2", "IMPORTANT", "defect", "OPEN", "claim two", "a.cpp:12"),
    ("DOCS-1", "SUGGESTION", "source-claim", "FIXED@abc1234", "docs one", "a.md:1"),
    ("API-1", "BLOCKING", "defect", "UNVERIFIED", "api one", "b.hpp:3"),
    ("TESTQ-1", "SUGGESTION", "test-gap", "OPEN", "test one", "t.cpp:4"),
    ("DUP-1", "IMPORTANT", "defect", "REFUTED", "dup one", "a.cpp:11"),
]


def make_ledger(ev, rl, ledger):
    """A LEDGER.md with FINDING_ROWS and no Threads table (a ledger of an older round)."""
    findings = rl.Table("Findings", ledger.FINDING_COLS)
    for fid, tier, kind, status, claim, where in FINDING_ROWS:
        findings.add(
            **{
                "ID": fid,
                "Tier": f"{tier} -> {tier}",
                "Kind": kind,
                "Status": status,
                "Claim": claim,
                "Where": where,
                "Evidence": "proof",
                "Round opened": "1",
            }
        )
    decisions = rl.Table("Decisions", ledger.DECISION_COLS)
    (ev / "LEDGER.md").write_text(rl.render_tables([findings, decisions]))


@pytest.fixture
def ev(tmp_path, rl, ledger):
    """Evidence with rounds r1 (PIN.md with Pinned) and r2, and a ledger without Threads."""
    make_ledger(tmp_path, rl, ledger)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "PIN.md").write_text(
        "Round: 1\nPinned: 2026-02-01T00:00:00Z\nReviewer: reviewer\nHEAD: "
        + "a" * 40
        + "\nTARGET: develop\nKEY: pr7\n"
    )
    (tmp_path / "r2").mkdir()
    return tmp_path


def threads_rows(rl, ev):
    tables = rl.parse_tables((ev / "LEDGER.md").read_text())
    return [r for t in tables if t.title == "Threads" for r in t.rows]


def posted(ev, *lines):
    path = ev / "posted.tsv"
    path.write_text("".join(f"{line}\n" for line in lines))
    return str(path)


def link_from(ledger, ev, *lines, rnd=1):
    return ledger.main(["link", str(ev), str(rnd), "--from", posted(ev, *lines)])


# ledger.py link -------------------------------------------------------------


def test_posted_tsv_links_every_id_and_an_old_ledger_gains_the_table(ledger, rl, ev):
    assert "## Threads" not in (ev / "LEDGER.md").read_text()
    assert (
        link_from(
            ledger, ev, f"{URL.format(1)}\tCORR-1", f"{URL.format(2)}\tCORR-2,DOCS-1"
        )
        == 0
    )
    rows = threads_rows(rl, ev)
    assert [(r["Thread URL"], r["Finding"]) for r in rows] == [
        (URL.format(1), "CORR-1"),
        (URL.format(2), "CORR-2"),
        (URL.format(2), "DOCS-1"),
    ]
    assert {
        (r["Opened by"], r["State"], r["Last reply"], r["Round seen"]) for r in rows
    } == {("review", "", "", "1")}
    # The findings table is untouched.
    findings = [
        t
        for t in rl.parse_tables((ev / "LEDGER.md").read_text())
        if t.title == "Findings"
    ]
    assert len(findings[0].rows) == len(FINDING_ROWS)


def test_link_one_comment_form_and_relinking_adds_no_rows(ledger, rl, ev):
    assert ledger.main(["link", str(ev), "1", URL.format(1), "CORR-1,CORR-2"]) == 0
    assert len(threads_rows(rl, ev)) == 2
    before = (ev / "LEDGER.md").read_text()
    assert link_from(ledger, ev, f"{URL.format(1)}\tCORR-1,CORR-2", rnd=2) == 0
    assert link_from(ledger, ev, f"{URL.format(1)}\tCORR-1,CORR-2", rnd=2) == 0
    assert (ev / "LEDGER.md").read_text() == before


def test_link_rejects_unknown_ids_naming_them_and_writes_nothing(ledger, ev, capsys):
    before = (ev / "LEDGER.md").read_text()
    assert (
        link_from(
            ledger, ev, f"{URL.format(1)}\tCORR-1", f"{URL.format(2)}\tNOPE-9,GONE-1"
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "GONE-1" in err and "NOPE-9" in err
    assert (ev / "LEDGER.md").read_text() == before


@pytest.mark.parametrize(
    "bad",
    [
        "https://api.github.com/repos/example/repo/pulls/comments/123",
        "https://github.com/example/repo/pull/7",
        "https://github.com/example/repo/pull/7#issuecomment-5",
    ],
)
def test_link_rejects_non_web_comment_urls(ledger, ev, capsys, bad):
    before = (ev / "LEDGER.md").read_text()
    assert link_from(ledger, ev, f"{bad}\tCORR-1") == 1
    assert bad in capsys.readouterr().err
    assert (ev / "LEDGER.md").read_text() == before


def test_seed_adds_the_threads_table_to_an_old_ledger(ledger, rl, ev):
    (ev / "r2" / "owners.tsv").write_text("CORR1\ta.cpp\n")
    (ev / "r2" / "numstat.txt").write_text("1\t0\ta.cpp\n")
    (ev / "r2" / "PIN.md").write_text("Round: 2\n")
    assert ledger.main(["seed", str(ev), "2"]) == 0
    assert any(
        t.title == "Threads" for t in rl.parse_tables((ev / "LEDGER.md").read_text())
    )


# thread intake ---------------------------------------------------------------


def refetch(pin, ev, monkeypatch, fake, rnd=1, *extra):
    monkeypatch.setattr(pin, "gh", fake)
    pin.main(
        [
            "threads",
            "--evidence",
            str(ev),
            "--round",
            str(rnd),
            "--owner-repo",
            "example/repo",
            "--pr",
            "7",
            *extra,
        ]
    )


def test_intake_fills_state_and_roles_across_pages_and_keeps_logins_out_of_the_ledger(
    pin, ledger, rl, ev, monkeypatch, capsys
):
    link_from(ledger, ev, f"{URL.format(1)}\tCORR-1", f"{URL.format(2)}\tCORR-2")
    fake = FakeGh(
        [
            thread(1, replies=[("author", "2026-03-05T09:00:00Z")]),
            thread(
                2,
                resolved=True,
                replies=[
                    ("other", "2026-03-06T09:00:00Z"),
                    ("reviewer", "2026-03-07T09:00:00Z"),
                ],
            ),
            thread(3, root="other"),
        ]
    )
    refetch(pin, ev, monkeypatch, fake, 1, "--reviewer", "reviewer")
    rows = {r["Finding"]: r for r in threads_rows(rl, ev)}
    assert (
        rows["CORR-1"]["State"],
        rows["CORR-1"]["Last reply"],
        rows["CORR-1"]["Round seen"],
    ) == (
        "unresolved",
        "author 2026-03-05",
        "1",
    )
    assert (rows["CORR-2"]["State"], rows["CORR-2"]["Last reply"]) == (
        "resolved",
        "review 2026-03-07",
    )
    ledger_text = (ev / "LEDGER.md").read_text()
    assert (
        "first line" not in ledger_text and "https://github.com/example" in ledger_text
    )
    out = capsys.readouterr().out
    assert f"unlinked: {URL.format(3)}" in out
    assert "unlinked: " + URL.format(1) not in out
    listing = (ev / "r1" / "threads.md").read_text()
    assert (
        URL.format(3) in listing and "other" in listing
    )  # all threads, logins allowed
    saved = json.loads((ev / "r1" / "pr" / "threads.json").read_text())
    assert len(saved["threads"]) == 3 and saved["reviewer"] == "reviewer"


def test_intake_never_changes_a_finding(pin, ledger, rl, ev, monkeypatch):
    link_from(ledger, ev, f"{URL.format(1)}\tCORR-1", f"{URL.format(2)}\tDOCS-1")
    before = [
        t
        for t in rl.parse_tables((ev / "LEDGER.md").read_text())
        if t.title == "Findings"
    ][0].rows
    refetch(
        pin,
        ev,
        monkeypatch,
        FakeGh([thread(1, resolved=True), thread(2)]),
        1,
        "--reviewer",
        "reviewer",
    )
    after = [
        t
        for t in rl.parse_tables((ev / "LEDGER.md").read_text())
        if t.title == "Findings"
    ][0].rows
    assert after == before


def test_graphql_failure_fails_loudly_and_writes_nothing(pin, ev, monkeypatch):
    with pytest.raises(SystemExit) as e:
        refetch(
            pin,
            ev,
            monkeypatch,
            FakeGh(errors="Resource not accessible"),
            1,
            "--reviewer",
            "reviewer",
        )
    assert "Resource not accessible" in str(e.value)
    assert not (ev / "r1" / "pr").exists()


def test_a_thread_with_over_100_comments_fails_loudly(pin, ev, monkeypatch):
    big = thread(1)
    big["comments"]["pageInfo"]["hasNextPage"] = True
    with pytest.raises(SystemExit) as e:
        refetch(pin, ev, monkeypatch, FakeGh([big]), 1, "--reviewer", "reviewer")
    assert "over 100 comments" in str(e.value)


def test_reviewer_is_required_when_the_round_has_none(pin, ev, monkeypatch):
    (ev / "r1" / "PIN.md").write_text("Round: 1\n")
    with pytest.raises(SystemExit) as e:
        refetch(pin, ev, monkeypatch, FakeGh([thread(1)]), 1)
    assert "--reviewer" in str(e.value)


def test_reviewer_comes_from_the_round_pin(pin, ev, monkeypatch):
    (ev / "r1" / "PIN.md").write_text(
        "Round: 1\nPinned: 2026-02-01T00:00:00Z\nReviewer: reviewer\n"
    )
    refetch(pin, ev, monkeypatch, FakeGh([thread(1)]), 1)
    assert json.loads((ev / "r1/pr/threads.json").read_text())["reviewer"] == "reviewer"


def affected(ev, rnd=1):
    return (ev / f"r{rnd}" / "reply-affected.txt").read_text().split()


def test_reply_affected_lists_exactly_linked_findings_with_a_new_reply(
    pin, ledger, ev, monkeypatch
):
    link_from(
        ledger,
        ev,
        f"{URL.format(1)}\tCORR-1",  # reply by author after Pinned
        f"{URL.format(2)}\tCORR-2",  # reply only before Pinned
        f"{URL.format(3)}\tDOCS-1,API-1",  # reply by the reviewer after Pinned
        f"{URL.format(4)}\tTESTQ-1",  # reply by other after Pinned
    )
    fake = FakeGh(
        [
            thread(1, replies=[("author", "2026-02-02T00:00:00Z")]),
            thread(2, replies=[("author", "2026-01-31T23:59:59Z")]),
            thread(3, replies=[("reviewer", "2026-02-05T00:00:00Z")]),
            thread(4, replies=[("other", "2026-02-03T00:00:00Z")]),
            thread(5, replies=[("author", "2026-02-04T00:00:00Z")]),  # unlinked
        ]
    )
    refetch(pin, ev, monkeypatch, fake, 1, "--reviewer", "reviewer")
    assert affected(ev) == ["CORR-1", "TESTQ-1"]


def test_reply_detection_needs_since_when_the_round_has_no_pinned(
    pin, ledger, ev, monkeypatch
):
    (ev / "r1" / "PIN.md").write_text("Round: 1\nReviewer: reviewer\n")
    link_from(ledger, ev, f"{URL.format(1)}\tCORR-1")
    fake = FakeGh([thread(1, replies=[("author", "2026-02-02T10:00:00Z")])])
    with pytest.raises(SystemExit) as e:
        refetch(pin, ev, monkeypatch, fake, 1, "--reviewer", "reviewer")
    assert "--since" in str(e.value)
    refetch(
        pin, ev, monkeypatch, fake, 1, "--reviewer", "reviewer", "--since", "2026-02-03"
    )
    assert affected(ev) == []
    refetch(
        pin, ev, monkeypatch, fake, 1, "--reviewer", "reviewer", "--since", "2026-02-02"
    )
    assert affected(ev) == ["CORR-1"]


def test_since_overrides_pinned_and_a_published_only_round_needs_it(
    pin, ledger, ev, monkeypatch
):
    (ev / "r1" / "PIN.md").unlink()  # a published-only round has no PIN.md
    link_from(ledger, ev, f"{URL.format(1)}\tCORR-1")
    fake = FakeGh([thread(1, replies=[("author", "2026-02-02T10:00:00Z")])])
    with pytest.raises(SystemExit):
        refetch(pin, ev, monkeypatch, fake, 1, "--reviewer", "reviewer")
    refetch(
        pin,
        ev,
        monkeypatch,
        fake,
        1,
        "--reviewer",
        "reviewer",
        "--since",
        "2026-02-02T09:00:00Z",
    )
    assert affected(ev) == ["CORR-1"]


def set_status(rl, ev, fid, status):
    tables = rl.parse_tables((ev / "LEDGER.md").read_text())
    for row in rl.find_table(tables, "Findings").rows:
        if row["ID"] == fid:
            row["Status"] = status
    (ev / "LEDGER.md").write_text(rl.render_tables(tables))


@pytest.mark.parametrize("status", ["ACCEPTED-RISK", "SUPERSEDED-BY:CORR-2"])
def test_an_unresolved_thread_of_a_closed_finding_is_a_mismatch_in_any_closed_status(
    pin, ledger, rl, ev, monkeypatch, capsys, status
):
    set_status(rl, ev, "CORR-1", status)
    link_from(ledger, ev, f"{URL.format(1)}\tCORR-1")
    refetch(pin, ev, monkeypatch, FakeGh([thread(1)]), 1, "--reviewer", "reviewer")
    assert f"mismatch: {URL.format(1)} unresolved, CORR-1 {status}" in (
        capsys.readouterr().out
    )
    resolved = FakeGh([thread(1, resolved=True)])
    refetch(pin, ev, monkeypatch, resolved, 1, "--reviewer", "reviewer")
    assert "mismatch" not in capsys.readouterr().out


def test_a_reopened_round_takes_replies_after_its_own_pinned(
    pin, ledger, ev, monkeypatch
):
    (ev / "r2" / "PIN.md").write_text(
        "Round: 2\nPinned: 2026-02-10T00:00:00Z\nReviewer: reviewer\n"
    )
    link_from(
        ledger,
        ev,
        f"{URL.format(1)}\tCORR-1",  # reply between r1's and r2's Pinned
        f"{URL.format(2)}\tCORR-2",  # reply after r2's Pinned
    )
    fake = FakeGh(
        [
            thread(1, replies=[("author", "2026-02-05T00:00:00Z")]),
            thread(2, replies=[("author", "2026-02-11T00:00:00Z")]),
        ]
    )
    refetch(pin, ev, monkeypatch, fake, 2)
    assert affected(ev, 2) == ["CORR-2"]
    (ev / "r2" / "PIN.md").write_text("Round: 2\nReviewer: reviewer\n")
    with pytest.raises(SystemExit) as e:  # r1's Pinned is not a fallback
        refetch(pin, ev, monkeypatch, fake, 2)
    assert "--since" in str(e.value)


def test_a_comment_with_no_author_counts_as_someone_other_than_the_reviewer(
    pin, ledger, rl, ev, monkeypatch
):
    link_from(ledger, ev, f"{URL.format(1)}\tCORR-1")
    t = thread(1, replies=[("x", "2026-02-02T00:00:00Z")])
    t["comments"]["nodes"][1]["author"] = None
    refetch(pin, ev, monkeypatch, FakeGh([t]), 1, "--reviewer", "reviewer")
    assert affected(ev) == ["CORR-1"]
    assert threads_rows(rl, ev)[0]["Last reply"] == "other 2026-02-02"


def test_a_linked_thread_missing_from_the_fetch_is_reported_and_keeps_its_row(
    pin, ledger, rl, ev, monkeypatch, capsys
):
    link_from(ledger, ev, f"{URL.format(1)}\tCORR-1", f"{URL.format(9)}\tCORR-2")
    refetch(pin, ev, monkeypatch, FakeGh([thread(1)]), 1, "--reviewer", "reviewer")
    out = capsys.readouterr().out
    assert f"missing: {URL.format(9)} (CORR-2)" in out
    assert "1 linked" in out
    rows = {r["Finding"]: r for r in threads_rows(rl, ev)}
    assert rows["CORR-2"]["State"] == "" and rows["CORR-2"]["Round seen"] == "1"
    assert rows["CORR-1"]["State"] == "unresolved"


def test_threads_md_marks_outdated_and_shows_the_line_and_last_reply(
    pin, ledger, ev, monkeypatch
):
    link_from(ledger, ev, f"{URL.format(1)}\tCORR-1")
    old = thread(
        1,
        replies=[("author", "2026-02-02T00:00:00Z"), ("other", "2026-02-03T00:00:00Z")],
    )
    old["isOutdated"] = True
    ranged = thread(2, root="other")
    ranged["originalLine"], ranged["originalStartLine"] = None, 42
    refetch(pin, ev, monkeypatch, FakeGh([old, ranged]), 1, "--reviewer", "reviewer")
    rows = {
        line.split(" | ")[0].lstrip("| "): line
        for line in (ev / "r1" / "threads.md").read_text().splitlines()
        if line.startswith("| https")
    }
    assert rows[URL.format(1)] == (
        f"| {URL.format(1)} | a.cpp:11 | unresolved (outdated) | reviewer | "
        "other 2026-02-03 | CORR-1 |"
    )
    assert rows[URL.format(2)] == (
        f"| {URL.format(2)} | a.cpp:42 | unresolved | other |  |  |"
    )


@pytest.mark.parametrize(
    "line",
    [
        "{u} CORR-1",  # spaces for the tab
        "{u}\t",  # no ID
        "{u}\t, ,",  # separators only
    ],
)
def test_link_rejects_a_malformed_posted_line_naming_it(ledger, ev, capsys, line):
    good, bad = f"{URL.format(1)}\tCORR-1", line.format(u=URL.format(2))
    assert link_from(ledger, ev, good, bad) == 1
    assert "line 2: expected <comment URL><TAB><ID>" in capsys.readouterr().err
    assert "## Threads" not in (ev / "LEDGER.md").read_text()


@pytest.mark.parametrize(
    "finding, resolved, expect",
    [
        ("CORR-1", True, "resolved, CORR-1 OPEN"),
        ("API-1", True, "resolved, API-1 UNVERIFIED"),
        ("DOCS-1", False, "unresolved, DOCS-1 FIXED@abc1234"),
        ("DUP-1", False, "unresolved, DUP-1 REFUTED"),
    ],
)
def test_mismatch_between_thread_and_finding_state_is_reported(
    pin, ledger, ev, monkeypatch, capsys, finding, resolved, expect
):
    link_from(ledger, ev, f"{URL.format(1)}\t{finding}")
    refetch(
        pin,
        ev,
        monkeypatch,
        FakeGh([thread(1, resolved=resolved)]),
        1,
        "--reviewer",
        "reviewer",
    )
    assert f"mismatch: {URL.format(1)} {expect}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "finding, resolved",
    [("CORR-1", False), ("TESTQ-1", False), ("DOCS-1", True), ("DUP-1", True)],
)
def test_agreeing_state_reports_no_mismatch(
    pin, ledger, ev, monkeypatch, capsys, finding, resolved
):
    link_from(ledger, ev, f"{URL.format(1)}\t{finding}")
    refetch(
        pin,
        ev,
        monkeypatch,
        FakeGh([thread(1, resolved=resolved)]),
        1,
        "--reviewer",
        "reviewer",
    )
    assert "mismatch" not in capsys.readouterr().out


# freshness -------------------------------------------------------------------


def freshness(pin, ev, monkeypatch, fake, capsys, *extra):
    shutil.rmtree(ev / "r2", ignore_errors=True)  # r1 is the newest round
    monkeypatch.setattr(pin, "gh", fake)
    pin.main(
        ["freshness", "--evidence", str(ev), "--owner-repo", "example/repo", *extra]
    )
    return capsys.readouterr().out.splitlines()


def test_freshness_fresh_reopens_the_round(pin, ev, monkeypatch, capsys):
    fake = FakeGh([thread(1, replies=[("reviewer", "2026-03-01T00:00:00Z")])])
    assert freshness(pin, ev, monkeypatch, fake, capsys) == ["fresh", "next: reopen r1"]


def test_freshness_new_thread_activity_reopens_the_round(pin, ev, monkeypatch, capsys):
    fake = FakeGh([thread(1, replies=[("author", "2026-03-01T00:00:00Z")])])
    assert freshness(pin, ev, monkeypatch, fake, capsys) == [
        "new thread activity",
        "next: reopen r1",
    ]


def test_freshness_a_new_thread_by_someone_else_is_activity(
    pin, ev, monkeypatch, capsys
):
    fake = FakeGh([thread(1, root="other", when="2026-03-01T00:00:00Z")])
    assert freshness(pin, ev, monkeypatch, fake, capsys)[0] == "new thread activity"


def test_freshness_head_moved_beats_thread_activity(pin, ev, monkeypatch, capsys):
    fake = FakeGh(
        [thread(1, replies=[("author", "2026-03-01T00:00:00Z")])], head="b" * 40
    )
    assert freshness(pin, ev, monkeypatch, fake, capsys) == [
        "stale: head moved",
        "next: start r2",
    ]


def test_freshness_retargeted_pr_beats_head_moved(pin, ev, monkeypatch, capsys):
    fake = FakeGh(base="release", head="b" * 40)
    assert freshness(pin, ev, monkeypatch, fake, capsys) == [
        "stale: base branch changed",
        "next: start r2",
    ]


def test_freshness_of_a_published_only_round_takes_flags(pin, ev, monkeypatch, capsys):
    (ev / "r1" / "PIN.md").unlink()
    fake = FakeGh(head="b" * 40)
    with pytest.raises(SystemExit) as e:
        freshness(pin, ev, monkeypatch, fake, capsys, "--pr", "7")
    assert "--prev-head" in str(e.value)
    out = freshness(
        pin,
        ev,
        monkeypatch,
        fake,
        capsys,
        "--pr",
        "7",
        "--prev-head",
        "a" * 12,
        "--prev-pinned",
        "2026-02-01",
    )
    assert out == [
        "skipped: base branch check (no previous target)",
        "stale: head moved",
        "next: start r2",
    ]
    # A date-only value is 00:00 UTC; a short previous head matches the full one.
    same = FakeGh(
        [thread(1, replies=[("author", "2026-02-01T00:00:01Z")])], head="a" * 40
    )
    out = freshness(
        pin,
        ev,
        monkeypatch,
        same,
        capsys,
        "--pr",
        "7",
        "--prev-head",
        "a" * 12,
        "--prev-pinned",
        "2026-02-01",
        "--prev-target",
        "develop",
        "--reviewer",
        "reviewer",
    )
    assert out == ["new thread activity", "next: reopen r1"]


def test_freshness_without_a_pr_is_not_applicable(pin, ev, monkeypatch, capsys):
    (ev / "r1" / "PIN.md").write_text("Round: 1\nKEY: my-branch\n")
    fake = FakeGh()
    assert freshness(pin, ev, monkeypatch, fake, capsys) == ["N/A"]
    assert fake.calls == []


# comments.py universe --------------------------------------------------------


def test_universe_lists_the_rows_draft_selects_with_the_contract_keys(
    comments, ledger, ev, capsys
):
    (ev / "r1" / "PIN.md").unlink()  # works without PIN.md
    (ev / "r1" / "lanes").mkdir()
    (ev / "r1" / "lanes" / "CORR1.md").write_text(
        "# CORR1 (round 1, aaaaaaaaaaaa)\n\n## Findings\n\n"
        "ID: CORR1-p1\nKind: defect\nTier: IMPORTANT\nWhere: a.cpp:11\nExpected: e\n"
        "Actual: a\nProof: p\nFix: clamp it\n"
    )
    (ev / "r1" / "ID-MAP.md").write_text(
        "## ID-MAP\n\n| Lane / provisional | Final ID | SUPERSEDED-BY |\n|---|---|---|\n"
        "| CORR1/CORR1-p1 | CORR-1 |  |\n"
    )
    assert comments.main(["universe", str(ev), "1"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert [e["id"] for e in entries] == ["CORR-1", "CORR-2", "API-1", "TESTQ-1"]
    assert all(
        set(e)
        == {"id", "tier", "kind", "status", "where", "claim", "evidence", "fix", "lane"}
        for e in entries
    )
    first = entries[0]
    assert (first["tier"], first["kind"], first["status"], first["where"]) == (
        "IMPORTANT",
        "defect",
        "OPEN",
        "a.cpp:11",
    )
    assert (first["claim"], first["evidence"], first["fix"], first["lane"]) == (
        "claim one",
        "proof",
        "clamp it",
        "CORR",
    )
    assert entries[1]["fix"] == "" and entries[2]["status"] == "UNVERIFIED"


# context packets -------------------------------------------------------------


def test_packets_list_unresolved_threads_the_reviewer_did_not_open(part, tmp_path):
    (tmp_path / "numstat.txt").write_text("1\t0\ta.cpp\n")
    (tmp_path / "files.txt").write_text("a.cpp\n")
    part.main(["areas", "--round-dir", str(tmp_path)])
    (tmp_path / "owners.tsv").write_text("CORR1\ta.cpp\n")
    (tmp_path / "pr").mkdir()
    doc = {
        "author": "author",
        "reviewer": "reviewer",
        "threads": [
            thread(1, root="reviewer"),
            thread(2, root="other"),
            thread(3, root="author", resolved=True),
        ],
    }
    (tmp_path / "pr" / "threads.json").write_text(json.dumps(doc))
    part.main(["packets", "--round-dir", str(tmp_path)])
    packet = (tmp_path / "packets" / "CORR1.md").read_text()
    assert f"- a.cpp:12 {URL.format(2)} first line" in packet
    assert URL.format(1) not in packet and URL.format(3) not in packet
    assert "other" not in packet.replace("## Open review threads", "")
