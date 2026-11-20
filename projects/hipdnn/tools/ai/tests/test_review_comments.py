# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the hipdnn-review-comments skill's review_comments.py: the mechanical gate over
FINDINGS.md, TRIAGE.md, COMMENTS.md and comments.json, the posting payloads and the
`posted` records, including the hand-off to the deep-review ledger.

GitHub is never called: patches come from a synthetic `pulls/<N>/files` dump or a throwaway
git repository."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parent.parent / "skills"
SCRIPT = SKILLS / "hipdnn-review-comments" / "scripts" / "review_comments.py"
DEEP = SKILLS / "hipdnn-deep-review" / "scripts"
HEAD = "b" * 40
BASE = "a" * 40
URL = "https://github.com/example/repo/pull/7#discussion_r{}"
REVIEW_URL = "https://github.com/example/repo/pull/7#pullrequestreview-9"

# New-side lines 10 (context), 11 (added), 12-13 (context); 21-23 (context, context, context)
# with a deleted line between 21 and 22.
PATCH = (
    "@@ -10,3 +10,4 @@ int f()\n ctx\n+added\n ctx\n ctx\n"
    "@@ -20,4 +21,3 @@ int g()\n ctx\n-gone\n ctx\n ctx"
)


def _load_module(name, path, extra_path=None):
    if extra_path:
        sys.path.insert(0, str(extra_path))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if extra_path:
            sys.path.remove(str(extra_path))


@pytest.fixture(scope="module")
def rc():
    return _load_module("review_comments", SCRIPT)


def base_state():
    """A run the gate accepts: a merged pair, a context-line anchor, a sibling-ID-shaped
    token, a backticked run ID, an UNANCHORED BLOCKING item and a review body."""
    return {
        "findings": {
            "CORR-1": ("IMPORTANT", "a.cpp:11", ""),
            "CORR-2": ("SUGGESTION", "a.cpp:11", ""),
            "DOCS-2": ("BLOCKING", "docs/x.md:1", ""),
            "F1": ("", "a.cpp:10", ""),
        },
        "triage": [
            ["CORR-1", "a.cpp:11", "KEEP", "P1", "Overflow on the cached index."],
            ["CORR-2", "-", "MERGE INTO CORR-1", "P2", "Same root cause as CORR-1."],
            ["DOCS-2", "-", "UNANCHORED", "P1", "Sits in a file outside the diff."],
            ["F1", "a.cpp:10", "KEEP", "P2", "Stale comment next to the change."],
        ],
        "comments": [
            {
                "path": "a.cpp",
                "line": 11,
                "start": None,
                "ids": ["CORR-1", "CORR-2"],
                "prio": "P1",
                "body": "The index can overflow here; widen it.",
            },
            {
                "path": "a.cpp",
                "line": 10,
                "start": None,
                "ids": ["F1"],
                "prio": "P2",
                "body": "Optional: this comment is stale, see SEM-015 and `CORR-1`.",
            },
        ],
        "review": "Two notes outside the diff: the build docs and `CORR-1`.",
        "json": {},
        "files": {"a.cpp": PATCH},
    }


def write_run(tmp, s):
    out = tmp / "run"
    out.mkdir(exist_ok=True)
    f = ["# Findings", "", "Head and decisions go here.", ""]
    for fid, (tier, evidence, decision) in s["findings"].items():
        f.append(f"## {fid}. claim of {fid}")
        if tier:
            f.append(f"Tier: {tier}")
        if evidence:
            f.append(f"Evidence: {evidence}")
        if decision:
            f.append(f"Decision: {decision}")
        f.append("")
    (out / "FINDINGS.md").write_text("\n".join(f))
    t = ["| ID | Anchor (or -) | Action | Priority | Reason |", "|---|---|---|---|---|"]
    t += ["| " + " | ".join(r) + " |" for r in s["triage"]]
    (out / "TRIAGE.md").write_text("\n".join(t) + "\n")
    c = [f"Head: {HEAD[:12]}", f"Base: {BASE[:12]}", ""]
    data = []
    for i, cm in enumerate(s["comments"], 1):
        span = f"{cm['start']}-{cm['line']}" if cm["start"] else f"{cm['line']}"
        c += [
            f"### {i}. {cm['prio']} {cm['path']}:{span} ({', '.join(cm['ids'])})",
            "",
            cm["body"],
            "",
        ]
        entry = {
            "commit_id": HEAD,
            "path": cm["path"],
            "line": cm["line"],
            "side": "RIGHT",
            "body": cm["body"],
            "finding_ids": cm["ids"],
        }
        if cm["start"]:
            entry |= {"start_line": cm["start"], "start_side": "RIGHT"}
        data.append(entry)
    if s["review"] is not None:
        c += ["## Review body", "", s["review"], ""]
    (out / "COMMENTS.md").write_text("\n".join(c))
    for index, patch in s["json"].items():
        data[index].update(patch)
    (out / "comments.json").write_text(json.dumps(data))
    files = [{"filename": n, "patch": p} for n, p in s["files"].items()]
    (tmp / "files.json").write_text(json.dumps(files))
    return out


def run_check(rc, tmp, s, capsys, *extra):
    out = write_run(tmp, s)
    code = rc.main(["check", str(out), "--files", str(tmp / "files.json"), *extra])
    return code, capsys.readouterr().err


def test_a_valid_run_passes(rc, tmp_path, capsys):
    code, err = run_check(rc, tmp_path, base_state(), capsys)
    assert (code, err) == (0, "")


def test_a_range_inside_one_hunk_and_a_dry_run_head_pass(rc, tmp_path, capsys):
    s = base_state()
    s["comments"][1].update(line=13, start=10)
    code, err = run_check(rc, tmp_path, s, capsys, "--head", HEAD)
    assert (code, err) == (0, "")


FAILURES = {
    "finding without a triage row": (
        lambda s: s["triage"].pop(),
        "F1 has no TRIAGE row",
    ),
    "empty reason": (
        lambda s: s["triage"][0].__setitem__(4, ""),
        "CORR-1 has an empty or bare-nit reason",
    ),
    "bare nit reason": (
        lambda s: s["triage"][2].__setitem__(4, "nit."),
        "DOCS-2 has an empty or bare-nit reason",
    ),
    "duplicate reason": (
        lambda s: s["triage"][3].__setitem__(4, "Overflow on the cached index"),
        "F1 repeats the reason of CORR-1",
    ),
    "merge reason that does not name its target": (
        lambda s: s["triage"][1].__setitem__(4, "Overflow on the cached index."),
        "CORR-2 repeats the reason of CORR-1",
    ),
    "undecided blocking drop": (
        lambda s: s["triage"][2].__setitem__(2, "DROP"),
        "DOCS-2 is a BLOCKING item dropped without a Decision",
    ),
    "comment citing a drop": (
        lambda s: s["triage"][3].__setitem__(2, "DROP"),
        "comment 2 cites F1, which TRIAGE marks DROP",
    ),
    "merged id missing from its target": (
        lambda s: s["comments"][0].__setitem__("ids", ["CORR-1"]),
        "CORR-2 merges into CORR-1 but is missing",
    ),
    "kept item in no comment": (
        lambda s: s["comments"].pop(),
        "F1 is KEEP in TRIAGE but in no comment",
    ),
    "json body differs": (
        lambda s: s["json"].update({0: {"body": "other"}}),
        "comments.json comment 1: body differs",
    ),
    "json ids differ": (
        lambda s: s["json"].update({1: {"finding_ids": ["F9"]}}),
        "comments.json comment 2: finding_ids differs",
    ),
    "json for another head": (
        lambda s: s["json"].update({0: {"commit_id": "c" * 40}}),
        "commit_id is not the head",
    ),
    "anchor on a deleted-side gap": (
        lambda s: s["comments"][0].update(line=20),
        "comment 1: a.cpp:20 is not added or context",
    ),
    "range across hunks": (
        lambda s: s["comments"][0].update(start=13, line=21),
        "comment 1: a.cpp:13-21",
    ),
    "anchor in an untouched file": (
        lambda s: s["comments"][0].update(path="zzz.cpp"),
        "zzz.cpp:11 is not added or context",
    ),
    "missing patch without a checkout": (
        lambda s: s["files"].update({"a.cpp": None}),
        "no patch for a.cpp",
    ),
    "own heading ID in the body": (
        lambda s: s["comments"][0].__setitem__("body", "See `CORR-2` and CORR-1."),
        "comment 1 contains its own heading ID",
    ),
    "another run ID outside backticks": (
        lambda s: s["comments"][1].__setitem__("body", "Related to CORR-1 here."),
        "comment 2 names run ID CORR-1 outside backticks",
    ),
    "run ID in the review body": (
        lambda s: s.__setitem__("review", "Also DOCS-2 is open."),
        "review body names run ID DOCS-2",
    ),
    "empty review body": (
        lambda s: s.__setitem__("review", ""),
        "Review body section is empty",
    ),
}


@pytest.mark.parametrize("name", FAILURES)
def test_check_fails_on_each_gate_rule(rc, tmp_path, capsys, name):
    change, message = FAILURES[name]
    s = base_state()
    change(s)
    code, err = run_check(rc, tmp_path, s, capsys)
    assert code == 1
    assert message in err


def test_a_heading_that_is_not_an_item_is_an_error(rc, tmp_path, capsys):
    out = write_run(tmp_path, base_state())
    (out / "FINDINGS.md").write_text(
        (out / "FINDINGS.md").read_text() + "\n## Notes\nfree\n"
    )
    assert rc.main(["check", str(out), "--files", str(tmp_path / "files.json")]) == 1
    assert "`## Notes` is not `## <ID>. <claim>`" in capsys.readouterr().err


def test_check_rejects_a_denied_term_and_an_evidence_path(rc, tmp_path, capsys):
    s = base_state()
    s["comments"][0]["body"] = "Ask Internal-Name about /var/ev/r1 first."
    (tmp_path / "terms.txt").write_text("\ninternal-name\n")
    code, err = run_check(
        rc,
        tmp_path,
        s,
        capsys,
        "--terms",
        str(tmp_path / "terms.txt"),
        "--evidence",
        "/var/ev",
    )
    assert code == 1
    assert "comment 1 contains a denied term" in err
    assert "comment 1 contains an evidence path" in err
    s["review"] = "Read $EVIDENCE/r1/REPORT.md."
    code, err = run_check(rc, tmp_path, s, capsys)
    assert "review body contains an evidence path" in err


def test_a_stale_head_fails(rc, tmp_path, capsys):
    code, err = run_check(rc, tmp_path, base_state(), capsys, "--head", "c" * 40)
    assert code == 1
    assert "is not the head" in err


def test_anchors_come_from_the_checkout_when_there_is_no_patch(rc, tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(repo),
    }

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()

    git("init", "-q")
    (repo / "a.cpp").write_text("".join(f"line{i}\n" for i in range(1, 31)))
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    lines = [f"line{i}\n" for i in range(1, 31)]
    lines.insert(14, "added\n")
    (repo / "a.cpp").write_text("".join(lines))
    git("commit", "-qam", "head")
    head = git("rev-parse", "HEAD")

    def run_at(s):
        out = write_run(tmp_path, s)
        for name in ("COMMENTS.md", "comments.json"):
            text = (out / name).read_text()
            text = (
                text.replace(HEAD, head)
                .replace(HEAD[:12], head)
                .replace(BASE[:12], base)
            )
            (out / name).write_text(text)
        return out

    s = base_state()
    s["comments"][0].update(line=15)  # the added line
    s["comments"][1].update(line=13)  # context above it
    assert rc.main(["check", str(run_at(s)), "--checkout", str(repo)]) == 0
    s["comments"][0].update(line=29)
    out = run_at(s)
    capsys.readouterr()
    assert rc.main(["check", str(out), "--checkout", str(repo)]) == 1
    assert "comment 1: a.cpp:29 is not added or context" in capsys.readouterr().err


def test_payload_has_no_finding_ids_and_the_review_is_one_comment_review(
    rc, tmp_path, capsys
):
    out = write_run(tmp_path, base_state())
    capsys.readouterr()
    assert rc.main(["payload", str(out / "comments.json"), "--count"]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert rc.main(["payload", str(out / "comments.json"), "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "finding_ids" not in payload
    assert payload["path"] == "a.cpp" and payload["commit_id"] == HEAD
    assert rc.main(["review-payload", str(out), "--head", HEAD]) == 0
    review = json.loads(capsys.readouterr().out)
    assert review["event"] == "COMMENT" and "comments" not in review
    assert review["body"].startswith("Two notes")


# Ledger mode -----------------------------------------------------------------


@pytest.fixture
def ledger_run(rc, tmp_path):
    """Evidence with a real LEDGER.md, and a run compiled from the sibling skill's `universe`."""
    rl = _load_module("dr_reviewlib", DEEP / "reviewlib.py", DEEP)
    ledger = _load_module("dr_ledger", DEEP / "ledger.py", DEEP)
    ev = tmp_path / "ev"
    (ev / "r3").mkdir(parents=True)
    findings = rl.Table("Findings", ledger.FINDING_COLS)
    for fid, tier, claim in (
        ("CORR-1", "IMPORTANT", "claim one"),
        ("CORR-2", "SUGGESTION", "claim two"),
        ("DOCS-1", "IMPORTANT", "docs one"),
        ("TESTQ-1", "SUGGESTION", "never commented"),
    ):
        findings.add(
            **{
                "ID": fid,
                "Tier": f"{tier} -> {tier}",
                "Kind": "defect",
                "Status": "OPEN",
                "Claim": claim,
                "Where": "a.cpp:11",
                "Evidence": "proof at a.cpp:11",
                "Round opened": "1",
            }
        )
    (ev / "LEDGER.md").write_text(
        rl.render_tables([findings, rl.Table("Decisions", ledger.DECISION_COLS)])
    )
    return ev, rl


def compile_ledger_run(rc, tmp_path, ev):
    out = tmp_path / "run"
    assert rc.main(["findings", str(ev), "3", "--out", str(out), "--head", HEAD]) == 0
    return out


def ledger_state():
    s = base_state()
    s["findings"] = {
        "CORR-1": ("IMPORTANT", "proof at a.cpp:11", ""),
        "CORR-2": ("SUGGESTION", "proof at a.cpp:11", ""),
        "DOCS-1": ("IMPORTANT", "proof at a.cpp:11", ""),
        "TESTQ-1": ("SUGGESTION", "proof at a.cpp:11", ""),
        "F1": ("", "a.cpp:10", ""),
    }
    s["triage"] = [
        ["CORR-1", "a.cpp:11", "KEEP", "P1", "Overflow on the cached index."],
        ["CORR-2", "-", "MERGE INTO CORR-1", "P2", "Same root cause as CORR-1."],
        ["DOCS-1", "a.cpp:13", "KEEP", "P1", "The documented default is wrong."],
        ["TESTQ-1", "-", "DROP", "P3", "Covered by an existing test."],
        ["F1", "a.cpp:10", "KEEP", "P2", "Stale comment next to the change."],
    ]
    s["comments"].insert(
        1,
        {
            "path": "a.cpp",
            "line": 13,
            "start": None,
            "ids": ["DOCS-1"],
            "prio": "P1",
            "body": "The documented default no longer holds.",
        },
    )
    return s


def test_ledger_compile_carries_every_universe_item_with_its_evidence(
    rc, ledger_run, tmp_path
):
    ev, _ = ledger_run
    out = compile_ledger_run(rc, tmp_path, ev)
    items = rc.parse_findings((out / "FINDINGS.md").read_text())
    assert list(items) == ["CORR-1", "CORR-2", "DOCS-1", "TESTQ-1"]
    assert items["DOCS-1"]["tier"] == "IMPORTANT"
    assert items["CORR-1"]["evidence"] == "proof at a.cpp:11"
    # the default out dir is the next comments run of the round
    assert rc.main(["findings", str(ev), "3"]) == 0
    assert (ev / "r3" / "comments" / "c1" / "FINDINGS.md").is_file()
    assert rc.main(["findings", str(ev), "3"]) == 0
    assert (ev / "r3" / "comments" / "c2" / "FINDINGS.md").is_file()


def test_ledger_run_requires_every_universe_item(rc, ledger_run, tmp_path, capsys):
    ev, _ = ledger_run
    compile_ledger_run(rc, tmp_path, ev)
    s = ledger_state()
    del s["findings"]["TESTQ-1"]
    s["triage"].pop(3)
    write_run(tmp_path, s)
    capsys.readouterr()
    code = rc.main(
        ["check", str(tmp_path / "run"), "--files", str(tmp_path / "files.json")]
    )
    assert code == 1
    assert "ledger item TESTQ-1 is missing from FINDINGS.md" in capsys.readouterr().err


def test_posted_round_trips_through_ledger_link_with_merged_ids(
    rc, ledger_run, tmp_path, capsys
):
    ev, rl = ledger_run
    compile_ledger_run(rc, tmp_path, ev)
    out = write_run(tmp_path, ledger_state())
    # keep the compiled ledger.json next to the hand-written run
    assert (out / "ledger.json").is_file()
    assert rc.main(["check", str(out), "--files", str(tmp_path / "files.json")]) == 0
    (out / "POSTED.urls").write_text(
        f"1 {URL.format(101)}\n2 {URL.format(102)}\n3 {URL.format(103)}\nreview {REVIEW_URL}\n"
    )
    capsys.readouterr()
    assert rc.main(["posted", str(out)]) == 0
    text = capsys.readouterr().out
    assert "ledger.py" in text and f"link {ev} 3 --from {out / 'posted.tsv'}" in text
    tsv = (out / "posted.tsv").read_text().splitlines()
    assert tsv == [f"{URL.format(101)}\tCORR-1,CORR-2", f"{URL.format(102)}\tDOCS-1"]
    posted_md = (out / "POSTED.md").read_text()
    assert (
        URL.format(103) in posted_md and REVIEW_URL in posted_md
    )  # F1-only comment and body
    assert URL.format(103) not in "".join(tsv) and REVIEW_URL not in "".join(tsv)

    done = subprocess.run(
        [
            sys.executable,
            str(DEEP / "ledger.py"),
            "link",
            str(ev),
            "3",
            "--from",
            str(out / "posted.tsv"),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    threads = [
        r
        for t in rl.parse_tables((ev / "LEDGER.md").read_text())
        if t.title == "Threads"
        for r in t.rows
    ]
    assert sorted((r["Thread URL"], r["Finding"]) for r in threads) == [
        (URL.format(101), "CORR-1"),
        (URL.format(101), "CORR-2"),
        (URL.format(102), "DOCS-1"),
    ]


def test_posted_rejects_an_api_url_and_keeps_a_partial_run(rc, tmp_path, capsys):
    out = write_run(tmp_path, base_state())
    (out / "POSTED.urls").write_text(
        "1 https://api.github.com/repos/example/repo/pulls/comments/5\n"
    )
    assert rc.main(["posted", str(out)]) == 1
    assert "not a web comment URL" in capsys.readouterr().err
    (out / "POSTED.urls").write_text(f"1 {URL.format(5)}\n")
    capsys.readouterr()
    assert rc.main(["posted", str(out)]) == 0
    assert "1 of 2 comments" in capsys.readouterr().out
    assert "not posted" in (out / "POSTED.md").read_text()
    assert not (out / "posted.tsv").exists()  # free-form mode writes no posted.tsv


# Portability -----------------------------------------------------------------


def test_free_form_mode_needs_no_deep_review(rc, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(rc, "SKILLS_DIR", tmp_path / "no-skills")
    code, err = run_check(rc, tmp_path, base_state(), capsys)
    assert (code, err) == (0, "")


def test_ledger_mode_fails_clearly_without_deep_review(
    rc, tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(rc, "SKILLS_DIR", tmp_path / "no-skills")
    assert rc.main(["findings", str(tmp_path), "1", "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert "hipdnn-deep-review" in err and "install" in err
    assert not (tmp_path / "o").exists()


def test_findings_from_a_universe_file_needs_no_sibling(rc, tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "SKILLS_DIR", tmp_path / "no-skills")
    universe = tmp_path / "u.json"
    universe.write_text(
        json.dumps([{"id": "CORR-1", "tier": "BLOCKING", "claim": "c", "evidence": ""}])
    )
    out = tmp_path / "o"
    assert (
        rc.main(["findings", "ev", "2", "--out", str(out), "--universe", str(universe)])
        == 0
    )
    assert "## CORR-1. c" in (out / "FINDINGS.md").read_text()
    assert "Evidence:" not in (out / "FINDINGS.md").read_text()


# TRIAGE table shape and P3 -----------------------------------------------------


def test_a_triage_table_without_outer_pipes_parses(rc, tmp_path, capsys):
    out = write_run(tmp_path, base_state())
    lines = (out / "TRIAGE.md").read_text().splitlines()
    (out / "TRIAGE.md").write_text(
        "\n".join(line.strip("|").strip() for line in lines) + "\n"
    )
    assert rc.main(["check", str(out), "--files", str(tmp_path / "files.json")]) == 0


def p3_state(drafted):
    """F1 is a KEEP row at P3; `drafted` keeps its comment."""
    s = base_state()
    s["triage"][3][3] = "P3"
    if not drafted:
        s["comments"].pop()
    return s


def test_a_p3_row_is_not_drafted_and_needs_no_comment(rc, tmp_path, capsys):
    code, err = run_check(rc, tmp_path, p3_state(False), capsys)
    assert (code, err) == (0, "")


def test_a_comment_may_not_cite_a_p3_row(rc, tmp_path, capsys):
    code, err = run_check(rc, tmp_path, p3_state(True), capsys)
    assert code == 1
    assert "comment 2 cites F1, which TRIAGE marks P3" in err


def test_a_comment_priority_equals_its_triage_priority(rc, tmp_path, capsys):
    s = base_state()
    s["comments"][0]["prio"] = "P2"
    code, err = run_check(rc, tmp_path, s, capsys)
    assert code == 1
    assert "comment 1 is P2 but TRIAGE gives CORR-1 P1" in err


def test_a_p1_row_still_needs_a_comment(rc, tmp_path, capsys):
    s = base_state()
    s["comments"].pop(0)
    s["comments"][0]["ids"] = ["F1"]
    code, err = run_check(rc, tmp_path, s, capsys)
    assert code == 1
    assert "CORR-1 is KEEP in TRIAGE but in no comment" in err
