# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the hipdnn-deep-review skill's comments.py (hunk anchoring)."""

import importlib.util
import os
import subprocess
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
def comments():
    return _load("comments")


@pytest.fixture(scope="module")
def rl():
    return _load("reviewlib")


def git(repo, *args):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(repo),
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """BASE has a.txt with lines L1..L10; HEAD rewrites L5, deletes L8 and appends N11, N12."""
    path = tmp_path / "clone"
    path.mkdir()
    git(path, "init", "-q")
    (path / "a.txt").write_text("".join(f"L{i}\n" for i in range(1, 11)))
    git(path, "add", ".")
    git(path, "commit", "-qm", "base")
    base = git(path, "rev-parse", "HEAD")
    lines = [f"L{i}\n" for i in range(1, 11)]
    lines[4] = "NEW5\n"
    del lines[7]
    (path / "a.txt").write_text("".join(lines) + "N11\nN12\n")
    git(path, "commit", "-qam", "head")
    return path, base, git(path, "rev-parse", "HEAD")


def finding(prov, tier, where, actual, kind="defect"):
    return (
        f"ID: {prov}\nKind: {kind}\nTier: {tier}\nWhere: {where}\nExpected: the contract\n"
        f"Actual: {actual}\nProof: p\nFix: smallest fix\n"
    )


ROW = "| {id} | {t} -> {t} | {kind} | {status} |  | {claim} | {where} | e | 1 |  |  |"


def build(tmp_path, repo, specs):
    """specs: (final ID, tier, where, actual, status, kind) rows."""
    path, base, head = repo
    ev = tmp_path / "ev"
    (ev / "r1" / "lanes").mkdir(parents=True)
    (ev / "r1" / "PIN.md").write_text(
        f"Round: 1\nCLONE: {path}\nBASE: {base}\nHEAD: {head}\n"
    )
    rows, idmap, blocks = [], [], []
    for i, (fid, tier, where, actual, status, kind) in enumerate(specs, 1):
        prov = f"CORR1-p{i}"
        blocks.append(finding(prov, tier, where, actual, kind))
        idmap.append(f"| CORR1/{prov} | {fid} |  |")
        rows.append(
            ROW.format(
                id=fid,
                t=tier,
                kind=kind,
                status=status,
                claim=actual.split("\n")[0],
                where=where,
            )
        )
    cols = "| ID | Tier | Kind | Status | Verdict | Claim | Where | Evidence | Round opened | Round closed | known-open-item |"
    (ev / "LEDGER.md").write_text(
        "## Findings\n\n" + cols + "\n" + "|---|" * 11 + "\n" + "\n".join(rows) + "\n"
    )
    (ev / "r1" / "ID-MAP.md").write_text(
        "## ID-MAP\n\n| Lane / provisional | Final ID | SUPERSEDED-BY |\n|---|---|---|\n"
        + "\n".join(idmap)
        + "\n"
    )
    (ev / "r1" / "lanes" / "CORR1.md").write_text(
        "# CORR1 (round 1, abc)\n\n## Findings\n\n"
        + "\n".join(blocks)
        + "\n## Coverage\n"
    )
    return ev


def draft(comments, ev):
    assert comments.main(["draft", str(ev), "1"]) == 0
    return (ev / "r1" / "COMMENTS.md").read_text()


def test_anchor_inside_hunk_is_inline_and_outside_or_removed_is_general(
    comments, repo, tmp_path
):
    ev = build(
        tmp_path,
        repo,
        [
            (
                "CORR-1",
                "BLOCKING",
                "`a.txt:5`",
                "Rewritten line is wrong.",
                "OPEN",
                "defect",
            ),
            (
                "CORR-2",
                "IMPORTANT",
                "a.txt:11, a.txt:2",
                "Appended line is wrong.",
                "OPEN",
                "defect",
            ),
            (
                "CORR-3",
                "IMPORTANT",
                "a.txt:2",
                "Untouched line is wrong.",
                "OPEN",
                "defect",
            ),
            (
                "CORR-4",
                "BLOCKING",
                "$BASE:a.txt:8",
                "Removed line mattered.",
                "OPEN",
                "defect",
            ),
            (
                "CORR-5",
                "IMPORTANT",
                "a.txt:7",
                "Line next to a pure deletion.",
                "OPEN",
                "defect",
            ),
        ],
    )
    out = draft(comments, ev)
    inline, general = out.split("## General")
    assert (
        "### a.txt:5 (CORR-1, BLOCKING)" in inline
        and "### a.txt:11 (CORR-2, IMPORTANT)" in inline
    )
    assert "Fix: smallest fix" in inline and "Expected: the contract" in inline
    for fid in ("CORR-3", "CORR-4", "CORR-5"):
        assert f"({fid}, " in general and f"({fid}, " not in inline
    assert "Removed line mattered." in general


def test_only_open_findings_are_drafted_and_suggestions_are_batched_by_theme(
    comments, repo, tmp_path
):
    ev = build(
        tmp_path,
        repo,
        [
            ("CORR-1", "BLOCKING", "a.txt:5", "closed one.", "FIXED@abc", "defect"),
            (
                "CORR-2",
                "IMPORTANT",
                "a.txt:5",
                "unverified one.",
                "UNVERIFIED",
                "defect",
            ),
            ("CORR-3", "SUGGESTION", "a.txt:5", "rename it.", "OPEN", "source-claim"),
            (
                "CORR-4",
                "SUGGESTION",
                "a.txt:11",
                "rename that too.",
                "OPEN",
                "source-claim",
            ),
            ("CORR-5", "FUTURE WORK", "a.txt:1", "extract helper.", "OPEN", "process"),
        ],
    )
    out = draft(comments, ev)
    assert "###" not in out and "CORR-1" not in out and "CORR-2" not in out
    themed = [line for line in out.splitlines() if line.startswith("- **")]
    assert len(themed) == 2
    assert (
        "rename it." in themed[0]
        and "rename that too." in themed[0]
        and "extract helper." in themed[1]
    )


def test_pasted_actual_has_no_code_fences(comments, repo, tmp_path):
    ev = build(
        tmp_path,
        repo,
        [
            (
                "CORR-1",
                "BLOCKING",
                "a.txt:5",
                "Fails on:\n```cpp\nint x = f();\n```\nso it crashes.",
                "OPEN",
                "defect",
            )
        ],
    )
    out = draft(comments, ev)
    assert "```" not in out
    assert "int x = f();" in out and "so it crashes." in out


def test_partial_verdict_pastes_the_narrowed_claim(comments, repo, tmp_path):
    ev = build(
        tmp_path,
        repo,
        [("CORR-1", "IMPORTANT", "a.txt:5", "Broad claim.", "OPEN", "defect")],
    )
    ledger = (
        (ev / "LEDGER.md")
        .read_text()
        .replace("|  | Broad claim.", "| PARTIAL | Narrow claim.")
    )
    (ev / "LEDGER.md").write_text(ledger)
    out = draft(comments, ev)
    assert "Actual: Narrow claim." in out and "Broad claim." not in out


def test_missing_pin_values_fail_with_reason(comments, repo, tmp_path, capsys):
    ev = build(tmp_path, repo, [])
    (ev / "r1" / "PIN.md").write_text("Round: 1\n")
    assert comments.main(["draft", str(ev), "1"]) == 1
    assert "PIN.md lacks" in capsys.readouterr().err


def test_write_is_utf8_lf_and_atomic(rl, tmp_path):
    target = tmp_path / "LEDGER.md"
    rl.write(target, "a \u2192 b \u201c\u201d\nsecond\n")
    assert target.read_bytes() == "a \u2192 b \u201c\u201d\nsecond\n".encode("utf-8")
    with pytest.raises(UnicodeEncodeError):
        rl.write(target, "\ud800")
    assert target.read_text(encoding="utf-8").startswith("a \u2192")
    assert [p.name for p in tmp_path.iterdir()] == ["LEDGER.md"]


def test_run_decodes_utf8_and_replaces_invalid_bytes(rl):
    code = "import sys; sys.stdout.buffer.write(b'\\xe2\\x80\\x9d \\xff')"
    rc, out, _ = rl.run([sys.executable, "-c", code])
    assert rc == 0 and out == "\u201d \ufffd"


def test_round_dirs_sort_numerically(rl, tmp_path):
    for name in ("r1", "r2", "r10", "r3x", "rx", "notes"):
        (tmp_path / name).mkdir()
    assert [p.name for p in rl.round_dirs(tmp_path)] == ["r1", "r2", "r10"]
