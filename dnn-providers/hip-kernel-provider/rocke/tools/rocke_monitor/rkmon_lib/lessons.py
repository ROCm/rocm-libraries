# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The weekly lessons-learned review of the agentic review process: every week, the issues and what we distilled from
them go into the process, so the agents embody the lessons and nobody repeats an erroneous cycle or re-discovers a known
fix.

The toolkit collects one week's raw material into `rocke_lessons_raw_<week>.md`, next to the weekly report:
- the reviewer model's claims that verification corrected (the daily log's `verify` saves), reviews dropped or
  failed, mis-posts;
- holds and their releases, and the auto-merge guard's actions;
- tool failures and held PRs merged anyway (the journal);
- the toolkit's own commits, each one a bug found or a rule added;
- the verifier notes archived that week (`<paths.lessons>/notes/`).
The orchestrator has it distilled into the skill's `lessons.md`, which the worker (through the review book), the verifiers
(through their team member file) and the orchestrator read. A lesson that recurs in spite of the prose becomes a toolkit
guard or a test.
"""

import datetime as dt
import pathlib
import subprocess

from . import daylog, paths
from .store import ROOT as MONITOR_ROOT  # the toolkit checkout: its own commits

NOTE_LINES = 40


def _in(ts: str, start: dt.datetime, end: dt.datetime) -> bool:
    try:
        return (
            start
            <= dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=dt.timezone.utc
            )
            < end
        )
    except ValueError:  # a journal line that does not start with a timestamp
        return False


def _section(title: str, lines: list[str]) -> list[str]:
    return [f"## {title} ({len(lines)})", ""] + (lines or ["None."]) + [""]


def collect(
    cfg: dict,
    store,
    start: dt.datetime,
    end: dt.datetime,
    label: str,
    notes_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Write the week's raw material for the lessons review and return its path. The verifier notes are read from
    `<paths.lessons>/notes`."""
    notes_dir = notes_dir or pathlib.Path(paths.get(cfg, "lessons")) / "notes"
    rows = [r for r in daylog.entries(store) if _in(r["ts"], start, end)]

    def log(kinds: tuple, save: str | None = None) -> list[str]:
        return [
            f"- {r['ts'][:16]}Z {'#' + str(r['pr']) if r.get('pr') else '—'}: {r['text']}"
            for r in rows
            if r["kind"] in kinds and (save is None or r.get("save") == save)
        ]

    guard = [
        f"- {r['ts'][:16]}Z #{r['pr']}: {r.get('note') or r['ref']}"
        for r in store.ledger()
        if r["kind"] == "automerge-off" and _in(r["ts"], start, end)
    ]
    journal = store.journal_lines()  # the rotated journal too (retention.py)
    trouble = [
        f"- {line[:240]}"
        for line in journal
        if _in(line, start, end)
        and (
            "TOOL failure" in line
            or "HELD_MERGED" in line
            or "mis-post" in line.lower()
        )
    ]
    try:
        # Only the toolkit's own folder: in a rocKE checkout, the rest of the repository is not ours.
        commits = subprocess.run(
            [
                "git",
                "-C",
                str(MONITOR_ROOT),
                "log",
                f"--since={start:%Y-%m-%dT%H:%M:%SZ}",
                f"--until={end:%Y-%m-%dT%H:%M:%SZ}",
                "--format=- %h %ad %s",
                "--date=short",
                "--",
                ".",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        commits = ["- (git log failed)"]
    notes = []
    for p in sorted(notes_dir.glob("*.md")) if notes_dir.exists() else []:
        if start <= dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) < end:
            body = p.read_text(encoding="utf-8", errors="replace").splitlines()
            notes += (
                [f"### {p.name}", ""]
                + body[:NOTE_LINES]
                + (["…"] if len(body) > NOTE_LINES else [])
                + [""]
            )
    out = [
        f"# Lessons review raw material — {label} ({start:%Y-%m-%d} to {end - dt.timedelta(days=1):%Y-%m-%d}, UTC)",
        "",
        "Distil this into the skill's `lessons.md`: recurring cycles (did a lesson already exist?), new lessons, "
        "lessons to retire, and where each one is embodied (review book §11, the verifier's team member file, "
        "SKILL.md, or a toolkit guard).",
        "",
    ]
    out += _section("Worker claims verification corrected", log(("save",), "verify"))
    out += _section("Reviews dropped or failed", log(("skipped",)))
    out += _section("Mis-posts", log(("mispost",)))
    out += _section("Holds, releases and disarms", log(("hold", "release", "disarm")))
    out += _section("Auto-merge guard", guard)
    out += _section("Tool failures and held PRs merged anyway", trouble)
    out += _section("Toolkit commits", commits)
    out += [
        f"## Verifier notes ({sum(1 for line in notes if line.startswith('### '))})",
        "",
    ] + (notes or ["None."])
    path = pathlib.Path(paths.get(cfg, "weekly_dir")) / f"rocke_lessons_raw_{label}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
