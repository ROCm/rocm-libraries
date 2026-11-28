# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk draft-lint N`: a first-pass draft checked against its own inputs (lessons L-501).

It flags and never edits: the flags go to `checks/lint.md` beside the draft, for the verifier to read after its blind
pass. Each flag names the lesson it enforces. The inputs are the worker's folder: `review_agent.md` (the draft),
`agent_log.txt` (its tool calls), `stats_agent.json`, and `pr/` (meta.md, commits.md, threads.md, reviews.md,
comments.md, checks.txt).

What it flags:
- L-001: ACCEPT or REJECT; line 1 other than meta.md's `Line 1:` (with none, a line 1 that gives no automated reviewer
  name: rkmon_lib/reviewer_name.py); a header without `<light> First-round|Follow-up:`,
  or with a round word other than meta.md's `Round:`; a footer whose round word differs from the header's;
- L-002: a behind/ahead count; L-010: "Stale", a fix SHA that is not in commits.md, one SHA for several items;
  L-011: "well-tested" and the like with no test run; L-012: a "Saved to" line; L-014: a ticket tag the PR does not
  carry (the configured tracker keys, else any Jira-style key);
- L-003: a "(ran)" point no run_pytest, rule_scan or merge_conflicts call backs, and test counts no pytest summary
  printed; L-005: merge words that disagree with merge_conflicts;
- L-004: CI words or counts that disagree with checks.txt ("green" while a check runs or fails, "unrelated", "infra");
- L-006: a prior row for the operator's own items (meta.md's `Operator:`); L-009: an author other than meta.md's;
- L-013, L-016: a finding that only echoes a rule_scan text-I/O or perf hint;
- L-017: a reviewer's ×N above their thread count; L-019: a status other than OPEN with no reply and no later commit,
  or FIXED for an item raised after the last commit;
- L-101: a degenerate draft: repeated table rows, a forced finalize, the step cap, a very long think, one file read or
  one pattern grepped in most steps.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

from . import reviewer_name

LINT_FILE = "lint.md"  # in the review folder's checks/
MAX_STEPS = 20  # the local worker's step cap (review_agent.py MAX_STEPS)
LONG_THINK = 30000  # characters of one step's thinking
REPEATS = 3  # a table row (or a first cell) seen more often than this is a loop

VERDICT = re.compile(r"(?<![\w-])(ACCEPT|REJECT)(?![\w-])")
HEADER = re.compile(r"^## PR #\d+ — .*$", re.M)
HEADER_TAIL = re.compile(r"— (\S+) (First-round|Follow-up): ")
HEADER_WHO = re.compile(r"\(([^(),]+?)(?:, ([^()]*))?\) — [^—]*$")
FOOTER = re.compile(r"^_(First-round|Follow-up) review", re.M | re.I)
ANY_KEY = r"[A-Z][A-Z0-9]{1,9}"  # a Jira-style key, when no tracker key is configured


def tag_rx(keys=()) -> re.Pattern:
    """A ticket tag, KEY-NNN, of the tracker keys (config.tracker_keys), else of any Jira-style key."""
    return re.compile(
        r"(?<![\w-])(?:"
        + ("|".join(map(re.escape, keys)) if keys else ANY_KEY)
        + r")-\d+(?![\w-])"
    )


SHA = re.compile(r"`([0-9a-f]{7,40})`")
TIMES = re.compile(r"×\s?(\d+)")
STALE = re.compile(r"(?<![\w-])[Ss]tale(?![\w-])")
UNEARNED = re.compile(r"correctness-validated|well-tested|CI-verified", re.I)
BEHIND = re.compile(
    r"\b\d+\s+(?:commits?\s+)?(?:behind|ahead)\b|\b(?:behind|ahead)\s+(?:by\s+)?\d+\b",
    re.I,
)
CI_EXCUSE = re.compile(r"\b(unrelated|pre-existing|preexisting|infra)\b", re.I)
GREEN = re.compile(
    r"\b(green|all pass(?:ing|ed)?|all checks pass|CI clean|clean CI|CI pass(?:es|ing)?)\b",
    re.I,
)
RUNNING_WORD = re.compile(r"\b(running|queued|pending|in progress|in_progress)\b", re.I)
TEXT_IO = re.compile(
    r"encoding=|text I/O|text-I/O|read_text|write_text|\bopen\(\)", re.I
)
PERF = re.compile(
    r"\b(perf|throughput|TFLOPS|GB/s|latency|time limit|timeout budget|speed-?up)\b",
    re.I,
)
TEST_CLAIM = re.compile(r"pytest|passed|\bpass\b|✓|\bskip|test run|tests? pass", re.I)
RULE_CLAIM = re.compile(
    r"tracker|encoding|text I/O|perf|rule|env read|env assignment", re.I
)
PYTEST_SUMMARY = re.compile(r"\b\d+ (?:passed|failed|skipped|errors?)\b|no tests ran")
NOT_OPEN = ("FIXED", "DEFERRED", "MOOT", "ADDRESSED", "ANSWERED", "RESOLVED", "DONE")
STEP = re.compile(r"^=== step (\d+) \(t\+\d+s\) think=(\d+)c calls=(\d+) ===", re.M)
CALL = re.compile(r"^  TOOL (\w+)\((.*)\)\s*$")
NOT_DONE = (
    "queued",
    "in_progress",
    "pending",
    "waiting",
    "requested",
    "running",
    "expected",
)
FAILED = ("failure", "timed_out", "startup_failure", "action_required", "error")


def _read(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def meta_fields(text: str) -> dict:
    """meta.md's `- Key: value` lines (the first of each key), with a leading @ dropped."""
    out = {}
    for m in re.finditer(r"^- ([A-Za-z0-9 ]+): (.*)$", text, re.M):
        out.setdefault(m.group(1).strip(), m.group(2).strip().lstrip("@"))
    return out


def commits(text: str) -> list[tuple[str, str]]:
    """commits.md's (short SHA, committed date), oldest first."""
    return re.findall(r"^- ([0-9a-f]{7,40}) (\S+)", text, re.M)


def threads(text: str) -> list[dict]:
    """threads.md: each thread's comments [(login, date)], in order."""
    out = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        cs = re.findall(r"^### @(\S+) \(([^)]+)\)", block, re.M)
        if cs:
            out.append({"comments": cs})
    return out


def dated_posts(text: str) -> list[tuple[str, str]]:
    """reviews.md and comments.md: (login, date) per entry."""
    return [
        (m.group(1), m.group(2))
        for m in re.finditer(
            r"^## @(\S+)[^(\n]*\((\d{4}-\d\d-\d\dT[^)]+)\)", text, re.M
        )
    ]


def checks_counts(text: str) -> Counter:
    """checks.txt by state: ok, failed, running, other (cancelled and skipped are not failures)."""
    c = Counter()
    for line in text.splitlines()[1:]:
        state = line.split("\t", 1)[0].strip().lower()
        if not state or "\t" not in line:
            continue
        c[
            (
                "ok"
                if state == "success"
                else (
                    "failed"
                    if state in FAILED
                    else "running" if state in NOT_DONE else "other"
                )
            )
        ] += 1
    return c


def tool_calls(log: str) -> list[dict]:
    """agent_log.txt's tool calls: name, arguments (as logged) and output (the logged head), with the step."""
    calls, cur, step = [], None, -1
    for line in log.splitlines():
        m = STEP.match(line)
        if m or line.startswith("=== "):
            step = int(m.group(1)) if m else step
            cur = None
            continue
        m = CALL.match(line)
        if m:
            cur = {"name": m.group(1), "args": m.group(2), "out": "", "step": step}
            calls.append(cur)
        elif cur is not None and line.startswith("  -> "):
            cur["out"] = line[5:]
        elif cur is not None and cur["out"]:
            cur["out"] += "\n" + line
    return calls


def _table_rows(draft: str, header: str) -> list[str]:
    """The rows of the table whose header line starts with `header`."""
    lines = draft.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header):
            rows = []
            for row in lines[i + 2 :]:
                if not row.startswith("|"):
                    break
                rows.append(row)
            return rows
    return []


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _logins(first_cell: str) -> list[str]:
    """The reviewers a prior row names, before the colon that starts its items: each name before its "(" or ×,
    comma-separated."""
    names = []
    who = re.sub(r"\*\*", "", first_cell).split(":", 1)[0]
    for part in re.split(r",\s*(?![^()]*\))", who):
        name = re.split(r"[\s(×:]", part.strip(), maxsplit=1)[0].strip("`@")
        if name:
            names.append(name)
    return names


def _same_login(row_name: str, login: str) -> bool:
    a, b = row_name.lower(), login.lower()
    return a == b or (a == "copilot" and "copilot" in b)


def lint(ws: pathlib.Path, keys=()) -> list[tuple[str, str]]:
    """Flags on the draft in folder `ws`: (lesson, what), in draft order where it has one. `keys`: the tracker keys
    (config.tracker_keys) of the ticket tags L-014 compares."""
    draft = _read(ws / "review_agent.md")
    pr = ws / "pr"
    meta = meta_fields(_read(pr / "meta.md"))
    meta_text = _read(pr / "meta.md")
    commits_text = _read(pr / "commits.md")
    cs = commits(commits_text)
    th = threads(_read(pr / "threads.md"))
    posts = dated_posts(_read(pr / "reviews.md")) + dated_posts(
        _read(pr / "comments.md")
    )
    log = _read(ws / "agent_log.txt")
    calls = tool_calls(log)
    out: list[tuple[str, str]] = []
    flag = out.append
    lines = draft.splitlines()

    # L-001: verdict words, line 1, header and footer
    for i, ln in enumerate(lines, 1):
        if VERDICT.search(ln):
            flag(("L-001", f"line {i}: the verdict word {VERDICT.search(ln).group(1)}"))
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    want = meta.get("Line 1", "")
    if want and first != want:
        flag(("L-001", f"line 1 is {first[:80]!r}, not meta.md's `Line 1:`"))
    elif not want and not reviewer_name.given(first):
        flag(
            (
                "L-001",
                f"line 1 is {first[:80]!r}: no `Automated review by <automated_reviewer_name>:` line",
            )
        )
    hm = HEADER.search(draft)
    header = hm.group(0) if hm else ""
    tail = HEADER_TAIL.search(header)
    if not tail:
        flag(
            (
                "L-001",
                "the header lacks `— <🔴|🟠|🟡|🟢> First-round|Follow-up: <2–5 word reason>`",
            )
        )
    elif meta.get("Round") and tail.group(2) != meta["Round"]:
        flag(
            (
                "L-001",
                f"the header says {tail.group(2)}; meta.md's `Round:` is {meta['Round']}",
            )
        )
    fm = FOOTER.search(draft)
    if tail and fm and fm.group(1).lower() != tail.group(2).lower():
        flag(
            ("L-001", f"the footer says {fm.group(1)}; the header says {tail.group(2)}")
        )

    # L-009, L-014: author and ticket tags
    who = HEADER_WHO.search(header)
    if who and meta.get("Author") and who.group(1).strip() != meta["Author"]:
        flag(
            (
                "L-009",
                f"the header's author is {who.group(1).strip()}; meta.md's `Author:` is {meta['Author']}",
            )
        )
    tag = tag_rx(keys)
    carried = set(tag.findall(meta_text + commits_text))
    for tag in sorted(set(tag.findall(draft)) - carried):
        flag(("L-014", f"{tag} is in neither meta.md nor commits.md"))

    # words the lessons forbid
    for i, ln in enumerate(lines, 1):
        if STALE.search(ln):
            flag(
                (
                    "L-010",
                    f'line {i}: "Stale" (write `APPROVED at <short sha>`, or the item\'s state)',
                )
            )
        if "Saved to" in ln:
            flag(("L-012", f'line {i}: a "Saved to" line'))
        if BEHIND.search(ln):
            flag(
                (
                    "L-002",
                    f"line {i}: a behind/ahead count ({BEHIND.search(ln).group(0)}): no input gives one",
                )
            )
    summaries = [
        c
        for c in calls
        if c["name"] == "run_pytest" and PYTEST_SUMMARY.search(c["out"])
    ]
    if UNEARNED.search(draft) and not summaries:
        flag(("L-011", f'"{UNEARNED.search(draft).group(0)}" with no test run'))

    # L-003, L-005: "(ran)" against the tool calls
    names = {c["name"] for c in calls}
    merges = [c for c in calls if c["name"] == "merge_conflicts"]
    for i, ln in enumerate(lines, 1):
        for seg in re.split(r"<br>\s*•|\|", ln):
            if "(ran)" not in seg:
                continue
            what = re.sub(r"\s+", " ", seg.strip())[:90]
            if re.search(r"\bmerge\b", seg, re.I):
                if not merges:
                    flag(("L-003", f'line {i}: "{what}": no merge_conflicts call'))
            elif TEST_CLAIM.search(seg) and not RULE_CLAIM.search(seg):
                if not summaries:
                    flag(
                        (
                            "L-003",
                            f'line {i}: "{what}": no run_pytest call printed a pytest summary',
                        )
                    )
                else:
                    _check_counts(seg, summaries, i, what, flag)
            elif RULE_CLAIM.search(seg):
                if "rule_scan" not in names:
                    flag(("L-003", f'line {i}: "{what}": no rule_scan call'))
            elif not _backed(seg, calls):
                flag(
                    (
                        "L-003",
                        f'line {i}: "{what}": no run_pytest, rule_scan or merge_conflicts output backs it '
                        "(read_file and grep_code are (read))",
                    )
                )
    if merges:
        last = merges[-1]["out"].strip()
        clean = last in ("(no output)", "(no conflicts)", "")
        not_run = "not prepared" in last
        said_clean = re.search(r"merge:?\s*clean", draft, re.I)
        said_conflict = re.search(
            r"merge:?\s*\d+ files? conflict|files? conflict", draft, re.I
        )
        if not_run and (said_clean or said_conflict):
            flag(
                ("L-005", "the merge is stated, but merge_conflicts had no merged tree")
            )
        elif clean and said_conflict:
            flag(("L-005", "the draft names conflicts; merge_conflicts printed none"))
        elif not clean and not not_run and said_clean:
            flag(
                (
                    "L-005",
                    f"the draft says merge clean; merge_conflicts listed {len(last.splitlines())} file(s)",
                )
            )

    # L-004: Bots / CI against checks.txt
    counts = checks_counts(_read(pr / "checks.txt"))
    pending, failed = counts["running"], counts["failed"]
    for i, ln in enumerate(lines, 1):
        if ln.startswith("| Bots / CI"):
            cell = " ".join(_cells(ln)[1:])
            if pending and not RUNNING_WORD.search(cell):
                flag(
                    (
                        "L-004",
                        f"line {i}: Bots / CI does not say running; checks.txt has {pending} not completed",
                    )
                )
            if CI_EXCUSE.search(cell):
                flag(
                    (
                        "L-004",
                        f'line {i}: "{CI_EXCUSE.search(cell).group(1)}": write "cause not checked"',
                    )
                )
            for n, sym, key in (
                (r"(\d+)\s*✓", "✓", "ok"),
                (r"(\d+)\s*✗", "✗", "failed"),
                (r"(\d+)\s*(?:running|queued|pending)", "running", "running"),
            ):
                got = [int(x) for x in re.findall(n, cell)]
                if got and sum(got) != counts[key]:
                    flag(
                        (
                            "L-004",
                            f"line {i}: Bots / CI says {sum(got)} {sym}; checks.txt has {counts[key]}",
                        )
                    )
        if (
            (pending or failed)
            and GREEN.search(ln)
            and "CI" in ln
            and not ln.startswith("| Our checks")
        ):
            flag(
                (
                    "L-004",
                    f'line {i}: "{GREEN.search(ln).group(0)}" while checks.txt has {failed} failed and '
                    f"{pending} not completed",
                )
            )

    # Prior rows: L-006, L-010, L-017, L-019
    me = meta.get("Operator", "").lower()
    last_commit = max((d for _, d in cs), default="")
    for row in _table_rows(draft, "| Prior item"):
        cells = _cells(row)
        if len(cells) < 2:
            continue
        item, status = cells[0], cells[1]
        if item.startswith(("Bots / CI", "Our checks", "Code quality", "Approvals")):
            continue
        short = item[:60]
        rnames = _logins(item)
        if re.match(r"(?i)ours?\b", item) or (
            me and any(n.lower() == me for n in rnames)
        ):
            flag(
                ("L-006", f'prior row "{short}": our own items; the verifier adds them')
            )
            continue
        word = next((w for w in NOT_OPEN if w in status.upper()), "")
        times = sum(int(x) for x in TIMES.findall(item)) or 1
        if word == "FIXED":
            shas = SHA.findall(status)
            for s in shas:
                if not any(c.startswith(s[:7]) or s.startswith(c[:7]) for c, _ in cs):
                    flag(
                        (
                            "L-010",
                            f'prior row "{short}": FIXED at `{s}`, which is not in commits.md',
                        )
                    )
            if len(set(shas)) == 1 and times > 1:
                flag(("L-010", f'prior row "{short}": one SHA for {times} items'))
        mine = [
            t for t in th if any(_same_login(n, t["comments"][0][0]) for n in rnames)
        ]
        listed = sum(int(x) for x in TIMES.findall(item))
        if listed and listed > len(mine) and th:
            flag(
                (
                    "L-017",
                    f'prior row "{short}": ×{listed}, but threads.md has {len(mine)} thread(s) by '
                    f"{', '.join(rnames)}",
                )
            )
        if word and word != "APPROVED":
            raised = sorted(
                [d for t in mine for (lg, d) in t["comments"][:1]]
                + [d for lg, d in posts if any(_same_login(n, lg) for n in rnames)]
            )
            if not raised:
                continue
            first_raised = raised[0]
            replied = any(
                len(t["comments"]) > 1
                and any(
                    not _same_login(n, lg)
                    for lg, _ in t["comments"][1:]
                    for n in rnames
                )
                for t in mine
            )
            replied = replied or any(
                lg.lower() == meta.get("Author", "").lower() and d > first_raised
                for lg, d in posts
            )
            later = any(d > first_raised for _, d in cs)
            if not replied and not later:
                flag(
                    (
                        "L-019",
                        f'prior row "{short}": {word} with no reply and no later commit (OPEN)',
                    )
                )
            elif word == "FIXED" and last_commit and min(raised) > last_commit:
                flag(
                    (
                        "L-019",
                        f'prior row "{short}": FIXED, but raised after the head\'s last commit',
                    )
                )

    # L-013, L-016: findings that echo rule_scan hints
    for row in _table_rows(draft, "| # | Sev"):
        cells = _cells(row)
        if len(cells) < 4:
            continue
        text = " ".join(cells[2:])
        if TEXT_IO.search(text):
            flag(
                (
                    "L-013",
                    f"finding {cells[0]}: a text-I/O hint as a finding (leave it out)",
                )
            )
        elif PERF.search(text) and not re.search(
            r"\d+(\.\d+)?\s*(ms|us|µs|s|TFLOPS|GB/s)\b", text
        ):
            flag(
                (
                    "L-016",
                    f"finding {cells[0]}: a perf hint with no measured time or throughput",
                )
            )

    # L-101: a degenerate draft
    rows = [
        ln for ln in lines if ln.startswith("|") and not re.match(r"^\|[\s|:-]+\|$", ln)
    ]
    for row, k in Counter(rows).items():
        if k > REPEATS:
            flag(("L-101", f'a table row repeated {k} times: "{row[:60]}"'))
    firsts = Counter(_cells(r)[0] for r in rows if _cells(r)[0])
    for cell, k in firsts.items():
        if k > REPEATS and not any(
            Counter(rows)[r] > REPEATS for r in rows if _cells(r)[0] == cell
        ):
            flag(("L-101", f'a first cell repeated {k} times: "{cell[:60]}"'))
    steps = [(int(a), int(b)) for a, b, _ in STEP.findall(log)]
    if "=== finalize" in log:
        flag(
            (
                "L-101",
                "the worker was forced to finalize (no review by the end of its loop)",
            )
        )
    stats = {}
    try:
        stats = json.loads(_read(ws / "stats_agent.json") or "{}")
    except ValueError:
        pass
    if (stats.get("steps") or 0) >= MAX_STEPS:
        flag(("L-101", f"the worker used all {MAX_STEPS} steps"))
    if steps and max(t for _, t in steps) >= LONG_THINK:
        flag(("L-101", f"a step thought {max(t for _, t in steps):,} characters"))
    n_steps = len({s for s, _ in steps})
    for name in ("read_file", "grep_code"):
        seen: dict[str, set] = {}
        for c in calls:
            if c["name"] == name:
                seen.setdefault(c["args"], set()).add(c["step"])
        for args, in_steps in seen.items():
            if n_steps >= 6 and len(in_steps) > n_steps / 2:
                flag(
                    (
                        "L-101",
                        f"{name}({args[:60]}) in {len(in_steps)} of {n_steps} steps",
                    )
                )
    return out


def _check_counts(seg: str, summaries: list[dict], i: int, what: str, flag) -> None:
    """The counts a test claim gives (✓, ✗, skip) must be one pytest summary's passed, failed and skipped."""
    want = {}
    for pat, key in (
        (r"(\d+)\s*✓", "passed"),
        (r"(\d+)\s*passed", "passed"),
        (r"(\d+)\s*✗", "failed"),
        (r"(\d+)\s*failed", "failed"),
        (r"(\d+)\s*skip", "skipped"),
    ):
        m = re.search(pat, seg)
        if m:
            want[key] = int(m.group(1))
    if not want:
        return
    for c in summaries:
        got = {
            k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped)", c["out"])
        }
        if all(got.get(k, 0) == v for k, v in want.items()):
            return
    flag(("L-003", f'line {i}: "{what}": no pytest summary printed these counts'))


def _backed(seg: str, calls: list[dict]) -> bool:
    """A "(ran)" point that names a file is backed when a run tool's output names that file."""
    paths = [
        p.rsplit("/", 1)[-1].split(":", 1)[0]
        for p in re.findall(r"`([^`\s]+\.[A-Za-z]{1,4}(?::\d+)?)`", seg)
    ]
    runs = [
        c for c in calls if c["name"] in ("run_pytest", "rule_scan", "merge_conflicts")
    ]
    if not paths:
        return bool(runs)
    return any(p in c["out"] or p in c["args"] for p in paths for c in runs)


def render(n: int, flags: list[tuple[str, str]]) -> str:
    head = [
        f"# Draft lint for #{n}",
        "",
        "rk draft-lint: the draft checked against its own inputs (lessons L-501). Flags only: the draft is "
        "unchanged. Read it after your blind pass, and judge each flag yourself.",
        "",
    ]
    if not flags:
        return "\n".join(head + ["No flags.", ""])
    by = Counter(lesson for lesson, _ in flags)
    head.append(
        f"{len(flags)} flag(s): "
        + ", ".join(f"{k} ×{v}" for k, v in sorted(by.items()))
    )
    head.append("")
    return "\n".join(head + [f"- {lesson}: {what}" for lesson, what in flags] + [""])


def run(n: int, ws: pathlib.Path, keys=()) -> tuple[list[str], int, int]:
    """Lint PR n's draft in `ws` and write checks/lint.md: (the report's lines, exit code, flag count). Exit 0 no flags,
    1 flags, 2 no draft."""
    if (
        not (ws / "review_agent.md").is_file()
        or not _read(ws / "review_agent.md").strip()
    ):
        return [f"draft-lint #{n}: no draft in {ws}"], 2, 0
    flags = lint(ws, keys)
    text = render(n, flags)
    (ws / "checks").mkdir(parents=True, exist_ok=True)
    (ws / "checks" / LINT_FILE).write_text(text, encoding="utf-8")
    return text.rstrip("\n").splitlines(), (1 if flags else 0), len(flags)


def ready_note(n: int, ws: pathlib.Path | str, keys=()) -> str:
    """REVIEW_READY's words on the lint: never raises, so a lint failure never holds a review back."""
    ws = pathlib.Path(ws)
    try:
        _, code, k = run(n, ws, keys)
    except Exception as exc:  # noqa: BLE001 (the lint is advice; the review goes on)
        return f" Draft lint failed: {exc!r}."
    if code == 2:
        return ""
    return f" Draft lint: {ws}/checks/{LINT_FILE} ({k} flag{'s' if k != 1 else ''}; open it after the blind pass)."
