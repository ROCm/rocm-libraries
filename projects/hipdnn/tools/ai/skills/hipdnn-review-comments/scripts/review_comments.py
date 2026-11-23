#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build, check and record curated PR comments.

Usage: review_comments.py findings <evidence dir> <round> [--out DIR] [--head SHA]
                          [--universe FILE] [--deep-review DIR]
       review_comments.py check <out dir> [--head SHA] [--checkout DIR] [--files FILE]
                          [--terms FILE] [--evidence DIR]
       review_comments.py payload <comments.json> (<n> | --count)
       review_comments.py review-payload <out dir> --head SHA
       review_comments.py posted <out dir>

`<out dir>` holds FINDINGS.md, TRIAGE.md, COMMENTS.md and comments.json (grammar in
RUNBOOK.md *Formats*). Exits 1 with a one-line reason per failure.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[2]
SIBLING = "hipdnn-deep-review"

ID = r"(?:F\d+|[A-Z][A-Z0-9]*-\d+)"
ID_RE = re.compile(ID)
TIERS = ("BLOCKING", "IMPORTANT", "SUGGESTION", "FUTURE WORK")
BLOCKING_TIERS = TIERS[:2]
ACTIONS = ("KEEP", "REVISE", "DROP", "UNANCHORED")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
HEAD_LINE = re.compile(r"^\s*[-*]?\s*(Head|Base):\s*`?([0-9a-fA-F]{7,40})`?\s*$")
COMMENT_HEADING = re.compile(
    rf"^### (\d+)\. (P1|P2) (\S+?):(\d+)(?:-(\d+))? \(({ID}(?:,\s*{ID})*)\)\s*$"
)
COMMENT_URL = re.compile(r"https://[^/\s]+/[^/\s]+/[^/\s]+/pull/\d+#discussion_r\d+")
REVIEW_URL = re.compile(
    r"https://[^/\s]+/[^/\s]+/[^/\s]+/pull/\d+#pullrequestreview-\d+"
)
JSON_KEYS = {
    "commit_id",
    "path",
    "line",
    "side",
    "body",
    "finding_ids",
    "start_line",
    "start_side",
}


class Fail(Exception):
    """A user-facing failure; the CLI prints it and exits 1."""


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read(path):
    path = Path(path)
    if not path.is_file():
        raise Fail(f"missing {path}")
    return path.read_text(encoding="utf-8")


def is_fence(line):
    return line.lstrip().startswith("```")


def same_sha(a, b):
    a, b = a.lower(), b.lower()
    return a.startswith(b) or b.startswith(a)


def id_in(fid, text):
    return re.search(rf"(?<![\w-]){re.escape(fid)}(?![\w-])", text) is not None


def outside_code(text):
    """text without fenced blocks and inline code spans."""
    kept, fenced = [], False
    for line in text.splitlines():
        if is_fence(line):
            fenced = not fenced
        elif not fenced:
            kept.append(line)
    return re.sub(r"`[^`\n]*`", "", "\n".join(kept))


# FINDINGS.md ----------------------------------------------------------------


def parse_findings(text):
    """{ID: {claim, tier, evidence, decision}} in file order. Text before the first `## `
    heading is the preamble; any `## ` heading that is not `## <ID>. <claim>` is an error.
    """
    items, cur, fenced = {}, None, False
    for n, line in enumerate(text.splitlines(), 1):
        if is_fence(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        if line.startswith("## "):
            m = re.fullmatch(rf"## ({ID})\. (\S.*)", line.rstrip())
            if not m:
                raise Fail(
                    f"FINDINGS.md line {n}: `{line.strip()}` is not `## <ID>. <claim>`"
                )
            if m.group(1) in items:
                raise Fail(f"FINDINGS.md line {n}: duplicate ID {m.group(1)}")
            cur = items[m.group(1)] = {
                "claim": m.group(2).strip(),
                "tier": "",
                "evidence": "",
                "decision": "",
            }
            continue
        m = re.fullmatch(r"(Tier|Evidence|Decision):[ \t]*(.*)", line.rstrip())
        if cur is not None and m:
            key = m.group(1).lower()
            if cur[key]:
                raise Fail(
                    f"FINDINGS.md line {n}: second `{m.group(1)}:` line in one item"
                )
            cur[key] = m.group(2).strip()
            if key == "tier" and cur[key] not in TIERS:
                raise Fail(
                    f"FINDINGS.md line {n}: Tier `{cur[key]}` is not one of "
                    + ", ".join(TIERS)
                )
    return items


def run_universe(evidence, rnd, deep_review):
    """The sibling `comments.py universe` output, parsed."""
    root = Path(deep_review) if deep_review else SKILLS_DIR / SIBLING
    script = root / "scripts" / "comments.py"
    if not script.is_file():
        raise Fail(
            f"ledger mode needs the {SIBLING} skill next to this one (missing {script}); "
            f"install it with `python3 tools/ai/install-skills.py --target <skills dir> {SIBLING}`, "
            "or pass a FINDINGS.md instead of ledger:<evidence>:<round>"
        )
    p = subprocess.run(
        [sys.executable, str(script), "universe", str(evidence), str(rnd)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if p.returncode:
        raise Fail(
            (p.stderr.strip().splitlines() or ["comments.py universe failed"])[0]
        )
    return load_universe(p.stdout), root


def load_universe(text):
    try:
        data = json.loads(text)
    except ValueError as e:
        raise Fail(f"universe is not JSON: {e}")
    if not isinstance(data, list) or not all(
        isinstance(r, dict) and "id" in r for r in data
    ):
        raise Fail("universe must be a JSON list of objects with an `id`")
    return data


def one_line(text):
    return " ".join(str(text).split())


def render_findings(evidence, rnd, head, universe):
    lines = [
        "# Findings for PR review comments",
        "",
        f"Source: deep-review ledger `{evidence}`, round {rnd}."
        + (f" Head `{head}`." if head else ""),
        "",
    ]
    for r in universe:
        lines += [f"## {r['id']}. {one_line(r.get('claim', '')) or r['id']}"]
        if r.get("tier"):
            lines.append(f"Tier: {r['tier']}")
        if one_line(r.get("evidence", "")):
            lines.append(f"Evidence: {one_line(r['evidence'])}")
        if one_line(r.get("where", "")):
            lines.append(f"Where: {one_line(r['where'])}")
        if one_line(r.get("fix", "")):
            lines.append(f"Fix: {one_line(r['fix'])}")
        if r.get("status") and r["status"] != "OPEN":
            lines.append(f"Status: {r['status']}")
        lines.append("")
    return "\n".join(lines)


def next_run_dir(evidence, rnd):
    base = Path(evidence) / f"r{rnd}" / "comments"
    taken = [int(p.name[1:]) for p in base.glob("c*") if re.fullmatch(r"c\d+", p.name)]
    return base / f"c{max(taken, default=0) + 1}"


def cmd_findings(a):
    """Write FINDINGS.md and ledger.json from the deep-review universe and print the out dir."""
    if a.universe:
        universe, root = load_universe(read(a.universe)), Path(
            a.deep_review or SKILLS_DIR / SIBLING
        )
    else:
        universe, root = run_universe(a.evidence, a.round, a.deep_review)
    out = Path(a.out) if a.out else next_run_dir(a.evidence, a.round)
    if (out / "FINDINGS.md").exists():
        raise Fail(
            f"{out / 'FINDINGS.md'} exists; earlier runs stay untouched, use a new out dir"
        )
    out.mkdir(parents=True, exist_ok=True)
    write(out / "FINDINGS.md", render_findings(a.evidence, a.round, a.head, universe))
    write(
        out / "ledger.json",
        json.dumps(
            {
                "evidence": str(a.evidence),
                "round": a.round,
                "scripts": str(root / "scripts"),
                "universe": universe,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"out: {out}")
    print(f"findings: {len(universe)} items")


def load_ledger_run(out):
    """The ledger.json of a ledger-mode run, or None in free-form mode."""
    path = Path(out) / "ledger.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise Fail(f"{path} is not JSON: {e}")


# TRIAGE.md ------------------------------------------------------------------


def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.strip().replace("\\|", "|") for c in cells]


def parse_triage(text):
    """[(ID, anchor, action, priority, reason)] from every `ID | Anchor | Action | Priority |
    Reason` table, with action normalised (`MERGE INTO <ID>` keeps its target)."""
    rows, in_table, found = [], False, False
    for n, line in enumerate(text.splitlines(), 1):
        if "|" not in line:
            in_table = False
            continue
        cells = split_row(line)
        if not in_table:
            if (
                len(cells) == 5
                and cells[0].lower() == "id"
                and cells[2].lower().startswith("action")
            ):
                in_table = found = True
            continue
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        if len(cells) != 5:
            raise Fail(f"TRIAGE.md line {n}: expected 5 columns, found {len(cells)}")
        fid = re.sub(r"[*`]", "", cells[0]).strip()
        action = re.sub(r"[*`]", "", cells[2]).strip().upper()
        if not re.fullmatch(ID, fid):
            raise Fail(f"TRIAGE.md line {n}: `{cells[0]}` is not an ID")
        if action not in ACTIONS and not re.fullmatch(rf"MERGE INTO {ID}", action):
            raise Fail(
                f"TRIAGE.md line {n}: action `{cells[2]}` is not KEEP, MERGE INTO <ID>, REVISE, UNANCHORED or DROP"
            )
        rows.append(
            (
                fid,
                cells[1],
                action,
                re.sub(r"[*`]", "", cells[3]).strip().upper(),
                cells[4],
            )
        )
    if not found:
        raise Fail("TRIAGE.md has no `ID | Anchor | Action | Priority | Reason` table")
    return rows


# COMMENTS.md ----------------------------------------------------------------


def parse_comments(text):
    """(head, base, [comment dict], review body or None). A comment has `n`, `prio`, `path`,
    `line`, `start` (first line of a range, else None), `ids` and `body`."""
    head = base = None
    comments, review, cur, fenced = [], None, None, False
    section = "header"
    for n, line in enumerate(text.splitlines(), 1):
        if is_fence(line):
            fenced = not fenced
        elif not fenced and line.startswith("## "):
            if line.strip() != "## Review body":
                raise Fail(f"COMMENTS.md line {n}: unknown section `{line.strip()}`")
            if review is not None:
                raise Fail(f"COMMENTS.md line {n}: second Review body section")
            review, section, cur = [], "review", None
            continue
        elif not fenced and line.startswith("### "):
            m = COMMENT_HEADING.match(line)
            if not m or section == "review":
                raise Fail(
                    f"COMMENTS.md line {n}: `{line.strip()}` is not `### <n>. <P1|P2> <path>:<line>[-<line>] (<IDs>)`"
                )
            first, last = int(m.group(4)), int(m.group(5) or m.group(4))
            if last < first:
                raise Fail(
                    f"COMMENTS.md line {n}: line range {first}-{last} runs backwards"
                )
            cur = {
                "n": int(m.group(1)),
                "prio": m.group(2),
                "path": m.group(3),
                "line": last,
                "start": first if m.group(5) else None,
                "ids": [i.strip() for i in m.group(6).split(",")],
                "body": [],
            }
            comments.append(cur)
            section = "comment"
            continue
        if section == "header":
            m = HEAD_LINE.match(line)
            if m:
                if m.group(1) == "Head":
                    head = m.group(2).lower()
                else:
                    base = m.group(2).lower()
        elif section == "comment":
            cur["body"].append(line)
        else:
            review.append(line)
    if not head or not base:
        raise Fail("COMMENTS.md needs `Head:` and `Base:` lines (7-40 hex digits)")
    for i, c in enumerate(comments, 1):
        if c["n"] != i:
            raise Fail(
                f"COMMENTS.md: comment {c['n']} is out of sequence, expected {i}"
            )
        c["body"] = "\n".join(c["body"]).strip()
    return (
        head,
        base,
        comments,
        ("\n".join(review).strip() if review is not None else None),
    )


def load_comments_json(path):
    try:
        data = json.loads(read(path))
    except ValueError as e:
        raise Fail(f"{path} is not JSON: {e}")
    if not isinstance(data, list) or not all(isinstance(c, dict) for c in data):
        raise Fail(f"{path} must be a list of objects")
    return data


# Diff hunks -----------------------------------------------------------------


def parse_hunks(patch):
    """[set of new-side line numbers that are added or context lines], one per hunk."""
    hunks, lines, i = [], patch.split("\n"), 0
    while i < len(lines):
        m = HUNK.match(lines[i])
        i += 1
        if not m:
            continue
        old = int(m.group(2) if m.group(2) is not None else 1)
        new = int(m.group(4) if m.group(4) is not None else 1)
        n, valid = int(m.group(3)), set()
        while (old > 0 or new > 0) and i < len(lines):
            tag, i = lines[i][:1], i + 1
            if tag == "-":
                old -= 1
            elif tag == "+":
                valid.add(n)
                n, new = n + 1, new - 1
            elif tag in (" ", ""):
                valid.add(n)
                n, old, new = n + 1, old - 1, new - 1
        hunks.append(valid)
    return hunks


def load_patches(path):
    """{filename: patch or None} from `gh api --paginate .../pulls/<N>/files` output (one
    JSON array per page, possibly concatenated)."""
    text, pos, out = read(path), 0, {}
    dec = json.JSONDecoder()
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return out
        try:
            page, pos = dec.raw_decode(text, pos)
        except ValueError as e:
            raise Fail(f"{path} is not JSON: {e}")
        for f in page:
            out[f.get("filename")] = f.get("patch")


class Diff:
    """Hunks of one path at head, from `git diff -U3 <base>...<head>` in a checkout, else
    from a pulls/<N>/files dump."""

    def __init__(self, base, head, checkout, files):
        self.base, self.head, self.checkout = base, head, checkout
        self.patches = load_patches(files) if files else None
        self.cache = {}

    def hunks(self, path):
        """(hunks, error)."""
        if path not in self.cache:
            self.cache[path] = self._load(path)
        return self.cache[path]

    def _load(self, path):
        if self.checkout:
            p = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.checkout),
                    "-c",
                    "core.quotepath=off",
                    "diff",
                    "-U3",
                    "--no-renames",
                    f"{self.base}...{self.head}",
                    "--",
                    path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if p.returncode:
                return (
                    None,
                    f"git diff {self.base}...{self.head} failed for {path}: {p.stderr.strip()}",
                )
            return parse_hunks(p.stdout), None
        if self.patches is None:
            return None, f"no checkout: or patch file to verify the anchor in {path}"
        if path not in self.patches:
            return [], None
        if self.patches[path] is None:
            return None, f"no patch for {path} (binary or too large); pass checkout:"
        return parse_hunks(self.patches[path]), None


# check ----------------------------------------------------------------------


def norm_reason(text):
    return re.sub(r"[\s.!:;,-]+", " ", text.lower()).strip()


def check_comment_ids(comments, rows, errs):
    """IDs each comment may cite, and where each KEEP/REVISE item lands."""
    action = {r[0]: r[2] for r in rows}
    prio = {r[0]: r[3] for r in rows}
    merged = {}
    for fid, _, act, _, _ in rows:
        if act.startswith("MERGE INTO "):
            merged.setdefault(act.split()[-1], []).append(fid)
    placed = {}
    for c in comments:
        for fid in c["ids"]:
            act = action.get(fid)
            target = act.split()[-1] if act and act.startswith("MERGE INTO ") else None
            if act is None:
                errs.append(f"comment {c['n']} cites {fid}, which has no TRIAGE row")
            elif act == "DROP":
                errs.append(f"comment {c['n']} cites {fid}, which TRIAGE marks DROP")
            elif act == "UNANCHORED":
                errs.append(
                    f"comment {c['n']} cites {fid}, which TRIAGE marks UNANCHORED"
                )
            elif prio[fid] == "P3":
                errs.append(
                    f"comment {c['n']} cites {fid}, which TRIAGE marks P3 (not drafted)"
                )
            elif target and target not in c["ids"]:
                errs.append(f"comment {c['n']} cites {fid}, merged into {target}")
            else:
                placed.setdefault(fid, []).append(c["n"])
                if act in ("KEEP", "REVISE") and prio[fid] != c["prio"]:
                    errs.append(
                        f"comment {c['n']} is {c['prio']} but TRIAGE gives {fid} {prio[fid]}"
                    )
    for fid, act in action.items():
        if act in ("KEEP", "REVISE"):
            hits = placed.get(fid, [])
            if not hits and prio[fid] == "P3":
                continue
            if not hits:
                errs.append(f"{fid} is {act} in TRIAGE but in no comment")
            elif len(hits) > 1:
                errs.append(
                    f"{fid} is in comments {', '.join(map(str, hits))}; one comment per item"
                )
            for m in merged.get(fid, []):
                if not any(m in c["ids"] for c in comments if fid in c["ids"]):
                    errs.append(
                        f"{m} merges into {fid} but is missing from {fid}'s comment IDs"
                    )
        elif act.startswith("MERGE INTO "):
            target = act.split()[-1]
            if action.get(target) not in ("KEEP", "REVISE"):
                errs.append(f"{fid} merges into {target}, which is not KEEP or REVISE")


def check_json(comments, data, head, errs):
    if len(data) != len(comments):
        errs.append(
            f"comments.json has {len(data)} comments, COMMENTS.md has {len(comments)}"
        )
    for c, j in zip(comments, data):
        n = c["n"]
        for extra in sorted(set(j) - JSON_KEYS):
            errs.append(f"comments.json comment {n}: unknown field {extra}")
        want = {
            "path": c["path"],
            "line": c["line"],
            "side": "RIGHT",
            "body": c["body"],
            "finding_ids": c["ids"],
        }
        if c["start"] is not None:
            want |= {"start_line": c["start"], "start_side": "RIGHT"}
        for key, value in want.items():
            got = j.get(key)
            if key == "body" and isinstance(got, str):
                got = got.strip()
            if got != value:
                errs.append(
                    f"comments.json comment {n}: {key} differs from COMMENTS.md"
                )
        if c["start"] is None:
            for key in ("start_line", "start_side"):
                if key in j:
                    errs.append(
                        f"comments.json comment {n}: {key} on a single-line comment"
                    )
        if not isinstance(j.get("commit_id"), str) or not same_sha(
            j["commit_id"], head
        ):
            errs.append(f"comments.json comment {n}: commit_id is not the head {head}")


def check_anchors(comments, diff, errs):
    for c in comments:
        hunks, err = diff.hunks(c["path"])
        if err:
            errs.append(f"comment {c['n']}: {err}")
            continue
        first = c["start"] if c["start"] is not None else c["line"]
        if not any(
            first in h
            and c["line"] in h
            and all(x in h for x in range(first, c["line"] + 1))
            for h in hunks
        ):
            errs.append(
                f"comment {c['n']}: {c['path']}:{first}"
                + (f"-{c['line']}" if first != c["line"] else "")
                + f" is not added or context lines of one hunk at {diff.head}"
            )


def check_text(label, text, all_ids, own, terms, evidence, errs):
    for fid in own:
        if id_in(fid, text):
            errs.append(f"{label} contains its own heading ID {fid}")
    bare = outside_code(text)
    for fid in sorted(all_ids - set(own)):
        if id_in(fid, bare):
            errs.append(f"{label} names run ID {fid} outside backticks")
    lower = text.lower()
    for term in terms:
        if term.lower() in lower:
            errs.append(f"{label} contains a denied term")
    for token in (
        "$EVIDENCE",
        *([str(evidence), str(Path(evidence).resolve())] if evidence else []),
    ):
        if token in text:
            errs.append(f"{label} contains an evidence path")
            break


def collect_check(a):
    out = Path(a.out)
    findings = parse_findings(read(out / "FINDINGS.md"))
    rows = parse_triage(read(out / "TRIAGE.md"))
    chead, base, comments, review = parse_comments(read(out / "COMMENTS.md"))
    data = load_comments_json(out / "comments.json")
    ledger = load_ledger_run(out)
    head = (a.head or chead).lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", head):
        raise Fail(f"--head `{head}` is not 7-40 hex digits")
    errs = []
    if not same_sha(head, chead):
        errs.append(
            f"COMMENTS.md Head: {chead} is not the head {head}; re-run from stage 2"
        )

    universe = {r["id"]: r for r in ledger["universe"]} if ledger else {}
    for fid in universe:
        if fid not in findings:
            errs.append(f"ledger item {fid} is missing from FINDINGS.md")
    for fid, item in findings.items():
        if not item["evidence"] and not (
            fid in universe and not one_line(universe[fid].get("evidence", ""))
        ):
            errs.append(f"{fid} has no Evidence: line")

    seen, by_id = {}, {}
    for fid, anchor, act, prio, reason in rows:
        if fid in by_id:
            errs.append(f"{fid} has two TRIAGE rows")
        by_id[fid] = act
        if fid not in findings:
            errs.append(f"TRIAGE row {fid} is not a FINDINGS item")
        norm = norm_reason(reason)
        named = act.startswith("MERGE INTO ") and id_in(act.split()[-1], reason)
        if not norm or norm in ("nit", "nits"):
            errs.append(f"{fid} has an empty or bare-nit reason")
        elif not named:
            if norm in seen:
                errs.append(f"{fid} repeats the reason of {seen[norm]}")
            seen.setdefault(norm, fid)
        if act == "DROP" and fid in findings:
            item = findings[fid]
            if item["tier"] in BLOCKING_TIERS and not item["decision"]:
                errs.append(
                    f"{fid} is a {item['tier']} item dropped without a Decision: line"
                )
    for fid in findings:
        if fid not in by_id:
            errs.append(f"{fid} has no TRIAGE row")

    check_comment_ids(comments, rows, errs)
    check_json(comments, data, head, errs)
    check_anchors(comments, Diff(base, head, a.checkout, a.files), errs)

    terms = (
        [t.strip() for t in read(a.terms).splitlines() if t.strip()] if a.terms else []
    )
    ids = set(findings)
    for c in comments:
        check_text(
            f"comment {c['n']}", c["body"], ids, c["ids"], terms, a.evidence, errs
        )
    if review is not None:
        if not review:
            errs.append("the Review body section is empty")
        check_text("review body", review, ids, [], terms, a.evidence, errs)
    return errs, len(comments), len(findings)


def cmd_check(a):
    """Run every gate rule over <out dir>; print each failure and exit 1, else print a summary."""
    errs, ncomments, nitems = collect_check(a)
    if errs:
        raise Fail("\n".join(errs))
    print(f"check: ok, {ncomments} comments for {nitems} items")


# payloads and posted --------------------------------------------------------


def cmd_payload(a):
    """Print comment <n> (1-based) of comments.json without finding_ids, or the count."""
    data = load_comments_json(a.comments)
    if a.count:
        print(len(data))
        return
    if a.n is None or not 1 <= a.n <= len(data):
        raise Fail(f"give a comment number 1-{len(data)} or --count")
    print(json.dumps({k: v for k, v in data[a.n - 1].items() if k != "finding_ids"}))


def cmd_review_payload(a):
    """Print the COMMENT review that posts the Review body."""
    _, _, _, review = parse_comments(read(Path(a.out) / "COMMENTS.md"))
    if not review:
        raise Fail("COMMENTS.md has no Review body")
    print(json.dumps({"commit_id": a.head, "event": "COMMENT", "body": review}))


def read_urls(path, count):
    """({comment number: html_url}, review URL or None) from POSTED.urls lines
    `<n> <html_url>` and `review <html_url>`."""
    urls, review = {}, None
    if not Path(path).is_file():
        return urls, review
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        key, _, url = line.strip().partition(" ")
        url = url.strip()
        if key == "review":
            if not REVIEW_URL.fullmatch(url):
                raise Fail(
                    f"{path} line {n}: not a review web URL (.../pull/<N>#pullrequestreview-<id>)"
                )
            review = url
        elif key.isdigit() and 1 <= int(key) <= count:
            if not COMMENT_URL.fullmatch(url):
                raise Fail(
                    f"{path} line {n}: not a web comment URL (.../pull/<N>#discussion_r<id>)"
                )
            if int(key) in urls:
                raise Fail(f"{path} line {n}: comment {key} listed twice")
            urls[int(key)] = url
        else:
            raise Fail(
                f"{path} line {n}: expected `<comment number> <html_url>` or `review <html_url>`"
            )
    return urls, review


def cmd_posted(a):
    """Write POSTED.md and, in ledger mode, posted.tsv from POSTED.urls; print the link command."""
    out = Path(a.out)
    head, _, comments, review = parse_comments(read(out / "COMMENTS.md"))
    urls, review_url = read_urls(out / "POSTED.urls", len(comments))
    lines = ["# Posted comments", "", f"Head: `{head}`", ""]
    for c in comments:
        where = (
            f"{c['path']}:{c['start']}-{c['line']}"
            if c["start"]
            else f"{c['path']}:{c['line']}"
        )
        lines.append(
            f"- {c['n']} `{where}` ({', '.join(c['ids'])}) — {urls.get(c['n'], 'not posted')}"
        )
    if review:
        lines += ["", f"## Review body — {review_url or 'not posted'}", "", review]
    write(out / "POSTED.md", "\n".join(lines) + "\n")
    print(
        f"posted: {len(urls)} of {len(comments)} comments"
        + (
            f", review body {'posted' if review_url else 'not posted'}"
            if review
            else ""
        )
    )
    ledger = load_ledger_run(out)
    if not ledger:
        return
    known = {r["id"] for r in ledger["universe"]}
    tsv = []
    for c in comments:
        ids = [i for i in c["ids"] if i in known]
        if c["n"] in urls and ids:
            tsv.append(f"{urls[c['n']]}\t{','.join(ids)}")
    write(out / "posted.tsv", "".join(line + "\n" for line in tsv))
    link = Path(ledger["scripts"]) / "ledger.py"
    print("link the new comments to the ledger:")
    print(
        f"python3 {link} link {ledger['evidence']} {ledger['round']} --from {out / 'posted.tsv'}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        s = sub.add_parser(name, help=fn.__doc__.split(". ")[0].strip("."))
        s.set_defaults(fn=fn)
        return s

    s = add("findings", cmd_findings)
    s.add_argument("evidence")
    s.add_argument("round", type=int)
    s.add_argument("--out")
    s.add_argument("--head")
    s.add_argument(
        "--universe",
        metavar="FILE",
        help="universe JSON instead of running the sibling skill",
    )
    s.add_argument(
        "--deep-review",
        metavar="DIR",
        help=f"the {SIBLING} skill directory (default: the sibling)",
    )
    s = add("check", cmd_check)
    s.add_argument("out")
    s.add_argument(
        "--head", help="head SHA to check against (default: COMMENTS.md Head:)"
    )
    s.add_argument("--checkout", help="git checkout holding base and head")
    s.add_argument(
        "--files", metavar="FILE", help="`gh api --paginate .../pulls/<N>/files` output"
    )
    s.add_argument(
        "--terms", metavar="FILE", help="one fixed string per line that must not appear"
    )
    s.add_argument("--evidence", help="evidence dir whose path must not appear")
    s = add("payload", cmd_payload)
    s.add_argument("comments", metavar="comments.json")
    s.add_argument("n", nargs="?", type=int)
    s.add_argument("--count", action="store_true")
    s = add("review-payload", cmd_review_payload)
    s.add_argument("out")
    s.add_argument("--head", required=True)
    s = add("posted", cmd_posted)
    s.add_argument("out")
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except Fail as e:
        print(f"review_comments.py {args.cmd}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
