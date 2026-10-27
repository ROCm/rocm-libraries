# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Parsers and table I/O shared by ledger.py and comments.py.

The formats are the ones in findings.md (finding block, lane report, verifier
file) and report.md (PIN.md, LEDGER.md, ID-MAP.md).
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

TIERS = ("BLOCKING", "IMPORTANT", "SUGGESTION", "FUTURE WORK")
FINDING_LABELS = ("ID", "Kind", "Tier", "Where", "Expected", "Actual", "Proof", "Fix")
VERIFIER_LABELS = (
    "ID",
    "Verdict",
    "Tier",
    "Claim",
    "Reason",
    "Proof",
    "Class",
    "Not established",
)
VERDICTS = ("CONFIRMED", "PARTIAL", "DOWNGRADED", "UPGRADED", "REFUTED", "UNVERIFIED")
SECTIONS = (
    "Findings",
    "Coverage",
    "Checked, no finding",
    "Handoffs",
    "Approval requests",
    "Needs human decision",
)
READ_DEPTHS = ("full", "diff-only", "sampled")


class Fail(Exception):
    """A user-facing failure; the CLI prints it as one line and exits 1."""


def write(path, text):
    """Write text as UTF-8 with LF newlines. The bytes go to a temp file that then replaces
    `path`, so a failed encode or write never truncates an existing file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def run(cmd, *, cwd=None, stdin=None):
    """Run a command with UTF-8 bytes on stdin and stdout (never the locale codec, never CRLF
    translation). Returns (returncode, stdout, stderr) as str; undecodable bytes become U+FFFD.
    """
    r = subprocess.run(
        cmd,
        cwd=cwd,
        input=None if stdin is None else stdin.encode("utf-8"),
        capture_output=True,
    )
    return (
        r.returncode,
        r.stdout.decode("utf-8", errors="replace"),
        r.stderr.decode("utf-8", errors="replace"),
    )


def git(clone, *args, cwd=None, stdin=None):
    """Run `git [-C clone] -c core.quotepath=off args`; returns (returncode, stdout, stderr)."""
    cmd = (
        ["git", "-c", "core.quotepath=off"]
        + (["-C", str(clone)] if clone else [])
        + list(args)
    )
    return run(cmd, cwd=cwd, stdin=stdin)


def round_dir(evidence, rnd):
    path = Path(evidence) / f"r{rnd}"
    if not path.is_dir():
        raise Fail(f"no such round directory: {path}")
    return path


def round_dirs(evidence):
    """Existing `r<N>` directories of the evidence dir in numeric order."""
    found = [
        p
        for p in Path(evidence).iterdir()
        if re.fullmatch(r"r\d+", p.name) and p.is_dir()
    ]
    return sorted(found, key=lambda p: int(p.name[1:]))


def pin_value(round_path, key):
    """Value of the first `<key>: <value>` line of PIN.md, or None."""
    pin = Path(round_path) / "PIN.md"
    if not pin.is_file():
        raise Fail(f"missing {pin}")
    for line in pin.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + ":"):
            return line[len(key) + 1 :].strip()
    return None


def is_fence(line):
    return line.lstrip().startswith("```")


def _scan_fields(text, labels, first):
    """(blocks, skipped) of parse_fields; skipped counts `<first>:` lines inside fenced code."""
    label_re = re.compile(
        r"^(" + "|".join(re.escape(x) for x in labels) + r"):[ \t]*(.*)$"
    )
    lines = text.splitlines()
    blocks, cur, field, fenced, wrapped, skipped = [], None, None, False, False, 0
    for i, line in enumerate(lines):
        if is_fence(line):
            bare = line.strip() == "```"
            if wrapped and not fenced and bare:
                wrapped = False
                continue
            if not wrapped and not fenced:
                nxt = next((x for x in lines[i + 1 :] if x.strip()), "")
                if label_re.match(nxt) and label_re.match(nxt).group(1) == first:
                    wrapped = True
                    continue
        m = None if fenced else label_re.match(line)
        if m:
            if m.group(1) == first:
                cur = {}
                blocks.append(cur)
            if cur is not None:
                field = m.group(1)
                cur[field] = m.group(2)
        elif cur is not None and field:
            cur[field] += "\n" + line
        if fenced and label_re.match(line) and line.startswith(f"{first}:"):
            skipped += 1
        if is_fence(line):
            fenced = not fenced
    for block in blocks:
        for key in block:
            block[key] = block[key].strip()
    return blocks, skipped


def parse_fields(text, labels, first="ID"):
    """Labelled-line blocks: a `<first>:` line starts a block; a later labelled line
    starts the next field; any other line (blank lines and fenced code included)
    continues the current field. A fence directly before a `<first>:` line wraps the
    blocks up to the next bare fence outside nested code; neither fence is read.
    Returns a list of {label: value} dicts."""
    return _scan_fields(text, labels, first)[0]


def unparsed_ids(text):
    """Number of `ID:` lines of a lane report's Findings section that parse_findings
    cannot read: those inside a code fence that does not wrap whole blocks."""
    return _scan_fields(split_sections(text).get("Findings", ""), FINDING_LABELS, "ID")[
        1
    ]


def split_sections(text):
    """Lane report sections by their (optionally marked-up) names, e.g. `## Coverage`,
    `**Coverage:**`, `Coverage:`. Returns {name: text}."""
    names = "|".join(re.escape(s) for s in SECTIONS)
    head = re.compile(rf"^\s*(?:#{{1,6}}\s+)?(?:\*\*)?({names}):?(?:\*\*)?:?\s*$")
    out, cur, fenced = {}, None, False
    for line in text.splitlines():
        m = None if fenced else head.match(line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur:
            out[cur].append(line)
        if is_fence(line):
            fenced = not fenced
    return {k: "\n".join(v) for k, v in out.items()}


def report_kind(text):
    """`NA` or `STOP` for a report that is only that line, else `report`."""
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            if line.startswith("N/A:"):
                return "NA"
            if line.startswith("STOP:"):
                return "STOP"
            break
    return "report"


def parse_findings(text):
    return [
        b
        for b in parse_fields(split_sections(text).get("Findings", ""), FINDING_LABELS)
        if "ID" in b
    ]


def coverage_rows(text):
    """(path, read depth cell, tests) per table row of a lane report's Coverage section;
    the header and separator rows come back too, with their text as the depth cell."""
    rows = []
    for line in split_sections(text).get("Coverage", "").splitlines():
        cells = [
            c.strip().strip("`")
            for c in re.sub(r"^\s*(?:[-*]\s+)?\|?", "", line).rstrip("|").split("|")
        ]
        if len(cells) >= 2:
            rows.append((cells[0], cells[1], " | ".join(cells[2:]).strip() or "-"))
    return rows


def parse_coverage(text):
    """(path, read depth, tests) rows of a lane report's Coverage section; rows whose
    depth is not a read depth (the header, the separator) are skipped."""
    return [r for r in coverage_rows(text) if r[1] in READ_DEPTHS]


M_LINE = re.compile(
    r"^\s*(?:[-*]\s+)?(?:\*\*)?M([1-5]):(?:\*\*)?\s*(PASS|FAIL|WAIVED\([^)]*\)|N/A|UNVERIFIED)[.,;:]?(?=\s|$)\s*(.*)$"
)


def parse_mlines(text):
    """(rule number, status, evidence) per `M<n>:` line of Checked, no finding."""
    return [
        (int(m.group(1)), m.group(2), m.group(3).strip())
        for line in split_sections(text).get("Checked, no finding", "").splitlines()
        if (m := M_LINE.match(line))
    ]


def parse_verifier(text):
    """The first verifier block of a verify file, or {}."""
    blocks = parse_fields(text, VERIFIER_LABELS)
    return blocks[0] if blocks else {}


class Table:
    def __init__(self, title, cols, rows=None):
        self.title, self.cols, self.rows = title, list(cols), rows or []
        self.sep_pending = False

    def add(self, **cells):
        self.rows.append({c: cells.get(c, "") for c in self.cols})


def _cells(line):
    inner = line.strip()[1:-1] if line.strip().endswith("|") else line.strip()[1:]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]


def parse_tables(text, what="file"):
    """Markdown headings each followed by one pipe table. Any other non-blank line is
    an error: a row in another format is a candidate, not a record."""
    tables, title, cur = [], None, None
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            title, cur = s.lstrip("#").strip(), None
        elif s.startswith("|"):
            cells = _cells(s)
            if cur is None:
                cur = Table(title, cells)
                tables.append(cur)
                cur.sep_pending = True
            elif cur.sep_pending:
                cur.sep_pending = False
            else:
                cells += [""] * (len(cur.cols) - len(cells))
                cur.rows.append(dict(zip(cur.cols, cells)))
        else:
            raise Fail(f"{what} line {n}: not a heading or table row")
    return tables


def render_tables(tables):
    def esc(v):
        return str(v).replace("|", "\\|").replace("\n", " ")

    out = []
    for t in tables:
        out += [
            f"## {t.title}",
            "",
            "| " + " | ".join(t.cols) + " |",
            "|" + "---|" * len(t.cols),
        ]
        out += [
            "| " + " | ".join(esc(r.get(c, "")) for c in t.cols) + " |" for r in t.rows
        ]
        out.append("")
    return "\n".join(out)


def find_table(tables, title):
    for t in tables:
        if t.title == title:
            return t
    raise Fail(f"missing table '{title}'")


def tier_pair(cell):
    """(reported, final) from a `<reported> -> <final>` cell."""
    reported, _, final = cell.partition("->")
    return reported.strip(), (final.strip() or reported.strip())


def lane_files(round_path):
    """Lane report files of a round (lanes/*.md and the verifier batch files
    verify/VER<batch>.md), by name."""
    rp = Path(round_path)
    return sorted(rp.glob("lanes/*.md")) + sorted(rp.glob("verify/VER[0-9]*.md"))
