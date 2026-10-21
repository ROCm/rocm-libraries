#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Draft COMMENTS.md for a deep review.

Usage: comments.py draft <evidence dir> <round>

Exits 1 with a one-line reason on failure.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reviewlib as rl  # noqa: E402

HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.M)
ANCHOR = re.compile(r"^\s*`?([^\s:`,$]+):(\d+)")


def new_side_ranges(clone, base, head, path):
    """Inclusive (first, last) new-side line ranges of the changed hunks of path between
    base and head; a hunk with count 0 (pure deletion) is an empty range."""
    code, out, err = rl.git(
        clone, "diff", "-U0", "--no-renames", base, head, "--", path
    )
    if code:
        raise rl.Fail(f"git diff failed for {path}: {err.strip()}")
    ranges = []
    for m in HUNK.finditer(out):
        start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        if count:
            ranges.append((start, start + count - 1))
    return ranges


def strip_fences(text):
    kept = [line for line in text.splitlines() if not rl.is_fence(line)]
    return "\n".join(kept).replace("```", "").strip()


def finding_blocks(evidence):
    """{final ID: finding block} from every round's ID-MAP.md and lane reports."""
    out = {}
    for rp in rl.round_dirs(evidence):
        map_path = rp / "ID-MAP.md"
        if not map_path.is_file():
            continue
        blocks = {
            f["ID"]: f
            for p in rl.lane_files(rp)
            for f in rl.parse_findings(p.read_text(encoding="utf-8"))
        }
        for row in rl.parse_tables(map_path.read_text(encoding="utf-8"), "ID-MAP.md")[
            0
        ].rows:
            block = blocks.get(row["Lane / provisional"].rsplit("/", 1)[-1])
            if block:
                out[row["Final ID"]] = block
    return out


def cmd_draft(a):
    """Write r<round>/COMMENTS.md from the OPEN ledger rows. An OPEN BLOCKING or IMPORTANT
    finding is an inline `### <path>:<line> (<ID>, <tier>)` entry (Actual, Expected, Fix)
    when its first `Where` location is `path:line` inside a changed new-side hunk between
    PIN.md's BASE and HEAD; otherwise (line outside every hunk, `$BASE:` removed content,
    another anchor form) it is a `## General` item. OPEN SUGGESTION and FUTURE WORK findings
    are one General item per tier and kind. Fenced code is dropped from pasted text."""
    rp = rl.round_dir(a.evidence, a.round)
    pins = {k: rl.pin_value(rp, k) for k in ("CLONE", "BASE", "HEAD")}
    if not all(pins.values()):
        raise rl.Fail("PIN.md lacks CLONE, BASE or HEAD")
    tables = rl.parse_tables(
        (Path(a.evidence) / "LEDGER.md").read_text(encoding="utf-8"), "LEDGER.md"
    )
    blocks = finding_blocks(a.evidence)
    ranges = {}
    inline, general, themes = [], [], {}
    for row in rl.find_table(tables, "Findings").rows:
        if row["Status"] != "OPEN":
            continue
        fid, tier = row["ID"], rl.tier_pair(row["Tier"])[1]
        block = blocks.get(fid, {})
        claim = strip_fences(row["Claim"])
        where = row["Where"]
        m = ANCHOR.match(where)
        if tier in rl.TIERS[:2]:
            inside = False
            if m:
                path, line = m.group(1), int(m.group(2))
                if path not in ranges:
                    ranges[path] = new_side_ranges(
                        pins["CLONE"], pins["BASE"], pins["HEAD"], path
                    )
                inside = any(lo <= line <= hi for lo, hi in ranges[path])
            fix = strip_fences(block.get("Fix", ""))
            if inside:
                narrowed = row["Verdict"] == "PARTIAL" or "Actual" not in block
                body = [
                    f"Actual: {claim if narrowed else strip_fences(block['Actual'])}"
                ]
                if block.get("Expected"):
                    body.append(f"Expected: {strip_fences(block['Expected'])}")
                if fix:
                    body.append(f"Fix: {fix}")
                inline.append(
                    (
                        rl.TIERS.index(tier),
                        path,
                        line,
                        f"### {path}:{line} ({fid}, {tier})\n\n" + "\n\n".join(body),
                    )
                )
            else:
                general.append(
                    f"- {where or fid} ({fid}, {tier}; not in a changed hunk): {claim}"
                    + (f" Fix: {fix}" if fix else "")
                )
        else:
            themes.setdefault((tier, row["Kind"]), []).append(
                f"{claim} (`{where}`, {fid})"
            )
    out = ["# Draft comments (nothing posted)", ""]
    for entry in sorted(inline, key=lambda e: e[:3]):
        out += [entry[3], ""]
    out += ["## General", ""] + general
    out += [
        f"- **{tier} {kind}**: " + "; ".join(items)
        for (tier, kind), items in sorted(
            themes.items(), key=lambda kv: (rl.TIERS.index(kv[0][0]), kv[0][1])
        )
    ]
    rl.write(rp / "COMMENTS.md", "\n".join(out) + "\n")
    print(
        f"comments: {len(inline)} inline, {len(general)} general findings, {len(themes)} themes"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("draft", help=cmd_draft.__doc__.split(". ")[0])
    s.add_argument("evidence")
    s.add_argument("round", type=int)
    s.set_defaults(fn=cmd_draft)
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except rl.Fail as e:
        print(f"comments.py {args.cmd}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
