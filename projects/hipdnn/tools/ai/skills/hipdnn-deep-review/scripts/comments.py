#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Draft COMMENTS.md for a deep review.

Usage: comments.py draft <evidence dir> <round>
       comments.py check-sweep <evidence dir> <round>

Exits 1 with a one-line reason on failure.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402
import partition  # noqa: E402
import reviewlib as rl  # noqa: E402

HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.M)
ANCHOR = re.compile(r"^\s*`?([^\s:`,$]+):(\d+)")
# `$HEAD:path:line` anchors a head-side line; `$BASE:` content is removed, so it stays General.
HEAD_ANCHOR = re.compile(r"^(\s*`?)\$HEAD:")
BASE_ANCHOR = re.compile(r"^\s*`?\$BASE:([^\s:`,$]+):(\d+)")
# Review-process references that mean nothing to the PR author: provisional IDs, lane
# reports, evidence paths and the $HEAD:/$BASE: prefixes of inline references.
_LANES = "|".join(
    ["CORR", "SCOUT", "SYNTH", "VER", "ORCH", *sorted(partition.CONCERN_LANES)]
)
_REF = (
    r"\$EVIDENCE/\S*|\br\d+/lanes/[\w.-]+\.md"
    rf"|\b(?:{_LANES})\d*(?:-f\d+)?-p\d+\b|\$(?:HEAD|BASE):"
)
PROCESS_REFS = re.compile(rf"\(\s*(?:{_REF})\s*\)|{_REF}")


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
    """text without code fences and review-process references (final IDs stay), and
    without the spacing those leave behind."""
    kept = [line for line in text.splitlines() if not rl.is_fence(line)]
    text = "\n".join(kept).replace("```", "")
    cleaned = PROCESS_REFS.sub("", text)
    if cleaned != text:
        cleaned = re.sub(r"(?<=\S)[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


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


def unverified_note(evidence, fid):
    """` (UNVERIFIED: <reason>)` from the newest verify file of fid (the `-2` file first),
    capped like a ledger Claim; ` (UNVERIFIED)` when no file gives a Reason."""
    for rp in reversed(rl.round_dirs(evidence)):
        for name in (f"{fid}-2.md", f"{fid}.md"):
            path = rp / "verify" / name
            if path.is_file():
                reason = ledger.cap(
                    strip_fences(
                        rl.parse_verifier(path.read_text(encoding="utf-8")).get(
                            "Reason", ""
                        )
                    )
                )
                return f" (UNVERIFIED: {reason})" if reason else " (UNVERIFIED)"
    return " (UNVERIFIED)"


def cmd_draft(a):
    """Write r<round>/COMMENTS.md from the OPEN ledger rows and the UNVERIFIED BLOCKING and
    IMPORTANT rows (their heading or item carries `(UNVERIFIED: <reason>)`, the Reason of
    the finding's verify file). Needs r<round>/ID-MAP.md. An OPEN BLOCKING or IMPORTANT
    finding is an inline `### <path>:<line> (<ID>, <tier>)` entry (Actual, Expected, Fix)
    when its first `Where` location is `path:line` inside a changed new-side hunk between
    PIN.md's BASE and HEAD; otherwise (line outside every hunk, `$BASE:` removed content,
    another anchor form) it is a `## General` item with one `path:line` anchor. A `$HEAD:`
    prefix is dropped before the anchor match. OPEN SUGGESTION and FUTURE WORK findings are
    one General item per tier and kind. Pasted text loses fenced code, provisional IDs, lane
    reports, `$EVIDENCE` paths and `$HEAD:`/`$BASE:` prefixes."""
    rp = rl.round_dir(a.evidence, a.round)
    if not (rp / "ID-MAP.md").is_file():
        raise rl.Fail(f"missing {rp / 'ID-MAP.md'}; run ledger.py ids")
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
        fid, tier = row["ID"], rl.tier_pair(row["Tier"])[1]
        if row["Status"] != "OPEN" and not (
            row["Status"] == "UNVERIFIED" and tier in rl.TIERS[:2]
        ):
            continue
        note = unverified_note(a.evidence, fid) if row["Status"] != "OPEN" else ""
        block = blocks.get(fid, {})
        claim = strip_fences(row["Claim"])
        where = row["Where"]
        m = ANCHOR.match(HEAD_ANCHOR.sub(r"\1", where))
        removed = BASE_ANCHOR.match(where)
        if m:
            anchor = f"{m.group(1)}:{m.group(2)}"
        elif removed:
            anchor = f"{removed.group(1)}:{removed.group(2)} (removed)"
        else:
            anchor = re.split(r"[,;]\s*", strip_fences(where))[0]
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
                        f"### {path}:{line} ({fid}, {tier}){note}\n\n"
                        + "\n\n".join(body),
                    )
                )
            else:
                general.append(
                    f"- {anchor or fid} ({fid}, {tier}; not in a changed hunk){note}: {claim}"
                    + (f" Fix: {fix}" if fix else "")
                )
        else:
            themes.setdefault((tier, row["Kind"]), []).append(
                f"{claim} (`{strip_fences(where)}`, {fid})"
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


def sweep_inputs(evidence, rp):
    """Every file the LEAK output sweep reads (lanes/leaks.md *Sweeping review output*):
    the round's REPORT.md, COMMENTS.md and CONTEXT.md, LEDGER.md, the `.md` files under
    orient/ and verify/ outside `*-evidence/`, and the published `$KNOWLEDGE/$KEY-r<N>/`
    copy."""
    files = [rp / n for n in ("REPORT.md", "COMMENTS.md", "CONTEXT.md")]
    files.append(Path(evidence) / "LEDGER.md")
    for name in files:
        if not name.is_file():
            raise rl.Fail(f"sweep input missing: {name}")
    for sub in ("orient", "verify"):
        files += [
            p
            for p in sorted((rp / sub).rglob("*.md"))
            if not any(part.endswith("-evidence") for part in p.relative_to(rp).parts)
        ]
    knowledge, key = rl.pin_value(rp, "KNOWLEDGE"), rl.pin_value(rp, "KEY")
    if not knowledge or not key:
        raise rl.Fail("PIN.md lacks KNOWLEDGE or KEY")
    copy = sorted((Path(knowledge) / f"{key}-{rp.name}").rglob("*.md"))
    if not copy:
        raise rl.Fail(f"missing published copy {Path(knowledge) / f'{key}-{rp.name}'}")
    return files + copy


def cmd_check_sweep(a):
    """Pass when lanes/LEAK-sweep.md says `Sweep: clean` and is newer than every file the
    sweep reads (sweep_inputs): a later edit leaves the sweep stale, not clean."""
    rp = rl.round_dir(a.evidence, a.round)
    sweep = rp / "lanes" / "LEAK-sweep.md"
    if not sweep.is_file():
        raise rl.Fail(f"missing {sweep}")
    if not re.search(r"^\W*Sweep: clean\b", sweep.read_text(encoding="utf-8"), re.M):
        raise rl.Fail("LEAK-sweep.md does not say 'Sweep: clean'")
    stamp = sweep.stat().st_mtime_ns
    stale = [p for p in sweep_inputs(a.evidence, rp) if p.stat().st_mtime_ns >= stamp]
    if stale:
        more = f" (+{len(stale) - 1} more)" if len(stale) > 1 else ""
        raise rl.Fail(f"sweep is stale: {stale[0]} is not older than it{more}")
    print("check-sweep: clean and current")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("draft", help=cmd_draft.__doc__.split(". ")[0])
    s.add_argument("evidence")
    s.add_argument("round", type=int)
    s.set_defaults(fn=cmd_draft)
    s = sub.add_parser("check-sweep", help=cmd_check_sweep.__doc__.split(". ")[0])
    s.add_argument("evidence")
    s.add_argument("round", type=int)
    s.set_defaults(fn=cmd_check_sweep)
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except rl.Fail as e:
        print(f"comments.py {args.cmd}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
