# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The fix-line premise of `rk post --premises` (lessons L-578, L-158): a fix line is advice the author applies
literally, so the ones that decide whether CI runs a test are checked before the post.

For each 🔴/🟠 row of the post's findings tables (`| N<k> | 🔴 | file:line | item |`, F rows too), its fix (the item's
`• fix:` bullet, or the text after its last `→`):
- a skip guard on a module CI lacks (`importorskip("torch")`, a `skipif` naming one; config `ci_absent_modules`) stops
  the post: CI would never run the test (move the import into the test that needs it, or skip per test);
- a claim that CI still runs the test ("CI still runs it") is checked with `rk lane-of --collect` at the head on the
  test files the row names: a verdict other than runs stops the post with lane-of's line;
- any other skip guard, a tier change or a test move needs a `fixcheck: ID ran|read HOW` line in the verifier's
  notes (what was run or read to settle that CI still runs it), as does a CI claim lane-of could not answer.
Each check prints one line; the first that fails stops the post.
"""

from __future__ import annotations

import re

from . import lanes

ROW = re.compile(r"^\|\s*([NF]\d+)\s*\|\s*(🔴|🟠)\s*\|")
FIX = re.compile(r"(?i)^\s*•\s*fix\s*:\s*")
CELL = re.compile(r"(?<!\\)\|")
CODE = re.compile(r"`([^`]+)`")
TEST_FILE = re.compile(r"(?:^|/)test_[\w.-]*\.py(?:::[\w\[\]:.-]+)?$")
SKIP_GUARD = re.compile(
    r"importorskip|skipif|pytest\.skip\b|\bmark\.skip\b|SKIP_RETURN_CODE|allow_module_level"
)
IMPORTORSKIP = re.compile(r"importorskip\(\s*\\?[\"']([\w.]+)")
SKIPIF = re.compile(r"skipif\(([^)]*)\)")
CI_CLAIM = re.compile(
    r"(?i)\bCI (?:still |also )?runs? (?:it|them|the (?:test|case|file|entry)s?)\b|"
    r"\b(?:still|keeps?|stays?) (?:running |run |runs )?in CI\b|\bCI keeps (?:running|it)\b|"
    r"\bstill runs? in (?:the )?CI\b"
)
TIER_OR_MOVE = re.compile(
    r"(?i)\b(?:quick|standard|comprehensive|full)[- ]tiers?\b|\btier (?:yaml|label|list)s?\b|\btiers? yaml\b|"
    r"ctest_lists|\bCTest list|test_categories|InstanceTests\.cmake|"
    r"\bmove (?:the |its |this |that )?(?:test|block|entry|registration|add_test)"
)
FIXCHECK = re.compile(r"(?mi)^[\s>*•-]*fixcheck:\s*([NF]\d+)\s+(ran|read)\b")


class FixStop(Exception):
    """The fix-line premise failed: its stop line."""


def rows(text: str) -> list[dict]:
    """The 🔴/🟠 rows of the post's findings tables: id, light, the files they name and the fix text."""
    out = []
    for line in text.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in CELL.split(line.strip().strip("|"))]
        if len(cells) < 4:
            continue
        where, item = cells[2], "|".join(cells[3:])
        bullets = re.split(r"<br\s*/?>", item)
        fix = " ".join(FIX.sub("", b) for b in bullets if FIX.match(b))
        if not fix and "→" in item:
            fix = item.rsplit("→", 1)[1]
        out.append(
            {
                "id": m.group(1),
                "light": m.group(2),
                "fix": fix.strip(),
                "tests": tests_named(where + " " + fix),
            }
        )
    return out


def tests_named(text: str) -> list[str]:
    """The test files (or node ids) a row's code spans name, `:line` cut, `…/` paths left out."""
    out = []
    for span in CODE.findall(text):
        for part in re.split(r"[\s,]+", span):
            p = re.sub(r":\d+(?:-\d+)?$", "", part.strip())
            if p and "…" not in p and TEST_FILE.search(p) and p not in out:
                out.append(p)
    return out


def fixchecks(notes: str) -> set[str]:
    return {m.group(1) for m in FIXCHECK.finditer(notes or "")}


def absent_guard(fix: str, absent: set[str]) -> str | None:
    """The module CI lacks that the fix guards a test on, if any."""
    for m in IMPORTORSKIP.finditer(fix):
        if m.group(1).split(".")[0] in absent:
            return m.group(1)
    for m in SKIPIF.finditer(fix):
        for mod in sorted(absent):
            if re.search(rf"\b{re.escape(mod)}\b", m.group(1)):
                return mod
    return None


def check(
    cfg: dict, text: str, notes: str, head: str, lane_of=lanes.lane_of
) -> list[str]:
    """The fix-line premise over a post: its lines; FixStop with the stop line when a fix fails it. `lane_of(cfg,
    targets, rev=, collect=)` is lanes.lane_of."""
    absent = set(lanes.absent_modules(cfg))
    checked = fixchecks(notes)
    out = []
    for r in rows(text):
        fix, rid = r["fix"], r["id"]
        if not fix:
            continue
        mod = absent_guard(fix, absent)
        if mod:
            raise FixStop(
                f"fix-line premise: {rid}'s fix guards the test on {mod}, which CI's test environment lacks "
                f"(ci_absent_modules): CI would never run it. Move the import into the test that needs it, or skip "
                "per test so the rest still runs: stop"
            )
        if CI_CLAIM.search(fix):
            if r["tests"]:
                got = lane_of(cfg, r["tests"], rev=head, collect=True)
                res = got.get("results") or []
                unknown = got.get("error") or next(
                    (x.get("why") for x in res if x.get("verdict") == "not run"), None
                )
                if unknown or not res:
                    # A file the head lacks (a test the fix adds), or no tree: the verifier's own check settles it.
                    if rid not in checked:
                        raise FixStop(
                            f"fix-line premise: {rid}'s fix says CI still runs the test, and lane-of could not answer "
                            f"({unknown or 'no result'}): add a `fixcheck: {rid} ran HOW` line to the notes "
                            "once you checked it: stop"
                        )
                    out.append(
                        f"fix lines: {rid} claims CI runs it; lane-of NOT RUN, fixcheck in the notes"
                    )
                    continue
                bad = [x for x in res if x.get("verdict") != "runs"]
                if bad:
                    x = bad[0]
                    raise FixStop(
                        f"fix-line premise: {rid}'s fix says CI still runs the test, but lane-of at the head reads "
                        f"{x['target']}: CI: {x['verdict']}({x.get('why', '')}): rewrite the fix line: stop"
                    )
                out.append(
                    f"fix lines: {rid} claims CI runs it; lane-of: "
                    + "; ".join(
                        f"{x['target']}: runs ({x.get('entry')}, {x.get('tier')})"
                        for x in res
                    )
                )
                continue
            if rid not in checked:
                raise FixStop(
                    f"fix-line premise: {rid}'s fix says CI still runs the test but names no test file to check: add "
                    f"a `fixcheck: {rid} ran HOW` line to the notes (rk lane-of): stop"
                )
        if (SKIP_GUARD.search(fix) or TIER_OR_MOVE.search(fix)) and rid not in checked:
            what = (
                "a skip guard"
                if SKIP_GUARD.search(fix)
                else "a tier change or a test move"
            )
            raise FixStop(
                f"fix-line premise: {rid}'s fix adds {what}; the notes need a `fixcheck: {rid} ran|read HOW` line "
                "saying whether CI still runs the test (rk lane-of, an installed run): stop"
            )
        if rid in checked:
            out.append(f"fix lines: {rid} fixcheck in the notes")
    return out
