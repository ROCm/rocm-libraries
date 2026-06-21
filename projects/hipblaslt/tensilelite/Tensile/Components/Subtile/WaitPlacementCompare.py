# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare split-counter wait placement between two gfx1250 asm emissions.

The subtile path emits its own split waits; StinkyTofu can instead strip and
re-insert them when ``SUBTILE_STINKYTOFU_WAITCNT`` is on. This utility diffs
*where* each emission places its split-counter waits so divergences can be
inspected for correctness/perf regressions.

Approach: parse each asm stream into a sequence of stable anchors (non-wait
instructions and labels) and attach each split-counter wait to the anchor that
immediately precedes it. Comparing the per-anchor wait multisets isolates wait
movement from the surrounding (unchanged) instruction stream. Divergence is a
signal to inspect, not necessarily a failure.

Tracked split-counter waits (gfx1250): ``s_wait_loadcnt``, ``s_wait_storecnt``,
``s_wait_dscnt``, ``s_wait_tensorcnt``, ``s_wait_kmcnt``, ``s_wait_expcnt`` and
the combined ``s_wait_loadcnt_dscnt`` / ``s_wait_storecnt_dscnt`` forms. Legacy
``s_waitcnt`` (non-split) is also tracked so its presence is reported.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Split-counter wait mnemonics (longest first so combined forms match before
# their single-counter prefixes).
WAIT_MNEMONICS = (
    "s_wait_loadcnt_dscnt",
    "s_wait_storecnt_dscnt",
    "s_wait_loadcnt",
    "s_wait_storecnt",
    "s_wait_tensorcnt",
    "s_wait_dscnt",
    "s_wait_kmcnt",
    "s_wait_expcnt",
    "s_wait_bvhcnt",
    "s_wait_samplecnt",
    "s_waitcnt_depctr",
    "s_waitcnt",
)

_WAIT_RE = re.compile(r"^(" + "|".join(WAIT_MNEMONICS) + r")\b(.*)$")
_LABEL_RE = re.compile(r"^[A-Za-z_.$][\w.$]*:\s*$")


@dataclass(frozen=True)
class WaitInst:
    """A single split-counter wait attached to a preceding anchor index."""
    anchorIdx: int      # index into the anchor list this wait follows (-1 = preamble)
    mnemonic: str       # e.g. "s_wait_loadcnt"
    count: Optional[int]  # parsed numeric count, or None if no operand
    text: str           # normalized instruction text

    def key(self) -> Tuple[str, Optional[int]]:
        return (self.mnemonic, self.count)


@dataclass
class WaitPlacement:
    """Parsed wait placement: stable anchors + waits keyed to them."""
    anchors: List[str] = field(default_factory=list)
    waits: List[WaitInst] = field(default_factory=list)

    def waitsByAnchor(self) -> Dict[int, List[WaitInst]]:
        out: Dict[int, List[WaitInst]] = {}
        for w in self.waits:
            out.setdefault(w.anchorIdx, []).append(w)
        return out

    def counts(self) -> Counter:
        """Multiset of (mnemonic, count) over all waits."""
        return Counter(w.key() for w in self.waits)


def _strip_comment(line: str) -> str:
    """Drop trailing ``//`` comments and surrounding whitespace."""
    idx = line.find("//")
    if idx != -1:
        line = line[:idx]
    return line.strip()


def _is_directive(text: str) -> bool:
    return text.startswith(".")


def parseWaitPlacement(asm: str) -> WaitPlacement:
    """Parse an asm string into anchors + split-counter waits.

    Anchors are the stable, non-wait instructions and labels surrounding the
    waits. Each wait records the index of the most recent anchor, so two streams
    that share anchors can be compared gap-by-gap.
    """
    placement = WaitPlacement()
    lastAnchor = -1
    for raw in asm.splitlines():
        text = _strip_comment(raw)
        if not text:
            continue
        m = _WAIT_RE.match(text)
        if m:
            mnemonic, rest = m.group(1), m.group(2).strip()
            count: Optional[int] = None
            cm = re.search(r"-?\d+", rest)
            if cm:
                count = int(cm.group(0))
            placement.waits.append(
                WaitInst(anchorIdx=lastAnchor, mnemonic=mnemonic,
                         count=count, text=text))
            continue
        # Labels and directives are still useful anchors; keep everything
        # that is not a wait so anchor sequences line up across streams.
        if _LABEL_RE.match(text) or not _is_directive(text):
            placement.anchors.append(text)
            lastAnchor = len(placement.anchors) - 1
    return placement


@dataclass
class WaitPlacementDiff:
    """Result of comparing two wait placements (e.g. subtile vs StinkyTofu)."""
    anchorsMatch: bool
    sharedAnchorCount: int
    leftAnchorCount: int
    rightAnchorCount: int
    leftTotals: Counter
    rightTotals: Counter
    # Per-anchor divergences: anchorIdx -> (anchorText, leftMultiset, rightMultiset)
    perAnchor: List[Tuple[int, str, Counter, Counter]] = field(default_factory=list)
    # Anchor-sequence mismatch detail (first differing index), if any.
    firstAnchorMismatch: Optional[Tuple[int, str, str]] = None

    @property
    def diverges(self) -> bool:
        return (not self.anchorsMatch) or bool(self.perAnchor) \
            or (self.leftTotals != self.rightTotals)

    def summary(self) -> str:
        lines: List[str] = []
        lines.append("Wait-placement comparison (left=subtile, right=StinkyTofu)")
        lines.append(f"  anchors: left={self.leftAnchorCount} "
                     f"right={self.rightAnchorCount} match={self.anchorsMatch}")
        if not self.anchorsMatch and self.firstAnchorMismatch is not None:
            i, lt, rt = self.firstAnchorMismatch
            lines.append(f"  first anchor mismatch @ {i}:")
            lines.append(f"    left : {lt}")
            lines.append(f"    right: {rt}")
        lines.append(f"  total split waits: left={sum(self.leftTotals.values())} "
                     f"right={sum(self.rightTotals.values())}")
        totalKeys = sorted(set(self.leftTotals) | set(self.rightTotals))
        for k in totalKeys:
            lc, rc = self.leftTotals.get(k, 0), self.rightTotals.get(k, 0)
            flag = "" if lc == rc else "  <-- differs"
            mnem, cnt = k
            cntStr = "" if cnt is None else f" {cnt}"
            lines.append(f"    {mnem}{cntStr}: left={lc} right={rc}{flag}")
        if self.perAnchor:
            lines.append(f"  per-anchor divergences: {len(self.perAnchor)}")
            for idx, text, lset, rset in self.perAnchor:
                lines.append(f"    after anchor[{idx}] '{text}':")
                lines.append(f"      left : {_fmt_multiset(lset)}")
                lines.append(f"      right: {_fmt_multiset(rset)}")
        else:
            lines.append("  per-anchor divergences: none")
        return "\n".join(lines)


def _fmt_multiset(ms: Counter) -> str:
    if not ms:
        return "(none)"
    parts = []
    for (mnem, cnt), n in sorted(ms.items()):
        cntStr = "" if cnt is None else f" {cnt}"
        parts.append(f"{mnem}{cntStr}x{n}")
    return ", ".join(parts)


def compareWaitPlacement(asmLeft: str, asmRight: str) -> WaitPlacementDiff:
    """Diff split-counter wait placement between two asm streams.

    ``asmLeft`` is the subtile (flag-off) emission; ``asmRight`` is the
    StinkyTofu (flag-on) emission. Anchors are compared positionally; per-anchor
    wait multisets are diffed where the anchor sequence agrees.
    """
    left = parseWaitPlacement(asmLeft)
    right = parseWaitPlacement(asmRight)

    n = min(len(left.anchors), len(right.anchors))
    firstMismatch: Optional[Tuple[int, str, str]] = None
    for i in range(n):
        if left.anchors[i] != right.anchors[i]:
            firstMismatch = (i, left.anchors[i], right.anchors[i])
            break
    anchorsMatch = (firstMismatch is None
                    and len(left.anchors) == len(right.anchors))

    leftByAnchor = left.waitsByAnchor()
    rightByAnchor = right.waitsByAnchor()
    perAnchor: List[Tuple[int, str, Counter, Counter]] = []
    # Only diff per-anchor gaps over the shared, positionally-matching prefix;
    # if anchors diverge, the totals diff still reports the overall picture.
    upper = n if firstMismatch is None else firstMismatch[0]
    for idx in range(-1, upper):
        lset = Counter(w.key() for w in leftByAnchor.get(idx, []))
        rset = Counter(w.key() for w in rightByAnchor.get(idx, []))
        if lset != rset:
            text = "<preamble>" if idx < 0 else left.anchors[idx]
            perAnchor.append((idx, text, lset, rset))

    return WaitPlacementDiff(
        anchorsMatch=anchorsMatch,
        sharedAnchorCount=n,
        leftAnchorCount=len(left.anchors),
        rightAnchorCount=len(right.anchors),
        leftTotals=left.counts(),
        rightTotals=right.counts(),
        perAnchor=perAnchor,
        firstAnchorMismatch=firstMismatch,
    )
