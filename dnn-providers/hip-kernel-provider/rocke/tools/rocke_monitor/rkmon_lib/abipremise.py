# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ABI premise of `rk post --premises` (lessons L-579, L-110, L-502): what a post says about the ABI is checked
against the checks' ABI section and develop's counters before the post.

- A post that says the PR needs no bump ("no bump", "no ABI change", "ABI-neutral", "does not need a bump") stops
  while the checks' ABI section (tools/abi_scan.py, in `<workers>/pr<N>/checks/checks.md`) lists a NOTE: a C struct
  layout, a public or generated header line or an exported function that changes without a binary bump. abi.h's "when
  to bump" rule decides, not the layout: a field gained in tail padding still needs one. The verifier's notes may
  carry an `abi:` line that names abi.h's rule (it must say `abi.h`) when it read the rule and the change is
  outside it.
- An ABI number the post names (`ROCKE_ABI_VERSION 17`, `BINARY_ABI = 17`, `ABI 16 -> 17`, `recipe ABI 5`) must be
  develop's or develop + 1 for its counter: a post never names a lane number (L-110); the same `abi:` line in the notes
  lets a number through.
Each check prints one line; the first that fails stops the post. Without a checks file or an ABI section the "no bump"
check says so and passes; without develop's abi.h the number check does.
"""

from __future__ import annotations

import pathlib
import re

from . import abi, gitops, paths

NO_BUMP = re.compile(
    r"(?i)\bno (?:ABI |binary |abi\.h )?bump\b|\b(?:needs|requires) no (?:ABI |binary )?bump\b|"
    r"\b(?:does not|doesn't|need not|needn't) (?:need |require )?(?:an? )?(?:ABI |binary )?bump\b|"
    r"\bno ABI (?:change|impact)\b|\bABI[- ]neutral\b"
)
NOTE = re.compile(r"(?m)^NOTE \(G9 waits[^)]*\):\s*(.+)$")
ABI_LINE = re.compile(r"(?mi)^[\s>*•-]*abi:\s*(.+)$")
# A number named for a counter: the binary one (ROCKE_ABI_VERSION, BINARY_ABI, a plain "ABI N") or the recipe one.
_ARROW = r"(?:\s*(?:->|→|to)\s*(\d{1,4}))?"
_SEP = r"\s*(?:=|==|>=|is|to|of|at|:|→|->)?\s*"
NUMBERS = {
    "recipe": re.compile(
        r"(?i)\b(?:ROCKE_RECIPE_ABI|RECIPE_ABI|recipe ABI(?: version| number)?)"
        + _SEP
        + r"(\d{1,4})\b"
        + _ARROW
    ),
    "binary": re.compile(
        r"(?i)(?<!recipe )\b(?:ROCKE_ABI_VERSION|BINARY_ABI|binary ABI(?: version| number)?|"
        r"ABI(?: version| number)?)" + _SEP + r"(\d{1,4})\b" + _ARROW
    ),
}


class AbiStop(Exception):
    """The ABI premise failed: its stop line."""


def checks_file(cfg: dict, n: int) -> pathlib.Path:
    return pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}" / "checks" / "checks.md"


def section(text: str) -> str | None:
    """The body of checks.md's ABI section (its `## ABI` heading to the next `## `); None when it has none."""
    m = re.search(r"(?m)^## ABI\b.*$", text)
    if not m:
        return None
    rest = text[m.end() :]
    nxt = re.search(r"(?m)^## ", rest)
    return rest[: nxt.start()] if nxt else rest


def notes_of(body: str | None) -> list[str]:
    return NOTE.findall(body or "")


def abi_line(notes: str) -> str | None:
    """The notes' `abi:` line that names abi.h's rule, if any."""
    for m in ABI_LINE.finditer(notes or ""):
        if "abi.h" in m.group(1):
            return m.group(1).strip()
    return None


def numbers(text: str) -> dict[str, list[int]]:
    """The ABI numbers a post names, per counter (both ends of an `A -> B`)."""
    out: dict[str, list[int]] = {}
    taken: list[tuple[int, int]] = []
    for lane in ("recipe", "binary"):
        for m in NUMBERS[lane].finditer(text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            taken.append(m.span())
            for g in m.groups():
                if g is not None:
                    out.setdefault(lane, []).append(int(g))
    return out


def develop_counters(cfg: dict, dev: str | None) -> dict:
    """develop's (or the base's) abi.h counters at `dev`, else at the mirror's develop; {} when unreadable."""
    try:
        repo, K = paths.get(cfg, "repo"), paths.repository_folder(cfg)
        ref = dev or gitops.DEVELOP_REF
        return abi.parse_header(abi._show(repo, ref, f"{K}/{abi.ABI_H}"))
    except (gitops.GitError, KeyError, OSError, paths.PathsError):
        return {}


def check(
    cfg: dict, n: int, text: str, notes: str, dev: str | None, counters=None
) -> list[str]:
    """The ABI premise over a post: its lines; AbiStop with the stop line when the post fails it. `counters`: develop's
    abi.h counters (develop_counters' form), read from the clone when None."""
    out = []
    why = abi_line(notes)
    claim = NO_BUMP.search(text)
    if claim:
        try:
            body = section(checks_file(cfg, n).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, KeyError, paths.PathsError):
            body = None
        notes_ = notes_of(body)
        if body is None:
            out.append(
                f"abi premise: the post says {claim.group(0)!r}; no ABI section in the checks to check it against"
            )
        elif notes_ and not why:
            raise AbiStop(
                f"abi premise: the post says {claim.group(0)!r}, but the checks' ABI section lists "
                f"{len(notes_)} change{'s' * (len(notes_) != 1)} without a bump ({notes_[0][:120]}): abi.h's "
                "when-to-bump rule decides, not the layout (a field gained in tail padding still needs one). "
                "Rewrite the ABI item, or add an `abi:` line naming abi.h's rule to the notes: stop"
            )
        elif notes_:
            out.append(
                f"abi premise: the post says {claim.group(0)!r} over {len(notes_)} ABI note(s); the notes' abi: "
                f"line: {why[:100]}"
            )
        else:
            out.append(
                f"abi premise: the post says {claim.group(0)!r}; the checks' ABI section lists no change without a "
                "bump"
            )
    named = numbers(text)
    if named:
        dc = develop_counters(cfg, dev) if counters is None else counters
        for lane, ns in named.items():
            d = dc.get(lane)
            if not isinstance(d, int):
                out.append(
                    f"abi premise: develop's {abi.DEFINES[lane]} not read: the post's {sorted(set(ns))} not checked"
                )
                continue
            bad = sorted({x for x in ns if x not in (d, d + 1)})
            if bad and not why:
                raise AbiStop(
                    f"abi premise: the post names {abi.DEFINES[lane]} {', '.join(map(str, bad))}, which is neither "
                    f"develop's {d} nor develop + 1 ({d + 1}): name develop's number and 'takes develop + 1 when "
                    "it lands', never a lane number (or add an `abi:` line naming abi.h to the notes): stop"
                )
            out.append(
                f"abi premise: {abi.DEFINES[lane]} {', '.join(map(str, sorted(set(ns))))} against develop {d}"
                + (f" (allowed by the notes' abi: line)" if bad else "")
            )
    return out
