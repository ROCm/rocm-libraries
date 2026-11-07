# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SLO106: copy-paste overlap between file pairs.

overlap(a, b) = distinct lines found in both / distinct lines of the smaller file.
Whitespace is squeezed; comment lines and lines shorter than
OVERLAP_MIN_LINE_CHARS are dropped.

A second, advisory score (`overlap_norm`) first replaces every variable name with
`V` (method and attribute names are kept). It catches copies where only the
variable names were changed, which the plain score misses.
"""
from __future__ import annotations

import io
import itertools
import keyword
import tokenize

import slo_config as C


def plain_lines(src: str) -> set[str]:
    out = set()
    for line in src.splitlines():
        s = " ".join(line.split())
        if len(s) >= C.OVERLAP_MIN_LINE_CHARS and not s.startswith("#"):
            out.add(s)
    return out


def normalized_lines(src: str) -> set[str]:
    """Lines with variable names replaced by V. Comments and strings dropped."""
    rows: dict[int, list[str]] = {}
    prev = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                continue
            text = tok.string
            if (
                tok.type == tokenize.NAME
                and not keyword.iskeyword(text)
                and not (prev is not None and prev.string == ".")
            ):
                text = "V"
            elif tok.type == tokenize.STRING:
                text = "S"
            rows.setdefault(tok.start[0], []).append(text)
            prev = tok
    except (tokenize.TokenError, IndentationError):
        return set()
    return {
        s
        for s in (" ".join(t) for t in rows.values())
        if len(s) >= C.OVERLAP_MIN_LINE_CHARS
    }


def pairs(sources: dict[str, str], slocs: dict[str, int]) -> list[dict]:
    """All file pairs at or above OVERLAP_LIST_FROM, worst first."""
    sel = [p for p in sources if slocs.get(p, 0) >= C.OVERLAP_MIN_SLOC]
    plain = {p: plain_lines(sources[p]) for p in sel}
    norm = {p: normalized_lines(sources[p]) for p in sel}
    out = []
    for a, b in itertools.combinations(sel, 2):
        shared = len(plain[a] & plain[b])
        small = min(len(plain[a]), len(plain[b]))
        nshared = len(norm[a] & norm[b])
        nsmall = min(len(norm[a]), len(norm[b]))
        ov = shared / small if small else 0.0
        nov = nshared / nsmall if nsmall else 0.0
        if (ov >= C.OVERLAP_LIST_FROM and shared >= C.OVERLAP_MIN_SHARED) or (
            nov >= C.OVERLAP_LIST_FROM and nshared >= C.OVERLAP_MIN_SHARED
        ):
            out.append(
                {
                    "a": a,
                    "b": b,
                    "shared": shared,
                    "overlap": round(ov, 3),
                    "overlap_norm": round(nov, 3),
                }
            )
    out.sort(key=lambda r: (-r["overlap"] * r["shared"], r["a"], r["b"]))
    return out


def worst_per_file(pair_list: list[dict]) -> dict[str, dict]:
    """For each file, its highest plain overlap that counts (shared >= OVERLAP_MIN_SHARED)."""
    worst: dict[str, dict] = {}
    for r in pair_list:
        if r["shared"] < C.OVERLAP_MIN_SHARED:
            continue
        for me, other in ((r["a"], r["b"]), (r["b"], r["a"])):
            if me not in worst or r["overlap"] > worst[me]["overlap"]:
                worst[me] = {
                    "overlap": r["overlap"],
                    "with": other,
                    "shared": r["shared"],
                }
    return worst
