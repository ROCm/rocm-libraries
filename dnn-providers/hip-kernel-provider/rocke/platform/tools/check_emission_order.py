#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# check_emission_order.py -- keep C++ builder calls from emitting ops in a
# compiler-dependent order.
#
# Every op the IR builder emits, constants included, takes the next SSA number.
# Python evaluates call arguments left to right; C and C++ leave the order
# unspecified (GCC goes right to left). So a call with two or more arguments
# that each emit an op, such as
#
#     rocke_b_add(b, rocke_b_mul(b, x, y), rocke_b_const_i32(b, 1))
#
# can number its ops differently from the Python twin, and byte identity then
# holds or breaks depending on the compiler. The fix is to bind the arguments to
# locals in the Python twin's order and make the call with plain values.
#
# An argument "emits" when it calls a rocke_b_* builder function other than the
# type and info helpers in NON_OP, or one of the wrappers in EMITTING_WRAPPERS.
# This is a lexical scan, not a C++ parser: comments, string literals and
# preprocessor lines are ignored, and an emitting call hidden behind a macro or
# an unlisted wrapper is not seen.
#
# With no arguments it scans cpp/ and fails on any site not in KNOWN_SITES, and
# on any KNOWN_SITES entry that no longer occurs. With file arguments (the
# pre-commit hook) it checks only those files and skips the staleness check.

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CPP_ROOT = Path(__file__).resolve().parents[1] / "cpp"
SUFFIXES = {".c", ".cpp", ".h", ".hpp"}

# rocke_b_* functions that return a type, a name or an existing value and emit
# nothing.
NON_OP = frozenset(
    {
        "rocke_b_fmoe_ensure_2byte_dtype",
        "rocke_b_fmoe_gemm_dtype_to_universal",
        "rocke_b_get_param",
        "rocke_b_io_ir_type",
        "rocke_b_mfma_atom_for_dtype",
        "rocke_b_quant_ir_type",
    }
)

# Helpers that are not rocke_b_* but emit ops: the cached lane-index values
# (they emit on first use, so whether a call emits depends on the spec) and the
# gfx1151 vector splat. Add a wrapper here when a call passes it beside another
# emitting argument.
EMITTING_WRAPPERS = frozenset(
    {
        "fh_lane_col",
        "fh_lane_col32",
        "fh_lane_half32",
        "fh_lane_rg",
        "lane_col32_of",
        "lane_col_of",
        "lane_half32_of",
        "lane_rg_of",
        "rocke__lane_half_base",
        "rocke_a2d_lane_col",
        "rocke_a2d_lane_rg",
        "rocke_g950a2d_lane_rg",
        "rocke_gfx1151_dfcp_splat_i32",
    }
)

# (path relative to cpp/, callee) -> occurrences allowed. Burn down only.
KNOWN_SITES: dict[tuple[str, str], int] = {
    # The 3D QQ-bias mask compares; fixed together with their parity configs
    # in a separate change. Delete both entries when it lands.
    (
        "instances/gfx942/attention_tiled_3d_gfx942_attn_tiled_3d_segment_compute.cpp",
        "rocke_b_land",
    ): 1,
    (
        "instances/gfx950/attention_tiled_3d_segment_loop_epilogue.cpp",
        "rocke_b_land",
    ): 1,
}

_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_NOT_CALLS = frozenset({"if", "for", "while", "switch", "return", "sizeof", "defined"})


@dataclass(frozen=True)
class Site:
    line: int
    callee: str


def strip_source(src: str) -> str:
    """Blank out comments, string/char literals and preprocessor lines.

    Newlines are kept so offsets still map to the original line numbers."""
    out: list[str] = []
    i, n = 0, len(src)
    at_line_start = True
    while i < n:
        c = src[i]
        if src.startswith("//", i) or (c == "#" and at_line_start):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            end = min(j + 1, n)
            out.append(" " * (end - i))
            i = end
        else:
            out.append(c)
            if c == "\n":
                at_line_start = True
            elif not c.isspace():
                at_line_start = False
            i += 1
            continue
        at_line_start = False
    return "".join(out)


def _arguments(src: str, open_paren: int) -> list[str]:
    """Top-level argument texts of the call whose '(' is at open_paren."""
    depth, start, args = 0, open_paren + 1, []
    for i in range(open_paren, len(src)):
        c = src[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append(src[start:i])
                return args
        elif c == "," and depth == 1:
            args.append(src[start:i])
            start = i + 1
    return args


def _emits(arg: str) -> bool:
    return any(
        (name.startswith("rocke_b_") and name not in NON_OP)
        or name in EMITTING_WRAPPERS
        for name in _CALL.findall(arg)
    )


def find_sites(src: str) -> list[Site]:
    """Calls in src with two or more op-emitting arguments."""
    text = strip_source(src)
    sites = []
    for m in _CALL.finditer(text):
        if m.group(1) in _NOT_CALLS:
            continue
        if sum(_emits(a) for a in _arguments(text, m.end() - 1)) >= 2:
            sites.append(Site(text.count("\n", 0, m.start()) + 1, m.group(1)))
    return sites


def scan(paths) -> dict[str, list[Site]]:
    """{path relative to cpp/: sites} for every C/C++ file in paths."""
    found = {}
    for path in paths:
        path = Path(path).resolve()
        if path.suffix not in SUFFIXES or not path.is_relative_to(CPP_ROOT):
            continue
        sites = find_sites(path.read_text(encoding="utf-8", errors="replace"))
        if sites:
            found[path.relative_to(CPP_ROOT).as_posix()] = sites
    return found


def problems(found: dict[str, list[Site]], *, check_stale: bool) -> list[str]:
    """Sites beyond KNOWN_SITES, plus stale KNOWN_SITES entries if asked."""
    out = []
    counts = Counter()
    for rel, sites in sorted(found.items()):
        for site in sites:
            counts[rel, site.callee] += 1
            if counts[rel, site.callee] > KNOWN_SITES.get((rel, site.callee), 0):
                out.append(
                    f"cpp/{rel}:{site.line}: {site.callee}(...) has two or more "
                    "arguments that emit ops; bind them to locals in the Python "
                    "twin's order (C++ leaves argument evaluation order unspecified)"
                )
    if check_stale:
        for (rel, callee), allowed in sorted(KNOWN_SITES.items()):
            if counts[rel, callee] < allowed:
                out.append(
                    f"cpp/{rel}: KNOWN_SITES allows {allowed} {callee} site(s) but "
                    f"{counts[rel, callee]} remain; lower or delete the entry"
                )
    return out


def main(argv: list[str]) -> int:
    if argv:
        found = scan(argv)
    else:
        found = scan(p for p in CPP_ROOT.rglob("*") if p.is_file())
    errors = problems(found, check_stale=not argv)
    for line in errors:
        print(line, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
