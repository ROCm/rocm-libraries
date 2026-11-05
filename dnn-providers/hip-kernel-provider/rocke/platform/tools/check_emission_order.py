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
# type and info helpers in NON_OP, or a function whose body emits. The second
# set is derived from every function and member function defined in the scanned
# trees, to a fixed point: a body that calls an emitting function emits too.
# A lambda bound to a name (auto f = [&](...) { ... };) emits the same way, but
# only within its own file, since its name is local. A lambda body inside an
# argument does not count, since it runs when the callee calls it, not while
# the arguments are evaluated, unless the argument calls it on the spot
# ([&] { ... }()).
#
# This is a lexical scan, not a C++ parser: comments, string literals and
# preprocessor lines are ignored, functions are matched by name across files,
# and an emitting call hidden behind a macro is not seen.
#
# It scans cpp/ and tests/instances/parity/ (plus any --also directories) and
# fails on any site not in KNOWN_SITES, and on any KNOWN_SITES entry that no
# longer occurs.

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
CPP_ROOT = PLATFORM / "cpp"
DEFAULT_ROOTS = (CPP_ROOT, PLATFORM / "tests" / "instances" / "parity")
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

# (path relative to platform/, callee) -> occurrences allowed. Burn down only.
KNOWN_SITES: dict[tuple[str, str], int] = {
    # The 3D QQ-bias mask compares; fixed together with their parity configs
    # in a separate change. Delete both entries when it lands.
    (
        "cpp/instances/gfx942/attention_tiled_3d_gfx942_attn_tiled_3d_segment_compute.cpp",
        "rocke_b_land",
    ): 1,
    (
        "cpp/instances/gfx950/attention_tiled_3d_segment_loop_epilogue.cpp",
        "rocke_b_land",
    ): 1,
}

_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_NOT_CALLS = frozenset({"if", "for", "while", "switch", "return", "sizeof", "defined"})
# Names that can stand before "(...) {" without being a function definition.
_NOT_DEFINITIONS = _NOT_CALLS | {"catch"}
# What may sit between a parameter list's ')' and the body's '{'.
_QUALIFIERS = re.compile(r"\)\s*(?:(?:const|noexcept|override|final|mutable)\s*)*$")
_NAME_BEFORE = re.compile(r"([A-Za-z_]\w*)\s*$")
# What may sit between a lambda's parameter list and its body.
_LAMBDA_TAIL = re.compile(
    r"(?:\s*(?:mutable|constexpr|noexcept)\b)*(?:\s*->\s*[\w:<>,\s*&]+?)?\s*$"
)
# "name = [" where the '[' opens a lambda's capture list.
_LAMBDA_BINDING = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*\[")


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


def _close(text: str, start: int) -> int:
    """Index of the bracket closing the one at start (len(text) if none)."""
    opening, closing = text[start], {"(": ")", "[": "]", "{": "}"}[text[start]]
    depth = 0
    for i in range(start, len(text)):
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def _open(text: str, end: int) -> int:
    """Index of the '(' matching the ')' at end (-1 if none)."""
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == ")":
            depth += 1
        elif text[i] == "(":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _header_name(text: str, brace: int) -> str | None:
    """For the '{' at brace: the name before its "(...)" header, if any."""
    head = _QUALIFIERS.search(text, max(0, brace - 200), brace)
    if head is None:
        return None
    paren = _open(text, head.start())
    if paren < 0:
        return None
    before = text[max(0, paren - 200) : paren]
    if before.rstrip().endswith("]"):
        return None  # a lambda: [captures](params) { ... }
    name = _NAME_BEFORE.search(before)
    if name is None or name.group(1) in _NOT_DEFINITIONS:
        return None
    return name.group(1)


def _is_lambda_body(text: str, brace: int) -> bool:
    """Whether the '{' at brace opens a lambda body."""
    tail = _LAMBDA_TAIL.search(text, max(0, brace - 200), brace)
    head = text[: tail.start() if tail else brace].rstrip()
    if head.endswith("]"):
        return True
    if head.endswith(")"):
        paren = _open(head, len(head) - 1)
        return paren > 0 and head[:paren].rstrip().endswith("]")
    return False


def _without_lambda_bodies(arg: str) -> str:
    """arg with the body of every lambda in it blanked out, except a lambda
    the argument calls on the spot."""
    out, i = arg, 0
    while True:
        i = out.find("{", i)
        if i < 0:
            return out
        if _is_lambda_body(out, i):
            j = _close(out, i)
            if not out[j + 1 :].lstrip().startswith("("):
                out = out[:i] + " " * (j + 1 - i) + out[j + 1 :]
        i += 1


def _lambda_body(text: str, capture: int) -> str | None:
    """Body of the lambda whose capture list opens at capture, if it is one."""
    end = _close(text, capture)
    rest = end + 1
    while rest < len(text) and text[rest].isspace():
        rest += 1
    if rest < len(text) and text[rest] == "(":
        rest = _close(text, rest) + 1
    brace = text.find("{", rest)
    if brace < 0 or ";" in text[rest:brace] or not _is_lambda_body(text, brace):
        return None
    return text[brace : _close(text, brace) + 1]


def named_lambdas(text: str) -> dict[str, str]:
    """{name: body} of every lambda bound to a name in text (stripped)."""
    found: dict[str, str] = {}
    for m in _LAMBDA_BINDING.finditer(text):
        body = _lambda_body(text, m.end() - 1)
        if body is not None:
            found[m.group(1)] = found.get(m.group(1), "") + body
    return found


def _derive(callees: dict[str, set[str]], emitting: set[str]) -> frozenset[str]:
    """emitting, grown to a fixed point over callees."""
    emitting = set(emitting)
    emitting |= {name for name, calls in callees.items() if _calls_an_op(calls)}
    while True:
        more = {
            name
            for name, calls in callees.items()
            if name not in emitting and calls & emitting
        }
        if not more:
            return frozenset(emitting)
        emitting |= more


def function_bodies(text: str) -> Iterable[tuple[str, str]]:
    """(name, body) of every function and member function defined in text.

    text must already be stripped (see strip_source)."""
    for brace in (m.start() for m in re.finditer(r"\{", text)):
        name = _header_name(text, brace)
        if name is not None:
            yield name, text[brace : _close(text, brace) + 1]


def _calls_an_op(names: Iterable[str]) -> bool:
    return any(n.startswith("rocke_b_") and n not in NON_OP for n in names)


def emitting_functions(texts: Iterable[str]) -> frozenset[str]:
    """Names of the functions in texts (stripped sources) whose bodies emit."""
    callees: dict[str, set[str]] = {}
    for text in texts:
        for name, body in function_bodies(text):
            callees.setdefault(name, set()).update(_CALL.findall(body))
    return _derive(callees, set())


def _with_local_lambdas(text: str, emitters: frozenset[str]) -> frozenset[str]:
    """emitters plus the lambdas named in text whose bodies emit."""
    callees = {
        name: set(_CALL.findall(body)) for name, body in named_lambdas(text).items()
    }
    if not callees:
        return emitters
    return emitters | _derive(callees, set(emitters))


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


def _emits(arg: str, emitters: frozenset[str]) -> bool:
    names = _CALL.findall(_without_lambda_bodies(arg))
    return _calls_an_op(names) or any(name in emitters for name in names)


def _sites(text: str, emitters: frozenset[str]) -> list[Site]:
    emitters = _with_local_lambdas(text, emitters)
    sites = []
    for m in _CALL.finditer(text):
        if m.group(1) in _NOT_CALLS:
            continue
        args = _arguments(text, m.end() - 1)
        if sum(_emits(a, emitters) for a in args) >= 2:
            sites.append(Site(text.count("\n", 0, m.start()) + 1, m.group(1)))
    return sites


def find_sites(src: str, emitters: frozenset[str] = frozenset()) -> list[Site]:
    """Calls in src with two or more op-emitting arguments.

    emitters adds functions to count as emitting; by default only rocke_b_*
    calls (and functions defined in src) do."""
    text = strip_source(src)
    return _sites(text, emitters | emitting_functions([text]))


def _display(path: Path) -> str:
    for base in (PLATFORM, Path.cwd()):
        if path.is_relative_to(base):
            return path.relative_to(base).as_posix()
    return path.as_posix()


def scan(roots: Iterable[Path]) -> dict[str, list[Site]]:
    """{path relative to platform/: sites} for every C/C++ file under roots."""
    paths = sorted(
        {
            p.resolve()
            for root in roots
            for p in Path(root).rglob("*")
            if p.is_file() and p.suffix in SUFFIXES
        }
    )
    texts = {
        p: strip_source(p.read_text(encoding="utf-8", errors="replace")) for p in paths
    }
    emitters = emitting_functions(texts.values())
    found = {}
    for path, text in texts.items():
        sites = _sites(text, emitters)
        if sites:
            found[_display(path)] = sites
    return found


def problems(found: dict[str, list[Site]]) -> list[str]:
    """Sites beyond KNOWN_SITES, and KNOWN_SITES entries that no longer occur."""
    out = []
    counts = Counter()
    for rel, sites in sorted(found.items()):
        for site in sites:
            counts[rel, site.callee] += 1
            if counts[rel, site.callee] > KNOWN_SITES.get((rel, site.callee), 0):
                out.append(
                    f"{rel}:{site.line}: {site.callee}(...) has two or more "
                    "arguments that emit ops; bind them to locals in the Python "
                    "twin's order (C++ leaves argument evaluation order unspecified)"
                )
    for (rel, callee), allowed in sorted(KNOWN_SITES.items()):
        if counts[rel, callee] < allowed:
            out.append(
                f"{rel}: KNOWN_SITES allows {allowed} {callee} site(s) but "
                f"{counts[rel, callee]} remain; lower or delete the entry"
            )
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--also",
        action="append",
        default=[],
        type=Path,
        metavar="DIR",
        help="also scan the C/C++ files under DIR (e.g. library/tests/parity)",
    )
    args = parser.parse_args(argv)
    errors = problems(scan([*DEFAULT_ROOTS, *args.also]))
    for line in errors:
        print(line, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
