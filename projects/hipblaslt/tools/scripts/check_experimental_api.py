#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fail if a change breaks the experimental API without bumping its version.

Compares the experimental API of a base revision (normally the PR's target branch) with
the working tree. A change is breaking if it

* removes or changes a declaration in hipblaslt-experimental.h or a
  hipblaslt-experimental-<feature>.h header (functions, types, enumerators, macros), or
* changes or deletes a contract test (clients/tests/src/experimental/contract_*.cpp).
  Contract tests define an experimental feature's promised behavior, so editing one
  means the behavior changed.

Additions are never breaking. A breaking change must bump HIPBLASLT_EXPERIMENTAL_API_VERSION
in the same change. Policy: docs/reference/experimental-api.rst.

Usage, from anywhere inside projects/hipblaslt:
    python3 tools/scripts/check_experimental_api.py --base origin/develop
    python3 tools/scripts/check_experimental_api.py --base-dir <other hipblaslt checkout>

Pure standard library (plus git for --base).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEADER_DIR = "library/include/hipblaslt"
MAIN_HEADER = f"{HEADER_DIR}/hipblaslt-experimental.h"
FEATURE_HEADER_GLOB = "hipblaslt-experimental-*.h"
CONTRACT_DIR = "clients/tests/src/experimental"
CONTRACT_GLOB = "contract_*.cpp"
VERSION_MACRO = "HIPBLASLT_EXPERIMENTAL_API_VERSION"


# ---------------------------------------------------------------------------
# Source trees: the working tree, another directory, or a git revision.
# ---------------------------------------------------------------------------


class DirTree:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.name = str(root)

    def read(self, rel: str) -> Optional[str]:
        p = self.root / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def glob(self, rel_dir: str, pattern: str) -> List[str]:
        d = self.root / rel_dir
        if not d.is_dir():
            return []
        return sorted(f"{rel_dir}/{p.name}" for p in d.iterdir() if fnmatch.fnmatch(p.name, pattern))


class GitTree:
    """Files of PROJECT_ROOT as of a git revision."""

    def __init__(self, rev: str, root: Path = PROJECT_ROOT):
        self.root = root
        self.rev = rev
        self.name = rev

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, text=True, encoding="utf-8"
        )

    def read(self, rel: str) -> Optional[str]:
        r = self._git("show", f"{self.rev}:./{rel}")
        return r.stdout if r.returncode == 0 else None

    def glob(self, rel_dir: str, pattern: str) -> List[str]:
        r = self._git("ls-tree", "--name-only", self.rev, f"{rel_dir}/")
        if r.returncode != 0:
            raise RuntimeError(f"git ls-tree {self.rev} failed: {r.stderr.strip()}")
        names = [line.rsplit("/", 1)[-1] for line in r.stdout.splitlines()]
        return sorted(f"{rel_dir}/{n}" for n in names if fnmatch.fnmatch(n, pattern))


# ---------------------------------------------------------------------------
# Extracting the API surface.
# ---------------------------------------------------------------------------

_COMMENT_OR_STRING = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', re.S)


def strip_comments(text: str) -> str:
    def repl(m: re.Match) -> str:
        s = m.group(0)
        return s if s.startswith('"') else " "

    text = text.replace("\r\n", "\n")
    text = _COMMENT_OR_STRING.sub(repl, text)
    return text.replace("\\\n", " ")


def normalize(text: str) -> str:
    """Whitespace-insensitive form: 'int * x' and 'int* x' compare equal."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s*([^\w\s])\s*", r"\1", text)


@dataclass
class Surface:
    version: Optional[int] = None
    decls: Dict[str, str] = field(default_factory=dict)  # key -> normalized text
    contracts: Dict[str, str] = field(default_factory=dict)  # path -> hash
    errors: List[str] = field(default_factory=list)


def _add_statement(stmt: str, where: str, surface: Surface) -> None:
    stmt = normalize(stmt)
    if not stmt:
        return
    enum = re.match(r"(?:typedef )?enum(?: (\w+))?(?::[\w ]+)?\{(.*)\}(\w*)$", stmt)
    if enum:
        name = enum.group(3) or enum.group(1)
        if not name:
            surface.errors.append(f"{where}: anonymous enum; give it a name")
            return
        surface.decls[f"enum {name}"] = "enum"
        for item in filter(None, (e.strip() for e in enum.group(2).split(","))):
            if "=" not in item:
                surface.errors.append(
                    f"{where}: enumerator {name}::{item} needs an explicit value"
                    " (implicit values shift silently when enumerators are inserted)"
                )
                continue
            ident, value = item.split("=", 1)
            surface.decls[f"enumerator {name}::{ident}"] = value
        return
    fnptr = re.search(r"\(\*(\w+)\)", stmt)
    if stmt.startswith("typedef"):
        name = fnptr.group(1) if fnptr else re.findall(r"\w+", stmt)[-1]
        surface.decls[f"type {name}"] = stmt
    elif re.match(r"(?:typedef )?(?:struct|union) \w+\{", stmt):
        surface.decls[f"type {re.match(r'(?:typedef )?(?:struct|union) (\w+)', stmt).group(1)}"] = stmt
    elif "(" in stmt:
        surface.decls[f"function {re.findall(r'(\w+)\(', stmt)[0]}"] = stmt
    else:
        surface.decls[f"decl {stmt}"] = stmt


def parse_header(text: str, where: str, surface: Surface) -> None:
    code_lines = []
    for line in strip_comments(text).split("\n"):
        directive = line.strip()
        if directive.startswith("#"):
            m = re.match(r"#\s*define\s+(\w+)(\([^)]*\))?\s+(.+)", directive)
            if m:
                name, value = m.group(1), m.group(3)
                if name == VERSION_MACRO:
                    surface.version = int(value.strip())
                else:
                    surface.decls[f"macro {name}"] = normalize((m.group(2) or "") + " " + value)
            continue
        code_lines.append(line)
    code = re.sub(r'extern\s*"C"\s*\{', " ", "\n".join(code_lines))

    depth, start = 0, 0
    for i, ch in enumerate(code):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:  # closing brace of extern "C"
                code = code[:i] + " " + code[i + 1 :]
                continue
            depth -= 1
        elif ch == ";" and depth == 0:
            _add_statement(code[start:i], where, surface)
            start = i + 1
    _add_statement(code[start:], where, surface)


def contract_hash(text: str) -> str:
    return hashlib.sha256(normalize(strip_comments(text)).encode("utf-8")).hexdigest()


def load_surface(tree) -> Optional[Surface]:
    main = tree.read(MAIN_HEADER)
    if main is None:
        return None
    surface = Surface()
    parse_header(main, MAIN_HEADER, surface)
    for path in tree.glob(HEADER_DIR, FEATURE_HEADER_GLOB):
        parse_header(tree.read(path), path, surface)
    for path in tree.glob(CONTRACT_DIR, CONTRACT_GLOB):
        surface.contracts[path] = contract_hash(tree.read(path))
    if surface.version is None:
        surface.errors.append(f"{MAIN_HEADER}: no #define {VERSION_MACRO}")
    return surface


# ---------------------------------------------------------------------------
# Comparing.
# ---------------------------------------------------------------------------


@dataclass
class Report:
    breaking: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def compare(base: Optional[Surface], head: Surface) -> Report:
    report = Report(errors=list(head.errors))
    if base is None:
        report.warnings.append("base has no experimental API; nothing to compare")
        return report

    for key, text in base.decls.items():
        if key not in head.decls:
            report.breaking.append(f"removed  {key}")
        elif head.decls[key] != text:
            report.breaking.append(f"changed  {key}: '{text}' -> '{head.decls[key]}'")
    report.added += [f"added    {k}" for k in head.decls if k not in base.decls]

    for path, digest in base.contracts.items():
        if path not in head.contracts:
            report.breaking.append(f"deleted contract test {path}")
        elif head.contracts[path] != digest:
            report.breaking.append(f"changed contract test {path}")
    report.added += [f"added contract test {p}" for p in head.contracts if p not in base.contracts]

    if None in (base.version, head.version):
        return report
    if head.version < base.version:
        report.errors.append(f"{VERSION_MACRO} went backwards: {base.version} -> {head.version}")
    elif report.breaking and head.version == base.version:
        report.errors.append(
            f"breaking change to the experimental API without bumping {VERSION_MACRO} "
            f"(still {base.version}).\n"
            f"  Bump it to {base.version + 1} in {MAIN_HEADER} and add a breaking entry under\n"
            '  "### Experimental" in CHANGELOG.md. If the change is not meant to be breaking,\n'
            "  restore the removed/changed declarations or contract tests instead."
        )
    elif head.version > base.version and not report.breaking:
        report.warnings.append(
            f"{VERSION_MACRO} bumped ({base.version} -> {head.version}) but no breaking change"
            " was detected. If behavior changed, a contract test should change with it."
        )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", help="git revision to compare against (uses its merge base with HEAD)")
    src.add_argument("--base-dir", type=Path, help="another hipblaslt project root to compare against")
    ap.add_argument("--head-dir", type=Path, default=PROJECT_ROOT, help="hipblaslt project root to check")
    args = ap.parse_args(argv)

    if args.base:
        mb = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "merge-base", args.base, "HEAD"],
            capture_output=True, text=True,
        )
        if mb.returncode != 0:
            print(f"error: git merge-base {args.base} HEAD failed: {mb.stderr.strip()}", file=sys.stderr)
            return 2
        base_tree = GitTree(mb.stdout.strip())
    else:
        base_tree = DirTree(args.base_dir)

    head = load_surface(DirTree(args.head_dir))
    if head is None:
        print(f"error: {MAIN_HEADER} not found under {args.head_dir}", file=sys.stderr)
        return 2
    report = compare(load_surface(base_tree), head)

    print(f"experimental API: base {base_tree.name}, version {head.version} in working tree")
    for line in report.breaking + report.added:
        print(f"  {line}")
    for w in report.warnings:
        print(f"warning: {w}")
    for e in report.errors:
        print(f"error: {e}")
    print("OK" if report.ok else "FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
