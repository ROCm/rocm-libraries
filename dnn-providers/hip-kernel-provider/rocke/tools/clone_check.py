#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# clone_check.py -- report Python code copies that a change ADDS (advisory).
#
# Compares a base commit with either the working tree (default) or a head
# commit. Every run tokenises all in-scope Python files on both sides,
# fingerprints every window of MIN_TOKENS consecutive code tokens (across
# statement boundaries, local variable names normalised), and reports a
# fingerprint only when the change adds a genuine new occurrence of code that
# now exists in two or more places. Python only: C++ duplication is not measured.
#
# Usage:
#   python tools/clone_check.py                   # base = merge-base with origin/develop, head = working tree
#   python tools/clone_check.py --base <commit>   # other base
#   python tools/clone_check.py --head <commit>   # compare two commits (PR mode)
#   python tools/clone_check.py --format json     # machine-readable
#   python tools/clone_check.py --fail-on-new     # exit 1 when a NEW or LIKELY copy is found (off by default)
#   python tools/clone_check.py --backlog         # EXPERIMENTAL: every existing copy group in the working tree (or --head), most duplicated lines first
#
# Exit codes: 0 ran (advisory), 1 a NEW or LIKELY copy was found with --fail-on-new,
# 2 tool error.
#
# Approved forks: tools/clone_groups.json is {"groups": [{"name": ..., "files": [at
# least two paths relative to the rocKE root], "reason": ...}]}. Files in one group
# are meant to stay copies of each other (for example per-architecture schedules):
#   - a copy between files of one group is a SIBLING COPY, not a NEW COPY;
#   - a second copy inside one file of a group is still a NEW COPY;
#   - editing one file of a group while a matching block in a sibling stays
#     unchanged adds a SIBLING NOTE naming the sibling;
#   - a copy into a file outside the group is a NEW COPY.
# A malformed file makes the tool exit with code 2.

from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from array import array
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

ROCKE = Path(__file__).resolve().parents[1]  # tools -> rocke
SCOPE_DIRS = ("library", "platform/python/rocke")
# Test, benchmark, verify and example code is out of scope: its duplication is
# harness scaffolding, not kernel-authoring debt.
_OUT_OF_SCOPE = re.compile(
    r"(^|/)tests/"
    r"|(^|/)(examples|experiments)/"
    r"|^library/benchmarks/"
    r"|^platform/python/rocke/benchmark/"
    r"|/[^/]*_(verify|sweep)\.py$"
    r"|/(bench|benchmark|reproduce)_[^/]*\.py$"
)
# Host-side command-line and test-input helpers inside builder scripts. They are
# skipped together with `if __name__ == "__main__":` blocks. `check` and `bench`
# stay in scope: copies of them were judged real duplication.
_HARNESS_FUNCTIONS = frozenset({"main", "make_inputs", "parse_args", "_parser"})
GROUPS_FILE = "tools/clone_groups.json"
DEFAULT_UPSTREAM = "origin/develop"
# Smallest copy reported, in code tokens (names, operators, literals; comments
# and layout excluded). Smaller windows report more look-alike code; larger ones
# miss short copies. 50 is a starting point, not a measured optimum.
MIN_TOKENS = 50
WORKTREE = "WORKTREE"

NEW_COPY = "NEW COPY"
LIKELY_COPY = "LIKELY COPY"
SIBLING_COPY = "SIBLING COPY"
EXISTING_EDITED = "EXISTING COPIES EDITED"
SIBLING_NOTE = "SIBLING NOTE"
MAX_EXISTING_SHOWN = 10  # text report; JSON always lists every entry
COPY_GROUP = "COPY GROUP"
APPROVED_FORK = "APPROVED FORK"
MAX_BACKLOG_SHOWN = 50  # text backlog; JSON always lists every group

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class ToolError(Exception):
    """A git or configuration failure; reported with exit code 2."""


# --------------------------------------------------------------------------- git


def _git_env() -> dict[str, str]:
    """The environment without variables that change how git prints diffs."""
    return {k: v for k, v in os.environ.items() if k != "GIT_DIFF_OPTS"}


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            input=stdin,
            capture_output=True,
            env=_git_env(),
        )
    except FileNotFoundError as exc:
        raise ToolError("git was not found on PATH") from exc
    except OSError as exc:
        raise ToolError(f"cannot run git in {root}: {exc}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise ToolError(f"git {' '.join(args)} failed: {err}")
    return proc.stdout


def in_scope(path: str) -> bool:
    return (
        path.endswith(".py")
        and any(path.startswith(d + "/") for d in SCOPE_DIRS)
        and not _OUT_OF_SCOPE.search(path)
    )


def resolve(root: Path, rev: str) -> str:
    return _git(root, "rev-parse", "--verify", rev + "^{commit}").decode().strip()


def _scoped_paths(raw: bytes) -> list[str]:
    """The in-scope paths of a NUL-separated git file list. Other names may hold
    any bytes; an in-scope one must be valid UTF-8 without a newline."""
    paths = []
    for name in raw.split(b"\0"):
        path = name.decode("utf-8", errors="surrogateescape")
        if not in_scope(path):
            continue
        if "\n" in path:
            raise ToolError(f"unsupported file name (newline): {name!r}")
        try:
            path.encode("utf-8")
        except UnicodeEncodeError:
            raise ToolError(f"file name is not valid UTF-8: {name!r}") from None
        paths.append(path)
    return paths


def commit_sources(root: Path, commit: str) -> dict[str, bytes]:
    """All in-scope files at ``commit``, read in one ``git cat-file --batch`` call."""
    listing = _git(
        root, "ls-tree", "-z", "-r", "--name-only", commit, "--", *SCOPE_DIRS
    )
    paths = sorted(_scoped_paths(listing))
    if not paths:
        return {}
    request = "".join(f"{commit}:./{p}\n" for p in paths).encode()
    out = _git(root, "cat-file", "--batch", stdin=request)
    sources: dict[str, bytes] = {}
    pos = 0
    for path in paths:
        newline = out.index(b"\n", pos)
        header = out[pos:newline].split()
        if header[-1] == b"missing":
            raise ToolError(f"git cat-file could not read {commit}:{path}")
        size = int(header[2])
        sources[path] = out[newline + 1 : newline + 1 + size]
        pos = newline + 1 + size + 1
    return sources


def worktree_sources(root: Path) -> tuple[dict[str, bytes], set[str]]:
    """In-scope files on disk (tracked and untracked) plus the untracked subset."""
    tracked = _scoped_paths(_git(root, "ls-files", "-z", "--", *SCOPE_DIRS))
    untracked = set(
        _scoped_paths(
            _git(
                root,
                "ls-files",
                "-z",
                "--others",
                "--exclude-standard",
                "--",
                *SCOPE_DIRS,
            )
        )
    )
    sources = {}
    for p in sorted(set(tracked) | untracked):
        path = root / p
        try:
            if path.is_symlink():  # git stores the link target as the content
                sources[p] = os.fsencode(os.readlink(path))
            elif path.is_file():
                sources[p] = path.read_bytes()
        except OSError as exc:
            raise ToolError(f"cannot read {p}: {exc}") from exc
    return sources, {p for p in untracked if p in sources}


# -------------------------------------------------------------------------- diff


@dataclass
class FileDiff:
    """Line correspondence for one file between base and head (``git diff -U0``)."""

    base_path: str | None  # None: the file does not exist at base
    head_path: str | None  # None: the file was deleted
    hunks: list[tuple[int, int, int, int]] = field(default_factory=list)

    def head_kind(self, line: int) -> str:
        """``unchanged``, ``added`` (no base line behind it) or ``modified``."""
        if self.base_path is None:
            return "added"
        for old_start, old_count, new_start, new_count in self.hunks:
            if new_count and new_start <= line < new_start + new_count:
                return "added" if old_count == 0 else "modified"
        return "unchanged"

    def _map(self, line: int, forward: bool) -> int | None:
        """Map a line of one revision to the other; forward is base to head."""
        delta = 0
        for old_start, old_count, new_start, new_count in self.hunks:
            a_start, a_count, b_start, b_count = (
                (old_start, old_count, new_start, new_count)
                if forward
                else (new_start, new_count, old_start, old_count)
            )
            if a_count and a_start <= line < a_start + a_count:
                if b_count == 0:
                    return None
                return b_start + min(line - a_start, b_count - 1)
            if a_start + max(a_count, 1) - 1 < line:
                delta += b_count - a_count
        return line + delta

    def to_base(self, line: int) -> int | None:
        return None if self.base_path is None else self._map(line, forward=False)

    def to_head(self, line: int) -> int | None:
        return None if self.head_path is None else self._map(line, forward=True)

    def base_changed(self, start: int, end: int) -> bool:
        """True when the diff deletes, replaces or inserts inside base lines start..end."""
        for old_start, old_count, _new_start, _new_count in self.hunks:
            if old_count and old_start <= end and start < old_start + old_count:
                return True
            if old_count == 0 and start <= old_start < end:
                return True
        return False


@dataclass
class Diffs:
    by_head: dict[str, FileDiff]
    by_base: dict[str, FileDiff]
    base_text: dict[str, bytes]  # in-scope base files, for comparing old code
    head_files: set[str]

    def for_head(self, path: str) -> FileDiff:
        if path in self.by_head:
            return self.by_head[path]
        return FileDiff(path if path in self.base_text else None, path)

    def for_base(self, path: str) -> FileDiff:
        if path in self.by_base:
            return self.by_base[path]
        return FileDiff(path, path if path in self.head_files else None)


def parse_hunks(text: str) -> list[list[tuple[int, int, int, int]]]:
    """The ``-U0`` hunks of every file section of a ``git diff`` patch, in order.

    File names are not read from the patch: git quotes some of them and they
    can hold spaces. ``parse_name_status`` supplies them, in the same order."""
    sections: list[list[tuple[int, int, int, int]]] = []
    for line in text.split("\n"):
        if line.startswith("diff --git "):
            sections.append([])
        elif sections and (match := _HUNK.match(line)):
            old_start, old_count, new_start, new_count = match.groups()
            sections[-1].append(
                (
                    int(old_start),
                    1 if old_count is None else int(old_count),
                    int(new_start),
                    1 if new_count is None else int(new_count),
                )
            )
    return sections


def parse_name_status(raw: bytes) -> list[tuple[str | None, str | None]]:
    """(base path, head path) of every file of ``git diff --name-status -z``;
    None when the file does not exist on that side."""
    fields = [f.decode("utf-8", errors="surrogateescape") for f in raw.split(b"\0")]
    pairs: list[tuple[str | None, str | None]] = []
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i][0]
        if status in "RC":  # rename or copy: status, old path, new path
            pairs.append((fields[i + 1], fields[i + 2]))
            i += 3
            continue
        if status == "T":  # file <-> symlink: git prints a delete and an add
            pairs += [(fields[i + 1], None), (None, fields[i + 1])]
            i += 2
            continue
        path = fields[i + 1]
        pairs.append(
            (None, path)
            if status == "A"
            else (path, None) if status == "D" else (path, path)
        )
        i += 2
    return pairs


def collect_diffs(
    root: Path,
    base: str,
    head: str | None,
    base_sources: dict[str, bytes],
    head_files: set[str],
    untracked: set[str],
) -> Diffs:
    # Pin the options that user git config or attributes can change, so hunks
    # (and therefore occurrence classes) are the same on every machine. File
    # names come from NUL-separated output, never from patch headers.
    args = [
        "-c",
        "diff.autoRefreshIndex=true",  # else an untouched, stat-dirty file has no hunk
        "diff",
        "-M",
        "-U0",
        "--diff-algorithm=myers",
        "--indent-heuristic",
        "--relative",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--text",
        "--submodule=short",
        "--inter-hunk-context=0",
        "-l1000",
    ]
    revisions = [base] if head is None else [base, head]
    names = parse_name_status(
        _git(root, *args, "--name-status", "-z", *revisions, "--", *SCOPE_DIRS)
    )
    patch = _git(root, *args, *revisions, "--", *SCOPE_DIRS)
    sections = parse_hunks(patch.decode(errors="replace"))
    if len(names) != len(sections):
        raise ToolError("could not pair the file list of git diff with its patch")
    by_head: dict[str, FileDiff] = {}
    by_base: dict[str, FileDiff] = {}
    for (base_path, head_path), hunks in zip(names, sections):
        # A side outside the scanned set (a rename from or to an excluded file)
        # does not exist for this tool: the file is then added or deleted.
        if base_path is not None and base_path not in base_sources:
            base_path, hunks = None, []
        if head_path is not None and head_path not in head_files:
            head_path = None
        fd = FileDiff(base_path, head_path, hunks)
        if head_path is not None:
            by_head[head_path] = fd
        if base_path is not None:
            by_base[base_path] = fd
    for path in untracked:
        by_head[path] = FileDiff(None, path)
    return Diffs(by_head, by_base, base_sources, head_files)


# ------------------------------------------------------------------ fingerprints


@dataclass(frozen=True)
class Window:
    path: str
    first: int  # index of the window's first token in its file
    start: int  # first source line
    end: int  # last source line
    exact: int
    renamed: int


def _function_args(args: ast.arguments):
    """Every parameter node of a function signature."""
    for arg in (
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        args.vararg,
        args.kwarg,
    ):
        if arg is not None:
            yield arg


def _scope_locals(scope: ast.AST) -> frozenset[str]:
    """Parameters and names bound in this scope.

    Nested functions, classes and lambdas are their own scopes and are skipped.
    Names declared ``global`` or ``nonlocal`` bind an outer scope, so they are
    not local even when assigned here.
    """
    names: set[str] = set()
    declared_outer: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        names.update(arg.arg for arg in _function_args(scope.args))
    stack: list[ast.AST] = list(scope.body)
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            declared_outer.update(node.names)
        stack.extend(ast.iter_child_nodes(node))
    return frozenset(names - declared_outer)


def _dotted(node: ast.AST) -> str:
    """``"sys.path.insert"`` for a Name/Attribute chain, ``""`` otherwise."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_bootstrap(value: ast.AST) -> bool:
    """Script path set-up: an ``os.path.*`` call or anything built from ``__file__``."""
    if any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(value)):
        return True
    return isinstance(value, ast.Call) and _dotted(value.func).startswith("os.path.")


def _is_setup(stmt: ast.stmt, in_class: bool = False) -> bool:
    """Recognised setup: constants, plain values, imports, field declarations,
    kernel parameter declarations, script path bootstrap, pass and docstrings.

    A def/class header is marked as setup by the caller, not here."""
    if isinstance(stmt, (ast.Pass, ast.Import, ast.ImportFrom)):
        return True
    if isinstance(stmt, ast.AnnAssign) and (stmt.value is None or in_class):
        return True  # `x: int`, or any typed field directly in a class body
    if isinstance(stmt, (ast.Return, ast.Expr)) and _is_forwarding_call(stmt.value):
        return True
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Call):
            return _dotted(stmt.value.func).startswith("sys.path.")
        return isinstance(stmt.value, ast.Constant) and isinstance(
            stmt.value.value, str
        )
    if isinstance(stmt, ast.Assign) or (isinstance(stmt, ast.AnnAssign) and stmt.value):
        value = stmt.value
        if (
            _is_bootstrap(value)
            or _is_signature_chain(value)
            or _is_forwarding_call(value)
        ):
            return True
        if isinstance(value, ast.Call):
            func = value.func
            if isinstance(func, ast.Name):
                return func.id == "IRBuilder"
            return isinstance(func, ast.Attribute) and (
                func.attr.startswith("const_") or func.attr == "param"
            )
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            value = value.operand
        while isinstance(value, ast.Attribute):
            value = value.value
        return isinstance(value, (ast.Constant, ast.Name))
    return False


def _is_forwarding_call(node: ast.AST | None) -> bool:
    """A call whose only arguments are pass-through keywords:
    ``GemmSpec(tile_m=self.tile_m, warp_m=self.warp_m)``."""
    return (
        isinstance(node, ast.Call)
        and not node.args
        and bool(node.keywords)
        and all(_is_pass_through(k) for k in node.keywords)
    )


_DIAGNOSTIC_LISTS = ("error", "problem", "reason", "issue", "warning", "message", "msg")


def _is_text(node: ast.AST) -> bool:
    """A message string: a string constant, an f-string or ``str(...)``/``repr(...)``."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    return isinstance(node, ast.JoinedStr) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("str", "repr")
    )


def _is_message(node: ast.AST | None) -> bool:
    """A rejection result: nothing, a constant, a message, or a tuple of those
    such as ``(False, f"...")``. A variable is a value, not a message."""
    if node is None or isinstance(node, ast.Constant) or _is_text(node):
        return True
    return isinstance(node, ast.Tuple) and all(
        isinstance(e, ast.Constant) or _is_text(e) for e in node.elts
    )


def _is_rejection(stmt: ast.stmt) -> bool:
    """``raise ...``, ``return <message>`` or ``errors.append(<message>)``
    on a diagnostics list (errors, problems, reasons, ...)."""
    if isinstance(stmt, ast.Raise):
        return True
    if isinstance(stmt, ast.Return):
        return _is_message(stmt.value)
    if not (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "append"
        and stmt.value.args
        and all(_is_text(a) for a in stmt.value.args)
    ):
        return False
    target = _dotted(stmt.value.func.value).rsplit(".", 1)[-1].lower()
    return any(word in target for word in _DIAGNOSTIC_LISTS)


def _is_guard(stmt: ast.stmt) -> bool:
    """A validation guard: an ``if`` whose only statement rejects, or a ``try``
    around one statement whose handlers only reject (``ArchTarget.from_gfx``)."""
    if isinstance(stmt, ast.If):
        return not stmt.orelse and len(stmt.body) == 1 and _is_rejection(stmt.body[0])
    if isinstance(stmt, ast.Try):
        return (
            len(stmt.body) == 1
            and isinstance(stmt.body[0], (ast.Assign, ast.Expr))
            and not stmt.orelse
            and not stmt.finalbody
            and all(
                len(h.body) == 1 and _is_rejection(h.body[0]) for h in stmt.handlers
            )
        )
    return False


def _is_signature_chain(node: ast.AST) -> bool:
    """``SignatureBuilder(...).ptr(...).scalar(...).build()``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        node = node.func.value
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SignatureBuilder"
    )


def _is_pass_through(node: ast.AST) -> bool:
    """A keyword argument that forwards a value of the same name:
    ``tile_m=self.tile_m`` or ``grid=grid``."""
    if not isinstance(node, ast.keyword) or node.arg is None:
        return False
    value = node.value
    if isinstance(value, ast.Name):
        return value.id == node.arg
    return isinstance(value, ast.Attribute) and value.attr == node.arg


def _is_harness(stmt: ast.stmt) -> bool:
    """A CLI/test-input helper function or an ``if __name__ == "__main__":`` block."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return stmt.name in _HARNESS_FUNCTIONS
    if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare):
        test = stmt.test
        return (
            isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )
    return False


def _child_blocks(stmt: ast.stmt):
    for name in ("body", "orelse", "finalbody"):
        block = getattr(stmt, name, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            yield block
    for handler in getattr(stmt, "handlers", ()):
        yield handler.body
    for case in getattr(stmt, "cases", ()):
        yield case.body


_DELIMITERS = frozenset({",", "(", ")", "[", "]", "{", "}", ":", "=", "."})
_SKIPPED_TOKENS = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)
# Polynomial rolling hash modulo a Mersenne prime: window keys are 61-bit.
_MOD = (1 << 61) - 1
_BASE = 1_000_003
_values: dict[str, int] = {}


def _value(text: str) -> int:
    """A fixed pseudo-random value per token text (stable across runs)."""
    value = _values.get(text)
    if value is None:
        digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
        value = _values[text] = int.from_bytes(digest, "little") % _MOD
    return value


# A local name's renamed value is this plus the distance (in tokens) to the
# previous use of the same name inside the window, or plus 0 for its first use.
# One rename map is therefore shared by the whole window: two windows match
# only when their variables flow the same way.
_LOCAL_VALUE = _value("\x00local")


Pos = tuple[int, int]  # (line, column) of a token or node start


@dataclass
class _Marks:
    """What the AST says about individual tokens of one file."""

    local: set[Pos] = field(default_factory=set)  # local variable names
    docstrings: dict[Pos, str] = field(default_factory=dict)  # -> cleandoc text
    # (start, end, is_setup) for every statement, outer before inner
    statements: list[tuple[Pos, Pos, bool]] = field(default_factory=list)
    # setup expressions inside other code (signature chains, pass-through
    # keyword arguments); applied after the statements
    setup_expressions: list[tuple[Pos, Pos]] = field(default_factory=list)
    harness: list[tuple[Pos, Pos]] = field(default_factory=list)


def _statement_start(stmt: ast.stmt) -> Pos:
    decorators = getattr(stmt, "decorator_list", None) or []
    return min(
        [(stmt.lineno, stmt.col_offset)]
        + [(d.lineno, d.col_offset - 1) for d in decorators]  # include the "@"
    )


def _mark(tree: ast.Module) -> _Marks:
    marks = _Marks()

    def docstring(body: list[ast.stmt]) -> None:
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                # Re-indenting copied code re-indents its docstrings, so they
                # are compared without their common indentation. Every other
                # string keeps its exact text.
                text = "docstring:" + repr(inspect.cleandoc(value.value))
                marks.docstrings[(value.lineno, value.col_offset)] = text

    def locals_of(body: list[ast.stmt], local_names: frozenset[str]) -> None:
        stack: list[ast.AST] = list(body)
        while stack:
            node = stack.pop()
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue  # its own scope
            if isinstance(node, ast.Name) and node.id in local_names:
                marks.local.add((node.lineno, node.col_offset))
            stack.extend(ast.iter_child_nodes(node))

    def visit(stmts: list[ast.stmt], in_class: bool = False) -> None:
        for stmt in stmts:
            span = (_statement_start(stmt), (stmt.end_lineno, stmt.end_col_offset))
            # Harness code is skipped entirely: no window may contain it, so it
            # can neither match nor complete a match with its neighbours.
            if _is_harness(stmt):
                marks.harness.append(span)
                continue
            # Validation guards are setup as a whole, body included.
            if _is_guard(stmt):
                marks.statements.append((*span, True))
                continue
            # A def/class header (decorators, signature) may be part of a copy
            # but cannot make one: its body statements override this mark.
            header = isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            marks.statements.append((*span, header or _is_setup(stmt, in_class)))
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in _function_args(stmt.args):
                    marks.local.add((arg.lineno, arg.col_offset))
                docstring(stmt.body)
                locals_of(stmt.body, _scope_locals(stmt))
                visit(stmt.body)
            elif isinstance(stmt, ast.ClassDef):
                docstring(stmt.body)
                locals_of(stmt.body, _scope_locals(stmt))
                visit(stmt.body, in_class=True)
            else:
                for child in _child_blocks(stmt):
                    visit(child)

    docstring(tree.body)
    locals_of(tree.body, _scope_locals(tree))
    visit(tree.body)
    for node in ast.walk(tree):
        if _is_pass_through(node) or _is_signature_chain(node):
            marks.setup_expressions.append(
                ((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset))
            )
    return marks


_FSTRING_START = getattr(tokenize, "FSTRING_START", None)  # Python 3.12+
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


def _in_utf8_columns(tok: tokenize.TokenInfo, lines: list[str]) -> tokenize.TokenInfo:
    """``tok`` with its columns as UTF-8 byte offsets, the unit ``ast`` positions
    use, so a non-ASCII character earlier on a line cannot shift them apart."""

    def column(pos: tuple[int, int]) -> tuple[int, int]:
        row, col = pos
        if row - 1 >= len(lines) or lines[row - 1].isascii():
            return pos
        return row, len(lines[row - 1][:col].encode())

    return tok._replace(start=column(tok.start), end=column(tok.end))


def _code_tokens(source: bytes) -> list[tokenize.TokenInfo]:
    """The code tokens of ``source``.

    From Python 3.12 an f-string is tokenized into many pieces. It is merged
    back into one STRING token, so token counts, and with them which windows
    exist, do not depend on the interpreter version.
    """
    encoding = tokenize.detect_encoding(io.BytesIO(source).readline)[0]
    lines = [raw.decode(encoding) for raw in io.BytesIO(source).readlines()]
    tokens: list[tokenize.TokenInfo] = []
    depth = 0  # open f-strings; replacement fields may hold nested ones
    opener = None
    for tok in tokenize.tokenize(io.BytesIO(source).readline):
        if tok.type in _SKIPPED_TOKENS:
            continue
        if tok.type == _FSTRING_START:
            if depth == 0:
                opener = tok
            depth += 1
        elif depth:
            if tok.type == _FSTRING_END:
                depth -= 1
                if depth == 0:
                    (srow, scol), (erow, ecol) = opener.start, tok.end
                    if srow == erow:
                        text = lines[srow - 1][scol:ecol]
                    else:
                        text = "".join(
                            [lines[srow - 1][scol:], *lines[srow : erow - 1]]
                            + [lines[erow - 1][:ecol]]
                        )
                    tokens.append(
                        tokenize.TokenInfo(
                            tokenize.STRING, text, opener.start, tok.end, opener.line
                        )
                    )
        else:
            tokens.append(tok)
    return [_in_utf8_columns(tok, lines) for tok in tokens]


@dataclass
class FileKeys:
    """Every non-trivial token window of one file, in compact arrays."""

    first: array = field(default_factory=lambda: array("l"))
    start: array = field(default_factory=lambda: array("l"))
    end: array = field(default_factory=lambda: array("l"))
    exact: array = field(default_factory=lambda: array("q"))
    renamed: array = field(default_factory=lambda: array("q"))


def file_windows(path: str, source: bytes) -> FileKeys:
    """Every window of ``MIN_TOKENS`` consecutive code tokens that is neither all
    setup nor touches harness code."""
    size = MIN_TOKENS
    # A CRLF working-tree file (Windows autocrlf) must hash like the LF blob git
    # stores: a multi-line string token would otherwise keep its "\r".
    source = source.replace(b"\r\n", b"\n")
    marks = _mark(ast.parse(source, filename=path))
    tokens = _code_tokens(source)
    keys = FileKeys()
    count = len(tokens)
    if count < size:
        return keys
    starts = [t.start for t in tokens]

    def token_range(span_start: Pos, span_end: Pos) -> range:
        return range(
            bisect.bisect_left(starts, span_start), bisect.bisect_left(starts, span_end)
        )

    # Tokens outside every statement (none in valid code) count as setup.
    setup = [True] * count
    for span_start, span_end, is_setup in marks.statements:  # inner ones override
        for i in token_range(span_start, span_end):
            setup[i] = is_setup
    for span_start, span_end in marks.setup_expressions:
        for i in token_range(span_start, span_end):
            setup[i] = True
    harness = [False] * count
    for span_start, span_end in marks.harness:
        for i in token_range(span_start, span_end):
            harness[i] = True

    text = [_value(marks.docstrings.get(t.start, t.string)) for t in tokens]
    local = [t.type == tokenize.NAME and t.start in marks.local for t in tokens]
    prev = [-1] * count  # previous use of the same local name
    nxt = [-1] * count  # next use of the same local name
    last_use: dict[str, int] = {}
    for i, t in enumerate(tokens):
        if local[i]:
            p = last_use.get(t.string, -1)
            prev[i] = p
            if p >= 0:
                nxt[p] = i
            last_use[t.string] = i

    def renamed_value(i: int, window_start: int) -> int:
        if not local[i]:
            return text[i]
        return _LOCAL_VALUE + (i - prev[i] if prev[i] >= window_start else 0)

    power = [1] * size
    for i in range(1, size):
        power[i] = power[i - 1] * _BASE % _MOD
    exact = renamed = 0
    for i in range(size):
        exact = (exact * _BASE + text[i]) % _MOD
        renamed = (renamed * _BASE + renamed_value(i, 0)) % _MOD
    # Delimiters belong to whatever surrounds them: a window whose only
    # non-setup tokens are commas and brackets holds no code of its own.
    plain = [
        setup[i] or (t.type == tokenize.OP and t.string in _DELIMITERS)
        for i, t in enumerate(tokens)
    ]
    plain_sum = [0]
    harness_sum = [0]
    for i in range(count):
        plain_sum.append(plain_sum[-1] + plain[i])
        harness_sum.append(harness_sum[-1] + harness[i])

    s = 0
    while True:
        e = s + size - 1
        if (
            harness_sum[e + 1] == harness_sum[s]
            and plain_sum[e + 1] - plain_sum[s] < size
        ):
            keys.first.append(s)
            keys.start.append(tokens[s].start[0])
            keys.end.append(tokens[e].end[0])
            keys.exact.append(exact)
            keys.renamed.append(renamed)
        if e + 1 == count:
            return keys
        # Slide by one token. When the outgoing token is a local name, its next
        # use inside the window becomes a first use: its value drops to 0.
        exact = ((exact - text[s] * power[-1]) * _BASE + text[e + 1]) % _MOD
        renamed = (renamed - renamed_value(s, s) * power[-1]) % _MOD
        k = nxt[s]
        if k != -1 and k <= e:
            renamed = (renamed - (k - s) * power[e - k]) % _MOD
        renamed = (renamed * _BASE + renamed_value(e + 1, s + 1)) % _MOD
        s += 1


def occurrence_table(windows: list[Window], attr: str) -> dict[int, list[Window]]:
    """key -> occurrences, one per run: same-key windows in one file that
    overlap one another (a block, or a repetitive stretch such as an unrolled
    loop) count once."""
    grouped: dict[tuple[int, str], list[Window]] = {}
    for w in windows:
        grouped.setdefault((getattr(w, attr), w.path), []).append(w)
    table: dict[int, list[Window]] = {}
    for (key, _path), ws in grouped.items():
        last = -1
        for w in sorted(ws, key=lambda w: w.first):
            if w.first > last:
                table.setdefault(key, []).append(w)
            last = w.first + MIN_TOKENS - 1
    for occurrences in table.values():
        occurrences.sort(key=lambda w: (w.path, w.start))
    return table


@dataclass
class Side:
    renamed: dict[int, list[Window]]
    exact: dict[int, list[Window]]
    by_path: dict[str, list[Window]]


def fingerprint(
    sources: dict[str, bytes],
    side: str,
    skipped: list[dict],
    cache: dict[tuple[str, bytes], FileKeys | str],
) -> dict[str, FileKeys]:
    """Token windows per file; files identical on both sides are keyed once."""
    keys: dict[str, FileKeys] = {}
    for path, source in sources.items():
        entry = cache.get((path, source))
        if entry is None:
            try:
                entry = file_windows(path, source)
            except (
                SyntaxError,
                ValueError,
                RecursionError,
                tokenize.TokenError,
            ) as exc:
                entry = f"{type(exc).__name__}: {exc}"
            cache[(path, source)] = entry
        if isinstance(entry, str):
            skipped.append({"side": side, "path": path, "error": entry})
        else:
            keys[path] = entry
    return keys


def build_sides(
    base: dict[str, FileKeys], head: dict[str, FileKeys]
) -> tuple[Side, Side]:
    """Window objects only for keys that repeat on either side: a key seen once
    on both sides can never be a copy. Its count on the other side is kept."""

    def repeated(files: dict[str, FileKeys]) -> set[int]:
        ordered = sorted(k for f in files.values() for k in f.renamed)
        return {a for a, b in zip(ordered, ordered[1:]) if a == b}

    relevant = repeated(base) | repeated(head)

    def side(files: dict[str, FileKeys]) -> Side:
        windows = [
            Window(path, *row)
            for path, f in files.items()
            for row in zip(f.first, f.start, f.end, f.exact, f.renamed)
            if row[4] in relevant
        ]
        renamed = occurrence_table(windows, "renamed")
        by_path: dict[str, list[Window]] = {}
        for occurrences in renamed.values():
            for w in occurrences:
                by_path.setdefault(w.path, []).append(w)
        return Side(renamed, occurrence_table(windows, "exact"), by_path)

    return side(base), side(head)


# ------------------------------------------------------------------------ groups


@dataclass(frozen=True)
class Group:
    name: str
    files: frozenset[str]
    reason: str


def _is_group_path(path: str) -> bool:
    """A path the tool can ever match: relative, normalised and in scope."""
    pure = PurePosixPath(path)
    return (
        str(pure) == path
        and not pure.is_absolute()
        and ".." not in pure.parts
        and in_scope(path)
    )


def parse_groups(text: str) -> list[Group]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolError(f"{GROUPS_FILE}: {exc}") from exc
    entries = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ToolError(
            f"{GROUPS_FILE}: top level must be an object with a 'groups' list"
        )
    groups = []
    seen: set[str] = set()
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ToolError(f"{GROUPS_FILE}: group #{i} must be an object")
        name = entry.get("name")
        files = entry.get("files")
        reason = entry.get("reason")
        if not isinstance(name, str) or not name.strip():
            raise ToolError(f"{GROUPS_FILE}: group #{i} has no 'name'")
        if name in seen:
            raise ToolError(f"{GROUPS_FILE}: group name {name!r} is used twice")
        seen.add(name)
        if (
            not isinstance(files, list)
            or not all(isinstance(f, str) for f in files)
            or len(set(files)) < 2
        ):
            raise ToolError(
                f"{GROUPS_FILE}: group {name!r} must list at least 2 different 'files'"
            )
        if not all(_is_group_path(f) for f in files):
            raise ToolError(
                f"{GROUPS_FILE}: group {name!r} 'files' must be normalised relative "
                f"Python paths this tool scans (under {', '.join(SCOPE_DIRS)})"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ToolError(f"{GROUPS_FILE}: group {name!r} has no 'reason'")
        groups.append(Group(name, frozenset(files), reason))
    return groups


def load_groups(root: Path, head: str | None) -> list[Group]:
    """The approved forks at ``head`` (the working tree when None); none when the
    file does not exist there. A file that cannot be read is an error."""
    if head is None:
        path = root / GROUPS_FILE
        if not path.is_file():
            return []
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ToolError(f"{GROUPS_FILE}: {exc}") from exc
    else:
        if not _git(root, "ls-tree", "--name-only", head, "--", GROUPS_FILE).strip():
            return []
        raw = _git(root, "show", f"{head}:./{GROUPS_FILE}")
    try:
        return parse_groups(raw.decode("utf-8-sig"))  # editors on Windows add a BOM
    except UnicodeDecodeError as exc:
        raise ToolError(f"{GROUPS_FILE}: not valid UTF-8 ({exc})") from exc


# ---------------------------------------------------------------------- analysis


@dataclass
class Location:
    path: str
    start: int
    end: int
    cls: str  # unchanged | maintained | new | replaced
    # Token index of the (latest merged) window start; used only to merge
    # windows of one copied region, never reported.
    first: int = -1
    # Window length in tokens, so same-file locations can be joined by token
    # interval instead of by line range (they can sit on one line).
    tokens: int = MIN_TOKENS


@dataclass
class Finding:
    kind: str
    match: str
    group: str | None
    before: int
    after: int
    locations: list[Location]


@dataclass
class ExistingCopies:
    """Copies that already existed and that this change edited."""

    kind: str  # EXISTING_EDITED, or SIBLING_NOTE for an approved fork group
    group: str | None
    locations: list[Location]  # cls: edited | unchanged
    # Which file revision ``Location.first`` indexes: "base" (copies that
    # existed in the base) or "head" (copies only the head side can see).
    origin: str = "base"


@dataclass
class Report:
    base: str
    head: str
    findings: list[Finding]
    existing: list[ExistingCopies]
    skipped: list[dict]


@dataclass
class Backlog:
    """Every copy group in one revision (``--backlog``)."""

    head: str
    groups: list[Finding]  # kind COPY_GROUP or APPROVED_FORK; location cls "copy"
    skipped: list[dict]


def _base_span(w: Window, fd: FileDiff) -> tuple[int, int] | None:
    lines = [
        fd.to_base(n) for n in range(w.start, w.end + 1) if fd.head_kind(n) != "added"
    ]
    lines = [n for n in lines if n is not None]
    return (min(lines), max(lines)) if lines else None


def _code_lines(source: bytes, start: int, end: int) -> tuple[str, ...]:
    """Non-blank, non-comment source lines ``start..end`` (1-based), with the
    common leading indent removed and relative indentation kept, so that
    different control flow never compares equal."""
    text = source.decode("utf-8-sig", errors="replace").split("\n")[start - 1 : end]
    lines = [t.rstrip() for t in text if t.strip() and not t.lstrip().startswith("#")]
    indent = min((len(t) - len(t.lstrip()) for t in lines), default=0)
    return tuple(t[indent:] for t in lines)


def _old_code(w: Window, diffs: Diffs) -> tuple[str, ...] | None:
    """The base-side code behind a head window (at least two lines), else None."""
    fd = diffs.for_head(w.path)
    span = _base_span(w, fd)
    if span is None or fd.base_path not in diffs.base_text:
        return None
    lines = _code_lines(diffs.base_text[fd.base_path], *span)
    return lines if len(lines) >= 2 else None


def classify(w: Window, peers: list[Window], diffs: Diffs, base: Side) -> str:
    fd = diffs.for_head(w.path)
    kinds = {fd.head_kind(n) for n in range(w.start, w.end + 1)}
    if kinds == {"unchanged"}:
        return "unchanged"
    if kinds == {"added"}:
        return "new"
    span = _base_span(w, fd)
    if span is not None and fd.base_path is not None:
        lo, hi = span
        for old in base.by_path.get(fd.base_path, []):
            if not (old.start <= hi and lo <= old.end):
                continue
            partners = [o for o in base.renamed.get(old.renamed, []) if o is not old]
            for peer in peers:
                if peer is w:
                    continue
                pfd = diffs.for_head(peer.path)
                pspan = _base_span(peer, pfd)
                if pspan is None or pfd.base_path is None:
                    continue
                if any(
                    p.path == pfd.base_path
                    and p.start <= pspan[1]
                    and pspan[0] <= p.end
                    for p in partners
                ):
                    return "maintained"
    # Copies too short for a window in the base: identical old code behind two
    # occurrences means they already existed and were edited, not pasted.
    mine = _old_code(w, diffs)
    if mine is not None and any(
        peer is not w and _old_code(peer, diffs) == mine for peer in peers
    ):
        return "maintained"
    return "replaced"


def _merge_findings(
    findings: list[Finding], anchor: tuple[str, ...] = ("new", "replaced")
) -> list[Finding]:
    """Join per-key findings of one copied region.

    Two findings merge only when they describe the same set of copies: equally
    many locations, paired in file order, each pair in one file with
    overlapping token intervals, and one token shift for every pair. A finding
    that matches only some of another's copies is a different, smaller shared
    region and stays separate. Merged locations are token-interval unions, so
    two copies that sit on one source line stay two copies.
    """
    rank = {"new": 3, "replaced": 2, "maintained": 1, "unchanged": 0, "copy": 0}

    def contributing(f: Finding) -> list[Location]:
        return [loc for loc in f.locations if loc.cls in anchor]

    order = sorted(
        findings,
        key=lambda f: (
            contributing(f)[0].path,
            contributing(f)[0].start,
            contributing(f)[0].first,
            f.kind,
        ),
    )
    merged: list[Finding] = []
    # Merged findings by file. A finding can only join one that has a location
    # in the file of its own leading location, ending at or after its start.
    active: dict[str, list[Finding]] = {}
    for f in order:
        lead = contributing(f)[0]
        pool = active.setdefault(lead.path, [])
        _prune(pool, lead, slack=0)
        target, pairs = None, None
        for m in pool:
            if m.kind == f.kind and m.group == f.group:
                pairs = _pair_locations(m.locations, f.locations)
                if pairs is not None:
                    target = m
                    break
        if target is None:
            new = Finding(
                f.kind,
                f.match,
                f.group,
                f.before,
                f.after,
                [replace(l) for l in f.locations],
            )
            merged.append(new)
            for path in {loc.path for loc in new.locations}:
                active.setdefault(path, []).append(new)
            continue
        if f.match == "renamed":
            target.match = "renamed"
        if f.after > target.after:
            target.before, target.after = f.before, f.after
        for old, loc in pairs:
            _widen(old, loc)
            if rank[loc.cls] > rank[old.cls]:
                old.cls = loc.cls
    merged.sort(key=lambda f: (contributing(f)[0].path, contributing(f)[0].start))
    return merged


def _prune(pool: list, lead: Location, slack: int) -> None:
    """Drop merged entries that end too far above ``lead`` to ever take it in.

    Leads arrive in (path, start) order, so an entry whose last line in the
    lead's file is more than ``slack`` lines above the lead can be forgotten."""
    pool[:] = [
        m
        for m in pool
        if max(l.end for l in m.locations if l.path == lead.path) + slack >= lead.start
    ]


def _widen(old: Location, loc: Location, tokens: bool = True) -> None:
    """Grow ``old`` to also cover ``loc``, another window of the same copy.

    The token interval is joined only when both come from one revision."""
    if tokens:
        lo = min(old.first, loc.first)
        old.tokens = max(old.first + old.tokens, loc.first + loc.tokens) - lo
        old.first = lo
    old.start = min(old.start, loc.start)
    old.end = max(old.end, loc.end)


def _pair_locations(
    a: list[Location], b: list[Location]
) -> list[tuple[Location, Location]] | None:
    if len(a) != len(b):
        return None
    order = lambda l: (l.path, l.first)  # noqa: E731
    pairs = list(zip(sorted(a, key=order), sorted(b, key=order)))
    shift = None
    for old, loc in pairs:
        if old.path != loc.path:
            return None
        if not (
            loc.first < old.first + old.tokens and old.first < loc.first + loc.tokens
        ):
            return None
        if shift is None:
            shift = loc.first - old.first
        elif shift != loc.first - old.first:
            return None
    return pairs


def _fork_group(paths: list[str], groups: list[Group]) -> Group | None:
    """The approved fork group holding these locations, one per sibling file."""
    return next(
        (g for g in groups if set(paths) <= g.files and len(set(paths)) == len(paths)),
        None,
    )


def find_copies(
    base: Side, head: Side, diffs: Diffs, groups: list[Group]
) -> tuple[list[Finding], list[ExistingCopies]]:
    """Copies the change added, plus existing copies it edited that only the head
    side can see (copies too short for a window in the base)."""
    findings = []
    edited: list[ExistingCopies] = []
    for key, occurrences in head.renamed.items():
        before = len(base.renamed.get(key, []))
        if len(occurrences) < 2 or len(occurrences) <= before:
            continue
        locations = [
            Location(
                w.path, w.start, w.end, classify(w, occurrences, diffs, base), w.first
            )
            for w in occurrences
        ]
        existing = [
            Location(
                l.path,
                l.start,
                l.end,
                "edited" if l.cls == "maintained" else "unchanged",
                l.first,
                l.tokens,
            )
            for l in locations
            if l.cls in ("maintained", "unchanged")
        ]
        if len(existing) >= 2 and any(l.cls == "edited" for l in existing):
            edited.append(ExistingCopies(EXISTING_EDITED, None, existing, "head"))
        added = [
            (w, loc)
            for w, loc in zip(occurrences, locations)
            if loc.cls in ("new", "replaced")
        ]
        if not added:
            continue
        exact = any(
            len(head.exact.get(w.exact, [])) >= 2
            and len(head.exact.get(w.exact, [])) > len(base.exact.get(w.exact, []))
            for w, _loc in added
        )
        # Approved forks cover copies BETWEEN sibling files: one occurrence per file.
        group = _fork_group([w.path for w in occurrences], groups)
        if group is not None:
            kind = SIBLING_COPY
        elif any(loc.cls == "new" for _w, loc in added):
            kind = NEW_COPY
        else:
            kind = LIKELY_COPY
        findings.append(
            Finding(
                kind,
                "exact" if exact else "renamed",
                group.name if group else None,
                before,
                len(occurrences),
                locations,
            )
        )
    return _merge_findings(findings), edited


def find_existing_copies(
    base: Side,
    diffs: Diffs,
    groups: list[Group],
    from_head: list[ExistingCopies],
    head_text: dict[str, bytes],
) -> list[ExistingCopies]:
    """Copy groups that already existed in the base and that the change edited.

    Each still-existing member is listed as ``edited`` or ``unchanged``; deleted
    members are dropped, so a moved copy is not an edit. Groups inside an
    approved fork become SIBLING NOTEs when a sibling was left unchanged or the
    edited siblings now differ; a fork whose sides end up identical is in sync.
    """
    raw = list(from_head)
    for occurrences in base.renamed.values():
        if len(occurrences) < 2:
            continue
        locations = []
        for w in occurrences:
            fd = diffs.for_base(w.path)
            if fd.head_path is None:
                continue
            mapped = [fd.to_head(n) for n in range(w.start, w.end + 1)]
            mapped = [n for n in mapped if n is not None]
            if not mapped:
                continue
            edited = fd.base_changed(w.start, w.end)
            locations.append(
                Location(
                    fd.head_path,
                    min(mapped),
                    max(mapped),
                    "edited" if edited else "unchanged",
                    w.first,
                )
            )
        if len(locations) >= 2 and any(l.cls == "edited" for l in locations):
            raw.append(ExistingCopies(EXISTING_EDITED, None, locations))
    items = []
    for item in raw:
        group = _fork_group([l.path for l in item.locations], groups)
        if group is None:
            items.append(item)
            continue
        codes = {
            _code_lines(head_text[l.path], l.start, l.end)
            for l in item.locations
            if l.path in head_text
        }
        if any(l.cls == "unchanged" for l in item.locations) or len(codes) > 1:
            items.append(
                ExistingCopies(SIBLING_NOTE, group.name, item.locations, item.origin)
            )
    return _merge_existing(items)


def _merge_existing(items: list[ExistingCopies]) -> list[ExistingCopies]:
    """Join entries from windows of one edited copy group.

    Entries merge only when they describe the same set of copies (same origin,
    equally many locations, paired by token position with one shift), so an
    entry for a region shared by five files is never grown by one that only
    four of them share."""

    def edited(e: ExistingCopies) -> list[Location]:
        return [l for l in e.locations if l.cls == "edited"]

    merged: list[ExistingCopies] = []
    by_path: dict[str, list[ExistingCopies]] = {}
    order = sorted(
        items,
        key=lambda e: (edited(e)[0].path, edited(e)[0].start, edited(e)[0].first),
    )
    for item in order:
        lead = edited(item)[0]
        pool = by_path.setdefault(lead.path, [])
        _prune(pool, lead, slack=1)
        target, pairs, same_origin = None, None, False
        for m in pool:
            if (m.kind, m.group) != (item.kind, item.group):
                continue
            same_origin = m.origin == item.origin
            pairs = (
                _pair_locations(m.locations, item.locations)
                if same_origin
                else _pair_by_lines(m.locations, item.locations)
            )
            if pairs is not None:
                target = m
                break
        if target is None:
            new = ExistingCopies(
                item.kind,
                item.group,
                [replace(l) for l in item.locations],
                item.origin,
            )
            merged.append(new)
            for path in {l.path for l in new.locations}:
                by_path.setdefault(path, []).append(new)
            continue
        for old, loc in pairs:
            _widen(old, loc, tokens=same_origin)
            if loc.cls == "edited":
                old.cls = "edited"
    return merged


def _pair_by_lines(
    a: list[Location], b: list[Location]
) -> list[tuple[Location, Location]] | None:
    """One-to-one pairing by file and touching line ranges, for entries whose
    token positions come from different revisions (base vs head)."""
    if len(a) != len(b):
        return None
    order = lambda l: (l.path, l.start)  # noqa: E731
    pairs = list(zip(sorted(a, key=order), sorted(b, key=order)))
    for old, loc in pairs:
        if old.path != loc.path or not (
            loc.start <= old.end + 1 and old.start <= loc.end + 1
        ):
            return None
    return pairs


def _non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def _require_root(root: Path) -> None:
    if not root.is_dir():
        raise ToolError(f"--root {root} is not a directory")


def _require_scope(*sources: dict[str, bytes]) -> None:
    """A run that scanned no Python file would report a clean bill it cannot
    justify, usually because --root is not the rocKE root."""
    if not any(sources):
        raise ToolError(
            f"no Python files under {', '.join(SCOPE_DIRS)}; is --root the rocKE root?"
        )


def analyze(root: Path, base_rev: str | None, head_rev: str | None) -> Report:
    _require_root(root)
    head = resolve(root, head_rev) if head_rev is not None else None
    if base_rev is None:
        try:
            base_rev = (
                _git(root, "merge-base", head or "HEAD", DEFAULT_UPSTREAM)
                .decode()
                .strip()
            )
        except ToolError as exc:
            raise ToolError(
                f"cannot find the merge-base with {DEFAULT_UPSTREAM} ({exc}); "
                "fetch it or pass --base <commit>"
            ) from exc
    base = resolve(root, base_rev)
    skipped: list[dict] = []
    base_sources = commit_sources(root, base)
    if head is None:
        head_sources, untracked = worktree_sources(root)
    else:
        head_sources, untracked = commit_sources(root, head), set()
    _require_scope(base_sources, head_sources)
    cache: dict[tuple[str, bytes], FileKeys | str] = {}
    base_side, head_side = build_sides(
        fingerprint(base_sources, "base", skipped, cache),
        fingerprint(head_sources, "head", skipped, cache),
    )
    diffs = collect_diffs(root, base, head, base_sources, set(head_sources), untracked)
    groups = load_groups(root, head)
    findings, from_head = find_copies(base_side, head_side, diffs, groups)
    return Report(
        base,
        head or WORKTREE,
        findings,
        find_existing_copies(base_side, diffs, groups, from_head, head_sources),
        sorted(skipped, key=lambda s: (s["side"], s["path"])),
    )


def duplicated_lines(group: Finding) -> int:
    """Lines a cleanup could remove: every copy but the largest."""
    spans = [loc.end - loc.start + 1 for loc in group.locations]
    return sum(spans) - max(spans)


def find_backlog(side: Side, groups: list[Group]) -> list[Finding]:
    """Every copy group in one side, most duplicated lines first."""
    found = []
    for occurrences in side.renamed.values():
        if len(occurrences) < 2:
            continue
        fork = _fork_group([w.path for w in occurrences], groups)
        found.append(
            Finding(
                APPROVED_FORK if fork else COPY_GROUP,
                "exact" if len({w.exact for w in occurrences}) == 1 else "renamed",
                fork.name if fork else None,
                len(occurrences),
                len(occurrences),
                [
                    Location(w.path, w.start, w.end, "copy", w.first)
                    for w in occurrences
                ],
            )
        )
    merged = _merge_findings(found, anchor=("copy",))
    return sorted(
        merged,
        key=lambda g: (-duplicated_lines(g), g.locations[0].path, g.locations[0].start),
    )


def backlog(root: Path, head_rev: str | None) -> Backlog:
    _require_root(root)
    head = resolve(root, head_rev) if head_rev is not None else None
    skipped: list[dict] = []
    sources = commit_sources(root, head) if head else worktree_sources(root)[0]
    _require_scope(sources)
    _, side = build_sides({}, fingerprint(sources, "head", skipped, {}))
    return Backlog(
        head or WORKTREE,
        find_backlog(side, load_groups(root, head)),
        sorted(skipped, key=lambda s: s["path"]),
    )


# ------------------------------------------------------------------------ output


_FIX = {
    NEW_COPY: "Fix: extract one helper and call it from every copy, or explain in the PR why it must stay a copy.",
    LIKELY_COPY: "Fix: call the existing code instead of repeating it, or explain in the PR why it must stay a copy.",
    SIBLING_COPY: "Note: copy between approved fork files; no action needed if intended.",
}


def _span(path: str, start: int, end: int) -> str:
    return f"{path}:{start}-{end}"


def _spans(locations: list[Location]) -> str:
    return ", ".join(_span(l.path, l.start, l.end) for l in locations)


def _rev_label(head: str) -> str:
    return "<working tree>" if head == WORKTREE else head[:11]


def render_text(report: Report) -> str:
    head = _rev_label(report.head)
    lines = [
        f"rocKE clone check  base {report.base[:11]}  head {head}",
        "Python only; C++ duplication not measured.",
        "",
    ]
    for f in report.findings:
        added = [loc for loc in f.locations if loc.cls in ("new", "replaced")]
        others = [loc for loc in f.locations if loc.cls not in ("new", "replaced")]
        label = (
            f"{f.kind} ({'replaced code, ' if f.kind == LIKELY_COPY else ''}{f.match})"
        )
        lines.append(
            f"{label}  "
            + ", ".join(f"{_span(l.path, l.start, l.end)} [{l.cls}]" for l in added)
        )
        if others:
            lines.append(
                "  same code at "
                + ", ".join(_span(l.path, l.start, l.end) for l in others)
            )
        lines.append(
            f"  copies {f.before} → {f.after}"
            + (f"   (group: {f.group})" if f.group else "")
        )
        lines.append("  " + _FIX[f.kind])
        lines.append("")
    shown_edits = 0
    for e in report.existing:
        edited = [l for l in e.locations if l.cls == "edited"]
        unchanged = [l for l in e.locations if l.cls == "unchanged"]
        if e.kind == SIBLING_NOTE:
            lines.append(f"SIBLING NOTE  {_spans(edited)} changed")
            if unchanged:
                lines.append(
                    f"  same block in {_spans(unchanged)} was not changed   (group: {e.group})"
                )
            else:
                lines.append(
                    f"  the sides were changed differently; check they should differ   (group: {e.group})"
                )
        elif shown_edits < MAX_EXISTING_SHOWN:
            shown_edits += 1
            lines.append(f"EXISTING COPIES EDITED  {_spans(edited)}")
            if unchanged:
                lines.append(f"  same code, not changed here, at {_spans(unchanged)}")
            lines.append(
                f"  This code already existed in {len(e.locations)} places before this change."
            )
            lines.append(
                "  Fix: extract one helper now, so the next change touches one place."
            )
        else:
            continue
        lines.append("")
    hidden = sum(e.kind == EXISTING_EDITED for e in report.existing) - shown_edits
    if hidden:
        lines.append(
            f"+{hidden} more existing copy groups edited; run with --format json for the full list"
        )
        lines.append("")
    for s in report.skipped:
        lines.append(f"SKIPPED ({s['side']})  {s['path']}: {s['error']}")
    count = lambda kind: sum(f.kind == kind for f in report.findings)  # noqa: E731
    edits = sum(e.kind == EXISTING_EDITED for e in report.existing)
    lines.append(
        f"Summary: new copies {count(NEW_COPY)} · likely copies {count(LIKELY_COPY)} · "
        f"sibling copies {count(SIBLING_COPY)} · existing copies edited {edits} · "
        f"sibling notes {len(report.existing) - edits} · "
        f"skipped files {len(report.skipped)}"
    )
    return "\n".join(lines)


def render_json(report: Report) -> str:
    def locations(items: list[Location]) -> list[dict]:
        return [
            {"path": l.path, "start": l.start, "end": l.end, "class": l.cls}
            for l in items
        ]

    data = {
        "base": report.base,
        "head": report.head,
        "scope": "python",
        "findings": [
            {
                "kind": f.kind,
                "match": f.match,
                "group": f.group,
                "copies_before": f.before,
                "copies_after": f.after,
                "locations": locations(f.locations),
            }
            for f in report.findings
        ],
        "existing_copies": [
            {"kind": e.kind, "group": e.group, "locations": locations(e.locations)}
            for e in report.existing
        ],
        "skipped": report.skipped,
    }
    return json.dumps(data, indent=2, sort_keys=True)


def render_backlog_text(report: Backlog) -> str:
    head = _rev_label(report.head)
    lines = [
        f"rocKE clone backlog  at {head}",
        "Experimental. Python only; C++ duplication not measured. Most duplicated lines first.",
        "",
    ]
    copies = [g for g in report.groups if g.kind == COPY_GROUP]
    for g in copies[:MAX_BACKLOG_SHOWN]:
        lines.append(
            f"{COPY_GROUP} ({g.match})  {len(g.locations)} copies, "
            f"{duplicated_lines(g)} duplicated lines"
        )
        lines.append("  " + _spans(g.locations))
        lines.append("")
    if len(copies) > MAX_BACKLOG_SHOWN:
        lines.append(
            f"+{len(copies) - MAX_BACKLOG_SHOWN} more copy groups; "
            "run with --format json for the full list"
        )
        lines.append("")
    for s in report.skipped:
        lines.append(f"SKIPPED  {s['path']}: {s['error']}")
    forks = len(report.groups) - len(copies)
    lines.append(
        f"Summary: copy groups {len(copies)} · "
        f"duplicated lines {sum(duplicated_lines(g) for g in copies)} · "
        f"approved-fork groups {forks} (JSON only) · "
        f"skipped files {len(report.skipped)}"
    )
    return "\n".join(lines)


def render_backlog_json(report: Backlog) -> str:
    data = {
        "mode": "backlog",
        "head": report.head,
        "scope": "python",
        "groups": [
            {
                "kind": g.kind,
                "match": g.match,
                "group": g.group,
                "copies": len(g.locations),
                "duplicated_lines": duplicated_lines(g),
                "locations": [
                    {"path": l.path, "start": l.start, "end": l.end}
                    for l in g.locations
                ],
            }
            for g in report.groups
        ],
        "skipped": report.skipped,
    }
    return json.dumps(data, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    # The text report holds non-ASCII symbols. A console or pipe that cannot show
    # them must not turn a report into a crash with exit code 1.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    ap = argparse.ArgumentParser(
        description="report Python code copies a change adds (advisory)"
    )
    ap.add_argument(
        "--base",
        type=_non_empty,
        help=f"base commit (default: merge-base with {DEFAULT_UPSTREAM})",
    )
    ap.add_argument(
        "--head", type=_non_empty, help="head commit (default: the working tree)"
    )
    ap.add_argument("--format", choices=("text", "json"), default="text")
    ap.add_argument(
        "--fail-on-new",
        action="store_true",
        help="exit 1 when a NEW or LIKELY copy is found (files that could not be "
        "parsed are listed but not counted)",
    )
    ap.add_argument(
        "--root",
        type=_non_empty,
        default=str(ROCKE),
        help="rocKE root (default: this tool's parent)",
    )
    ap.add_argument(
        "--backlog",
        action="store_true",
        help="EXPERIMENTAL: list every copy group in one revision (--head, default: "
        "the working tree), most duplicated lines first; no comparison",
    )
    args = ap.parse_args(argv)
    if args.backlog and (args.base or args.fail_on_new):
        ap.error(
            "--backlog lists one revision; it takes --head, not --base or --fail-on-new"
        )
    try:
        if args.backlog:
            listing = backlog(Path(args.root), args.head)
        else:
            report = analyze(Path(args.root), args.base, args.head)
    except ToolError as exc:
        print(f"clone_check: error: {exc}", file=sys.stderr)
        return 2
    if args.backlog:
        render = render_backlog_json if args.format == "json" else render_backlog_text
        print(render(listing))
        return 0
    print(render_json(report) if args.format == "json" else render_text(report))
    blocking = any(f.kind in (NEW_COPY, LIKELY_COPY) for f in report.findings)
    return 1 if args.fail_on_new and blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
