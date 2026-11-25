# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Clashes with other open PRs, in every review's checks: land-order traps that merge with no conflict marker and break
later (one PR moves headers another's new builders include; one renames roles another uses; one changes a selector
another's parity test relies on). A plain script: git and parsers, no model and no GitHub call.

For PR N at its head, against each other open PR M of the state (drafts too; a PR stacked on the other is left out):

**Both land orders.** N lands on develop (`git merge-tree --write-tree`, then `git commit-tree` for a temporary commit:
objects only, never a ref, never a checkout), then M merges on top; then M first and N on top. Each order's conflicting
files, with modify/delete pairs (one side deletes or moves a file the other changes) named as such. A PR that does not
merge with develop stands in by its head, and its own develop conflicts are left out of the pair's. Only pairs that
share a changed file, or that have a silent clash to check, are merged: two PRs with no file in common merge clean.

**Silent clashes** (the main target: they merge clean and break later), both ways:
- a name N removes that M's added lines use. Removed: a Python def, class, module-level name or class field; a C/C++
  function, macro, enum value, type or struct field; whose defining line N removes and that no file N changes defines
  at its head. It counts only when the merged tree of that order defines the name nowhere (`git grep` for it, then each
  hit's definitions): another definition answers the use. A file-local definition (a static function, a macro or type
  of a source file) counts only for uses in its own file;
- a file N deletes or moves that M's added lines include (C/C++), import (Python) or name (a path string), when the
  merged tree holds no other file the reference resolves to. A move that keeps the reference's tail (another include
  root, another sys.path root) says so: it breaks unless the build follows the move.

- a C/C++ function whose parameter list N changes (a regex pass over both sides' declarations and definitions: an
  arity the old side took and the new one does not) that M's added lines call with an argument count the new list does
  not take. It merges clean and fails to compile. Counted with the silent clashes, as a lead: build the merge.

- a name N removes that M's added Python lines spell as a string (lessons L-573): a CTest entry, target or option
  name N's CMake lines drop (`== "rocke_pytest_x"`, a `-R` filter) when the merged tree's CMake files name it nowhere,
  and an attribute N removes that M's added lines patch or look up by name (`mock.patch.object(mod, "_name")`,
  `monkeypatch.setattr(mod, "_name", ...)`, `getattr`, `patch("pkg.mod._name")`) when the merged tree defines it
  nowhere. Both merge clean and fail at run time (no test, an AttributeError).

**Changed under a new test** (a lead): a Python function whose logic N changes (docstrings and comments aside) and that
M's added test lines call, when the merged tree defines that name once. It merges clean, and M's test may fail on it.

**ABI lane**: the abi.h counters N bumps against every other open PR's claim on the same counter (abi_lane): two
claims both set it, so whichever lands second renumbers in every home; an `ABI NUMBER CLASH` fails the status.

**Land order**: per pair, what the second to land ports, and who that is in the merge plan's order.

**Open PRs the checks did not compare.** The checks write the scan's record (`clash.json` in the checks' folder: when
it ran, the PRs it compared); `unscanned` names the open PRs opened (or reopened) since, which `rk clash N --new`
scans, and `rk post --premises` names them.

**The pair merges built** (`rk clash N --build`, recipe R3): for each other PR, the merge with this PR landing first is
extracted (`git archive` of the merged tree's rocKE folder into a scratch folder) and the C/C++ files both change are
compiled `-fsyntax-only` with the review's shared build's flags (its core build's `compile_commands.json`, read as data
while the build's slot is kept for it; else the tree's include folders), headers as headers, and both PRs' changed test
files collected (`pytest --collect-only`), in a PR runner's slot (`prcode.Slot`; the scratch folder is the monitor's,
which the runners read): a clean textual merge that does not compile or collect reads `BUILD fail`.

**Cost and limits.** A PR's facts (merge base, files, removed names, its added lines' uses and references) are cached by
head and merge base in `cache` (a directory; none: computed each run). Git runs without lazy fetch; a PR whose changed
files' blobs are not local fetches them (one `git diff` request) while `max_fetch` allows, else it is NOT RUN, as are
heads that are not local, merge bases beyond the shallow mirror and the PRs left when `budget_s` runs out. A check that
ran nothing never reads as clean.
"""

from __future__ import annotations

import ast
import bisect
import contextlib
import copy
import datetime as dt
import functools
import hashlib
import json
import keyword
import os
import pathlib
import posixpath
import re
import shlex
import shutil
import subprocess
import tempfile
import time

from . import abi, gitops, overlaps, paths, prcode, prtree, registry, slots, whole_file
from .conflicts import short

SECTION = "Clashes with other open PRs"
RECORD = "clash.json"  # the checks' scan record, in the checks' folder
TEST_FILE = re.compile(r"(?:^|/)test_[\w.-]*\.py$")
C_SOURCE = re.compile(r"\.(?:c|cc|cpp|cxx|hip)$")
C_HEADER = re.compile(r"\.(?:h|hh|hpp|hxx|inl)$")
BUILD_MOST, TESTS_MOST = (
    12,
    30,
)  # C/C++ files compiled and test files collected per pair
DEVELOP = "refs/remotes/origin/develop"
CACHE_VERSION = (
    3  # bump when the facts change shape or meaning, so every cached PR is read again
)
GIT_ERRORS = (gitops.GitError, OSError, subprocess.SubprocessError, ValueError)
NO_LAZY = {"GIT_NO_LAZY_FETCH": "1"}
# A landing's temporary commit: a fixed identity and date, so the same landing is the same object on every run.
TEMP_ENV = {
    "GIT_AUTHOR_NAME": "rkmon",
    "GIT_AUTHOR_EMAIL": "rkmon@localhost",
    "GIT_AUTHOR_DATE": "1970-01-01T00:00:00+0000",
    "GIT_COMMITTER_NAME": "rkmon",
    "GIT_COMMITTER_EMAIL": "rkmon@localhost",
    "GIT_COMMITTER_DATE": "1970-01-01T00:00:00+0000",
}
SHOWN = 3  # uses, files and names shown per item
USES_KEPT, USES_PER_FILE = 8, 2  # use sites kept per name in a PR's facts, and per file
COMMON_FILES = 200  # a name in more files than this of the merged tree is defined somewhere: not looked up further
MAX_REFS = 5000  # path references kept per PR
GENERIC = overlaps.STOP | {
    "get",
    "set",
    "add",
    "update",
    "name",
    "names",
    "value",
    "values",
    "size",
    "data",
    "type",
    "types",
    "key",
    "keys",
    "items",
    "result",
    "config",
    "options",
    "path",
    "paths",
    "args",
    "kwargs",
    "self",
    "cls",
    "op",
    "ops",
    "info",
    "fn",
    "func",
    "test",
    "tests",
    "setup",
    "teardown",
    "default",
    "defaults",
    "kind",
    "mode",
    "shape",
    "dtype",
    "index",
    "count",
    "length",
    "format",
    "parse",
    "load",
    "save",
    "read",
    "write",
    "check",
    "validate",
    "reset",
    "clear",
    "close",
    "open",
    "start",
    "stop",
    "init",
    "create",
    "copy",
    "apply",
    "emit",
}
REFS_PER_FILE = 500  # path references kept per file (a data file can name thousands)
CODE = ("py", "c")
DOCS = (
    ".md",
    ".rst",
    ".txt",
    ".html",
    ".svg",
    ".png",
    ".jpg",
)  # prose and pictures: a path in them breaks nothing
TESTISH = overlaps.TEST_PATH
# Language words: never a name another PR can remove.
NOT_NAMES = set(keyword.kwlist) | {
    "None",
    "True",
    "False",
    "self",
    "cls",
    "int",
    "str",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "len",
    "range",
    "print",
    "isinstance",
    "const",
    "char",
    "void",
    "unsigned",
    "signed",
    "long",
    "short",
    "double",
    "auto",
    "static",
    "inline",
    "struct",
    "class",
    "enum",
    "union",
    "typedef",
    "using",
    "namespace",
    "template",
    "typename",
    "return",
    "sizeof",
    "nullptr",
    "NULL",
    "include",
    "define",
    "ifdef",
    "ifndef",
    "endif",
    "else",
    "elif",
    "size_t",
    "uint32_t",
    "int32_t",
    "uint64_t",
    "int64_t",
    "uint8_t",
    "int8_t",
    "std",
}


class NotRun(Exception):
    """A PR the scan could not read (the reason is the message)."""


@functools.lru_cache(maxsize=32)
def _tree(text: str) -> ast.Module:
    """One parse per file text: a file's definitions, logic, imports and bindings all read the same tree (none
    changes it)."""
    return ast.parse(text)


def _is_doc(path: str) -> bool:
    return (
        path.endswith(DOCS) and overlaps.language(path) != "cmake"
    )  # CMakeLists.txt is code


def settings(cfg: dict) -> dict:
    """The clash scan's limits: config `clashes` over the package's (config.defaults.json), entry by entry."""
    return dict(registry.default("clashes"), **(cfg.get("clashes") or {}))


def _git(repo: str, *args: str, timeout: int = 300, check: bool = True) -> str:
    return gitops.git(repo, *args, timeout=timeout, check=check, lazy=False)


def _base(name: str) -> str:
    """A qualified name's own part: `Class.method`, `ns::Type::f` and `~T` read `method`, `f` and `T`."""
    return re.split(r"\.|::", name)[-1].lstrip("~")


def _line_of(starts: list[int], i: int) -> int:
    return bisect.bisect_right(starts, i)


def _starts(text: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer("\n", text)]


# --- one file: its definitions, its code's names and references ------------------------------------------------------


def _py_defs(text: str) -> list[tuple[str, str, int, bool]]:
    """Python definitions: functions and methods, classes, module-level names and class fields (an assignment in a
    class body, a dataclass field), also under a module-level `if`, `try` or `with`."""
    out: list[tuple[str, str, int, bool]] = []

    def targets(t) -> list[ast.Name]:
        if isinstance(t, ast.Name):
            return [t]
        if isinstance(t, (ast.Tuple, ast.List)):
            return [x for e in t.elts for x in targets(e)]
        return []

    def visit(body: list, in_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(
                    ("method" if in_class else "def", node.name, node.lineno, False)
                )
            elif isinstance(node, ast.ClassDef):
                out.append(("class", node.name, node.lineno, False))
                visit(node.body, True)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                ):
                    out.extend(
                        ("field" if in_class else "global", x.id, x.lineno, False)
                        for x in targets(t)
                    )
            elif (
                isinstance(node, (ast.If, ast.Try, ast.With))
                or type(node).__name__ == "TryStar"
            ):
                for block in (
                    getattr(node, "body", []),
                    getattr(node, "orelse", []),
                    getattr(node, "finalbody", []),
                    *[h.body for h in getattr(node, "handlers", [])],
                ):
                    visit(block, in_class)

    visit(_tree(text).body, False)
    return out


ENUM_HEAD = re.compile(
    r"\benum\b(?:\s+(?:class|struct))?(?:\s+[A-Za-z_]\w*)?(?:\s*:\s*[\w:\s]+?)?\s*\{"
)
RECORD_HEAD = re.compile(
    r"\b(?:struct|union|class)\b(?:\s+[A-Za-z_]\w*)*\s*(?::[^{;()]*)?\{"
)
FN_POINTER = re.compile(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\(")
NOT_FIELD = re.compile(
    r"\s*(?:typedef|using|friend|static_assert|template|enum|struct|class|union|return|namespace)\b"
)


def _pieces(
    text: str, start: int, end: int, sep: str, angle: bool = False
) -> list[tuple[int, str]]:
    """(offset, piece) of text[start:end] split at `sep` outside brackets (and `<…>` when `angle`)."""
    opens, closes = ("([{<", ")]}>") if angle else ("([{", ")]}")
    out, depth, a = [], 0, start
    for i in range(start, end):
        ch = text[i]
        if ch in opens:
            depth += 1
        elif ch in closes:
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            out.append((a, text[a:i]))
            a = i + 1
    out.append((a, text[a:end]))
    return out


def _c_members(text: str, private: bool) -> list[tuple[str, str, int, bool]]:
    """Enum values and struct, class and union fields (function-pointer fields too) of a C/C++ file."""
    masked = overlaps._c_masks(text)[
        0
    ]  # one home for what is code: the overlap scan's masks
    starts, out = _starts(text), []
    for m in ENUM_HEAD.finditer(masked):
        o = m.end() - 1
        c = overlaps._closing(masked, o, len(masked))
        for off, piece in _pieces(masked, o + 1, c, ","):
            nm = re.match(r"\s*(?:\[\[.*?\]\]\s*)?([A-Za-z_]\w*)", piece)
            if nm:
                out.append(
                    (
                        "enumerator",
                        nm.group(1),
                        _line_of(starts, off + nm.start(1)),
                        private,
                    )
                )
    for m in RECORD_HEAD.finditer(masked):
        o = m.end() - 1
        c = overlaps._closing(masked, o, len(masked))
        i = stmt = o + 1
        while i < c:
            ch = masked[i]
            if ch == "{":
                close = overlaps._closing(masked, i, c)
                if "(" in masked[stmt:i]:  # a method's body: the statement ends with it
                    stmt = close + 1
                i = close + 1
                continue
            if ch == ";":
                out.extend(_fields(masked, stmt, i, starts, private))
                stmt = i + 1
            elif ch == ":" and re.fullmatch(
                r"\s*(?:public|private|protected)\s*", masked[stmt:i]
            ):
                stmt = i + 1
            i += 1
    return out


def _fields(
    masked: str, a: int, b: int, starts: list[int], private: bool
) -> list[tuple[str, str, int, bool]]:
    """The field names one member statement declares (masked[a:b], without its `;`)."""
    s = masked[a:b]
    if not s.strip() or NOT_FIELD.match(s):
        return []
    if "(" in s:
        fp = FN_POINTER.search(s)
        return (
            [("field", fp.group(1), _line_of(starts, a + fp.start(1)), private)]
            if fp
            else []
        )
    out = []
    for off, piece in _pieces(masked, a, b, ",", angle=True):
        decl = re.split(r"=|\{|\[|(?<!:):(?!:)", piece, maxsplit=1)[0]
        words = list(re.finditer(r"[A-Za-z_]\w*", decl))
        if not words or (off == a and len(words) < 2):
            continue  # the first declarator needs a type before its name
        w = words[-1]
        if w.group() not in overlaps.C_SPEC:
            out.append(("field", w.group(), _line_of(starts, off + w.start()), private))
    return out


def defs(path: str, text: str) -> list[tuple[str, str, int, bool]]:
    """(kind, name, line, file-local) of every definition in one file; a qualified name by its own part. A file the
    parsers cannot read contributes nothing."""
    lang = overlaps.language(path)
    try:
        if lang == "py":
            try:
                return _py_defs(text)
            except (SyntaxError, ValueError, RecursionError):
                return [
                    (d["kind"], _base(d["name"]), d["line"], False)
                    for d in overlaps.file_defs(path, text)
                ]
        if lang == "c":
            local = not path.endswith(overlaps.HEADER)
            out = [
                (d["kind"], _base(d["name"]), d["line"], bool(d.get("private")))
                for d in overlaps.file_defs(path, text)
            ]
            return out + _c_members(text, local)
    except Exception:  # noqa: BLE001 - one odd file must not stop the scan
        return []
    return []


def py_logic(text: str) -> dict[str, tuple[str, int, str]]:
    """Each Python function and method (`Class.method`) -> (name, line, digest of its logic: the syntax tree without
    the docstring; comments are not in it)."""
    out: dict[str, tuple[str, int, str]] = {}

    def visit(body: list, prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n2 = copy.copy(node)
                doc = (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )
                n2.body = node.body[1:] if doc else node.body
                digest = hashlib.sha1(
                    ast.dump(n2, include_attributes=False).encode()
                ).hexdigest()[:12]
                out[prefix + node.name] = (node.name, node.lineno, digest)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for block in (
                    node.body,
                    getattr(node, "orelse", []),
                    getattr(node, "finalbody", []),
                ):
                    visit(block, prefix)

    try:
        visit(_tree(text).body, "")
    except (SyntaxError, ValueError, RecursionError):
        return {}
    return out


IDENT = re.compile(r"[A-Za-z_]\w*")

# C/C++ parameter lists, from a regex pass over the masked text (no compiler): a name followed by `(`. A declaration or
# definition has a type before its name (a word, `*`, `&` or `>`) and, after its `)`, a body, a `;`, a qualifier or a
# constructor's initialiser list; a call has neither. Arity: (required, most) parameters; defaults lower the first, a
# `...` lifts the second to VARIADIC.
C_NAME_PAREN = re.compile(r"(?<![\w.:>])([A-Za-z_]\w*)\s*\(")
NOT_FN = {
    "if",
    "while",
    "for",
    "switch",
    "return",
    "sizeof",
    "alignof",
    "alignas",
    "decltype",
    "catch",
    "new",
    "delete",
    "throw",
    "defined",
    "static_assert",
    "noexcept",
    "typeid",
    "__attribute__",
    "__declspec",
    "operator",
    "case",
    "else",
    "do",
    "goto",
    "co_return",
    "co_await",
    "co_yield",
}
DECL_AFTER = re.compile(
    r"\s*(?:\{|;|const\b|noexcept\b|override\b|final\b|->|throw\s*\(|=\s*(?:0|default|delete)\b|"
    r":(?!:))"
)
TYPE_BEFORE = re.compile(r"([A-Za-z_]\w*|[*&>])\s*$")
VARIADIC = 999
CALLS_KEPT = 8  # call sites kept per name in a PR's facts


def _close(text: str, i: int) -> int:
    """The index of the `)` that closes the `(` at text[i], or -1."""
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return j if c == ")" else -1
    return -1


def _top_parts(s: str) -> list[str]:
    """`s` split at its top-level commas (parentheses, brackets, braces and angle brackets nest)."""
    parts, depth, cur = [], 0, ""
    for c in s:
        if c in "([{<":
            depth += 1
        elif c in ")]}>" and depth:
            depth -= 1
        if c == "," and not depth:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    return parts + [cur]


def arity(params: str) -> tuple[int, int]:
    """(required, most) parameters of a parameter list's text."""
    s = " ".join(params.split())
    if s in ("", "void"):
        return 0, 0
    parts = [p.strip() for p in _top_parts(s)]
    variadic = parts[-1].endswith("...")
    return (
        sum(1 for p in parts if "=" not in p and not p.endswith("...")),
        VARIADIC if variadic else len(parts),
    )


def c_signatures(text: str) -> tuple[dict[str, list[list]], list[list]]:
    """A C/C++ file's declarations and definitions, name -> [[first line, last line, required, most], ...], and its
    calls, [[name, line, argument count], ...] (comments and literals masked)."""
    masked = overlaps.C_LIT.sub(overlaps._blank, text)
    starts = _starts(masked)
    sigs: dict[str, list[list]] = {}
    calls: list[list] = []
    for m in C_NAME_PAREN.finditer(masked):
        name = m.group(1)
        if name in NOT_FN:
            continue
        close = _close(masked, m.end() - 1)
        if close < 0:
            continue
        inside = masked[m.end() : close]
        before = TYPE_BEFORE.search(masked[max(0, m.start() - 120) : m.start()])
        line = _line_of(starts, m.start())
        if (
            before
            and before.group(1) not in NOT_FN | {"define"}
            and DECL_AFTER.match(masked, close + 1)
        ):
            sigs.setdefault(name, []).append(
                [line, _line_of(starts, close), *arity(inside)]
            )
        elif not (before and before.group(1) == "define"):
            parts = _top_parts(" ".join(inside.split()))
            calls.append([name, line, 0 if parts == [""] else len(parts)])
    return sigs, calls


def changed_signatures(
    path: str, old: str, new: str, minus: list, plus: list
) -> list[dict]:
    """The C/C++ functions of a file whose parameter lists the change makes refuse an arity they took: a name declared
    or defined at both sides, one of its new declarations on added lines or one of its old ones on removed lines, and
    an old (required, most) the new side no longer has. Each: name, path, line (its first new declaration), old and new
    arities."""
    before, _ = c_signatures(old)
    after, _ = c_signatures(new)
    out = []
    for name in sorted(set(before) & set(after)):
        if len(name) < 4 or name in NOT_NAMES or name in GENERIC:
            continue
        touched = any(_in_span(plus, a, b) for a, b, *_ in after[name]) or any(
            _in_span(minus, a, b) for a, b, *_ in before[name]
        )
        old_ar = sorted({(r, m) for _, _, r, m in before[name]})
        new_ar = sorted({(r, m) for _, _, r, m in after[name]})
        gone = [a for a in old_ar if a not in new_ar]
        if touched and gone:
            out.append(
                {
                    "name": name,
                    "kind": "function",
                    "path": path,
                    "line": after[name][0][0],
                    "old": [list(a) for a in gone],
                    "new": [list(a) for a in new_ar],
                }
            )
    return out


def _in_span(ranges: list, a: int, b: int) -> bool:
    return any(x <= b and a <= y for x, y in ranges)


def code_lines(path: str, text: str) -> list[str] | None:
    """The file's lines with comments and the insides of literals blank (the overlap scan's masks), for names used in
    code; None for a file that is not Python or C/C++."""
    lang = overlaps.language(path)
    if lang == "py":
        return overlaps.PY_LIT.sub(overlaps._blank, text).split("\n")
    if lang == "c":
        return overlaps.C_LIT.sub(overlaps._blank, text).split("\n")
    return None


STRING_NAME = re.compile(r"^[A-Za-z_]\w*$")
PATCH_CALLS = {
    "object",
    "setattr",
    "getattr",
    "hasattr",
    "delattr",
}  # (obj, "name") calls: an attribute by name
CMAKE_FILE = re.compile(r"(?:^|/)CMakeLists\.txt$|\.cmake$")
CMAKE_WORDS = {
    "COMMAND",
    "WORKING_DIRECTORY",
    "PROPERTIES",
    "ENVIRONMENT",
    "LABELS",
    "TIMEOUT",
    "DEPENDS",
    "PRIVATE",
    "PUBLIC",
    "INTERFACE",
    "STATUS",
    "WARNING",
    "FATAL_ERROR",
    "APPEND",
    "endif",
    "endforeach",
    "endfunction",
    "endmacro",
    "foreach",
    "function",
    "add_test",
    "set_tests_properties",
    "list",
    "string",
    "message",
}
CMAKE_KEPT = 2000  # CMake names kept per PR


def py_strings(text: str, plus: list[tuple[int, int]]) -> list[tuple[str, int, str]]:
    """The identifier-shaped string constants on a Python file's added lines: (name, line, how); how is "patch" for an
    attribute named by a string (`patch.object(mod, "x")`, `setattr`/`getattr`/`hasattr`/`delattr(obj, "x")`, the last
    part of `patch("pkg.mod.x")`), else "string"."""
    try:
        tree = _tree(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    out, done = [], set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name in PATCH_CALLS and len(node.args) >= 2:
            a = node.args[1]
        elif name == "patch" and node.args:
            a = node.args[0]
        else:
            continue
        if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
            continue
        last = a.value.rsplit(".", 1)[-1] if name == "patch" else a.value
        if STRING_NAME.match(last) and _in(plus, a.lineno):
            out.append((last, a.lineno, "patch"))
            done.add(id(a))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in done
            and STRING_NAME.match(node.value)
            and len(node.value) >= 5
            and _in(plus, node.lineno)
        ):
            out.append((node.value, node.lineno, "string"))
    return [x for x in out if x[0] not in GENERIC and x[0] not in NOT_NAMES]


def cmake_names(text: str) -> set[str]:
    """The identifier-shaped words of a CMake file, comments aside: its test, target, option and variable names."""
    words = set()
    for line in text.split("\n"):
        words.update(IDENT.findall(line.split("#", 1)[0]))
    return words


def cmake_removed(
    path: str, old: str, new: str, minus: list[tuple[int, int]]
) -> list[dict]:
    """The names a CMake file's removed lines carry that its new text names nowhere (a CTest entry, a target, an
    option the PR drops or renames): {"name", "kind": "CMake name", "path", "line"}."""
    keep = cmake_names(new)
    out, seen = [], set()
    lines = old.split("\n")
    for ln in _lines_in(minus, len(lines)):
        for w in IDENT.findall(lines[ln - 1].split("#", 1)[0]):
            if (
                len(w) >= 5
                and w not in keep
                and w not in seen
                and w not in CMAKE_WORDS
                and w not in GENERIC
                and not w.startswith("CMAKE_")
            ):
                seen.add(w)
                out.append({"name": w, "kind": "CMake name", "path": path, "line": ln})
    return out


def module_words(text: str) -> list[str]:
    """The module names a Python file reaches: every part of each module it imports and each name it imports from one,
    and the stem of every `.py` path it spells in a string (a test that loads a file by path).
    """
    out: set[str] = set()
    try:
        tree = _tree(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(part for a in node.names for part in a.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            out.update((node.module or "").split("."))
            out.update(a.name for a in node.names)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and ".py" in node.value
        ):
            out.update(re.findall(r"([A-Za-z_]\w*)\.py\b", node.value))
    return sorted(out - {""})


def py_bindings(text: str) -> set[str]:
    """The names a Python file binds itself, in any scope: assignment and loop targets, parameters, defs and classes,
    `as` names (of an import, a `with`, an `except`, a match). Not `from m import name`: that is a use of m's name.
    """
    try:
        tree = _tree(text)
    except (SyntaxError, ValueError, RecursionError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.alias) and node.asname:
            out.add(node.asname)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif type(node).__name__ in ("MatchAs", "MatchStar") and getattr(
            node, "name", None
        ):
            out.add(node.name)
    return out


def module_stem(path: str) -> str:
    """The name a module is imported by: its file's stem, or its package's for an `__init__.py`."""
    stem = posixpath.splitext(posixpath.basename(path))[0]
    return posixpath.basename(posixpath.dirname(path)) if stem == "__init__" else stem


def _lines_in(ranges: list[tuple[int, int]], n: int):
    for a, b in ranges:
        yield from range(max(a, 1), min(b, n) + 1)


def _after_dot(line: str, i: int) -> bool:
    """Whether the name at line[i] follows a `.` (spaces between): an attribute, not a bare name."""
    i -= 1
    while i >= 0 and line[i] in " \t":
        i -= 1
    return i >= 0 and line[i] == "."


INCLUDE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"\n]+)[>\"]")
PATH_TOKEN = re.compile(r"(?<![\w/.-])(?:\.{1,2}/)*[\w.+-]+(?:/[\w.+-]+)+")
HAS_EXT = re.compile(r"\.[A-Za-z0-9_]{1,8}$")


def _norm(path: str) -> str:
    p = posixpath.normpath(path)
    return "" if p.startswith("..") or p == "." else p


def _module_paths(module: str) -> list[str]:
    stem = module.replace(".", "/")
    return [stem + ".py", stem + "/__init__.py"]


def refs(path: str, text: str, plus: list[tuple[int, int]]) -> list[dict]:
    """The file references on a file's added lines: C/C++ includes, Python imports, path strings. Each: the reference,
    its line, the exact paths it can name (relative to the file) and the path tails it can end (2+ parts).
    """
    lines, here = text.split("\n"), posixpath.dirname(path)
    added = set(_lines_in(plus, len(lines)))
    out: list[dict] = []
    lang = overlaps.language(path)

    def add(kind: str, ref: str, line: int, exact: list[str], tails: list[str]) -> None:
        exact = [e for e in (_norm(x) for x in exact) if e]
        tails = [t for t in tails if t.count("/") >= 1]
        if exact or tails:
            out.append(
                {"kind": kind, "ref": ref, "line": line, "exact": exact, "tails": tails}
            )

    skip: set[int] = set()
    if lang == "c":
        for ln in sorted(added):
            m = INCLUDE.match(lines[ln - 1])
            if m:
                s = m.group(1).strip()
                add(
                    "include", s, ln, [posixpath.join(here, s)], [s] if "/" in s else []
                )
                skip.add(ln)
    elif lang == "py":
        try:
            tree = _tree(text)
        except (SyntaxError, ValueError, RecursionError):
            tree = None
        for node in ast.walk(tree) if tree else ():
            if (
                not isinstance(node, (ast.Import, ast.ImportFrom))
                or node.lineno not in added
            ):
                continue
            skip.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            if isinstance(node, ast.Import):
                for a in node.names:
                    add(
                        "import",
                        a.name,
                        node.lineno,
                        _module_paths(posixpath.join(here, a.name.replace(".", "/"))),
                        _module_paths(a.name) if "." in a.name else [],
                    )
                continue
            mod, level = node.module or "", node.level or 0
            mods = ([mod] if mod else []) + [
                f"{mod}.{a.name}" if mod else a.name
                for a in node.names
                if a.name != "*"
            ]
            for m in mods:
                if level:
                    root = here
                    for _ in range(level - 1):
                        root = posixpath.dirname(root)
                    exact = _module_paths(posixpath.join(root, m.replace(".", "/")))
                    add("import", "." * level + m, node.lineno, exact, [])
                else:
                    add(
                        "import",
                        m,
                        node.lineno,
                        [],
                        _module_paths(m) if "." in m else [],
                    )
    if not _is_doc(path):
        for ln, s in _path_strings(path, text, added - skip):
            tail = re.sub(r"^(?:\.{1,2}/)+", "", s)
            add("path", s, ln, [posixpath.join(here, s)], [tail] if "/" in tail else [])
    return out[:REFS_PER_FILE]


def _path_strings(path: str, text: str, lines: set[int]):
    """(line, token) for the path-like tokens (a `/` and an extension) on the given lines: in a Python or C/C++ file
    inside string literals only, in a CMake file outside comments, elsewhere anywhere. A URL's path is not a file.
    """
    lang = overlaps.language(path)
    if lang in CODE:
        rx = overlaps.PY_LIT if lang == "py" else overlaps.C_LIT
        starts = _starts(text)
        spans = [
            (m.start(), m.group())
            for m in rx.finditer(text)
            if m.group()[:1] not in "#/"
        ]
    else:
        body = re.sub(r"#[^\n]*", "", text) if lang == "cmake" else text
        spans, starts = [(0, body)], _starts(body)
    for off, s in spans:
        first = _line_of(starts, off)
        if lang in CODE and not any(
            first <= ln <= first + s.count("\n") for ln in lines
        ):
            continue
        for m in PATH_TOKEN.finditer(s):
            ln = _line_of(starts, off + m.start())
            tok = m.group()
            if (
                ln in lines
                and HAS_EXT.search(tok)
                and "://" not in s[max(0, m.start() - 3) : m.start()]
            ):
                yield ln, tok


# --- one PR: what it removes, what its added lines use and name ------------------------------------------------------

HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def hunks(diff: str) -> dict[str, dict]:
    """Each file of a `git diff -U0 --no-renames` patch: the old-side line ranges it removes (`minus`) and the new-side
    ranges it adds (`plus`). Keyed by path (the old path of a deleted file)."""
    out: dict[str, dict] = {}
    cur, skip = None, 0
    for line in diff.split("\n"):
        if line.startswith("\\"):
            continue  # "\ No newline at end of file" is not counted in the hunk
        if skip:
            skip -= 1
            continue
        if line.startswith("diff --git "):
            cur = None
        elif line.startswith("--- "):
            cur = {
                "old": line[6:] if line.startswith("--- a/") else None,
                "new": None,
                "minus": [],
                "plus": [],
            }
        elif line.startswith("+++ ") and cur is not None:
            cur["new"] = line[6:] if line.startswith("+++ b/") else None
            key = cur["new"] or cur["old"]
            if key:
                out[key] = cur
        elif line.startswith("@@ ") and cur is not None:
            m = HUNK.match(line)
            if not m:
                continue
            o0, oc = int(m.group(1)), int(m.group(2)) if m.group(2) is not None else 1
            n0, nc = int(m.group(3)), int(m.group(4)) if m.group(4) is not None else 1
            skip = oc + nc
            if oc:
                cur["minus"].append((o0, o0 + oc - 1))
            if nc:
                cur["plus"].append((n0, n0 + nc - 1))
    return out


def _in(ranges: list[tuple[int, int]], line: int) -> bool:
    return any(a <= line <= b for a, b in ranges)


def _raw_blobs(raw: str) -> tuple[list[str], set[str]]:
    """The paths and the blob ids (both sides) of a `git diff --raw -z --no-abbrev --no-renames`."""
    parts, paths, blobs, i = raw.split("\0"), [], set(), 0
    while i < len(parts) - 1:
        meta = parts[i]
        if not meta.startswith(":"):
            i += 1
            continue
        f = meta.split()
        blobs.update(x for x in f[2:4] if x.strip("0"))
        paths.append(parts[i + 1])
        i += 2
    return paths, blobs


def missing(repo: str, blobs: set[str]) -> int:
    """How many of the blob ids the clone does not have (no fetch)."""
    if not blobs:
        return 0
    p = subprocess.run(
        ["git", "-C", repo, "cat-file", "--batch-check"],
        input="\n".join(sorted(blobs)) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **NO_LAZY},
    )
    return sum(1 for line in p.stdout.splitlines() if line.endswith(" missing"))


def fetch_blobs(repo: str, a: str, b: str, paths: list[str] | None = None) -> None:
    """Fetch the blobs a diff of a..b (over `paths`) needs, in one request: git's diff prefetches what a partial clone
    lacks, as the overlap scan's head reads do (lazy fetch on, whatever the caller's environment says).
    """
    env = {k: v for k, v in os.environ.items() if k != "GIT_NO_LAZY_FETCH"}
    subprocess.run(
        [
            "git",
            "-C",
            repo,
            "diff",
            "--no-renames",
            "--no-color",
            "-U0",
            a,
            b,
            *(["--", *paths] if paths else []),
        ],
        capture_output=True,
        timeout=900,
        env=env,
    )


def _moves(repo: str, mb: str, head: str) -> list[list]:
    """[old, new] for each file the PR deletes (new None) or moves, from git's rename detection (no lazy fetch)."""
    return [
        [c["old"], c["new"]]
        for c in prtree.changes(repo, mb, head, lazy=False)
        if c["status"] in ("D", "R")
    ]


def facts(
    repo: str, head: str, base: str, fetch=None, cache: str | None = None
) -> dict:
    """What PR `head` does against merge-base(base, head) (module docstring). `fetch()` says whether a blob fetch is
    still allowed (and spends it); NotRun when the head is not local, the merge base is beyond the shallow mirror, or
    blobs are missing and no fetch is allowed."""
    if not gitops.has_commit(repo, head):
        raise NotRun("its head is not in the mirror")
    try:
        mb = _git(repo, "merge-base", base, head).strip()
    except gitops.GitError:
        mb = ""
    if not mb:
        raise NotRun("no merge base in the mirror (beyond its shallow history)")
    path = (
        pathlib.Path(cache, f"{head[:16]}-{mb[:16]}-v{CACHE_VERSION}.json")
        if cache
        else None
    )
    if path and path.exists():
        try:
            got = json.loads(path.read_text())
            os.utime(path)  # in use: prune_cache keeps it
            return got
        except (OSError, ValueError):
            pass
    raw = _git(
        repo,
        "-c",
        "core.quotepath=false",
        "diff",
        "--raw",
        "-z",
        "--no-abbrev",
        "--no-renames",
        mb,
        head,
        timeout=600,
    )
    paths, blobs = _raw_blobs(raw)
    if missing(repo, blobs):
        if not (fetch and fetch()):
            raise NotRun("its changed files' blobs are not local (fetch budget spent)")
        fetch_blobs(repo, mb, head)
        if missing(repo, blobs):
            raise NotRun("its changed files' blobs could not be fetched")
    diff = _git(
        repo,
        "-c",
        "core.quotepath=false",
        "diff",
        "-U0",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        mb,
        head,
        timeout=600,
    )
    hs = hunks(diff)
    code = {p: h for p, h in hs.items() if overlaps.language(p) in CODE}
    old_specs = {
        p: f"{mb}:{h['old']}" for p, h in code.items() if h["old"] and h["minus"]
    }
    new_specs = {p: f"{head}:{h['new']}" for p, h in code.items() if h["new"]}
    texts = gitops.cat_blobs(
        repo, sorted(set(old_specs.values()) | set(new_specs.values())), timeout=600
    )
    new_names: set[str] = set()
    added_names: set[str] = set()
    for p, spec in new_specs.items():
        text = texts.get(spec)
        if text is None:
            continue
        for kind, name, line, _ in defs(code[p]["new"], text):
            new_names.add(name)
            if _in(code[p]["plus"], line):
                added_names.add(name)
    removed, logic, sigs = [], [], []
    for p, spec in old_specs.items():
        text = texts.get(spec)
        if text is None:
            continue
        for kind, name, line, local in defs(code[p]["old"], text):
            if (
                _in(code[p]["minus"], line)
                and name not in new_names
                and len(name) >= 3
                and name not in GENERIC
                and not name.startswith("__")
            ):
                removed.append(
                    {
                        "name": name,
                        "kind": kind,
                        "path": code[p]["old"],
                        "line": line,
                        "local": local,
                    }
                )
        if (
            overlaps.language(p) == "py"
            and p in new_specs
            and texts.get(new_specs[p]) is not None
        ):
            before, after = py_logic(text), py_logic(texts[new_specs[p]])
            for q, (
                name,
                line,
                digest,
            ) in after.items():  # the name kept, its signature or body changed
                if q in before and before[q][2] != digest:
                    logic.append({"name": name, "path": code[p]["new"], "line": line})
        elif (
            overlaps.language(p) == "c"
            and p in new_specs
            and texts.get(new_specs[p]) is not None
        ):
            try:
                sigs += changed_signatures(
                    code[p]["new"],
                    text,
                    texts[new_specs[p]],
                    code[p]["minus"],
                    code[p]["plus"],
                )
            except Exception:  # noqa: BLE001 - one odd file must not stop the scan
                pass
    cmake = {
        p: h for p, h in hs.items() if CMAKE_FILE.search(p) and h["old"] and h["minus"]
    }
    cm_specs = [f"{mb}:{h['old']}" for h in cmake.values()] + [
        f"{head}:{h['new']}" for h in cmake.values() if h["new"]
    ]
    cm_texts = gitops.cat_blobs(repo, cm_specs, timeout=600) if cm_specs else {}
    cm_removed: list[dict] = []
    for p, h in sorted(cmake.items()):
        old = cm_texts.get(f"{mb}:{h['old']}")
        if old is None:
            continue
        new = cm_texts.get(f"{head}:{h['new']}") or "" if h["new"] else ""
        cm_removed += cmake_removed(h["old"], old, new, h["minus"])
    strs: dict[str, list] = {}
    uses: dict[str, list] = {}
    test_uses: dict[str, list] = {}
    test_mods: dict[str, list] = {}
    bound: dict[str, list] = {}
    calls: dict[str, list] = {}
    found_refs: list[dict] = []
    others = [
        p
        for p, h in hs.items()
        if h["new"] and h["plus"] and p not in new_specs and not _is_doc(p)
    ]
    texts.update(
        gitops.cat_blobs(repo, [f"{head}:{hs[p]['new']}" for p in others], timeout=600)
        if others
        else {}
    )
    # Code first, so the references cap never cuts an include or an import for a data file's path strings.
    for p in sorted(
        (p for p, h in hs.items() if h["new"] and h["plus"]),
        key=lambda p: (p not in new_specs, p),
    ):
        h = hs[p]
        text = texts.get(new_specs.get(p) or f"{head}:{h['new']}")
        if text is None:
            continue
        if len(found_refs) < MAX_REFS:
            found_refs += [
                dict(r, path=h["new"])
                for r in refs(h["new"], text, h["plus"])[:REFS_PER_FILE]
            ]
        lines = code_lines(p, text)
        if lines is None:
            continue
        test = bool(TESTISH.search(p))
        py = overlaps.language(p) == "py"
        if test and py:
            test_mods[h["new"]] = module_words(text)
        if py:  # names spelled as strings: a CTest name, an attribute patched by name
            for name, ln, how in py_strings(text, h["plus"]):
                got = strs.setdefault(name, [])
                if len(got) < USES_KEPT and [p, ln, how] not in got:
                    got.append([p, ln, how])
        if not py:  # its added lines' calls, with their argument counts (arity_clashes)
            try:
                for name, ln, argc in c_signatures(text)[1]:
                    got = (
                        calls.setdefault(name, [])
                        if _in(h["plus"], ln) and len(name) >= 4
                        else None
                    )
                    if got is not None and len(got) < CALLS_KEPT:
                        got.append([p, ln, argc])
            except Exception:  # noqa: BLE001 - one odd file must not stop the scan
                pass
        used: set[str] = set()
        for ln in _lines_in(h["plus"], len(lines)):
            line, seen = lines[ln - 1], {}
            for m in IDENT.finditer(
                line
            ):  # bare: not after a `.` (an attribute is no use of a bound local)
                if m.group() not in NOT_NAMES and len(m.group()) >= 3:
                    seen[m.group()] = seen.get(m.group(), 0) | (
                        not _after_dot(line, m.start())
                    )
            for name, bare in seen.items():
                used.add(name)
                for d in (uses, test_uses) if test else (uses,):
                    got = d.setdefault(name, [])
                    if (
                        len(got) < USES_KEPT
                        and sum(1 for g in got if g[0] == p) < USES_PER_FILE
                    ):
                        got.append([p, ln, bare])
        if py:
            local = py_bindings(text) & used
            if local:
                bound[p] = sorted(local)
    out = {
        "head": head,
        "mb": mb,
        "paths": sorted(paths),
        "moves": _moves(repo, mb, head),
        "removed": removed,
        "added_names": sorted(added_names),
        "uses": uses,
        "test_uses": test_uses,
        "test_mods": test_mods,
        "bound": bound,
        "refs": found_refs[:MAX_REFS],
        "logic": logic,
        "sigs": sigs,
        "calls": calls,
        "strs": strs,
        "cmake_removed": cm_removed[:CMAKE_KEPT],
    }
    if path:
        _write_json(path, out)
    return out


def _write_json(path: pathlib.Path, data) -> None:
    """A cache file written whole (whole_file) at the mode of the files the monitor shares (worker/perms.py); a cache
    that cannot be written is only a slower next run."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        whole_file.write(
            path,
            json.dumps(data, separators=(",", ":")),
            paths.worker_module("perms").SHARED_FILE_MODE,
        )
    except OSError:
        pass


# --- merges, and what one develop's merges cost once ------------------------------------------------------------------


def merge(repo: str, a: str, b: str, timeout: int = 60) -> dict | None:
    """`git merge-tree --write-tree` of a and b (objects only): the tree, the conflicting paths, and each conflict's
    type and message. None when git could not merge (an object it cannot read, a time-out).
    """
    try:
        p = subprocess.run(
            [
                "git",
                "-C",
                repo,
                "merge-tree",
                "--write-tree",
                "-z",
                "--name-only",
                a,
                b,
            ],
            capture_output=True,
            timeout=timeout,
            env={**os.environ, **NO_LAZY},
        )
    except subprocess.TimeoutExpired:
        return None
    parts = p.stdout.decode("utf-8", "replace").split("\0")
    tree = parts[0].strip()
    if p.returncode not in (0, 1) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", tree
    ):
        return None
    conflicts, kinds, i = [], {}, 1
    if p.returncode == 1:
        while i < len(parts) and parts[i] != "":
            conflicts.append(parts[i])
            i += 1
        i += 1
        while i < len(parts) - 1 and parts[i].strip().isdigit():
            k = int(parts[i])
            ps, typ, msg = (
                parts[i + 1 : i + 1 + k],
                parts[i + 1 + k],
                parts[i + 2 + k].strip(),
            )
            i += 3 + k
            if typ.startswith("CONFLICT"):
                for x in ps:
                    kinds.setdefault(x, [typ, msg])
    return {"tree": tree, "conflicts": conflicts, "kinds": kinds}


def _exists(repo: str, oid: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", repo, "cat-file", "-e", oid],
            capture_output=True,
            env={**os.environ, **NO_LAZY},
        ).returncode
        == 0
    )


class DevCache:
    """One develop commit's landings, pair merges and name lookups, kept in the cache directory so the next review on
    the same develop reads them instead of merging again (one file per develop; a lost write costs only a merge). A
    landing or merge whose objects git has pruned since is made again."""

    def __init__(self, cache: str | None, develop: str):
        self.path = (
            pathlib.Path(cache, f"develop-{develop[:16]}-v{CACHE_VERSION}.json")
            if cache
            else None
        )
        self.data = {"landings": {}, "merges": {}, "names": {}}
        self.dirty = False
        if self.path and self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text()))
            except (OSError, ValueError):
                pass

    def get(self, kind: str, key: str):
        return self.data[kind].get(key)

    def put(self, kind: str, key: str, value) -> None:
        self.data[kind][key] = value
        self.dirty = True

    def save(self) -> None:
        if self.path and self.dirty:
            _write_json(self.path, self.data)


def prune_cache(cache: str | None, days: float = 3.0) -> None:
    """Drop cache files not used for `days` (a develop's merges are stale once develop moves; a head's facts once the
    PR moves on), and a write's temp file an interrupted run left."""
    if not cache or not pathlib.Path(cache).is_dir():
        return
    cutoff = time.time() - days * 86400
    for f in pathlib.Path(cache).glob("*"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def land(
    repo: str,
    develop: str,
    f: dict,
    fetch=None,
    timeout: int = 60,
    dc: DevCache | None = None,
) -> dict:
    """A PR (its facts `f`) landed on develop: a temporary commit of the merge (develop, then the head, as parents), or
    the head itself when it does not merge with develop (`dev_conflicts`). {"error": why} when git could not merge:
    develop's copies of the PR's files are checked first (a merge with a blob that is not local fails only after half
    a minute), and fetched while `fetch()` allows."""
    got = dc.get("landings", f["head"]) if dc else None
    if got and _exists(repo, got["commit"]):
        return got
    want = (
        set(gitops.tree_blobs(repo, develop, f["paths"]).values())
        if f["paths"]
        else set()
    )
    if missing(repo, want):
        if not (fetch and fetch()):
            return {
                "error": "develop's copies of its files are not local (fetch budget spent)"
            }
        fetch_blobs(repo, f["mb"], develop, f["paths"])
        if missing(repo, want):
            return {"error": "develop's copies of its files could not be fetched"}
    m = merge(repo, develop, f["head"], timeout)
    if m is None:
        return {"error": "git could not merge it with develop"}
    if m["conflicts"]:
        out = {"commit": f["head"], "dev_conflicts": m["conflicts"]}
    else:
        p = subprocess.run(
            [
                "git",
                "-C",
                repo,
                "commit-tree",
                m["tree"],
                "-p",
                develop,
                "-p",
                f["head"],
                "-m",
                "rkmon clash scan: a temporary landing (no ref)",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, **NO_LAZY, **TEMP_ENV},
        )
        if p.returncode:
            return {"error": f"git commit-tree failed: {p.stderr.strip()[:120]}"}
        out = {"commit": p.stdout.strip(), "dev_conflicts": []}
    if dc:
        dc.put("landings", f["head"], out)
    return out


def merged(repo: str, a: str, b: str, timeout: int, dc: DevCache | None) -> dict | None:
    """merge(a, b), read from the develop's cache when its tree is still in the clone."""
    key = f"{a}:{b}"
    got = dc.get("merges", key) if dc else None
    if got and _exists(repo, got["tree"]):
        return got
    got = merge(repo, a, b, timeout)
    if got and dc:
        dc.put("merges", key, got)
    return got


def _deleter(msg: str, sides: dict[str, int]) -> int | None:
    """Who deleted the file in a modify/delete or rename/delete message: a value of `sides` (commit id -> PR)."""
    m = re.search(r"deleted in (\S+?)[ .,]", msg)
    return sides.get(m.group(1)) if m else None


# --- a tree: what it still defines, which paths it holds -------------------------------------------------------------


class Tree:
    """One tree's lookups (develop, or a land order's merge), cached: the files that define a name, the paths a
    reference can name. Files in `skip` (the order's conflicting files: their text is not settled) are left out.
    """

    def __init__(
        self, repo: str, tree: str, skip: set[str], dc: DevCache | None = None
    ):
        self.repo, self.tree, self.skip, self.dc = repo, tree, skip, dc
        self._paths: set[str] | None = None
        self._by_base: dict[str, list[str]] = {}
        self._definers: dict[str, dict | None] = {}

    def paths(self) -> set[str]:
        if self._paths is None:
            self._paths = set(
                _git(
                    self.repo,
                    "-c",
                    "core.quotepath=false",
                    "ls-tree",
                    "-r",
                    "--name-only",
                    self.tree,
                ).splitlines()
            )
            for p in self._paths:
                self._by_base.setdefault(posixpath.basename(p), []).append(p)
        return self._paths

    def resolve(self, exact: list[str], tails: list[str]) -> list[str]:
        """The tree's paths a reference can name: an exact path, or one that ends with a tail."""
        self.paths()
        out = [e for e in exact if e in self._paths]
        for t in tails:
            out += [
                p
                for p in self._by_base.get(posixpath.basename(t), [])
                if p == t or p.endswith("/" + t)
            ]
        return sorted(set(out))

    def definers(self, names: set[str], areas: list[str]) -> dict[str, dict | None]:
        """For each name, the code files under `areas` that define it: {"files": [...], "common": False}, or
        {"files": [], "common": True} for a name in more than COMMON_FILES files (defined somewhere; not read further);
        None when the tree could not be read. One `git grep` for every name not looked up yet.
        """
        tag = f"{self.tree}:{','.join(areas)}:"
        for n in names:
            if (
                n not in self._definers
                and self.dc
                and self.dc.get("names", tag + n) is not None
            ):
                self._definers[n] = self.dc.get("names", tag + n)
        todo = sorted(n for n in names if n not in self._definers)
        if todo:
            args = ["grep", "-o", "-w", "-F", "-I"]
            for n in todo:
                args += ["-e", n]
            p = subprocess.run(
                ["git", "-C", self.repo, *args, self.tree, "--", *areas],
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, **NO_LAZY},
            )
            if p.returncode not in (0, 1):
                self._definers.update({n: None for n in todo})
            else:
                files: dict[str, set[str]] = {n: set() for n in todo}
                for line in p.stdout.splitlines():
                    _, _, rest = line.partition(":")
                    f, _, word = rest.rpartition(":")
                    if word in files and overlaps.language(f) in CODE:
                        files[word].add(f)
                want = sorted(
                    {f for fs in files.values() if len(fs) <= COMMON_FILES for f in fs}
                )
                texts = (
                    gitops.cat_blobs(
                        self.repo, [f"{self.tree}:{f}" for f in want], timeout=600
                    )
                    if want
                    else {}
                )
                names_in = {
                    f: {
                        nm
                        for _, nm, _, _ in defs(f, texts.get(f"{self.tree}:{f}") or "")
                    }
                    for f in want
                }
                for n, fs in files.items():
                    got = (
                        {"files": [], "common": True}
                        if len(fs) > COMMON_FILES
                        else {
                            "files": sorted(f for f in fs if n in names_in.get(f, ())),
                            "common": False,
                        }
                    )
                    self._definers[n] = got
                    if self.dc:
                        self.dc.put("names", tag + n, got)
        return {n: self._definers.get(n) for n in names}

    def cmake_files(self, names: set[str]) -> dict[str, set[str] | None]:
        """For each name, the tree's CMake files (CMakeLists.txt, *.cmake) that carry it as a word; None when the tree
        could not be read. One `git grep` for every name not looked up yet."""
        if not hasattr(self, "_cmake"):
            self._cmake: dict[str, set[str] | None] = {}
        todo = sorted(n for n in names if n not in self._cmake)
        if todo:
            args = ["grep", "-o", "-w", "-F", "-I"]
            for n in todo:
                args += ["-e", n]
            p = subprocess.run(
                [
                    "git",
                    "-C",
                    self.repo,
                    *args,
                    self.tree,
                    "--",
                    ":(glob)**/CMakeLists.txt",
                    ":(glob)**/*.cmake",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, **NO_LAZY},
            )
            if p.returncode not in (0, 1):
                self._cmake.update({n: None for n in todo})
            else:
                found: dict[str, set[str]] = {n: set() for n in todo}
                for line in p.stdout.splitlines():
                    _, _, rest = line.partition(":")
                    f, _, word = rest.rpartition(":")
                    if word in found:
                        found[word].add(f)
                self._cmake.update(found)
        return {n: self._cmake.get(n) for n in names}

    def defining(self, names: set[str], areas: list[str]) -> dict[str, int | None]:
        """For each name: how many code files define it, the skipped files aside (COMMON_FILES + 1: a common name);
        None when the tree could not be read."""
        out = {}
        for n, d in self.definers(names, areas).items():
            out[n] = (
                None
                if d is None
                else (
                    COMMON_FILES + 1
                    if d["common"]
                    else sum(1 for f in d["files"] if f not in self.skip)
                )
            )
        return out


def areas_of(paths) -> list[str]:
    """The pathspecs a name is looked up in: the first two parts of each path (such as `team/provider`)."""
    return sorted({"/".join(p.split("/")[:2]) for p in paths if p})


# --- one pair ---------------------------------------------------------------------------------------------------------


def _uses(user: dict, key: str, name: str, home: str | None = None) -> list[list]:
    """`user`'s sites of a name ([path, line]): a bare use in a Python file that binds the name itself (an `as` name,
    a local, a parameter, its own def) is that binding's, not another PR's definition; an attribute use stays. Not in
    `home`, the file the other PR removes the name from: the binding there is the one that goes.
    """
    bound = user.get("bound") or {}
    return [
        [u[0], u[1]]
        for u in user[key].get(name) or []
        if not (len(u) > 2 and u[2] and u[0] != home and name in bound.get(u[0], ()))
    ]


def name_clashes(remover: dict, user: dict) -> list[dict]:
    """Names `remover` removes that `user`'s added lines use (not defined by those lines themselves); a file-local
    definition only for uses in its own file."""
    added, out, seen = set(user["added_names"]), [], set()
    for r in remover["removed"]:
        if r["name"] in added or r["name"] in seen:
            continue
        occ = _uses(user, "uses", r["name"], r["path"])
        if r["local"]:
            occ = [o for o in occ if o[0] == r["path"]]
        if occ:
            seen.add(r["name"])
            out.append(dict(r, uses=occ))
    return out


def path_clashes(remover: dict, user: dict) -> list[dict]:
    """Files `remover` deletes or moves that a reference on `user`'s added lines names: one item per file and
    reference, with every site (`uses`)."""
    moved = {old: new for old, new in remover["moves"]}
    by_base: dict[str, list[str]] = {}
    for old in moved:
        by_base.setdefault(posixpath.basename(old), []).append(old)
    out: dict[tuple, dict] = {}
    for r in user["refs"]:
        for cand in r["exact"] + r["tails"]:
            for old in by_base.get(posixpath.basename(cand), ()):
                if old == cand or (cand in r["tails"] and old.endswith("/" + cand)):
                    x = out.setdefault(
                        (old, r["ref"]),
                        {"old": old, "new": moved[old], "ref": r, "uses": []},
                    )
                    if [r["path"], r["line"]] not in x["uses"]:
                        x["uses"].append([r["path"], r["line"]])
    return list(out.values())


def arity_clashes(changer: dict, user: dict) -> list[dict]:
    """C/C++ functions whose parameter list `changer` changes that `user`'s added lines call with an argument count the
    new list does not take (one PR adds a parameter, another's new code calls the function with the old count: "too
    few arguments" once both land). Not when `user`'s
    added lines declare the name themselves."""
    added, out, seen = set(user["added_names"]), [], set()
    calls = user.get("calls") or {}
    for s in (
        changer.get("sigs") or []
    ):  # one item per name: its first changed declaration (a header's, a source's)
        if s["name"] in added or s["name"] in seen:
            continue
        seen.add(s["name"])
        bad = [
            c
            for c in calls.get(s["name"]) or []
            if not any(lo <= c[2] <= hi for lo, hi in s["new"])
        ]
        if bad:
            out.append(
                dict(
                    s,
                    uses=[[c[0], c[1]] for c in bad],
                    argc=sorted({c[2] for c in bad}),
                )
            )
    return out


def string_clashes(remover: dict, user: dict) -> list[dict]:
    """Names `remover` removes that `user`'s added Python lines spell as a string: an attribute (a Python name it
    removes) that `user` patches or looks up by name, and a CMake name its CMake lines drop (a CTest entry, a target)
    that `user` spells anywhere as a string. Not the names `user`'s added lines define themselves.
    """
    added, strs = set(user["added_names"]), user.get("strs") or {}
    out, seen = [], set()
    for r in remover["removed"]:
        if (
            overlaps.language(r["path"]) != "py"
            or r["name"] in added
            or r["name"] in seen
        ):
            continue
        occ = [[u[0], u[1]] for u in strs.get(r["name"]) or [] if u[2] == "patch"]
        if occ:
            seen.add(r["name"])
            out.append(dict(r, uses=occ, how="patch"))
    for r in remover.get("cmake_removed") or []:
        if r["name"] in added or r["name"] in seen:
            continue
        occ = [[u[0], u[1]] for u in strs.get(r["name"]) or []]
        if occ:
            seen.add(r["name"])
            out.append(dict(r, uses=occ, how="cmake"))
    return out


def logic_leads(changer: dict, user: dict) -> list[dict]:
    """Python functions whose logic `changer` changes that `user`'s added test lines call, in test files that import
    the function's module (or name its file): a same-named call on something else (`tensor.permute`) is no lead.
    """
    added, out, seen = set(user["added_names"]), [], set()
    mods = user.get("test_mods") or {}
    for f in changer["logic"]:
        name, stem = f["name"], module_stem(f["path"])
        if name in added or name in GENERIC or len(name) < 4 or name in seen:
            continue
        occ = [
            u
            for u in _uses(user, "test_uses", name, f["path"])
            if stem in mods.get(u[0], ())
        ]
        if occ:
            seen.add(name)
            out.append(dict(f, uses=occ))
    return out


def candidates(nf: dict, mf: dict) -> dict:
    """A pair's silent-clash candidates both ways, and their shared files: what the merges then confirm."""
    return {
        "shared": sorted(set(nf["paths"]) & set(mf["paths"])),
        "fwd": {
            "names": name_clashes(nf, mf),
            "paths": path_clashes(nf, mf),
            "leads": logic_leads(nf, mf),
            "arity": arity_clashes(nf, mf),
            "strings": string_clashes(nf, mf),
        },
        "rev": {
            "names": name_clashes(mf, nf),
            "paths": path_clashes(mf, nf),
            "leads": logic_leads(mf, nf),
            "arity": arity_clashes(mf, nf),
            "strings": string_clashes(mf, nf),
        },
    }


NAMED = (
    "names",
    "leads",
    "arity",
    "strings",
)  # the candidates that are names looked up in a tree


def _names(c: dict) -> set[str]:
    """The names looked up among a tree's code definitions (a CMake name is looked up in its CMake files)."""
    return {
        x["name"]
        for d in ("fwd", "rev")
        for k in NAMED
        for x in c[d].get(k, [])
        if x.get("how") != "cmake"
    }


def _areas(c: dict) -> list[str]:
    return areas_of(
        [x["path"] for d in ("fwd", "rev") for k in NAMED for x in c[d].get(k, [])]
        + [
            u[0]
            for d in ("fwd", "rev")
            for k in NAMED
            for x in c[d].get(k, [])
            for u in x["uses"]
        ]
    )


def prune(c: dict, changed: set[str], dev: Tree, found: dict[str, dict | None]) -> None:
    """Drop the candidates develop answers without a merge (`found`: develop's definers of the names): a name develop
    defines in a file neither PR changes is still defined once both land (a lead's name then has two definitions), and
    a reference develop resolves to a file neither PR changes still resolves."""

    def outside(name: str) -> bool:
        d = found.get(name)
        return d is not None and (
            d["common"] or any(f not in changed for f in d["files"])
        )

    for d in ("fwd", "rev"):
        c[d]["names"] = [x for x in c[d]["names"] if not outside(x["name"])]
        c[d]["leads"] = [x for x in c[d]["leads"] if not outside(x["name"])]
        c[d]["arity"] = [x for x in c[d].get("arity", []) if not outside(x["name"])]
        cm = {x["name"] for x in c[d].get("strings", []) if x["how"] == "cmake"}
        named = dev.cmake_files(cm) if cm else {}
        c[d]["strings"] = [
            x
            for x in c[d].get("strings", [])
            if not (
                outside(x["name"])
                if x["how"] == "patch"
                else any(f not in changed for f in named.get(x["name"]) or ())
            )
        ]
        c[d]["paths"] = [
            x
            for x in c[d]["paths"]
            if not any(
                h not in changed
                for h in dev.resolve(x["ref"]["exact"], x["ref"]["tails"])
            )
        ]


def _verify(tree: Tree | None, c: dict, areas: list[str]) -> dict:
    """The candidates (one direction) a land order's merged tree confirms: names it defines nowhere, references it
    resolves to no other file, leads whose name it defines once. Arity clashes stand (develop's own definitions of the
    name pruned them already). Without a tree they stand, marked unverified."""
    names, paths, leads, ar = c["names"], c["paths"], c["leads"], c.get("arity", [])
    strs = c.get("strings", [])
    if tree is None:
        for x in names + paths + leads + ar + strs:
            x["unverified"] = True
        return {
            "names": names,
            "paths": paths,
            "leads": leads,
            "arity": ar,
            "strings": strs,
        }
    for x in (
        names + paths + leads + ar + strs
    ):  # a site in a file this order conflicts on is seen when the conflict is resolved
        x["uses"] = sorted(
            ([u[0], u[1], u[0] in tree.skip] for u in x["uses"]),
            key=lambda u: (u[2], u[0], u[1]),
        )
    patched = [x for x in strs if x["how"] == "patch"]
    got = (
        tree.defining({x["name"] for x in names + leads + patched}, areas)
        if names or leads or patched
        else {}
    )
    keep_names = []
    for x in names:
        if got.get(x["name"]) is None:
            x["unverified"] = True
        if not got.get(x["name"]):
            keep_names.append(x)
    keep_leads = [x for x in leads if got.get(x["name"]) in (1, None)]
    keep_paths = []
    for x in paths:
        hits = tree.resolve(x["ref"]["exact"], x["ref"]["tails"])
        if any(h != x["new"] for h in hits):
            continue  # the reference still names another file
        x["moved"] = bool(hits)
        keep_paths.append(x)
    cm = {x["name"] for x in strs if x["how"] == "cmake"}
    named = tree.cmake_files(cm) if cm else {}
    keep_strs = []
    for x in strs:
        if x["how"] == "patch":
            if got.get(x["name"]) is None:
                x["unverified"] = True
            if not got.get(x["name"]):
                keep_strs.append(x)
            continue
        files = named.get(x["name"])
        if files is None:
            x["unverified"] = True
        if not [f for f in files or () if f not in tree.skip]:
            keep_strs.append(x)
    return {
        "names": keep_names,
        "paths": keep_paths,
        "leads": keep_leads,
        "arity": ar,
        "strings": keep_strs,
    }


def pair(
    repo: str,
    n: dict,
    m: dict,
    nf: dict,
    mf: dict,
    c: dict,
    landed: dict,
    timeout: int,
    fetch=None,
    dc: DevCache | None = None,
) -> dict:
    """PR n (the one under review) against PR m (`nf`, `mf`: their facts' head, merge base and paths; `c`: the pair's
    candidates, already pruned on develop): both land orders and the candidates each order's merged tree confirms.
    Merged only when they share a file or a candidate is left. `landed` caches each PR's landing (key: its number;
    "develop": the develop commit)."""
    res = {"n": m["n"], "shared": c["shared"]}
    if not c["shared"] and not any(c["fwd"].values()) and not any(c["rev"].values()):
        res.update(merged=False, fwd=c["fwd"], rev=c["rev"], orders={})
        return res
    res["merged"] = True
    for pr, f in ((n, nf), (m, mf)):
        if pr["n"] not in landed:
            landed[pr["n"]] = land(repo, landed["develop"], f, fetch, timeout, dc)
    orders, why = {}, []
    for first, second, key in ((n, m, "nm"), (m, n, "mn")):
        a, b = landed[first["n"]], landed[second["n"]]
        if a.get("error") or b.get("error"):
            orders[key] = None
            why.append(
                f"#{first['n'] if a.get('error') else second['n']}: {a.get('error') or b.get('error')}"
            )
            continue
        got = merged(repo, a["commit"], second["head"], timeout, dc)
        if got is None:
            orders[key] = None
            why.append("git could not merge the two")
            continue
        own = set(a["dev_conflicts"]) | set(b["dev_conflicts"])
        sides = {
            a["commit"]: first["n"],
            first["head"]: first["n"],
            second["head"]: second["n"],
        }
        kinds = sorted((p, k) for p, k in got["kinds"].items() if p not in own)
        orders[key] = {
            "tree": got["tree"],
            "conflicts": sorted(set(got["conflicts"]) - own),
            "moddel": [
                {"path": p, "deleted_by": _deleter(k[1], sides), "type": k[0]}
                for p, k in kinds
                if "modify/delete" in k[0] or "rename/delete" in k[0]
            ],
            "special": [
                {"path": p, "type": k[0]}
                for p, k in kinds
                if not any(
                    t in k[0]
                    for t in ("contents", "add/add", "modify/delete", "rename/delete")
                )
            ],
            "dev_conflicts": sorted(own),
            "first_stands_in": bool(a["dev_conflicts"]),
            "all_conflicts": got["conflicts"],
        }
    res["why_unmerged"] = sorted(set(why))
    areas = _areas(c)

    def tree_of(key: str) -> Tree | None:
        o = orders.get(key)
        return Tree(repo, o["tree"], set(o["all_conflicts"]), dc) if o else None

    res.update(
        fwd=_verify(tree_of("nm"), c["fwd"], areas),
        rev=_verify(tree_of("mn"), c["rev"], areas),
        orders=orders,
    )
    return res


# --- the scan ---------------------------------------------------------------------------------------------------------


def open_prs(st: dict) -> dict[int, dict]:
    """Every open PR of the state: its head, the base its own change is read against (develop, or the head of the open
    PR it is stacked on), its base branch when that is neither develop nor an open PR's (a feature branch:
    `base_branch`, read against when the mirror has it) and the PRs it is stacked with (never compared).
    """
    recs = {
        int(k): r
        for k, r in (st.get("prs") or {}).items()
        if r.get("status") == "open" and r.get("head")
    }
    by_ref = {r.get("head_ref"): k for k, r in recs.items() if r.get("head_ref")}
    out = {}
    for k, r in recs.items():
        stack, cur, base = set(), r, None
        while cur.get("base", "develop") != "develop" and by_ref.get(
            cur["base"]
        ) not in (None, k, *stack):
            b = by_ref[cur["base"]]
            base = base or recs[b]["head"]
            stack.add(b)
            cur = recs[b]
        branch = (
            r.get("base") if not base and not gitops.is_develop(r.get("base")) else None
        )
        out[k] = {
            "n": k,
            "head": r["head"],
            "base": base,
            "base_branch": branch,
            "draft": bool(r.get("draft")),
            "title": r.get("title") or "",
            "author": r.get("author") or "",
            "stack": stack,
        }
    for k, r in out.items():
        for b in r["stack"]:
            out[b]["stack"] = out[b]["stack"] | {k}
    return out


def own_base(repo: str, pr: dict, develop: str) -> str:
    """What a PR's own change is read against: the head of the open PR it is stacked on, else its base branch when
    that is a feature branch the mirror has (`base_branch`), else develop."""
    if pr.get("base"):
        return pr["base"]
    branch = pr.get("base_branch")
    if (
        branch
        and gitops.git(
            repo,
            "rev-parse",
            "--verify",
            "-q",
            gitops.base_ref(branch),
            check=False,
            lazy=False,
        ).strip()
    ):
        return gitops.base_ref(branch)
    return develop


def scan(
    repo: str,
    n: int,
    prs: dict[int, dict],
    develop: str,
    budget_s: float | None = None,
    max_fetch: int | None = None,
    cache: str | None = None,
    merge_timeout_s: int | None = None,
    order=None,
) -> dict:
    """PR n against every other PR of `prs` (open_prs' records) on `develop` (a commit id), in three steps: each PR's
    facts and the pair's candidates; one look on develop for the names and references it already answers; the merges
    and the merged trees' answers. Returns each pair's result and what was not run, with why. A limit not given
    (budget_s, max_fetch, merge_timeout_s) is the package's: config.defaults.json's clashes.
    """
    limits = registry.default("clashes")
    budget_s = limits["budget_s"] if budget_s is None else budget_s
    max_fetch = limits["max_fetch"] if max_fetch is None else max_fetch
    if merge_timeout_s is None:
        merge_timeout_s = limits["merge_timeout_s"]
    start = time.monotonic()
    fetches = {"left": max_fetch}

    def fetch() -> bool:
        if fetches["left"] <= 0:
            return False
        fetches["left"] -= 1
        return True

    me = prs[n]
    res = {
        "n": n,
        "head": me["head"],
        "develop": develop,
        "pairs": [],
        "not_run": [],
        "skipped_stack": [],
        "secs": 0.0,
        "fetched": 0,
    }
    prune_cache(cache)
    try:
        nf = facts(repo, me["head"], own_base(repo, me, develop), fetch, cache)
    except (NotRun, *GIT_ERRORS) as exc:
        res["error"] = (
            str(exc)
            if isinstance(exc, NotRun)
            else f"could not be read ({str(exc)[:160]})"
        )
        res["secs"] = round(time.monotonic() - start, 1)
        return res
    res["mb"] = nf["mb"]
    res["tests"] = [p for p in nf["paths"] if TEST_FILE.search(p)]
    keep = ("head", "mb", "paths")
    todo = []
    for k in [
        k
        for k in (order or sorted(prs, key=lambda k: (prs[k]["draft"], -k)))
        if k != n and k in prs
    ]:
        m = prs[k]
        if k in me["stack"]:
            res["skipped_stack"].append(k)
        elif time.monotonic() - start >= budget_s:
            res["not_run"].append((k, "the time budget ran out"))
        else:
            try:
                mf = facts(repo, m["head"], own_base(repo, m, develop), fetch, cache)
                todo.append((m, {x: mf[x] for x in keep}, candidates(nf, mf)))
            except NotRun as exc:
                res["not_run"].append((k, str(exc)))
            except GIT_ERRORS as exc:
                res["not_run"].append((k, f"could not be read ({str(exc)[:120]})"))
    dc = DevCache(cache, develop)
    dev = Tree(repo, develop, set(), dc)
    names = set().union(*(_names(c) for _, _, c in todo)) if todo else set()
    found = (
        dev.definers(names, areas_of([a for _, _, c in todo for a in _areas(c)]))
        if names
        else {}
    )
    nf = {x: nf[x] for x in keep}
    landed: dict = {"develop": develop}
    for m, mf, c in todo:
        if time.monotonic() - start >= budget_s:
            res["not_run"].append((m["n"], "the time budget ran out before its merges"))
            continue
        prune(c, set(nf["paths"]) | set(mf["paths"]), dev, found)
        try:
            res["pairs"].append(
                dict(
                    pair(repo, me, m, nf, mf, c, landed, merge_timeout_s, fetch, dc),
                    draft=m["draft"],
                    title=m["title"],
                    author=m["author"],
                    head=m["head"],
                    tests=[x for x in mf["paths"] if TEST_FILE.search(x)],
                )
            )
        except GIT_ERRORS as exc:
            res["not_run"].append((m["n"], f"could not be merged ({str(exc)[:120]})"))
    dc.save()
    res["dev_conflicts"] = len((landed.get(n) or {}).get("dev_conflicts") or [])
    res["fetched"] = max_fetch - fetches["left"]
    res["secs"] = round(time.monotonic() - start, 1)
    return res


# --- the report -------------------------------------------------------------------------------------------------------


def _where(path: str, line: int | None, K: str) -> str:
    return f"{short(path, K)}:{line}" if line else short(path, K)


def _some(items: list[str], most: int = SHOWN) -> str:
    return ", ".join(items[:most]) + (
        f" (+{len(items) - most} more)" if len(items) > most else ""
    )


def findings(p: dict) -> dict:
    """A pair's counts: silent clashes (names and paths, both ways), modify/delete pairs, conflicting files, leads, and
    whether a land order could not be merged."""
    orders = [o for o in (p.get("orders") or {}).values() if o]
    return {
        "silent": sum(
            len(p[d]["names"])
            + len(p[d]["paths"])
            + len(p[d].get("arity", []))
            + len(p[d].get("strings", []))
            for d in ("fwd", "rev")
        ),
        "moddel": len({x["path"] for o in orders for x in o["moddel"]}),
        "conflicts": len({c for o in orders for c in o["conflicts"]}),
        "leads": len(p["fwd"]["leads"]) + len(p["rev"]["leads"]),
        "unmerged": bool(
            p.get("merged") and any(v is None for v in (p.get("orders") or {}).values())
        ),
    }


def _sites(uses: list, K: str) -> str:
    """Where a name or file is used, with its preposition: sites in files that merge clean first (the silent ones); a
    site in a file the land order conflicts on is seen when the conflict is resolved."""
    clean = [_where(u[0], u[1], K) for u in uses if not (len(u) > 2 and u[2])]
    conf = [_where(u[0], u[1], K) for u in uses if len(u) > 2 and u[2]]
    out = f"at {_some(clean)} (merges clean)" if clean else ""
    if conf:
        out += f"{'; also' if out else 'only'} in conflicting files: {_some(conf, 2)}"
    return out


def _name_line(x: dict, remover: str, user: str, K: str) -> str:
    tag = (
        " (not checked for another definition: the merged tree was not read)"
        if x.get("unverified")
        else ""
    )
    return (
        f"  silent: `{x['name']}` ({x['kind']}): {remover} removes it ({_where(x['path'], x['line'], K)}); "
        f"{user} uses it {_sites(x['uses'], K)}{tag}"
    )


def _path_line(x: dict, remover: str, user: str, K: str) -> str:
    what = f"moves it to {short(x['new'], K)}" if x["new"] else "deletes it"
    how = {"include": "includes", "import": "imports"}.get(x["ref"]["kind"], "names")
    tail = (
        (
            " — the reference also matches the new path, so it breaks unless the include or import path follows the "
            "move"
        )
        if x.get("moved")
        else ""
    )
    return (
        f"  silent: {short(x['old'], K)}: {remover} {what}; {user} {how} `{x['ref']['ref']}` "
        f"{_sites(x['uses'], K)}{tail}"
    )


def _arity_line(x: dict, changer: str, user: str, K: str) -> str:
    def ar(a) -> str:
        lo, hi = a
        return str(lo) if lo == hi else f"{lo}+" if hi >= VARIADIC else f"{lo}-{hi}"

    tag = (
        " (not checked for another definition: the merged tree was not read)"
        if x.get("unverified")
        else ""
    )
    n = ", ".join(map(str, x["argc"]))
    return (
        f"  silent (arity, a lead from a parse of the signatures: build the merge): `{x['name']}` "
        f"({_where(x['path'], x['line'], K)}): {changer} changes its parameters ({' or '.join(map(ar, x['old']))} "
        f"-> {' or '.join(map(ar, x['new']))}); {user} calls it with {n} argument{'s' if n != '1' else ''} "
        f"{_sites(x['uses'], K)}{tag}"
    )


def _string_line(x: dict, remover: str, user: str, K: str) -> str:
    tag = (
        " (not checked for another definition: the merged tree was not read)"
        if x.get("unverified")
        else ""
    )
    if x["how"] == "cmake":
        what = f"{remover} drops it from its CMake lines ({_where(x['path'], x['line'], K)}); {user} spells it"
        after = "no CMake file of the merge names it (no such test or target: the run finds nothing)"
    else:
        what = (
            f"{remover} removes it ({x['kind']}, {_where(x['path'], x['line'], K)}); {user} patches or looks it up "
            "by name"
        )
        after = "the merge defines it nowhere (an AttributeError at run time)"
    return f"  silent (a string): `{x['name']}`: {what} {_sites(x['uses'], K)}; {after}{tag}"


def _lead_line(x: dict, changer: str, user: str, K: str) -> str:
    return (
        f"  lead: `{x['name']}` ({_where(x['path'], x['line'], K)}): {changer} changes its logic; {user}'s new "
        f"test lines call it {_sites(x['uses'], K)}: run them on the merge"
    )


def render_pair(p: dict, n: int, K: str, order_note: str = "") -> list[str]:
    me, other = "this PR", f"#{p['n']}"
    title = p.get("title", "")[:70]
    out = [
        f"#{p['n']} {p.get('head', '')[:10]}{' (draft)' if p.get('draft') else ''} \"{title}\""
        f"{' (' + p['author'] + ')' if p.get('author') else ''}:"
    ]
    for x in p["fwd"]["names"]:
        out.append(_name_line(x, me, other, K))
    for x in p["rev"]["names"]:
        out.append(_name_line(x, other, me, K))
    for x in p["fwd"]["paths"]:
        out.append(_path_line(x, me, other, K))
    for x in p["rev"]["paths"]:
        out.append(_path_line(x, other, me, K))
    for x in p["fwd"].get("arity", []):
        out.append(_arity_line(x, me, other, K))
    for x in p["rev"].get("arity", []):
        out.append(_arity_line(x, other, me, K))
    for x in p["fwd"].get("strings", []):
        out.append(_string_line(x, me, other, K))
    for x in p["rev"].get("strings", []):
        out.append(_string_line(x, other, me, K))
    orders = p.get("orders") or {}
    nm, mn = orders.get("nm"), orders.get("mn")
    who = {n: me, p["n"]: other}
    seen = set()
    for o in (nm, mn):
        for x in (o or {}).get("moddel", []):
            if x["path"] in seen:
                continue
            seen.add(x["path"])
            by = who.get(x["deleted_by"], "one side")
            keeper = (
                other
                if x["deleted_by"] == n
                else me if x["deleted_by"] == p["n"] else "the other"
            )
            out.append(
                f"  modify/delete: {short(x['path'], K)}: {by} deletes or moves it, {keeper} changes it"
            )
    for x in {
        x["path"]: x for o in (nm, mn) for x in (o or {}).get("special", [])
    }.values():
        out.append(f"  {x['type']}: {short(x['path'], K)}")
    if (
        nm is not None
        and mn is not None
        and nm["conflicts"] == mn["conflicts"]
        and nm["conflicts"]
    ):
        out.append(
            f"  conflicts, either order ({len(nm['conflicts'])}): "
            f"{_some([short(c, K) for c in nm['conflicts']], 6)}"
        )
    else:
        for o, label in ((nm, "this PR first"), (mn, f"#{p['n']} first")):
            if o and o["conflicts"]:
                out.append(
                    f"  conflicts, {label} ({len(o['conflicts'])}): "
                    f"{_some([short(c, K) for c in o['conflicts']], 6)}"
                )
    if mn and mn["first_stands_in"]:
        out.append(
            f"  (#{p['n']} does not merge with develop: its head stands in for its landing)"
        )
    if p.get("merged") and (nm is None or mn is None):
        out.append(
            f"  NOT RUN: {'one land order' if (nm or mn) else 'both land orders'} could not be merged "
            f"({'; '.join(p.get('why_unmerged') or ['git could not merge'])}): conflicts and the merged tree "
            f"behind the silent clashes are not checked there"
        )
    for x in p["fwd"]["leads"]:
        out.append(_lead_line(x, me, other, K))
    for x in p["rev"]["leads"]:
        out.append(_lead_line(x, other, me, K))
    note = land_note(p, n)
    if note:
        out.append(f"  land order: {note}{'; ' + order_note if order_note else ''}")
    return out


def land_note(p: dict, n: int) -> str:
    """What the second to land ports or resolves (the silent clashes break either order; so do conflicts)."""
    f = findings(p)
    if not (f["silent"] or f["moddel"] or f["conflicts"] or f["leads"]):
        return ""
    names = sorted({x["name"] for d in ("fwd", "rev") for x in p[d]["names"]})
    paths = sorted(
        {posixpath.basename(x["old"]) for d in ("fwd", "rev") for x in p[d]["paths"]}
    )
    calls = sorted({x["name"] for d in ("fwd", "rev") for x in p[d].get("arity", [])})
    leads = sorted({x["name"] for d in ("fwd", "rev") for x in p[d]["leads"]})
    strs = sorted({x["name"] for d in ("fwd", "rev") for x in p[d].get("strings", [])})
    work = []
    if names:
        work.append(f"ports the uses of {_some([f'`{x}`' for x in names])}")
    if paths:
        work.append(f"moves the references to {_some(paths)}")
    if calls:
        work.append(
            f"ports the calls of {_some([f'`{x}`' for x in calls])} to the new parameters"
        )
    if strs:
        work.append(
            f"ports the string uses of {_some([f'`{x}`' for x in strs])} (a test name, a patched attribute)"
        )
    if f["moddel"]:
        work.append(
            f"carries {f['moddel']} modify/delete file{'s' if f['moddel'] != 1 else ''} to the new home (or "
            f"drops the change)"
        )
    if f["conflicts"]:
        work.append(
            f"resolves {f['conflicts']} conflicting file{'s' if f['conflicts'] != 1 else ''}"
        )
    if leads:
        work.append(
            f"re-runs the new tests against the changed {_some([f'`{x}`' for x in leads])}"
        )
    return (
        f"whichever lands second {', '.join(work)} (this PR first: #{p['n']} does; #{p['n']} first: this PR "
        f"does)"
    )


def abi_lane(cfg: dict, st: dict, n: int, prs: dict, dev: str) -> dict:
    """The ABI lane of the clash scan: the abi.h counters PR n bumps (rkmon_lib/abi.py) against every other open PR's
    claim on the same counter. Two claims on one counter both set it; whichever lands second renumbers to the next
    number in every home (abi.h's define and history line, abi.py, the layout canaries), even when git merges the two
    clean (lessons L-110). One stack, or PRs carrying the identical bump hunk, are one claim. A PR's claim is the
    tick's (`abi_cache`) while its head is the one compared, else scanned now.
    {"develop": {lane: N}, "mine": {lane: claim}, "clashes": [{"lane", "n", "to", "mine"}], "not_run": [n, ...]}
    """
    repo, K = cfg["paths"]["repo"], paths.repository_folder(cfg)
    recs = st.get("prs") or {}

    def claim(k: int) -> dict:
        p = prs[k]
        c = (recs.get(str(k)) or {}).get("abi_cache") or {}
        if c.get("head") == p["head"] and c.get("v") == abi.CACHE_VERSION:
            return c
        mb = _git(repo, "merge-base", p.get("base") or dev, p["head"]).strip()
        return abi.scan(repo, K, mb, p["head"], dev)

    out = {"develop": {}, "mine": {}, "clashes": [], "not_run": []}
    try:
        d = abi.parse_header(abi._show(repo, dev, f"{K}/{abi.ABI_H}"))
        mine = claim(n).get("lanes") or {}
    except GIT_ERRORS:
        out["not_run"].append(n)
        return out
    out["develop"] = {lane: d.get(lane) for lane in abi.DEFINES}
    out["mine"] = mine
    if not mine:
        return out
    stack = set(prs[n].get("stack") or ())
    for k in sorted(prs):
        if k == n or k in stack or n in set(prs[k].get("stack") or ()):
            continue
        try:
            theirs = claim(k).get("lanes") or {}
        except GIT_ERRORS:
            out["not_run"].append(k)
            continue
        for lane, cl in mine.items():
            o = theirs.get(lane)
            if o and not (o.get("hunk") and o.get("hunk") == cl.get("hunk")):
                out["clashes"].append(
                    {"lane": lane, "n": k, "to": o["to"], "mine": cl["to"]}
                )
    return out


def abi_lines(a: dict | None) -> list[str]:
    """The ABI lane's lines: per counter PR N bumps, develop's number, its claim and each other claim on it."""
    if not a or not a.get("mine"):
        return []
    out = []
    for lane, cl in a["mine"].items():
        title = abi.LANE_TITLE.get(lane, lane)
        dv = a["develop"].get(lane)
        others = [c for c in a["clashes"] if c["lane"] == lane]
        if not others:
            out.append(
                f"ABI lane {title}: develop {dv}; this PR claims {cl['to']}; no other open PR claims this counter"
            )
            continue
        nxt = (dv + 2) if isinstance(dv, int) else "the next number"
        out.append(
            f"ABI NUMBER CLASH {title}: develop {dv}; this PR claims {cl['to']}; "
            + ", ".join(f"#{c['n']} claims {c['to']}" for c in others)
            + f": whichever lands second renumbers to {nxt} in every home (abi.h define and history line, "
            f"abi.py {abi.MIRRORS[lane]}"
            + (", the layout canaries" if lane == "binary" else "")
            + ")"
        )
    if a.get("not_run"):
        out.append(
            "ABI lane NOT RUN for: " + " ".join(f"#{k}" for k in a["not_run"][:12])
        )
    return out


def verdict(res: dict) -> tuple[str, str]:
    """(pass|fail|skip, detail) for the status line: fail on any silent clash, modify/delete pair or ABI number clash,
    pass when the scan ran and found none (textual conflicts and leads are named), skip when it ran nothing.
    """
    v, d = _verdict(res)
    hits = (res.get("abi") or {}).get("clashes") or []
    if hits and not res.get("error"):
        ks = sorted({c["n"] for c in hits})
        return "fail", (
            f"ABI number clash with {', '.join(f'#{k}' for k in ks[:4])}"
            + ("; " + d if v == "fail" else "")
        )
    return v, d


def _verdict(res: dict) -> tuple[str, str]:
    if res.get("error"):
        return "skip", f"not run: #{res['n']} {res['error']}"
    pairs = res["pairs"]
    if not pairs:
        return (
            "skip",
            f"no other open PR could be compared ({len(res['not_run'])} NOT RUN)",
        )
    bad = [(p, findings(p)) for p in pairs]
    hard = [(p, f) for p, f in bad if f["silent"] or f["moddel"]]
    soft = [
        (p, f)
        for p, f in bad
        if not (f["silent"] or f["moddel"]) and (f["conflicts"] or f["leads"])
    ]
    tail = f"; {len(res['not_run'])} NOT RUN" if res["not_run"] else ""
    if hard:
        return "fail", (
            f"{len(hard)} PR{'s' if len(hard) != 1 else ''} clash: "
            + ", ".join(
                f"#{p['n']} ({f['silent']} silent, {f['moddel']} modify/delete, {f['conflicts']} conflicts)"
                for p, f in hard[:4]
            )
            + tail
        )
    if soft:
        return "pass", (
            f"no silent clash in {len(pairs)} PRs; conflicts or leads with "
            + ", ".join(f"#{p['n']}" for p, _ in soft[:6])
            + tail
        )
    return "pass", f"no clash with {len(pairs)} open PRs{tail}"


def render(
    res: dict, K: str, orders: dict[int, str] | None = None, limit: int = 56
) -> list[str]:
    """The checks.md section's body: the scan's line, each PR with findings (silent clashes first), what was not run,
    and the PRs that merge clean. `orders` gives each PR's land-order note from the merge plan.
    """
    n = res["n"]
    if res.get("error"):
        return [
            f"NOT RUN: #{n} at {res['head'][:10]}: {res['error']}",
            "CLASH skip " + verdict(res)[1],
        ]
    pairs = res["pairs"]
    scored = [(findings(p), p) for p in pairs]
    hit = sorted(
        [(f, p) for f, p in scored if any(f.values())],
        key=lambda t: (
            -t[0]["silent"],
            -t[0]["moddel"],
            -t[0]["conflicts"],
            -t[0]["leads"],
            t[1]["n"],
        ),
    )
    shown = {p["n"] for _, p in hit}
    clean = sorted(p["n"] for p in pairs if p["n"] not in shown)
    out = [
        f"#{n} at {res['head'][:10]} against {len(pairs)} open PRs on develop {res['develop'][:10]} "
        f"(merge base {res.get('mb', '')[:10]}): {sum(1 for f, _ in hit if f['silent'] or f['moddel'])} with a "
        f"silent clash or modify/delete, {sum(1 for f, _ in hit if not (f['silent'] or f['moddel']))} with "
        f"conflicts or leads only, {len(clean)} clean, {len(res['not_run'])} NOT RUN "
        f"({res['secs']} s{', ' + str(res['fetched']) + ' blob fetches' if res.get('fetched') else ''})"
    ]
    if res.get("dev_conflicts"):
        out.append(
            f"(this PR does not merge with develop ({res['dev_conflicts']} files): its head stands in for its "
            f"landing; each pair leaves out either PR's own develop conflicts)"
        )
    body = abi_lines(res.get("abi"))
    for f, p in hit:
        body += render_pair(p, n, K, (orders or {}).get(p["n"], ""))
    if len(body) > limit:
        cut = len(body) - limit
        body = body[:limit] + [f"(… {cut} more lines: `rk clash {n}` prints them all)"]
    out += body
    if res["not_run"]:
        out.append(
            f"NOT RUN ({len(res['not_run'])}): "
            + "; ".join(f"#{k} ({why})" for k, why in res["not_run"][:12])
            + (
                f"; +{len(res['not_run']) - 12} more"
                if len(res["not_run"]) > 12
                else ""
            )
        )
    if res["skipped_stack"]:
        out.append(
            "stacked with this PR (not compared): "
            + " ".join(f"#{k}" for k in sorted(res["skipped_stack"]))
        )
    if clean:
        out.append(
            f"clean ({len(clean)}: no shared file, or both orders merge clean; no silent clash): "
            + " ".join(f"#{k}" for k in clean)
        )
    v, d = verdict(res)
    out.append(f"CLASH {v} {d}")
    return out


def plan_orders(st: dict, n: int, ks: list[int]) -> dict[int, str]:
    """Each PR's expected land order against n (overlaps.land_order: the merge plan's, else readiness), when the queue
    knows the first of the two."""
    out = {}
    for k in ks:
        try:
            order = overlaps.land_order(st, [n, k])
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if order and order[0][1] and order[0][1] != "not in the queue":
            first = order[0][0]
            out[k] = (
                f"expected: {'this PR' if first == n else '#' + str(first)} first ({order[0][1]}), so "
                f"{'#' + str(k) if first == n else 'this PR'} does it"
            )
    return out


def run(
    cfg: dict,
    st: dict,
    n: int,
    heads: dict[int, str] | None = None,
    develop: str | None = None,
    others: list[int] | None = None,
    cache: str | None = None,
    budget_s: float | None = None,
    max_fetch: int | None = None,
) -> tuple[dict, list[str]]:
    """The scan for PR n with the state's open PRs (`heads` overrides heads, e.g. a snapshot's; `others` limits the
    PRs compared; `develop` a commit id or ref), and its rendered section."""
    c = settings(cfg)
    repo, K = cfg["paths"]["repo"], paths.repository_folder(cfg)
    prs = open_prs(st)
    for k, h in (heads or {}).items():
        rec = (st.get("prs") or {}).get(str(k)) or {}
        prs.setdefault(
            k,
            {
                "n": k,
                "head": h,
                "base": None,
                "draft": bool(rec.get("draft")),
                "title": rec.get("title") or "",
                "author": rec.get("author") or "",
                "stack": set(),
            },
        )
        prs[k]["head"] = h
    if n not in prs:
        res = {
            "n": n,
            "head": "",
            "error": "is not an open PR of the state (give its head with --heads N=SHA)",
            "pairs": [],
            "not_run": [],
            "secs": 0,
        }
        return res, render(res, K)
    if others is not None:
        prs = {k: v for k, v in prs.items() if k == n or k in set(others)}
    dev = _git(
        repo, "rev-parse", "--verify", (develop or DEVELOP) + "^{commit}"
    ).strip()
    res = scan(
        repo,
        n,
        prs,
        dev,
        budget_s if budget_s is not None else c["budget_s"],
        max_fetch if max_fetch is not None else c["max_fetch"],
        cache,
        c["merge_timeout_s"],
    )
    res["abi"] = abi_lane(cfg, st, n, prs, dev)
    res["at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res["scanned"] = sorted(k for k in prs if k != n)
    hit = [
        p["n"]
        for p in res["pairs"]
        if any(v for k, v in findings(p).items() if k != "unmerged")
    ]
    return res, render(res, K, plan_orders(st, n, hit))


# --- open PRs the checks' scan did not compare -----------------------------------------------------------------------


def record_path(cfg: dict, n: int) -> pathlib.Path:
    return pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}" / "checks" / RECORD


def unscanned(cfg: dict, st: dict, n: int) -> tuple[str | None, list[int]]:
    """(when the checks' scan ran, the open PRs it did not compare): PRs opened or reopened since, its stack aside;
    (None, []) when the checks wrote no record."""
    try:
        rec = json.loads(record_path(cfg, n).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, paths.PathsError):
        return None, []
    if not isinstance(rec, dict) or "scanned" not in rec:
        return None, []
    prs = open_prs(st)
    seen = set(rec.get("scanned") or ())
    stack = set((prs.get(n) or {}).get("stack") or ())
    return rec.get("at") or "?", sorted(
        k for k in prs if k != n and k not in seen and k not in stack
    )


# --- the pair merges built ------------------------------------------------------------------------------------------


def compile_db(cfg: dict, n: int) -> tuple[dict[str, list[str]], str]:
    """The review's shared build's compile commands ({repo path: argv}) and the tree folder they were built from
    (`<workers>/pr<N>/checks/shared_build`: the build folder, its tree and the tree's commit, one to a line); ({}, "")
    without them, or once the build's slot is no longer kept for it (slots.kept_build). The core build's
    `compile_commands.json` is what PR code's build wrote: it is read as JSON by the result reader
    (worker/pr_results.py read_json: a regular file, no link on the way, a size cap, no JSON refused) and taken as
    data (an entry of another shape is left out).
    """
    pr_results = paths.worker_module("pr_results")
    from . import review  # which imports this module

    try:
        line = review.shared_build_record(cfg, n)
        bd, tree = line[0], line[1].rstrip("/")
        if not slots.kept_build(bd):
            return {}, ""
        root, rel = slots.split(bd)
        db = pr_results.read_json(
            root, posixpath.join(rel, "core", "compile_commands.json")
        )
    except (OSError, IndexError, KeyError, paths.PathsError):
        return {}, ""
    out = {}
    for e in db if isinstance(db, list) else []:
        if not isinstance(e, dict):
            continue
        try:
            f = os.path.normpath(os.path.join(e.get("directory", ""), e["file"]))
            args = e.get("arguments") or shlex.split(e.get("command", ""))
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            continue
        if f.startswith(tree + "/") and args:
            out[f[len(tree) + 1 :]] = args
    return out, tree


def compile_argv(
    rel: str, db: dict, tree: str, scratch: str, K: str, header: bool = False
) -> tuple[list[str], str]:
    """The `-fsyntax-only` command for `rel` in the scratch tree, and where its flags came from: its own compile
    command, one of its folder's, or the tree's include folders (c++ -std=c++20)."""
    args = db.get(rel)
    how = "its compile command"
    if args is None:
        d = posixpath.dirname(rel)
        args = next(
            (a for r, a in sorted(db.items()) if posixpath.dirname(r) == d), None
        )
        how = "its folder's flags"
    flags: list[str] = []
    cc = "c++"
    if args is None:
        how = "the tree's include folders"
        for root, dirs, _ in os.walk(os.path.join(scratch, K)):
            dirs.sort()
            if os.path.basename(root) == "include":
                flags.append(f"-I{root}")
        flags.insert(0, "-std=c++20")
    else:
        cc, skip = args[0], False
        for a in args[1:]:
            if skip:
                skip = False
                continue
            if a in ("-o", "-MF", "-MT", "-MQ"):
                skip = True
                continue
            if a in ("-c", "-MD", "-MMD") or (
                not a.startswith("-") and C_SOURCE.search(a)
            ):
                continue
            flags.append(a.replace(tree + "/", scratch + "/") if tree else a)
    lang = ["-x", "c++-header"] if header else []
    return [cc, "-fsyntax-only", *flags, *lang, os.path.join(scratch, rel)], how


def _first_error(text: str, scratch: str) -> str:
    for line in text.splitlines():
        if "error" in line:
            return line.replace(scratch + "/", "")[:200]
    return (text.strip().splitlines() or ["(no output)"])[-1].replace(
        scratch + "/", ""
    )[:200]


def extract(repo: str, tree: str, K: str, scratch: str) -> None:
    """The merged tree's rocKE folder into `scratch` (readable by the PR runners; nothing is written there later)."""
    a = subprocess.Popen(
        ["git", "-C", repo, "archive", "--format=tar", tree, "--", K],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    t = subprocess.run(
        ["tar", "-x", "-C", scratch], stdin=a.stdout, capture_output=True
    )
    a.stdout.close()
    if a.wait(timeout=600) or t.returncode:
        raise gitops.GitError(f"git archive {tree[:10]} -- {K} could not be extracted")


def build_pair(
    cfg: dict, n: int, p: dict, mine: list[str], run=None, python: str | None = None
) -> tuple[list[str], int]:
    """Pair `p`'s merge with this PR first: its shared C/C++ files compiled and both PRs' changed tests collected, in a
    PR runner's slot (`run`: a slot's run; by default one prcode.Slot for the pair). (lines, failures); no lines when
    there is nothing to build."""
    o = (p.get("orders") or {}).get("nm")
    conf = set((o or {}).get("all_conflicts") or [])
    shared = [f for f in p.get("shared") or [] if f not in conf]
    tus = [f for f in shared if C_SOURCE.search(f)]
    hdrs = [f for f in shared if C_HEADER.search(f)]
    tests = sorted({f for f in mine + (p.get("tests") or []) if f not in conf})
    if not (tus or hdrs or tests):
        return [], 0
    head = f"#{p['n']} (this PR first):"
    if not o:
        return [f"{head} NOT RUN: that merge could not be made"], 0
    repo, K = cfg["paths"]["repo"], paths.repository_folder(cfg)
    files = (tus + hdrs)[:BUILD_MOST]
    left = len(tus) + len(hdrs) - len(files)
    base = None
    try:
        base = paths.get(cfg, "tmp")
        os.makedirs(base, exist_ok=True)
    except (KeyError, OSError, paths.PathsError):
        base = None
    scratch = tempfile.mkdtemp(prefix=f"clash-build-{n}-{p['n']}-", dir=base)
    try:
        # The merged tree: the monitor's, which the runners read.
        why = prcode.tree(scratch)
        if why:
            return [f"{head} NOT RUN: the scratch folder was not made ({why})"], 0
        extract(repo, o["tree"], K, scratch)
        db, tree = compile_db(cfg, n)
        with contextlib.ExitStack() as held:
            if run is None:
                run = held.enter_context(prcode.Slot(tmp=base)).run
            return _build_pair(
                cfg, run, scratch, db, tree, files, hdrs, tests, left, python, head
            )
    except prcode.NoSlot as exc:
        why = (exc.lines or ["no slot could be taken"])[-1]
        return [f"{head} NOT RUN: {why[:200]}"], 0
    except (gitops.GitError, OSError, subprocess.SubprocessError) as exc:
        return [f"{head} NOT RUN: {str(exc)[:200]}"], 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _build_pair(
    cfg: dict,
    run,
    scratch: str,
    db,
    tree,
    files: list[str],
    hdrs: list[str],
    tests: list[str],
    left: int,
    python: str | None,
    head: str,
) -> tuple[list[str], int]:
    """build_pair's compiles and collect, each through `run` (a slot's run): (lines, failures)."""
    from . import lanes

    K = paths.repository_folder(cfg)
    out, fails = [], 0
    for rel in files:
        argv, how = compile_argv(rel, db, tree, scratch, K, header=rel in hdrs)
        r = run(argv, cwd=scratch, timeout=300)
        if r.returncode:
            fails += 1
            out.append(
                f"  ERROR {short(rel, K)} ({how}): {_first_error(r.stderr + r.stdout, scratch)}"
            )
        else:
            out.append(f"  ok {short(rel, K)} ({how})")
    if left:
        out.append(
            f"  ({left} more C/C++ files both change, not compiled: the cap is {BUILD_MOST})"
        )
    there = [t for t in tests if os.path.isfile(os.path.join(scratch, t))][:TESTS_MOST]
    if there:
        py = python or lanes._venv_python(cfg)
        env = prcode_env(scratch, K)
        r = run(
            [
                py,
                "-B",
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                *there,
            ],
            cwd=scratch,
            env=env,
            timeout=600,
        )
        said = (r.stdout + r.stderr).replace(scratch + "/", "")
        last = next(
            (
                x
                for x in reversed(said.splitlines())
                if "collected" in x or "error" in x.lower()
            ),
            "",
        )
        if r.returncode not in (0, 5):
            fails += 1
            out.append(
                f"  ERROR collect {len(there)} test file{'s' * (len(there) != 1)}: "
                f"{(last or _first_error(said, scratch))[:200]}"
            )
        else:
            out.append(
                f"  ok collect {len(there)} test file{'s' * (len(there) != 1)}: {last[:120]}"
            )
    n_c = len(files)
    out.insert(
        0,
        f"{head} {n_c} C/C++ file{'s' * (n_c != 1)} compiled, {len(there)} test file"
        f"{'s' * (len(there) != 1)} collected: {'FAIL' if fails else 'OK'}",
    )
    return out, fails


def prcode_env(scratch: str, K: str) -> dict:
    """The collect's environment: this process's (PR code keeps only what worker/pr_env.py lists of it), the merged
    tree's packages first, no bytecode written."""
    env = dict(os.environ)
    root = os.path.join(scratch, K)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(root, "platform", "python"), os.path.join(root, "library")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def build(cfg: dict, res: dict, run=None, python: str | None = None) -> list[str]:
    """`rk clash N --build`: each compared PR's merge with this PR first, built and collected (build_pair); the last
    line `BUILD pass|fail|skip <detail>`."""
    out = [
        "",
        "Pair merges built (this PR first; C/C++ files both change, both PRs' changed tests):",
    ]
    if res.get("error"):
        return out + [f"BUILD skip not run: {res['error']}"]
    ran = fails = 0
    for p in res.get("pairs") or []:
        lines, f = build_pair(cfg, res["n"], p, res.get("tests") or [], run, python)
        if lines:
            ran += 1
            fails += f
            out += lines
    if not ran:
        return out + [
            "BUILD skip no other open PR shares a C/C++ file, and no changed test file"
        ]
    return out + [
        f"BUILD {'fail' if fails else 'pass'} {ran} pair merge{'s' * (ran != 1)}, {fails} failure"
        f"{'s' * (fails != 1)}"
    ]
