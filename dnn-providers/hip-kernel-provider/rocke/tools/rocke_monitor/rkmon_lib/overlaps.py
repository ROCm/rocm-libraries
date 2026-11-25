# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Overlapping work across open PRs, shown in every review's checks: work two open PRs both add is not duplicated, one
of them is the authoritative source of the overlapping code, and the PRs coordinate through comments on them.

**What a PR adds.** The definitions whose name line its head adds against `merge-base(develop, head)` (a stacked PR:
against its base PR's head), from `git diff -U0 --no-renames`, then the head blob of each changed file, read whole, so a
body ends at its closing brace or its dedent: Python functions, classes, methods and module-level UPPER constants; C/C++
functions (definitions and declarations), macros, struct/class/enum/union types and namespace-scope constants; CMake
`add_test(NAME …)`. Each keeps its file, line, kind, name, signature line and a body normalized for comparison:
comments, docstrings and whitespace dropped, identifiers kept, parameters renamed by position (`rocke_ew_streq(a, b)`
and `streq(a, b)` compare equal), at most `MAX_BODY_LINES` lines, as a bottom-k sketch of its token shingles.

**Develop.** The same extractor over develop's code files under the areas the open PRs touch (the first three path
parts, e.g. `team/provider/rocke`), cached per blob, so a develop move re-reads only the files it
changed. A name develop defines (same language and kind) is not new work; one it defines in the same file is an edit.

**Groups** (leads; the reviewers decide, `rocke_code_quality_reviewer.md` §8):
- *same new thing*: the same name and kind added by 2+ open PRs, not on develop, when `same_thing` holds: similar
  bodies, the same file, one global name space (CTest, header macros), or a visible function with a descriptive name.
  The rest is a coincidence (`_cfg`, `_div`, `_fake`; `ARMS`, `Plan`, `family`);
- *same logic, different names*: bodies of `MIN_LOGIC_LINES`+ lines, `SIM_LOGIC` similar or more, in 2+ open PRs, not
  all test-local;
- *another copy of a develop pattern*: a body that matches a develop body with `MIN_DEVELOP_COPIES`+ exact copies, or
  that is `SIM_SAME` similar to develop's definitions of its own name (e.g. several `findTensor`, or `rocke_*_streq`
  in three variants). A lead for the code quality pass's "search before you add" (at most `LEADS_PER_PR` a
  PR); such a copy joins no cross-PR group, since its shared home is develop's pattern, not another PR.
Copies inside one PR are the code quality pass's to find, not this module's.

**Land order** (`land_order`): the merge plan's order for the PRs it places, else develop fixes and the user's
priorities first, then readiness (ready, candidate, the rest), fewer failing gates, the older PR. Its first PR is the
proposed owner, unless the reviewers pick the one that puts the code in its natural shared home. A posted decision is
recorded (`rk overlap-decide`, `state/overlaps_decided.json`), and every later review of a member shows it, also after
the owner has landed.

**Cost.** `state/overlap_cache.json.gz` holds every open head's definitions and develop's index (about 6 MB).
A tick reads at most `max_heads` new heads in `budget_s` seconds (0.1-5 s a head), of which at most
`max_fetches` may fetch blobs that are not local yet (`git diff` fetches them in one request), re-reads develop's
changed files when it moves, and recomputes the groups (about 2 s); a tick with nothing new costs nothing. No GitHub API
call; a failure is reported and never fails the tick.
"""

from __future__ import annotations

import base64
import binascii
import bisect
import contextlib
import gzip
import hashlib
import json
import os
import re
import struct
import subprocess
import time

from . import gitops, idle, paths, registry, state_store, whole_file
from .conflicts import short
from .paths import repository_folder
from .store import minutes_since, now

CACHE_FILE = "overlap_cache.json.gz"
DECIDED_FILE = "overlaps_decided.json"
CACHE_VERSION = 1  # bump when the extractor or the sketch changes, so every head and blob is read again
SECTION = "Overlapping work in other open PRs (leads; the reviewers decide)"
DEVELOP = "refs/remotes/origin/develop"
DEFAULTS = registry.package_section("overlaps")
GIT_ERRORS = (gitops.GitError, OSError, subprocess.SubprocessError, ValueError)

MAX_BODY_LINES = 80
SHINGLE = 3  # tokens per shingle
SKETCH = 48  # the smallest shingle hashes kept per body (bottom-k): exact Jaccard below that many shingles
SIM_SAME = (
    0.6  # a same-name pair this similar is the same thing, whatever its visibility
)
MIN_SIM_TOKENS = 10  # a shorter body (a constant's value, a one-call wrapper) proves sameness only by being identical
SIM_LOGIC = 0.85  # same logic under different names
MIN_LOGIC_LINES = 6
SIM_DEVELOP = 0.85  # a copy of a develop body
MIN_DEVELOP_COPIES = 2
MIN_DEVELOP_TOKENS = 16  # shorter bodies (getters, `return x;`) match everywhere
LEADS_PER_PR = 5  # the leads shown a PR: first those other open PRs also copy, then the biggest patterns
FAMILY_MAX = 12  # a name family on develop larger than this (`*_spec`, `*_name`) is too generic to call a pattern
MIN_KEY = (
    4  # the shortest name part a family is looked up by (`streq` in `rocke_ew_streq`)
)
POSTING_CAP = 300  # a sketch hash more bodies share than this is too common to look for candidates with
RETRY_MIN = 30  # a head that could not be read (not fetched, blobs not local) is tried again after this many minutes
DEPTH = 3  # develop's index covers the first DEPTH path parts of every file an open PR changes

PY = (".py",)
CPP = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".inl",
    ".cu",
    ".hip",
)
HEADER = (".h", ".hh", ".hpp", ".hxx", ".inc", ".inl")
CODE_SPECS = ["*.py", *(f"*{e}" for e in CPP), "*.cmake", "*CMakeLists.txt"]
SHARED = (
    "platform/cpp/include/",
    "platform/cpp/support/",
    "platform/python/rocke/",
    "library/kernels/common/",
)
SHARED_PACKS = re.compile(r"/src/engines/.+/packs/[^/]+\.hpp$")
TEST_PATH = re.compile(
    r"(^|/)(tests?|testing|test_sdk|unittests?)/|(^|/)(test_[^/]*|conftest|[^/]*_test)\.py$|"
    r"(^|/)(Test[A-Z][^/]*|[^/]*_test|test_[^/]*)\.(c|cc|cpp|cxx|h|hpp)$"
)
STOP = {
    "main",
    "__init__",
    "__call__",
    "__repr__",
    "__str__",
    "__eq__",
    "__ne__",
    "__hash__",
    "__post_init__",
    "__enter__",
    "__exit__",
    "__iter__",
    "__next__",
    "__len__",
    "__getitem__",
    "__setitem__",
    "__contains__",
    "__lt__",
    "__bool__",
    "setUp",
    "tearDown",
    "setUpClass",
    "tearDownClass",
    "run",
    "build",
    "_build",
    "helper",
    "_helper",
    "make",
    "_make",
    "wrapper",
    "inner",
    "fn",
    "cb",
    "impl",
    "_impl",
    "parse_args",
    "_parse_args",
    "cli",
    "_main",
    "pytest_configure",
    "pytest_collection_modifyitems",
    "pytest_addoption",
    "pytest_generate_tests",
    "operator",
}


# --- tokens, bodies, similarity ---------------------------------------------------------------------------------------

TOKEN = re.compile(
    r"[A-Za-z_]\w*|\d[\w.']*|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|::|->|<<=?|>>=?|[-+*/%&|^!=<>]="
    r"|&&|\|\||\+\+|--|\*\*|//|\S"
)
_TOKEN_HASH: dict[str, int] = {}


def tokens(text: str, params: list[str] | None = None) -> list[str]:
    """Code text (comments already dropped) as tokens; each parameter becomes `$1`, `$2`… by position."""
    toks = TOKEN.findall(text)
    if params:
        ren = {p: f"${i}" for i, p in enumerate(params, 1)}
        toks = [ren.get(t, t) for t in toks]
    return toks


def _token_hash(t: str) -> int:
    h = _TOKEN_HASH.get(t)
    if h is None:
        h = _TOKEN_HASH[t] = binascii.crc32(t.encode("utf-8", "replace"))
    return h


def sketch(toks: list[str]) -> list[int]:
    """The SKETCH smallest hashes of the body's token shingles (bottom-k), stable across runs: two sketches estimate
    the Jaccard similarity of the two shingle sets."""
    hs = [_token_hash(t) for t in toks]
    if len(hs) < SHINGLE:
        return [binascii.crc32(" ".join(map(str, hs)).encode())] if hs else []
    sh = {
        ((a * 0x9E3779B1 + b) * 0x85EBCA77 + c) & 0xFFFFFFFF
        for a, b, c in zip(hs, hs[1:], hs[2:])
    }
    return sorted(sh)[:SKETCH]


def similarity(a: list[int], b: list[int]) -> float:
    """Jaccard of two bodies estimated from their sketches; exact while the two have at most SKETCH shingles
    together."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union = sorted(sa | sb)[:SKETCH]
    return sum(1 for x in union if x in sa and x in sb) / len(union)


def body_hash(toks: list[str]) -> str:
    return hashlib.sha1("\x1f".join(toks).encode("utf-8", "replace")).hexdigest()[:12]


def _definition(
    kind: str,
    name: str,
    line: int,
    sig: str,
    body: list[str],
    n_lines: int,
    private: bool,
    decl: bool = False,
) -> dict:
    return {
        "kind": kind,
        "name": name,
        "line": line,
        "sig": " ".join(sig.split())[:160],
        "lines": n_lines,
        "toks": len(body),
        "h": body_hash(body),
        "sk": sketch(body),
        "private": private,
        "decl": decl,
    }


_NOT_NL = re.compile(r"[^\n]")
_QUOTE = re.compile(r"[rRbBuUfF]{0,2}(\"\"\"|'''|\"|')")


def _spaces(s: str) -> str:
    return _NOT_NL.sub(" ", s) if "\n" in s else " " * len(s)


def _blank(m: re.Match) -> str:
    """A comment as spaces, or a literal with its inside as spaces (prefix and quotes kept); newlines stay, so offsets
    and line numbers do not move."""
    s = m.group()
    if s[0] == "#" or s.startswith(("//", "/*")):
        return _spaces(s)
    q = _QUOTE.match(s)
    if not q:
        return s
    lead, close = q.end(), len(q.group(1))
    return s[:lead] + _spaces(s[lead : len(s) - close]) + s[len(s) - close :]


def _uncomment(m: re.Match) -> str:
    s = m.group()
    return _spaces(s) if s[0] == "#" or s.startswith(("//", "/*")) else s


def _split_top(text: str, sep: str) -> list[str]:
    """`text` split at `sep` outside brackets."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p for p in parts if p.strip()]


# --- Python -----------------------------------------------------------------------------------------------------------

PY_LIT = re.compile(
    r"(?:[rRbBuUfF]{1,2})?(?:\"\"\"(?:\\.|[^\\])*?\"\"\"|'''(?:\\.|[^\\])*?'''|\"(?:\\.|[^\"\\\n])*\""
    r"|'(?:\\.|[^'\\\n])*')|#[^\n]*",
    re.S,
)
PY_DEF = re.compile(r"([ \t]*)(async[ \t]+)?(def|class)[ \t]+([A-Za-z_]\w*)\b")
PY_CONST = re.compile(r"([A-Z][A-Z0-9_]{3,})[ \t]*(?::[^=\n]*)?=(?!=)")
_BRACKET = {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}


def _py_params(sig: str) -> list[str]:
    """Parameter names from a `def name(…)` signature (self and cls left out)."""
    i = sig.find("(")
    if i < 0:
        return []
    depth, j = 0, i
    for j in range(i, len(sig)):
        depth += _BRACKET.get(sig[j], 0)
        if depth == 0:
            break
    out = []
    for part in _split_top(sig[i + 1 : j], ","):
        m = re.match(r"\s*\**\s*([A-Za-z_]\w*)", part)
        if m and m.group(1) not in ("self", "cls"):
            out.append(m.group(1))
    return out


def python_defs(text: str) -> list[dict]:
    """Top-level functions, methods (`Class.method`), classes and module-level UPPER constants. A function nested in a
    function is part of its body, not a definition of its own."""
    masked, plain = PY_LIT.sub(_blank, text), PY_LIT.sub(_uncomment, text)
    mlines, plines = masked.split("\n"), plain.split("\n")
    n = len(mlines)
    indent, depth, joined = [], 0, False
    for (
        ln
    ) in (
        mlines
    ):  # a line inside brackets, or after a `\`, continues a statement: no indentation of its own
        s = ln.lstrip(" \t")
        indent.append(len(ln) - len(s) if s and not depth and not joined else -1)
        depth = max(
            0,
            depth
            + ln.count("(")
            + ln.count("[")
            + ln.count("{")
            - ln.count(")")
            - ln.count("]")
            - ln.count("}"),
        )
        joined = ln.rstrip().endswith("\\")

    def header_end(i: int) -> int:
        """The line on which the bracketed statement that starts on line i closes."""
        depth = 0
        for k in range(i, min(n, i + 200)):
            for ch in mlines[k]:
                depth += _BRACKET.get(ch, 0)
            if depth <= 0 and not mlines[k].rstrip().endswith("\\"):
                return k
        return i

    def block_end(i: int, ind: int) -> int:
        last = i
        for k in range(i + 1, n):
            if indent[k] == -1:
                continue
            if indent[k] <= ind:
                break
            last = k
        return last

    out, stack = [], []  # stack: (indent, "def" | "class", name)
    for i in range(n):
        ind = indent[i]
        if ind == -1:
            continue
        while stack and stack[-1][0] >= ind:
            stack.pop()
        m = PY_DEF.match(mlines[i])
        if m:
            is_def, name = m.group(3) == "def", m.group(4)
            if any(kind == "def" for _, kind, _ in stack):
                stack.append((ind, "def", name))
                continue  # nested in a function: part of its body
            qual = ".".join([s[2] for s in stack] + [name])
            h_end = header_end(i)
            b_end = block_end(h_end, ind)
            sig = " ".join(plines[k].strip() for k in range(i, h_end + 1))
            first = h_end + 1
            while first <= b_end and indent[first] == -1:
                first += 1
            q = _QUOTE.match(mlines[first].lstrip()) if first <= b_end else None
            if q:  # the docstring is documentation, not logic
                rest = mlines[first].lstrip()[q.end() :]
                if q.group(1) not in rest:
                    while first < b_end and q.group(1) not in mlines[first + 1]:
                        first += 1
                    first += 1
                first += 1
            body = "\n".join(plines[first : min(b_end, h_end + MAX_BODY_LINES) + 1])
            params = _py_params(sig) if is_def else []
            head = tokens(sig.split("(", 1)[1] if "(" in sig else "", params)
            kind = ("method" if stack else "def") if is_def else "class"
            out.append(
                _definition(
                    kind,
                    qual,
                    i + 1,
                    sig,
                    head + tokens(body, params),
                    b_end - i + 1,
                    name.startswith("_"),
                )
            )
            stack.append((ind, "def" if is_def else "class", name))
            continue
        if ind == 0:
            c = PY_CONST.match(mlines[i])
            if c:
                v_end = header_end(i)
                value = "\n".join(plines[i : min(v_end, i + MAX_BODY_LINES) + 1]).split(
                    "=", 1
                )[-1]
                out.append(
                    _definition(
                        "const",
                        c.group(1),
                        i + 1,
                        plines[i],
                        tokens(value),
                        v_end - i + 1,
                        False,
                    )
                )
    return out


# --- C and C++ --------------------------------------------------------------------------------------------------------

C_LIT = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S
)
C_PP = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", re.M)
C_DEFINE = re.compile(r"[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\()?")
C_GUARD = re.compile(r"_*[A-Z0-9_]*_(H|HH|HPP|HXX|INC|INCLUDED|GUARD)_*")
C_STRUCT = re.compile(r"[{};()]")
C_BRACE = re.compile(r"[{}]")
C_ATTR = re.compile(r"\[\[.*?\]\]")
C_NAMESPACE = re.compile(
    r"(?:inline\s+)?namespace\b\s*([\w:]*)\s*(?:\[\[.*?\]\]\s*)?$", re.S
)
C_EXTERN = re.compile(
    r"extern\s+\"\s*\""
)  # extern "C" (masked: the literal's inside is blank)
C_TYPE = re.compile(
    r"(?:template\s*<.*?>\s*)?(typedef\s+)?(struct|class|union|enum(?:\s+(?:class|struct))?)\b(.*)$",
    re.S,
)
C_EXPORT = re.compile(r"[A-Z0-9_]+_(API|EXPORT|PUBLIC)|ROCKE_[A-Z0-9_]+|final|alignas")
C_NAME = re.compile(r"((?:[A-Za-z_]\w*\s*::\s*)*~?[A-Za-z_]\w*)\s*$")
C_WORD = re.compile(r"[A-Za-z_]\w*|[*&>]")
C_ASSIGN = re.compile(r"(?<![=!<>])=(?!=)")
C_SPEC = {
    "static",
    "inline",
    "constexpr",
    "consteval",
    "extern",
    "virtual",
    "explicit",
    "friend",
    "__device__",
    "__host__",
    "__global__",
    "__forceinline__",
    "__inline__",
    "noexcept",
    "const",
    "volatile",
    "mutable",
    "thread_local",
    "register",
    "typename",
    "unsigned",
    "signed",
    "struct",
    "class",
    "enum",
    "union",
    "auto",
}
C_NOT_NAME = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "alignof",
    "decltype",
    "static_assert",
    "defined",
    "else",
    "do",
    "case",
    "new",
    "delete",
    "throw",
    "operator",
    "typeid",
    "co_return",
    "alignas",
    "__attribute__",
    "__declspec",
    "requires",
    "assert",
}
# What may follow a declaration's parameter list. Every token is atomic, read one way only, so a long UPPER name,
# `&& &&`, `[[a]] [[b]]` or `-> A -> B` before a character no token reads fails at once (every split of them was tried
# before: an hour on one long macro name). A lookahead that captures the token, then the capture, reads it so: Python
# reads an atomic group only from 3.11, and docker/rkmon loads this module on the host's Python, an older one. The `->`
# return type stops before a macro call: `-> T MACRO(x)`.
C_AFTER_DECL = re.compile(
    r"(?:\s*(?=(const|noexcept|override|final|&&|&|=\s*(?:0|default|delete)|\[\[.*?\]\]"
    r"|[A-Z_][A-Z0-9_]*(?:\([^)]*\))?|->(?:(?![A-Z_][A-Z0-9_]*\()[\w:<>,\s*&])*))\1)*\s*"
)
C_ACCESS = re.compile(r"(?:\s*(?:public|private|protected)\s*:(?!:))*\s*")
C_TAIL = re.compile(r"\s*(?:[A-Za-z_]\w*\s*)?;")
CONST_NAME = re.compile(r"k[A-Z]\w*|[A-Z][A-Z0-9_]{3,}")


def _c_masks(text: str) -> tuple[str, str]:
    """(masked, plain): masked has comments and literal insides blank and preprocessor lines blanked, for the
    structure; plain has only the comments blank, for signatures and bodies. Offsets and line numbers are the text's.
    """
    blanked = C_LIT.sub(_blank, text)
    return C_PP.sub(lambda m: _spaces(m.group()), blanked), C_LIT.sub(_uncomment, text)


def _closing(text: str, i: int, end: int) -> int:
    """The index of the `}` that closes the `{` at i; end - 1 when it never closes (an #if branch may leave one)."""
    depth = 0
    for m in C_BRACE.finditer(text, i, end):
        depth += 1 if m.group() == "{" else -1
        if depth == 0:
            return m.start()
    return end - 1


def _paren_end(text: str, i: int) -> int:
    """The index of the `)` that closes the `(` at i (len(text) when it never closes)."""
    depth = 0
    for j in range(i, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            return j
    return len(text)


def _c_params(params: str) -> list[str] | None:
    """Parameter names of a C/C++ parameter list; None when it is no parameter list (a call's arguments)."""
    out = []
    for p in _split_top(params, ","):
        p = p.split("=", 1)[0].strip()
        if not p or p in ("void", "..."):
            continue
        if not re.match(r"[A-Za-z_:]", p):
            return None  # a literal: `Foo g(1, 2);` constructs an object, it declares no function
        names = re.findall(r"[A-Za-z_]\w*", re.sub(r"\[[^\]]*\]", "", p))
        if len(names) >= 2 and names[-1] not in C_SPEC:
            out.append(names[-1])
    return out


def _c_head(h: str) -> tuple[str, int, int, int] | None:
    """A function header `ret name(params) quals` in masked text: (name, start and end of the name, index of its `(`),
    or None when it is not one (no return type, a control statement, an initialization).
    """
    if "[[" in h:
        h = C_ATTR.sub(lambda m: " " * len(m.group()), h)
    depth = 0
    for k, ch in enumerate(h):  # the first `(` outside template brackets
        if ch == "<":
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif ch in "={};" and depth == 0:
            return None
        elif ch == "(" and depth == 0:
            break
    else:
        return None
    m = C_NAME.search(h, 0, k)
    if not m:
        return None
    name = re.sub(r"\s+", "", m.group(1))
    base = name.rsplit("::", 1)[-1].lstrip("~")
    if not base or base in C_NOT_NAME or base in C_SPEC:
        return None
    before = C_WORD.findall(h, 0, m.start())
    if not before and "::" not in name:
        return None  # no return type: a macro call (`TEST(Suite, Name) {`), or a constructor (the caller's case)
    if before and before[-1] in (
        "return",
        "else",
        "new",
        "delete",
        "throw",
        "case",
        "goto",
    ):
        return None
    return name, m.start(1), m.end(1), k


def _ctor(h: str, cls: str) -> tuple[str, int, int, int] | None:
    """A constructor or destructor inside class `cls` (it has no return type)."""
    m = re.match(
        r"\s*(?:explicit\s+|constexpr\s+|inline\s+)*(~?"
        + re.escape(cls.rsplit("::", 1)[-1])
        + r")\s*\(",
        h,
    )
    return (m.group(1), m.start(1), m.end(1), m.end() - 1) if m else None


class _CFile:
    """One C/C++ file's definitions: macros from the preprocessor lines, then the statements at namespace and class
    scope, split at `;` and at balanced `{…}` blocks outside parentheses."""

    def __init__(self, text: str):
        self.masked, self.plain = _c_masks(text)
        self.starts = [0] + [m.end() for m in re.finditer("\n", text)]
        self.out: list[dict] = []
        self.macros(text)
        self.scan(0, len(self.masked), None, False)

    def line(self, i: int) -> int:
        return bisect.bisect_right(self.starts, i)

    def macros(self, text: str) -> None:
        for m in C_PP.finditer(C_LIT.sub(_blank, text)):
            d = C_DEFINE.match(m.group())
            if not d or C_GUARD.fullmatch(d.group(1)):
                continue
            name, raw = d.group(1), self.plain[m.start() : m.end()]
            value, params = raw[raw.find(name) + len(name) :].replace("\\\n", " "), []
            if d.group(2):
                close = value.find(")")
                params = [p.strip() for p in value[1:close].split(",") if p.strip()]
                value = value[close + 1 :]
            self.out.append(
                _definition(
                    "fmacro" if d.group(2) else "macro",
                    name,
                    self.line(m.start()),
                    raw.split("\n", 1)[0],
                    tokens(value, params),
                    raw.count("\n") + 1,
                    False,
                )
            )

    def scan(self, a: int, b: int, cls: str | None, private: bool) -> None:
        i = stmt = a
        paren = 0
        while True:
            m = C_STRUCT.search(self.masked, i, b)
            if not m:
                return
            c, i = m.group(), m.end()
            if c in "()":
                paren = paren + 1 if c == "(" else max(0, paren - 1)
                continue
            if paren:
                continue
            if c == "}":
                stmt = i
                continue
            s0 = C_ACCESS.match(self.masked, stmt, m.start()).end()
            h = self.masked[s0 : m.start()].rstrip()
            if c == ";":
                if h:
                    self.statement(h, s0, m.start(), cls, private)
                stmt = i
                continue
            close = _closing(self.masked, m.start(), b)
            if h:
                self.block(h, s0, m.start(), close, cls, private)
            i = stmt = close + 1
            tail = C_TAIL.match(self.masked, i, min(b, i + 200))  # `};` or `} name_t;`
            if tail:
                i = stmt = tail.end()

    def block(
        self, h: str, s0: int, open_i: int, close: int, cls: str | None, private: bool
    ) -> None:
        ns = C_NAMESPACE.match(h)
        if ns:
            self.scan(open_i + 1, close, cls, private or not ns.group(1))
            return
        if C_EXTERN.fullmatch(h):
            self.scan(open_i + 1, close, cls, private)
            return
        t = C_TYPE.match(h)
        if t and "(" not in t.group(3).split(":", 1)[0]:
            words = [
                w
                for w in re.findall(r"[A-Za-z_]\w*", t.group(3).split(":", 1)[0])
                if not C_EXPORT.fullmatch(w)
            ]
            after = re.match(
                r"\s*([A-Za-z_]\w*)\s*;", self.masked[close + 1 : close + 200]
            )
            name = (
                after.group(1) if t.group(1) and after else words[-1] if words else None
            )
            if not name:
                return
            qual = f"{cls}::{name}" if cls else name
            kind = "enum" if t.group(2).startswith("enum") else "type"
            body = self.plain[open_i + 1 : close]
            self.out.append(
                _definition(
                    kind,
                    qual,
                    self.line(s0),
                    self.plain[s0:open_i] + "{",
                    tokens("\n".join(body.split("\n")[:MAX_BODY_LINES])),
                    body.count("\n") + 1,
                    private,
                )
            )
            if kind == "type":
                self.scan(open_i + 1, close, qual, private)
            return
        if h.endswith("="):  # an initializer: `constexpr int kTable[] = {…};`
            self.constant(h[:-1], s0, self.plain[open_i : close + 1], private)
            return
        f = _c_head(h) or (_ctor(h, cls) if cls else None)
        if not f:
            return
        p1 = _paren_end(h, f[3])
        params = _c_params(h[f[3] + 1 : p1])
        if params is None or C_ASSIGN.search(h[p1 + 1 :].split("->", 1)[0]):
            return
        body = self.plain[open_i : close + 1]
        self.function(
            f,
            h,
            s0,
            self.plain[s0:open_i],
            tokens("\n".join(body.split("\n")[:MAX_BODY_LINES]), params),
            body.count("\n") + 1,
            params,
            cls,
            private,
            False,
        )

    def statement(
        self, h: str, s0: int, end: int, cls: str | None, private: bool
    ) -> None:
        if h.startswith(
            ("using ", "typedef ", "friend ", "return ", "template<>", "template <>")
        ):
            return
        if C_ASSIGN.search(
            (re.sub(r"operator\s*\S+", "", h) if "operator" in h else h).split("(", 1)[
                0
            ]
        ):
            self.constant(h, s0, None, private)
            return
        f = _c_head(h)
        if not f:
            return
        p1 = _paren_end(h, f[3])
        params = _c_params(h[f[3] + 1 : p1])
        if params is None or not C_AFTER_DECL.fullmatch(h[p1 + 1 :]):
            return
        self.function(
            f, h, s0, self.plain[s0:end] + ";", [], 1, params, cls, private, True
        )

    def function(
        self,
        f: tuple,
        h: str,
        s0: int,
        sig: str,
        body: list[str],
        n_lines: int,
        params: list[str],
        cls: str | None,
        private: bool,
        decl: bool,
    ) -> None:
        name, n0, n1, _ = f
        qual = f"{cls}::{name}" if cls and "::" not in name else name
        static = not cls and bool(re.search(r"\bstatic\b", h[:n0]))
        head = tokens(
            sig[:n0] + sig[n1:], params
        )  # the types: a renamed copy still compares by them
        d = _definition(
            "cfunc",
            qual,
            self.line(s0 + n0),
            sig,
            head + body,
            n_lines,
            private or static,
            decl=decl,
        )
        if cls and "::" not in name:
            d["member"] = True  # defined in its class's body: as visible as the class
        self.out.append(d)

    def constant(self, h: str, s0: int, init: str | None, private: bool) -> None:
        left = re.sub(r"\[[^\]]*\]", "", h.split("=", 1)[0])
        words = re.findall(r"[A-Za-z_]\w*", left)
        if (
            not words
            or "(" in left
            or not re.search(r"\b(const|constexpr)\b", left)
            or not CONST_NAME.fullmatch(words[-1])
        ):
            return
        text = self.plain[s0 : s0 + len(h)]
        value = init if init is not None else text.split("=", 1)[1]
        self.out.append(
            _definition(
                "cconst",
                words[-1],
                self.line(s0 + left.rfind(words[-1])),
                text.split("\n")[0],
                tokens("\n".join(value.split("\n")[:MAX_BODY_LINES])),
                value.count("\n") + 1,
                private or bool(re.search(r"\bstatic\b", left)),
            )
        )


def cpp_defs(text: str) -> list[dict]:
    """Functions (definitions and declarations; `Class::method` inside a class), macros, struct/class/enum/union types
    and namespace-scope constants (`k…` or UPPER names, const or constexpr)."""
    return _CFile(text).out


# --- CMake ------------------------------------------------------------------------------------------------------------

CM_TEST = re.compile(r"\b(add_test|[A-Za-z_]*_add_test|add_[A-Za-z_]*_test)\s*\(", re.I)
CM_COMMENT = re.compile(r"#[^\n]*")


def cmake_defs(text: str) -> list[dict]:
    """CTest registrations: `add_test(NAME x …)`, `add_test(x cmd …)` and project wrappers named like them."""
    plain = CM_COMMENT.sub(lambda m: " " * len(m.group()), text)
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    out = []
    for m in CM_TEST.finditer(plain):
        j = _paren_end(plain, m.end() - 1)
        args = plain[m.end() : j].split()
        if not args:
            continue
        name = args[args.index("NAME") + 1] if "NAME" in args[:-1] else args[0]
        if not re.fullmatch(r"[\w.+-]+", name) or name in ("NAME", "COMMAND"):
            continue  # a variable (`${name}`): not a name to compare
        out.append(
            _definition(
                "ctest",
                name,
                bisect.bisect_right(starts, m.start()),
                plain[m.start() : j + 1].split("\n")[0],
                tokens(" ".join(args)),
                plain[m.start() : j].count("\n") + 1,
                False,
            )
        )
    return out


# --- per file ---------------------------------------------------------------------------------------------------------

FAMILY = {
    "def": "func",
    "method": "func",
    "class": "type",
    "const": "const",
    "cfunc": "func",
    "fmacro": "const",
    "macro": "const",
    "type": "type",
    "enum": "type",
    "cconst": "const",
    "ctest": "test",
}
KIND_WORD = {
    "def": "function",
    "method": "method",
    "class": "class",
    "const": "constant",
    "cfunc": "function",
    "fmacro": "macro",
    "macro": "macro",
    "type": "type",
    "enum": "enum",
    "cconst": "constant",
    "ctest": "ctest",
}
PLURAL = {"class": "classes"}


def language(path: str) -> str | None:
    if path.endswith(PY):
        return "py"
    if path.endswith(CPP):
        return "c"
    if path.endswith(".cmake") or path.rsplit("/", 1)[-1] == "CMakeLists.txt":
        return "cmake"
    return None


def family(path: str, kind: str) -> str:
    """The name space a definition lives in, language and kind: a Python and a C++ `streq` are different things."""
    return f"{language(path)}.{FAMILY[kind]}"


def file_defs(path: str, text: str) -> list[dict]:
    """Every definition in one file. Private: a Python name starting with `_`; a static C/C++ function, or one in an
    anonymous namespace; a macro, type, constant or class member of a source file (no other file sees it). Test-local:
    in a test file. A file the parsers cannot read contributes nothing: one odd file must not stop every tick.
    """
    lang = language(path)
    try:
        defs = (
            python_defs(text)
            if lang == "py"
            else (
                cpp_defs(text)
                if lang == "c"
                else cmake_defs(text) if lang == "cmake" else []
            )
        )
    except Exception:  # noqa: BLE001 - see the docstring
        return []
    if lang == "c" and not path.endswith(HEADER):
        for d in defs:
            if d["kind"] in ("macro", "fmacro", "type", "enum", "cconst") or d.get(
                "member"
            ):
                d["private"] = True
    test = bool(TEST_PATH.search(path))
    for d in defs:
        d["test"] = test
    return defs


def shared_layer(path: str) -> bool:
    """A file in a layer several features share: the public C++ headers and support code, the rocke Python package (not
    its tests), library/kernels/common, a provider engine's pack headers."""
    p = "/" + path
    return (any(f"/{s}" in p for s in SHARED) and not TEST_PATH.search(path)) or bool(
        SHARED_PACKS.search(p)
    )


GENERATED = re.compile(r"/include/rocke/(instance_|helper_)[^/]*\.h$")


def layer_tag(path: str) -> str:
    """How a definition's file reads to the owner rule: a generated header, or a shared layer (a natural home)."""
    if GENERATED.search(path):
        return " [generated header]"
    return " [shared layer]" if shared_layer(path) else ""


def words(name: str) -> int:
    """How many words a name has (`sha1_hex` 2, `findTensor` 2, `family` 1): one word is a generic name."""
    return len(
        re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", re.split(r"\.|::", name)[-1])
    )


def name_keys(name: str) -> list[str]:
    """The keys a definition's name family is looked up by, most specific first: the name and each `_` suffix of
    MIN_KEY characters or more (`rocke_ew_streq`: itself, `ew_streq`, `streq`)."""
    parts = re.split(r"\.|::", name)[-1].split("_")
    keys = ["_".join(parts[i:]) for i in range(len(parts))]
    return [k for k in keys if len(k) >= MIN_KEY]


def skipped(name: str) -> bool:
    """Names left out: test functions and classes (`test_*`, `Test*`), the stoplist's generic names, and constructors
    and destructors (`Builder::~Builder`: the class's own group says it)."""
    parts = re.split(r"\.|::", name)
    return (
        parts[-1] in STOP
        or any(p.startswith("test_") or re.match(r"Test[A-Z_]", p) for p in parts)
        or (len(parts) > 1 and parts[-1].lstrip("~") == parts[-2])
    )


# --- git: what a head adds, develop's index ---------------------------------------------------------------------------

HUNK = re.compile(r"@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class NeedsBlobs(Exception):
    """The head's changed files are not all local (a partial clone), and this pass may not fetch them."""


@contextlib.contextmanager
def no_fetch(on: bool = True):
    """git may not fetch a missing blob from the promisor remote while this is on (GIT_NO_LAZY_FETCH)."""
    old = os.environ.get("GIT_NO_LAZY_FETCH")
    if on:
        os.environ["GIT_NO_LAZY_FETCH"] = "1"
    try:
        yield
    finally:
        if on and old is None:
            os.environ.pop("GIT_NO_LAZY_FETCH", None)
        elif on:
            os.environ["GIT_NO_LAZY_FETCH"] = old


def added_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """The line ranges each file gains (new side), from `git diff -U0`; deleted files have none."""
    out: dict[str, list[tuple[int, int]]] = {}
    path, skip = None, 0
    for line in diff.split("\n"):
        if line.startswith("\\"):
            continue  # "\ No newline at end of file" is not counted in the hunk
        if skip:
            skip -= 1
            continue
        if line.startswith("diff --git "):
            path = None
        elif line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@ "):
            m = HUNK.match(line)
            if not m:
                continue
            old = int(m.group(1)) if m.group(1) is not None else 1
            start, new = int(m.group(2)), (
                int(m.group(3)) if m.group(3) is not None else 1
            )
            skip = old + new
            if path and new:
                out.setdefault(path, []).append((start, start + new - 1))
    return out


def missing_blobs(repo: str, mb: str, head: str) -> int:
    """How many blobs a diff of mb..head over the code files needs that this clone does not have (no fetch)."""
    with no_fetch():
        raw = gitops.git(
            repo,
            "diff",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--diff-filter=AM",
            mb,
            head,
            "--",
            *CODE_SPECS,
        )
    want = {
        x
        for line in raw.splitlines()
        if line.startswith(":")
        for x in line.split()[2:4]
        if x.strip("0")
    }
    if not want:
        return 0
    p = subprocess.run(
        ["git", "-C", repo, "cat-file", "--batch-check"],
        input="\n".join(sorted(want)) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    return sum(1 for line in p.stdout.splitlines() if line.endswith(" missing"))


def head_defs(repo: str, head: str, base: str, fetch: bool = False) -> dict:
    """The definitions `head` adds against merge-base(base, head), each with its path; and the code files it changes.
    Raises NeedsBlobs when a needed blob is not local and `fetch` is off."""
    mb = gitops.git(repo, "merge-base", base, head).strip()
    if missing_blobs(repo, mb, head) and not fetch:
        raise NeedsBlobs(head)
    with no_fetch(not fetch):
        diff = gitops.git(
            repo,
            "-c",
            "core.quotepath=false",
            "diff",
            "-U0",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--diff-filter=AM",
            mb,
            head,
            "--",
            *CODE_SPECS,
            timeout=600,
        )
        ranges = {p: r for p, r in added_ranges(diff).items() if language(p)}
        texts = gitops.cat_blobs(repo, [f"{head}:{p}" for p in sorted(ranges)])
    defs = []
    for path in sorted(ranges):
        text = texts.get(f"{head}:{path}")
        if text is None:
            continue
        starts = [a for a, _ in ranges[path]]
        for d in file_defs(path, text):
            k = bisect.bisect_right(starts, d["line"]) - 1
            if k >= 0 and d["line"] <= ranges[path][k][1]:
                d["path"] = path
                defs.append(d)
    return {"mb": mb, "paths": sorted(ranges), "defs": defs}


def tops(paths: list[str]) -> list[str]:
    """Develop's areas for a set of changed files: their first DEPTH path parts; a file nearer the root stands alone."""
    out = set()
    for p in paths:
        parts = p.split("/")
        out.add("/".join(parts[:DEPTH]) if len(parts) > DEPTH else p)
    return sorted(out)


def _compact(d: dict) -> list:
    """A develop definition as stored: kind, name, line, and the body hash when the body is long enough to count as a
    pattern copy (declarations and short bodies match everything)."""
    return [
        d["kind"],
        d["name"],
        d["line"],
        d["h"] if not d["decl"] and d["toks"] >= MIN_DEVELOP_TOKENS else "",
    ]


def develop_update(repo: str, sha: str, areas: list[str], cache: dict) -> dict:
    """Develop's index at `sha` over `areas`: every code file's definitions, read only for blobs the cache lacks, and
    the sketches of the bodies develop has 2+ exact copies of (develop's patterns). Updates `cache` in place; returns
    {"files": n, "read": n, "missing": n}."""
    listing = (
        gitops.git(repo, "ls-tree", "-r", "-z", sha, "--", *areas, timeout=300)
        if areas
        else ""
    )
    files = {}
    for entry in listing.split("\0"):
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob" and language(path):
            files[path] = parts[2]
    old, fresh = cache.get("blobs") or {}, {}
    need = sorted({b for b in files.values() if b not in old})
    with no_fetch():
        texts = gitops.cat_blobs(repo, need, timeout=600) if need else {}
    sketches = {}
    by_blob = {}
    for path, blob in files.items():
        if blob in old:
            by_blob[blob] = old[blob]
            continue
        if texts.get(blob) is None:
            continue  # not local: the next update reads it
        defs = file_defs(path, texts[blob])
        by_blob[blob] = [_compact(d) for d in defs]
        fresh[blob] = True
        for d in defs:
            if _compact(d)[3]:
                sketches[d["h"]] = d["sk"]
    counts: dict[str, int] = {}
    for path, blob in files.items():
        for row in by_blob.get(blob) or []:
            if row[3]:
                counts[row[3]] = counts.get(row[3], 0) + 1
    patterns = {
        h: sk
        for h, sk in (cache.get("patterns") or {}).items()
        if counts.get(h, 0) >= MIN_DEVELOP_COPIES
    }
    lacking = {
        h
        for h, c in counts.items()
        if c >= MIN_DEVELOP_COPIES and h not in patterns and h not in sketches
    }
    if (
        lacking
    ):  # a pattern formed by blobs read in earlier updates: read one copy again for its sketch
        again = {}
        for path, blob in files.items():
            for row in by_blob.get(blob) or []:
                if row[3] in lacking and row[3] not in again:
                    again[row[3]] = (path, blob)
        with no_fetch():
            more = gitops.cat_blobs(
                repo, sorted({b for _, b in again.values()}), timeout=600
            )
        for h, (path, blob) in again.items():
            for d in file_defs(path, more.get(blob) or ""):
                if d["h"] == h:
                    sketches[h] = d["sk"]
    for h, c in counts.items():
        if c >= MIN_DEVELOP_COPIES and h not in patterns and h in sketches:
            patterns[h] = sketches[h]
    cache["develop"] = {"sha": sha, "areas": areas, "files": files, "at": now()}
    cache["blobs"] = {
        b: by_blob[b] for b in sorted(set(files.values())) if b in by_blob
    }
    cache["patterns"] = patterns
    return {
        "files": len(files),
        "read": len(fresh),
        "missing": sum(1 for b in need if texts.get(b) is None),
    }


def develop_view(cache: dict) -> dict:
    """What grouping reads from develop's index: names per family, names per file, the exact-copy buckets of pattern
    bodies with an inverted index of their sketches, and the name families (each key of `name_keys`).
    """
    dev = cache.get("develop") or {}
    blobs = cache.get("blobs") or {}
    patterns = cache.get("patterns") or {}
    names: dict[str, set] = {}
    by_path: dict[str, set] = {}
    buckets: dict[str, list] = {}
    suffix: dict[tuple, list] = {}
    for path, blob in (dev.get("files") or {}).items():
        for kind, name, line, h in blobs.get(blob) or []:
            fam = family(path, kind)
            names.setdefault(fam, set()).add(name)
            by_path.setdefault(path, set()).add(f"{fam}/{name}")
            if h in patterns:
                buckets.setdefault(h, []).append((path, line, name))
            if FAMILY[kind] != "test" and not skipped(name):
                for key in name_keys(name):
                    suffix.setdefault((fam, key), []).append((path, line, name))
    index: dict[int, list] = {}
    for h, sk in patterns.items():
        for x in sk:
            index.setdefault(x, []).append(h)
    return {
        "sha": dev.get("sha"),
        "names": names,
        "by_path": by_path,
        "buckets": buckets,
        "patterns": patterns,
        "index": index,
        "suffix": suffix,
    }


# --- groups ----------------------------------------------------------------------------------------------------------


def _visible(d: dict) -> bool:
    """Seen beyond its own file and the tests."""
    return not d["private"] and not d["test"]


def _rep(defs: list[dict]) -> dict:
    """The definition that stands for a PR in a same-name group: a definition before a declaration, then the biggest
    body, then one in a shared layer."""
    return min(
        defs,
        key=lambda d: (
            d["decl"],
            -d["toks"],
            not shared_layer(d["path"]),
            d["path"],
            d["line"],
        ),
    )


def same_thing(x: dict, y: dict, sim: float) -> bool:
    """Whether a same-name pair from two PRs is the same thing, not a coincidence:
    - similar bodies (SIM_SAME), or identical short ones that both PRs make visible (a shared constant's value);
    - both PRs add it to the same file (a conflict in waiting), or it is a CTest name or a header macro (one global
      name space each);
    - a function both make visible whose name says what it does (two words or more: `sha1_hex`, `resolve_num_cus`).
    The rest is a coincidence: private or test-local helpers with unrelated bodies (`_cfg`, `_div`, `_fake`), and
    constants, types and one-word functions with unrelated bodies in different modules (`ARMS`, `Plan`, `family`).
    """
    if not (x["decl"] or y["decl"]):
        if min(x["toks"], y["toks"]) >= MIN_SIM_TOKENS:
            if sim >= SIM_SAME:
                return True
        elif x["h"] == y["h"] and _visible(x) and _visible(y):
            return True
    if x["path"] == y["path"] or x["kind"] == "ctest":
        return True
    if (
        {x["kind"], y["kind"]} <= {"macro", "fmacro"}
        and x["path"].endswith(HEADER)
        and y["path"].endswith(HEADER)
    ):
        return True
    return (
        FAMILY[x["kind"]] == "func"
        and _visible(x)
        and _visible(y)
        and words(x["name"]) >= 2
    )


def _components(nodes: list, edges: list[tuple]) -> list[list]:
    parent = {v: v for v in nodes}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b, *_ in edges:
        parent[find(a)] = find(b)
    comps: dict = {}
    for v in nodes:
        comps.setdefault(find(v), []).append(v)
    return [sorted(c) for c in comps.values() if len(c) > 1]


def _member(n: int, d: dict, also: list[dict] | None = None) -> dict:
    m = {
        "n": n,
        "path": d["path"],
        "line": d["line"],
        "kind": d["kind"],
        "name": d["name"],
        "sig": d["sig"][:120],
        "lines": d["lines"],
        "private": d["private"],
        "test": d["test"],
        "decl": d["decl"],
    }
    if also:
        m["also"] = [[x["path"], x["line"]] for x in also[:3]]
    return m


def develop_copies(d: dict, view: dict) -> list[tuple]:
    """Develop's exact-copy bodies (2+ identical) this definition matches at SIM_DEVELOP or more: (path, line, name,
    similarity)."""
    if d["decl"] or d["toks"] < MIN_DEVELOP_TOKENS or not d["sk"]:
        return []
    hits: dict[str, int] = {}
    for x in d["sk"]:
        for h in view["index"].get(x, ())[:POSTING_CAP]:
            hits[h] = hits.get(h, 0) + 1
    need = max(1, len(d["sk"]) // 4)
    out, lang = [], language(d["path"])
    for h, c in hits.items():
        if c < need:
            continue
        s = similarity(d["sk"], view["patterns"][h])
        if (
            s >= SIM_DEVELOP
        ):  # a copy in the PR's own file is its old self (a rename, a move), not a pattern
            out += [
                (p, ln, nm, round(s, 2))
                for p, ln, nm in view["buckets"].get(h, ())
                if language(p) == lang and p != d["path"]
            ]
    return sorted(out, key=lambda t: (-t[3], t[0], t[1]))


def name_family(d: dict, fam: str, view: dict) -> list[tuple]:
    """Develop's definitions of this definition's name, outside its own file: the same name, or develop's prefixed
    copies of it (`streq` and `rocke_*_streq` in three variants), when there are 2 to FAMILY_MAX of them (a larger
    family is a generic name). Only the full name counts: a shared suffix alone (`*_bool`, `*_spec_t`) finds siblings
    with a similar shape, not copies. [] when none fits."""
    keys = name_keys(d["name"])
    if d["decl"] or d["toks"] < MIN_DEVELOP_TOKENS or not keys:
        return []
    members = [m for m in view["suffix"].get((fam, keys[0]), ()) if m[0] != d["path"]]
    return members if MIN_DEVELOP_COPIES <= len(members) <= FAMILY_MAX else []


def name_copies(d: dict, members: list[tuple], sketches: dict) -> list[tuple]:
    """The family `members` with their similarity to this body, most similar first, when one is SIM_SAME similar or
    more (else []); `sketches` holds develop's bodies by (path, line)."""
    sims = [
        (p, ln, nm, round(similarity(d["sk"], sketches[(p, ln)]), 2))
        for p, ln, nm in members
        if (p, ln) in sketches
    ]
    if sims and max(s for *_, s in sims) >= SIM_SAME:
        return sorted(sims, key=lambda t: (-t[3], t[0], t[1]))
    return []


def compute(
    prs: dict[int, list[dict]], view: dict, related: set | None = None, sketches=None
) -> dict:
    """The groups and leads (module docstring) for the open PRs' added definitions `prs` (n -> definitions with their
    paths) against develop's `view`. `related` holds frozenset pairs of PRs whose overlap is by construction (a stack);
    `sketches` reads develop bodies for name families ((path, line, name) list -> {(path, line): sketch}), None to skip
    them."""
    related = related or set()
    cands, leads, todo, families = [], [], [], {}
    for n in sorted(prs):
        for d in prs[n]:
            if skipped(d["name"]):
                continue
            fam = family(d["path"], d["kind"])
            if f"{fam}/{d['name']}" in view["by_path"].get(d["path"], ()):
                continue  # develop has it in this file: an edit, not an addition
            todo.append((n, d, fam))
            if sketches is not None:
                families[id(d)] = name_family(d, fam, view)
    wanted = sorted({m for ms in families.values() for m in ms})
    bodies = (
        sketches(wanted) if wanted else {}
    )  # one batch: a read per definition takes about 25 s cold
    for n, d, fam in todo:
        copies, by = develop_copies(d, view), "body"
        if len(copies) < MIN_DEVELOP_COPIES and families.get(id(d)):
            copies, by = name_copies(d, families[id(d)], bodies), "name"
        if len(copies) >= MIN_DEVELOP_COPIES:
            leads.append(
                {
                    **_member(n, d),
                    "copies": [list(c[:3]) for c in copies[:3]],
                    "count": len(copies),
                    "sim": max(c[3] for c in copies),
                    "by": by,
                    "at": sorted(f"{c[0]}:{c[1]}" for c in copies),
                }
            )
            continue
        cands.append((n, d, fam))
    groups = _same_name(cands, view, related) + _same_logic(cands, related)
    users: dict[str, set] = {}
    for lead in leads:
        for at in lead["at"]:
            users.setdefault(at, set()).add(lead["n"])
    for lead in leads:
        lead["also"] = sorted(
            set().union(*(users[at] for at in lead.pop("at"))) - {lead["n"]}
        )
    kept: list[dict] = []
    for n in sorted({x["n"] for x in leads}):
        mine = sorted(
            (x for x in leads if x["n"] == n),
            key=lambda x: (
                x["test"],
                -len(x["also"]),
                -x["count"],
                -x["lines"],
                x["path"],
            ),
        )
        kept += mine[:LEADS_PER_PR]
    return {"groups": groups, "leads": kept, "lead_total": len(leads)}


def _same_name(cands: list[tuple], view: dict, related: set) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for n, d, fam in cands:
        if d["name"] in view["names"].get(fam, ()):
            continue  # develop defines it: not new work
        by_key.setdefault((fam, d["name"]), {}).setdefault(n, []).append(d)
    groups = []
    for (fam, name), per in sorted(by_key.items()):
        if len(per) < 2:
            continue
        reps = {n: _rep(ds) for n, ds in per.items()}
        ns, edges = sorted(reps), []
        for i, a in enumerate(ns):
            for b in ns[i + 1 :]:
                if frozenset((a, b)) in related:
                    continue
                s = similarity(reps[a]["sk"], reps[b]["sk"])
                if same_thing(reps[a], reps[b], s):
                    edges.append((a, b, round(s, 2)))
        for comp in _components(ns, edges):
            rep = reps[comp[0]]
            groups.append(
                {
                    "key": f"{fam}/{name}",
                    "rule": "same",
                    "kind": KIND_WORD[rep["kind"]],
                    "name": name,
                    "members": [
                        _member(n, reps[n], [x for x in per[n] if x is not reps[n]])
                        for n in comp
                    ],
                    "sims": {f"{a}-{b}": s for a, b, s in edges if a in comp},
                }
            )
    types = {
        (g["key"].split("/", 1)[0].split(".")[0], g["name"]): {
            m["n"] for m in g["members"]
        }
        for g in groups
        if g["key"].split("/", 1)[0].endswith(".type")
    }

    def under_type(
        g: dict,
    ) -> bool:  # a method of a class the same PRs both add: the class's group says it
        lang, sep = g["key"].split(".", 1)[0], (
            "." if g["key"].startswith("py.") else "::"
        )
        owner = g["name"].rsplit(sep, 1)[0] if sep in g["name"] else None
        return bool(owner) and {m["n"] for m in g["members"]} <= types.get(
            (lang, owner), set()
        )

    return [g for g in groups if not under_type(g)]


def _same_logic(cands: list[tuple], related: set) -> list[dict]:
    pool = [
        (n, d, fam)
        for n, d, fam in cands
        if not d["decl"] and d["lines"] >= MIN_LOGIC_LINES and d["sk"]
    ]
    index: dict[int, list[int]] = {}
    for i, (_, d, _) in enumerate(pool):
        for x in d["sk"]:
            index.setdefault(x, []).append(i)
    edges, seen = [], set()
    for i, (n, d, fam) in enumerate(pool):
        hits: dict[int, int] = {}
        for x in d["sk"]:
            post = index[x]
            if len(post) > POSTING_CAP:
                continue
            for j in post:
                if j > i:
                    hits[j] = hits.get(j, 0) + 1
        need = max(1, int(len(d["sk"]) * SIM_LOGIC / 2))
        for j, c in hits.items():
            m, e, gam = pool[j]
            if (
                c < need
                or m == n
                or e["name"] == d["name"]
                or gam != fam
                or frozenset((m, n)) in related
            ):
                continue
            s = similarity(d["sk"], e["sk"])
            if s >= SIM_LOGIC:
                edges.append((i, j, round(s, 2)))
                seen.update((i, j))
    groups = []
    for comp in _components(sorted(seen), edges):
        if len({pool[i][0] for i in comp}) < 2 or all(pool[i][1]["test"] for i in comp):
            continue  # one PR's copies are the code quality pass's; test helpers alone are not shared work
        names = sorted({pool[i][1]["name"] for i in comp})
        fam = pool[comp[0]][2]
        key = f"like/{fam}/" + "+".join(names)
        if len(key) > 110:
            key = key[:96] + "+" + hashlib.sha1(key.encode()).hexdigest()[:8]
        members = [_member(pool[i][0], pool[i][1]) for i in comp]
        groups.append(
            {
                "key": key,
                "rule": "logic",
                "kind": KIND_WORD[pool[comp[0]][1]["kind"]],
                "name": names[0],
                "members": sorted(
                    members, key=lambda m: (m["n"], m["path"], m["line"])
                ),
                "sims": {
                    f"{pool[a][0]}:{pool[a][1]['name']}-{pool[b][0]}:{pool[b][1]['name']}": s
                    for a, b, s in edges
                    if a in comp
                },
            }
        )
    return groups


# --- land order -------------------------------------------------------------------------------------------------------

STAGE_ORDER = {"ready": 0, "candidate": 1, "held": 2}


def land_order(st: dict, ns: list[int]) -> list[tuple[int, str]]:
    """The PRs `ns` in the order they are expected to land, each with why: the merge plan's order for the PRs it places
    (its `rank`, which sets the queue's `plan_rank`), then the rest by develop fixes and the user's priorities,
    readiness (ready, candidate, the rest), fewer failing gates, the older PR. The first proposes the owner of shared
    code. One function, so the plan's order and the fallback never disagree in two places.
    """
    from .mergeplan import place, why_first

    mp = st.get("merge_plan") or {}
    plan = (
        {}
        if mp.get("unavailable")
        else {
            **{k: v.get("rank") for k, v in (mp.get("dev_conflicts") or {}).items()},
            **{k: v.get("rank") for k, v in (mp.get("nodes") or {}).items()},
        }
    )
    rows = {r["n"]: r for r in st.get("queue") or []}
    fixes = st.get("develop_fixes") or {}

    def key(n: int) -> tuple:
        row, rank = rows.get(n) or {}, plan.get(str(n))
        return (
            rank is None,
            rank or 0,
            str(n) not in fixes,
            STAGE_ORDER.get(row.get("stage"), 3),
            len(row.get("blockers") or []),
            n,
        )

    out = []
    for n in sorted(ns, key=key):
        row, rank = rows.get(n) or {}, plan.get(str(n))
        why = []
        if rank is not None:
            w = place(st, n)
            why.append(f"merge plan: {w}" if w else "in the merge plan")
            if (mp.get("why") or {}).get(str(n)):
                why[-1] += f" ({mp['why'][str(n)]})"
        else:
            if str(n) in fixes:
                why.append(why_first(st, n))
            stage, b = row.get("stage"), len(row.get("blockers") or [])
            why.append(
                stage
                if stage
                else (
                    "not in the queue"
                    if not row
                    else (
                        f"not ready: {b} failing gate{'s' if b != 1 else ''}"
                        if b
                        else "not ready"
                    )
                )
            )
        out.append((n, "; ".join(why)))
    return out


# --- the cache and the tick step --------------------------------------------------------------------------------------


def settings(cfg: dict) -> dict:
    return dict(DEFAULTS, **(cfg.get("overlaps") or {}))


def _pack(sk: list[int]) -> str:
    return base64.b64encode(struct.pack(f"<{len(sk)}I", *sk)).decode("ascii")


def _unpack(s: str) -> list[int]:
    raw = base64.b64decode(s)
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def load_cache(store) -> dict:
    """The tick's cache (gzip JSON, sketches packed): heads, develop's index, its patterns and the develop bodies read
    for name families. A cache from another CACHE_VERSION, or one that does not read, starts over.
    """
    path = store.dir / CACHE_FILE
    try:
        cache = json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}
    except (OSError, ValueError, EOFError):
        cache = {}
    if cache.get("v") != CACHE_VERSION:
        return {
            "v": CACHE_VERSION,
            "heads": {},
            "develop": {},
            "blobs": {},
            "patterns": {},
            "named": {},
        }
    for v in cache["heads"].values():
        for d in v["defs"]:
            d["sk"] = _unpack(d["sk"])
    cache["patterns"] = {h: _unpack(s) for h, s in cache["patterns"].items()}
    cache.setdefault("named", {})
    return cache


def save_cache(store, cache: dict) -> None:
    """Written whole (whole_file), so an interrupted write leaves the last good cache, at the mode of the files the
    monitor and the writer share (worker/perms.py)."""
    out = dict(
        cache,
        patterns={h: _pack(sk) for h, sk in cache["patterns"].items()},
        heads={
            k: dict(v, defs=[dict(d, sk=_pack(d["sk"])) for d in v["defs"]])
            for k, v in cache["heads"].items()
        },
    )
    whole_file.write(
        store.dir / CACHE_FILE,
        gzip.compress(
            json.dumps(out, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            6,
        ),
        paths.worker_module("perms").SHARED_FILE_MODE,
    )


def develop_sketches(repo: str, cache: dict):
    """A reader of develop bodies for name families: (path, line, name) list -> {(path, line): sketch}. A blob is read
    once (no fetch) and its bodies' sketches kept in cache["named"]."""
    named, files = (
        cache.setdefault("named", {}),
        (cache.get("develop") or {}).get("files") or {},
    )

    def read(members: list) -> dict:
        want: dict[str, tuple] = {}
        for p, ln, _ in members:
            b = files.get(p)
            if b and str(ln) not in (named.get(b) or {}):
                want.setdefault(b, (p, set()))[1].add(str(ln))
        if want:
            with no_fetch():
                texts = gitops.cat_blobs(repo, sorted(want), timeout=600)
            for blob, (path, lines) in want.items():
                if texts.get(blob) is None:
                    continue
                got = named.setdefault(blob, {})
                for d in file_defs(path, texts[blob]):
                    if str(d["line"]) in lines and not d["decl"]:
                        got[str(d["line"])] = _pack(d["sk"])
                for ln in lines - set(got):
                    got[ln] = ""  # no body there (a declaration): not read again
        out = {}
        for p, ln, _ in members:
            got = (named.get(files.get(p)) or {}).get(str(ln))
            if got:
                out[(p, ln)] = _unpack(got)
        return out

    return read


def _open(st: dict) -> dict[str, dict]:
    return {
        k: r
        for k, r in st["prs"].items()
        if r.get("status") == "open" and not r.get("draft") and r.get("head")
    }


def head_keys(st: dict) -> dict[str, tuple[str, str]]:
    """Each open PR's cache key and diff base: its head (and, stacked on another open PR, that PR's head), and develop
    or the base PR's head."""
    recs = _open(st)
    by_ref = {r.get("head_ref"): r for r in recs.values() if r.get("head_ref")}
    out = {}
    for k, r in recs.items():
        base = (
            by_ref.get(r.get("base")) if r.get("base", "develop") != "develop" else None
        )
        if base and base["n"] != r["n"]:
            out[k] = (f"{r['head'][:12]}^{base['head'][:12]}", base["head"])
        else:
            out[k] = (r["head"][:12], DEVELOP)
    return out


def related_pairs(st: dict) -> set:
    """PR pairs whose shared definitions are by construction: one is stacked on the other."""
    recs = _open(st)
    by_ref = {r.get("head_ref"): r["n"] for r in recs.values() if r.get("head_ref")}
    out = set()
    for r in recs.values():
        seen, cur = {r["n"]}, r
        while cur.get("base", "develop") != "develop" and by_ref.get(
            cur["base"]
        ) not in (None, *seen):
            m = by_ref[cur["base"]]
            out.add(frozenset((r["n"], m)))
            seen.add(m)
            cur = recs[str(m)]
    return out


def _areas(cfg: dict, st: dict, cache: dict) -> list[str]:
    """Develop's areas to index: those of every scanned head's code files, of the files the queue knows an open PR
    changes (so a cold start indexes them at once), and the rocKE tree."""
    paths = [p for v in cache["heads"].values() for p in v["paths"]]
    paths += [
        f
        for r in _open(st).values()
        for f in (r.get("files_cache") or {}).get("files") or []
        if language(f)
    ]
    return sorted(set(tops(paths + [repository_folder(cfg)])) - {""})


def _develop(repo: str, dev: str, areas: list[str], cache: dict, out: dict) -> bool:
    """Bring develop's index to `dev` over `areas` when either moved; True when the cache changed."""
    d = cache.get("develop") or {}
    if d.get("sha") == dev and d.get("areas") == areas and d.get("files"):
        return False
    try:
        got = develop_update(repo, dev, areas, cache)
    except GIT_ERRORS as exc:
        out["errors"].append(f"develop: {str(exc)[:120]}")
        return False
    out["develop_read"] += got["read"]
    if got["missing"]:
        out["errors"].append(f"develop: {got['missing']} blobs not local")
    return True


def _read_heads(
    cfg: dict, st: dict, cache: dict, keys: dict, rest: dict, out: dict, stop: float
) -> dict[str, str]:
    """Read the heads the cache lacks (develop fixes first, then the newest PRs), at most `max_heads`, until `stop`; of
    those whose blobs are not local, at most `max_fetches` fetch them. Returns why each unread PR waits; a PR that could
    not be read rests in `rest` for RETRY_MIN minutes."""
    c, repo = settings(cfg), cfg["paths"]["repo"]
    pending: dict[str, str] = {}
    fetches = c["max_fetches"]
    fixes = st.get("develop_fixes") or {}
    for k in sorted(
        (k for k in keys if keys[k][0] not in cache["heads"]),
        key=lambda k: (k not in fixes, -int(k)),
    ):
        rec, (key, base) = st["prs"][k], keys[k]
        if k in rest:
            pending[k] = rest[k]["why"]
            continue
        if out["heads"] >= c["max_heads"] or time.monotonic() >= stop:
            out["deferred"] += 1
            pending[k] = "waits for the next tick"
            continue
        why, res, t0 = None, None, time.monotonic()
        try:
            if not gitops.has_commit(repo, rec["head"]):
                why = "its head is not fetched yet"
            else:
                try:
                    res = head_defs(repo, rec["head"], base)
                except NeedsBlobs:
                    if c["max_fetches"] <= 0:
                        why = "its changed files are not local yet"
                    elif fetches <= 0:
                        out["deferred"] += 1
                        pending[k] = "waits for the next tick (blob fetch budget spent)"
                        continue
                    else:
                        fetches -= 1
                        out["fetched"] += 1
                        res = head_defs(repo, rec["head"], base, fetch=True)
        except GIT_ERRORS as exc:
            out["errors"].append(f"#{k}: {str(exc)[:120]}")
            why = "could not be read"
        if why:
            pending[k] = why
            rest[k] = {"key": key, "at": now(), "why": why}
            continue
        cache["heads"][key] = {
            "n": int(k),
            "head": rec["head"],
            "at": now(),
            "secs": round(time.monotonic() - t0, 2),
            "paths": res["paths"],
            "defs": res["defs"],
        }
        out["heads"] += 1
    return pending


def refresh(cfg: dict, st: dict, store) -> dict:
    """One tick's refresh (module docstring): develop when it moved, then the heads not read yet in what is left of
    `budget_s`, then the groups into st["overlaps"]. A head that could not be read (not fetched, blobs not local) is
    tried again after RETRY_MIN minutes; a pass with nothing new reads nothing. Returns the counts; `errors` lists what
    failed."""
    repo, dev = (cfg.get("paths") or {}).get("repo"), st["develop"].get("sha")
    out = {"heads": 0, "fetched": 0, "deferred": 0, "develop_read": 0, "errors": []}
    if not repo or not dev:
        return out
    keys = head_keys(st)
    prev = st.get("overlaps") or {}
    done, waiting = prev.get("keys") or {}, prev.get("waiting") or {}

    def resting(k: str) -> bool:  # tried at this head lately and could not read it
        w = waiting.get(k) or {}
        return (
            w.get("key") == keys[k][0]
            and minutes_since(w.get("at") or "2000-01-01T00:00:00Z") < RETRY_MIN
        )

    if (
        prev.get("indexed") == dev[:12]
        and all(keys.get(k, (None,))[0] == v for k, v in done.items())
        and all(resting(k) for k in keys if done.get(k) != keys[k][0])
    ):
        return out  # nothing moved since the last pass
    start = time.monotonic()
    cache = load_cache(store)
    t0 = time.monotonic()
    dirty = _develop(repo, dev, _areas(cfg, st, cache), cache, out)
    t1 = time.monotonic()
    secs = {"load": round(t0 - start, 2), "develop": round(t1 - t0, 2)}
    rest = {k: w for k, w in waiting.items() if k in keys and resting(k)}
    pending = _read_heads(
        cfg, st, cache, keys, rest, out, start + settings(cfg)["budget_s"]
    )
    live = {key for key, _ in keys.values()}
    if out["heads"] or set(cache["heads"]) - live:
        cache["heads"] = {key: v for key, v in cache["heads"].items() if key in live}
        dirty = True
    dirty = (
        _develop(repo, dev, _areas(cfg, st, cache), cache, out) or dirty
    )  # a head in a new area
    blobs = set((cache.get("develop") or {}).get("files", {}).values())
    cache["named"] = {b: v for b, v in (cache.get("named") or {}).items() if b in blobs}
    secs["heads"] = round(time.monotonic() - t1, 2)
    t2 = time.monotonic()
    view = develop_view(cache)
    prs = {v["n"]: v["defs"] for v in cache["heads"].values()}
    before = sum(len(v) for v in cache["named"].values())
    try:
        res = compute(prs, view, related_pairs(st), develop_sketches(repo, cache))
    except (
        GIT_ERRORS
    ) as exc:  # a develop body that cannot be read: the name families wait for the next pass
        out["errors"].append(f"name families: {str(exc)[:120]}")
        res = compute(prs, view, related_pairs(st))
    dirty = dirty or sum(len(v) for v in cache["named"].values()) != before
    secs["groups"] = round(time.monotonic() - t2, 2)
    t3 = time.monotonic()
    if dirty:
        save_cache(store, cache)
    secs["save"] = round(time.monotonic() - t3, 2)
    st["overlaps"] = {
        "at": now(),
        "develop": dev[:12],
        "indexed": (view["sha"] or "")[:12],
        "keys": {k: key for k, (key, _) in keys.items() if key in cache["heads"]},
        "scanned": {str(v["n"]): v["head"][:12] for v in cache["heads"].values()},
        "pending": pending,
        "waiting": rest,
        "groups": res["groups"],
        "leads": res["leads"],
        "lead_total": res["lead_total"],
        "secs": dict(secs, total=round(time.monotonic() - start, 2)),
    }
    return out


# --- decisions --------------------------------------------------------------------------------------------------------


def decisions(store) -> dict:
    path = store.dir / DECIDED_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError:
        return {}


def decide(
    st: dict,
    store,
    key: str,
    owner: int,
    members: list[int] | None,
    note: str,
    by: str,
    same_as: str = "",
) -> dict:
    """Record who owns a group's shared code (the posted proposal) in DECIDED_FILE, written whole and durable
    (state_store.write_file). The members default to the group's current ones;
    an owner that is not a member is refused (ValueError). Local only: no GitHub write, so no ledger entry.
    `same_as` files an earlier decision again under another group's key (a decision that also names a constant, which
    becomes a group of its own once another PR adopts it): it must
    name a recorded decision with the same owner, and that decision's overlap notices count for this key too.
    """
    if not members:
        g = next(
            (
                g
                for g in (st.get("overlaps") or {}).get("groups") or []
                if g["key"] == key
            ),
            None,
        )
        if not g:
            raise ValueError(f"no current group {key}: pass --members")
        members = sorted({m["n"] for m in g["members"]})
    members = sorted(set(members))
    if owner not in members:
        raise ValueError(
            f"#{owner} is not a member of {key} ({', '.join('#%d' % n for n in members)})"
        )
    allrec = decisions(store)
    if same_as:
        if same_as not in allrec:
            raise ValueError(
                f"--same-as {same_as}: no decision recorded under that key"
            )
        if allrec[same_as]["owner"] != owner:
            raise ValueError(
                f"--same-as {same_as}: that decision's owner is #{allrec[same_as]['owner']}, not #{owner}"
            )
    rec = {"owner": owner, "members": members, "note": note, "at": now(), "by": by}
    if same_as:
        rec["same_as"] = same_as
    allrec[key] = rec
    state_store.write_file(store.dir / DECIDED_FILE, allrec)
    store.journal(
        "overlap",
        f"{key}: #{owner} owns it; members {', '.join('#%d' % n for n in members)}"
        + (f"; the same decision as {same_as}" if same_as else "")
        + (f"; {note}" if note else ""),
    )
    return rec


def notices_due(st: dict, store, hint=lambda n: "") -> list[str]:
    """`OVERLAP_NOTICE_DUE #N` for each open PR a recorded decision asks to drop its copy, while no overlap notice for
    the group is on it yet (the notice names the duplication, its owner and resolution, the landing order and the
    rebase that removes the copy). One notice per PR may cover
    several groups: its ledger key joins them with ",". A group key may itself contain "+" (a "like/" group names its
    members' types: like/c.type/A+B+C), so a key counts whole first; older "+" joins still count. A decision filed
    again under another key (`same_as`) counts the notices of the key it repeats. `hint(n)` ends a PR's line (the
    first draft's command while models.notice_draft is bound: first_pass.notice_hint).
    """
    done = set()
    for r in store.ledger():
        if r["kind"] == "notice-overlap":
            key = str(r.get("head") or "")
            done |= {(r["pr"], k) for k in {key, *key.split(","), *key.split("+")}}
    due: dict[int, list[str]] = {}
    for key, rec in sorted(decisions(store).items()):
        for n in rec.get("members") or []:
            pr = (st.get("prs") or {}).get(str(n))
            if (
                n == rec["owner"]
                or (n, key) in done
                or (n, rec.get("same_as")) in done
                or not pr
                or pr.get("status") != "open"
            ):
                continue
            due.setdefault(n, []).append(f"{key} (owner #{rec['owner']})")
    return [
        f"OVERLAP_NOTICE_DUE #{n}: {'; '.join(v)}: post the notice (`rk notice {n} FILE --kind overlap --key "
        f"K[,K]`){hint(n)}"
        for n, v in sorted(due.items())
    ]


# --- reading: `rk overlaps`, the checks section -----------------------------------------------------------------------


def _who(st: dict, n: int) -> str:
    rec = st["prs"].get(str(n)) or {}
    return f"#{n} ({rec.get('author', '?')})"


def _where(m: dict, rocke_path: str) -> str:
    return f"{short(m['path'], rocke_path)}:{m['line']}"


def _decision_line(st: dict, key: str, rec: dict) -> str:
    owner = st["prs"].get(str(rec["owner"])) or {}
    state = owner.get("status")
    tail = (
        "; it has landed: reuse develop's copy"
        if state == "merged"
        else "; it was closed unmerged: decide again" if state == "closed" else ""
    )
    return (
        f"decided: #{rec['owner']} owns it ({rec['at'][:10]}, {rec.get('by') or '?'})"
        + (f": {rec['note']}" if rec.get("note") else "")
        + tail
    )


def _sim(g: dict, a: dict, b: dict) -> float | None:
    if g["rule"] == "same":
        x, y = sorted((a["n"], b["n"]))
        return g["sims"].get(f"{x}-{y}")
    return g["sims"].get(f"{a['n']}:{a['name']}-{b['n']}:{b['name']}") or g["sims"].get(
        f"{b['n']}:{b['name']}-{a['n']}:{a['name']}"
    )


def group_title(g: dict) -> str:
    if g["rule"] == "same":
        n = len({m["n"] for m in g["members"]})
        return f"{g['kind']} `{g['name']}`: the same new name in {n} open PRs"
    names = ", ".join(f"`{x}`" for x in sorted({m["name"] for m in g["members"]}))
    return f"{PLURAL.get(g['kind'], g['kind'] + 's')} with the same logic under other names ({names})"


def order_line(st: dict, g: dict, decided: dict | None = None) -> str:
    """The group's land order and the owner it proposes; a recorded decision replaces the proposal."""
    order = land_order(st, sorted({m["n"] for m in g["members"]}))
    tail = (
        f" → decided: #{decided['owner']} owns it (below)"
        if decided
        else f" → proposed owner #{order[0][0]}, unless another member puts it in its natural shared home"
    )
    return "land order: " + ", ".join(f"#{n} ({why})" for n, why in order) + tail


def render_group(
    st: dict, g: dict, i: int, decided: dict, rocke_path: str, n: int | None = None
) -> list[str]:
    """A group as text: its line, then a line per member (for PR n: the other PRs, with their similarity to n's), the
    land order, and a recorded decision."""
    lines = [f"{i}. {group_title(g)} [key {g['key']}]"]
    mine = [m for m in g["members"] if m["n"] == n]
    for m in mine:
        lines.append(
            f"   here: {_where(m, rocke_path)}{layer_tag(m['path'])}  {m['sig'][:100]}"
        )
    for m in g["members"]:
        if n is not None and m["n"] == n:
            continue
        s = (
            max(
                (x for x in (_sim(g, m, y) for y in mine) if x is not None),
                default=None,
            )
            if mine
            else None
        )
        tags = layer_tag(m["path"]) + (" [declaration]" if m["decl"] else "")
        lines.append(
            f"   {_who(st, m['n'])}: {_where(m, rocke_path)}{tags}"
            + (f", similarity {s:.2f}" if s is not None else "")
            + (f"  {m['sig'][:100]}" if n is None else "")
        )
    if n is None:
        sims = list(g["sims"].values())
        if sims:
            lines.append(
                f"   similarity {min(sims):.2f}–{max(sims):.2f}"
                if min(sims) != max(sims)
                else f"   similarity {sims[0]:.2f}"
            )
    lines.append("   " + order_line(st, g, decided.get(g["key"])))
    if g["key"] in decided:
        lines.append("   " + _decision_line(st, g["key"], decided[g["key"]]))
    return lines


def render_lead(st: dict, lead: dict, rocke_path: str, with_pr: bool) -> str:
    copies = ", ".join(
        f"`{nm}` ({short(p, rocke_path)}:{ln})" for p, ln, nm in lead["copies"]
    )
    more = (
        f" (+{lead['count'] - len(lead['copies'])})"
        if lead["count"] > len(lead["copies"])
        else ""
    )
    like = "like it" if lead.get("by") != "name" else "of its name family"
    return (
        f"- {_who(st, lead['n']) + ' ' if with_pr else ''}{KIND_WORD[lead['kind']]} `{lead['name']}` at "
        f"{_where(lead, rocke_path)}: develop has {lead['count']} {like}, e.g. {copies}{more}; best similarity "
        f"{lead['sim']:.2f}"
        + (
            f"; also added by {', '.join('#%d' % x for x in lead['also'])}"
            if lead["also"]
            else ""
        )
    )


def _status_line(st: dict, ov: dict, n: int | None) -> str:
    recs = _open(st)
    scanned = ov.get("scanned") or {}
    pend = ov.get("pending") or {}
    head = ""
    if n is not None:
        rec = st["prs"].get(str(n)) or {}
        at = scanned.get(str(n))
        head = f"; #{n} at {rec.get('head', '')[:10]}" + (
            ""
            if at and rec.get("head", "").startswith(at)
            else f" not scanned yet ({pend.get(str(n), 'next tick')})"
        )
    waiting = sorted(int(k) for k in recs if k not in scanned)
    return (
        f"(as of {ov.get('at', '?')[:16]}Z, develop {ov.get('indexed') or '-'}{head}; {len(scanned)} of "
        f"{len(recs)} open PRs scanned"
        + (
            f", not yet: {', '.join('#%d' % x for x in waiting[:8])}"
            + (f" (+{len(waiting) - 8})" if len(waiting) > 8 else "")
            if waiting
            else ""
        )
        + ")"
    )


def view_for(cfg: dict, st: dict, store, n: int) -> dict:
    """st["overlaps"], or, when PR n's current head is not in it yet, the same computed now with n's head read from the
    clone (read-only: no fetch, no state written)."""
    ov = st.get("overlaps") or {}
    rec = st["prs"].get(str(n)) or {}
    key = head_keys(st).get(str(n), (None, None))
    if not rec.get("head") or (ov.get("keys") or {}).get(str(n)) == key[0] or not ov:
        return ov
    cache = load_cache(store)
    repo = cfg["paths"]["repo"]
    entry = cache["heads"].get(key[0])
    if entry is None:
        try:
            with no_fetch():
                res = head_defs(repo, rec["head"], key[1])
            entry = {
                "n": n,
                "head": rec["head"],
                "paths": res["paths"],
                "defs": res["defs"],
            }
        except NeedsBlobs:
            return dict(
                ov,
                pending={
                    **(ov.get("pending") or {}),
                    str(n): "its changed files are not local here",
                },
            )
        except GIT_ERRORS as exc:
            return dict(
                ov,
                pending={
                    **(ov.get("pending") or {}),
                    str(n): f"could not be read: {str(exc)[:80]}",
                },
            )
    current = {v[0] for v in head_keys(st).values()}
    prs = {
        v["n"]: v["defs"]
        for k, v in cache["heads"].items()
        if k in current and v["n"] != n
    }
    prs[n] = entry["defs"]
    with no_fetch():
        res = compute(
            prs, develop_view(cache), related_pairs(st), develop_sketches(repo, cache)
        )
    return dict(
        ov,
        groups=res["groups"],
        leads=res["leads"],
        scanned={**(ov.get("scanned") or {}), str(n): rec["head"][:12]},
        pending={k: v for k, v in (ov.get("pending") or {}).items() if k != str(n)},
    )


def adds(defs: list[dict], key: str) -> bool:
    """Whether these definitions hold what a group key names: `fam/name`, or `like/fam/a+b`."""
    like = key.startswith("like/")
    fam, _, rest = key[5:].partition("/") if like else key.partition("/")
    names = set(rest.split("+")) if like else {rest}
    return any(d["name"] in names and family(d["path"], d["kind"]) == fam for d in defs)


def reminders(st: dict, store, decided: dict, n: int, shown: set) -> list[str]:
    """Recorded decisions naming PR n as a member, not the owner, that no current group shows, while n still adds what
    was decided (the owner landed, so develop has the name now and no group forms): n should drop its copy.
    """
    todo = {
        k: r
        for k, r in decided.items()
        if k not in shown and n in r.get("members", []) and r["owner"] != n
    }
    if not todo:
        return []
    key = head_keys(st).get(str(n), (None,))[0]
    entry = load_cache(store)["heads"].get(key)
    return [
        f"- [key {k}] " + _decision_line(st, k, r)
        for k, r in sorted(todo.items())
        if entry is None or adds(entry["defs"], k)
    ]


def section(cfg: dict, st: dict, store, n: int) -> list[str]:
    """The checks.md section for PR n (under the SECTION header)."""
    rocke_path = repository_folder(cfg)
    if str(n) not in _open(st):
        return [
            f"#{n} is not an open, non-draft PR the monitor tracks: nothing to compare"
        ]
    ov = view_for(cfg, st, store, n)
    if not ov:
        return ["not computed yet: the next tick scans the open PRs"]
    decided = decisions(store)
    groups = [
        g for g in ov.get("groups") or [] if any(m["n"] == n for m in g["members"])
    ]
    leads = [x for x in ov.get("leads") or [] if x["n"] == n]
    lines = [_status_line(st, ov, n)]
    for i, g in enumerate(groups, 1):
        lines += render_group(st, g, i, decided, rocke_path, n)
    more = reminders(st, store, decided, n, {g["key"] for g in groups})
    if more:
        lines += [
            "Decisions recorded earlier for shared code this PR also adds:"
        ] + more
    if leads:
        lines += ['Copies of a develop pattern (leads for "search before you add"):']
        lines += [render_lead(st, x, rocke_path, False) for x in leads]
    if not groups and not leads and not more:
        lines.append("none found")
    return lines


def render_all(cfg: dict, st: dict, store, n: int | None = None) -> list[str]:
    """`rk overlaps [N]`: every current group with its members (PR n's groups only, with n), then the develop-pattern
    leads, then recorded decisions no current group shows."""
    rocke_path = repository_folder(cfg)
    if n is not None and str(n) not in _open(st):
        return [f"#{n} is not an open, non-draft PR the monitor tracks"]
    ov = view_for(cfg, st, store, n) if n is not None else st.get("overlaps") or {}
    if not ov:
        return ["not computed yet: the next tick scans the open PRs"]
    decided = decisions(store)
    groups = [
        g
        for g in ov.get("groups") or []
        if n is None or any(m["n"] == n for m in g["members"])
    ]
    leads = [x for x in ov.get("leads") or [] if n is None or x["n"] == n]
    lines = [f"Overlapping work across the open PRs {_status_line(st, ov, n)}"]
    for i, g in enumerate(groups, 1):
        lines += render_group(st, g, i, decided, rocke_path)
    if not groups:
        lines.append("no group: none found")
    total = ov.get("lead_total", len(leads)) if n is None else len(leads)
    lines += [
        "",
        f"Copies of a develop pattern ({len(leads)} shown of {total}; at most {LEADS_PER_PR} per PR):",
    ]
    lines += [render_lead(st, x, rocke_path, True) for x in leads] or ["none found"]
    stale = sorted(
        k
        for k, rec in decided.items()
        if k not in {g["key"] for g in ov.get("groups") or []}
        and (n is None or n in rec.get("members", []))
    )
    if stale:
        lines += ["", "Recorded decisions with no current group:"]
        lines += [
            f"- [key {k}] "
            + _decision_line(st, k, decided[k])
            + "; members "
            + ", ".join("#%d" % x for x in decided[k]["members"])
            for k in stale
        ]
    return lines


# --- the change fingerprint (idle.py) ---------------------------------------------------------------------------------

# An owner decision recorded by hand (`rk overlap-decide`) makes the next tick full.
idle.file_part("owner decisions", lambda store: store.dir / DECIDED_FILE)


@idle.timer("overlap reads", again=True)
def _reads_due(cfg: dict, st: dict, store):
    """Heads a pass left for the next tick (its time or fetch budget spent) are due at the next tick."""
    ov = st.get("overlaps") or {}
    if any(
        str(why).startswith("waits for the next tick")
        for why in (ov.get("pending") or {}).values()
    ):
        yield "overlap reads left", ov.get("at") or now()


@idle.timer("overlap retries")
def _retries_due(cfg: dict, st: dict, store):
    """A head that could not be read is tried again RETRY_MIN minutes later."""
    for k, w in ((st.get("overlaps") or {}).get("waiting") or {}).items():
        if w.get("at"):
            yield f"overlap retry #{k}", idle.plus(w["at"], RETRY_MIN)
