# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared source readers for the maps and lints: file sets, Python symbols,
the C++ symbol index, header twin tables and "not ported" claims.

Everything here reads the rocKE tree and never writes to it.
"""

from __future__ import annotations

import ast
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cscan  # noqa: E402


def _default_rocke() -> Optional[str]:
    """$ROCKE_ROOT, else the rocKE tree this file sits in (a copy under
    platform/tools/), else None: the caller passes --rocke."""
    env = os.environ.get("ROCKE_ROOT")
    if env:
        return env
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p):
        if os.path.isdir(os.path.join(p, "platform", "cpp")) and os.path.isdir(
            os.path.join(p, "library", "kernels")
        ):
            return p
        p = os.path.dirname(p)
    return None


DEFAULT_ROCKE = _default_rocke()

# Python builder modules the backward path uses (under library/kernels/common).
BWD_PY_GLOBS = ["*attention*bwd*.py"]
# C++ files of the backward path (include/rocke and instances/common).
BWD_H_GLOBS = [
    "helper_rocke.instances.common.*attention*bwd*.h",
    "instance_*attention*bwd*.h",
]
BWD_CPP_GLOBS = ["*attention*bwd*.cpp"]


# ------------------------------------------------------------------ paths


class Tree:
    """Paths inside a rocKE root."""

    def __init__(self, root: Optional[str] = DEFAULT_ROCKE):
        self.given = bool(root)
        self.root = os.path.abspath(root or ".")
        self.kernels = os.path.join(self.root, "library", "kernels", "common")
        self.include = os.path.join(self.root, "platform", "cpp", "include", "rocke")
        self.instances = os.path.join(
            self.root, "platform", "cpp", "instances", "common"
        )
        self.parity = os.path.join(self.root, "library", "tests", "parity")
        self.ir_py = os.path.join(
            self.root, "platform", "python", "rocke", "core", "ir.py"
        )
        self.ir_h = os.path.join(self.include, "ir.h")
        self.pyroot = os.path.join(self.root, "platform", "python")

    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.root)

    def check(self) -> Optional[str]:
        if not self.given:
            return "no rocKE tree: pass --rocke <root> or set ROCKE_ROOT"
        for p in (self.kernels, self.include, self.instances):
            if not os.path.isdir(p):
                return f"not a rocKE root (missing {p})"
        return None

    def bwd_py(self) -> List[str]:
        out = set()
        for g in BWD_PY_GLOBS:
            out.update(glob.glob(os.path.join(self.kernels, g)))
        return sorted(out)

    def bwd_cxx(self, parity: bool = False) -> List[str]:
        out = set()
        for g in BWD_H_GLOBS:
            out.update(glob.glob(os.path.join(self.include, g)))
        for g in BWD_CPP_GLOBS:
            out.update(glob.glob(os.path.join(self.instances, g)))
        if parity:
            out.update(glob.glob(os.path.join(self.parity, "*attention*bwd*_emit.c")))
        return sorted(out)


# ------------------------------------------------------------------ names


def snake(s: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", s).lower()


def norm(s: str) -> str:
    """Key for comparing a Python name with a C name: lower case, no underscores."""
    return re.sub(r"_", "", s).lower()


# ------------------------------------------------------------------ python symbols


@dataclass
class PySym:
    module: str  # file stem, e.g. _attention_bwd_valu
    qual: str  # name or Class.name
    kind: str  # function, class, method, property, const
    line: int
    params: List[dict] = field(default_factory=list)
    returns: str = ""
    private: bool = False
    cls: str = ""

    @property
    def key(self) -> str:
        return f"{self.module}:{self.qual}"

    @property
    def name(self) -> str:
        return self.qual.split(".")[-1]


def _params(fn: ast.FunctionDef) -> List[dict]:
    a = fn.args
    out = []
    pos = list(a.posonlyargs) + list(a.args)
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(pos, defaults):
        out.append(_param(arg, d, "pos"))
    if a.vararg:
        out.append(_param(a.vararg, None, "var"))
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        out.append(_param(arg, d, "kwonly"))
    if a.kwarg:
        out.append(_param(a.kwarg, None, "varkw"))
    return out


def _param(arg: ast.arg, default, kind: str) -> dict:
    return {
        "name": arg.arg,
        "kind": kind,
        "annotation": ast.unparse(arg.annotation) if arg.annotation else "",
        "default": ast.unparse(default) if default is not None else None,
    }


def _is_property(fn) -> bool:
    for d in fn.decorator_list:
        s = ast.unparse(d)
        if s in (
            "property",
            "cached_property",
            "functools.cached_property",
        ) or s.endswith(".setter"):
            return True
    return False


def py_symbols(path: str) -> List[PySym]:
    mod = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    out: List[PySym] = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(
                PySym(
                    mod,
                    n.name,
                    "function",
                    n.lineno,
                    _params(n),
                    ast.unparse(n.returns) if n.returns else "",
                    n.name.startswith("_"),
                )
            )
        elif isinstance(n, ast.ClassDef):
            out.append(
                PySym(mod, n.name, "class", n.lineno, [], "", n.name.startswith("_"))
            )
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "property" if _is_property(m) else "method"
                    if kind == "property" and any(
                        s.qual == f"{n.name}.{m.name}" for s in out
                    ):
                        continue  # a setter
                    dunder = m.name.startswith("__") and m.name.endswith("__")
                    out.append(
                        PySym(
                            mod,
                            f"{n.name}.{m.name}",
                            kind,
                            m.lineno,
                            _params(m),
                            ast.unparse(m.returns) if m.returns else "",
                            (m.name.startswith("_") and not dunder)
                            or n.name.startswith("_"),
                            n.name,
                        )
                    )
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                names = (
                    [t]
                    if isinstance(t, ast.Name)
                    else (list(t.elts) if isinstance(t, ast.Tuple) else [])
                )
                for x in names:
                    if isinstance(x, ast.Name) and x.id not in ("__all__", "__slots__"):
                        val = getattr(n, "value", None)
                        alias = (
                            isinstance(val, ast.Subscript)
                            and ast.unparse(val).startswith(
                                (
                                    "Union[",
                                    "Optional[",
                                    "Tuple[",
                                    "tuple[",
                                    "Callable[",
                                    "Sequence[",
                                    "List[",
                                    "Dict[",
                                )
                            )
                            or (
                                isinstance(val, ast.BinOp)
                                and isinstance(val.op, ast.BitOr)
                            )
                        )
                        out.append(
                            PySym(
                                mod,
                                x.id,
                                "type_alias" if alias else "const",
                                n.lineno,
                                [],
                                "",
                                x.id.startswith("_"),
                            )
                        )
    return out


# ------------------------------------------------------------------ C++ index


class CIndex:
    """Symbols of a set of C/C++ files, looked up by name and by normalised key."""

    def __init__(self, tree: Tree, files: List[str]):
        self.tree = tree
        self.files = files
        self.syms: List[cscan.Sym] = []
        self.comments: Dict[str, List[cscan.Comment]] = {}
        for f in files:
            rel = tree.rel(f)
            s, c = cscan.scan_file(f, rel)
            self.syms += s
            self.comments[rel] = c
        self.by_name: Dict[str, List[cscan.Sym]] = {}
        self.by_qual: Dict[str, List[cscan.Sym]] = {}
        self.by_key: Dict[str, List[cscan.Sym]] = {}
        for s in self.syms:
            if s.kind == "namespace" or s.guard:
                continue
            self.by_name.setdefault(s.name, []).append(s)
            self.by_qual.setdefault(s.qualified, []).append(s)
            for k in c_keys(s):
                self.by_key.setdefault(k, []).append(s)

    def lookup(self, name: str) -> List[cscan.Sym]:
        """Symbols named `name` (bare or qualified)."""
        if "::" in name:
            return [
                s
                for s in self.syms
                if s.qualified == name or s.qualified.endswith("::" + name)
            ]
        return self.by_name.get(name, [])


_C_PREFIXES = ("rocke_attn_bwd_", "rocke_attn_", "rocke_")


def c_keys(s: cscan.Sym) -> List[str]:
    """Normalised keys a C symbol could be the twin of."""
    name = s.name
    lo = name.lower()
    keys = {lo}
    for p in ("rocke_attn_bwd_", "rocke_"):
        if lo.startswith(p):
            keys.add(lo[len(p) :])
    more = set()
    for k in keys:
        for suf in ("_t", "_new"):
            if k.endswith(suf) and len(k) > len(suf):
                more.add(k[: -len(suf)])
    keys |= more
    out = {norm(k) for k in keys}
    if s.in_struct and s.scope:
        owner = s.scope.split("::")[-1]
        out.add(norm(owner) + norm(name))
    return sorted(out)


def mod_suffix(module: str) -> str:
    """band for _attention_bwd_band, aux for attention_bwd_aux; '' for attention_bwd."""
    m = re.sub(r"^_?attention_(?:dense_)?bwd_?", "", module)
    return m


def py_keys(p: PySym) -> List[str]:
    base = p.name.lstrip("_")
    if p.kind in ("method", "property"):
        cls = p.cls.lstrip("_")
        m = p.name
        if m == "__init__":
            m = "init"
        elif m == "__post_init__":
            m = "validate"
        else:
            m = m.lstrip("_")
        cands = {norm(cls) + norm(m)}
        if cls.startswith("Bwd"):
            cands.add(norm(cls[3:]) + norm(m))
        if cls.startswith("AttnBwd"):
            cands.add(norm(cls[7:]) + norm(m))
        return sorted(cands)
    out = {norm(base)}
    if "ttention" in base.lower():
        out.add(norm(re.sub(r"(?i)attention", "attn", base)))
    if base.startswith("attn_bwd_"):
        out.add(norm(base[len("attn_bwd_") :]))
    suf = mod_suffix(p.module)
    if suf:
        out.add(norm(suf) + norm(base))
    if p.kind == "class":
        for pre in ("Bwd", "AttnBwd"):
            if base.startswith(pre):
                out.add(norm(base[len(pre) :]))
    return sorted(out)


_KIND_OK = {
    "function": {"function"},
    "class": {"typedef", "struct", "class", "enum", "union"},
    "method": {"function"},
    "property": {"function"},
    "const": {"macro", "variable", "enumerator"},
    "type_alias": {"typedef", "struct"},
}


# ------------------------------------------------------------------ module <-> C++ files


def module_stems(py_files: List[str]) -> List[str]:
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in py_files)


def cxx_base(path: str) -> str:
    b = os.path.basename(path)
    b = re.sub(r"\.(h|hpp|cpp|c)$", "", b)
    for p in ("helper_rocke.instances.common.", "instance_"):
        if b.startswith(p):
            b = b[len(p) :]
    return re.sub(r"_(internal|emit)$", "", b)


def module_of_cxx(path: str, stems: List[str]) -> Optional[str]:
    """The Python module a C++ file ports: the longest stem that prefixes its base name."""
    b = cxx_base(path)
    best = None
    for s in stems:
        if b == s or b.startswith(s + "_"):
            if best is None or len(s) > len(best):
                best = s
    return best


# ------------------------------------------------------------------ header tables


@dataclass
class TableRow:
    file: str
    line: int
    left: str
    right: str
    module_hint: str = ""
    parent_left: str = ""  # Python side of the row above, for an indented sub-row


_TABLE_HEAD = re.compile(r"^(\s*)(Python(?:\s*\(([^)]*)\))?)(\s{2,})(C\+\+|C)\b")


def _comment_lines(text: str) -> List[Tuple[int, str]]:
    """(line number, text without leading ` * `) for the lines of leading block comments."""
    out = []
    for i, raw in enumerate(text.split("\n"), 1):
        out.append((i, raw))
    return out


def parse_tables(path: str, rel: str) -> List[TableRow]:
    """Rows of the `Python (...)  C (...)` tables in a file's comments."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    rows: List[TableRow] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        body = (
            re.sub(r"^\s*(/\*+|\*|//)\s?", "", raw, count=1)
            if raw.strip().startswith(("*", "/*", "//"))
            else None
        )
        m = _TABLE_HEAD.match(body) if body is not None else None
        if not m:
            i += 1
            continue
        split = len(m.group(1)) + len(m.group(2)) + len(m.group(4))
        start = len(m.group(1))
        hint = m.group(3) or ""
        i += 1
        if i < len(lines) and re.match(
            r"^\s*(\*|//)?\s*-{3,}",
            re.sub(r"^\s*(/\*+|\*|//)\s?", "", lines[i], count=1),
        ):
            i += 1
        cur: Optional[TableRow] = None
        top: Optional[TableRow] = None  # the last row at the table's own indent
        while i < len(lines):
            raw = lines[i]
            if not raw.strip().startswith(("*", "//")):
                break
            body = re.sub(r"^\s*(\*|//)\s?", "", raw, count=1)
            if body.strip() == "" or body.strip() == "/":
                break
            if body.strip().endswith("*/") and body.strip() == "*/":
                break
            left = body[:split].strip() if len(body) > start else ""
            right = body[split:].strip() if len(body) > split else ""
            lead = len(body) - len(body.lstrip())
            new_row = bool(left) and lead <= start and not left.startswith("(")
            # an indented line starts a sub-row unless the row above is a list
            # that is still open (ends in `/` or `,`)
            open_list = cur is not None and cur.left.rstrip().endswith(("/", ","))
            sub_row = (
                bool(left)
                and lead > start
                and not left.startswith("(")
                and cur is not None
                and not open_list
            )
            if new_row or sub_row or cur is None:
                if cur is not None:
                    rows.append(cur)
                cur = TableRow(
                    rel, i + 1, left, right, hint, top.left if (sub_row and top) else ""
                )
                if not sub_row:
                    top = cur
            else:
                cur.left = (cur.left + " " + left).strip()
                cur.right = (cur.right + " " + right).strip()
            i += 1
        if cur is not None:
            rows.append(cur)
    return rows


def parse_arrows(path: str) -> List[Tuple[int, str, str]]:
    """`* A / B -> C` bullets in comments: (line, left, right). Wrapped bullets are joined."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    items: List[Tuple[int, str]] = []
    cur: Optional[List] = None
    for i, raw in enumerate(lines, 1):
        t = raw.strip()
        if not t.startswith(("*", "//")) or t.startswith("*/"):
            if cur:
                items.append((cur[0], cur[1]))
            cur = None
            continue
        body = re.sub(r"^(\*+|//)\s?", "", raw.strip(), count=1)
        m = re.match(r"^\s*[*-]\s+(.*)$", body)
        if m:
            if cur:
                items.append((cur[0], cur[1]))
            cur = [i, m.group(1).strip()]
        elif cur and body.strip() and len(body) - len(body.lstrip()) >= 2:
            cur[1] += " " + body.strip()
        else:
            if cur:
                items.append((cur[0], cur[1]))
            cur = None
    if cur:
        items.append((cur[0], cur[1]))
    out = []
    for ln, text in items:
        if "->" in text:
            l, r = text.split("->", 1)
            out.append((ln, l, r))
    return out


# ------------------------------------------------------------------ not-ported claims


@dataclass
class Claim:
    file: str
    line: int
    phrase: str  # "not ported" / "no twin" ...
    text: str  # the clause the claim covers
    tokens: List[str]  # identifier tokens in the clause


_NOT_PORTED_HEAD = re.compile(
    r"\bnot\s+ported(?:\s+yet)?(?:\s*\([^)]*\))?\s*[:\-]", re.I
)
_HOST_ONLY_HEAD = re.compile(
    r"(?:^|\.\s+)(?:Host-only\b[^:.]{0,40}|[A-Z][\w -]{0,30}\bstays?\s+(?:in\s+)?Python)\s*:"
)
_NOT_PORTED_INLINE = re.compile(
    r"\b(?:is|are|stays?|remains?|has|have)?\s*(?:not\s+ported|no\s+(?:C\+*\s+)?(?:twin|object|port|counterpart)(?:\s+yet)?)",
    re.I,
)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def _paragraphs(comments: List[cscan.Comment]) -> List[Tuple[int, str]]:
    """Comment paragraphs: (first line, text) with comment decoration removed."""
    paras: List[Tuple[int, str]] = []
    for c in comments:
        cur: List[str] = []
        first = c.line
        for k, raw in enumerate(c.text.split("\n")):
            t = re.sub(r"^\s*\*+ ?", "", raw) if c.block else raw.strip()
            if t.strip() == "":
                if cur:
                    paras.append((first, " ".join(cur)))
                    cur = []
                continue
            if not cur:
                first = c.line + k
            cur.append(t.strip())
        if cur:
            paras.append((first, " ".join(cur)))
    # adjacent // comment lines are separate Comment objects; merge neighbours
    merged: List[Tuple[int, str]] = []
    for ln, t in paras:
        if merged and ln == merged[-1][0] + merged[-1][1].count("\x00") + 1 and False:
            pass
        merged.append((ln, t))
    return merged


_LEAD = {"and", "the", "a", "an", "its", "their", "also", "of", "only", "both", "or"}


def _heading_items(clause: str) -> List[str]:
    """First symbol-like token of each item in a `Not ported: A, B (why), C` list."""
    toks = []
    for item in re.split(r";|,|/|\band\b|\bor\b", clause):
        if re.search(r"\b(constructor|ctor)\b", item):
            continue  # a claim about the constructor, not the type
        if re.match(r"\s*(?:the\s+)?[A-Za-z_][\w.]*'s\b", item):
            continue  # `X's optional check`: a part of X is not ported, not X
        words = [w for w in re.findall(r"[A-Za-z_][\w.]*", item)]
        while words and words[0].lower() in _LEAD:
            words = words[1:]
        if words:
            toks.append(words[0])
    return toks


def _trailing_names(before: str) -> List[str]:
    """The names a `... X, Y and Z are not ported` clause is about (the list before the phrase)."""
    m = re.search(
        r"((?:[A-Za-z_][\w.]*)(?:\s*(?:,|/|\band\b|\bor\b)\s*[A-Za-z_][\w.]*)*)\s*(?:\([^)]*\))?\s*$",
        before.rstrip(),
    )
    if not m:
        return []
    return [
        t for t in re.findall(r"[A-Za-z_][\w.]*", m.group(1)) if t.lower() not in _LEAD
    ]


def parse_claims(index_comments: List[cscan.Comment], rel: str) -> List[Claim]:
    """Statements in comments that something is not ported / has no twin.

    A `Not ported: A, B (why), C.` heading claims the first name of each item.
    An inline `X is not ported` / `X (no C twin yet)` claims the names right
    before the phrase.
    """
    claims: List[Claim] = []
    # merge consecutive line comments so a claim split over lines is read whole
    merged: List[cscan.Comment] = []
    for c in index_comments:
        if (
            merged
            and not c.block
            and not merged[-1].block
            and c.line == merged[-1].end_line + 1
        ):
            m = merged[-1]
            merged[-1] = cscan.Comment(
                m.line, c.end_line, m.text + "\n" + c.text, False, m.level
            )
        else:
            merged.append(c)
    for ln, para in _paragraphs(merged):
        m = _NOT_PORTED_HEAD.search(para)
        if m:
            rest = para[m.end() :]
            end = re.search(r"\.(?:\s|$)(?=[A-Z]|$)", rest)
            clause = rest[: end.start()] if end else rest
            clause = re.sub(r"\([^)]*\)", " ", clause)
            # "X has no C twin of its own" items keep only their subject
            claims.append(
                Claim(rel, ln, "not ported", clause.strip(), _heading_items(clause))
            )
            continue
        mh = _HOST_ONLY_HEAD.search(para)
        if mh:
            rest = para[mh.end() :]
            end = re.search(r"\.(?:\s|$)(?=[A-Z]|$)", rest)
            clause = rest[: end.start()] if end else rest
            names = [
                t
                for t in _TOKEN.findall(clause)
                if "_" in t or any(ch.isupper() for ch in t)
            ]
            claims.append(Claim(rel, ln, "host-only", clause.strip(), names))
            continue
        for m in _NOT_PORTED_INLINE.finditer(para):
            before = para[: m.start()]
            s = max(before.rfind(". "), before.rfind("; "), -1)
            clause = before[s + 1 :]
            claims.append(
                Claim(
                    rel, ln, m.group(0).strip(), clause.strip(), _trailing_names(clause)
                )
            )
    return claims


# ------------------------------------------------------------------ twin map


_TOKEN_DOT = re.compile(r"\.?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def _row_tokens_left(left: str) -> Tuple[List[str], List[str]]:
    """(main tokens, '(+ ...)' extras) from the Python side of a table row."""
    extras = []
    for m in re.finditer(r"\(\+\s*([^)]*)\)", left):
        extras += _TOKEN_DOT.findall(m.group(1))
    main = re.sub(r"\([^)]*\)", " ", left)
    toks = [t for t in _TOKEN_DOT.findall(main)]
    return toks, extras


def _row_tokens_right(right: str) -> Tuple[List[str], List[str]]:
    extras = []
    for m in re.finditer(r"\(\+\s*([^)]*)\)", right):
        extras += re.findall(r"[A-Za-z_][\w:]*\*?", m.group(1))
    main = re.sub(r"\([^)]*\)", " ", right)
    toks = re.findall(r"[A-Za-z_][\w:]*\*?", main)
    return toks, extras


def _is_c_token(t: str) -> bool:
    return bool(re.match(r"^(rocke_|ROCKE_|_[a-z]|[A-Za-z_]\w*::)", t))


_CONNECTORS = {"and", "or", "plus"}


def _looks_like_identifier(t: str) -> bool:
    return "_" in t.strip("_") or any(ch.isdigit() for ch in t)


def _expand_c_tokens(tokens: List[str], index: CIndex) -> List[Tuple[str, bool]]:
    """Resolve `_suffix` shorthand against the previous full name.

    Returns (name, exists-in-index) pairs. Reading stops at the first prose
    word: a bare token (no rocke_/ROCKE_/namespace prefix) counts as a C name
    only when it looks like an identifier (has `_` or a digit) or is the first
    token of the cell; any other bare word ends the list, so the prose after
    `rocke_x_t, one struct with ...` never becomes a twin.
    """
    out: List[Tuple[str, bool]] = []
    prev = ""
    for t in tokens:
        if not _is_c_token(t):
            if t.lower() in _CONNECTORS:
                continue
            if not (_looks_like_identifier(t) or not out):
                break
            if not index.lookup(t):
                continue
        name = t
        if t.startswith("_") and prev:
            parts = prev.split("_")
            found = ""
            for k in range(len(parts) - 1, 0, -1):
                cand = "_".join(parts[:k]) + t
                if index.lookup(cand):
                    found = cand
                    break
            if found:
                name = found
        if not t.startswith("_"):
            prev = t
        out.append((name, bool(index.lookup(name))))
    return out


@dataclass
class Twin:
    symbol: str
    file: str
    line: int
    kind: str
    evidence: str  # table, table-group, comment, convention
    definition: str = ""  # file:line of the definition when it is not the listed one


def build_twin_map(tree: Tree, parity: bool = False) -> dict:
    py_files = tree.bwd_py()
    stems = module_stems(py_files)
    cxx = tree.bwd_cxx(parity)
    index = CIndex(tree, cxx)
    mod_files: Dict[str, List[str]] = {s: [] for s in stems}
    for f in cxx:
        m = module_of_cxx(f, stems)
        if m:
            mod_files[m].append(tree.rel(f))

    pysyms: Dict[str, List[PySym]] = {}
    for f in py_files:
        pysyms[os.path.splitext(os.path.basename(f))[0]] = py_symbols(f)
    by_simple: Dict[str, List[PySym]] = {}
    by_qual: Dict[str, List[PySym]] = {}
    for syms in pysyms.values():
        for s in syms:
            by_simple.setdefault(s.name, []).append(s)
            by_qual.setdefault(s.qual, []).append(s)

    twins: Dict[str, List[Twin]] = {}
    claimed_missing: Dict[str, List[str]] = {}
    claimed_at: Dict[str, Dict[str, list]] = {}
    elsewhere: Dict[str, List[str]] = {}

    def add_twin(p: PySym, c: cscan.Sym, ev: str):
        lst = twins.setdefault(p.key, [])
        for t in lst:
            if t.symbol == c.qualified and t.kind == c.kind:
                # one entry per C symbol: the header declaration, with the definition noted
                if (
                    c.is_def
                    and (t.file, t.line) != (c.file, c.line)
                    and not t.definition
                ):
                    if t.file.endswith(".cpp") or not t.file.endswith((".h", ".hpp")):
                        pass
                    t.definition = f"{c.file}:{c.line}"
                elif (
                    not c.is_def
                    and t.file.endswith(".cpp")
                    and c.file.endswith((".h", ".hpp"))
                ):
                    t.definition = f"{t.file}:{t.line}"
                    t.file, t.line = c.file, c.line
                return
        lst.append(
            Twin(
                c.qualified,
                c.file,
                c.line,
                c.kind,
                ev,
                f"{c.file}:{c.line}" if c.is_def and c.file.endswith(".cpp") else "",
            )
        )

    # 1. header tables and `A / B -> C` bullets
    def pick_sym(name):
        syms = index.lookup(name)
        if not syms:
            return None
        return sorted(syms, key=lambda s: (not s.is_def, s.file != syms[0].file))[0]

    def name_match(p, name):
        """True when the C name is the one the naming convention gives p."""
        c = pick_sym(name)
        return bool(c) and bool(set(py_keys(p)) & set(c_keys(c)))

    def claim(p, name, at):
        if re.match(r"^(rocke_|ROCKE_)\w*[A-Za-z0-9]$", name) or "::" in name:
            claimed_missing.setdefault(p.key, []).append(name)
            claimed_at.setdefault(p.key, {}).setdefault(name, list(at))

    def link_one(p, name, ok, ev, at):
        c = pick_sym(name) if ok else None
        if c:
            add_twin(p, c, ev)
        else:
            claim(p, name, at)

    def link(p_list, cs, at):
        """Pair the Python symbols of a row with its C names.

        Equal counts zip. One C name is shared by every Python symbol (a kind
        tag struct, a merged helper). One Python symbol takes every C name (a
        struct and its _init). Anything else pairs only by the naming
        convention inside the row: no guessing, no broadcast.
        """
        n_p, n_c = len(p_list), len(cs)
        if n_p == n_c and n_p > 1:
            for p, (name, ok) in zip(p_list, cs):
                link_one(p, name, ok, "table", at)
        elif n_c == 1:
            for p in p_list:
                link_one(p, cs[0][0], cs[0][1], "table", at)
        elif n_p == 1:
            p = p_list[0]
            named = [c for c in cs if name_match(p, c[0])]
            for name, ok in cs:
                ev = "table" if (name, ok) in named or len(cs) == 1 else "table-group"
                link_one(p, name, ok, ev, at)
        else:
            for p in p_list:
                for name, ok in cs:
                    if ok and name_match(p, name):
                        link_one(p, name, ok, "table", at)

    def wildcard_link(p_list, wild):
        """`rocke_attn_bwd_band_*`: every Python symbol whose key matches a C name with that prefix."""
        for w in wild:
            pre = w[:-1]
            matches = [
                s
                for s in index.syms
                if s.name.startswith(pre)
                and s.kind in ("function", "macro", "variable", "typedef", "struct")
            ]
            for p in p_list:
                keys = set(py_keys(p))
                for s in matches:
                    if keys & set(c_keys(s)):
                        add_twin(p, s, "table")

    def py_tokens(left: str, mod: str, parent: str = ""):
        ltoks, lextra = _row_tokens_left(left)
        pys: List[PySym] = []
        main_cls = ""
        if parent:  # an indented sub-row works on the class of the row above
            for t in _row_tokens_left(parent)[0]:
                hit = [
                    s
                    for s in pysyms.get(mod, [])
                    if s.qual == t.lstrip(".") and s.kind == "class"
                ]
                if hit:
                    main_cls = hit[0].qual
                    break
        for t in ltoks + lextra:
            if t.startswith("."):  # `.method`: a method of the class in play
                c0 = (
                    [
                        s
                        for s in pysyms.get(mod, [])
                        if s.cls == main_cls and s.name == t[1:]
                    ]
                    if main_cls
                    else []
                )
                if c0:
                    pys.append(c0[0])
                continue
            if "." in t:
                c0 = [s for s in by_qual.get(t, []) if s.module == mod] or by_qual.get(
                    t, []
                )
                if not c0:
                    c0 = [
                        s
                        for s in by_simple.get(t.split(".")[-1], [])
                        if s.module == mod and s.cls == t.split(".")[0]
                    ]
            else:
                c0 = [s for s in pysyms.get(mod, []) if s.qual == t]
                if not c0 and main_cls:
                    c0 = [
                        s
                        for s in pysyms.get(mod, [])
                        if s.cls == main_cls and s.name == t
                    ]
            if c0:
                pys.append(c0[0])
                if c0[0].kind == "class" and not main_cls:
                    main_cls = c0[0].qual
                elif "." in t and c0[0].cls:
                    main_cls = c0[0].cls  # `Class.method / .other`
        return pys

    for f in cxx:
        rel = tree.rel(f)
        mod = module_of_cxx(f, stems)
        if mod is None:
            continue
        for row in parse_tables(f, rel):
            pys = py_tokens(row.left, mod, row.parent_left)
            rtoks, rextra = _row_tokens_right(row.right)
            alltoks = rtoks + rextra
            wild = [w for w in alltoks if w.endswith("*")]
            cs = _expand_c_tokens([w for w in alltoks if not w.endswith("*")], index)
            if pys and cs:
                link(pys, cs, (rel, row.line))
            if pys:
                wildcard_link(pys, wild)
        for line, left, right in parse_arrows(f):
            pys = py_tokens(left, mod)
            if not pys:
                continue
            wild = [w for w in re.findall(r"[A-Za-z_]\w*\*", right)]
            plain = [
                w
                for w in re.findall(
                    r"[A-Za-z_][\w:]*", re.sub(r"[A-Za-z_]\w*\*", " ", right)
                )
            ]
            cs = _expand_c_tokens(plain, index)
            if cs:
                link(pys, cs, (rel, line))
            wildcard_link(pys, wild)

    # 2. Python:/py: comments that sit at declaration level, next to a definition
    cmt_re = re.compile(r"\b(?:Python|py)\s*:\s*([A-Za-z_][\w.]*)")
    for f in cxx:
        rel = tree.rel(f)
        mod = module_of_cxx(f, stems)
        if mod is None:
            continue
        fsyms = sorted(
            [s for s in index.syms if s.file == rel and s.kind == "function"],
            key=lambda s: s.line,
        )
        for c in index.comments.get(rel, []):
            if c.level != "top":
                continue
            m = cmt_re.search(c.text)
            if not m:
                continue
            tok = m.group(1)
            cands = [s for s in pysyms.get(mod, []) if s.qual == tok or s.name == tok]
            nxt = [
                s for s in fsyms if s.line >= c.end_line and s.line - c.end_line <= 3
            ]
            if len(cands) == 1 and nxt:
                add_twin(cands[0], nxt[0], "comment")

    # 3. naming conventions
    for mod, syms in pysyms.items():
        own = set(mod_files.get(mod, []))
        for p in syms:
            ok_kinds = _KIND_OK[p.kind]
            cands: List[cscan.Sym] = []
            for k in py_keys(p):
                cands += [s for s in index.by_key.get(k, []) if s.kind in ok_kinds]
            # methods may also be struct members named by their bare method name
            if p.kind in ("method", "property") and p.cls:
                owner_norm = {norm(p.cls.lstrip("_"))}
                for s in index.by_name.get(p.name.lstrip("_"), []):
                    if (
                        s.in_struct
                        and s.kind == "function"
                        and norm(s.scope.split("::")[-1]) in owner_norm
                    ):
                        cands.append(s)
            seen = set()
            uniq = []
            for s in cands:
                if (s.file, s.line, s.qualified) not in seen:
                    seen.add((s.file, s.line, s.qualified))
                    uniq.append(s)
            mine = [s for s in uniq if s.file in own]
            if mine:
                # a C function with the convention's name beats a table pick that
                # points somewhere else
                names = {s.qualified for s in mine}
                twins[p.key] = [
                    t
                    for t in twins.get(p.key, [])
                    if t.symbol in names or t.evidence != "table"
                ]
            for s in mine:
                # a struct tag next to the table's `<tag>_t` typedef is the same type
                if any(t.symbol == s.qualified + "_t" for t in twins.get(p.key, [])):
                    continue
                add_twin(p, s, "convention")
            if not mine and uniq and p.kind not in ("method", "property"):
                elsewhere[p.key] = [
                    f"{s.qualified} ({s.file}:{s.line})" for s in uniq[:4]
                ]

    # 3a. `Class.method(args) ...` comments above a global declaration link a method that has no
    # twin yet (conventions and tables win; namespace-scope helpers are not method twins)
    meth_re = re.compile(r"^\s*([A-Za-z_]\w*\.[A-Za-z_]\w*)\s*\(")
    for f in cxx:
        rel = tree.rel(f)
        mod = module_of_cxx(f, stems)
        if mod is None:
            continue
        fsyms = sorted(
            [
                s
                for s in index.syms
                if s.file == rel
                and s.kind == "function"
                and not s.in_struct
                and not s.scope
            ],
            key=lambda s: s.line,
        )
        for c in index.comments.get(rel, []):
            if c.level != "top":
                continue
            m = meth_re.match(c.text)
            if not m:
                continue
            cands = [
                s
                for s in pysyms.get(mod, [])
                if s.qual == m.group(1) and s.kind in ("method", "property")
            ]
            nxt = [
                s for s in fsyms if s.line >= c.end_line and s.line - c.end_line <= 2
            ]
            if len(cands) == 1 and nxt and not twins.get(cands[0].key):
                add_twin(cands[0], nxt[0], "comment")

    # 3b. methods of a class folded into a C struct: try `<struct prefix>_<method>`,
    # e.g. MaskPolicy -> rocke_attn_bwd_mask_policy_t gives rocke_attn_bwd_mask_<method>
    for mod, syms in pysyms.items():
        own = set(mod_files.get(mod, []))
        for p in syms:
            if p.kind not in ("method", "property") or not p.cls or twins.get(p.key):
                continue
            ctw = [
                t
                for t in twins.get(f"{mod}:{p.cls}", [])
                if t.kind in ("struct", "typedef", "class")
            ]
            if not ctw:
                continue
            m = "init" if p.name == "__init__" else p.name.lstrip("_")
            found = None
            for t in ctw:
                parts = re.sub(r"_t$", "", t.symbol.split("::")[-1]).split("_")
                for k in range(len(parts), 1, -1):
                    cand = "_".join(parts[:k]) + "_" + m
                    hits = [
                        s
                        for s in index.by_name.get(cand, [])
                        if s.kind == "function" and not s.in_struct and s.file in own
                    ]
                    if hits:
                        found = sorted(
                            hits, key=lambda s: (s.file.endswith(".cpp"), not s.is_def)
                        )[0]
                        break
                if found:
                    break
            if found:
                add_twin(p, found, "convention")

    # 4. claims
    claims_by_file: Dict[str, List[Claim]] = {}
    for f in cxx:
        rel = tree.rel(f)
        claims_by_file[rel] = parse_claims(index.comments.get(rel, []), rel)
    declared: Dict[str, List[dict]] = {}
    for f in cxx:
        rel = tree.rel(f)
        mod = module_of_cxx(f, stems)
        for cl in claims_by_file[rel]:
            for tok in cl.tokens:
                for p in claim_symbols(tok, mod, pysyms, by_qual):
                    declared.setdefault(p.key, []).append(
                        {"file": rel, "line": cl.line, "phrase": cl.phrase}
                    )

    for mod, syms in pysyms.items():
        for p in syms:
            if p.cls and f"{mod}:{p.cls}" in declared and p.key not in declared:
                declared[p.key] = [
                    dict(d, via="class") for d in declared[f"{mod}:{p.cls}"]
                ]

    result = {"modules": {}, "symbols": {}}
    for mod in stems:
        result["modules"][mod] = {
            "python": tree.rel(os.path.join(tree.kernels, mod + ".py")),
            "cxx_files": sorted(mod_files[mod]),
        }
    for mod, syms in pysyms.items():
        for p in syms:
            tw = twins.get(p.key, [])
            entry = {
                "module": mod,
                "name": p.qual,
                "kind": p.kind,
                "line": p.line,
                "private": p.private,
                "status": "ported" if tw else "not ported",
                "twins": [
                    t.__dict__
                    for t in sorted(tw, key=lambda t: (t.file, t.line, t.symbol))
                ],
            }
            if not tw:
                entry["reason"] = (
                    "module has no C++ file" if not mod_files[mod] else "no twin found"
                )
            if p.key in declared:
                entry["declared_not_ported"] = declared[p.key]
            if p.key in elsewhere and not tw:
                entry["same_name_elsewhere"] = elsewhere[p.key]
            if p.key in claimed_missing:
                entry["claimed_missing"] = sorted(set(claimed_missing[p.key]))
                entry["claimed_missing_at"] = {
                    n: claimed_at[p.key][n] for n in entry["claimed_missing"]
                }
            result["symbols"][p.key] = entry
    return result


def claim_symbols(
    tok: str, mod: Optional[str], pysyms: Dict[str, List[PySym]], by_qual
) -> List[PySym]:
    """Python symbols a comment token names, in the file's own module first."""
    parts = tok.split(".")
    cands: List[PySym] = []
    # try the longest dotted suffix that names a symbol: a.b.Class.method, Class.method, name
    for k in range(min(2, len(parts)), 0, -1):
        q = ".".join(parts[-k:])
        cands = list(by_qual.get(q, []))
        if cands:
            break
    if not cands:
        return []
    own = [c for c in cands if c.module == mod]
    if own:
        return own
    # a token naming a symbol of another module counts only when it is dotted (qualified)
    return cands if len(cands) == 1 and "." in tok else []
