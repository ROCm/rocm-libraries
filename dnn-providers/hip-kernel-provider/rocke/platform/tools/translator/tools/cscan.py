# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A small C/C++ declaration scanner (no compiler needed).

It lexes out comments and string literals, then walks the statements of a file
and reports the names it declares or defines: functions, variables, typedefs,
structs, enums and enumerators, namespaces and macros. It is a regex-level
tool, tuned for the rocKE headers and instances; it does not parse templates
or expressions.

    from cscan import scan_text
    syms, comments = scan_text(text, "x.h")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Sym:
    name: str  # bare name
    qualified: str  # namespace/struct scope + name
    kind: str  # function, variable, typedef, struct, enum, enumerator, namespace, macro
    file: str
    line: int
    end_line: int = 0
    is_def: bool = True  # False for a prototype / extern declaration / forward decl
    linkage: str = "external"  # external or internal (static, anonymous ns, const)
    scope: str = ""  # qualifier part of `qualified`
    in_struct: bool = False  # member of a struct/class body
    guard: bool = False  # macro that is an include guard
    params: str = ""  # raw parameter text of a function
    ret: str = ""  # raw return-type text of a function
    value: str = ""  # macro body or variable initializer, whitespace-collapsed


@dataclass
class Comment:
    line: int  # first line
    end_line: int
    text: str  # without the comment markers
    block: bool
    level: str = ""  # "body" inside a function body, else "top"


def lex(text: str) -> Tuple[str, List[Comment]]:
    """Blank comments and string/char literals; return (code, comments).

    String contents become 'S' so an `extern "C"` block can still be spotted.
    Newlines are kept, so line numbers match the input.
    """
    out: List[str] = []
    comments: List[Comment] = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            # a line comment continued by a backslash is rare; ignore that
            comments.append(Comment(line, line, text[i + 2 : j], False))
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            body = text[i:j]
            nl = body.count("\n")
            comments.append(
                Comment(
                    line,
                    line + nl,
                    body[2:-2] if body.endswith("*/") else body[2:],
                    True,
                )
            )
            out.append(re.sub(r"[^\n]", " ", body))
            line += nl
            i = j
        elif c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 1
                if j < n and text[j] == "\n":
                    break
                j += 1
            j = min(j + 1, n)
            lit = text[i:j]
            if c == '"' and lit.startswith('"') and lit.endswith('"') and len(lit) >= 2:
                out.append('"' + "S" * (len(lit) - 2) + '"')
            else:
                out.append(c + "S" * (len(lit) - 2) + c if len(lit) >= 2 else lit)
            i = j
        else:
            if c == "\n":
                line += 1
            out.append(c)
            i += 1
    return "".join(out), comments


_PP = re.compile(r"^[ \t]*#[ \t]*(\w+)[ \t]*(.*)$")


def _take_preprocessor(code: str):
    """Blank preprocessor lines; return (code, [(line, directive, rest)])."""
    lines = code.split("\n")
    pp = []
    i = 0
    while i < len(lines):
        m = _PP.match(lines[i])
        if not m:
            i += 1
            continue
        start = i
        rest = m.group(2)
        while rest.rstrip().endswith("\\") and i + 1 < len(lines):
            rest = rest.rstrip()[:-1] + " " + lines[i + 1].strip()
            i += 1
            lines[i] = ""
        pp.append((start + 1, m.group(1), rest.strip()))
        lines[start] = ""
        i += 1
    return "\n".join(lines), pp


_IDENT = r"[A-Za-z_]\w*"
_NS_RE = re.compile(
    r"^(?:inline\s+)?namespace\s*(" + _IDENT + r"(?:::" + _IDENT + r")*)?\s*$"
)
_EXTERN_RE = re.compile(r'^extern\s*"S+"\s*$')
_AGG_RE = re.compile(
    r"^(typedef\s+)?(?:(?:alignas\s*\([^)]*\)|__attribute__\s*\(\(.*?\)\))\s*)*"
    r"(struct|union|class|enum\s+class|enum\s+struct|enum)\s*"
    r"(?:(?:alignas\s*\([^)]*\))\s*)?(" + _IDENT + r")?\s*(?::\s*[\w:\s,<>]+)?\s*$"
)
_ACCESS = re.compile(r"^(?:public|private|protected)\s*:\s*")


def _line_of(starts: List[int], pos: int) -> int:
    lo, hi = 0, len(starts)
    while lo < hi:
        mid = (lo + hi) // 2
        if starts[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _match_close(code: str, i: int, open_c: str, close_c: str) -> int:
    d = 0
    n = len(code)
    while i < n:
        c = code[i]
        if c == open_c:
            d += 1
        elif c == close_c:
            d -= 1
            if d == 0:
                return i
        i += 1
    return n - 1


def _split_top(s: str, sep: str = ",") -> List[str]:
    parts, d, cur = [], 0, []
    for c in s:
        if c in "([{<":
            d += 1
        elif c in ")]}>":
            d -= 1
        if c == sep and d == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


_KW_NOT_NAME = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "operator",
    "static_assert",
    "alignas",
    "decltype",
    "noexcept",
    "__attribute__",
    "defined",
}


def _func_name(head: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (name, qualified-extra, ret, params) if head declares a function."""
    d = 0
    for i, c in enumerate(head):
        if c == "=" and d == 0:
            return None  # a variable with an initializer
        if c == "(":
            pre = head[:i].rstrip()
            m = re.search(
                r"((?:~?" + _IDENT + r"::)*(?:~?" + _IDENT + r"|operator\s*\S+?))$", pre
            )
            if not m:
                # (*fnptr)(...) style or a stray paren
                return None
            qn = m.group(1)
            name = qn.split("::")[-1]
            if name in _KW_NOT_NAME:
                # an attribute or macro wrapping; keep scanning past it
                end = _match_close(head, i, "(", ")")
                sub = head[end + 1 :]
                r = _func_name(pre + " " + sub) if sub.strip() else None
                return r
            end = _match_close(head, i, "(", ")")
            return (
                name,
                "::".join(qn.split("::")[:-1]),
                pre[: m.start(1)].strip(),
                head[i + 1 : end],
            )
    return None


def _var_names(head: str) -> List[Tuple[str, str]]:
    """Declarator names of a variable declaration: [(name, initializer)]."""
    out = []
    # remove the type part before the first declarator is hard; take each
    # comma-separated piece and look at the identifier before `=`, `[` or the end.
    for piece in _split_top(head):
        init = ""
        lhs = piece
        if "=" in piece:
            lhs, init = piece.split("=", 1)
        elif "{" in piece:
            lhs, init = piece.split("{", 1)
        lhs = re.sub(r"\[[^\]]*\]", "", lhs).strip()
        m = re.search(r"(" + _IDENT + r")\s*$", lhs)
        if m and m.group(1) not in ("const", "static", "extern", "volatile"):
            out.append((m.group(1), init.strip()))
    return out


def _strip_template(h: str) -> str:
    while h.startswith("template"):
        j = h.find("<")
        if j < 0:
            break
        d, k = 0, j
        while k < len(h):
            if h[k] == "<":
                d += 1
            elif h[k] == ">":
                d -= 1
                if d == 0:
                    break
            k += 1
        h = h[k + 1 :].lstrip()
    return h


def scan_text(text: str, path: str = "<mem>") -> Tuple[List[Sym], List[Comment]]:
    code, comments = lex(text)
    code, pp = _take_preprocessor(code)
    syms: List[Sym] = []
    starts = [0]
    for m in re.finditer("\n", code):
        starts.append(m.end())

    # macros; an #ifndef X directly followed by #define X is an include guard
    for idx, (ln, d, rest) in enumerate(pp):
        if d != "define":
            continue
        m = re.match(r"(" + _IDENT + r")(\([^)]*\))?\s*(.*)$", rest)
        if not m:
            continue
        guard = False
        if (
            idx > 0
            and pp[idx - 1][1] == "ifndef"
            and pp[idx - 1][2].split()[:1] == [m.group(1)]
        ):
            guard = m.group(3) == "" and m.group(2) is None
        syms.append(
            Sym(
                m.group(1),
                m.group(1),
                "macro",
                path,
                ln,
                ln,
                True,
                "external",
                "",
                False,
                guard,
                m.group(2) or "",
                "",
                re.sub(r"\s+", " ", m.group(3)),
            )
        )

    # scope stack: (kind, name, extra); kinds: namespace, extern, struct
    stack: List[tuple] = []

    def scope_qual() -> str:
        return "::".join(
            e[1] for e in stack if e[0] in ("namespace", "struct") and e[1]
        )

    def in_anon() -> bool:
        return any(e[0] == "namespace" and e[1] == "" for e in stack)

    def in_struct() -> bool:
        return bool(stack) and stack[-1][0] == "struct"

    def add(name, kind, line, end_line, is_def, linkage, qual_extra="", **kw):
        q = scope_qual()
        scope = "::".join(x for x in (q, qual_extra) if x)
        if in_anon():
            linkage = "internal"
        syms.append(
            Sym(
                name,
                "::".join(x for x in (scope, name) if x),
                kind,
                path,
                line,
                end_line,
                is_def,
                linkage,
                scope,
                in_struct(),
                **kw,
            )
        )

    pending = [None]  # (kind, name, typedef, line) of an aggregate just closed

    def statement(head: str, hstart: int, term: str, end_pos: int):
        h = _ACCESS.sub("", head.strip())
        pa, pending[0] = pending[0], None
        if not h:
            return
        line = _line_of(starts, hstart + (len(head) - len(head.lstrip())))
        end_line = _line_of(starts, end_pos)
        if pa is not None:
            kind, name, is_td, aline = pa
            for d in _split_top(h):
                x = d.strip().lstrip("*& ").split("[")[0].strip()
                if re.fullmatch(_IDENT, x):
                    add(
                        x,
                        "typedef" if is_td else "variable",
                        aline,
                        end_line,
                        True,
                        "external",
                    )
            return
        h = _strip_template(h)
        if h.startswith("using "):
            m = re.match(r"using\s+(" + _IDENT + r")\s*=", h)
            if m:
                add(m.group(1), "typedef", line, end_line, True, "external")
            return
        if h.startswith("typedef"):
            m = re.search(r"\(\s*\*\s*(" + _IDENT + r")\s*\)", h)
            if not m:
                m = re.search(r"(" + _IDENT + r")\s*$", re.sub(r"\[[^\]]*\]", "", h))
            if m:
                add(m.group(1), "typedef", line, end_line, True, "external")
            return
        if h.startswith(("static_assert", "friend")):
            return
        is_static = bool(re.search(r"\bstatic\b", h))
        is_extern = bool(re.search(r"\bextern\b", h))
        is_const = bool(re.search(r"\bconst(expr)?\b", h))
        fn = _func_name(h)
        if fn is not None:
            name, qx, ret, params = fn
            link = "internal" if (is_static or in_anon()) else "external"
            add(
                name,
                "function",
                line,
                end_line,
                term == "{",
                link,
                qx,
                params=re.sub(r"\s+", " ", params).strip(),
                ret=re.sub(r"\s+", " ", ret).strip(),
            )
            return
        if term == "{":
            return
        m = re.match(
            r"^(struct|union|class|enum(?:\s+class)?)\s+(" + _IDENT + r")\s*$", h
        )
        if m:
            add(m.group(2), m.group(1).split()[0], line, end_line, False, "external")
            return
        if in_struct():
            return  # data members are not tracked
        for name, init in _var_names(h):
            link = (
                "internal"
                if (is_static or in_anon() or (is_const and not is_extern))
                else "external"
            )
            add(
                name,
                "variable",
                line,
                end_line,
                (not is_extern) or "=" in h,
                link,
                value=re.sub(r"\s+", " ", init),
            )

    n = len(code)
    i = 0
    head_start = 0
    while i < n:
        c = code[i]
        if c in "([":
            i = _match_close(code, i, c, ")" if c == "(" else "]") + 1
            continue
        if c == ";":
            statement(code[head_start:i], head_start, ";", i)
            i += 1
            head_start = i
            continue
        if c == "}":
            if stack:
                e = stack.pop()
                if e[0] == "struct":
                    pending[0] = (e[2], e[1], e[3], e[4])
            i += 1
            head_start = i
            continue
        if c != "{":
            i += 1
            continue
        head = code[head_start:i]
        h = _strip_template(_ACCESS.sub("", head.strip()))
        hl = _line_of(starts, head_start + (len(head) - len(head.lstrip())))
        mns = _NS_RE.match(h)
        if mns is not None:
            nm = mns.group(1) or ""
            if nm:
                add(nm, "namespace", hl, hl, True, "external")
            stack.append(("namespace", nm))
            i += 1
            head_start = i
            continue
        if _EXTERN_RE.match(h):
            stack.append(("extern", ""))
            i += 1
            head_start = i
            continue
        magg = _AGG_RE.match(h) if h else None
        if magg:
            is_td = bool(magg.group(1))
            kw = magg.group(2).split()[0]
            nm = magg.group(3) or ""
            close = _match_close(code, i, "{", "}")
            if nm:
                add(nm, kw, hl, _line_of(starts, close), True, "external")
            if kw == "enum":
                body = code[i + 1 : close]
                pos = 0
                for piece in _split_top(body):
                    mm = re.match(r"(" + _IDENT + r")", piece.strip())
                    off = body.find(piece, pos) if piece else pos
                    pos = off + len(piece)
                    if mm:
                        ln = _line_of(
                            starts, i + 1 + off + (len(piece) - len(piece.lstrip()))
                        )
                        add(mm.group(1), "enumerator", ln, ln, True, "external")
                pending[0] = (kw, nm, is_td, hl)
                i = close + 1
                head_start = i
                continue
            stack.append(
                ("struct", nm, nm, is_td, hl)
                if False
                else ("struct", nm, kw, is_td, hl)
            )
            i += 1
            head_start = i
            continue
        close = _match_close(code, i, "{", "}")
        fn = _func_name(h) if h else None
        if fn is not None:
            statement(head, head_start, "{", close)
            i = close + 1
            head_start = i
            continue
        i = close + 1  # an initializer; the statement runs on to its ';'

    # comments inside a function body are marked so callers can tell them from
    # declaration-level comments
    bodies = [
        (s.line, s.end_line)
        for s in syms
        if s.kind == "function" and s.is_def and s.end_line > s.line
    ]
    for cm in comments:
        cm.level = (
            "body"
            if any(a < cm.line and cm.end_line <= b for a, b in bodies)
            else "top"
        )
    return syms, comments


def scan_file(path: str, shown: str = "") -> Tuple[List[Sym], List[Comment]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return scan_text(f.read(), shown or path)
