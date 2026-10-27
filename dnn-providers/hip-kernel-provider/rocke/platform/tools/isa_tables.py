#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx11 instruction description behind the direct compiler.

``rust/rocke-codeobj/isa/gfx11.isa`` is one declarative description of every
instruction the direct compiler (``rust/rocke-codeobj``) emits: mnemonic,
encoding format and opcode, operand classes and sizes, modifiers, implicit
registers, execution unit, latency class, hazard participation and VOPD form,
plus the encoding layouts and operand codes they share and the BURS grammar
for integer and address arithmetic. The Rust tables the compiler uses are
generated from it.

    python tools/isa_tables.py import     # refresh the description
    python tools/isa_tables.py generate   # write the generated Rust modules
    python tools/isa_tables.py check      # fail if import or generate would change a file
    python tools/isa_tables.py switch     # point the compiler at the generated tables
    python tools/isa_tables.py status     # which switch-over edits are applied
    python tools/isa_tables.py generate --isa gfx950   # the gfx9 back end's tables
    python tools/isa_tables.py check --isa gfx950      # fail if that would change a file

``isa/gfx950.isa`` is authored, not imported: it has no overlay, ``import``,
``switch`` or ``status``; ``generate --isa gfx950`` writes
``src/target/gfx9/gen_isa.rs`` from it.

``import`` reads what the Rust sources state about instructions: the opcode
rows (``opcodes!`` in ``src/rir/mir.rs``), the flag and modifier bits, the VOPD
forms (``part`` in ``src/rir/vopd.rs``) and names, the classification rules of
``src/rir/post.rs`` (result latency, ``s_delay_alu`` class, wait counters,
VALU test), evaluated per opcode, and which opcodes ``src/rir/asm.rs``'s hand
encoder treats specially. It then applies the overlay ``isa/gfx11.overlay``,
the hand-written knowledge (encoding layouts, operand codes, operand
signatures, encoding forms, hazards, VOPD and compare tables, BURS grammar).
Re-importing never loses overlay knowledge. Once ``switch`` has run, the
opcode rows, VOPD forms and classifications live in the generated files and
``import`` takes them from the description itself.

An opcode that the hand encoder special-cases without a matching overlay
form, or whose encoder arm differs from the arms the table-driven encoder
mirrors, is marked ``table=no``: the hand encoder keeps encoding it, so
additions to ``asm.rs`` stay byte-exact without editing the overlay.

``switch`` applies anchored, idempotent edits to ``mod.rs``, ``mir.rs``,
``asm.rs``, ``vopd.rs``, ``post.rs`` and ``isel.rs``, preserving each file's
line endings. An edit whose anchor is not found as expected is reported and
skipped, never guessed; the command then exits non-zero unless
``--allow-skip`` is given. Generated files and the description use the line
endings of ``mir.rs``.

``--crate DIR`` runs a command on another copy of the crate.

Description format (``gfx11.isa`` and ``gfx11.overlay``): line oriented. ``#``
starts a comment, ``[name]`` a section. Every other line is a record: a kind
word, positional fields, then ``key=value`` fields. Values contain no blanks
unless double-quoted; lists are comma-separated. The sections and keys are
documented at the top of the overlay.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# isa_burs.py imports this module by name; run as a script, it is `__main__`.
sys.modules.setdefault("isa_tables", sys.modules[__name__])

PLATFORM = Path(__file__).resolve().parent.parent
DEFAULT_CRATE = PLATFORM / "rust" / "rocke-codeobj"
TOOL = "platform/tools/isa_tables.py"
DESCRIPTION = "isa/gfx11.isa"
OVERLAY = "isa/gfx11.overlay"
MIR = "src/rir/mir.rs"
ASM = "src/rir/asm.rs"
VOPD = "src/rir/vopd.rs"
POST = "src/rir/post.rs"
MODRS = "src/rir/mod.rs"
ISEL = "src/rir/isel.rs"
GEN_OPCODES = "src/rir/gen_opcodes.rs"
GEN_ISA = "src/rir/gen_isa.rs"
GEN_BURS = "src/rir/gen_burs.rs"

HEADER_RS = "// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.\n// SPDX-License-Identifier: MIT\n"


class DescError(Exception):
    """A malformed description, overlay or Rust source."""


# --------------------------------------------------------------------------
# Files: exact bytes, line endings preserved
# --------------------------------------------------------------------------


@dataclass
class Tree:
    crate: Path

    def path(self, rel: str) -> Path:
        return self.crate / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def raw(self, rel: str) -> str:
        return self.path(rel).read_bytes().decode("utf-8")

    def read(self, rel: str) -> str:
        """Text with LF line endings."""
        return self.raw(rel).replace("\r\n", "\n")

    def newline(self, rel: str) -> str:
        raw = self.raw(rel)
        crlf = raw.count("\r\n")
        if crlf and crlf != raw.count("\n"):
            raise DescError(f"{rel}: mixed line endings")
        return "\r\n" if crlf else "\n"

    def convention(self) -> str:
        """Line endings of generated files: those of mir.rs."""
        return self.newline(MIR)

    @staticmethod
    def encode(text: str, newline: str) -> bytes:
        return (text.replace("\n", newline) if newline != "\n" else text).encode(
            "utf-8"
        )

    def write(self, rel: str, text: str, newline: str) -> None:
        self.path(rel).parent.mkdir(parents=True, exist_ok=True)
        self.path(rel).write_bytes(self.encode(text, newline))


# --------------------------------------------------------------------------
# Description format
# --------------------------------------------------------------------------

_KEY = re.compile(r"[a-z_][a-z0-9_]*\Z")


@dataclass
class Record:
    kind: str
    args: list[str] = field(default_factory=list)
    kv: dict[str, str] = field(default_factory=dict)
    where: str = ""

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.kv.get(key, default)


def _fields(line: str, where: str) -> list[tuple[str | None, str]]:
    """Splits a record line into (key or None, value) fields."""
    out: list[tuple[str | None, str]] = []
    key: str | None = None
    cur: list[str] = []
    started = quoted = False
    i = 0
    while i < len(line):
        c = line[i]
        if quoted:
            if c == "\\" and i + 1 < len(line):
                cur.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                quoted = False
            else:
                cur.append(c)
        elif c == '"':
            quoted = started = True
        elif c == "#":
            break
        elif c.isspace():
            if started:
                out.append((key, "".join(cur)))
            key, cur, started = None, [], False
        elif c == "=" and key is None and _KEY.match("".join(cur)):
            key, cur, started = "".join(cur), [], True
        else:
            cur.append(c)
            started = True
        i += 1
    if quoted:
        raise DescError(f"{where}: unterminated quote")
    if started:
        out.append((key, "".join(cur)))
    return out


def parse(text: str, name: str) -> dict[str, list[Record]]:
    """Sections of a description or overlay, records in file order."""
    sections: dict[str, list[Record]] = {}
    current: list[Record] | None = None
    for number, raw in enumerate(text.splitlines(), 1):
        where = f"{name}:{number}"
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            m = re.fullmatch(r"\[([a-z_]+)\]\s*(#.*)?", line)
            if not m:
                raise DescError(f"{where}: bad section header {line!r}")
            current = sections.setdefault(m.group(1), [])
            continue
        if current is None:
            raise DescError(f"{where}: record outside a section")
        fields = _fields(line, where)
        if not fields:
            continue
        (first_key, kind), rest = fields[0], fields[1:]
        if first_key is not None:
            raise DescError(f"{where}: a record starts with its kind, not {first_key}=")
        rec = Record(kind, where=where)
        for k, v in rest:
            if k is None:
                rec.args.append(v)
            elif k in rec.kv:
                raise DescError(f"{where}: duplicate key {k}")
            else:
                rec.kv[k] = v
        current.append(rec)
    return sections


def _quote(value: str, positional: bool = False) -> str:
    plain = value != "" and not any(c.isspace() or c in '#"\\' for c in value)
    if plain and positional and "=" in value and _KEY.match(value.split("=", 1)[0]):
        plain = False
    if plain:
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def format_record(rec: Record, widths: list[int] | None = None) -> str:
    cells = [rec.kind] + [_quote(a, positional=True) for a in rec.args]
    cells += [f"{k}={_quote(v)}" for k, v in rec.kv.items()]
    if not widths:
        return " ".join(cells)
    out = []
    for i, c in enumerate(cells):
        if i < len(widths) and i < len(cells) - 1:
            out.append(c.ljust(widths[i] - 1) + " ")
        else:
            out.append(c + " ")
    return "".join(out).rstrip()


# --------------------------------------------------------------------------
# Rust source scanning
# --------------------------------------------------------------------------


def _int(text: str) -> int:
    return int(text.replace("_", ""), 0)


def _skip_literal(text: str, i: int) -> int | None:
    """Index after the string, char literal or comment starting at `i`, if any."""
    n = len(text)
    c = text[i]
    if c == '"':
        i += 1
        while i < n and text[i] != '"':
            i += 2 if text[i] == "\\" else 1
        return i + 1
    if text.startswith("//", i):
        j = text.find("\n", i)
        return n if j < 0 else j
    if text.startswith("/*", i):
        j = text.find("*/", i + 2)
        return n if j < 0 else j + 2
    if c == "'":
        if i + 2 < n and text[i + 1] != "\\" and text[i + 2] == "'":
            return i + 3
        if i + 3 < n and text[i + 1] == "\\" and text[i + 3] == "'":
            return i + 4
    return None


def _matching_brace(text: str, open_index: int) -> int:
    """Index of the bracket closing the one at `open_index`."""
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack = []
    i, n = open_index, len(text)
    while i < n:
        skip = _skip_literal(text, i)
        if skip is not None:
            i = skip
            continue
        c = text[i]
        if c in pairs:
            stack.append(pairs[c])
        elif c in ")}]":
            if not stack or stack.pop() != c:
                raise DescError("unbalanced brackets")
            if not stack:
                return i
        i += 1
    raise DescError("unbalanced brackets")


def _body(text: str, head: str, start: int = 0) -> tuple[int, int] | None:
    """(open, close) brace indices of the block after the first match of the
    `head` regex at or after `start`."""
    m = re.compile(head, re.MULTILINE).search(text, start)
    if not m:
        return None
    k = m.end()
    while text[k] != "{":
        skip = _skip_literal(text, k)
        k = skip if skip is not None else k + 1
    return k, _matching_brace(text, k)


def _strip_comments(code: str) -> str:
    out, i, n = [], 0, len(code)
    while i < n:
        if code.startswith("//", i) or code.startswith("/*", i):
            i = _skip_literal(code, i)
            continue
        skip = _skip_literal(code, i)
        if skip is not None:
            out.append(code[i:skip])
            i = skip
            continue
        out.append(code[i])
        i += 1
    return "".join(out)


def _norm(code: str) -> str:
    return " ".join(_strip_comments(code).split())


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _fp(code: str) -> str:
    return f"{fnv1a64(_norm(code).encode()):016x}"


def split_arms(body: str) -> list[tuple[str, str | None, str, int, int]]:
    """Arms of a `match` body: (pattern, guard, value, start, end) with text
    offsets of each whole arm."""
    arms = []
    i, n = 0, len(body)
    while True:
        while i < n and (body[i].isspace() or body[i] == ","):
            i += 1
        if i >= n:
            return arms
        if body.startswith("//", i) or body.startswith("/*", i):
            i = _skip_literal(body, i)
            continue
        start = i
        depth = 0
        arrow = -1
        while i < n:
            skip = _skip_literal(body, i)
            if skip is not None:
                i = skip
                continue
            c = body[i]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif depth == 0 and body.startswith("=>", i):
                arrow = i
                break
            i += 1
        if arrow < 0:
            raise DescError("match arm without =>")
        header = body[start:arrow].strip()
        i = arrow + 2
        while body[i].isspace():
            i += 1
        if body[i] == "{":
            end = _matching_brace(body, i) + 1
            value = body[i:end]
            i = end
        else:
            vstart = i
            depth = 0
            while i < n:
                skip = _skip_literal(body, i)
                if skip is not None:
                    i = skip
                    continue
                c = body[i]
                if c in "([{":
                    depth += 1
                elif c in ")]}":
                    if depth == 0:
                        break
                    depth -= 1
                elif c == "," and depth == 0:
                    break
                i += 1
            value = body[vstart:i].strip()
        pattern, guard = header, None
        m = re.search(r"\s+if\s+", header)
        if m:
            pattern, guard = header[: m.start()].strip(), header[m.end() :].strip()
        arms.append((pattern, guard, value, start, i))


# --------------------------------------------------------------------------
# A small evaluator for the Rust conditions post.rs classifies with
# --------------------------------------------------------------------------

_ETOK = re.compile(
    r"\s*(0x[0-9a-fA-F_]+|\d[\d_]*|matches!|[A-Za-z_]\w*(?:::\w+)*(?:\(\))?(?:\.\w+(?:\(\))?)*|&&|\|\||==|!=|[&|!(),])"
)
_FLAGS_ATOMS = {"flags", "inst.op.flags()", "op.flags()"}
_FMT_ATOMS = {"fmt", "inst.op.fmt()", "op.fmt()"}
_OP_ATOMS = {"inst.op", "op"}


class RustExprError(DescError):
    pass


def compile_expr(text: str, consts: dict[str, int]):
    """A function of {flags, fmt, op} evaluating a Rust condition over an
    instruction's flags, format (`Fmt::X`) and opcode (`Opc::X`)."""
    toks: list[str] = []
    pos = 0
    src = text.strip()
    while pos < len(src):
        m = _ETOK.match(src, pos)
        if not m or m.end() == pos:
            raise RustExprError(f"cannot read {src!r} at {src[pos:]!r}")
        toks.append(m.group(1))
        pos = m.end()
    k = 0

    def peek() -> str | None:
        return toks[k] if k < len(toks) else None

    def take(expect: str | None = None) -> str:
        nonlocal k
        if k >= len(toks):
            raise RustExprError(f"{src!r} ends early")
        t = toks[k]
        if expect is not None and t != expect:
            raise RustExprError(f"{src!r}: expected {expect!r}, got {t!r}")
        k += 1
        return t

    def pattern() -> str:
        t = take()
        if peek() == "(":
            take()
            while take() != ")":
                pass
            t += "(..)"
        return t

    def primary():
        t = take()
        if t == "(":
            f = or_()
            take(")")
            return f
        if t == "matches!":
            take("(")
            target = bitor()
            take(",")
            pats = {pattern()}
            while peek() == "|":
                take()
                pats.add(pattern())
            if peek() == ",":
                take()
            take(")")
            return lambda env: target(env) in pats
        if t[0].isdigit():
            v = _int(t)
            return lambda env: v
        if t in _FLAGS_ATOMS:
            return lambda env: env["flags"]
        if t in _FMT_ATOMS:
            return lambda env: env["fmt"]
        if t in _OP_ATOMS:
            return lambda env: env["op"]
        if t.startswith(("Fmt::", "Opc::", "AluKind::")):
            if peek() == "(":
                take()
                while take() != ")":
                    pass
                t += "(..)"
            return lambda env: t
        if t in consts:
            v = consts[t]
            return lambda env: v
        raise RustExprError(f"{src!r}: unknown name {t!r}")

    def unary():
        if peek() == "!":
            take()
            f = unary()
            return lambda env: not f(env)
        return primary()

    def binary(sub, ops, combine):
        def parse_level():
            f = sub()
            while peek() in ops:
                o = take()
                g = sub()
                f = combine(o, f, g)
            return f

        return parse_level

    bitand = binary(unary, ("&",), lambda o, f, g: lambda env: f(env) & g(env))
    bitor = binary(bitand, ("|",), lambda o, f, g: lambda env: f(env) | g(env))

    def cmp():
        f = bitor()
        if peek() in ("==", "!="):
            o = take()
            g = bitor()
            return (
                (lambda env: f(env) == g(env))
                if o == "=="
                else (lambda env: f(env) != g(env))
            )
        return f

    and_ = binary(
        cmp, ("&&",), lambda o, f, g: lambda env: bool(f(env)) and bool(g(env))
    )
    or_ = binary(
        and_, ("||",), lambda o, f, g: lambda env: bool(f(env)) or bool(g(env))
    )
    result = or_()
    if k != len(toks):
        raise RustExprError(f"{src!r}: trailing {toks[k:]}")
    return result


def _patterns(pattern: str) -> set[str] | None:
    """Alternatives of an arm pattern; None for `_`."""
    if pattern.strip() == "_":
        return None
    return {p.strip() for p in pattern.split("|")}


def _arm_match(pats: set[str] | None, value: str) -> bool:
    return pats is None or value in pats


# --------------------------------------------------------------------------
# mir.rs
# --------------------------------------------------------------------------


@dataclass
class Row:
    name: str
    fmt: str
    code: int
    mnemonic: str
    flags: list[str]
    group: str | None = None
    note: str | None = None


@dataclass
class MirFacts:
    fmts: list[str]
    flags: dict[str, int]
    mods: dict[str, int]
    aliases: dict[str, list[str]]
    rows: list[Row] | None
    vopd_names: dict[int, str] | None
    vcmp_ok: bool


_ROW = re.compile(
    r'^\s*([A-Z][A-Za-z0-9]*)\s*=\s*\(\s*([A-Za-z0-9]+)\s*,\s*([0-9a-fA-Fx_]+)\s*,\s*"([^"]*)"\s*,\s*([^)]*?)\s*\)\s*,?\s*(?://.*)?$'
)

# The VCmp family as the importer models it: names from `vcmp_name`, flags
# from the `opcodes!` macro. A change to either is reported, not guessed.
_VCMP_FLAGS_ARM = (
    "Opc::VCmp(op) => F_USE_EXEC | if (op & 0xf0) == 0x50 { F_64BIT_SRC } else { 0 },"
)
_VCMP_NAME_FP = "afe6e819c86d911c"


def vcmp_table() -> list[tuple[int, str, str]]:
    """(code, mnemonic, source type) of every compare `vcmp_name` names."""
    out = []
    f32 = [
        "f",
        "lt",
        "eq",
        "le",
        "gt",
        "lg",
        "ge",
        "o",
        "u",
        "nge",
        "nlg",
        "ngt",
        "nle",
        "neq",
        "nlt",
        "t",
    ]
    ints = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"]
    for k, p in enumerate(f32):
        out.append((0x10 + k, f"v_cmp_{p}_f32", "f32"))
    for base, ty in ((0x40, "i32"), (0x48, "u32"), (0x50, "i64"), (0x58, "u64")):
        for k, p in enumerate(ints):
            out.append((base + k, f"v_cmp_{p}_{ty}", ty))
    out.append((0x7E, "v_cmp_class_f32", "f32"))
    return out


def vcmp_flag_names(code: int) -> set[str]:
    return {"USE_EXEC"} | ({"64BIT_SRC"} if code & 0xF0 == 0x50 else set())


def scan_mir(text: str) -> MirFacts:
    span = _body(text, r"^pub enum Fmt\b")
    if not span:
        raise DescError("mir.rs: no `pub enum Fmt`")
    fmts = re.findall(
        r"^\s*([A-Z][A-Za-z0-9]*)\s*,",
        _strip_comments(text[span[0] + 1 : span[1]]),
        re.MULTILINE,
    )
    flags = {
        n: int(b)
        for n, b in re.findall(
            r"^pub const F_([A-Z0-9_]+): u32 = 1 << (\d+);", text, re.MULTILINE
        )
    }
    mods = {
        n: int(b)
        for n, b in re.findall(
            r"^pub const M_([A-Z0-9_]+): u32 = 1 << (\d+);", text, re.MULTILINE
        )
    }
    aliases = {}
    for name, expr in re.findall(
        r"^const ([A-Z][A-Z0-9_]*): u32 = ([^;]+);", text, re.MULTILINE
    ):
        terms = [t.strip() for t in expr.split("|")]
        if all(t.startswith("F_") and t[2:] in flags for t in terms):
            aliases[name] = [t[2:] for t in terms]
    rows = None
    inv = re.search(r"^opcodes!\s*\{", text, re.MULTILINE)
    if inv:
        start = text.index("{", inv.start())
        body = text[start + 1 : _matching_brace(text, start)]
        rows = []
        group = None
        notes: list[str] = []
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("//"):
                c = s[2:].strip()
                if c and " " not in c and len(c) <= 10:
                    group = c
                    notes = []
                else:
                    notes.append(c)
                continue
            m = _ROW.match(line)
            if not m:
                raise DescError(f"mir.rs: unrecognised opcodes! row {s!r}")
            name, fmt, code, mn, fexpr = m.groups()
            toks = []
            for t in (x.strip() for x in fexpr.split("|")):
                if t == "0":
                    continue
                if t.startswith("F_") and t[2:] in flags:
                    toks.append(t[2:])
                elif t in aliases:
                    toks.append(t)
                else:
                    raise DescError(f"mir.rs: {name}: unsupported flags term {t!r}")
            rows.append(
                Row(name, fmt, _int(code), mn, toks, group, " ".join(notes) or None)
            )
            notes = []
    elif 'include!("gen_opcodes.rs")' not in text:
        raise DescError(
            'mir.rs: neither an opcodes! table nor include!("gen_opcodes.rs")'
        )
    vopd_names = None
    vspan = _body(text, r"^pub fn vopd_name\(code: u8\) -> &'static str")
    if vspan:
        vbody = text[vspan[0] + 1 : vspan[1]]
        if "gen_isa::vopd_name" not in vbody:
            vopd_names = {
                int(c): n
                for c, n in re.findall(
                    r'^\s*(\d+)\s*=>\s*"([^"]+)"', vbody, re.MULTILINE
                )
            }
    nspan = _body(text, r"^pub fn vcmp_name\(op: u8\) -> String")
    vcmp_ok = bool(nspan) and _fp(text[nspan[0] : nspan[1] + 1]) == _VCMP_NAME_FP
    vcmp_ok = vcmp_ok and _norm(_VCMP_FLAGS_ARM) in _norm(text)
    return MirFacts(fmts, flags, mods, aliases, rows, vopd_names, vcmp_ok)


# --------------------------------------------------------------------------
# vopd.rs
# --------------------------------------------------------------------------


@dataclass
class VopdFacts:
    forms: dict[str, tuple[int, bool]] | None  # None once switched over
    special: list[str]  # opcodes `part` sends elsewhere before the table


def scan_vopd(text: str) -> VopdFacts:
    span = _body(text, r"^pub fn part\(")
    if not span:
        raise DescError("vopd.rs: no `pub fn part`")
    body = text[span[0] + 1 : span[1]]
    table = re.search(r"let \(code, y_only\) = match inst\.op \{", body)
    switched = re.search(
        r"let \(code, y_only\) = super::gen_isa::vopd\(inst\.op\)\?;", body
    )
    anchor = table or switched
    if not anchor:
        raise DescError("vopd.rs: `part` has no `let (code, y_only) = ...` statement")
    special = []
    for cond in re.findall(r"if (.+?) \{", body[: anchor.start()]):
        for name in re.findall(r"Opc::(\w+)", cond):
            if name not in special:
                special.append(name)
    if switched:
        return VopdFacts(None, special)
    tbody_start = body.index("{", table.start())
    tbody = body[tbody_start + 1 : _matching_brace(body, tbody_start)]
    forms = {}
    for pattern, guard, value, _s, _e in split_arms(tbody):
        if pattern == "_":
            continue
        m = re.fullmatch(r"\((\d+),\s*(true|false)\)", value)
        if guard or not m:
            raise DescError(f"vopd.rs: unsupported VOPD table arm {pattern} => {value}")
        for p in pattern.split("|"):
            name = p.strip()
            if not name.startswith("Opc::"):
                raise DescError(f"vopd.rs: unsupported VOPD table pattern {p}")
            forms[name[5:]] = (int(m.group(1)), m.group(2) == "true")
    return VopdFacts(forms, special)


# --------------------------------------------------------------------------
# post.rs: per-opcode classification, evaluated from the rules as written
# --------------------------------------------------------------------------


@dataclass
class PostFacts:
    """Classifiers found in post.rs, as functions of (flags, fmt, op), or
    `"switched"` once they call the generated tables."""

    result_latency: object = None
    delay_kind: object = None
    delay_clear: object = None
    delay_cycles: dict[str, int] | None = None
    is_valu: object = None
    wait: object = None
    wmma: object = None
    notes: list[str] = field(default_factory=list)


def _env_eval(arms, env):
    for pats, guard, value in arms:
        if _arm_match(pats, env["fmt"]) and (guard is None or guard(env)):
            return value
    raise DescError(f"no arm matches {env}")


def _fn_span(text: str, sig: str) -> tuple[int, int] | None:
    return _body(text, r"^\s*(?:pub(?:\([^)]*\))? )?" + sig)


def scan_post(text: str, consts: dict[str, int]) -> PostFacts:
    facts = PostFacts()
    # result latency
    span = _fn_span(text, r"fn result_latency\(inst: &MInst\) -> u32")
    if not span:
        facts.notes.append("post.rs: no result_latency")
    else:
        body = text[span[0] + 1 : span[1]]
        if "gen_isa::result_latency" in body:
            facts.result_latency = "switched"
        else:
            try:
                m = re.match(
                    r"\s*(?:let flags = inst\.op\.flags\(\);\s*)?match inst\.op\.fmt\(\) \{",
                    body,
                )
                if not m:
                    raise DescError("unexpected shape")
                ms = body.index("{", m.end() - 1)
                me = _matching_brace(body, ms)
                if body[me + 1 :].strip():
                    raise DescError("statements after the match")
                arms = []
                for pattern, guard, value, _s, _e in split_arms(body[ms + 1 : me]):
                    arms.append(
                        (
                            _patterns(pattern),
                            compile_expr(guard, consts) if guard else None,
                            _int(value),
                        )
                    )
                facts.result_latency = lambda env, arms=arms: _env_eval(arms, env)
            except DescError as e:
                facts.notes.append(f"post.rs: result_latency not modelled ({e})")
    # s_delay_alu class
    if "gen_isa::delay(inst.op)" in text:
        facts.delay_kind = facts.delay_clear = "switched"
    else:
        try:
            m = re.search(r"let kind = match fmt \{", text)
            if not m:
                raise DescError("no `let kind = match fmt`")
            ms = text.index("{", m.start())
            me = _matching_brace(text, ms)
            arms = []
            for pattern, guard, value, _s, _e in split_arms(text[ms + 1 : me]):
                v = value.strip()
                vm = re.fullmatch(r"AluKind::(\w+)", v)
                im = re.fullmatch(
                    r"\{\s*if (.+?) \{\s*AluKind::(\w+)\s*\}\s*else\s*\{\s*AluKind::(\w+)\s*\}\s*\}",
                    v,
                    re.DOTALL,
                )
                if vm:
                    val = (None, vm.group(1), vm.group(1))
                elif im:
                    val = (compile_expr(im.group(1), consts), im.group(2), im.group(3))
                else:
                    raise DescError(f"kind arm value {v!r}")
                arms.append(
                    (
                        _patterns(pattern),
                        compile_expr(guard, consts) if guard else None,
                        val,
                    )
                )

            def kind(env, arms=arms):
                cond, a, b = _env_eval(arms, env)
                return a if cond is None or cond(env) else b

            rest = text[me + 1 :]
            cm = re.match(
                r"\s*;\s*if (.+?) \{\s*state\.clear\(\);\s*\} else if kind != AluKind::Other \{",
                rest,
                re.DOTALL,
            )
            if not cm:
                raise DescError(
                    "no `if <memory> { state.clear(); } else if kind != AluKind::Other` after it"
                )
            clear = compile_expr(cm.group(1), consts)
            lm = re.search(r"let latency = match kind \{", rest)
            if not lm:
                raise DescError("no `let latency = match kind`")
            ls = rest.index("{", lm.start())
            le = _matching_brace(rest, ls)
            cycles = {}
            for pattern, guard, value, _s, _e in split_arms(rest[ls + 1 : le]):
                if guard:
                    raise DescError("guarded latency arm")
                for p in _patterns(pattern) or {"_"}:
                    cycles[p.replace("AluKind::", "")] = _int(value)
            facts.delay_kind, facts.delay_clear, facts.delay_cycles = (
                kind,
                clear,
                cycles,
            )
        except DescError as e:
            facts.notes.append(
                f"post.rs: insert_delay_alu classification not modelled ({e})"
            )
    # VALU test
    span = _fn_span(text, r"fn is_valu\(op: Opc\) -> bool")
    if span:
        body = text[span[0] + 1 : span[1]].strip()
        if "gen_isa::is_valu" in body:
            facts.is_valu = "switched"
        else:
            try:
                facts.is_valu = compile_expr(body, consts)
            except DescError as e:
                facts.notes.append(f"post.rs: is_valu not modelled ({e})")
    # wait counters and the WMMA hazard: fixed shapes
    facts.wait = (
        "switched"
        if "gen_isa::counters(inst.op)" in text
        else (_WAIT_ISSUE.search(text) is not None)
    )
    facts.wmma = (
        "switched"
        if "gen_isa::HZ_WMMA_AB" in text
        else (_WMMA_HAZARD.search(text) is not None)
    )
    if facts.wait is False:
        facts.notes.append(
            "post.rs: the wait-counter issue block is not in the expected form"
        )
    if facts.wmma is False:
        facts.notes.append("post.rs: the WMMA hazard test is not in the expected form")
    return facts


_WAIT_ISSUE = re.compile(
    r"if flags & F_VMEM_LOAD != 0 \{\s*state\.vm\.issue\(&inst\.defs\);\s*state\.vm_any = true;\s*\}\s*"
    r"if flags & \(F_LDS \| F_SMEM\) != 0 \{\s*state\.lgkm\.issue\(&inst\.defs\);\s*state\.lgkm_any = true;\s*"
    r"if flags & F_SMEM != 0 \{\s*state\.smem = true;\s*\}\s*\}"
)

_WMMA_HAZARD = re.compile(
    r"if inst\.op\.flags\(\) & F_WMMA != 0 \{(\s*)let prev = out\.iter\(\)\.rev\(\)\.find\(\|i\| is_valu\(i\.op\)\);\s*"
    r"let hazard = \|p: &MInst\| \{\s*p\.op\.flags\(\) & F_WMMA != 0"
)


# --------------------------------------------------------------------------
# asm.rs: which opcodes the hand encoder special-cases, and whether its arms
# are still the ones the table-driven encoder (isa.rs) mirrors
# --------------------------------------------------------------------------

# Fingerprints of the hand encoder's arms and helpers that isa.rs mirrors
# (normalised text, FNV-1a 64). `fp` prints the current ones.
KNOWN_ASM = {
    "prelude": "4ccad524b4f86610",
    "Sop1": "e8fdf7ce41976a47",
    "Sop2": "ee4a537eaa91e5dd",
    "Sopc": "e71f8a9a4f20e546",
    "Sopk": "7b8b0ca9f0ddf4f5",
    "Sopp": "5c8b5d863de36065",
    "Smem": "6352df3fadb62dc6",
    "Vop1 if inst.defs.is_empty()": "1115e23dee562e7f",
    "Vop1 if matches!(op, Opc::VMovB32Dpp | Opc::VMovB32Dpp8)": "d965a4043d339c8c",
    "Vop1": "41cacd4cedef653c",
    "Vop2": "5ab4f76e1cb7244e",
    "Vop3 if op == Opc::VWritelaneB32": "762ee45c71a5da42",
    "Vop3": "cba151c16dac1bae",
    "Vop3b": "daec053735891f23",
    "Vopc": "c11e3b90882d24a3",
    "Vop3p": "262c30c8f17aac6d",
    "Vopd": "995e1db3be76e3d0",
    "Ds": "c350220f4f4a0f33",
    "Global if is_scratch(op)": "7c78493e574dfb53",
    "Global": "84aa72381ff84bc1",
    "Mubuf if inst.defs.is_empty() && inst.uses.is_empty()": "d792eefd41604abd",
    "Mubuf": "c2254803b8b4de84",
    "Pseudo": "7a1adfd5dade06c6",
    "fn bad": "1e624fedee1f83d3",
    "fn src_lit": "55c4eb49dc3af7d7",
    "fn src_code": "67c0d409bdbd5514",
    "fn ssrc_code": "d428cf63a17d624b",
    "fn sdst_code": "cd6f416a8e434b1b",
    "fn vgpr": "becd9584bea9c1e6",
    "fn is_vgpr": "72c8ec6b134be45e",
    "fn is_scratch": "4f3ed5151bce24d3",
    "fn push_literal": "5e443b0749fd51c9",
    "fn vop3_words": "192e38472f65eecf",
}

ASM_DRIVER_FNS = {
    "encode",
    "encode_legacy",
    "encode_legacy_words",
    "lane_mask_source",
    "encode_blocks",
    "branch_delta",
    "assemble",
    "listing",
}


@dataclass
class AsmArm:
    fmts: list[str]
    guard: str | None
    fp: str
    ops: set[str]
    calls: set[str]


@dataclass
class AsmFacts:
    parsed: bool
    arms: list[AsmArm]
    helpers: dict[str, str]
    helper_calls: dict[str, set[str]]
    prelude_fp: str
    special: set[str]
    helper_ops: dict[str, set[str]]
    notes: list[str] = field(default_factory=list)


def _top_fns(text: str) -> dict[str, tuple[int, int]]:
    """Top-level functions outside `#[cfg(test)]` modules: name -> (start, end)."""
    test = re.search(r"^#\[cfg\(test\)\]", text, re.MULTILINE)
    limit = test.start() if test else len(text)
    out = {}
    for m in re.finditer(
        r"^(?:pub(?:\([^)]*\))? )?fn (\w+)\b", text[:limit], re.MULTILINE
    ):
        span = _body(text, re.escape(m.group(0)), m.start())
        if span:
            out[m.group(1)] = (m.start(), span[1] + 1)
    return out


def _calls(code: str) -> set[str]:
    return set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", _strip_comments(code)))


def scan_asm(text: str) -> AsmFacts:
    fns = _top_fns(text)
    name = "encode_legacy" if "encode_legacy" in fns else "encode"
    facts = AsmFacts(False, [], {}, {}, "", set(), {})
    if name not in fns:
        facts.notes.append("asm.rs: no hand encoder function")
        return facts
    s, e = fns[name]
    code = text[s:e]
    m = re.search(r"match op\.fmt\(\) \{", code)
    if not m:
        facts.notes.append("asm.rs: the hand encoder has no `match op.fmt()`")
        return facts
    ms = code.index("{", m.start())
    me = _matching_brace(code, ms)
    facts.prelude_fp = _fp(code[code.index("{") + 1 : m.start()])
    for pattern, guard, value, a, b in split_arms(code[ms + 1 : me]):
        arm_text = code[ms + 1 + a : ms + 1 + b]
        fmts = re.findall(r"Fmt::(\w+)", pattern)
        facts.arms.append(
            AsmArm(
                fmts,
                guard,
                _fp(arm_text),
                set(re.findall(r"Opc::(\w+)", arm_text)),
                _calls(arm_text),
            )
        )
    for fname, (fs, fe) in fns.items():
        if fname in ASM_DRIVER_FNS:
            continue
        facts.helpers[fname] = _fp(text[fs:fe])
        facts.helper_calls[fname] = _calls(text[fs:fe])
        facts.helper_ops[fname] = set(re.findall(r"Opc::(\w+)", text[fs:fe]))
    facts.special = set(re.findall(r"Opc::(\w+)", _strip_comments(code)))
    for ops in facts.helper_ops.values():
        facts.special |= ops
    facts.special.discard("VCmp")
    facts.parsed = True
    return facts


def asm_fingerprints(facts: AsmFacts) -> dict[str, str]:
    out = {"prelude": facts.prelude_fp}
    for arm in facts.arms:
        out[_arm_label(arm)] = arm.fp
    for name, fp in facts.helpers.items():
        out[f"fn {name}"] = fp
    return out


def _arm_label(arm: AsmArm) -> str:
    return "|".join(arm.fmts) + (f" if {_norm(arm.guard)}" if arm.guard else "")


def legacy_encoded(
    asm: AsmFacts, rows: list[Row], forms: dict[str, str]
) -> tuple[set[str], list[str]]:
    """Opcodes the table-driven encoder must leave to the hand encoder, with
    the reasons."""
    names = [r.name for r in rows]
    if not asm.parsed:
        return set(names), asm.notes + [
            "the table-driven encoder is off: asm.rs not understood"
        ]
    known = KNOWN_ASM
    notes: list[str] = []
    out: set[str] = set()
    changed_helpers = {n for n, fp in asm.helpers.items() if known.get(f"fn {n}") != fp}

    def tainted(calls: set[str], seen: set[str] | None = None) -> bool:
        seen = seen or set()
        for c in calls:
            if c in changed_helpers:
                return True
            if c in asm.helper_calls and c not in seen:
                seen.add(c)
                if tainted(asm.helper_calls[c], seen):
                    return True
        return False

    if known.get("prelude") != asm.prelude_fp:
        return set(names), [
            "asm.rs: the hand encoder's prelude changed; every opcode stays with it"
        ]
    by_fmt = {}
    for r in rows:
        by_fmt.setdefault(r.fmt, []).append(r.name)
    for arm in asm.arms:
        ok = known.get(_arm_label(arm)) == arm.fp and not tainted(arm.calls)
        if ok:
            continue
        guard_ops = set(re.findall(r"Opc::(\w+)", arm.guard or ""))
        for c in _calls(arm.guard or ""):
            guard_ops |= asm.helper_ops.get(c, set())
        if arm.guard and guard_ops:
            out |= guard_ops
            notes.append(
                f"asm.rs: arm `{_arm_label(arm)}` is new or changed; {sorted(guard_ops)} stay with the hand encoder"
            )
        else:
            for f in arm.fmts:
                out |= set(by_fmt.get(f, []))
            notes.append(
                f"asm.rs: arm `{_arm_label(arm)}` is new or changed; format(s) {arm.fmts} stay with the hand encoder"
            )
    labels = {_arm_label(a) for a in asm.arms}
    for label in known:
        if label not in labels and label != "prelude" and not label.startswith("fn "):
            fmts = label.split(" if ")[0].split("|")
            for f in fmts:
                out |= set(by_fmt.get(f, []))
            notes.append(
                f"asm.rs: arm `{label}` is gone; format(s) {fmts} stay with the hand encoder"
            )
    for op in sorted(asm.special):
        if op in names and op not in forms:
            out.add(op)
            notes.append(
                f"asm.rs: {op} is special-cased by the hand encoder and has no overlay form; it stays with it"
            )
    for op, form in sorted(forms.items()):
        if form != "plain" and op not in asm.special and op in names:
            out.add(op)
            notes.append(
                f"asm.rs: {op} has overlay form {form} but the hand encoder no longer special-cases it"
            )
    return out, notes


# --------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------

FMT_ENC_DEFAULT = {
    "Sop1": "SOP1",
    "Sop2": "SOP2",
    "Sopc": "SOPC",
    "Sopk": "SOPK",
    "Sopp": "SOPP",
    "Smem": "SMEM",
    "Vop1": "VOP1",
    "Vop2": "VOP2",
    "Vop3": "VOP3",
    "Vop3b": "VOP3SD",
    "Vopc": "VOPC",
    "Vop3p": "VOP3P",
    "Vopd": "VOPD",
    "Ds": "DS",
    "Global": "GLOBAL",
    "Mubuf": "MUBUF",
    "Pseudo": "PSEUDO",
}


def _dwords(bits: int) -> int:
    return max(1, bits // 32)


def _bits(mn: str) -> list[int]:
    return [int(b) for b in re.findall(r"_[biuf](\d+)", mn)]


SOPP_IMM = {
    "s_nop": "nop",
    "s_clause": "clause",
    "s_delay_alu": "delay",
    "s_waitcnt": "waitcnt",
    "s_sendmsg": "sendmsg",
    "s_setprio": "simm16",
}


def derive_signature(fmt: str, mn: str, flags: set[str]) -> dict[str, str]:
    """Default operand signature (defs, uses, immediate, modifier sets) of an
    instruction from its format, mnemonic and flags."""
    sig = {"d": "", "u": "", "imm": "", "m": ""}
    bits = _bits(mn) or [32]
    n = _dwords(bits[-1])
    load = "VMEM_LOAD" in flags or (fmt == "Ds" and "load" in mn)
    store = "VMEM_STORE" in flags or (fmt == "Ds" and "store" in mn)
    tied = "TIED" in flags
    if fmt == "Sop1":
        sig.update(d=f"sdst:s{n}", u=f"ssrc0:ss{n}")
    elif fmt == "Sop2":
        n1 = 1 if re.match(r"s_(lshl|lshr|ashr)_[bi]64$", mn) else n
        sig.update(d=f"sdst:s{n}", u=f"ssrc0:ss{n},ssrc1:ss{n1}")
    elif fmt == "Sopc":
        sig.update(u=f"ssrc0:ss{n},ssrc1:ss{n}")
    elif fmt == "Sopk":
        sig.update(d="sdst:s1", imm="simm16")
    elif fmt == "Sopp":
        sig["imm"] = "label" if "BRANCH" in flags else SOPP_IMM.get(mn, "")
    elif fmt == "Smem":
        sig.update(d=f"sdata:s{n}", u="sbase:s2", imm="soff21")
    elif fmt == "Vop1":
        if mn == "v_nop":
            return sig
        dst = "vh" if "16BIT_DST" in flags else f"v{n}"
        src = (
            "vsh"
            if "16BIT_SRC" in flags
            else f"vs{_dwords(bits[0]) if '64BIT_SRC' in flags else 1}"
        )
        sig.update(d=f"vdst:{dst}", u=f"src0:{src}", m="neg,opsel,opsel_dst")
    elif fmt == "Vop2":
        sig.update(d="vdst:v1", u="src0:vs1,vsrc1:v1", m="neg,opsel,opsel_dst")
    elif fmt == "Vop3":
        sig.update(d=f"vdst:v{n}", u="src0:vs1,src1:vs1", m="neg,opsel,opsel_dst")
    elif fmt == "Vop3b":
        sig.update(
            d=f"vdst:v{_dwords(bits[0])},sdst:s1", u="src0:vs1,src1:vs1", m="neg"
        )
    elif fmt == "Vopc":
        sig.update(d="sdst:s1", u=f"src0:vs{n},src1:vs{n}", m="neg,opsel")
    elif fmt == "Vop3p":
        sig.update(d="vdst:v1", u="src0:vi1,src1:vi1,src2:vi1", m="neg")
    elif fmt == "Vopd":
        sig.update(d="vdstx:v1,vdsty:v1", u="srcx:vs1,srcy:vs1", imm="vopd")
    elif fmt == "Ds":
        if load:
            sig.update(
                d=f"vdst:v{n}", u="addr:v1" + (",vdst:v1" if tied else ""), imm="off16"
            )
        elif store:
            sig.update(u=f"addr:v1,data0:v{n}", imm="off16")
        else:
            sig.update(u="addr:v1,data0:v1", imm="off16")
    elif fmt == "Global":
        if mn.startswith("scratch_"):
            (
                sig.update(d=f"vdst:v{n}", imm="off13s")
                if load
                else sig.update(u=f"vdata:v{n}", imm="off13s")
            )
        elif load and store:
            sig.update(
                d=f"vdst:v{n}", u=f"vaddr:v2,vdata:v{n}", imm="off13s", m="saddr,glc"
            )
        elif load:
            sig.update(
                d=f"vdst:v{n}",
                u="vaddr:v2" + (",vdst:v1" if tied else ""),
                imm="off13s",
                m="saddr",
            )
        else:
            sig.update(u=f"vaddr:v2,vdata:v{n}", imm="off13s", m="saddr")
    elif fmt == "Mubuf":
        if load:
            sig.update(
                d=f"vdata:v{n}",
                u="vaddr:v1,srsrc:s4,soffset:ss1" + (",vdata:v1" if tied else ""),
                imm="off12",
            )
        elif store:
            sig.update(u=f"vdata:v{n},vaddr:v1,srsrc:s4,soffset:ss1", imm="off12")
    return sig


def derive_unit(fmt: str, flags: set[str], valu: bool) -> str:
    if fmt == "Pseudo":
        return "PSEUDO"
    if "WMMA" in flags:
        return "WMMA"
    if valu:
        return "TRANS" if "TRANS" in flags else "VALU"
    if "SMEM" in flags:
        return "SMEM"
    if "LDS" in flags:
        return "LDS"
    if flags & {"VMEM_LOAD", "VMEM_STORE"} or fmt in ("Global", "Mubuf"):
        return "VMEM"
    if "BRANCH" in flags:
        return "BRANCH"
    if fmt == "Sopp":
        return "CTRL"
    if fmt.startswith("Sop"):
        return "SALU"
    return "OTHER"


# The wave-3 baseline's post.rs rules, used when post.rs no longer states
# them (after `switch`, or if it was restructured) for opcodes that are new
# to the description.
def base_model(
    fmt: str, flags: set[str]
) -> tuple[int, str, bool, int, list[str], bool]:
    if fmt == "Pseudo" and "USE_EXEC" not in flags:
        sched = 0
    elif "VMEM_LOAD" in flags:
        sched = 300
    elif flags & {"LDS", "SMEM"}:
        sched = 40
    elif "WMMA" in flags:
        sched = 16
    elif "TRANS" in flags:
        sched = 10
    elif fmt in ("Sop1", "Sop2", "Sopc", "Sopk"):
        sched = 2
    else:
        sched = 5
    valu_fmts = ("Vop1", "Vop2", "Vop3", "Vop3b", "Vopc", "Vop3p", "Vopd")
    if fmt == "Pseudo":
        kind, cycles = "skip", 0
    elif fmt in valu_fmts:
        kind = "trans" if "TRANS" in flags else "valu"
        cycles = 10 if kind == "trans" else 5
    elif fmt in ("Sop1", "Sop2", "Sopc", "Sopk", "Sopp"):
        kind, cycles = "salu", 2
    else:
        kind, cycles = "other", 0
    clear = fmt in ("Ds", "Global", "Mubuf")
    return sched, kind, clear, cycles, base_wait(flags), fmt in valu_fmts


def base_wait(flags: set[str]) -> list[str]:
    wait = []
    if "VMEM_LOAD" in flags:
        wait.append("vm")
    if flags & {"LDS", "SMEM"}:
        wait.append("lgkm")
    if "SMEM" in flags:
        wait.append("smem")
    return wait


def post_model(post: PostFacts, fmt: str, flags_int: int, flags: set[str], op: str):
    """(sched, delay kind, clears, cycles, wait counters, is VALU) from the
    classifiers post.rs states, or None for each one it does not."""
    env = {"flags": flags_int, "fmt": f"Fmt::{fmt}", "op": f"Opc::{op}"}
    sched = post.result_latency(env) if callable(post.result_latency) else None
    kind = clear = cycles = None
    if callable(post.delay_kind):
        if fmt == "Pseudo":
            kind, clear, cycles = "skip", False, 0
        else:
            k = post.delay_kind(env)
            kind = k.lower()
            clear = bool(post.delay_clear(env))
            cycles = (
                0
                if kind == "other"
                else post.delay_cycles.get(k, post.delay_cycles.get("_", 0))
            )
    valu = bool(post.is_valu(env)) if callable(post.is_valu) else None
    wait = base_wait(flags) if post.wait is True else None
    return sched, kind, clear, cycles, wait, valu


# --------------------------------------------------------------------------
# Import: Rust sources + overlay -> description
# --------------------------------------------------------------------------

INST_KEYS = [
    "fmt",
    "flags",
    "d",
    "u",
    "imm",
    "m",
    "imp",
    "unit",
    "lat",
    "haz",
    "vopd",
    "vopd_special",
    "form",
    "table",
    "issue",
    "note",
]

OVERLAY_INST_KEYS = {
    "enc",
    "d",
    "u",
    "imm",
    "m",
    "imp",
    "imp+",
    "haz",
    "vopd_special",
    "form",
    "issue",
    "note",
    "lat",
}


@dataclass
class Imported:
    text: str
    warnings: list[str]


def _flag_set(tokens: list[str], aliases: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    for t in tokens:
        out.update(aliases.get(t, [t]))
    return out


def _flags_field(value: str | None) -> list[str]:
    return [] if value in (None, "", "0") else value.split("|")


def _lat_model(rec: Record) -> tuple:
    dk, _, dc = (rec.get("delay") or "").partition(":")
    return (
        int(rec.get("sched", "-1")),
        dk,
        rec.get("clear", "no") == "yes",
        int(dc or 0),
        tuple(sorted(filter(None, (rec.get("wait") or "").split(",")))),
    )


def run_import(tree: Tree) -> Imported:
    warnings: list[str] = []
    overlay = parse(tree.read(OVERLAY), OVERLAY)
    old = parse(tree.read(DESCRIPTION), DESCRIPTION) if tree.exists(DESCRIPTION) else {}
    mir = scan_mir(tree.read(MIR))
    consts = {f"F_{n}": 1 << b for n, b in mir.flags.items()}
    vopd = scan_vopd(tree.read(VOPD))
    post = scan_post(tree.read(POST), consts)
    asm = scan_asm(tree.read(ASM))
    warnings += post.notes + asm.notes
    if not mir.vcmp_ok:
        warnings.append(
            "mir.rs: vcmp_name or the VCmp flags differ from the modelled ones; the [vcmp] table may be stale"
        )
    old_insts = {r.args[0]: r for r in old.get("inst", []) if r.kind == "inst"}

    # Opcode rows: from mir.rs, or from the description once switched over.
    if mir.rows is not None:
        rows = mir.rows
    else:
        if not old_insts:
            raise DescError(
                "mir.rs is switched over but the description has no instructions"
            )
        fmt_of_enc = {r.get("enc"): r.args[0] for r in old.get("fmt", [])}
        rows, group = [], None
        for rec in old.get("inst", []):
            if rec.kind == "group":
                group = rec.args[0]
            elif rec.kind == "inst":
                name, mn, enc, code = rec.args[:4]
                fmt = rec.get("fmt") or fmt_of_enc[enc]
                rows.append(
                    Row(
                        name,
                        fmt,
                        _int(code),
                        mn,
                        _flags_field(rec.get("flags")),
                        group,
                        rec.get("note"),
                    )
                )
    names = [r.name for r in rows]
    if len(set(names)) != len(names):
        raise DescError("duplicate opcode variant names")
    row_by = {r.name: r for r in rows}

    # VOPD forms.
    if vopd.forms is not None:
        vopd_forms = vopd.forms
    else:
        vopd_forms = {}
        for n, rec in old_insts.items():
            if rec.get("vopd"):
                v = rec.get("vopd")
                vopd_forms[n] = (int(v.rstrip("y")), v.endswith("y"))
    for n in list(vopd_forms) + vopd.special:
        if n not in row_by:
            raise DescError(f"vopd.rs: VOPD form for unknown opcode {n}")

    # Formats.
    fmt_recs = {r.args[0]: r for r in overlay.get("fmt", [])}
    enc_recs = {r.args[0]: r for r in overlay.get("enc", [])}
    fmt_out, enc_of_fmt = [], {}
    for f in mir.fmts:
        rec = fmt_recs.get(f) or Record(
            "fmt", [f], {"enc": FMT_ENC_DEFAULT.get(f, f.upper()), "table": "no"}
        )
        if f not in fmt_recs:
            warnings.append(
                f"mir.rs: format {f} has no [fmt] entry in the overlay; its opcodes stay with the hand encoder"
            )
        fmt_out.append(Record("fmt", [f], dict(rec.kv)))
        enc_of_fmt[f] = rec.get("enc")
    fmt_of_enc_name = {v: k for k, v in enc_of_fmt.items()}

    flag_imp = {}
    for r in overlay.get("flagimp", []):
        flag_imp[r.args[0]] = r.get("imp")

    # Overlay instruction entries.
    inst_over: dict[str, Record] = {}
    for r in overlay.get("inst", []):
        if r.kind != "inst" or not r.args:
            raise DescError(
                f"{r.where}: [inst] overlay records are `inst Variant key=value ...`"
            )
        if r.args[0] in inst_over:
            raise DescError(f"{r.where}: duplicate overlay entry for {r.args[0]}")
        for k in r.kv:
            if k not in OVERLAY_INST_KEYS:
                raise DescError(f"{r.where}: unknown overlay instruction key {k}")
        if r.args[0] not in row_by and r.args[0] != "VCmp":
            raise DescError(f"{r.where}: {r.args[0]} is not an opcode of mir.rs")
        inst_over[r.args[0]] = r
    forms = {n: r.get("form") for n, r in inst_over.items() if r.get("form")}
    legacy, legacy_notes = legacy_encoded(asm, rows, forms)
    if mir.rows is None and not asm.parsed:
        legacy = {n for n, r in old_insts.items() if r.get("table") == "no"}
        legacy_notes = []
    warnings += legacy_notes

    # Latency classes.
    lat_recs = [Record("lat", list(r.args), dict(r.kv)) for r in overlay.get("lat", [])]
    lat_by_name = {r.args[0]: r for r in lat_recs}
    # Once switched over, post.rs no longer states the classes import
    # synthesized: the previous description keeps them.
    old_lat = {r.args[0]: r for r in old.get("lat", []) if r.kind == "lat"}

    def prev_model(rec: Record | None) -> tuple | None:
        name = rec.get("lat") if rec else None
        found = lat_by_name.get(name) or old_lat.get(name)
        return _lat_model(found) if found is not None else None

    def lat_for(model: tuple) -> str:
        for r in lat_recs:
            if _lat_model(r) == model:
                return r.args[0]
        sched, kind, clear, cycles, wait = model
        name = f"auto_{sched}_{kind}{cycles}{'_clear' if clear else ''}{'_' + '_'.join(wait) if wait else ''}"
        rec = Record(
            "lat",
            [name, "synthesized by import: no overlay class has this model"],
            {"sched": str(sched), "delay": f"{kind}:{cycles}" if cycles else kind},
        )
        if clear:
            rec.kv["clear"] = "yes"
        if wait:
            rec.kv["wait"] = ",".join(wait)
        lat_recs.append(rec)
        lat_by_name[name] = rec
        warnings.append(
            f"latency class {name} synthesized: add it to the overlay's [lat] section"
        )
        return name

    out_inst: list[Record] = []
    last_group = None
    for row in rows:
        if row.group and row.group != last_group:
            out_inst.append(Record("group", [row.group]))
            last_group = row.group
        if row.fmt not in enc_of_fmt:
            raise DescError(f"{row.name}: format {row.fmt} is not a mir.rs format")
        flags = _flag_set(row.flags, mir.aliases)
        unknown = flags - set(mir.flags)
        if unknown:
            raise DescError(f"{row.name}: unknown flags {sorted(unknown)}")
        flags_int = sum(1 << mir.flags[f] for f in flags)
        ov = inst_over.get(row.name)
        prev = old_insts.get(row.name)
        # Classification: post.rs where it states it; else the previous
        # description; else the baseline rules.
        bsched, bkind, bclear, bcycles, bwait, bvalu = base_model(row.fmt, flags)
        psched, pkind, pclear, pcycles, pwait, pvalu = post_model(
            post, row.fmt, flags_int, flags, row.name
        )
        p = prev_model(prev)
        if p is not None:
            bsched, bkind, bclear, bcycles, bwait = p[0], p[1], p[2], p[3], list(p[4])
        model = (
            psched if psched is not None else bsched,
            pkind if pkind is not None else bkind,
            pclear if pclear is not None else bclear,
            pcycles if pcycles is not None else bcycles,
            tuple(sorted(pwait if pwait is not None else bwait)),
        )
        valu = (
            pvalu
            if pvalu is not None
            else (prev.get("unit") in ("VALU", "TRANS", "WMMA") if prev else bvalu)
        )
        kv: dict[str, str] = {}
        enc = enc_of_fmt[row.fmt]
        if ov and ov.get("enc"):
            enc = ov.get("enc")
            kv["fmt"] = row.fmt
        kv["flags"] = "|".join(row.flags) if row.flags else "0"
        sig = derive_signature(row.fmt, row.mnemonic, flags)
        for k in ("d", "u", "imm", "m"):
            if sig[k]:
                kv[k] = sig[k]
        imp = [flag_imp[f] for f in sorted(flags) if f in flag_imp]
        fmt_imp = (fmt_recs.get(row.fmt) or Record("fmt")).get("imp")
        imp += [x for x in (fmt_imp or "").split(",") if x and x not in imp]
        if imp:
            kv["imp"] = ",".join(imp)
        kv["unit"] = derive_unit(row.fmt, flags, valu)
        kv["lat"] = lat_for(model)
        if "WMMA" in flags:
            kv["haz"] = "wmma_d,wmma_ab"
        if row.name in vopd_forms:
            code, y_only = vopd_forms[row.name]
            kv["vopd"] = f"{code}{'y' if y_only else ''}"
        if row.name in vopd.special:
            kv["vopd_special"] = "?"
        if ov:
            for k, v in ov.kv.items():
                if k == "enc":
                    continue
                if k == "imp+":
                    have = kv["imp"].split(",") if kv.get("imp") else []
                    kv["imp"] = ",".join(
                        have + [x for x in v.split(",") if x not in have]
                    )
                elif k == "lat":
                    if v not in lat_by_name or _lat_model(lat_by_name[v]) != model:
                        raise DescError(
                            f"{ov.where}: {row.name}: lat={v} does not have the model post.rs gives ({model})"
                        )
                    kv["lat"] = v
                elif k == "vopd_special" and row.name not in vopd.special:
                    warnings.append(
                        f"overlay: {row.name} has vopd_special but vopd.rs's part() does not single it out"
                    )
                elif v == "-":
                    kv.pop(k, None)
                else:
                    kv[k] = v
        if row.name in legacy:
            kv["table"] = "no"
        if row.note and "note" not in kv:
            kv["note"] = row.note
        kv = {k: kv[k] for k in INST_KEYS if k in kv}
        out_inst.append(
            Record("inst", [row.name, row.mnemonic, enc, f"0x{row.code:02x}"], kv)
        )

    # The VCmp family: flags differ by code; every code must classify alike.
    fam_models = set()
    fam_prev = prev_model(
        next((r for r in old.get("inst", []) if r.kind == "family"), None)
    )
    for code in (0x41, 0x51):
        fl = vcmp_flag_names(code)
        fi = sum(1 << mir.flags[f] for f in fl)
        b = base_model("Vopc", fl)
        if fam_prev is not None:
            b = (*fam_prev[:4], list(fam_prev[4]), b[5])
        p = post_model(post, "Vopc", fi, fl, "VCmp(..)")
        fam_models.add(
            (
                p[0] if p[0] is not None else b[0],
                p[1] if p[1] is not None else b[1],
                p[2] if p[2] is not None else b[2],
                p[3] if p[3] is not None else b[3],
                tuple(sorted(p[4] if p[4] is not None else b[4])),
            )
        )
    if len(fam_models) != 1:
        raise DescError(f"VCmp codes classify differently: {fam_models}")
    vc = {
        "flags": "macro",
        "d": "sdst:s1",
        "u": "src0:vs1,src1:vs1",
        "m": "neg,opsel",
        "imp": "use:exec",
    }
    vc.update(unit="VALU", lat=lat_for(fam_models.pop()))
    if "VCmp" in inst_over:
        vc.update({k: v for k, v in inst_over["VCmp"].kv.items() if k != "enc"})
    out_inst.append(
        Record(
            "family",
            ["VCmp", "v_cmp_*", enc_of_fmt.get("Vopc", "VOPC"), "code"],
            {k: vc[k] for k in INST_KEYS if k in vc},
        )
    )

    # VOPD tables.
    vopd_hw = {int(r.args[0]): r for r in overlay.get("vopd", [])}
    for n, (code, y_only) in vopd_forms.items():
        hw = vopd_hw.get(code)
        if hw is None:
            raise DescError(
                f"{n}: VOPD component {code} is not in the overlay's [vopd] table"
            )
        if y_only != (hw.args[2] == "y"):
            raise DescError(
                f"{n}: VOPD component {code} is {hw.args[2]} in [vopd] but Y-only={y_only} in vopd.rs"
            )
    if mir.vopd_names is not None:
        for code, name in mir.vopd_names.items():
            hw = vopd_hw.get(code)
            if hw is None or hw.args[1] != name:
                raise DescError(
                    f"mir.rs vopd_name({code}) = {name} disagrees with the [vopd] table"
                )
        named = set(mir.vopd_names)
    else:
        named = {int(r.args[0]) for r in old.get("vopd", []) if r.get("named") == "yes"}
    vopd_out = []
    for code, r in sorted(vopd_hw.items()):
        kv = dict(r.kv)
        if code in named:
            kv["named"] = "yes"
        vopd_out.append(Record("vopd", list(r.args), kv))

    secs: dict[str, list[Record]] = {}
    secs["meta"] = [
        Record(r.kind, list(r.args), dict(r.kv)) for r in overlay.get("meta", [])
    ]
    secs["flags"] = [
        Record(
            "flag",
            [n],
            {"bit": str(b), **({"imp": flag_imp[n]} if n in flag_imp else {})},
        )
        for n, b in mir.flags.items()
    ]
    secs["alias"] = [
        Record("alias", [n], {"flags": "|".join(v)}) for n, v in mir.aliases.items()
    ]
    secs["mods"] = [Record("mod", [n], {"bit": str(b)}) for n, b in mir.mods.items()]
    for s in ("modset",):
        secs[s] = [Record(r.kind, list(r.args), dict(r.kv)) for r in overlay.get(s, [])]
    secs["fmt"] = fmt_out
    for s in ("enc", "opnd", "unit"):
        secs[s] = [Record(r.kind, list(r.args), dict(r.kv)) for r in overlay.get(s, [])]
    secs["lat"] = lat_recs
    secs["hazard"] = [
        Record(r.kind, list(r.args), dict(r.kv)) for r in overlay.get("hazard", [])
    ]
    secs["vopd"] = vopd_out
    secs["vcmp"] = [
        Record("vcmp", [f"0x{c:02x}", mn], {"src": ty}) for c, mn, ty in vcmp_table()
    ]
    secs["inst"] = out_inst
    for s in ("burs",):
        secs[s] = [Record(r.kind, list(r.args), dict(r.kv)) for r in overlay.get(s, [])]
    validate(secs, set(enc_recs), fmt_of_enc_name)
    return Imported(write_description(secs), warnings)


def validate(
    secs: dict[str, list[Record]], enc_names: set[str], fmt_of_enc: dict[str, str]
) -> None:
    lats = {r.args[0] for r in secs["lat"]}
    units = {r.args[0] for r in secs["unit"]}
    hazards = {r.args[0] for r in secs["hazard"]}
    modsets = {r.args[0] for r in secs["modset"]}
    fmt_recs = {r.args[0]: r for r in secs["fmt"]}
    for rec in secs["inst"]:
        if rec.kind == "group":
            continue
        name = rec.args[0]
        if rec.get("lat") not in lats:
            raise DescError(f"{name}: unknown latency class {rec.get('lat')}")
        if rec.get("unit") not in units:
            raise DescError(f"{name}: unknown unit {rec.get('unit')}")
        for h in filter(None, (rec.get("haz") or "").split(",")):
            if h not in hazards:
                raise DescError(f"{name}: unknown hazard {h}")
        for m in filter(None, (rec.get("m") or "").split(",")):
            if m not in modsets:
                raise DescError(f"{name}: unknown modifier set {m}")
        if rec.kind == "family":
            continue
        fmt = rec.get("fmt") or fmt_of_enc.get(rec.args[2])
        f = fmt_recs.get(fmt)
        if f is None:
            raise DescError(f"{name}: encoding {rec.args[2]} has no [fmt] entry")
        if f.get("table", "yes") == "yes" and rec.get("table") != "no":
            layout = f.get("layout", f.get("enc"))
            if layout not in enc_names:
                raise DescError(f"{name}: format {fmt} has no layout {layout} in [enc]")


SECTION_DOC = {
    "meta": "Target facts.",
    "flags": "Compiler flag bits (mir.rs F_*), with the implicit register each one states.",
    "alias": "Flag aliases (mir.rs `const NAME: u32 = F_..`).",
    "mods": "MInst modifier bits (mir.rs M_*).",
    "modset": "Modifier sets an instruction may accept (`m=`), by M_ bit.",
    "fmt": "Rust formats (mir.rs Fmt) and their encodings.",
    "enc": "Encoding layouts: fixed bits per word, fields as name:word:lsb:width.",
    "opnd": "Operand codes and classes.",
    "unit": "Execution units.",
    "lat": "Latency classes: scheduler result latency, s_delay_alu class and cycles, wait counters.",
    "hazard": "Hazards and where the compiler handles them.",
    "vopd": "VOPD components: code, mnemonic, half (x: either, y: Y only); named=yes when mir.rs names it.",
    "vcmp": "VOPC compares the compiler names (Opc::VCmp(code)).",
    "inst": "Instructions: variant, mnemonic, encoding, opcode, then key=value facts (keys in the overlay header).",
    "burs": "BURS grammar for integer and address arithmetic (see the overlay header).",
}


def write_description(secs: dict[str, list[Record]]) -> str:
    out = [
        "# gfx11 instruction description for the direct compiler (rust/rocke-codeobj).",
        f"# Written by `{TOOL} import` from the Rust sources and isa/gfx11.overlay;",
        "# keep hand-written knowledge in the overlay. After `switch`, this file is the",
        "# source of the opcode rows, VOPD forms and latency classes. Keys are",
        "# documented in the overlay.",
    ]
    for name, recs in secs.items():
        out.append("")
        out.append(f"[{name}]")
        out.append(f"# {SECTION_DOC.get(name, '')}")
        widths = [7, 18, 26, 8, 7] if name == "inst" else None
        for r in recs:
            if name == "inst" and r.kind == "group":
                out.append(f"group {r.args[0]}")
                continue
            out.append(format_record(r, widths))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Generation: description -> Rust
# --------------------------------------------------------------------------


def camel(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_\-]", name) if p)


@dataclass
class Desc:
    secs: dict[str, list[Record]]
    text: str

    def section(self, name: str) -> list[Record]:
        return self.secs.get(name, [])

    @property
    def insts(self) -> list[Record]:
        return [r for r in self.section("inst") if r.kind == "inst"]

    @property
    def family(self) -> Record:
        return next(r for r in self.section("inst") if r.kind == "family")

    def fmt_recs(self) -> dict[str, Record]:
        return {r.args[0]: r for r in self.section("fmt")}

    def fmt_of(self, rec: Record) -> str:
        if rec.get("fmt"):
            return rec.get("fmt")
        return {r.get("enc"): r.args[0] for r in self.section("fmt")}[rec.args[2]]

    def aliases(self) -> set[str]:
        return {r.args[0] for r in self.section("alias")}


def _wrap_alts(alts: list[str], indent: int, width: int = 112) -> str:
    lines, cur = [], ""
    for a in alts:
        piece = a if not cur else f" | {a}"
        if cur and len(cur) + len(piece) + indent > width:
            lines.append(cur)
            cur = f"| {a}"
        else:
            cur += piece
    lines.append(cur)
    return ("\n" + " " * indent).join(lines)


def _match_fn(
    name: str, ret: str, doc: str, arms: dict[str, list[str]], family: str | None
) -> str:
    out = [
        f"/// {doc}",
        "#[inline]",
        f"pub fn {name}(op: Opc) -> {ret} {{",
        "    match op {",
    ]
    for value, ops in arms.items():
        out.append(f"        {_wrap_alts([f'Opc::{o}' for o in ops], 8)} => {value},")
    if family is not None:
        out.append(f"        Opc::VCmp(_) => {family},")
    out.append("    }")
    out.append("}")
    return "\n".join(out)


def _group(insts: list[Record], value) -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {}
    for r in insts:
        arms.setdefault(value(r), []).append(r.args[0])
    return arms


def parse_sig_list(text: str | None) -> list[tuple[str, str, int]]:
    out = []
    for item in filter(None, (text or "").split(",")):
        role, _, cls = item.partition(":")
        m = re.fullmatch(r"([a-z]+?)(\d*)", cls)
        if not m:
            raise DescError(f"bad operand class {cls!r}")
        out.append((role, m.group(1), int(m.group(2) or 1)))
    return out


def _wrap_text(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def _chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def generate_opcodes(desc: Desc) -> str:
    aliases = desc.aliases()
    lines = [
        HEADER_RS,
        f"// @generated by `{TOOL} generate` from `{DESCRIPTION}`. Do not edit.",
        "// The rows of mir.rs's `opcodes!` table: variant = (format, opcode, mnemonic,",
        "// flags). Change the description (or its overlay), then rerun the tool.",
        "",
        "opcodes! {",
    ]
    for rec in desc.section("inst"):
        if rec.kind == "group":
            lines.append(f"    // {rec.args[0]}")
            continue
        if rec.kind != "inst":
            continue
        name, mn, _enc, code = rec.args[:4]
        flags = _flags_field(rec.get("flags"))
        fexpr = " | ".join(t if t in aliases else f"F_{t}" for t in flags) or "0"
        if rec.get("note"):
            for chunk in _wrap_text(rec.get("note"), 96):
                lines.append(f"    // {chunk}")
        lines.append(
            f'    {name} = ({desc.fmt_of(rec)}, 0x{_int(code):02x}, "{mn}", {fexpr}),'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _enc_structs(desc: Desc) -> str:
    out = []
    for rec in desc.section("enc"):
        name = rec.args[0]
        words = int(rec.get("words", "1"))
        fixed = [_int(x) for x in rec.get("fixed", "0").split(",")]
        fixed += [0] * (words - len(fixed))
        fields = []
        for f in rec.get("fields", "").split(","):
            fname, w, lsb, width = f.split(":")
            fields.append((fname, int(w), int(lsb), int(width)))
        sname = "Enc" + camel(name.lower())
        doc = rec.args[1] if len(rec.args) > 1 else name
        out.append(f"/// {name}: {doc}")
        out.append("#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]")
        out.append(f"pub struct {sname} {{")
        for fname, *_ in fields:
            out.append(f"    pub {fname}: u32,")
        out.append("}")
        out.append("")
        out.append(f"impl {sname} {{")
        out.append("    #[inline(always)]")
        ret = "u32" if words == 1 else f"[u32; {words}]"
        out.append(f"    pub fn pack(&self) -> {ret} {{")
        exprs = []
        for w in range(words):
            terms = [f"0x{fixed[w] >> 16:04x}_{fixed[w] & 0xFFFF:04x}"]
            for fname, fw, lsb, width in sorted(fields, key=lambda x: x[2]):
                if fw != w:
                    continue
                t = f"(self.{fname} & 0x{(1 << width) - 1:x})"
                terms.append(f"{t} << {lsb}" if lsb else t)
            exprs.append(" | ".join(terms))
        if words == 1:
            out.append(f"        {exprs[0]}")
        else:
            out.append("        [")
            for e in exprs:
                out.append(f"            {e},")
            out.append("        ]")
        out.append("    }")
        out.append("}")
        out.append("")
    return "\n".join(out)


def generate_isa(desc: Desc, digest: int) -> str:
    insts = desc.insts
    fam = desc.family
    fmt_recs = desc.fmt_recs()
    o: list[str] = [
        HEADER_RS,
        f"//! @generated by `{TOOL} generate` from `{DESCRIPTION}`. Do not edit:",
        f"//! change the overlay (`{OVERLAY}`) or the description, then run",
        "//! `isa_tables.py import` and `isa_tables.py generate`.",
        "//!",
        "//! Instruction tables of the gfx11 description: encoding layouts, operand",
        "//! codes and signatures, encoding forms, units, latency classes, wait",
        "//! counters, hazards and VOPD forms.",
        "",
        "#![allow(dead_code)]",
        "",
        "use super::mir::{Fmt, Opc};",
        "",
        "/// FNV-1a 64 of the description (line endings as LF) this file was generated from.",
        f"pub const DESCRIPTION_DIGEST: u64 = 0x{digest:016x};",
        "",
        "// ----- operand codes -----",
        "",
    ]
    f32_pairs: list[tuple[int, int]] = []
    f64_pairs: list[tuple[int, int]] = []
    for rec in desc.section("opnd"):
        if rec.kind == "reg":
            o.append(
                f"pub const REG_{rec.args[0].upper()}: u32 = {_int(rec.get('code'))};"
            )
        elif rec.kind == "src":
            for k, v in rec.kv.items():
                o.append(f"pub const SRC_{k.upper()}: u32 = {_int(v)};")
        elif rec.kind == "inline_int":
            o.append(f"pub const INLINE_INT_MIN: i32 = {int(rec.get('lo'))};")
            o.append(f"pub const INLINE_INT_MAX: i32 = {int(rec.get('hi'))};")
            o.append(f"pub const INLINE_INT_ZERO: u32 = {int(rec.get('zero'))};")
            o.append(f"pub const INLINE_INT_NEG: u32 = {int(rec.get('neg'))};")
        elif rec.kind == "inline_f32":
            f32_pairs = [(_int(b), int(c)) for b, c in (a.split("=") for a in rec.args)]
            o.append("/// f32 inline constants: (bits, operand code).")
            items = ", ".join(f"(0x{b:08x}, {c})" for b, c in f32_pairs)
            o.append(
                f"pub const INLINE_F32: [(u32, u32); {len(f32_pairs)}] = [{items}];"
            )
        elif rec.kind == "inline_f64":
            f64_pairs = [(_int(b), int(c)) for b, c in (a.split("=") for a in rec.args)]
            o.append(
                "/// f64 inline constants: (high dword, operand code); the low dword is zero."
            )
            items = ", ".join(f"(0x{b:08x}, {c})" for b, c in f64_pairs)
            o.append(
                f"pub const INLINE_F64: [(u32, u32); {len(f64_pairs)}] = [{items}];"
            )
    o += [
        "",
        "/// Operand code of an inline integer constant.",
        "#[inline]",
        "pub fn inline_int(v: u32) -> Option<u32> {",
        "    let s = v as i32;",
        "    if (0..=INLINE_INT_MAX).contains(&s) {",
        "        Some(INLINE_INT_ZERO + s as u32)",
        "    } else if (INLINE_INT_MIN..0).contains(&s) {",
        "        Some(INLINE_INT_NEG + (-s) as u32)",
        "    } else {",
        "        None",
        "    }",
        "}",
        "",
        "/// Operand code of an f32 inline constant.",
        "#[inline]",
        "pub fn inline_f32(v: u32) -> Option<u32> {",
        "    match v {",
        *[f"        0x{b:08x} => Some({c})," for b, c in f32_pairs],
        "        _ => None,",
        "    }",
        "}",
        "",
        "/// Operand code of the f64 inline constant with high dword `hi` (and a zero low dword).",
        "#[inline]",
        "pub fn inline_f64(hi: u32) -> Option<u32> {",
        "    match hi {",
        *[f"        0x{b:08x} => Some({c})," for b, c in f64_pairs],
        "        _ => None,",
        "    }",
        "}",
        "",
        "// ----- encoding layouts -----",
        "",
        _enc_structs(desc),
        "// ----- per-opcode facts -----",
        "",
    ]
    names = [r.args[0] for r in insts]
    o.append(
        "/// The named opcodes, in description order (the `VCmp` family excluded)."
    )
    o.append(f"pub const ALL: [Opc; {len(names)}] = [")
    for chunk in _chunks([f"Opc::{n}" for n in names], 6):
        o.append("    " + ", ".join(chunk) + ",")
    o.append("];")
    o.append("")
    o.append(
        _match_fn(
            "variant",
            "&'static str",
            "Rust variant name of an opcode.",
            {f'"{n}"': [n] for n in names},
            '"VCmp"',
        )
    )
    o.append("")

    o.append("/// Execution unit.")
    o.append("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]")
    o.append("pub enum Unit {")
    for rec in desc.section("unit"):
        o.append(f"    /// {rec.args[1] if len(rec.args) > 1 else rec.args[0]}")
        o.append(f"    {camel(rec.args[0].lower())},")
    o.append("}")
    o.append("")
    o.append(
        _match_fn(
            "unit",
            "Unit",
            "Execution unit of an opcode.",
            _group(insts, lambda r: f"Unit::{camel(r.get('unit').lower())}"),
            f"Unit::{camel(fam.get('unit').lower())}",
        )
    )
    o.append("")
    unit_issue = {r.args[0]: r.get("issue") for r in desc.section("unit")}

    def issue_value(r: Record) -> str:
        return str(int(r.get("issue") or unit_issue.get(r.get("unit")) or 1))

    o.append(
        _match_fn(
            "issue",
            "u32",
            "Issue cycles of an opcode on one SIMD (the BURS prototype's cost; nothing else reads it).",
            _group(insts, issue_value),
            issue_value(fam),
        )
    )
    o.append("")
    o.append(
        "/// Whether an opcode issues to the vector ALU (VALU, transcendental or WMMA)."
    )
    o.append("#[inline]")
    o.append("pub fn is_valu(op: Opc) -> bool {")
    o.append("    matches!(unit(op), Unit::Valu | Unit::Trans | Unit::Wmma)")
    o.append("}")
    o.append("")

    lat_recs = desc.section("lat")
    o += [
        "/// s_delay_alu class of an instruction: which kind of result a reader",
        "/// waits for (`Other`: none; `Skip`: a pseudo, which the pass passes by).",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Delay {",
        "    Valu,",
        "    Trans,",
        "    Salu,",
        "    Other,",
        "    Skip,",
        "}",
        "",
        "/// Wait counters an instruction increments: VMEM loads (`vmcnt`), LDS and",
        "/// SMEM (`lgkmcnt`), SMEM alone (returns out of order).",
        "pub const CNT_VM: u8 = 1;",
        "pub const CNT_LGKM: u8 = 2;",
        "pub const CNT_SMEM: u8 = 4;",
        "",
        "/// Latency class.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Lat {",
    ]
    for rec in lat_recs:
        o.append(f"    /// {rec.args[1] if len(rec.args) > 1 else rec.args[0]}")
        o.append(f"    {camel(rec.args[0])},")
    o.append("}")
    o.append("")
    o.append("impl Lat {")
    o.append(
        "    /// Cycles from issue until the scheduler lets a reader of the result issue."
    )
    o.append("    #[inline]")
    o.append("    pub fn sched(self) -> u32 {")
    o.append("        match self {")
    for rec in lat_recs:
        o.append(f"            Lat::{camel(rec.args[0])} => {int(rec.get('sched'))},")
    o.append("        }")
    o.append("    }")
    o.append("")
    o.append("    /// s_delay_alu class and the cycles a reader of the result waits.")
    o.append("    #[inline]")
    o.append("    pub fn delay(self) -> (Delay, u8) {")
    o.append("        match self {")
    for rec in lat_recs:
        dk, _, dc = rec.get("delay").partition(":")
        o.append(
            f"            Lat::{camel(rec.args[0])} => (Delay::{camel(dk)}, {int(dc or 0)}),"
        )
    o.append("        }")
    o.append("    }")
    o.append("")
    o.append("    /// Whether the instruction clears the s_delay_alu state (a memory")
    o.append("    /// instruction, which waits for VALU results itself).")
    o.append("    #[inline]")
    o.append("    pub fn delay_clears(self) -> bool {")
    clears = [f"Lat::{camel(r.args[0])}" for r in lat_recs if r.get("clear") == "yes"]
    o.append(
        f"        matches!(self, {' | '.join(clears)})" if clears else "        false"
    )
    o.append("    }")
    o.append("")
    o.append("    /// Wait counters (`CNT_*`) the instruction increments.")
    o.append("    #[inline]")
    o.append("    pub fn counters(self) -> u8 {")
    o.append("        match self {")
    for rec in lat_recs:
        w = [
            f"CNT_{x.upper()}" for x in filter(None, (rec.get("wait") or "").split(","))
        ]
        o.append(f"            Lat::{camel(rec.args[0])} => {' | '.join(w) or '0'},")
    o.append("        }")
    o.append("    }")
    o.append("}")
    o.append("")
    o.append(
        _match_fn(
            "lat",
            "Lat",
            "Latency class of an opcode.",
            _group(insts, lambda r: f"Lat::{camel(r.get('lat'))}"),
            f"Lat::{camel(fam.get('lat'))}",
        )
    )
    for fn_name, ret, doc, expr in (
        (
            "result_latency",
            "u32",
            "Scheduler result latency of an opcode.",
            "lat(op).sched()",
        ),
        (
            "delay",
            "(Delay, u8)",
            "s_delay_alu class and cycles of an opcode.",
            "lat(op).delay()",
        ),
        (
            "delay_clears",
            "bool",
            "Whether an opcode clears the s_delay_alu state.",
            "lat(op).delay_clears()",
        ),
        ("counters", "u8", "Wait counters an opcode increments.", "lat(op).counters()"),
    ):
        o += [
            "",
            f"/// {doc}",
            "#[inline]",
            f"pub fn {fn_name}(op: Opc) -> {ret} {{",
            f"    {expr}",
            "}",
        ]
    o.append("")

    for i, rec in enumerate(desc.section("hazard")):
        o.append(f"/// {rec.args[1] if len(rec.args) > 1 else rec.args[0]}")
        o.append(f"pub const HZ_{rec.args[0].upper()}: u32 = 1 << {i};")
    o.append("")

    def hz_value(r: Record) -> str:
        hs = [f"HZ_{h.upper()}" for h in filter(None, (r.get("haz") or "").split(","))]
        return " | ".join(hs) or "0"

    o.append(
        _match_fn(
            "hazards",
            "u32",
            "Hazards an opcode takes part in (`HZ_*`).",
            _group(insts, hz_value),
            hz_value(fam),
        )
    )
    o.append("")

    def vopd_value(r: Record) -> str:
        v = r.get("vopd")
        return (
            f"Some(({int(v.rstrip('y'))}, {str(v.endswith('y')).lower()}))"
            if v
            else "None"
        )

    o.append(
        _match_fn(
            "vopd",
            "Option<(u8, bool)>",
            "VOPD component code of an opcode's plain form and whether it may only be the Y half.",
            _group(insts, vopd_value),
            "None",
        )
    )
    o.append("")

    def special_value(r: Record) -> str:
        v = r.get("vopd_special")
        return (
            "None" if not v else ("Some(None)" if v == "?" else f"Some(Some({int(v)}))")
        )

    o.append(
        _match_fn(
            "vopd_special",
            "Option<Option<u8>>",
            "Opcodes vopd.rs pairs through their own path (`Some`), with the component code when known.",
            _group(insts, special_value),
            "None",
        )
    )
    o.append("")
    o.append("/// Mnemonic of a VOPD component opcode the compiler names.")
    o.append("pub fn vopd_name(code: u8) -> &'static str {")
    o.append("    match code {")
    for rec in desc.section("vopd"):
        if rec.get("named") == "yes":
            o.append(f'        {int(rec.args[0])} => "{rec.args[1]}",')
    o.append('        _ => "v_dual_?",')
    o.append("    }")
    o.append("}")
    o.append("")
    o.append(
        "/// Mnemonic of any VOPD component the hardware has, and whether it is Y only."
    )
    o.append("pub fn vopd_component(code: u8) -> Option<(&'static str, bool)> {")
    o.append("    match code {")
    for rec in desc.section("vopd"):
        o.append(
            f'        {int(rec.args[0])} => Some(("{rec.args[1]}", {str(rec.args[2] == "y").lower()})),'
        )
    o.append("        _ => None,")
    o.append("    }")
    o.append("}")
    o.append("")

    forms = ["plain"]
    for r in insts:
        f = r.get("form")
        if f and f not in forms:
            forms.append(f)
    o.append("/// How the table-driven encoder treats an opcode within its format.")
    o.append("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]")
    o.append("pub enum Form {")
    form_doc = {
        r.args[0]: (r.args[1] if len(r.args) > 1 else r.args[0])
        for r in desc.section("opnd")
        if r.kind == "form"
    }
    for f in forms:
        o.append(f"    /// {form_doc.get(f, f)}")
        o.append(f"    {camel(f)},")
    o.append("}")
    o.append("")

    def table_value(r: Record) -> str:
        f = fmt_recs[desc.fmt_of(r)]
        return (
            "false"
            if f.get("table", "yes") == "no" or r.get("table") == "no"
            else "true"
        )

    def vop3_value(r: Record, code: int) -> int:
        return code + _int(fmt_recs[desc.fmt_of(r)].get("vop3") or "0")

    bits = {r.args[0]: int(r.get("bit")) for r in desc.section("flags")}
    alias = {r.args[0]: r.get("flags").split("|") for r in desc.section("alias")}

    def flags_value(r: Record) -> int:
        v = 0
        for t in _flags_field(r.get("flags")):
            for f in alias.get(t, [t]):
                v |= 1 << bits[f]
        return v

    def info_row(r: Record, code: int, flags: int) -> str:
        return (
            f"    Info {{ code: 0x{code:03x}, vop3: 0x{vop3_value(r, code):03x}, flags: 0x{flags:05x}, "
            f"fmt: Fmt::{desc.fmt_of(r)}, form: Form::{camel(r.get('form') or 'plain')}, table: {table_value(r)} }},"
        )

    o += [
        "/// What the table-driven encoder reads of an opcode, in one load.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub struct Info {",
        "    /// Opcode field (`Opc::code`).",
        "    pub code: u16,",
        "    /// Opcode of the VOP3 encoding (VOP1 and VOP2 promoted, VOPC, native VOP3).",
        "    pub vop3: u16,",
        "    /// `Opc::flags`.",
        "    pub flags: u32,",
        "    pub fmt: Fmt,",
        "    pub form: Form,",
        "    /// Whether the table-driven encoder encodes the opcode (else asm.rs's hand encoder does).",
        "    pub table: bool,",
        "}",
        "",
        "/// `Info` of each opcode in `ALL` order, then the compare family's",
        "/// fields that do not depend on its code.",
        f"static INFO: [Info; {len(insts) + 1}] = [",
    ]
    o += [info_row(r, _int(r.args[3]), flags_value(r)) for r in insts]
    o += [info_row(fam, 0, 0), "];", ""]
    o += [
        "/// Position of an opcode in `INFO`.",
        "#[inline(always)]",
        "fn index(op: Opc) -> usize {",
        "    match op {",
    ]
    o += [f"        Opc::{r.args[0]} => {i}," for i, r in enumerate(insts)]
    o += [f"        Opc::VCmp(_) => {len(insts)},", "    }", "}", ""]
    fam_vop3 = vop3_value(fam, 0)
    o += [
        "/// `Info` of an opcode. A match whose arms compute a value compiles to a",
        "/// jump table, so the compare family's code-dependent fields are patched",
        "/// in after the load.",
        "#[inline]",
        "pub fn info(op: Opc) -> Info {",
        "    let mut info = INFO[index(op)];",
        "    if let Opc::VCmp(code) = op {",
        "        info.code = code as u16;",
        f"        info.vop3 = code as u16{f' + 0x{fam_vop3:x}' if fam_vop3 else ''};",
        "        info.flags = op.flags();",
        "    }",
        "    info",
        "}",
        "",
        "/// Encoding form of an opcode.",
        "#[inline]",
        "pub fn form(op: Opc) -> Form {",
        "    info(op).form",
        "}",
        "",
        "/// Whether the table-driven encoder encodes an opcode (else asm.rs's hand encoder does).",
        "#[inline]",
        "pub fn table_encoded(op: Opc) -> bool {",
        "    info(op).table",
        "}",
        "",
    ]

    roles: list[str] = []
    classes: list[str] = []
    imms: list[str] = [""]

    def note(lst: list[str], v: str) -> None:
        if v not in lst:
            lst.append(v)

    all_recs = insts + [fam]
    for r in all_recs:
        for key in ("d", "u"):
            for role, cls, _n in parse_sig_list(r.get(key)):
                note(roles, role)
                note(classes, cls)
        note(imms, r.get("imm") or "")
    modbits = {
        r.args[0]: [int(b) for b in r.get("bits").split(",")]
        for r in desc.section("modset")
    }
    class_doc = {
        r.args[0]: (r.args[1] if len(r.args) > 1 else r.args[0])
        for r in desc.section("opnd")
        if r.kind == "class"
    }
    imm_doc = {
        r.args[0]: (r.args[1] if len(r.args) > 1 else r.args[0])
        for r in desc.section("opnd")
        if r.kind == "imm"
    }
    o += [
        "/// Operand role: the encoding field an operand goes to.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Role {",
    ]
    o += [f"    {camel(v)}," for v in roles]
    o += [
        "}",
        "",
        "/// Operand class.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Class {",
    ]
    for v in classes:
        o.append(f"    /// {class_doc.get(v, v)}")
        o.append(f"    {camel(v)},")
    o += [
        "}",
        "",
        "/// What `MInst::imm` holds.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Imm {",
    ]
    for v in imms:
        o.append(f"    /// {imm_doc.get(v, 'nothing') if v else 'nothing'}")
        o.append(f"    {camel(v) or 'None'},")
    o += [
        "}",
        "",
        "/// One operand of a signature: role, class and size in dwords.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub struct Operand {",
        "    pub role: Role,",
        "    pub class: Class,",
        "    pub dwords: u8,",
        "}",
        "",
        "/// The operands an instruction takes, in `MInst` order, what its immediate",
        "/// holds and the modifier bits it accepts.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub struct Sig {",
        "    pub defs: &'static [Operand],",
        "    pub uses: &'static [Operand],",
        "    pub imm: Imm,",
        "    pub mods: u32,",
        "}",
        "",
    ]

    def operands(text: str | None) -> str:
        items = [
            f"Operand {{ role: Role::{camel(role)}, class: Class::{camel(cls)}, dwords: {n} }}"
            for role, cls, n in parse_sig_list(text)
        ]
        return "&[" + ", ".join(items) + "]"

    def mods_value(r: Record) -> int:
        v = 0
        for m in filter(None, (r.get("m") or "").split(",")):
            for b in modbits[m]:
                v |= 1 << b
        return v

    sigs: dict[tuple, str] = {}
    sig_of: dict[str, str] = {}

    def sig_const(key: tuple) -> str:
        if key not in sigs:
            sigs[key] = f"SIG_{len(sigs)}"
            o.append(f"const {sigs[key]}: Sig = Sig {{")
            o.append(f"    defs: {operands(key[0])},")
            o.append(f"    uses: {operands(key[1])},")
            o.append(f"    imm: Imm::{camel(key[2]) or 'None'},")
            o.append(f"    mods: 0x{key[3]:x},")
            o.append("};")
        return sigs[key]

    for r in all_recs:
        sig_of[r.args[0]] = sig_const(
            (r.get("d") or "", r.get("u") or "", r.get("imm") or "", mods_value(r))
        )
    vcmp64 = [
        int(c.args[0], 0)
        for c in desc.section("vcmp")
        if c.get("src") in ("i64", "u64", "f64")
    ]
    fam64 = sig_const(
        (
            fam.get("d") or "",
            (fam.get("u") or "").replace("vs1", "vs2"),
            fam.get("imm") or "",
            mods_value(fam),
        )
    )
    o.append("")
    arms: dict[str, list[str]] = {}
    for r in insts:
        arms.setdefault(f"&{sig_of[r.args[0]]}", []).append(r.args[0])
    out = [
        "/// Operand signature of an opcode.",
        "#[inline]",
        "pub fn signature(op: Opc) -> &'static Sig {",
        "    match op {",
    ]
    for value, ops in arms.items():
        out.append(f"        {_wrap_alts([f'Opc::{x}' for x in ops], 8)} => {value},")
    out.append(f"        Opc::VCmp(c) if VCMP64.contains(&c) => &{fam64},")
    out.append(f"        Opc::VCmp(_) => &{sig_of['VCmp']},")
    out += ["    }", "}"]
    o.append("\n".join(out))
    o.append("")
    o.append(
        f"const VCMP64: [u8; {len(vcmp64)}] = [{', '.join(f'0x{c:02x}' for c in vcmp64)}];"
    )
    o.append("")
    vc = [(int(c.args[0], 0), c.args[1]) for c in desc.section("vcmp")]
    o.append("/// Compares the compiler names: (code, mnemonic).")
    o.append(f"pub const VCMP: [(u8, &str); {len(vc)}] = [")
    for chunk in _chunks([f'(0x{c:02x}, "{n}")' for c, n in vc], 4):
        o.append("    " + ", ".join(chunk) + ",")
    o.append("];")
    o.append("")
    o.append("/// Format of an encoding family name (`[fmt]`).")
    o.append("pub fn fmt_of(enc: &str) -> Option<Fmt> {")
    o.append("    match enc {")
    for rec in desc.section("fmt"):
        o.append(f'        "{rec.get("enc")}" => Some(Fmt::{rec.args[0]}),')
    o.append("        _ => None,")
    o.append("    }")
    o.append("}")
    return "\n".join(o).rstrip() + "\n"


def load_desc(tree: Tree) -> Desc:
    text = tree.read(DESCRIPTION)
    return Desc(parse(text, DESCRIPTION), text)


def run_generate(tree: Tree) -> dict[str, str]:
    desc = load_desc(tree)
    digest = fnv1a64(desc.text.encode("utf-8"))
    out = {GEN_OPCODES: generate_opcodes(desc), GEN_ISA: generate_isa(desc, digest)}
    if desc.section("burs"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from isa_burs import generate_burs

        out[GEN_BURS] = generate_burs(desc)
    return out


# --------------------------------------------------------------------------
# Switch-over: anchored, idempotent edits that keep each file's line endings
# --------------------------------------------------------------------------


@dataclass
class Hunk:
    file: str
    name: str
    done: str
    apply: object  # callable(text: str) -> str | None


def _replace_once(text: str, old: str, new: str) -> str | None:
    return text.replace(old, new, 1) if text.count(old) == 1 else None


def _sub_once(text: str, pattern: re.Pattern, repl) -> str | None:
    found = list(pattern.finditer(text))
    if len(found) != 1:
        return None
    m = found[0]
    return text[: m.start()] + (repl(m) if callable(repl) else repl) + text[m.end() :]


def _line_start(text: str, i: int) -> int:
    return text.rfind("\n", 0, i) + 1


def _indent_at(text: str, i: int) -> str:
    s = _line_start(text, i)
    return text[s:i] if text[s:i].strip() == "" else re.match(r"\s*", text[s:]).group(0)


def _mir_rows(text: str) -> str | None:
    inv = re.search(r"^opcodes!\s*\{", text, re.MULTILINE)
    if not inv:
        return None
    start = text.index("{", inv.start())
    end = _matching_brace(text, start)
    new = (
        "// The rows are generated from the gfx11 instruction description\n"
        f"// (`{DESCRIPTION}`) by `{TOOL} generate`.\n"
        'include!("gen_opcodes.rs");'
    )
    return text[: inv.start()] + new + text[end + 1 :]


def _mir_vopd_name(text: str) -> str | None:
    span = _body(text, r"^pub fn vopd_name\(code: u8\) -> &'static str")
    if not span:
        return None
    return (
        text[: span[0]]
        + "{\n    super::gen_isa::vopd_name(code)\n}"
        + text[span[1] + 1 :]
    )


def _vopd_part(text: str) -> str | None:
    m = re.search(r"let \(code, y_only\) = match inst\.op \{", text)
    if not m or text.count("let (code, y_only) = match inst.op {") != 1:
        return None
    start = text.index("{", m.start())
    end = _matching_brace(text, start)
    if not text[end + 1 :].startswith(";"):
        return None
    return (
        text[: m.start()]
        + "let (code, y_only) = super::gen_isa::vopd(inst.op)?;"
        + text[end + 2 :]
    )


ASM_HOOK = """/// Encodes one instruction: the table-driven encoder over the gfx11
/// description (`isa.rs`), or the hand encoder below for the opcodes the
/// description leaves to it (`table_encoded`).
fn encode(inst: &MInst, e: &mut Enc) -> Result<(), CompileError> {
    if super::isa::encode(inst, &mut e.words)? {
        return Ok(());
    }
    encode_legacy(inst, e)
}

/// The hand encoder's words for one instruction: the reference the
/// table-driven encoder is tested against.
pub fn encode_legacy_words(inst: &MInst) -> Result<Vec<u32>, CompileError> {
    let mut e = Enc { words: Vec::new() };
    encode_legacy(inst, &mut e)?;
    Ok(e.words)
}

fn encode_legacy(inst: &MInst, e: &mut Enc) -> Result<(), CompileError> {"""


def _asm_hook(text: str) -> str | None:
    return _replace_once(
        text,
        "fn encode(inst: &MInst, e: &mut Enc) -> Result<(), CompileError> {",
        ASM_HOOK,
    )


def _replace_fn_body(text: str, sig: str, new_body: str) -> str | None:
    span = _fn_span(text, sig)
    if not span:
        return None
    return text[: span[0]] + "{\n" + new_body + "\n}" + text[span[1] + 1 :]


def _post_result_latency(text: str) -> str | None:
    span = _fn_span(text, r"fn result_latency\(inst: &MInst\) -> u32")
    if not span:
        return None
    return _replace_fn_body(
        text,
        r"fn result_latency\(inst: &MInst\) -> u32",
        "    super::gen_isa::result_latency(inst.op)",
    )


def _post_delay(text: str) -> str | None:
    m = re.search(r"let kind = match fmt \{", text)
    if not m:
        return None
    ms = text.index("{", m.start())
    me = _matching_brace(text, ms)
    cm = re.compile(
        r"\s*;\s*if (.+?) \{(\s*state\.clear\(\);\s*\} else if kind != AluKind::Other \{)",
        re.DOTALL,
    ).match(text, me + 1)
    if not cm:
        return None
    ind = _indent_at(text, m.start())
    new = (
        "let (class, latency) = super::gen_isa::delay(inst.op);\n"
        f"{ind}let kind = match class {{\n"
        f"{ind}    super::gen_isa::Delay::Valu => AluKind::Valu,\n"
        f"{ind}    super::gen_isa::Delay::Trans => AluKind::Trans,\n"
        f"{ind}    super::gen_isa::Delay::Salu => AluKind::Salu,\n"
        f"{ind}    _ => AluKind::Other,\n"
        f"{ind}}};\n"
        f"{ind}if super::gen_isa::delay_clears(inst.op) {{"
    )
    text = text[: m.start()] + new + cm.group(2) + text[cm.end() :]
    lm = re.search(r"let latency = match kind \{", text)
    if not lm:
        return None
    ls = text.index("{", lm.start())
    le = _matching_brace(text, ls)
    if text[le + 1] != ";":
        return None
    line_s = _line_start(text, lm.start())
    line_e = text.index("\n", le) + 1
    return text[:line_s] + text[line_e:]


def _post_is_valu(text: str) -> str | None:
    return _replace_fn_body(
        text, r"fn is_valu\(op: Opc\) -> bool", "    super::gen_isa::is_valu(op)"
    )


def _post_wait(text: str) -> str | None:
    def repl(m: re.Match) -> str:
        ind = _indent_at(text, m.start())
        return (
            "let counters = super::gen_isa::counters(inst.op);\n"
            f"{ind}if counters & super::gen_isa::CNT_VM != 0 {{\n"
            f"{ind}    state.vm.issue(&inst.defs);\n"
            f"{ind}    state.vm_any = true;\n"
            f"{ind}}}\n"
            f"{ind}if counters & super::gen_isa::CNT_LGKM != 0 {{\n"
            f"{ind}    state.lgkm.issue(&inst.defs);\n"
            f"{ind}    state.lgkm_any = true;\n"
            f"{ind}    if counters & super::gen_isa::CNT_SMEM != 0 {{\n"
            f"{ind}        state.smem = true;\n"
            f"{ind}    }}\n"
            f"{ind}}}"
        )

    return _sub_once(text, _WAIT_ISSUE, repl)


def _post_wmma(text: str) -> str | None:
    def repl(m: re.Match) -> str:
        nl = m.group(1)
        return (
            "if super::gen_isa::hazards(inst.op) & super::gen_isa::HZ_WMMA_AB != 0 {"
            f"{nl}let prev = out.iter().rev().find(|i| is_valu(i.op));"
            f"{nl}let hazard = |p: &MInst| {{"
            f"{nl}    super::gen_isa::hazards(p.op) & super::gen_isa::HZ_WMMA_D != 0"
        )

    return _sub_once(text, _WMMA_HAZARD, repl)


def _mod_hook(text: str) -> str | None:
    return _replace_once(
        text,
        "pub mod fx;\n",
        "pub mod fx;\n#[rustfmt::skip]\npub mod gen_isa;\npub mod isa;\n",
    )


def hunks() -> list[Hunk]:
    return [
        Hunk(MODRS, "mod: declare gen_isa and isa", "pub mod gen_isa;", _mod_hook),
        Hunk(
            MIR,
            "mir: opcode rows from gen_opcodes.rs",
            'include!("gen_opcodes.rs");',
            _mir_rows,
        ),
        Hunk(
            MIR,
            "mir: vopd_name from the VOPD table",
            "super::gen_isa::vopd_name(code)",
            _mir_vopd_name,
        ),
        Hunk(
            ASM,
            "asm: table-driven encoder first",
            "super::isa::encode(inst, &mut e.words)",
            _asm_hook,
        ),
        Hunk(
            VOPD,
            "vopd: VOPD forms from the table",
            "super::gen_isa::vopd(inst.op)?",
            _vopd_part,
        ),
        Hunk(
            POST,
            "post: result latency from the latency class",
            "super::gen_isa::result_latency(inst.op)",
            _post_result_latency,
        ),
        Hunk(
            POST,
            "post: s_delay_alu class from the latency class",
            "super::gen_isa::delay(inst.op)",
            _post_delay,
        ),
        Hunk(
            POST,
            "post: VALU test from the unit",
            "super::gen_isa::is_valu(op)",
            _post_is_valu,
        ),
        Hunk(
            POST,
            "post: wait counters from the latency class",
            "super::gen_isa::counters(inst.op)",
            _post_wait,
        ),
        Hunk(
            POST,
            "post: WMMA hazard from the hazard table",
            "super::gen_isa::HZ_WMMA_AB",
            _post_wmma,
        ),
    ] + burs_hunks()


def burs_hunks() -> list[Hunk]:
    """The BURS prototype's hooks in isel.rs (`isa_burs.py`)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from isa_burs import isel_hunks

        return [Hunk(ISEL, name, done, apply) for name, done, apply in isel_hunks()]
    except ImportError:
        return []


def run_switch(
    tree: Tree, dry_run: bool, skip: list[str]
) -> tuple[list[str], list[str]]:
    report, skipped = [], []
    texts: dict[str, str] = {}
    newlines: dict[str, str] = {}
    for h in hunks():
        if any(h.name.startswith(s) for s in skip):
            report.append(f"skip     {h.name} (--skip)")
            continue
        if not tree.exists(h.file):
            report.append(f"MISSING  {h.name}: no {h.file}")
            skipped.append(h.name)
            continue
        if h.file not in texts:
            texts[h.file] = tree.read(h.file)
            newlines[h.file] = tree.newline(h.file)
        text = texts[h.file]
        if h.done in text:
            report.append(f"applied  {h.name}")
            continue
        new = h.apply(text)
        if new is None:
            report.append(
                f"MISSING  {h.name}: anchor not found in {h.file}; left unchanged"
            )
            skipped.append(h.name)
            continue
        texts[h.file] = new
        report.append(f"apply    {h.name}")
    if not dry_run:
        for rel, text in texts.items():
            data = Tree.encode(text, newlines[rel])
            if data != tree.path(rel).read_bytes():
                tree.path(rel).write_bytes(data)
    return report, skipped


def run_status(tree: Tree) -> list[str]:
    out = []
    for h in hunks():
        if not tree.exists(h.file):
            out.append(f"MISSING  {h.name} (no {h.file})")
            continue
        text = tree.read(h.file)
        state = (
            "applied"
            if h.done in text
            else ("pending" if h.apply(text) is not None else "MISSING")
        )
        out.append(f"{state:8} {h.name}")
    return out


# --------------------------------------------------------------------------
# gfx950: an authored description (no import or switch) and its tables
# --------------------------------------------------------------------------

GFX950_DESCRIPTION = "isa/gfx950.isa"
GFX950_GEN_ISA = "src/target/gfx9/gen_isa.rs"


def _partial_match_fn(
    name: str,
    ret: str,
    doc: str,
    arms: dict[str, list[str]],
    vcmp: str | None,
    default: str,
) -> str:
    """A match over the opcodes the description has, `default` for the rest."""
    out = [
        f"/// {doc}",
        "#[inline]",
        f"pub fn {name}(op: Opc) -> {ret} {{",
        "    match op {",
    ]
    for value, ops in arms.items():
        out.append(f"        {_wrap_alts([f'Opc::{o}' for o in ops], 8)} => {value},")
    if vcmp is not None:
        out.append(f"        Opc::VCmp(_) => {vcmp},")
    out += [
        "        #[allow(unreachable_patterns)]",
        f"        _ => {default},",
        "    }",
        "}",
    ]
    return "\n".join(out)


def generate_isa_gfx9(desc: Desc, digest: int) -> str:
    """src/target/gfx9/gen_isa.rs: what the gfx9 back end reads of the gfx950
    description. Opcodes without a row have no gfx950 encoding."""
    insts = desc.insts
    fam = desc.family
    fmt_recs = desc.fmt_recs()
    o: list[str] = [
        HEADER_RS,
        f"//! @generated by `{TOOL} generate --isa gfx950` from `{GFX950_DESCRIPTION}`. Do not edit:",
        "//! change the description, then run the tool.",
        "//!",
        "//! Instruction tables of the gfx950 description: operand codes, encoding layouts, and per",
        "//! opcode its gfx950 mnemonic, encoding, opcode, operand signature, unit and latency class.",
        "//! A shared opcode without a gfx950 row has no gfx950 encoding (`info` and `signature` give",
        "//! `None`).",
        "",
        "#![allow(dead_code)]",
        "",
        "use crate::rir::mir::{Fmt, Opc};",
        "",
        "/// FNV-1a 64 of the description (line endings as LF) this file was generated from.",
        f"pub const DESCRIPTION_DIGEST: u64 = 0x{digest:016x};",
        "",
        "// ----- operand codes -----",
        "",
    ]
    f32_pairs: list[tuple[int, int]] = []
    f64_pairs: list[tuple[int, int]] = []
    for rec in desc.section("opnd"):
        if rec.kind == "reg":
            o.append(
                f"pub const REG_{rec.args[0].upper()}: u32 = {_int(rec.get('code'))};"
            )
        elif rec.kind == "src":
            for k, v in rec.kv.items():
                o.append(f"pub const SRC_{k.upper()}: u32 = {_int(v)};")
        elif rec.kind == "inline_int":
            o.append(f"pub const INLINE_INT_MIN: i32 = {int(rec.get('lo'))};")
            o.append(f"pub const INLINE_INT_MAX: i32 = {int(rec.get('hi'))};")
            o.append(f"pub const INLINE_INT_ZERO: u32 = {int(rec.get('zero'))};")
            o.append(f"pub const INLINE_INT_NEG: u32 = {int(rec.get('neg'))};")
        elif rec.kind == "inline_f32":
            f32_pairs = [(_int(b), int(c)) for b, c in (a.split("=") for a in rec.args)]
        elif rec.kind == "inline_f64":
            f64_pairs = [(_int(b), int(c)) for b, c in (a.split("=") for a in rec.args)]
    o += [
        "",
        "/// Operand code of an inline integer constant.",
        "#[inline]",
        "pub fn inline_int(v: u32) -> Option<u32> {",
        "    let s = v as i32;",
        "    if (0..=INLINE_INT_MAX).contains(&s) {",
        "        Some(INLINE_INT_ZERO + s as u32)",
        "    } else if (INLINE_INT_MIN..0).contains(&s) {",
        "        Some(INLINE_INT_NEG + (-s) as u32)",
        "    } else {",
        "        None",
        "    }",
        "}",
        "",
        "/// Operand code of an f32 inline constant.",
        "#[inline]",
        "pub fn inline_f32(v: u32) -> Option<u32> {",
        "    match v {",
        *[f"        0x{b:08x} => Some({c})," for b, c in f32_pairs],
        "        _ => None,",
        "    }",
        "}",
        "",
        "/// Operand code of the f64 inline constant with high dword `hi` (and a zero low dword).",
        "#[inline]",
        "pub fn inline_f64(hi: u32) -> Option<u32> {",
        "    match hi {",
        *[f"        0x{b:08x} => Some({c})," for b, c in f64_pairs],
        "        _ => None,",
        "    }",
        "}",
        "",
        "// ----- encoding layouts -----",
        "",
        _enc_structs(desc),
        "// ----- per-opcode facts -----",
        "",
    ]
    names = [r.args[0] for r in insts]
    o.append(
        "/// The opcodes with a gfx950 row, in description order (the `VCmp` family excluded)."
    )
    o.append(f"pub const ALL: [Opc; {len(names)}] = [")
    for chunk in _chunks([f"Opc::{n}" for n in names], 6):
        o.append("    " + ", ".join(chunk) + ",")
    o += ["];", ""]
    o.append(
        _partial_match_fn(
            "mnemonic",
            "Option<&'static str>",
            "gfx950 mnemonic of an opcode.",
            _group(insts, lambda r: f'Some("{r.args[1]}")'),
            None,
            "None",
        ).replace(
            "    match op {",
            "    match op {\n        Opc::VCmp(code) => vcmp(code).map(|(_, name)| name),",
            1,
        )
    )
    o.append("")
    o.append("/// Execution unit.")
    o.append("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]")
    o.append("pub enum Unit {")
    for rec in desc.section("unit"):
        o.append(f"    /// {rec.args[1] if len(rec.args) > 1 else rec.args[0]}")
        o.append(f"    {camel(rec.args[0].lower())},")
    o += ["}", ""]
    o.append(
        _partial_match_fn(
            "unit",
            "Unit",
            "Execution unit of an opcode (`Pseudo` without a gfx950 row).",
            _group(insts, lambda r: f"Unit::{camel(r.get('unit').lower())}"),
            f"Unit::{camel(fam.get('unit').lower())}",
            "Unit::Pseudo",
        )
    )
    o += [
        "",
        "/// Whether an opcode issues to the vector ALU (VALU or transcendental).",
        "#[inline]",
        "pub fn is_valu(op: Opc) -> bool {",
        "    matches!(unit(op), Unit::Valu | Unit::Trans)",
        "}",
        "",
        "/// Wait counters an instruction increments: vector memory loads and stores (`vmcnt`),",
        "/// LDS and SMEM (`lgkmcnt`), SMEM alone (returns out of order).",
        "pub const CNT_VM: u8 = 1;",
        "pub const CNT_LGKM: u8 = 2;",
        "pub const CNT_SMEM: u8 = 4;",
        "",
        "/// Latency class.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Lat {",
    ]
    lat_recs = desc.section("lat")
    for rec in lat_recs:
        o.append(f"    /// {rec.args[1] if len(rec.args) > 1 else rec.args[0]}")
        o.append(f"    {camel(rec.args[0])},")
    o += [
        "}",
        "",
        "impl Lat {",
        "    /// Cycles from issue until a reader of the result may issue.",
        "    #[inline]",
        "    pub fn sched(self) -> u32 {",
        "        match self {",
    ]
    for rec in lat_recs:
        o.append(f"            Lat::{camel(rec.args[0])} => {int(rec.get('sched'))},")
    o += [
        "        }",
        "    }",
        "",
        "    /// Wait counters (`CNT_*`) the instruction increments.",
        "    #[inline]",
        "    pub fn counters(self) -> u8 {",
        "        match self {",
    ]
    for rec in lat_recs:
        w = [
            f"CNT_{x.upper()}" for x in filter(None, (rec.get("wait") or "").split(","))
        ]
        o.append(f"            Lat::{camel(rec.args[0])} => {' | '.join(w) or '0'},")
    o += ["        }", "    }", "}", ""]
    o.append(
        _partial_match_fn(
            "lat",
            "Lat",
            "Latency class of an opcode (`Pseudo` without a gfx950 row).",
            _group(insts, lambda r: f"Lat::{camel(r.get('lat'))}"),
            f"Lat::{camel(fam.get('lat'))}",
            "Lat::Pseudo",
        )
    )
    o += [
        "",
        "/// Scheduler result latency of an opcode.",
        "#[inline]",
        "pub fn result_latency(op: Opc) -> u32 {",
        "    lat(op).sched()",
        "}",
        "",
        "/// Wait counters an opcode increments.",
        "#[inline]",
        "pub fn counters(op: Opc) -> u8 {",
        "    lat(op).counters()",
        "}",
        "",
    ]
    forms = ["plain"]
    for r in insts:
        f = r.get("form")
        if f and f not in forms:
            forms.append(f)
    form_doc = {
        r.args[0]: (r.args[1] if len(r.args) > 1 else r.args[0])
        for r in desc.section("opnd")
        if r.kind == "form"
    }
    o += [
        "/// How the encoder treats an opcode within its format.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Form {",
    ]
    for f in forms:
        o.append(f"    /// {form_doc.get(f, f)}")
        o.append(f"    {camel(f)},")
    o += ["}", ""]
    bits = {r.args[0]: int(r.get("bit")) for r in desc.section("flags")}
    alias = {r.args[0]: r.get("flags").split("|") for r in desc.section("alias")}

    def flags_value(r: Record) -> int:
        v = 0
        for t in _flags_field(r.get("flags")):
            for f in alias.get(t, [t]):
                v |= 1 << bits[f]
        return v

    def info_row(r: Record, code: int, flags: int) -> str:
        f = fmt_recs[desc.fmt_of(r)]
        vop3 = code + _int(f.get("vop3") or "0")
        table = (
            "false"
            if f.get("table", "yes") == "no" or r.get("table") == "no"
            else "true"
        )
        return (
            f"    Info {{ code: 0x{code:03x}, vop3: 0x{vop3:03x}, flags: 0x{flags:05x}, "
            f"fmt: Fmt::{desc.fmt_of(r)}, form: Form::{camel(r.get('form') or 'plain')}, table: {table} }},"
        )

    fam_vop3 = _int(fmt_recs[desc.fmt_of(fam)].get("vop3") or "0")
    o += [
        "/// What the encoder reads of an opcode, in one load.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub struct Info {",
        "    /// Opcode field of the encoding.",
        "    pub code: u16,",
        "    /// Opcode of the VOP3 encoding (VOP1 and VOP2 promoted, VOPC, native VOP3).",
        "    pub vop3: u16,",
        "    /// The shared flags (`Opc::flags`).",
        "    pub flags: u32,",
        "    pub fmt: Fmt,",
        "    pub form: Form,",
        "    /// Whether the encoder encodes the opcode (pseudos are not).",
        "    pub table: bool,",
        "}",
        "",
        "/// `Info` of each opcode in `ALL` order, then the compare family's fields that do not",
        "/// depend on its code.",
        f"static INFO: [Info; {len(insts) + 1}] = [",
    ]
    o += [info_row(r, _int(r.args[3]), flags_value(r)) for r in insts]
    o += [info_row(fam, 0, 0), "];", ""]
    o += [
        "/// Position of an opcode in `INFO`.",
        "#[inline]",
        "fn index(op: Opc) -> Option<usize> {",
        "    match op {",
    ]
    o += [f"        Opc::{r.args[0]} => Some({i})," for i, r in enumerate(insts)]
    o += [
        f"        Opc::VCmp(_) => Some({len(insts)}),",
        "        #[allow(unreachable_patterns)]",
        "        _ => None,",
        "    }",
        "}",
        "",
    ]
    o += [
        "/// `Info` of an opcode, or `None` when gfx950 has no encoding for it.",
        "#[inline]",
        "pub fn info(op: Opc) -> Option<Info> {",
        "    let mut info = INFO[index(op)?];",
        "    if let Opc::VCmp(code) = op {",
        "        let (gfx9, _) = vcmp(code)?;",
        "        info.code = gfx9;",
        f"        info.vop3 = gfx9{f' + 0x{fam_vop3:x}' if fam_vop3 else ''};",
        "        info.flags = op.flags();",
        "    }",
        "    Some(info)",
        "}",
        "",
    ]
    roles: list[str] = []
    classes: list[str] = []
    imms: list[str] = [""]
    for r in insts + [fam]:
        for key in ("d", "u"):
            for role, cls, _n in parse_sig_list(r.get(key)):
                if role not in roles:
                    roles.append(role)
                if cls not in classes:
                    classes.append(cls)
        if (r.get("imm") or "") not in imms:
            imms.append(r.get("imm") or "")
    modbits = {
        r.args[0]: [int(b) for b in r.get("bits").split(",")]
        for r in desc.section("modset")
    }
    class_doc = {
        r.args[0]: (r.args[1] if len(r.args) > 1 else r.args[0])
        for r in desc.section("opnd")
        if r.kind == "class"
    }
    imm_doc = {
        r.args[0]: (r.args[1] if len(r.args) > 1 else r.args[0])
        for r in desc.section("opnd")
        if r.kind == "imm"
    }
    o += [
        "/// Operand role: the encoding field an operand goes to.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Role {",
    ]
    o += [f"    {camel(v)}," for v in roles]
    o += [
        "}",
        "",
        "/// Operand class.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Class {",
    ]
    for v in classes:
        o.append(f"    /// {class_doc.get(v, v)}")
        o.append(f"    {camel(v)},")
    o += [
        "}",
        "",
        "/// What `MInst::imm` holds.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
        "pub enum Imm {",
    ]
    for v in imms:
        o.append(f"    /// {imm_doc.get(v, 'nothing') if v else 'nothing'}")
        o.append(f"    {camel(v) or 'None'},")
    o += [
        "}",
        "",
        "/// One operand of a signature: role, class and size in dwords.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub struct Operand {",
        "    pub role: Role,",
        "    pub class: Class,",
        "    pub dwords: u8,",
        "}",
        "",
        "/// The operands an instruction takes, in `MInst` order, what its immediate holds and the",
        "/// modifier bits it accepts.",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub struct Sig {",
        "    pub defs: &'static [Operand],",
        "    pub uses: &'static [Operand],",
        "    pub imm: Imm,",
        "    pub mods: u32,",
        "}",
        "",
    ]

    def operands(text: str | None) -> str:
        items = [
            f"Operand {{ role: Role::{camel(role)}, class: Class::{camel(cls)}, dwords: {n} }}"
            for role, cls, n in parse_sig_list(text)
        ]
        return "&[" + ", ".join(items) + "]"

    def mods_value(r: Record) -> int:
        v = 0
        for m in filter(None, (r.get("m") or "").split(",")):
            for b in modbits[m]:
                v |= 1 << b
        return v

    sigs: dict[tuple, str] = {}

    def sig_const(key: tuple) -> str:
        if key not in sigs:
            sigs[key] = f"SIG_{len(sigs)}"
            o.append(f"const {sigs[key]}: Sig = Sig {{")
            o.append(f"    defs: {operands(key[0])},")
            o.append(f"    uses: {operands(key[1])},")
            o.append(f"    imm: Imm::{camel(key[2]) or 'None'},")
            o.append(f"    mods: 0x{key[3]:x},")
            o.append("};")
        return sigs[key]

    sig_of = {
        r.args[0]: sig_const(
            (r.get("d") or "", r.get("u") or "", r.get("imm") or "", mods_value(r))
        )
        for r in insts + [fam]
    }
    vc = [
        (int(c.args[0], 0), _int(c.get("op")), c.args[1], c.get("src"))
        for c in desc.section("vcmp")
    ]
    fam64 = sig_const(
        (
            (fam.get("d") or ""),
            (fam.get("u") or "").replace("vs1", "vs2"),
            fam.get("imm") or "",
            mods_value(fam),
        )
    )
    o.append("")
    arms: dict[str, list[str]] = {}
    for r in insts:
        arms.setdefault(f"Some(&{sig_of[r.args[0]]})", []).append(r.args[0])
    o.append(
        _partial_match_fn(
            "signature",
            "Option<&'static Sig>",
            "Operand signature of an opcode.",
            arms,
            None,
            "None",
        ).replace(
            "    match op {",
            f"    match op {{\n        Opc::VCmp(c) if VCMP64.contains(&c) => Some(&{fam64}),\n"
            f"        Opc::VCmp(_) => Some(&{sig_of['VCmp']}),",
            1,
        )
    )
    o.append("")
    vcmp64 = [c for c, _op, _n, src in vc if src in ("i64", "u64", "f64")]
    o.append(
        f"const VCMP64: [u8; {len(vcmp64)}] = [{', '.join(f'0x{c:02x}' for c in vcmp64)}];"
    )
    o.append("")
    o.append(
        "/// Compares the compiler names: (shared code, gfx950 VOPC opcode, mnemonic)."
    )
    o.append(f"pub const VCMP: [(u8, u16, &str); {len(vc)}] = [")
    for chunk in _chunks([f'(0x{c:02x}, 0x{op:02x}, "{n}")' for c, op, n, _s in vc], 3):
        o.append("    " + ", ".join(chunk) + ",")
    o += [
        "];",
        "",
        "/// gfx950 VOPC opcode and mnemonic of the compare with shared code `code`.",
        "#[inline]",
        "pub fn vcmp(code: u8) -> Option<(u16, &'static str)> {",
        "    match code {",
    ]
    for c, op, n, _s in vc:
        o.append(f'        0x{c:02x} => Some((0x{op:02x}, "{n}")),')
    o += ["        _ => None,", "    }", "}"]
    return "\n".join(o).rstrip() + "\n"


def run_generate_gfx950(tree: Tree) -> dict[str, str]:
    text = tree.read(GFX950_DESCRIPTION)
    secs = parse(text, GFX950_DESCRIPTION)
    validate(
        secs,
        {r.args[0] for r in secs.get("enc", [])},
        {r.get("enc"): r.args[0] for r in secs.get("fmt", [])},
    )
    desc = Desc(secs, text)
    return {GFX950_GEN_ISA: generate_isa_gfx9(desc, fnv1a64(text.encode("utf-8")))}


def cmd_generate_gfx950(tree: Tree, check: bool) -> int:
    """`generate --isa gfx950` (or with `check`, `check --isa gfx950`): the
    description is authored, so there is nothing to import."""
    nl = tree.convention()
    bad = 0
    for rel, text in run_generate_gfx950(tree).items():
        data = Tree.encode(text, nl)
        if _bytes_or_none(tree, rel) == data:
            print(f"{rel}: up to date")
            continue
        if check:
            print(f"{rel} is stale: run `{TOOL} generate --isa gfx950`")
            bad += 1
            continue
        tree.path(rel).parent.mkdir(parents=True, exist_ok=True)
        tree.path(rel).write_bytes(data)
        print(f"{rel}: written")
    if check and not bad:
        print("isa tables (gfx950): description and generated files are up to date")
    return 1 if bad else 0


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _diff(rel: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}", n=2
        )
    )


def _bytes_or_none(tree: Tree, rel: str) -> bytes | None:
    return tree.path(rel).read_bytes() if tree.exists(rel) else None


def cmd_import(tree: Tree, args) -> int:
    res = run_import(tree)
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    data = Tree.encode(res.text, tree.convention())
    if _bytes_or_none(tree, DESCRIPTION) == data:
        print(f"{DESCRIPTION}: up to date")
        return 0
    tree.path(DESCRIPTION).parent.mkdir(parents=True, exist_ok=True)
    tree.path(DESCRIPTION).write_bytes(data)
    print(f"{DESCRIPTION}: written")
    return 0


def cmd_generate(tree: Tree, args) -> int:
    if getattr(args, "isa", "gfx11") == "gfx950":
        return cmd_generate_gfx950(tree, check=False)
    nl = tree.convention()
    for rel, text in run_generate(tree).items():
        data = Tree.encode(text, nl)
        if _bytes_or_none(tree, rel) == data:
            print(f"{rel}: up to date")
            continue
        tree.path(rel).write_bytes(data)
        print(f"{rel}: written")
    return 0


def cmd_check(tree: Tree, args) -> int:
    if getattr(args, "isa", "gfx11") == "gfx950":
        return cmd_generate_gfx950(tree, check=True)
    bad = 0
    nl = tree.convention()
    res = run_import(tree)
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    cur = _bytes_or_none(tree, DESCRIPTION)
    if cur != Tree.encode(res.text, nl):
        print(f"{DESCRIPTION} is stale: run `{TOOL} import`")
        if cur is not None:
            print(
                _diff(DESCRIPTION, cur.decode("utf-8").replace("\r\n", "\n"), res.text)[
                    :6000
                ]
            )
        bad += 1
    for rel, text in run_generate(tree).items():
        cur = _bytes_or_none(tree, rel)
        if cur != Tree.encode(text, nl):
            print(f"{rel} is stale: run `{TOOL} generate`")
            if cur is not None:
                print(
                    _diff(rel, cur.decode("utf-8").replace("\r\n", "\n"), text)[:6000]
                )
            bad += 1
    if not bad:
        print("isa tables: description and generated files are up to date")
    return 1 if bad else 0


def cmd_switch(tree: Tree, args) -> int:
    report, skipped = run_switch(tree, args.dry_run, args.skip or [])
    for line in report:
        print(line)
    if skipped and not args.allow_skip:
        print(
            f"{len(skipped)} edit(s) not applied; rerun with --allow-skip to accept",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_status(tree: Tree, args) -> int:
    for line in run_status(tree):
        print(line)
    return 0


def cmd_fp(tree: Tree, args) -> int:
    """Prints the hand encoder's arm and helper fingerprints (to refresh
    KNOWN_ASM after isa.rs is taught a changed arm)."""
    for label, fp in asm_fingerprints(scan_asm(tree.read(ASM))).items():
        print(f"    {label!r}: {fp!r},")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--crate",
        type=Path,
        default=DEFAULT_CRATE,
        help="rocke-codeobj crate directory",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "import", help="refresh the description from the Rust sources and the overlay"
    )
    gen = sub.add_parser("generate", help="write the generated Rust modules")
    chk = sub.add_parser(
        "check", help="fail if import or generate would change a checked-in file"
    )
    for p in (gen, chk):
        p.add_argument(
            "--isa",
            choices=("gfx11", "gfx950"),
            default="gfx11",
            help="the description: isa/gfx11.isa (default) or the authored isa/gfx950.isa",
        )
    sw = sub.add_parser(
        "switch",
        help="point mir.rs, asm.rs, vopd.rs, post.rs and isel.rs at the tables",
    )
    sw.add_argument("--dry-run", action="store_true")
    sw.add_argument(
        "--allow-skip",
        action="store_true",
        help="exit 0 even if an edit's anchor is missing",
    )
    sw.add_argument(
        "--skip",
        action="append",
        help="skip edits whose name starts with this (e.g. post:)",
    )
    sub.add_parser("status", help="show which edits of `switch` are applied")
    sub.add_parser("fp", help="print the hand encoder's arm fingerprints")
    args = ap.parse_args(argv)
    tree = Tree(args.crate.resolve())
    commands = {
        "import": cmd_import,
        "generate": cmd_generate,
        "check": cmd_check,
        "switch": cmd_switch,
        "status": cmd_status,
        "fp": cmd_fp,
    }
    try:
        return commands[args.cmd](tree, args)
    except DescError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
