# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read the enums in hipBLASLt's public headers, and the list of values the bindings leave out.

One parser serves two checks:
- the `hipblaslt-python-enum-sync` pre-commit hook, which reads the raw headers and
  csrc/enums.cpp as text, with nothing built (scripts/check_enum_sync.py);
- the coverage tests, which read the headers after the real preprocessor has run.

This module uses only the standard library and never imports hipblaslt._core, so the hook can
load it by file path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "ALLOWED_MISSING",
    "BoundValue",
    "EnumDef",
    "EnumValue",
    "bound_values",
    "parse_enums",
]


@dataclass(frozen=True)
class EnumValue:
    name: str
    value: int
    # The `#if` condition the value sits under, when it's an opt-in feature guard
    # (HIPBLASLT_HAS_...), else None.
    guard: str | None = None


@dataclass
class EnumDef:
    name: str
    values: list[EnumValue] = field(default_factory=list)

    def by_name(self) -> dict[str, EnumValue]:
        return {v.name: v for v in self.values}


# Values the bindings deliberately don't bind, keyed by the C enum's name. Only two kinds belong
# here: sentinels that grow whenever a value is added, and values the header marks "not
# supported yet". hipDataType and hipblasComputeType_t also list the HIP types hipBLASLt doesn't
# accept, because those enums belong to HIP and hipBLAS, not to hipBLASLt.
_SENTINEL = "a sentinel; it grows whenever a value is added"
_NOT_YET = 'the header marks it "Not supported yet"'
_COMPLEX = "a complex type; hipBLASLt doesn't accept complex matrices"
_UNSUPPORTED_TYPE = "hipBLASLt doesn't accept this type"
_UNSUPPORTED_COMPUTE = (
    "hipBLASLt's compute-type handling (library/src/amd_detail) doesn't accept it"
)
ALLOWED_MISSING: dict[str, dict[str, str]] = {
    "hipblasLtMatmulDescAttributes_t": {"HIPBLASLT_MATMUL_DESC_MAX": _SENTINEL},
    "hipblasLtMatmulPreferenceAttributes_t": {"HIPBLASLT_MATMUL_PREF_MAX": _SENTINEL},
    "hipblasLtInt4Encoding_t": {"HIPBLASLT_INT4_ENCODING_END_EXT": _SENTINEL},
    "hipblasLtMatmulMatrixScale_t": {
        "HIPBLASLT_MATMUL_MATRIX_SCALE_END": _SENTINEL,
        **{
            f"HIPBLASLT_MATMUL_MATRIX_SCALE_{v}": _NOT_YET
            for v in (
                "VEC16_UE4M3",
                "VEC128_32F",
                "BLK128x128_32F",
                "VEC16_UE8M0_EXT",
                "VEC32_UE4M3_EXT",
                "VEC16_UE5M3_EXT",
                "VEC32_UE5M3_EXT",
            )
        },
    },
    "hipblasComputeType_t": {
        f"HIPBLAS_COMPUTE_{v}": _UNSUPPORTED_COMPUTE
        for v in (
            "16F_PEDANTIC",
            "32F_PEDANTIC",
            "64F_PEDANTIC",
            "32I_PEDANTIC",
            "32F_FAST_8F_FNUZ",
            "32F_FAST_8BF_FNUZ",
            "32F_FAST_8F8BF_FNUZ",
            "32F_FAST_8BF8F_FNUZ",
        )
    },
    "hipDataType": {
        **{
            f"HIP_C_{t}": _COMPLEX
            for t in (
                "32F",
                "64F",
                "16F",
                "8I",
                "8U",
                "32I",
                "32U",
                "16BF",
                "4I",
                "4U",
                "16I",
                "16U",
                "64I",
                "64U",
            )
        },
        **{
            f"HIP_R_{t}": _UNSUPPORTED_TYPE
            for t in ("8U", "32U", "4U", "16I", "16U", "64I", "64U")
        },
        "HIP_R_8F_UE8M0": "hipBLASLt takes UE8M0 scales through ScaleMode, not as a matrix type",
    },
}

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_ENUM = re.compile(
    r"(?:typedef\s+enum(?:\s+(?P<tag>\w+))?\s*\{(?P<body>[^{}]*)\}\s*(?P<name>\w+)\s*;)"
    r"|(?:enum\s+class\s+(?P<cname>\w+)(?:\s*:\s*[\w:]+)?\s*\{(?P<cbody>[^{}]*)\}\s*;)",
    re.S,
)
_ENTRY = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?:=\s*(?P<expr>.+?))?\s*$", re.S)
_GUARD = re.compile(r"\bHIPBLASLT_HAS_\w+")


def _strip_comments(text: str) -> str:
    """Remove comments but keep every newline, so offsets still map to lines."""
    text = _BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n") + " ", text)
    return _LINE_COMMENT.sub("", text)


def _line_guards(text: str) -> list[str | None]:
    """For each line, the innermost HIPBLASLT_HAS_... condition it sits under, or None."""
    stack: list[str | None] = []
    out: list[str | None] = []
    for line in text.split("\n"):
        s = line.strip()
        directive = s[1:].strip() if s.startswith("#") else ""
        word = directive.split(None, 1)[0] if directive else ""
        current = next((g for g in reversed(stack) if g), None)
        if word in ("if", "ifdef", "ifndef"):
            m = _GUARD.search(directive)
            stack.append(directive if m else None)
        elif word in ("elif", "else") and stack:
            top = stack.pop()
            stack.append(
                f"!({top})"
                if top and word == "else"
                else (directive if _GUARD.search(directive) else None)
            )
        elif word == "endif" and stack:
            stack.pop()
        out.append(current)
    return out


def _eval(expr: str, known: dict[str, int]) -> int:
    """Evaluate an enumerator's value: integer literals, earlier members, | & << >> + - ~ ( )."""
    expr = re.sub(
        r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b", lambda m: str(int(m.group(1), 0)), expr
    )
    expr = re.sub(r"\(\s*(?:int|unsigned|uint\d+_t|int\d+_t)\s*\)", "", expr)

    def name(m: re.Match[str]) -> str:
        ident = m.group(0).split("::")[-1]
        if ident not in known:
            raise ValueError(f"unknown name {ident!r} in enumerator value {expr!r}")
        return str(known[ident])

    expr = re.sub(r"[A-Za-z_][\w:]*", name, expr)
    if not re.fullmatch(r"[\d\s|&^<>+\-~()*]+", expr):
        raise ValueError(f"can't evaluate enumerator value {expr!r}")
    return int(
        eval(expr, {"__builtins__": {}})
    )  # noqa: S307 - digits and operators only


def parse_enums(
    text: str, select: Callable[[str], bool] | None = None
) -> dict[str, EnumDef]:
    """Every `typedef enum` and `enum class` in a C or C++ header, by type name.

    Handles comments that contain braces, explicit, implicit and hex values, trailing commas,
    CRLF line endings, and values inside `#if HIPBLASLT_HAS_...` blocks (each value records its
    guard). Other preprocessor conditions are ignored, so run this on preprocessed text when the
    build configuration matters.

    `select` limits the result to the enums it accepts by name. Preprocessed text includes the
    C++ standard library's enums, which can use values this parser doesn't evaluate; an enum
    that `select` accepts must parse, or this raises.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    clean = _strip_comments(text)
    guards = _line_guards(clean)
    line_starts = [0]
    for i, ch in enumerate(clean):
        if ch == "\n":
            line_starts.append(i + 1)

    def guard_at(offset: int) -> str | None:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return guards[lo]

    enums: dict[str, EnumDef] = {}
    for m in _ENUM.finditer(clean):
        name = m.group("name") or m.group("cname")
        if select is not None and not select(name):
            continue
        body_key = "body" if m.group("name") else "cbody"
        body = m.group(body_key)
        start = m.start(body_key)
        enum = EnumDef(name)
        known: dict[str, int] = {}
        nxt = 0
        pos = 0
        for raw in body.split(","):
            entry_offset = start + pos + (len(raw) - len(raw.lstrip()))
            pos += len(raw) + 1
            # Drop preprocessor lines inside the body; their effect is in the guards.
            entry = "\n".join(
                l for l in raw.split("\n") if not l.strip().startswith("#")
            ).strip()
            if not entry:
                continue
            em = _ENTRY.match(entry)
            if not em:
                raise ValueError(f"can't read enumerator {entry!r} in {name}")
            value = _eval(em.group("expr"), known) if em.group("expr") else nxt
            ident = em.group("name")
            # The guard is taken where the enumerator's name is, not where the raw chunk starts.
            ident_offset = (
                entry_offset + raw.lstrip().find(ident)
                if ident in raw
                else entry_offset
            )
            known[ident] = value
            enum.values.append(EnumValue(ident, value, guard_at(ident_offset)))
            nxt = value + 1
        enums[name] = enum
    return enums


_REG = re.compile(
    r'\breg\(\s*\w+\s*,\s*"(?P<enum>\w+)"\s*,\s*"(?P<py>\w+)"\s*,\s*(?P<c>[\w:]+)\s*\)'
)


@dataclass(frozen=True)
class BoundValue:
    py_name: str
    guard: str | None = None


def bound_values(enums_cpp: str) -> dict[str, dict[str, BoundValue]]:
    """What csrc/enums.cpp binds: {C enum name: {C value name: BoundValue}}.

    It reads `reg(e, "<C enum>", "<Python name>", <C value>)` calls, the one registration form
    enums.cpp uses, and records the `#if HIPBLASLT_HAS_...` guard each sits under.
    """
    text = _strip_comments(enums_cpp.replace("\r\n", "\n"))
    guards = _line_guards(text)
    out: dict[str, dict[str, BoundValue]] = {}
    # A call can span lines (clang-format wraps long ones); its guard is its first line's.
    for m in _REG.finditer(text):
        lineno = text.count("\n", 0, m.start())
        c_name = m.group("c").split("::")[-1]
        out.setdefault(m.group("enum"), {})[c_name] = BoundValue(
            m.group("py"), guards[lineno]
        )
    return out
