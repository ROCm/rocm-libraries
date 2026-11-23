#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fail if a change breaks an experimental feature without bumping its revision.

Compares the experimental API of a base revision (normally the PR's target branch) with
the working tree. Each experimental feature lives in its own header,
hipblaslt-experimental-<feature>.h, which defines HIPBLASLT_EXPERIMENTAL_<FEATURE> to the
feature's revision, and has contract tests in
clients/tests/src/experimental/contract_<feature>_gtest.cpp. Headers next to them named
contract_<feature>*.hpp (or .h), such as shared test helpers, are part of the contract too. A
change to a feature is breaking if it

* removes a declaration from the feature's header (functions, C++ methods, types,
  enumerators, macros),
* changes a macro, or
* changes or deletes one of the feature's contract tests or contract headers. Contract tests
  define the feature's promised behavior, so editing one means the behavior changed.

Additions are never breaking; that includes a new C++ method or overload. A breaking change
must bump that feature's revision in the same change. Some changes are never allowed, with or
without a bump:

* changing a function's signature, or a struct, union or typedef, under its existing name
  (give it a new name, such as ..._v2, and remove the old one),
* changing a C++ function or method in place: its return type, default arguments or
  qualifiers (a different parameter list is a different overload, so it is a removal plus an
  addition),
* changing a C++ class's bases, data members or virtual functions, which are its layout (give
  the class a new name),
* changing an enumerator's value, or giving a new enumerator a value at or below
  HIPBLASLT_EXPERIMENTAL_ENUM_HIGH_WATER (experimental enum values are never reused; this
  includes C++ scoped enums),
* declaring a template, which the checker cannot compare,
* changing or removing anything declared in hipblaslt-experimental.h itself, such as
  hipblasLtExperimentalGetFeatureRevision.

Removing a whole feature header (graduation or removal) needs no bump: the feature's macro
disappears, which is the signal to consumers. Policy: docs/reference/experimental-api.rst.

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
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEADER_DIR = "library/include/hipblaslt"
MAIN_HEADER = f"{HEADER_DIR}/hipblaslt-experimental.h"
FEATURE_HEADER_GLOB = "hipblaslt-experimental-*.h"
CONTRACT_DIR = "clients/tests/src/experimental"
CONTRACT_GLOBS = ("contract_*.cpp", "contract_*.hpp", "contract_*.h")
REVISION_PREFIX = "HIPBLASLT_EXPERIMENTAL_"
ENUM_BIT = "HIPBLASLT_EXPERIMENTAL_ENUM_BIT"
HIGH_WATER_MACRO = "HIPBLASLT_EXPERIMENTAL_ENUM_HIGH_WATER"
CORE = "<core>"  # declarations in hipblaslt-experimental.h, which never change


def feature_of_header(path: str) -> str:
    """hipblaslt-experimental-gemm-a2a.h -> GEMM_A2A"""
    stem = path.rsplit("/", 1)[-1][len("hipblaslt-experimental-") : -len(".h")]
    return stem.replace("-", "_").upper()


def feature_of_contract(path: str, features) -> str:
    """contract_gemm_a2a_gtest.cpp, contract_gemm_a2a_errors_gtest.cpp or
    contract_gemm_a2a_helpers.hpp -> GEMM_A2A

    The longest known feature name that the file name starts with; otherwise the whole stem.
    """
    stem = path.rsplit("/", 1)[-1][len("contract_") :].rsplit(".", 1)[0]
    stem = re.sub(r"_gtest$", "", stem).upper()
    matches = [f for f in features if stem == f or stem.startswith(f + "_")]
    return max(matches, key=len) if matches else stem


def contract_files(tree) -> List[str]:
    """The contract tests and contract headers in a tree."""
    return sorted({p for pattern in CONTRACT_GLOBS for p in tree.glob(CONTRACT_DIR, pattern)})


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


def enum_value(text: str) -> Optional[int]:
    """Value of an experimental enumerator, ignoring the marker bit. None if not a literal."""
    text = re.sub(rf"^\(*{ENUM_BIT}\|", "", text).strip("()")
    try:
        return int(text, 0)
    except ValueError:
        return None


@dataclass
class Surface:
    revisions: Dict[str, int] = field(default_factory=dict)  # feature -> revision
    decls: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # key -> (feature, text)
    enums: Dict[str, int] = field(default_factory=dict)  # enumerator key -> value
    contracts: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # path -> (feature, hash)
    high_water: Optional[int] = None
    errors: List[str] = field(default_factory=list)


def _add_enumerator(key: str, value: str, feature: str, where: str, surface: Surface) -> None:
    n = enum_value(value)
    if n is None:
        surface.errors.append(
            f"{where}: {key} = {value}: experimental enumerators need an integer literal value"
            f" (optionally {ENUM_BIT} | n)"
        )
        return
    surface.decls[key] = (feature, value)
    surface.enums[key] = n


def _add_statement(stmt: str, feature: str, where: str, surface: Surface) -> None:
    stmt = normalize(stmt)
    if not stmt:
        return
    enum = re.match(r"(?:typedef )?enum(?: (\w+))?(?::[\w ]+)?\{(.*)\}(\w*)$", stmt)
    if enum:
        name = enum.group(3) or enum.group(1)
        if not name:
            surface.errors.append(f"{where}: anonymous enum; give it a name")
            return
        surface.decls[f"enum {name}"] = (feature, "enum")
        for item in filter(None, (e.strip() for e in enum.group(2).split(","))):
            if "=" not in item:
                surface.errors.append(
                    f"{where}: enumerator {name}::{item} needs an explicit value"
                    " (implicit values shift silently when enumerators are inserted)"
                )
                continue
            ident, value = item.split("=", 1)
            _add_enumerator(f"enumerator {name}::{ident}", value, feature, where, surface)
        return
    fnptr = re.search(r"\(\*(\w+)\)", stmt)
    if stmt.startswith("typedef"):
        name = fnptr.group(1) if fnptr else re.findall(r"\w+", stmt)[-1]
        surface.decls[f"type {name}"] = (feature, stmt)
    elif re.match(r"(?:typedef )?(?:struct|union) \w+\{", stmt):
        name = re.match(r"(?:typedef )?(?:struct|union) (\w+)", stmt).group(1)
        surface.decls[f"type {name}"] = (feature, stmt)
    elif "(" in stmt:
        surface.decls[f"function {re.findall(r'(\w+)\(', stmt)[0]}"] = (feature, stmt)
    else:
        surface.decls[f"decl {stmt}"] = (feature, stmt)


# C++ declarations: namespaces, classes, scoped enums and overloaded functions. Functions and
# methods are keyed by their scope, name and parameter types (the parts that decide which
# symbol a caller binds to), so an overload is its own declaration and renaming a parameter is
# no change. The rest of a declaration (return type, default arguments, qualifiers such as
# noexcept or = delete) is compared under that key. A class's layout (bases, data members and
# virtual functions) is one declaration; its other methods are separate.

_ACCESS = re.compile(r"(public|protected|private):(?!:)")
_TYPE_WORDS = {
    "auto", "bool", "char", "char8_t", "char16_t", "char32_t", "const", "double", "float",
    "int", "long", "short", "signed", "unsigned", "void", "volatile", "wchar_t",
}
_QUALIFIER_WORDS = {"class", "const", "enum", "struct", "typename", "volatile"}


def _match_brace(code: str, i: int) -> int:
    """Index of the '}' that closes the '{' at code[i] (len(code) if it is not closed)."""
    depth = 0
    for j in range(i, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return len(code)


def _split_top(text: str, sep: str) -> List[str]:
    """Split at `sep` outside (), <>, [] and {}."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _parameter(param: str) -> Tuple[str, str]:
    """The type and the default argument of a parameter: 'const T& x = {}' -> ('const T&', '{}')."""
    pieces = _split_top(param, "=")
    decl, default = pieces[0], "=".join(pieces[1:])
    m = re.match(r"(.*[\w>&*\]])\s*\b([A-Za-z_]\w*)$", decl)
    # 'unsigned int' and 'const T' have no parameter name.
    if m and m.group(2) not in _TYPE_WORDS and not set(m.group(1).split()) <= _QUALIFIER_WORDS:
        decl = m.group(1).strip()
    return decl, default


def _function(stmt: str) -> Optional[Tuple[str, str]]:
    """(key, text) of a function declaration, or None if `stmt` is not one.

    key is 'name(parameter types)' plus cv and ref qualifiers; text is the whole declaration
    without parameter names.
    """
    m = re.search(r"\boperator\s*(?:\(\)|\[\]|[^\w\s(]+)\s*\(|(~?\w+)\s*\(", stmt)
    if not m or "(*" in stmt[: m.end() + 1]:
        return None
    name_end = m.end() - 1
    name = normalize(stmt[m.start() : name_end])
    depth, close = 0, -1
    for j in range(name_end, len(stmt)):
        if stmt[j] == "(":
            depth += 1
        elif stmt[j] == ")":
            depth -= 1
            if depth == 0:
                close = j
                break
    if close < 0:
        return None
    prefix = normalize(stmt[: m.start()])
    suffix = normalize(stmt[close + 1 :])
    params = [_parameter(p) for p in _split_top(stmt[name_end + 1 : close], ",")]
    if [p[0] for p in params] == ["void"]:
        params = []
    types = ",".join(t for t, _ in params)
    qualifiers = re.findall(r"\bconst\b|\bvolatile\b|&&|&", suffix.split("=")[0].split("noexcept")[0])
    key = f"{name}({types}){''.join(qualifiers)}"
    text = normalize(
        f"{prefix} {name}({','.join(t + ('=' + d if d else '') for t, d in params)}) {suffix}"
    )
    return key, text


def _strip_access(stmt: str, access: str) -> Tuple[str, str]:
    while True:
        m = _ACCESS.match(stmt)
        if not m:
            return stmt, access
        access, stmt = m.group(1), stmt[m.end() :]


def _scoped(scope: str, name: str) -> str:
    return f"{scope}::{name}" if scope else name


class _Class:
    def __init__(self, kind: str, qualified: str):
        self.kind = kind
        self.qualified = qualified
        self.layout: List[str] = []


def _cpp_statement(stmt: str, scope: str, cls: Optional[_Class], access: str,
                   feature: str, where: str, surface: Surface) -> None:
    if stmt.startswith("template"):
        surface.errors.append(f"{where}: {scope or 'global scope'}: templates are not supported"
                              " in experimental headers (the checker cannot compare them)")
        return
    if stmt.startswith(("friend ", "static_assert")):
        return
    alias = re.match(r"using (\w+)=", stmt)
    if alias or stmt.startswith("typedef "):
        name = alias.group(1) if alias else re.findall(r"\w+", stmt)[-1]
        if cls is None or access != "private":
            surface.decls[f"type {_scoped(scope, name)}"] = (feature, stmt)
        return
    function = _function(stmt) if not re.match(r"(?:class|struct|union|enum)\b", stmt) else None
    if function:
        key, text = function
        if cls is None:
            surface.decls[f"function {_scoped(scope, key)}"] = (feature, text)
            return
        if re.search(r"\bvirtual\b", text):
            cls.layout.append(text)
        if access != "private":
            surface.decls[f"method {_scoped(scope, key)}"] = (feature, text)
        return
    if cls is not None:
        cls.layout.append(stmt)  # a data member or a nested type
    else:
        surface.decls[f"decl {_scoped(scope, stmt)}"] = (feature, stmt)


def _cpp_enum(head: str, body: str, scope: str, feature: str, where: str, surface: Surface) -> None:
    m = re.match(r"enum(?: (?:class|struct))?(?: (\w+))?(?::(.+))?$", head)
    if not m or not m.group(1):
        surface.errors.append(f"{where}: anonymous enum in {scope or 'global scope'}; give it a name")
        return
    name = _scoped(scope, m.group(1))
    surface.decls[f"enum {name}"] = (feature, head[: m.start(1)].strip() + (":" + m.group(2) if m.group(2) else ""))
    for item in _split_top(normalize(body), ","):
        if "=" not in item:
            surface.errors.append(
                f"{where}: enumerator {name}::{item} needs an explicit value"
                " (implicit values shift silently when enumerators are inserted)"
            )
            continue
        ident, value = item.split("=", 1)
        _add_enumerator(f"enumerator {name}::{ident}", value, feature, where, surface)


def _cpp_class(head: str, body: str, scope: str, outer: Optional[_Class],
               feature: str, where: str, surface: Surface) -> None:
    m = re.match(r"(class|struct|union) (?:\w+ )*?(\w+)( final)?(?::(.+))?$", head)
    if not m:
        surface.errors.append(f"{where}: unnamed {head.split()[0]} in {scope or 'global scope'}")
        return
    kind, name = m.group(1), m.group(2)
    cls = _Class(kind, _scoped(scope, name))
    if m.group(4):
        cls.layout.append(":" + m.group(4))
    parse_cpp(body, cls.qualified, feature, where, surface, cls)
    surface.decls[f"class {cls.qualified}"] = (feature, f"{kind}{m.group(3) or ''}{{{';'.join(cls.layout)}}}")
    if outer is not None:
        outer.layout.append(f"{kind} {name}")


def parse_cpp(code: str, scope: str, feature: str, where: str, surface: Surface,
              cls: Optional[_Class] = None) -> None:
    """Declarations in C++ code: a namespace body, a class body or top-level C++ declarations."""
    access = "private" if cls is not None and cls.kind == "class" else "public"
    i = start = 0
    while i < len(code):
        ch = code[i]
        if ch == "{":
            head, access = _strip_access(normalize(code[start:i]), access)
            end = _match_brace(code, i)
            namespace = re.fullmatch(r"namespace(?: ([\w:]+))?", head)
            if namespace:
                if cls is not None or not namespace.group(1):
                    surface.errors.append(f"{where}: {'anonymous namespace' if cls is None else 'namespace in a class'}"
                                          " in an experimental header")
                else:
                    parse_cpp(code[i + 1 : end], _scoped(scope, namespace.group(1)), feature, where, surface)
                start = i = end + 1
                continue
            if re.match(r"(?:class|struct|union|enum)\b", head):
                semi = code.find(";", end)
                semi = len(code) if semi < 0 else semi
                if head.startswith("enum"):
                    _cpp_enum(head, code[i + 1 : end], scope, feature, where, surface)
                else:
                    _cpp_class(head, code[i + 1 : end], scope, cls, feature, where, surface)
                trailing = normalize(code[end + 1 : semi])
                if trailing:
                    _cpp_statement(trailing, scope, cls, access, feature, where, surface)
                start = i = semi + 1
                continue
            if "(" in head and head.count("(") == head.count(")"):
                # An inline function body: the declaration is the head, the body is not API.
                _cpp_statement(head, scope, cls, access, feature, where, surface)
                after = re.match(r"\s*;", code[end + 1 :])
                start = i = end + 1 + (after.end() if after else 0)
                continue
            i = end + 1  # a braced initializer or default argument inside a declaration
            continue
        if ch == ";":
            stmt, access = _strip_access(normalize(code[start:i]), access)
            if stmt:
                _cpp_statement(stmt, scope, cls, access, feature, where, surface)
            start = i + 1
        i += 1
    stmt, access = _strip_access(normalize(code[start:]), access)
    if stmt:
        _cpp_statement(stmt, scope, cls, access, feature, where, surface)


def parse_header(text: str, feature: str, where: str, surface: Surface) -> None:
    revision_macro = f"{REVISION_PREFIX}{feature}"
    code_lines = []
    for line in strip_comments(text).split("\n"):
        directive = line.strip()
        if directive.startswith("#"):
            m = re.match(r"#\s*define\s+(\w+)(\([^)]*\))?\s+(.+)", directive)
            if m:
                name, value = m.group(1), normalize(m.group(3))
                if feature != CORE and name == revision_macro:
                    surface.revisions[feature] = int(value)
                elif feature == CORE and name == HIGH_WATER_MACRO:
                    surface.high_water = int(value)
                elif ENUM_BIT in value and name != ENUM_BIT:
                    # An experimental value of a stable enum, e.g.
                    # #define X ((hipblasLtEpilogue_t)(HIPBLASLT_EXPERIMENTAL_ENUM_BIT | 3))
                    cast = re.match(r"\(\((\w+)\)(.*)\)$", value)
                    enum_type, raw = (cast.group(1), cast.group(2)) if cast else ("macro", value)
                    _add_enumerator(f"enumerator {enum_type}::{name}", raw, feature, where, surface)
                else:
                    surface.decls[f"macro {name}"] = (feature, normalize((m.group(2) or "") + " " + value))
            continue
        code_lines.append(line)
    code = re.sub(r'extern\s*"C"\s*\{', " ", "\n".join(code_lines))

    depth, start, i = 0, 0, 0
    while i < len(code):
        ch = code[i]
        if ch == "{" and depth == 0:
            head = normalize(code[start:i])
            if re.fullmatch(r"namespace(?: [\w:]+)?", head) or re.match(r"(?:class|template)\b", head):
                end = _match_brace(code, i)
                if head.startswith("namespace"):
                    stop = end + 1
                else:
                    semi = code.find(";", end)
                    stop = len(code) if semi < 0 else semi + 1
                parse_cpp(code[start:stop], "", feature, where, surface)
                start = i = stop
                continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:  # closing brace of extern "C"
                code = code[:i] + " " + code[i + 1 :]
            else:
                depth -= 1
        elif ch == ";" and depth == 0:
            _add_statement(code[start:i], feature, where, surface)
            start = i + 1
        i += 1
    _add_statement(code[start:], feature, where, surface)

    if feature != CORE and feature not in surface.revisions:
        surface.errors.append(f"{where}: no #define {revision_macro} <revision>")


def contract_hash(text: str) -> str:
    return hashlib.sha256(normalize(strip_comments(text)).encode("utf-8")).hexdigest()


def load_surface(tree) -> Optional[Surface]:
    main = tree.read(MAIN_HEADER)
    if main is None:
        return None
    surface = Surface()
    parse_header(main, CORE, MAIN_HEADER, surface)
    for path in tree.glob(HEADER_DIR, FEATURE_HEADER_GLOB):
        parse_header(tree.read(path), feature_of_header(path), path, surface)
    for path in contract_files(tree):
        feature = feature_of_contract(path, surface.revisions)
        if feature not in surface.revisions:
            surface.errors.append(f"{path}: no experimental feature {feature} for this contract test")
        surface.contracts[path] = (feature, contract_hash(tree.read(path)))
    if surface.high_water is None:
        surface.errors.append(f"{MAIN_HEADER}: no #define {HIGH_WATER_MACRO}")
    elif surface.enums and max(surface.enums.values()) > surface.high_water:
        surface.errors.append(
            f"{HIGH_WATER_MACRO} is {surface.high_water}, but an experimental enumerator has value"
            f" {max(surface.enums.values())}; raise it"
        )
    return surface


# ---------------------------------------------------------------------------
# Comparing.
# ---------------------------------------------------------------------------


@dataclass
class Report:
    breaking: Dict[str, List[str]] = field(default_factory=dict)  # feature -> changes
    notes: List[str] = field(default_factory=list)  # additions, graduations, revisions
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

    def breaking(feature: str, what: str) -> None:
        report.breaking.setdefault(feature, []).append(what)

    gone = set(base.revisions) - set(head.revisions)
    for feature in sorted(gone):
        report.notes.append(f"feature {feature} removed or graduated (revision macro gone)")

    for key, (feature, text) in base.decls.items():
        if feature in gone:
            continue
        if key not in head.decls:
            if feature == CORE:
                report.errors.append(f"removed {key} from {MAIN_HEADER}, which never changes")
            else:
                breaking(feature, f"removed  {key}")
            continue
        new = head.decls[key][1]
        if new == text:
            continue
        if feature == CORE:
            report.errors.append(f"changed {key} in {MAIN_HEADER}, which never changes")
        elif key.startswith(("function ", "method ")) and "(" in key:
            report.errors.append(
                f"changed {key} in place: '{text}' -> '{new}'.\n"
                "  The return type, default arguments and qualifiers of a C++ function or method"
                " never change under the same parameter list: add an overload or a new name"
                " and remove the old one (a removal needs a revision bump)."
            )
        elif key.startswith("function "):
            report.errors.append(
                f"changed {key} in place: '{text}' -> '{new}'.\n"
                "  A breaking change to a function gives it a new name (for example ..._v2) and"
                " removes the old one, so a stale binary fails to load instead of calling it"
                " with the wrong arguments."
            )
        elif key.startswith(("class ", "enum ")):
            report.errors.append(
                f"changed {key} in place: '{text}' -> '{new}'.\n"
                "  A change to a class's bases, data members or virtual functions, or to an"
                " enum's underlying type, gives it a new name (for example ..._v2) and removes"
                " the old one, so a stale binary never uses the old layout as the new one."
                " Adding a non-virtual method is an addition."
            )
        elif key.startswith("type "):
            report.errors.append(
                f"changed {key} in place: '{text}' -> '{new}'.\n"
                "  A breaking change to a struct, union or typedef gives it a new name (for"
                " example ..._v2) and removes the old one, along with the functions that use it,"
                " so a stale binary never reads the old layout as the new one."
            )
        elif key.startswith("enumerator "):
            report.errors.append(
                f"changed value of {key}: {text} -> {new}. Experimental enum values are never"
                " renumbered; retire the enumerator and add a new one above"
                f" {HIGH_WATER_MACRO}."
            )
        else:
            breaking(feature, f"changed  {key}: '{text}' -> '{new}'")
    for key, (feature, _) in head.decls.items():
        if key not in base.decls:
            report.notes.append(f"added    {key} ({feature})")

    if base.high_water is not None and head.high_water is not None:
        if head.high_water < base.high_water:
            report.errors.append(
                f"{HIGH_WATER_MACRO} went down: {base.high_water} -> {head.high_water}"
            )
        for key, value in head.enums.items():
            if key not in base.enums and value <= base.high_water:
                report.errors.append(
                    f"new {key} has value {value}, at or below {HIGH_WATER_MACRO}"
                    f" ({base.high_water}); experimental enum values are never reused. Use"
                    f" {base.high_water + 1} or above and raise {HIGH_WATER_MACRO}."
                )

    for path, (feature, digest) in base.contracts.items():
        if feature in gone:
            continue
        if path not in head.contracts:
            breaking(feature, f"deleted contract test {path}")
        elif head.contracts[path][1] != digest:
            breaking(feature, f"changed contract test {path}")
    report.notes += [f"added contract test {p}" for p in head.contracts if p not in base.contracts]

    for feature, rev in sorted(head.revisions.items()):
        macro = f"{REVISION_PREFIX}{feature}"
        if feature not in base.revisions:
            report.notes.append(f"new feature {feature} at revision {rev}")
            if rev < 1:
                report.errors.append(f"{macro} must start at 1, not {rev}")
            continue
        old = base.revisions[feature]
        changes = report.breaking.get(feature)
        if rev < old:
            report.errors.append(f"{macro} went backwards: {old} -> {rev}")
        elif changes and rev == old:
            report.errors.append(
                f"breaking change to experimental feature {feature} without bumping {macro}"
                f" (still {old}).\n"
                f"  Bump it to {old + 1} in its header and add a breaking entry under\n"
                '  "### Experimental" in CHANGELOG.md. If the change is not meant to be breaking,\n'
                "  restore the removed/changed declarations or contract tests instead."
            )
        elif rev > old and not changes:
            report.warnings.append(
                f"{macro} bumped ({old} -> {rev}) but no breaking change was detected."
                " If behavior changed, a contract test should change with it."
            )
        elif rev > old:
            report.notes.append(f"feature {feature}: revision {old} -> {rev}")
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

    revisions = ", ".join(f"{f} {r}" for f, r in sorted(head.revisions.items())) or "none"
    print(f"experimental API: base {base_tree.name}; feature revisions in working tree: {revisions}")
    for feature, changes in sorted(report.breaking.items()):
        for line in changes:
            print(f"  [{feature}] {line}")
    for line in report.notes:
        print(f"  {line}")
    for w in report.warnings:
        print(f"warning: {w}")
    for e in report.errors:
        print(f"error: {e}")
    print("OK" if report.ok else "FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
