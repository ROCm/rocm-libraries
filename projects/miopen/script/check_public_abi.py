#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Public-ABI symbol gate for the MIOpen public/private library split.

What this actually verifies
---------------------------
This is a *symbol-surface* gate, not a byte-for-byte comparison. It reads the
ELF dynamic symbol table and dynamic section of a built library directly (no
nm/readelf, no shell, no locale-dependent sorting) and asserts:

``check`` -- run on a default MIOPEN_ENABLE_HIPDNN_WRAPPER=OFF build:

  1. SONAME is libMIOpen.so.1 (unchanged).
  2. The exported public C API symbol set (``miopen`` followed by an uppercase
     letter) exactly matches the committed baseline.
  3. No ``*_impl`` symbols are exported -- the private rename must not leak
     into a flag-off build.
  4. libMIOpen_private is NOT in DT_NEEDED (flag-off is self-contained).
  5. Optionally, the full DT_NEEDED list matches a committed baseline, so a
     flag-off build cannot silently acquire a new runtime dependency.

``check-wrapper`` -- run on a MIOPEN_ENABLE_HIPDNN_WRAPPER=ON build. Under the
flag the public surface is split: the thin wrapper libMIOpen.so re-exports only
the miopen.h public-header contract, while a handful of baseline exports that
were never in miopen.h (experimental miopen_internal.h APIs and the "Hidden"
MIGraphX shims) stay on libMIOpen_private.so under their original names.

  1. SONAME is libMIOpen.so.1 (unchanged).
  2. The wrapper's exported public C API set equals ``baseline - excluded``.
  3. No ``*_impl`` symbols are exported from the wrapper.
  4. libMIOpen_private IS in DT_NEEDED.
  5. Every excluded symbol is genuinely absent from the installed public header
     (``--public-header``), so the exclusion file cannot be used to silence a
     red gate by quietly dropping a real miopen.h entry point.
  6. Every excluded symbol is still exported, un-suffixed, from the private
     library (``--private-lib``), so the carve-out cannot delete a symbol from
     the whole installed surface.

``check-headers`` -- a source-only cross-check of the five hand-maintained
artifacts of the split. It needs no build, no GPU and no flag-on configuration,
so unlike the two gates above it can run in a lint lane on every PR:

  1. include/miopen/miopen.h        -- the public contract (MIOPEN_EXPORT decls)
  2. src/private/miopen_impl.h      -- the matching _impl declarations
  3. src/private/miopen_private_rename.h -- the compile-time rename
  4. src/private/wrapper.cpp        -- the forwarding stubs
  5. the hipDNN provider's MiopenApiPrivateRename.hpp -- the provider's mirror
     of (3), force-included when it links the private library

Every public entry point must appear in all five, the wrapper stub's signature
must match miopen.h, the _impl declaration's signature must match it too (modulo
the suffix), and each stub must forward to its own _impl symbol and nothing
else. No artifact may carry an entry the public header does not.

Each stub must also open with MIOPEN_WRAPPER_DISPATCH naming itself. The macro's
own runtime assert cannot cover this: it is compiled out under NDEBUG, it never
fires for a stub nothing calls, and a stub missing the macro entirely has no
assert to fire at all. Such a stub silently loses the ability to ever route to
hipDNN, which is invisible until the entry point joins the forwarding set and
keeps running MIOpen anyway.

The provider mirror lives in a sibling project that a MIOpen-only checkout does
not ship, so it is the one artifact that can be skipped. It is skipped only once
git confirms the commit under test does not track it -- being absent from a
sparse working tree is not enough, or CI, which checks out only the subtrees a
PR touches, would report a green gate on unchecked drift.

``check-headers`` additionally cross-checks the three
miopenConvolution*GetWorkSpaceSizeRange entry points. These are exported with
MIOPEN_EXPORT but never declared in miopen.h, so every consumer forward-declares
them by hand and none of the five artifacts above covers them. Their definitions
in src/convolution_api.cpp are the reference; the declarations in the gtest that
exercises them and in the hipDNN provider's MiopenApi.hpp must match it. The
provider header is located and skipped on the same terms as the mirror.

This is the only check in this script that can see *signature* drift. It matters
because the _impl entry points have C linkage: the wrapper's stub definitions are
compiler-checked against miopen.h, and the private library's _impl definitions
are the miopen.h declarations with a macro applied, but nothing compiles
miopen_impl.h against the definitions it describes. A divergence there links
cleanly and corrupts arguments at runtime. The symbol-set gates above cannot see
it -- the exported names are unchanged.

Deliberately NOT verified: struct/enum layout, behaviour, exported symbols
outside the ``miopen[A-Z]`` C API convention (including the mangled
``_ZN6miopen...`` internals that the driver, gtests and CK plugins link
against), symbol addresses/sizes, and dynamic-symbol ordering. ``check`` and
``check-wrapper`` additionally do not verify signatures; that is what
``check-headers`` is for. Signature comparison is textual after normalisation,
not semantic: it will not resolve a typedef, so two spellings of the same
underlying type read as a mismatch. That is the intended bias -- these files
are meant to be copies of one another.

``compare-pair`` diffs two ``dump`` outputs and reports a content hash for
information only; a content hash is build-path dependent and is never gated.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import subprocess
import sys
from pathlib import Path

EXPECTED_SONAME = "libMIOpen.so.1"
PRIVATE_LIB_PREFIX = "libMIOpen_private"

# Public MIOpen C API naming convention: "miopen" followed by an uppercase
# letter. This deliberately excludes internal exported shims such as
# miopen_sqlite3_memvfs_init.
PUBLIC_API_RE = re.compile(r"^miopen[A-Z]")
IMPL_RE = re.compile(r"^miopen[A-Za-z0-9_]*_impl$")


class AbiError(Exception):
    """Fatal, non-assertion problem (bad file, missing input, unreadable ELF)."""


# --------------------------------------------------------------------------
# Minimal ELF reader
#
# Parsing the ELF directly rather than shelling out to nm/readelf keeps this
# script runnable anywhere Python is (including a Windows host inspecting a
# Linux build artifact), and removes the locale-collation and pipeline-exit-code
# hazards that a shell implementation has.
# --------------------------------------------------------------------------

SHT_DYNSYM = 11
SHT_DYNAMIC = 6
SHN_UNDEF = 0
STB_GLOBAL, STB_WEAK, STB_GNU_UNIQUE = 1, 2, 10
STT_FUNC, STT_GNU_IFUNC = 2, 10
DT_NULL, DT_NEEDED, DT_SONAME = 0, 1, 14


class Elf:
    """Just enough ELF to read .dynsym and .dynamic from a 32/64-bit object."""

    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        if self.data[:4] != b"\x7fELF":
            raise AbiError(f"not an ELF file: {path}")
        self.is64 = self.data[4] == 2
        self.end = "<" if self.data[5] == 1 else ">"
        self._read_sections()

    def _unpack(self, fmt: str, off: int):
        fmt = self.end + fmt
        return struct.unpack_from(fmt, self.data, off)

    def _read_sections(self) -> None:
        if self.is64:
            (e_shoff,) = self._unpack("Q", 0x28)
            e_shentsize, e_shnum, e_shstrndx = self._unpack("HHH", 0x3A)
        else:
            (e_shoff,) = self._unpack("I", 0x20)
            e_shentsize, e_shnum, e_shstrndx = self._unpack("HHH", 0x2E)
        if e_shoff == 0 or e_shnum == 0:
            raise AbiError(f"ELF has no section headers: {self.path}")

        raw = []
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            if self.is64:
                name, stype, _flags, _addr, sh_off, size, link, _info, _al, entsize = (
                    self._unpack("IIQQQQIIQQ", off)
                )
            else:
                name, stype, _flags, _addr, sh_off, size, link, _info, _al, entsize = (
                    self._unpack("IIIIIIIIII", off)
                )
            raw.append(
                {
                    "name_off": name,
                    "type": stype,
                    "offset": sh_off,
                    "size": size,
                    "link": link,
                    "entsize": entsize,
                }
            )

        shstr = raw[e_shstrndx]
        for sec in raw:
            sec["name"] = self._cstr(shstr["offset"] + sec["name_off"])
        self.sections = raw

    def _cstr(self, off: int) -> str:
        end = self.data.index(b"\x00", off)
        return self.data[off:end].decode("utf-8", "replace")

    def _section(self, stype: int):
        for sec in self.sections:
            if sec["type"] == stype:
                return sec
        return None

    def defined_dynamic_functions(self) -> set[str]:
        """Names of defined, externally visible function symbols in .dynsym.

        Includes weak and IFUNC definitions (an alias or IFUNC export is still a
        real entry point), and strips any ``@VERSION`` / ``@@VERSION`` suffix so
        that introducing a version script does not make every symbol mismatch.
        """
        dynsym = self._section(SHT_DYNSYM)
        if dynsym is None:
            raise AbiError(f"no .dynsym section in {self.path} (stripped or static?)")
        strtab = self.sections[dynsym["link"]]
        entsize = dynsym["entsize"] or (24 if self.is64 else 16)

        names: set[str] = set()
        for off in range(dynsym["offset"], dynsym["offset"] + dynsym["size"], entsize):
            if self.is64:
                st_name, st_info, _other, st_shndx, _val, _size = self._unpack(
                    "IBBHQQ", off
                )
            else:
                st_name, _val, _size, st_info, _other, st_shndx = self._unpack(
                    "IIIBBH", off
                )
            if st_shndx == SHN_UNDEF or st_name == 0:
                continue
            bind, sym_type = st_info >> 4, st_info & 0xF
            if bind not in (STB_GLOBAL, STB_WEAK, STB_GNU_UNIQUE):
                continue
            if sym_type not in (STT_FUNC, STT_GNU_IFUNC):
                continue
            names.add(self._cstr(strtab["offset"] + st_name).split("@", 1)[0])
        return names

    def _dynamic_entries(self):
        dyn = self._section(SHT_DYNAMIC)
        if dyn is None:
            raise AbiError(f"no .dynamic section in {self.path}")
        strtab = self.sections[dyn["link"]]
        entsize = dyn["entsize"] or (16 if self.is64 else 8)
        fmt = "QQ" if self.is64 else "II"
        for off in range(dyn["offset"], dyn["offset"] + dyn["size"], entsize):
            tag, val = self._unpack(fmt, off)
            if tag == DT_NULL:
                break
            yield tag, val, strtab["offset"]

    def soname(self) -> str:
        for tag, val, stroff in self._dynamic_entries():
            if tag == DT_SONAME:
                return self._cstr(stroff + val)
        return ""

    def needed(self) -> list[str]:
        return sorted(
            self._cstr(stroff + val)
            for tag, val, stroff in self._dynamic_entries()
            if tag == DT_NEEDED
        )


# --------------------------------------------------------------------------
# Symbol-list helpers
# --------------------------------------------------------------------------


def open_elf(path_str: str, what: str) -> Elf:
    path = Path(path_str)
    if not path.is_file():
        raise AbiError(f"{what} not found: {path}")
    return Elf(path)


def public_api_symbols(elf: Elf) -> set[str]:
    syms = {n for n in elf.defined_dynamic_functions() if PUBLIC_API_RE.match(n)}
    if not syms:
        raise AbiError(
            f"no exported miopen* public API symbols found in {elf.path} "
            "(stripped, wrong file, or a link failure?)"
        )
    return syms


def impl_symbols(elf: Elf) -> set[str]:
    return {n for n in elf.defined_dynamic_functions() if IMPL_RE.match(n)}


def read_symbol_list(path_str: str, what: str) -> set[str]:
    """Read a committed symbol list, ignoring comments and blank lines."""
    path = Path(path_str)
    if not path.is_file():
        raise AbiError(f"{what} not found: {path}")
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def format_set_diff(expected: set[str], got: set[str]) -> str:
    lines = [f"  - missing from build: {s}" for s in sorted(expected - got)]
    lines += [f"  + unexpected in build: {s}" for s in sorted(got - expected)]
    return "\n".join(lines)


BASELINE_REMEDY = """
If this change to the public C API is intentional, regenerate the baseline with:
  projects/miopen/script/check_public_abi.py dump-symbols <lib> \\
      -o projects/miopen/test/public_abi/public_symbols.baseline
and update src/private/miopen_private_rename.h + src/private/miopen_impl.h to match.
A baseline change is an ABI change and requires API review -- do not regenerate
the baseline just to turn this gate green.""".rstrip()


HEADERS_REMEDY = """
Every public entry point must be spelled identically in all five places:
  include/miopen/miopen.h              MIOPEN_EXPORT <ret> miopenFoo(<params>);
  src/private/miopen_impl.h            extern "C" <ret> miopenFoo_impl(<params>);
  src/private/miopen_private_rename.h  #define miopenFoo miopenFoo_impl
  src/private/wrapper.cpp              extern "C" <ret> miopenFoo(<params>)
                                       { return miopenFoo_impl(<args>); }
  <repo>/dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp
                                       #define miopenFoo miopenFoo_impl
Update the private files and the provider's mirror to match the public header.
Do not edit miopen.h to match them -- that changes the public C API.""".rstrip()

DISPATCH_REMEDY = """
Every stub in src/private/wrapper.cpp must open with the dispatch macro, passing
its own function token:
  extern "C" miopenStatus_t miopenFoo(<params>)
  {
      MIOPEN_WRAPPER_DISPATCH(miopenFoo);
      return miopenFoo_impl(<args>);
  }
Pass the token, never a string and never a neighbouring stub's name. A stub
without the macro can never route to hipDNN, and nothing else reports that: the
macro's assert needs the macro to be there, and is compiled out under NDEBUG in
any case.""".rstrip()

RANGE_REMEDY = """
Each miopenConvolution*GetWorkSpaceSizeRange entry point is spelled by hand in
three places, because it is exported without being declared in miopen.h:
  src/convolution_api.cpp                      the definition (the reference)
  test/gtest/conv_workspace_size_range.cpp     local extern "C" declaration
  <repo>/dnn-providers/miopen-provider/MiopenApi.hpp
                                               local extern "C" declaration
Update the two declarations to match the definition. These have C linkage, so a
divergence links cleanly and corrupts arguments at run time.""".rstrip()

PROVIDER_REMEDY = """
--require-provider was passed, so the provider's copies must be readable rather
than skipped. Make dnn-providers/miopen-provider part of the tree being checked:
in a sparse checkout, add that directory to the checkout's path list. Drop
--require-provider only for a tree that genuinely has no provider, such as a
MIOpen-only checkout or a source tarball -- not to get past this message.""".rstrip()

# The hipDNN provider's copies, relative to the repository root (the MIOpen
# source root's grandparent in the monorepo layout).
PROVIDER_RENAME_RELPATH = "dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp"
PROVIDER_API_RELPATH = "dnn-providers/miopen-provider/MiopenApi.hpp"

# Exported with MIOPEN_EXPORT from src/convolution_api.cpp, absent from
# miopen.h, and so declared by hand wherever they are called. Listed explicitly
# rather than discovered from the definitions so that dropping a declaration
# from a consumer fails the check instead of shrinking the comparison set.
RANGE_ENTRY_POINTS = frozenset(
    {
        "miopenConvolutionForwardGetWorkSpaceSizeRange",
        "miopenConvolutionBackwardDataGetWorkSpaceSizeRange",
        "miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange",
    }
)

# Stubs that must NOT carry MIOPEN_WRAPPER_DISPATCH, and why. The macro returns
# forward_to_hipdnn's miopenStatus_t, so a stub returning anything else cannot
# host it. Exemptions are checked in both directions: an exempt stub that grows
# the macro fails here rather than failing to compile somewhere less obvious.
DISPATCH_EXEMPT = {
    "miopenGetErrorString": "returns const char*, not miopenStatus_t",
}


# --------------------------------------------------------------------------
# Minimal C declaration parser.
#
# Parsing these files textually rather than invoking a compiler keeps this
# check free of any build, toolchain or GPU dependency, so it can run in a lint
# lane on every PR. The cost is that the comparison is textual: the parser
# reduces a prototype to a return type plus a list of parameter types with the
# parameter names dropped, and compares those strings. It does not resolve
# typedefs, so two spellings of the same underlying type read as a mismatch.
# That bias is deliberate -- the private files are meant to be transcriptions of
# the public header, so an intentional divergence in spelling is itself worth a
# human look.
#
# Two constructs the parser cannot handle are function-pointer parameters and
# array parameters. Neither occurs in miopen.h today (function-pointer types are
# introduced via typedef and passed by that name). A declaration that fails to
# parse raises rather than being skipped, so adding one produces a loud failure
# here instead of a silent coverage hole.
# --------------------------------------------------------------------------

BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT_RE = re.compile(r"//[^\n]*")
EXPORT_ANCHOR_RE = re.compile(r"\bMIOPEN_EXPORT\b")
EXTERN_C_ANCHOR_RE = re.compile(r'\bextern\s+"C"(?!\s*\{)')
RENAME_RE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+(miopen[A-Za-z0-9_]*)[ \t]+(\S+)[ \t]*$", re.M
)
WRAPPER_DEF_RE = re.compile(
    r'\bextern\s+"C"\s+(?P<decl>[^;{}]*?)\s*\{(?P<body>[^{}]*)\}', re.S
)
DECL_RE = re.compile(
    r"(?P<ret>.*?)(?P<name>miopen[A-Za-z0-9_]*)\s*\((?P<params>[^()]*)\)"
)
IMPL_CALL_RE = re.compile(r"\b(miopen[A-Za-z0-9_]*_impl)\s*\(")
DISPATCH_RE = re.compile(
    r"\bMIOPEN_WRAPPER_DISPATCH\s*\(\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
)
# A declarator for one of the range entry points, in either of the two forms it
# is written in: a declaration ending in ';' and a definition followed by its
# body's '{'.
RANGE_DECL_RE = re.compile(
    r"\b(?P<name>miopen[A-Za-z0-9_]*GetWorkSpaceSizeRange)\s*"
    r"\((?P<params>[^()]*)\)\s*(?=[;{])"
)
# Characters that can precede a declarator's return type at file scope. A call
# site is preceded by one of these too, but with nothing between it and the
# name, which is how the two are told apart. '=' is in the set so that an
# assignment from a call (`auto r = miopenFoo(...)`) leaves nothing before the
# name either.
DECLARATOR_STOPS = ";{}(),="
# Tokens that can end a statement but never a return type. Without these, a
# call in statement position (`return miopenFoo(...)`) would read as a
# declaration of miopenFoo returning `return`.
NON_TYPE_TOKENS = frozenset({"return", "co_return", "auto", "case", "else", "do"})

# An unnamed parameter can end in one of these, so a trailing identifier that is
# one of them is part of the type rather than a parameter name.
TYPE_TAIL_KEYWORDS = frozenset(
    {
        "void",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "bool",
        "signed",
        "unsigned",
        "const",
        "struct",
        "enum",
    }
)


def strip_comments(text: str) -> str:
    return LINE_COMMENT_RE.sub("", BLOCK_COMMENT_RE.sub(" ", text))


def squash(text: str) -> str:
    """Collapse whitespace and normalise pointer spelling to ``T* x``."""
    text = " ".join(text.split())
    text = re.sub(r"\s*\*", "*", text)
    text = re.sub(r"\*(?=[A-Za-z_])", "* ", text)
    return text.strip()


def split_params(params: str) -> list[str]:
    """Split a parameter list on top-level commas."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in params:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur)
    if out or tail.strip():
        out.append(tail)
    return out


def drop_top_level_const(type_text: str) -> str:
    """Remove a top-level const, which does not participate in type identity.

    ``const float`` and ``float`` are the same parameter type, as are
    ``void* const`` and ``void*``: a const on the parameter object itself is
    ignored when the compiler matches declarations. A const *below* the top
    level is part of the type and must be kept, so ``const void*`` -- a pointer
    to const -- is left alone. The distinction is positional: a const after the
    last ``*``, or a leading const on a type with no ``*`` at all, is top level.
    """
    text = re.sub(r"\bconst\b\s*$", "", type_text).strip()
    if "*" not in text:
        text = re.sub(r"^const\b\s*", "", text)
    return squash(text)


def normalise_param(param: str) -> str:
    """Reduce one parameter to its type, dropping name and default argument.

    A few entry points carry a C++ default argument. It is a property of the
    declaration, not of the type, so a wrapper stub that omits it -- as a
    definition must -- is not drift.
    """
    text = squash(param.split("=", 1)[0])
    if not text or text == "void":
        return "void"
    match = re.fullmatch(r"(.*?)([A-Za-z_]\w*)", text)
    if match:
        prefix, tail = match.group(1).strip(), match.group(2)
        if prefix and tail not in TYPE_TAIL_KEYWORDS and not tail.endswith("_t"):
            text = prefix
    return drop_top_level_const(text)


def parse_prototype(decl: str, what: str) -> tuple[str, tuple[str, ...]]:
    """Turn one declarator into ``(name, (return_type, *param_types))``."""
    text = squash(decl.replace('extern "C"', " ").replace("MIOPEN_EXPORT", " "))
    match = DECL_RE.fullmatch(text)
    if match is None:
        raise AbiError(f"cannot parse declaration in {what}: {text!r}")
    params = [normalise_param(p) for p in split_params(match.group("params"))] or [
        "void"
    ]
    return match.group("name"), (squash(match.group("ret")), *params)


def render_signature(sig: tuple[str, ...]) -> str:
    return f"{sig[0]}({', '.join(sig[1:])})"


def parse_declarations(
    source: str, anchor: re.Pattern[str], what: str
) -> dict[str, tuple[str, ...]]:
    """Collect every miopen* prototype introduced by ``anchor``, keyed by name."""
    text = strip_comments(source)
    out: dict[str, tuple[str, ...]] = {}
    for match in anchor.finditer(text):
        end = text.find(";", match.end())
        if end < 0:
            raise AbiError(f"unterminated declaration in {what}")
        decl = text[match.end() : end]
        if "{" in decl:
            raise AbiError(f"unexpected block after declaration anchor in {what}")
        name, sig = parse_prototype(decl, what)
        if name in out:
            raise AbiError(f"duplicate declaration of {name} in {what}")
        out[name] = sig
    return out


def parse_range_prototypes(source: str, what: str) -> dict[str, tuple[str, ...]]:
    """Collect the range entry points declared or defined in one source file.

    Unlike the five rename artifacts these are not introduced by an anchor that
    marks a declaration: they appear inside an ``extern "C" { ... }`` block in
    the consumers and as ``MIOPEN_EXPORT extern "C"`` definitions in the
    library, and the consumers also *call* them. Matches are therefore found by
    name, and a call is rejected by the absence of a return type between the
    name and the punctuation that precedes it.
    """
    text = strip_comments(source)
    out: dict[str, tuple[str, ...]] = {}
    for match in RANGE_DECL_RE.finditer(text):
        start = max(text.rfind(ch, 0, match.start()) for ch in DECLARATOR_STOPS)
        if start < 0:
            continue  # no declarator boundary precedes the match; not a declaration
        prefix = text[start + 1 : match.start()]
        prefix = prefix.replace('extern "C"', " ").replace("MIOPEN_EXPORT", " ")
        tokens = prefix.split()
        if not tokens or NON_TYPE_TOKENS & set(tokens):
            continue  # a call, not a declarator
        name, sig = parse_prototype(
            f"{prefix} {match.group('name')}({match.group('params')})", what
        )
        if name in out and out[name] != sig:
            raise AbiError(f"conflicting declarations of {name} in {what}")
        out[name] = sig
    return out


def parse_wrapper(
    source: str,
) -> tuple[dict[str, tuple[str, ...]], dict[str, set[str]], dict[str, list[str]]]:
    """Collect each stub's prototype, the _impl symbols it calls, and its
    MIOPEN_WRAPPER_DISPATCH arguments.

    Dispatches stay an ordered list rather than a set so that a stub carrying the
    macro twice is visible as such instead of collapsing into a correct-looking
    single entry.
    """
    text = strip_comments(source)
    protos: dict[str, tuple[str, ...]] = {}
    forwards: dict[str, set[str]] = {}
    dispatches: dict[str, list[str]] = {}
    for match in WRAPPER_DEF_RE.finditer(text):
        name, sig = parse_prototype(match.group("decl"), "wrapper")
        if name in protos:
            raise AbiError(f"duplicate definition of {name} in wrapper")
        protos[name] = sig
        forwards[name] = set(IMPL_CALL_RE.findall(match.group("body")))
        dispatches[name] = DISPATCH_RE.findall(match.group("body"))
    return protos, forwards, dispatches


def parse_renames(source: str, what: str = "rename header") -> dict[str, str]:
    """Collect a rename header's #defines, folding backslash continuations."""
    text = strip_comments(source.replace("\\\n", " "))
    out: dict[str, str] = {}
    for name, target in RENAME_RE.findall(text):
        if name in out:
            raise AbiError(f"duplicate #define of {name} in {what}")
        out[name] = target
    return out


def describe_git_failure(repo_root: Path, rel: str, stderr: bytes) -> tuple[bool, str]:
    """Say why git could not produce HEAD:<rel>, and whether that is fatal.

    These causes are not interchangeable. A path the commit does not carry is a
    legitimately absent artifact and skipping it costs nothing. A path the commit
    *does* carry whose blob git cannot materialize -- a blobless clone whose
    promisor fetch failed -- is a broken checkout in which this gate would
    otherwise stop checking a file that is under test, and go green doing it.
    That one is fatal: the drift this gate exists to catch links cleanly and
    corrupts arguments at run time, so a broken checkout must be repaired and
    the job rerun, not quietly narrowed. Returns (fatal, reason).

    Only the case git positively confirms is escalated. If ls-tree cannot answer
    -- it errors, or git is gone -- the cause is ambiguous and stays a skip,
    because a false failure on every tarball build would be its own outage.
    """
    text = stderr.decode("utf-8", "replace").lower()
    if "not a git repository" in text:
        return False, "not on disk, and the source root is not a git checkout"
    try:
        listed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-tree", "--name-only", "HEAD", "--", rel],
            capture_output=True,
            check=False,
        )
    except OSError:
        listed = None
    if listed is not None and listed.returncode == 0 and listed.stdout.strip():
        detail = stderr.decode("utf-8", "replace").strip() or "(git printed nothing)"
        return True, (
            "tracked at HEAD but its content could not be read, so this checkout"
            " cannot be checked for drift; expect this when a blobless clone's"
            " promisor fetch fails, and retry the job or refetch the blob."
            f" git cat-file said: {detail}"
        )
    return False, "not tracked at HEAD"


def read_tracked_source(
    path: Path, repo_root: Path, reasons: dict[str, str] | None = None
) -> str | None:
    """Read a file that a sparse checkout may not have materialized.

    CI checks out only the subtrees a PR touches, so a file this check compares
    can be missing from the working tree while still being part of the commit
    under test -- the provider's copies for a MIOpen-only PR, MIOpen's own
    sources for a provider-only PR. Skipping on "not on disk" alone would let
    this gate go green on precisely the drift it exists to catch, so a missing
    file is only treated as genuinely absent once git confirms the commit does
    not track it. Returns None in that case, and the caller skips the check,
    recording why in ``reasons`` so the skip line can name the actual cause.
    Raises AbiError instead when git confirms the commit *does* track the file
    and still cannot produce it, which is a broken checkout rather than an
    absent artifact.
    """
    if path.is_file():
        return path.read_text(encoding="utf-8")

    def unavailable(key: str, reason: str) -> None:
        if reasons is not None:
            reasons[key] = reason

    try:
        rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        unavailable(path.as_posix(), "not on disk, and outside the repository root")
        return None
    try:
        blob = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-p", f"HEAD:{rel}"],
            capture_output=True,
            check=False,
        )
    except OSError:  # no git available: a source tarball, not a checkout
        unavailable(rel, "not on disk, and git is unavailable to read it from HEAD")
        return None
    if blob.returncode != 0:
        fatal, reason = describe_git_failure(repo_root, rel, blob.stderr)
        if fatal:
            raise AbiError(f"{rel}: {reason}")
        unavailable(rel, reason)
        return None
    print(f"NOTE: {rel} is not checked out; reading it from HEAD")
    return blob.stdout.decode("utf-8")


def read_source(path: Path, what: str) -> str:
    if not path.is_file():
        raise AbiError(f"{what} source not found: {path}")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Individual assertions. Each returns True on pass and prints its own verdict,
# so `check` and `check-wrapper` are assembled from the same primitives rather
# than duplicating them.
# --------------------------------------------------------------------------


def check_soname(elf: Elf) -> bool:
    got = elf.soname()
    if got == EXPECTED_SONAME:
        print(f"PASS: SONAME is {got}")
        return True
    print(f"FAIL: SONAME is '{got}', expected '{EXPECTED_SONAME}'")
    return False


def check_symbols(elf: Elf, expected: set[str], label: str, remedy: str = "") -> bool:
    got = public_api_symbols(elf)
    if got == expected:
        print(f"PASS: {label} ({len(expected)} symbols)")
        return True
    print(f"FAIL: {label} -- exported set differs:")
    print(format_set_diff(expected, got))
    if remedy:
        print(remedy)
    return False


def check_no_impl(elf: Elf, where: str) -> bool:
    leaked = impl_symbols(elf)
    if not leaked:
        print(f"PASS: no *_impl symbols exported from {where}")
        return True
    print(f"FAIL: {where} exported *_impl symbols (private rename leaked):")
    for sym in sorted(leaked):
        print(f"  {sym}")
    return False


def check_private_dep(elf: Elf, expect_present: bool) -> bool:
    present = any(n.startswith(PRIVATE_LIB_PREFIX) for n in elf.needed())
    if present == expect_present:
        print(
            f"PASS: DT_NEEDED on {PRIVATE_LIB_PREFIX} is "
            f"{'present' if present else 'absent'} as expected"
        )
        return True
    if expect_present:
        print(f"FAIL: flag-on wrapper has no DT_NEEDED on {PRIVATE_LIB_PREFIX}")
    else:
        print(
            f"FAIL: flag-off libMIOpen.so has DT_NEEDED on {PRIVATE_LIB_PREFIX} "
            "(not self-contained)"
        )
    return False


def check_needed_baseline(elf: Elf, baseline_path: str) -> bool:
    expected = read_symbol_list(baseline_path, "DT_NEEDED baseline")
    got = set(elf.needed())
    if got == expected:
        print(f"PASS: DT_NEEDED list matches baseline ({len(expected)} entries)")
        return True
    print("FAIL: DT_NEEDED list differs from baseline:")
    print(format_set_diff(expected, got))
    return False


def check_excluded_not_public(excluded: set[str], header_path: str) -> bool:
    path = Path(header_path)
    if not path.is_file():
        raise AbiError(f"public header not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    offenders = sorted(s for s in excluded if re.search(rf"\b{re.escape(s)}\b", text))
    if not offenders:
        print(
            f"PASS: no excluded symbol appears in {path.name} "
            f"({len(excluded)} checked)"
        )
        return True
    print(
        f"FAIL: excluded-symbols file lists entry points that ARE declared in "
        f"{path.name}. Excluding a real public API silently removes it from "
        "libMIOpen.so:"
    )
    for sym in offenders:
        print(f"  {sym}")
    return False


def check_excluded_on_private(excluded: set[str], private_lib: str) -> bool:
    elf = open_elf(private_lib, "private library")
    exported = elf.defined_dynamic_functions()
    missing = sorted(excluded - exported)
    if not missing:
        print(
            f"PASS: all {len(excluded)} excluded symbols are still exported "
            f"un-suffixed from {Path(private_lib).name}"
        )
        return True
    print(
        f"FAIL: excluded symbols are absent from {Path(private_lib).name} -- they "
        "have vanished from the entire installed surface (renamed by mistake?):"
    )
    for sym in missing:
        print(f"  {sym}")
    return False


def check_entry_point_set(public: set[str], other: set[str], what: str) -> bool:
    if public == other:
        print(f"PASS: {what} covers exactly the {len(public)} miopen.h entry points")
        return True
    print(f"FAIL: {what} does not match the miopen.h entry point set")
    for sym in sorted(public - other):
        print(f"  - declared in miopen.h, absent from {what}: {sym}")
    for sym in sorted(other - public):
        print(f"  + present in {what}, not declared in miopen.h: {sym}")
    return False


def check_prototypes(
    public: dict[str, tuple[str, ...]],
    other: dict[str, tuple[str, ...]],
    what: str,
    reference: str = "miopen.h",
) -> bool:
    drifted = [n for n in sorted(public.keys() & other.keys()) if public[n] != other[n]]
    if not drifted:
        print(f"PASS: {what} prototypes match {reference}")
        return True
    print(f"FAIL: {what} prototypes have drifted from {reference}")
    for name in drifted:
        print(f"  {name}")
        print(f"      {reference}: {render_signature(public[name])}")
        print(f"      {what}: {render_signature(other[name])}")
    return False


def check_range_entry_point_set(found: dict[str, tuple[str, ...]], what: str) -> bool:
    if set(found) == RANGE_ENTRY_POINTS:
        print(f"PASS: {what} covers all {len(RANGE_ENTRY_POINTS)} range entry points")
        return True
    print(f"FAIL: {what} does not cover exactly the range entry points")
    for sym in sorted(RANGE_ENTRY_POINTS - set(found)):
        print(f"  - missing from {what}: {sym}")
    for sym in sorted(set(found) - RANGE_ENTRY_POINTS):
        print(f"  + present in {what}, not a known range entry point: {sym}")
    return False


def check_rename_targets(renames: dict[str, str]) -> bool:
    bad = {n: t for n, t in sorted(renames.items()) if t != f"{n}_impl"}
    if not bad:
        print(f"PASS: all {len(renames)} renames map miopenFoo to miopenFoo_impl")
        return True
    print("FAIL: renames point at the wrong symbol -- calls would be misrouted:")
    for name, target in bad.items():
        print(f"  {name} -> {target} (expected {name}_impl)")
    return False


def check_provider_rename_mirror(lib: dict[str, str], provider: dict[str, str]) -> bool:
    """The provider's force-included rename header must mirror the library's.

    The hipDNN miopen-provider links libMIOpen_private.so and force-includes its
    own copy of the rename set so its public-name calls bind the _impl entry
    points. If the two copies diverge, a flag-on provider build fails to link --
    an undefined _impl symbol, or a public name that no longer resolves.
    """
    if lib == provider:
        print(f"PASS: provider rename header mirrors the {len(lib)} library renames")
        return True
    print("FAIL: the provider rename header has drifted from the library's")
    for name in sorted(set(lib) - set(provider)):
        print(f"  - in the library, missing from the provider: {name}")
    for name in sorted(set(provider) - set(lib)):
        print(f"  + in the provider, missing from the library: {name}")
    for name in sorted(set(lib) & set(provider)):
        if lib[name] != provider[name]:
            print(f"  ~ {name}: library -> {lib[name]}, provider -> {provider[name]}")
    return False


def check_wrapper_forwards(forwards: dict[str, set[str]]) -> bool:
    bad = {n: c for n, c in sorted(forwards.items()) if c != {f"{n}_impl"}}
    if not bad:
        print(
            f"PASS: all {len(forwards)} wrapper stubs forward to their own _impl symbol"
        )
        return True
    print(
        "FAIL: wrapper stubs do not forward to their own _impl symbol. A stub must "
        "be a pure forward -- logic here diverges the two libraries:"
    )
    for name, calls in bad.items():
        got = ", ".join(sorted(calls)) if calls else "no _impl call"
        print(f"  {name} calls {got} (expected {name}_impl)")
    return False


def check_wrapper_dispatch(dispatches: dict[str, list[str]]) -> bool:
    """Assert every stub opens with MIOPEN_WRAPPER_DISPATCH naming itself."""
    findings: list[str] = []
    for name, args in sorted(dispatches.items()):
        exempt_reason = DISPATCH_EXEMPT.get(name)
        if exempt_reason is not None:
            if args:
                findings.append(
                    f"  {name} carries MIOPEN_WRAPPER_DISPATCH but is exempt"
                    f" ({exempt_reason}); drop the macro or the exemption"
                )
        elif not args:
            findings.append(f"  {name} has no MIOPEN_WRAPPER_DISPATCH")
        elif len(args) > 1:
            findings.append(
                f"  {name} has {len(args)} MIOPEN_WRAPPER_DISPATCH calls"
                f" ({', '.join(args)}); expected exactly one"
            )
        elif args[0] != name:
            findings.append(
                f"  {name} dispatches as {args[0]}; expected {name}."
                " A stub cloned from its neighbour keeps the neighbour's name"
            )
    # An exemption for a stub that no longer exists has stopped documenting
    # anything, and would silently excuse the name if it ever came back.
    exempt = sorted(DISPATCH_EXEMPT.keys() & dispatches.keys())
    for name in sorted(DISPATCH_EXEMPT.keys() - dispatches.keys()):
        findings.append(
            f"  {name} is exempt from MIOPEN_WRAPPER_DISPATCH but has no stub;"
            " drop the exemption"
        )
    if not findings:
        print(
            f"PASS: all {len(dispatches) - len(exempt)} routable wrapper stubs"
            f" dispatch under their own name ({len(exempt)} exempt)"
        )
        return True
    print("FAIL: wrapper stubs are not all routable through the dispatch seam:")
    for line in findings:
        print(line)
    return False


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_dump_symbols(args) -> int:
    elf = open_elf(args.lib, "library")
    text = "\n".join(sorted(public_api_symbols(elf))) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_dump(args) -> int:
    elf = open_elf(args.lib, "library")
    prefix = Path(args.prefix)
    prefix.with_suffix(prefix.suffix + ".symbols").write_text(
        "\n".join(sorted(public_api_symbols(elf))) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(prefix.suffix + ".soname").write_text(
        elf.soname() + "\n", encoding="utf-8"
    )
    prefix.with_suffix(prefix.suffix + ".needed").write_text(
        "\n".join(elf.needed()) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(prefix.suffix + ".sha256").write_text(
        hashlib.sha256(elf.data).hexdigest() + "\n", encoding="utf-8"
    )
    print(f"wrote {prefix}.{{symbols,soname,needed,sha256}}")
    return 0


def cmd_check(args) -> int:
    elf = open_elf(args.lib, "library")
    baseline = read_symbol_list(args.baseline, "baseline symbols file")

    ok = check_soname(elf)
    ok &= check_symbols(
        elf,
        baseline,
        "exported public API symbol set matches baseline",
        BASELINE_REMEDY,
    )
    ok &= check_no_impl(elf, "flag-off libMIOpen.so")
    ok &= check_private_dep(elf, expect_present=False)
    if args.needed_baseline:
        ok &= check_needed_baseline(elf, args.needed_baseline)

    print(f"public-abi symbol check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_check_wrapper(args) -> int:
    elf = open_elf(args.lib, "wrapper library")
    baseline = read_symbol_list(args.baseline, "baseline symbols file")
    excluded = read_symbol_list(args.excluded, "excluded symbols file")

    ok = check_soname(elf)

    # A stale exclusion entry (a symbol no longer in the baseline at all) would
    # silently mask real drift, so reject it before using the set.
    stray = sorted(excluded - baseline)
    if stray:
        print("FAIL: excluded-symbols file lists symbols absent from the baseline:")
        for sym in stray:
            print(f"  {sym}")
        ok = False

    ok &= check_symbols(
        elf,
        baseline - excluded,
        "wrapper public API set == baseline - excluded",
        BASELINE_REMEDY,
    )
    ok &= check_no_impl(elf, "wrapper")
    ok &= check_private_dep(elf, expect_present=True)

    if args.public_header:
        ok &= check_excluded_not_public(excluded, args.public_header)
    if args.private_lib:
        ok &= check_excluded_on_private(excluded, args.private_lib)

    print(f"wrapper public-abi symbol check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_check_headers(args) -> int:
    root = Path(args.source_root).resolve()
    repo_root = root.parent.parent
    # Every input is resolved the same way: an explicitly requested path must
    # exist, while a default one falls back to git so that a sparse checkout,
    # where the file is part of the commit but not on disk, still gets checked.
    # A file git does not track either is genuinely not part of this tree, and
    # the caller decides whether that is a skip or a failure.
    reasons: dict[str, str] = {}

    def resolve(override: str | None, default_rel: str, what: str) -> str | None:
        if override:
            return read_source(Path(override), what)
        return read_tracked_source(repo_root / default_rel, repo_root, reasons)

    miopen_rel = root.relative_to(repo_root).as_posix()
    public_source = resolve(
        args.public_header, f"{miopen_rel}/include/miopen/miopen.h", "miopen.h"
    )
    impl_source = resolve(
        args.impl_header, f"{miopen_rel}/src/private/miopen_impl.h", "miopen_impl.h"
    )
    rename_source = resolve(
        args.rename_header,
        f"{miopen_rel}/src/private/miopen_private_rename.h",
        "miopen_private_rename.h",
    )
    wrapper_source = resolve(
        args.wrapper, f"{miopen_rel}/src/private/wrapper.cpp", "wrapper.cpp"
    )
    range_defs_source = resolve(
        args.range_definitions,
        f"{miopen_rel}/src/convolution_api.cpp",
        "convolution_api.cpp",
    )
    range_test_source = resolve(
        args.range_test,
        f"{miopen_rel}/test/gtest/conv_workspace_size_range.cpp",
        "conv_workspace_size_range.cpp",
    )
    provider_source = resolve(
        args.provider_rename, PROVIDER_RENAME_RELPATH, "provider rename header"
    )
    provider_api_source = resolve(
        args.provider_api, PROVIDER_API_RELPATH, "MiopenApi.hpp"
    )

    # A checkout that does not carry projects/miopen -- a provider-only PR,
    # whose pre-commit lane materializes only the projects its diff touches --
    # has no reference for this check to compare the provider against, so it is
    # not a finding. Reaching here means git confirmed the commit does not carry
    # the file, so it is absent rather than unreadable: a checkout that carries
    # it but cannot produce it has already failed hard above. Absence is never
    # itself the drift the gate catches, since drift is a content mismatch
    # between files that are all part of the commit.
    miopen_sources = {
        f"{miopen_rel}/include/miopen/miopen.h": public_source,
        f"{miopen_rel}/src/private/miopen_impl.h": impl_source,
        f"{miopen_rel}/src/private/miopen_private_rename.h": rename_source,
        f"{miopen_rel}/src/private/wrapper.cpp": wrapper_source,
        f"{miopen_rel}/src/convolution_api.cpp": range_defs_source,
        f"{miopen_rel}/test/gtest/conv_workspace_size_range.cpp": range_test_source,
    }
    missing = sorted(rel for rel, source in miopen_sources.items() if source is None)
    if missing:
        print(
            f"SKIP: {len(missing)} of MIOpen's own sources could not be read;"
            " nothing to check the split against"
        )
        for rel in missing:
            print(f"  {rel}: {reasons.get(rel, 'unreadable')}")
        return 0

    public = parse_declarations(public_source, EXPORT_ANCHOR_RE, "miopen.h")
    if not public:
        raise AbiError("no MIOPEN_EXPORT declarations found in miopen.h")
    impl = parse_declarations(impl_source, EXTERN_C_ANCHOR_RE, "miopen_impl.h")
    renames = parse_renames(rename_source)
    wrapper, forwards, dispatches = parse_wrapper(wrapper_source)

    # The private declarations carry the suffix; strip it so every comparison
    # below is against the public header under one common set of names. An
    # unsuffixed declaration is a defect, not something to silently rekey.
    unsuffixed = sorted(n for n in impl if not n.endswith("_impl"))
    if unsuffixed:
        print("FAIL: miopen_impl.h declares entry points without the _impl suffix:")
        for name in unsuffixed:
            print(f"  {name}")
    impl = {n[: -len("_impl")]: sig for n, sig in impl.items() if n.endswith("_impl")}

    ok = not unsuffixed
    ok &= check_entry_point_set(set(public), set(impl), "miopen_impl.h")
    ok &= check_prototypes(public, impl, "miopen_impl.h")
    ok &= check_entry_point_set(set(public), set(renames), "miopen_private_rename.h")
    ok &= check_rename_targets(renames)
    ok &= check_entry_point_set(set(public), set(wrapper), "wrapper.cpp")
    ok &= check_prototypes(public, wrapper, "wrapper.cpp")
    ok &= check_wrapper_forwards(forwards)
    # Kept out of ``ok`` until the remedies are printed: a routing defect is not
    # the five-artifact drift HEADERS_REMEDY talks about and has its own fix.
    dispatch_ok = check_wrapper_dispatch(dispatches)
    provider_missing = False
    if provider_source is None:
        provider_missing = args.require_provider
        print(
            f"{'FAIL' if provider_missing else 'SKIP'}: provider rename header is"
            f" not part of this tree ({PROVIDER_RENAME_RELPATH}:"
            f" {reasons.get(PROVIDER_RENAME_RELPATH, 'unreadable')})"
        )
    else:
        provider_renames = parse_renames(provider_source, "provider rename header")
        ok &= check_provider_rename_mirror(renames, provider_renames)
    headers_ok = ok

    # The range entry points are a separate family: exported but undeclared in
    # miopen.h, so their reference is the definition rather than the header.
    definitions = parse_range_prototypes(range_defs_source, "convolution_api.cpp")
    range_ok = check_range_entry_point_set(definitions, "convolution_api.cpp")
    consumers = [("conv_workspace_size_range.cpp", range_test_source)]
    if provider_api_source is None:
        provider_missing |= args.require_provider
        print(
            f"{'FAIL' if args.require_provider else 'SKIP'}:"
            f" {PROVIDER_API_RELPATH} is not part of this tree"
            f" ({reasons.get(PROVIDER_API_RELPATH, 'unreadable')})"
        )
    else:
        consumers.append(("MiopenApi.hpp", provider_api_source))
    for what, source in consumers:
        declared = parse_range_prototypes(source, what)
        range_ok &= check_range_entry_point_set(declared, what)
        range_ok &= check_prototypes(definitions, declared, what, "convolution_api.cpp")

    if not headers_ok:
        print(HEADERS_REMEDY)
    if not dispatch_ok:
        print(DISPATCH_REMEDY)
    if not range_ok:
        print(RANGE_REMEDY)
    if provider_missing:
        print(PROVIDER_REMEDY)
    ok &= dispatch_ok
    ok &= range_ok
    # A skipped file is not a checked one. Folding this in last keeps it out of
    # the two remedies above, which are about mismatches rather than readability.
    ok &= not provider_missing
    print(f"public/private header consistency check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_compare_pair(args) -> int:
    ok = True
    for ext in ("soname", "symbols", "needed"):
        base = Path(f"{args.base_prefix}.{ext}")
        cand = Path(f"{args.candidate_prefix}.{ext}")
        if not base.is_file() or not cand.is_file():
            raise AbiError(f"missing dump file: {base if not base.is_file() else cand}")
        b = set(base.read_text(encoding="utf-8").split())
        c = set(cand.read_text(encoding="utf-8").split())
        if b == c:
            print(f"PASS: {ext} identical")
        else:
            print(f"FAIL: {ext} differs:")
            print(format_set_diff(b, c))
            ok = False

    base_hash = Path(f"{args.base_prefix}.sha256")
    cand_hash = Path(f"{args.candidate_prefix}.sha256")
    if base_hash.is_file() and cand_hash.is_file():
        same = base_hash.read_text().strip() == cand_hash.read_text().strip()
        print(
            "INFO: SHA256 identical"
            if same
            else "INFO: SHA256 differs (expected under build-path nondeterminism; not gated)"
        )
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_public_abi.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="gate a flag-off (default) build")
    p.add_argument("lib", help="path to the built libMIOpen.so")
    p.add_argument("--baseline", required=True, help="committed public symbol baseline")
    p.add_argument(
        "--needed-baseline",
        help="optional committed DT_NEEDED baseline; when given, the full "
        "runtime dependency list must match it exactly",
    )
    p.set_defaults(func=cmd_check)

    p = sub.add_parser(
        "check-headers",
        help="cross-check the hand-maintained split sources (no build needed)",
    )
    p.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="MIOpen source root; the checked files are located under it by "
        "convention, read from HEAD when the checkout is sparse, and the check "
        "skipped only when git confirms the commit tracks none of them "
        "(default: the tree containing this script)",
    )
    p.add_argument("--public-header", help="override path to include/miopen/miopen.h")
    p.add_argument("--impl-header", help="override path to src/private/miopen_impl.h")
    p.add_argument(
        "--rename-header", help="override path to src/private/miopen_private_rename.h"
    )
    p.add_argument("--wrapper", help="override path to src/private/wrapper.cpp")
    p.add_argument(
        "--provider-rename",
        help="override path to the hipDNN provider's MiopenApiPrivateRename.hpp, "
        "which must then exist; by default it is located relative to the "
        "repository root, read from HEAD when the checkout is sparse, and "
        "skipped only when git confirms the commit does not track it",
    )
    p.add_argument(
        "--provider-api",
        help="override path to the hipDNN provider's MiopenApi.hpp, which must "
        "then exist; located and skipped on the same terms as --provider-rename",
    )
    p.add_argument(
        "--range-definitions",
        help="override path to src/convolution_api.cpp, which defines the "
        "miopenConvolution*GetWorkSpaceSizeRange entry points",
    )
    p.add_argument(
        "--range-test",
        help="override path to test/gtest/conv_workspace_size_range.cpp",
    )
    p.add_argument(
        "--require-provider",
        action="store_true",
        help="fail instead of skipping when the provider's two copies cannot be "
        "read, so that a tree which does not include them cannot report a pass "
        "for files it never opened; pass this wherever the provider is expected "
        "to be present, such as a monorepo CI lane",
    )
    p.set_defaults(func=cmd_check_headers)

    p = sub.add_parser("check-wrapper", help="gate a flag-on wrapper build")
    p.add_argument("lib", help="path to the built wrapper libMIOpen.so")
    p.add_argument("--baseline", required=True, help="committed public symbol baseline")
    p.add_argument("--excluded", required=True, help="committed excluded symbol list")
    p.add_argument(
        "--private-lib",
        help="path to libMIOpen_private.so; when given, every excluded symbol "
        "must still be exported from it under its original name",
    )
    p.add_argument(
        "--public-header",
        help="path to include/miopen/miopen.h; when given, no excluded symbol "
        "may be declared there",
    )
    p.set_defaults(func=cmd_check_wrapper)

    p = sub.add_parser("dump-symbols", help="print/write the public symbol set")
    p.add_argument("lib")
    p.add_argument("-o", "--output", help="write here instead of stdout")
    p.set_defaults(func=cmd_dump_symbols)

    p = sub.add_parser("dump", help="write <prefix>.{symbols,soname,needed,sha256}")
    p.add_argument("lib")
    p.add_argument("prefix")
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("compare-pair", help="diff two dump outputs")
    p.add_argument("base_prefix")
    p.add_argument("candidate_prefix")
    p.set_defaults(func=cmd_compare_pair)

    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except AbiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
