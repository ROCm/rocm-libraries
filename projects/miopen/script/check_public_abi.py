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
MIGraphX shims) stay on libMIOpen_private.so under their original names. The
wrapper also exports a few functions of its own that report on hipDNN
forwarding; a flag-off build has no forwarding, so they are in no baseline.

  1. SONAME is libMIOpen.so.1 (unchanged).
  2. The wrapper's exported public C API set equals
     ``(baseline - excluded) | wrapper-only``.
  3. No wrapper-only symbol is in the baseline. A function both builds export
     belongs in miopen.h and the baseline, not in the wrapper-only list.
  4. No ``*_impl`` symbols are exported from the wrapper.
  5. libMIOpen_private IS in DT_NEEDED, and libhipdnn_backend is NOT: the
     wrapper opens the backend on first use, so a machine without hipDNN must
     still be able to load libMIOpen.so.
  6. With ``--public-header``, no excluded or wrapper-only symbol is declared in
     miopen.h, so the exclusion file cannot hide a real miopen.h entry point,
     and a function both builds export cannot hide in the wrapper-only list.
  7. Every excluded symbol is still exported, un-suffixed, from the private
     library (``--private-lib``), so the carve-out cannot drop a symbol from
     the whole installed surface.
  8. With ``--private-lib``, the libraries are co-versioned: their SONAME majors
     agree and the wrapper's DT_NEEDED entry names that same major, so the
     wrapper cannot bind to an implementation it was not built against.
  9. With ``--private-lib``, every ``miopenFoo_impl`` the private library
     exports has a matching ``miopenFoo`` on the wrapper; otherwise it is
     unreachable through libMIOpen.so.
  10. Optionally (``--needed-baseline``), the full DT_NEEDED list matches a
      committed baseline. Opt-in because it names ROCm soversions and the
      platform's loader, so it changes with routine ROCm updates.

``check-installed-headers`` -- run on the staged include tree of a flag-on
build. Fails if the private rename spelling (``miopenFoo_impl``) appears in any
installed header outside the private include directory. The installed headers
are the public contract; a rename that leaks into one makes a consumer compile
against a name only the private library carries.

``check-headers`` -- a source-only cross-check. It needs no build, no GPU and no
flag-on configuration, so unlike the two gates above it can run in a lint lane
on every PR. It covers only what generation cannot replace.
script/generate_wrapper_sources.py writes the rename headers, miopen_impl.h, the
provider's rename mirror and the pass-through stubs in wrapper_passthrough.cpp
from include/miopen/miopen.h, and its --check mode is the gate for all of them,
signatures included. What is left here:

  1. The stubs in src/private/wrapper.cpp, the few with logic of their own. Each
     must forward to its own _impl symbol and nothing else.

     Each must also open with a dispatch macro -- MIOPEN_WRAPPER_DISPATCH or
     MIOPEN_WRAPPER_FORWARD -- naming itself. The macro's runtime assert cannot
     cover this: it is compiled out under NDEBUG, never fires for a stub nothing
     calls, and a stub missing the macro has no assert at all. Such a stub can
     never route to hipDNN, and nobody notices until the entry point joins the
     forwarding set and keeps running MIOpen anyway.

  2. The three miopenConvolution*GetWorkSpaceSizeRange entry points. These are
     exported with MIOPEN_EXPORT but never declared in miopen.h, so the
     generator does not see them. Their definitions in src/convolution_api.cpp
     are the reference, and the declaration in the hipDNN provider's
     MiopenApi.hpp must match it. MIOpen's own callers share the declaration in
     src/include/miopen/miopen_internal.h, which the compiler checks against the
     definitions. The provider cannot include MIOpen's internal headers, so its
     copy stays by hand and stays checked here.

  3. The wrapper's own functions, which only a flag-on build exports. They are
     declared in include_wrapper/miopen/hipdnn_forwarding.h, never in miopen.h,
     and listed in test/public_abi/wrapper_only_symbols.txt for
     ``check-wrapper``; the header and the list must name the same functions.
     Each has a stub in wrapper.cpp that must match that header's signature but
     does the work itself: it calls no _impl and carries no dispatch macro. The
     generator writes nothing for them. The header and the list are optional:
     if the commit tracks neither, there are no wrapper-only functions and
     every other check still runs.

The provider's header lives in a sibling project that a MIOpen-only checkout does
not ship, so it is the one input that can be skipped. It is skipped only once
git confirms the commit under test does not track it -- being absent from a
sparse working tree is not enough, or CI, which checks out only the subtrees a
PR touches, would report a green gate on unchecked drift.

The Range declarations have C linkage, so a divergence from the definitions links
cleanly and corrupts arguments at runtime. The symbol-set gates above cannot see
it -- the exported names are unchanged.

Deliberately NOT verified: struct/enum layout, behaviour, exported symbols
outside the ``miopen[A-Z]`` C API convention (including the mangled
``_ZN6miopen...`` internals that the driver, gtests and CK plugins link
against), symbol addresses/sizes, and dynamic-symbol ordering. ``check`` and
``check-wrapper`` additionally do not verify signatures; the generator's
``--check`` and ``check-headers`` do. Signature comparison is textual, not
semantic: typedefs are not resolved, by design.

``compare-pair`` diffs two ``dump`` outputs and reports a content hash for
information only; a content hash is build-path dependent and is never gated.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXPECTED_SONAME = "libMIOpen.so.1"
PRIVATE_LIB_PREFIX = "libMIOpen_private"
HIPDNN_BACKEND_PREFIX = "libhipdnn_backend"

# Public MIOpen C API naming convention: "miopen" followed by an uppercase
# letter. This deliberately excludes internal exported shims such as
# miopen_sqlite3_memvfs_init.
PUBLIC_API_RE = re.compile(r"^miopen[A-Z]")
IMPL_RE = re.compile(r"^miopen[A-Za-z0-9_]*_impl$")

# Directories, relative to the staged include tree, that may spell entry points
# by their private names. Nothing stages private declarations today; this is
# here for a future install rule.
PRIVATE_INCLUDE_DIRS = ("miopen/private",)

# Where MIOpen stages its public headers, relative to the staged include tree.
PUBLIC_INCLUDE_DIR = "miopen"

# What counts as a header in the staged include tree. MIOpen stages .h and .hpp;
# the rest are here because a consumer can include any of them.
HEADER_SUFFIXES = frozenset({".h", ".hpp", ".hh", ".hxx", ".inc", ".ipp"})


class AbiError(Exception):
    """Fatal, non-assertion problem (bad file, missing input, unreadable ELF)."""


# --------------------------------------------------------------------------
# Minimal ELF reader
#
# Parsed directly instead of via nm/readelf, so the script runs anywhere Python
# does (including a Windows host inspecting a Linux build artifact) and avoids
# the locale-collation and pipeline-exit-code hazards of a shell implementation.
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
    return parse_symbol_list(path.read_text(encoding="utf-8"))


def parse_symbol_list(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
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
and rerun projects/miopen/script/generate_wrapper_sources.py to regenerate the
wrapper's derived files from miopen.h.
A baseline change is an ABI change and requires API review -- do not regenerate
the baseline just to turn this gate green.
A function that only the flag-on wrapper exports does not go in the baseline:
list it in projects/miopen/test/public_abi/wrapper_only_symbols.txt and declare
it in include_wrapper/miopen/hipdnn_forwarding.h, not in miopen.h.""".rstrip()


WRAPPER_ONLY_REMEDY = """
A function only the flag-on wrapper exports is declared in
include_wrapper/miopen/hipdnn_forwarding.h, never in miopen.h, listed in
test/public_abi/wrapper_only_symbols.txt, and given a stub in
src/private/wrapper.cpp that matches that declaration, calls no _impl and
carries no dispatch macro. It goes in none of the generated files.""".rstrip()

DISPATCH_REMEDY = """
Every stub in src/private/wrapper.cpp must open with exactly one dispatch macro,
passing its own function token. Use MIOPEN_WRAPPER_DISPATCH where there is
nothing to forward to yet:
  extern "C" miopenStatus_t miopenFoo(<params>)
  {
      MIOPEN_WRAPPER_DISPATCH(miopenFoo);
      return miopenFoo_impl(<args>);
  }
and MIOPEN_WRAPPER_FORWARD where there is, passing the hipDNN call as well:
  extern "C" miopenStatus_t miopenBar(<params>)
  {
      MIOPEN_WRAPPER_FORWARD(miopenBar, ::miopen::wrapper::hipdnn::Bar(<args>));
      return miopenBar_impl(<args>);
  }
Pass the token, never a string and never a neighbouring stub's name. A stub
without either macro can never route to hipDNN, and nothing else reports that:
the macro's assert needs the macro to be there, and is compiled out under NDEBUG
in any case.""".rstrip()

RANGE_REMEDY = """
Each miopenConvolution*GetWorkSpaceSizeRange entry point is exported without
being declared in miopen.h, so the provider spells it by hand:
  src/convolution_api.cpp                      the definition (the reference)
  <repo>/dnn-providers/miopen-provider/MiopenApi.hpp
                                               local extern "C" declaration
Update the declaration to match the definition. It has C linkage, so a
divergence links cleanly and corrupts arguments at run time. MIOpen's own
callers use the declaration in src/include/miopen/miopen_internal.h, which the
compiler checks.""".rstrip()

INSTALLED_HEADERS_REMEDY = """
An installed header names an entry point by its private spelling. The installed
headers are what a consumer compiles against, and only libMIOpen_private.so
carries a miopenFoo_impl symbol, so such a consumer either fails to link or
binds past the wrapper.
The usual cause is a header that was force-included through
src/private/miopen_private_rename.h while being generated or copied, so the
rename was baked into the staged text. Regenerate it with the rename inactive.
Do not add the offending directory to the exemption list unless it genuinely
holds the private declarations -- the exemption is by path so that a leak into a
similarly-named header elsewhere still trips.""".rstrip()

COVERSION_REMEDY = """
The wrapper and the private library are two halves of one build and must ship
the same soversion, with the wrapper's DT_NEEDED edge naming it. A mismatch
means the two binaries came from different builds -- almost always a stale build
or install directory holding one half of an earlier configuration. Rebuild both
from a clean directory rather than adjusting a version to match.""".rstrip()

SUPERSET_REMEDY = """
The private library exports a renamed entry point that the wrapper does not
re-export under its public name, so nothing outside libMIOpen_private.so can
call it. Only entry points declared in miopen.h are renamed, and the wrapper's
stubs are generated from the same declarations, so the generated files are
stale: rerun projects/miopen/script/generate_wrapper_sources.py.""".rstrip()

PROVIDER_REMEDY = """
--require-provider was passed, so the provider's MiopenApi.hpp must be readable
rather than skipped. Make dnn-providers/miopen-provider part of the tree being checked:
in a sparse checkout, add that directory to the checkout's path list. Drop
--require-provider only for a tree that genuinely has no provider, such as a
MIOpen-only checkout or a source tarball -- not to get past this message.""".rstrip()

# The hipDNN provider's hand-written declarations, relative to the repository
# root (the MIOpen source root's grandparent in the monorepo layout).
PROVIDER_API_RELPATH = "dnn-providers/miopen-provider/MiopenApi.hpp"

# Exported with MIOPEN_EXPORT from src/convolution_api.cpp, absent from
# miopen.h, and so declared by hand in the provider, which cannot include
# MIOpen's internal headers. Listed explicitly rather than discovered from the
# definitions so that dropping a declaration from a consumer fails the check
# instead of shrinking the comparison set.
RANGE_ENTRY_POINTS = frozenset(
    {
        "miopenConvolutionForwardGetWorkSpaceSizeRange",
        "miopenConvolutionBackwardDataGetWorkSpaceSizeRange",
        "miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange",
    }
)

# Stubs that must NOT carry a dispatch macro, and why. MIOPEN_WRAPPER_DISPATCH
# returns ForwardingNotImplemented's miopenStatus_t, so a stub returning anything
# else cannot host it. Exemptions are checked in both directions: an exempt
# stub that grows the macro fails here rather than failing to compile somewhere
# less obvious.
DISPATCH_EXEMPT = {
    "miopenGetErrorString": "returns const char*, not miopenStatus_t",
}


# --------------------------------------------------------------------------
# Minimal C declaration parser.
#
# Reading the files as text instead of invoking a compiler keeps this check free
# of any build, toolchain or GPU dependency, so it can run in a lint lane on
# every PR. A prototype is reduced to its return type and parameter types (names
# dropped) and compared as strings. Typedefs are not resolved, so two spellings
# of one type read as a mismatch. That is deliberate: the provider's declarations
# should be transcriptions of the definitions, so a different spelling deserves
# a human look.
#
# Function-pointer and array parameters cannot be parsed; none occur in the range
# entry points. A declaration that fails to parse raises instead of being
# skipped, so adding one fails loudly rather than leaving a silent coverage hole.
# --------------------------------------------------------------------------

WRAPPER_DEF_RE = re.compile(
    r'\bextern\s+"C"\s+(?P<decl>[^;{}]*?)\s*\{(?P<body>[^{}]*)\}', re.S
)
DECL_RE = re.compile(
    r"(?P<ret>.*?)(?P<name>miopen[A-Za-z0-9_]*)\s*\((?P<params>[^()]*)\)"
)
IMPL_CALL_RE = re.compile(r"\b(miopen[A-Za-z0-9_]*_impl)\s*\(")
# One pattern matches both dispatch macros, and only as far as the entry point
# token: FORWARD's second argument is an arbitrary call expression with its own
# parentheses, which a regex should not chase. Matching both together reports a
# stub carrying one of each as two dispatch macros. Each macro keeps its own
# arity, so a DISPATCH given a second argument (a half-converted FORWARD, or the
# reverse) does not count and the stub is reported.
DISPATCH_RE = re.compile(
    r"\bMIOPEN_WRAPPER_(?:"
    r"DISPATCH\s*\(\s*(?P<dispatch>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
    r"|FORWARD\s*\(\s*(?P<forward>[A-Za-z_][A-Za-z0-9_]*)\s*,"
    r")"
)
# A declarator for one of the range entry points, in either of the two forms it
# is written in: a declaration ending in ';' and a definition followed by its
# body's '{'.
RANGE_DECL_RE = re.compile(
    r"\b(?P<name>miopen[A-Za-z0-9_]*GetWorkSpaceSizeRange)\s*"
    r"\((?P<params>[^()]*)\)\s*(?=[;{])"
)
# Characters that can precede a declarator's return type. A call site is also
# preceded by one, but with nothing between it and the name, which is how the two
# are told apart. '=' is included so that `auto r = miopenFoo(...)` leaves
# nothing before the name either.
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
    """Remove a top-level const, which is not part of the parameter's type.

    ``const float`` and ``float`` are the same type, as are ``void* const`` and
    ``void*``. A const below the top level is part of the type, so ``const
    void*`` (a pointer to const) is kept. A const after the last ``*``, or a
    leading const on a type with no ``*``, is top level.
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
    if "[" in param:
        raise AbiError(
            f"cannot parse array parameter {param.strip()!r}; give it a typedef"
        )
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


# The value each macro has while miopen.h and hipdnn_forwarding.h are read, as
# the library's build sees it. Conditionals are evaluated against this table
# rather than flattened, so an #else branch cannot contribute declarations.
# MIOPEN_BETA_API is defined for the whole build in CMakeLists.txt. The include
# guard reads as undefined because this is the first time the header is seen.
# __cplusplus and __clang__ only guard text that declares no entry point.
KNOWN_MACROS = {
    "MIOPEN_BETA_API": True,
    "__cplusplus": True,
    "__clang__": True,
    "MIOPEN_GUARD_MIOPEN_H_": False,
}

_COMMENT_OR_LITERAL_RE = re.compile(
    r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|(?P<open>/\*)",
    re.S,
)
# Words that qualify or introduce a type name without being part of its spelling.
_TYPE_KEYWORDS = frozenset({"const", "volatile", "struct", "enum", "union"})
_DIRECTIVE_RE = re.compile(r"^\s*#\s*(?P<kw>\w*)\s*(?P<rest>.*)$")
_IDENT_TAIL_RE = re.compile(r"([A-Za-z_]\w*)\s*$")
_HEADER_DECL_RE = re.compile(
    r"(?P<ret>[^()]*?)\s*(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*)\)", re.S
)


def strip_comments(text: str, what: str) -> str:
    """Blank out comments without moving any line, so errors can name one.

    Strings are matched too, so a ``//`` or ``/*`` inside one is not taken for a
    comment.
    """

    def blank(match: re.Match) -> str:
        if match.group("open"):
            line = text.count("\n", 0, match.start()) + 1
            raise AbiError(f"{what}:{line}: unterminated /* comment")
        found = match.group(0)
        if found[0] in "\"'":
            return found
        return " " + "\n" * found.count("\n")

    return _COMMENT_OR_LITERAL_RE.sub(blank, text)


def active_text(text: str, what: str) -> str:
    """Return ``text`` with comments, directives and inactive branches blanked.

    Only #ifdef, #ifndef, #else and #endif are understood. Anything that would
    need a real preprocessor is refused rather than guessed at, because a
    declaration dropped by a wrong guess is an entry point that silently has no
    stub.
    """
    lines = strip_comments(text, what).split("\n")
    out: list[str] = []
    # One entry per open conditional: [branch is taken, #else seen, opening line].
    stack: list[list] = []
    continued = False

    for number, line in enumerate(lines, start=1):
        if continued:
            continued = line.rstrip().endswith("\\")
            out.append("")
            continue
        directive = _DIRECTIVE_RE.match(line)
        if directive is None:
            out.append(line if all(entry[0] for entry in stack) else "")
            continue

        out.append("")
        continued = line.rstrip().endswith("\\")
        keyword, rest = directive.group("kw"), directive.group("rest").strip()
        if keyword in ("if", "elif", "elifdef", "elifndef"):
            raise AbiError(
                f"{what}:{number}: #{keyword} is not supported. Only #ifdef, #ifndef,"
                " #else and #endif are evaluated, against KNOWN_MACROS in"
                " check_public_abi.py"
            )
        if keyword in ("ifdef", "ifndef"):
            macro = rest.split()[0] if rest else ""
            if macro not in KNOWN_MACROS:
                raise AbiError(
                    f"{what}:{number}: #{keyword} {macro or '(nothing)'}: this macro is"
                    " not in KNOWN_MACROS in check_public_abi.py, so it is not"
                    " known whether the declarations it guards belong in the wrapper."
                    " Add it with the value the library build gives it"
                )
            defined = KNOWN_MACROS[macro]
            stack.append(
                [defined if keyword == "ifdef" else not defined, False, number]
            )
        elif keyword == "else":
            if not stack or stack[-1][1]:
                raise AbiError(f"{what}:{number}: #else without a matching #ifdef")
            stack[-1][0] = not stack[-1][0]
            stack[-1][1] = True
        elif keyword == "endif":
            if not stack:
                raise AbiError(f"{what}:{number}: #endif without a matching #ifdef")
            stack.pop()

    if stack:
        raise AbiError(f"{what}:{stack[-1][2]}: conditional is never closed by #endif")
    return "\n".join(out)


@dataclass(frozen=True)
class Param:
    type: str
    name: str
    default: str | None

    def declaration_text(self) -> str:
        """The parameter as an _impl declaration spells it.

        A top-level const is dropped: it is not part of the function's type, so it
        adds nothing to a declaration. A default argument stays, as the public
        header has it.
        """
        text = f"{drop_top_level_const(self.type)} {self.name}"
        return f"{text} = {self.default}" if self.default is not None else text

    def definition_text(self) -> str:
        """The parameter as written in miopen.h, minus the default argument.

        Repeating a default argument in a definition is a compile error.
        """
        return f"{self.type} {self.name}"


@dataclass(frozen=True)
class EntryPoint:
    name: str
    ret: str
    params: tuple[Param, ...]
    line: int


def parse_param(text: str, function: str, where: str) -> Param:
    if "..." in text:
        raise AbiError(f"{where}: {function}: variadic parameters are not supported")
    if "(" in text or "[" in text:
        raise AbiError(
            f"{where}: {function}: cannot parse parameter {text.strip()!r}; a function"
            " pointer or array parameter must be given a typedef"
        )
    declared, equals, default = text.partition("=")
    declared = " ".join(declared.split())
    tail = _IDENT_TAIL_RE.search(declared)
    type_text = declared[: tail.start()].strip() if tail else ""
    name = tail.group(1) if tail else ""
    # A trailing identifier that is a type word, is spelled like a typedef, or
    # follows only qualifiers and struct/enum/union keywords is part of an unnamed
    # parameter's type, not a name. The call into the _impl function needs every
    # name, so an unnamed one cannot be generated.
    if (
        not type_text
        or name in TYPE_TAIL_KEYWORDS
        or name.endswith("_t")
        or set(type_text.split()) <= _TYPE_KEYWORDS
    ):
        raise AbiError(
            f"{where}: {function}: parameter {declared!r} has no name; name every"
            " parameter in miopen.h"
        )
    return Param(
        type=squash(type_text),
        name=name,
        default=default.strip() if equals else None,
    )


def parse_public_header(
    source: str, what: str = "miopen.h", allow_empty: bool = False
) -> list[EntryPoint]:
    """Collect every MIOPEN_EXPORT declaration, in declaration order."""
    text = active_text(source, what)
    entries: dict[str, EntryPoint] = {}
    for match in re.finditer(r"\bMIOPEN_EXPORT\b", text):
        line = text.count("\n", 0, match.start()) + 1
        where = f"{what}:{line}"
        end = text.find(";", match.end())
        brace = text.find("{", match.end())
        if end < 0:
            raise AbiError(f"{where}: MIOPEN_EXPORT declaration is never terminated")
        if 0 <= brace < end:
            raise AbiError(
                f"{where}: a '{{' comes before the ';' of this MIOPEN_EXPORT"
                " declaration; only plain function declarations are supported"
            )
        declaration = " ".join(text[match.end() : end].split())
        parsed = _HEADER_DECL_RE.fullmatch(declaration)
        if parsed is None or not parsed.group("ret").strip():
            raise AbiError(
                f"{where}: cannot parse {declaration!r} as"
                " '<return type> <name>(<parameters>)'"
            )
        name = parsed.group("name")
        if name in entries:
            raise AbiError(
                f"{where}: {name} is declared again; first declared at line"
                f" {entries[name].line}"
            )
        params_text = parsed.group("params")
        if "(" in params_text or ")" in params_text:
            raise AbiError(
                f"{where}: {name}: a function-pointer parameter must be given a typedef"
            )
        params: list[Param] = []
        if params_text.strip() not in ("", "void"):
            for piece in split_params(params_text):
                if not piece.strip():
                    raise AbiError(f"{where}: {name}: empty parameter")
                params.append(parse_param(piece, name, where))
        entries[name] = EntryPoint(
            name=name, ret=squash(parsed.group("ret")), params=tuple(params), line=line
        )
    if not entries and not allow_empty:
        raise AbiError(f"no MIOPEN_EXPORT declarations found in {what}")
    return list(entries.values())


def stub_signature(entry: EntryPoint) -> tuple[str, ...]:
    """``entry`` reduced the way parse_wrapper reduces a stub, so the two compare.

    Parameter names, default arguments and a top-level const drop out: none of
    them changes the type.
    """
    params = ", ".join(p.definition_text() for p in entry.params)
    return parse_prototype(f"{entry.ret} {entry.name}({params})", "miopen.h")[1]


def parse_range_prototypes(source: str, what: str) -> dict[str, tuple[str, ...]]:
    """Collect the range entry points declared or defined in one source file.

    No anchor marks them as declarations: they sit in an ``extern "C" { ... }``
    block in the provider and are ``MIOPEN_EXPORT extern "C"`` definitions in
    the library, and the provider also *calls* them. Matches are therefore found
    by name, and a call is rejected because no return type sits between the name
    and the punctuation before it.
    """
    text = strip_comments(source, what)
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
    source: str, what: str = "wrapper.cpp"
) -> tuple[dict[str, tuple[str, ...]], dict[str, set[str]], dict[str, list[str]]]:
    """Collect each stub's prototype, the _impl symbols it calls, and the entry
    point named by its MIOPEN_WRAPPER_DISPATCH / MIOPEN_WRAPPER_FORWARD macros.

    Dispatches stay an ordered list rather than a set so that a stub carrying a
    dispatch macro twice is visible as such instead of collapsing into a
    correct-looking single entry.
    """
    text = strip_comments(source, what)
    protos: dict[str, tuple[str, ...]] = {}
    forwards: dict[str, set[str]] = {}
    dispatches: dict[str, list[str]] = {}
    for match in WRAPPER_DEF_RE.finditer(text):
        name, sig = parse_prototype(match.group("decl"), "wrapper")
        if name in protos:
            raise AbiError(f"duplicate definition of {name} in wrapper")
        protos[name] = sig
        forwards[name] = set(IMPL_CALL_RE.findall(match.group("body")))
        dispatches[name] = [
            m.group("dispatch") or m.group("forward")
            for m in DISPATCH_RE.finditer(match.group("body"))
        ]
    return protos, forwards, dispatches


def run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git against `repo_root`, immune to an ambient git environment.

    GIT_DIR overrides -C, and git exports it to the hooks it runs. This check runs
    as one, so without stripping it every read below would answer from whichever
    repository invoked the hook, whatever --repo-root was given.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        check=False,
        env=env,
    )


def describe_git_failure(repo_root: Path, rel: str, stderr: bytes) -> tuple[bool, str]:
    """Say why git could not produce HEAD:<rel>, and whether that is fatal.

    A path the commit does not carry is legitimately absent, and skipping it
    costs nothing. A path the commit *does* carry but whose blob git cannot
    produce (a blobless clone whose promisor fetch failed) is a broken checkout:
    skipping it would let this gate go green on a file under test, and the drift
    it catches links cleanly and corrupts arguments at run time. That case is
    fatal, so the checkout must be repaired and the job rerun.

    Only a case git positively confirms is escalated. If ls-tree cannot answer
    (it errors, or git is gone) the cause is ambiguous and stays a skip, because
    a false failure on every tarball build would be its own outage.
    Returns (fatal, reason).
    """
    text = stderr.decode("utf-8", "replace").lower()
    if "not a git repository" in text:
        return False, "not on disk, and the source root is not a git checkout"
    try:
        listed = run_git(repo_root, "ls-tree", "--name-only", "HEAD", "--", rel)
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
    can be missing from the working tree yet still be part of the commit under
    test (the provider's files for a MIOpen-only PR). Skipping on "not on disk"
    alone would let the gate go green on exactly the drift it exists to catch,
    so a missing file counts as absent only once git confirms the commit does
    not track it. Then this returns None, and the caller skips the check,
    recording the cause in ``reasons``. If git says the commit *does* track the
    file but cannot produce it, that is a broken checkout and this raises
    AbiError.
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
        blob = run_git(repo_root, "cat-file", "-p", f"HEAD:{rel}")
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


def check_no_backend_dep(elf: Elf) -> bool:
    linked = sorted(n for n in elf.needed() if n.startswith(HIPDNN_BACKEND_PREFIX))
    if not linked:
        print(f"PASS: DT_NEEDED on {HIPDNN_BACKEND_PREFIX} is absent as expected")
        return True
    print(
        f"FAIL: wrapper has DT_NEEDED on {', '.join(linked)}, which makes hipDNN a"
        " hard install dependency of every MIOpen consumer. Link the frontend's"
        " dynamic-loading variant instead"
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


def check_wrapper_only_not_public(wrapper_only: set[str], header_path: str) -> bool:
    path = Path(header_path)
    if not path.is_file():
        raise AbiError(f"public header not found: {path}")
    # Comments are stripped so miopen.h docs can mention a wrapper-only function
    # without it counting as a declaration.
    text = strip_comments(path.read_text(encoding="utf-8", errors="replace"), path.name)
    offenders = sorted(
        s for s in wrapper_only if re.search(rf"\b{re.escape(s)}\b", text)
    )
    if not offenders:
        print(
            f"PASS: no wrapper-only symbol appears in {path.name} "
            f"({len(wrapper_only)} checked)"
        )
        return True
    print(
        f"FAIL: wrapper-only symbols file lists entry points that ARE declared in "
        f"{path.name}. A function both builds export belongs in the baseline, not "
        "in the wrapper-only list:"
    )
    for sym in offenders:
        print(f"  {sym}")
    return False


def check_excluded_on_private(excluded: set[str], private: Elf) -> bool:
    exported = private.defined_dynamic_functions()
    missing = sorted(excluded - exported)
    if not missing:
        print(
            f"PASS: all {len(excluded)} excluded symbols are still exported "
            f"un-suffixed from {Path(private.path).name}"
        )
        return True
    print(
        f"FAIL: excluded symbols are absent from {Path(private.path).name} -- they "
        "have vanished from the entire installed surface (renamed by mistake?):"
    )
    for sym in missing:
        print(f"  {sym}")
    return False


def soname_major(soname: str) -> str:
    """Return the major soversion of an SONAME, or '' if it carries none.

    'libMIOpen.so.1' -> '1'; 'libMIOpen.so.1.2.3' -> '1'; 'libMIOpen.so' -> ''.
    """
    _, sep, tail = soname.partition(".so.")
    if not sep:
        return ""
    return tail.split(".", 1)[0]


def check_coversioned(wrapper: Elf, private: Elf) -> bool:
    """The wrapper and the private library must be halves of the same build.

    Two things have to agree: the major soversion each library declares, and
    the major named by the wrapper's DT_NEEDED edge onto the private library.
    The second is what a loader actually resolves, so checking only the
    declared SONAMEs would miss a wrapper linked against an older private
    library that happens to still be on disk.
    """
    wrapper_soname = wrapper.soname()
    private_soname = private.soname()
    needed = next(
        (n for n in wrapper.needed() if n.startswith(PRIVATE_LIB_PREFIX)),
        "",
    )

    wrapper_major = soname_major(wrapper_soname)
    private_major = soname_major(private_soname)
    needed_major = soname_major(needed)

    if wrapper_major and wrapper_major == private_major == needed_major:
        print(
            f"PASS: wrapper and private library are co-versioned at "
            f"soversion {wrapper_major}"
        )
        return True

    # Print all three strings: which one is the odd one out is the whole
    # diagnosis, and naming only the mismatched pair would point the reader at
    # the wrong place.
    print("FAIL: wrapper and private library are not co-versioned:")
    print(f"  wrapper SONAME:            {wrapper_soname or '(none)'}")
    print(f"  private SONAME:            {private_soname or '(none)'}")
    print(f"  wrapper DT_NEEDED on it:   {needed or '(none)'}")
    print(COVERSION_REMEDY)
    return False


def check_impl_superset(wrapper: Elf, private: Elf) -> bool:
    """Every renamed private entry point must have a wrapper stub.

    The baseline check cannot catch this: it compares the wrapper against a
    recorded list, so an entry point the wrapper never re-exported is simply
    absent from both sides and passes. Only the private library knows the full
    set of renamed entry points.
    """
    renamed = impl_symbols(private)
    # An empty set would pass every comparison below having compared nothing.
    # Guarded here, not in impl_symbols(), because check_no_impl() treats an
    # empty set as a pass.
    if not renamed:
        raise AbiError(
            f"no miopen*_impl symbols exported from {private.path} -- "
            "not a flag-on private library (wrong file, or the rename was "
            "not applied?)"
        )
    exported = wrapper.defined_dynamic_functions()
    # Report the public spelling: it is the name the missing stub would carry.
    missing = sorted(
        sym[: -len("_impl")] for sym in renamed if sym[: -len("_impl")] not in exported
    )
    if not missing:
        print(
            f"PASS: all {len(renamed)} renamed private entry points have a "
            "wrapper stub"
        )
        return True
    print(
        "FAIL: private library exports renamed entry points with no matching "
        "wrapper stub (unreachable through libMIOpen.so):"
    )
    for sym in missing:
        print(f"  {sym}")
    print(SUPERSET_REMEDY)
    return False


def check_installed_headers(include_dir: str, exempt: list[str]) -> bool:
    """No installed header may name an entry point by its private spelling.

    ``exempt`` holds directories, relative to ``include_dir``, that legitimately
    carry the private declarations. Exempting by directory, not file name, keeps
    a leak into a similarly-named header elsewhere from slipping through.

    Only header files are read: anything else staged here is not something a
    consumer compiles against, and counting it would overstate what the reported
    number covers.
    """
    root = Path(include_dir)
    if not root.is_dir():
        raise AbiError(f"installed include directory not found: {root}")

    exempt_roots = [root / rel for rel in exempt]
    public_root = root / PUBLIC_INCLUDE_DIR
    offenders: list[tuple[Path, int, str]] = []
    scanned = 0
    scanned_public = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in HEADER_SUFFIXES:
            continue
        if any(path.is_relative_to(d) for d in exempt_roots):
            continue
        scanned += 1
        if path.is_relative_to(public_root):
            scanned_public += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Match whole identifiers with the symbol checks' pattern, so a name
            # that merely contains '_impl' (ck_impl_interface) is not mistaken
            # for a renamed entry point.
            for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line):
                if IMPL_RE.match(word):
                    offenders.append((path, lineno, word))

    # Nothing scanned is a tree this never looked at, not a clean one, and
    # nothing else guards the staged include tree. Only MIOpen's own headers
    # count: a prefix shared with other packages would otherwise pass without
    # reading a single MIOpen header.
    if scanned_public == 0:
        raise AbiError(
            f"no headers found under {public_root} -- nothing was checked "
            "(wrong prefix, or an install staged without C++ headers?)"
        )

    if not offenders:
        print(f"PASS: no private rename spelling in {scanned} installed headers")
        return True
    print("FAIL: installed headers name entry points by their private spelling:")
    for path, lineno, word in offenders:
        print(f"  {path}:{lineno}: {word}")
    print(INSTALLED_HEADERS_REMEDY)
    return False


def check_wrapper_only_list(declared: set[str], listed: set[str]) -> bool:
    """The header and the list describe the same functions to two different gates.

    check-wrapper reads the list because a built tree has no source headers, so a
    function declared but not listed would fail that gate only once someone builds
    with the flag on.
    """
    if declared == listed:
        print(
            f"PASS: wrapper_only_symbols.txt lists exactly the {len(declared)}"
            " hipdnn_forwarding.h entry points"
        )
        return True
    print("FAIL: wrapper_only_symbols.txt does not match hipdnn_forwarding.h")
    for sym in sorted(declared - listed):
        print(f"  - declared in hipdnn_forwarding.h, not listed: {sym}")
    for sym in sorted(listed - declared):
        print(f"  + listed, not declared in hipdnn_forwarding.h: {sym}")
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
    """Assert every stub opens with a dispatch macro naming itself."""
    findings: list[str] = []
    for name, args in sorted(dispatches.items()):
        exempt_reason = DISPATCH_EXEMPT.get(name)
        if exempt_reason is not None:
            if args:
                findings.append(
                    f"  {name} carries a dispatch macro but is exempt"
                    f" ({exempt_reason}); drop the macro or the exemption"
                )
        elif not args:
            findings.append(
                f"  {name} has no MIOPEN_WRAPPER_DISPATCH or MIOPEN_WRAPPER_FORWARD"
            )
        elif len(args) > 1:
            findings.append(
                f"  {name} has {len(args)} dispatch macros"
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
            f"  {name} is exempt from the dispatch macros but has no stub;"
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


def check_wrapper_only_stubs(
    names: set[str], forwards: dict[str, set[str]], dispatches: dict[str, list[str]]
) -> bool:
    """A wrapper-only stub does the work itself.

    There is no _impl for it to call. A dispatch macro does not fit either: its
    MIOpen route expects to fall through to an _impl, and it clears the record of
    the last forwarded failure, which miopenGetLastForwardedError reads.
    """
    findings: list[str] = []
    for name in sorted(names - forwards.keys()):
        findings.append(f"  {name} has no stub in wrapper.cpp")
    for name in sorted(names & forwards.keys()):
        if forwards[name]:
            findings.append(
                f"  {name} calls {', '.join(sorted(forwards[name]))};"
                " a wrapper-only function has no _impl"
            )
        if dispatches[name]:
            findings.append(f"  {name} carries a dispatch macro")
    if not findings:
        print(
            f"PASS: all {len(names)} wrapper-only functions have a stub in wrapper.cpp"
            " that calls no _impl and carries no dispatch macro"
        )
        return True
    print("FAIL: wrapper-only stubs are missing or forward to the private library:")
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
    wrapper_only = read_symbol_list(args.wrapper_only, "wrapper-only symbols file")

    ok = check_soname(elf)

    # A stale exclusion entry (a symbol no longer in the baseline at all) would
    # silently mask real drift, so reject it before using the set.
    stray = sorted(excluded - baseline)
    if stray:
        print("FAIL: excluded-symbols file lists symbols absent from the baseline:")
        for sym in stray:
            print(f"  {sym}")
        ok = False
    shared = sorted(wrapper_only & baseline)
    if shared:
        print(
            "FAIL: wrapper-only symbols file lists symbols that are in the baseline."
            " A symbol the flag-off build also exports does not belong in this list:"
        )
        for sym in shared:
            print(f"  {sym}")
        ok = False

    ok &= check_symbols(
        elf,
        (baseline - excluded) | wrapper_only,
        "wrapper public API set == (baseline - excluded) + wrapper-only",
        BASELINE_REMEDY,
    )
    ok &= check_no_impl(elf, "wrapper")
    ok &= check_private_dep(elf, expect_present=True)
    ok &= check_no_backend_dep(elf)

    if args.needed_baseline:
        ok &= check_needed_baseline(elf, args.needed_baseline)

    if args.public_header:
        ok &= check_excluded_not_public(excluded, args.public_header)
        ok &= check_wrapper_only_not_public(wrapper_only, args.public_header)
    if args.private_lib:
        private = open_elf(args.private_lib, "private library")
        ok &= check_excluded_on_private(excluded, private)
        ok &= check_coversioned(elf, private)
        ok &= check_impl_superset(elf, private)

    print(f"wrapper public-abi symbol check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_check_installed_headers(args) -> int:
    exempt = args.exempt if args.exempt is not None else list(PRIVATE_INCLUDE_DIRS)
    ok = check_installed_headers(args.include_dir, exempt)
    print(f"installed-header rename check: {'PASS' if ok else 'FAIL'}")
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
    wrapper_source = resolve(
        args.wrapper, f"{miopen_rel}/src/private/wrapper.cpp", "wrapper.cpp"
    )
    wrapper_header_source = resolve(
        args.wrapper_header,
        f"{miopen_rel}/include_wrapper/miopen/hipdnn_forwarding.h",
        "hipdnn_forwarding.h",
    )
    wrapper_only_source = resolve(
        args.wrapper_only,
        f"{miopen_rel}/test/public_abi/wrapper_only_symbols.txt",
        "wrapper_only_symbols.txt",
    )
    range_defs_source = resolve(
        args.range_definitions,
        f"{miopen_rel}/src/convolution_api.cpp",
        "convolution_api.cpp",
    )
    provider_api_source = resolve(
        args.provider_api, PROVIDER_API_RELPATH, "MiopenApi.hpp"
    )

    # Without MIOpen's own sources there is nothing to compare the provider
    # against, so this is a skip, not a failure. Reaching here means git
    # confirmed the commit does not carry the file; one it carries but cannot
    # produce has already failed hard above. A missing file is not drift: drift
    # is a content mismatch between files the commit carries.
    miopen_sources = {
        f"{miopen_rel}/src/private/wrapper.cpp": wrapper_source,
        f"{miopen_rel}/src/convolution_api.cpp": range_defs_source,
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

    wrapper, forwards, dispatches = parse_wrapper(wrapper_source)

    # Not part of the skip above: either file can be removed on its own (the
    # function moves into miopen.h, or the list empties), and skipping then would
    # switch off every other check. An untracked one means no wrapper-only
    # functions, so such a stub is judged as a forwarding stub here; the generator
    # refuses a stub that neither header declares.
    for rel, source in (
        (
            f"{miopen_rel}/include_wrapper/miopen/hipdnn_forwarding.h",
            wrapper_header_source,
        ),
        (f"{miopen_rel}/test/public_abi/wrapper_only_symbols.txt", wrapper_only_source),
    ):
        if source is None:
            print(
                f"NOTE: {rel} is not part of this tree ({reasons.get(rel, 'unreadable')});"
                " treating it as empty"
            )
    wrapper_only = (
        {
            entry.name: stub_signature(entry)
            for entry in parse_public_header(
                wrapper_header_source, "hipdnn_forwarding.h", allow_empty=True
            )
        }
        if wrapper_header_source is not None
        else {}
    )
    wrapper_only_listed = (
        parse_symbol_list(wrapper_only_source)
        if wrapper_only_source is not None
        else set()
    )

    wrapper_only_ok = check_wrapper_only_list(set(wrapper_only), wrapper_only_listed)
    wrapper_only_ok &= check_prototypes(
        wrapper_only, wrapper, "wrapper.cpp", "hipdnn_forwarding.h"
    )
    wrapper_only_ok &= check_wrapper_only_stubs(set(wrapper_only), forwards, dispatches)
    forwards = {n: c for n, c in forwards.items() if n not in wrapper_only}
    dispatches = {n: d for n, d in dispatches.items() if n not in wrapper_only}

    ok = check_wrapper_forwards(forwards)
    # Kept out of ``ok`` until the remedies are printed: a routing defect has its
    # own fix.
    dispatch_ok = check_wrapper_dispatch(dispatches)

    # The range entry points are exported but undeclared in miopen.h, so their
    # reference is the definition rather than the header.
    definitions = parse_range_prototypes(range_defs_source, "convolution_api.cpp")
    range_ok = check_range_entry_point_set(definitions, "convolution_api.cpp")
    provider_missing = False
    if provider_api_source is None:
        provider_missing = args.require_provider
        print(
            f"{'FAIL' if provider_missing else 'SKIP'}:"
            f" {PROVIDER_API_RELPATH} is not part of this tree"
            f" ({reasons.get(PROVIDER_API_RELPATH, 'unreadable')})"
        )
    else:
        declared = parse_range_prototypes(provider_api_source, "MiopenApi.hpp")
        range_ok &= check_range_entry_point_set(declared, "MiopenApi.hpp")
        range_ok &= check_prototypes(
            definitions, declared, "MiopenApi.hpp", "convolution_api.cpp"
        )

    if not wrapper_only_ok:
        print(WRAPPER_ONLY_REMEDY)
    if not dispatch_ok:
        print(DISPATCH_REMEDY)
    if not range_ok:
        print(RANGE_REMEDY)
    if provider_missing:
        print(PROVIDER_REMEDY)
    ok &= wrapper_only_ok
    ok &= dispatch_ok
    ok &= range_ok
    # A skipped file is not a checked one. Folding this in last keeps it out of
    # the remedies above, which are about mismatches rather than readability.
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
        help="check the hand-written wrapper stubs, the wrapper-only functions "
        "and the provider's range declarations (no build needed)",
    )
    p.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="MIOpen source root; the checked files are located under it by "
        "convention, read from HEAD when the checkout is sparse, and the check "
        "skipped only when git confirms the commit tracks none of them "
        "(default: the tree containing this script)",
    )
    p.add_argument("--wrapper", help="override path to src/private/wrapper.cpp")
    p.add_argument(
        "--wrapper-header",
        help="override path to include_wrapper/miopen/hipdnn_forwarding.h",
    )
    p.add_argument(
        "--wrapper-only",
        help="override path to test/public_abi/wrapper_only_symbols.txt",
    )
    p.add_argument(
        "--provider-api",
        help="override path to the hipDNN provider's MiopenApi.hpp, which must "
        "then exist; by default it is located relative to the repository root, "
        "read from HEAD when the checkout is sparse, and skipped only when git "
        "confirms the commit does not track it",
    )
    p.add_argument(
        "--range-definitions",
        help="override path to src/convolution_api.cpp, which defines the "
        "miopenConvolution*GetWorkSpaceSizeRange entry points",
    )
    p.add_argument(
        "--require-provider",
        action="store_true",
        help="fail instead of skipping when the provider's MiopenApi.hpp cannot "
        "be read, so that a tree which does not include it cannot report a pass "
        "for a file it never opened; pass this wherever the provider is expected "
        "to be present, such as a monorepo CI lane",
    )
    p.set_defaults(func=cmd_check_headers)

    p = sub.add_parser("check-wrapper", help="gate a flag-on wrapper build")
    p.add_argument("lib", help="path to the built wrapper libMIOpen.so")
    p.add_argument("--baseline", required=True, help="committed public symbol baseline")
    p.add_argument("--excluded", required=True, help="committed excluded symbol list")
    p.add_argument(
        "--wrapper-only",
        required=True,
        help="committed list of the symbols only the wrapper exports, which "
        "the baseline does not list",
    )
    p.add_argument(
        "--private-lib",
        help="path to libMIOpen_private.so; when given, every excluded symbol "
        "must still be exported from it under its original name, the two "
        "libraries must be co-versioned, and every renamed entry point it "
        "exports must have a wrapper stub",
    )
    p.add_argument(
        "--public-header",
        help="path to include/miopen/miopen.h; when given, no excluded or "
        "wrapper-only symbol may be declared there",
    )
    p.add_argument(
        "--needed-baseline",
        help="optional committed DT_NEEDED baseline; when given, the full "
        "runtime dependency list must match it exactly. The private-library and "
        "hipDNN-backend entries are checked either way",
    )
    p.set_defaults(func=cmd_check_wrapper)

    p = sub.add_parser(
        "check-installed-headers",
        help="gate the staged include tree of a flag-on build",
    )
    p.add_argument(
        "include_dir", help="staged include directory, e.g. <install-prefix>/include"
    )
    p.add_argument(
        "--exempt",
        action="append",
        default=None,
        help="directory, relative to include_dir, that legitimately carries the "
        f"private declarations (default: {' '.join(PRIVATE_INCLUDE_DIRS)}); "
        "repeatable",
    )
    p.set_defaults(func=cmd_check_installed_headers)

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
