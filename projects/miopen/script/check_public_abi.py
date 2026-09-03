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

``check-headers`` -- a source-only cross-check of the four hand-maintained
artifacts of the split. It needs no build, no GPU and no flag-on configuration,
so unlike the two gates above it can run in a lint lane on every PR:

  1. include/miopen/miopen.h        -- the public contract (MIOPEN_EXPORT decls)
  2. src/private/miopen_impl.h      -- the matching _impl declarations
  3. src/private/miopen_private_rename.h -- the compile-time rename
  4. src/private/wrapper.cpp        -- the forwarding stubs

Every public entry point must appear in all four, the wrapper stub's signature
must match miopen.h, the _impl declaration's signature must match it too (modulo
the suffix), and each stub must forward to its own _impl symbol and nothing
else. No artifact may carry an entry the public header does not.

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
underlying type read as a mismatch. That is the intended bias -- the four files
are meant to be copies of one another.

``compare-pair`` diffs two ``dump`` outputs and reports a content hash for
information only; a content hash is build-path dependent and is never gated.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
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
Every public entry point must be spelled identically in all four places:
  include/miopen/miopen.h              MIOPEN_EXPORT <ret> miopenFoo(<params>);
  src/private/miopen_impl.h            extern "C" <ret> miopenFoo_impl(<params>);
  src/private/miopen_private_rename.h  #define miopenFoo miopenFoo_impl
  src/private/wrapper.cpp              extern "C" <ret> miopenFoo(<params>)
                                       { return miopenFoo_impl(<args>); }
Update the three private files to match the public header. Do not edit
miopen.h to match them -- that changes the public C API.""".rstrip()


# --------------------------------------------------------------------------
# Minimal C declaration parser.
#
# Parsing the four files textually rather than invoking a compiler keeps this
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
    path: Path, anchor: re.Pattern[str], what: str
) -> dict[str, tuple[str, ...]]:
    """Collect every miopen* prototype introduced by ``anchor``, keyed by name."""
    text = strip_comments(read_source(path, what))
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


def parse_wrapper(path: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, set[str]]]:
    """Collect the wrapper's stub prototypes and the _impl symbols each one calls."""
    text = strip_comments(read_source(path, "wrapper"))
    protos: dict[str, tuple[str, ...]] = {}
    forwards: dict[str, set[str]] = {}
    for match in WRAPPER_DEF_RE.finditer(text):
        name, sig = parse_prototype(match.group("decl"), "wrapper")
        if name in protos:
            raise AbiError(f"duplicate definition of {name} in wrapper")
        protos[name] = sig
        forwards[name] = set(IMPL_CALL_RE.findall(match.group("body")))
    return protos, forwards


def parse_renames(path: Path) -> dict[str, str]:
    """Collect the rename header's #defines, folding backslash continuations."""
    text = strip_comments(read_source(path, "rename header").replace("\\\n", " "))
    out: dict[str, str] = {}
    for name, target in RENAME_RE.findall(text):
        if name in out:
            raise AbiError(f"duplicate #define of {name} in rename header")
        out[name] = target
    return out


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
    public: dict[str, tuple[str, ...]], other: dict[str, tuple[str, ...]], what: str
) -> bool:
    drifted = [n for n in sorted(public.keys() & other.keys()) if public[n] != other[n]]
    if not drifted:
        print(f"PASS: {what} prototypes match miopen.h")
        return True
    print(f"FAIL: {what} prototypes have drifted from miopen.h")
    for name in drifted:
        print(f"  {name}")
        print(f"      miopen.h: {render_signature(public[name])}")
        print(f"      {what}: {render_signature(other[name])}")
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
    root = Path(args.source_root)
    public_path = Path(args.public_header or root / "include/miopen/miopen.h")
    impl_path = Path(args.impl_header or root / "src/private/miopen_impl.h")
    rename_path = Path(
        args.rename_header or root / "src/private/miopen_private_rename.h"
    )
    wrapper_path = Path(args.wrapper or root / "src/private/wrapper.cpp")

    public = parse_declarations(public_path, EXPORT_ANCHOR_RE, "miopen.h")
    if not public:
        raise AbiError(f"no MIOPEN_EXPORT declarations found in {public_path}")
    impl = parse_declarations(impl_path, EXTERN_C_ANCHOR_RE, "miopen_impl.h")
    renames = parse_renames(rename_path)
    wrapper, forwards = parse_wrapper(wrapper_path)

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

    if not ok:
        print(HEADERS_REMEDY)
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
        help="cross-check the four hand-maintained split sources (no build needed)",
    )
    p.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="MIOpen source root; the four files are located under it by "
        "convention (default: the tree containing this script)",
    )
    p.add_argument("--public-header", help="override path to include/miopen/miopen.h")
    p.add_argument("--impl-header", help="override path to src/private/miopen_impl.h")
    p.add_argument(
        "--rename-header", help="override path to src/private/miopen_private_rename.h"
    )
    p.add_argument("--wrapper", help="override path to src/private/wrapper.cpp")
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
