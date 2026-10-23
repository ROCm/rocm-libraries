#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate the MIOpen wrapper's derived files from include/miopen/miopen.h.

include/miopen/miopen.h is the only file where a public entry point is written
by hand. Everything below copies its list of entry points, so one run of this
script writes all of them:

  src/private/miopen_private_rename.h   #define miopenFoo miopenFoo_impl
  src/private/miopen_impl.h             extern "C" <ret> miopenFoo_impl(<params>);
  src/private/wrapper_passthrough.cpp   a pass-through stub for every entry point
                                        that src/private/wrapper.cpp does not
                                        define itself
  <repo>/dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp
                                        the provider's copy of the rename list

The files are committed, not generated at build time, so an ABI-visible change
shows up in the PR diff. ``--check`` is the gate: it regenerates in memory and
fails on any difference from what is committed.

    generate_wrapper_sources.py            rewrite the four files
    generate_wrapper_sources.py --check    exit 1 and print a diff on drift

Exit codes: 0 in sync (or written), 1 drift (--check only), 2 bad input, a
parse error, or the wrong clang-format.

To give an entry point custom behaviour, move its stub from
wrapper_passthrough.cpp into wrapper.cpp, edit it, and rerun this script; it
skips every entry point wrapper.cpp already defines.

The output is formatted with clang-format, so the committed text depends on its
version. The version is pinned in .pre-commit-config.yaml and any other is
refused: different versions wrap lines differently, so the gate could pass on
one machine and fail on another.

Python standard library only, plus the sibling check_public_abi.py, whose
parsing helpers this reuses rather than keeping a second copy.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from check_public_abi import (
    TYPE_TAIL_KEYWORDS,
    AbiError,
    drop_top_level_const,
    parse_wrapper,
    read_tracked_source,
    split_params,
    squash,
)

# The value each macro has while miopen.h is read, as the library's build sees
# it. Conditionals are evaluated against this table rather than flattened, so an
# #else branch cannot contribute declarations. MIOPEN_BETA_API is defined for the
# whole build in CMakeLists.txt. The include guard reads as undefined because
# this is the first time the header is seen. __cplusplus and __clang__ only
# guard text that declares no entry point.
KNOWN_MACROS = {
    "MIOPEN_BETA_API": True,
    "__cplusplus": True,
    "__clang__": True,
    "MIOPEN_GUARD_MIOPEN_H_": False,
}

STATUS_TYPE = "miopenStatus_t"

PUBLIC_HEADER_RELPATH = "include/miopen/miopen.h"
WRAPPER_RELPATH = "src/private/wrapper.cpp"
RENAME_RELPATH = "src/private/miopen_private_rename.h"
IMPL_RELPATH = "src/private/miopen_impl.h"
PASSTHROUGH_RELPATH = "src/private/wrapper_passthrough.cpp"
MIOPEN_CLANG_FORMAT_RELPATH = ".clang-format"

# Relative to the repository root.
PROVIDER_RENAME_RELPATH = "dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp"
PROVIDER_CLANG_FORMAT_RELPATH = "dnn-providers/miopen-provider/.clang-format"
PRE_COMMIT_CONFIG_RELPATH = ".pre-commit-config.yaml"

# Where cmake/ClangCheck.cmake looks, and where the MIOpen dev image installs
# the pinned version.
FALLBACK_CLANG_FORMAT = "/usr/local/bin/clang-format"

GENERATOR_COMMAND = "python3 projects/miopen/script/generate_wrapper_sources.py"
REGENERATE_HOOK_COMMAND = (
    "pre-commit run --hook-stage manual miopen-wrapper-sources-regenerate --all-files"
)

MAX_DIFF_LINES = 50


# --------------------------------------------------------------------------
# Preambles: each file's own copyright and SPDX lines, the shared generated-file
# banner, then that file's explanation.
# --------------------------------------------------------------------------

MIOPEN_COPYRIGHT = """\
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
"""

# The provider's file spells these two lines differently; kept as is.
PROVIDER_COPYRIGHT = """\
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
"""

BANNER = f"""\
//
// GENERATED FILE. Do not edit.
// Source: projects/miopen/include/miopen/miopen.h
// Regenerate: {GENERATOR_COMMAND}
//
"""

RENAME_EXPLANATION = """\
// Symbol-rename header for the MIOpen public/private library split. When
// MIOPEN_ENABLE_HIPDNN_WRAPPER is on, it is force-included (-include) into every
// translation unit of the private library (MIOpen_private). Each #define renames
// a public C entry point to its _impl form, which frees the public names in
// libMIOpen.so for the wrapper (wrapper.cpp, wrapper_passthrough.cpp) to define
// as forwarding stubs.
//
// It holds only #defines, so it is safe to force-include into every private
// source (C++, HIP, and assembled kernels), and the rest of the MIOpen source
// stays untouched. It is not installed: the hipDNN provider force-includes its
// own generated copy of this list
// (dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp).
"""

IMPL_EXPLANATION = """\
// Declarations of the _impl-form public C entry points that the private library
// (libMIOpen_private.so) exports when the public/private split is enabled. Its
// definitions are renamed to _impl at build time by force-including
// miopen_private_rename.h; this is the matching declaration set for callers.
//
// It is the single declaration source for the wrapper's stubs (wrapper.cpp,
// wrapper_passthrough.cpp) and for the hipDNN routing code that falls back to
// MIOpen (hipdnn_conv.cpp, hipdnn_graph.cpp). The _impl names have C linkage, so
// a signature that differed from the definition would link cleanly and corrupt
// arguments at run time. Each declaration is generated from the miopen.h
// declaration the definition is compiled from, so the two cannot differ.
//
// This header is not installed, and the hipDNN provider does not use it; it
// force-includes its own copy of the rename list instead.
"""

PASSTHROUGH_EXPLANATION = """\
// Pass-through stubs for the public wrapper of the MIOpen public/private split,
// compiled into libMIOpen.so next to wrapper.cpp. Each public C entry point in
// <miopen/miopen.h> has an extern "C" stub, either here or in wrapper.cpp. Every
// stub opens with MIOPEN_WRAPPER_DISPATCH (routing.hpp), which forwards the call
// to hipDNN when MIOPEN_HIPDNN_FORWARDING and the compile-time forwarding set
// say so; otherwise the stub calls the matching _impl symbol in the private
// library (declared in miopen_impl.h).
//
// A stub that needs more than a pass-through belongs in wrapper.cpp. The
// generator skips every entry point wrapper.cpp defines, so move the stub there,
// edit it, and rerun the generator.
//
// This file is compiled without the rename header, so it sees the public names
// from <miopen/miopen.h>.
"""

PASSTHROUGH_GUARDS = """\
// If the rename header ever reached this file (say, because -include or
// MIOPEN_BUILDING_PRIVATE were widened from PRIVATE to PUBLIC on the private
// target), every stub below would silently become
// `miopenFoo_impl(...) { return miopenFoo_impl(...); }`. That links cleanly,
// drops the public symbol, and recurses forever at runtime, so fail at compile
// time instead.
#ifdef MIOPEN_BUILDING_PRIVATE
#error "wrapper_passthrough.cpp must not be compiled as part of the private library"
#endif

#include <miopen/miopen.h>

#include "hipdnn_graph.hpp"
#include "miopen_impl.h"
#include "routing.hpp"

#ifdef miopenCreate
#error "miopen_private_rename.h leaked into the public wrapper"
#endif
"""

PROVIDER_EXPLANATION = """\
// Symbol-rename header for the hipDNN MIOpen provider, force-included (-include,
// see CMakeLists.txt) into every provider and provider-test translation unit
// only when the provider links MIOpen's private library
// (MIOPEN_ENABLE_HIPDNN_WRAPPER=ON in the MIOpen build, detected as
// TARGET MIOpen_private).
//
// Each #define renames a public MIOpen C entry point to its _impl form, so the
// provider, whose sources call the public names, binds the private library's
// entry points directly instead of going back through the public wrapper. With
// the split off, this header is not force-included and the provider calls the
// public C API exactly as before.
//
// It is generated from the same list as the library's own
// src/private/miopen_private_rename.h, so the two cannot differ: it renames
// exactly the entry points declared in <miopen/miopen.h>. It holds only
// #defines, so it is safe to force-include everywhere.
//
// The three miopenConvolution*GetWorkSpaceSizeRange entry points are deliberately
// not renamed. They are exported with MIOPEN_EXPORT from src/convolution_api.cpp
// but never declared in <miopen/miopen.h>, so the split leaves them on
// libMIOpen_private.so under their original names. The provider declares them in
// MiopenApi.hpp and binds them directly.
"""


# --------------------------------------------------------------------------
# Reading miopen.h
# --------------------------------------------------------------------------

_COMMENT_OR_LITERAL_RE = re.compile(
    r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S
)
_DIRECTIVE_RE = re.compile(r"^\s*#\s*(?P<kw>\w*)\s*(?P<rest>.*)$")
_IDENT_TAIL_RE = re.compile(r"([A-Za-z_]\w*)\s*$")
_DECL_RE = re.compile(
    r"(?P<ret>[^()]*?)\s*(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*)\)", re.S
)


def strip_comments_keep_lines(text: str, what: str) -> str:
    """Blank out comments without moving any line, so errors can name one.

    Strings are matched too, so a ``//`` inside one is not taken for a comment.
    """

    def blank(match: re.Match) -> str:
        found = match.group(0)
        if found[0] in "\"'":
            return found
        return " " + "\n" * found.count("\n")

    stripped = _COMMENT_OR_LITERAL_RE.sub(blank, text)
    start = stripped.find("/*")
    if start >= 0:
        raise AbiError(
            f"{what}:{stripped.count(chr(10), 0, start) + 1}: unterminated /* comment"
        )
    return stripped


def active_text(text: str, what: str) -> str:
    """Return ``text`` with comments, directives and inactive branches blanked.

    Only #ifdef, #ifndef, #else and #endif are understood. Anything that would
    need a real preprocessor is refused rather than guessed at, because a
    declaration dropped by a wrong guess is an entry point that silently has no
    stub.
    """
    lines = strip_comments_keep_lines(text, what).split("\n")
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
                " generate_wrapper_sources.py"
            )
        if keyword in ("ifdef", "ifndef"):
            macro = rest.split()[0] if rest else ""
            if macro not in KNOWN_MACROS:
                raise AbiError(
                    f"{what}:{number}: #{keyword} {macro or '(nothing)'}: this macro is"
                    " not in KNOWN_MACROS in generate_wrapper_sources.py, so it is not"
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

        A top-level const is dropped: it is not part of the function's type, and
        clang-tidy's readability-avoid-const-params-in-decls rejects it in a
        declaration. A default argument stays, as the public header has it.
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
    # A trailing identifier that is a type word, or spelled like a typedef, is
    # the end of an unnamed parameter's type rather than a name. The call into
    # the _impl function needs every name, so an unnamed one cannot be generated.
    if not type_text or name in TYPE_TAIL_KEYWORDS or name.endswith("_t"):
        raise AbiError(
            f"{where}: {function}: parameter {declared!r} has no name; name every"
            " parameter in miopen.h"
        )
    return Param(
        type=squash(type_text),
        name=name,
        default=default.strip() if equals else None,
    )


def parse_public_header(source: str, what: str = "miopen.h") -> list[EntryPoint]:
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
        parsed = _DECL_RE.fullmatch(declaration)
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
    if not entries:
        raise AbiError(f"no MIOPEN_EXPORT declarations found in {what}")
    return list(entries.values())


def hand_written_stubs(wrapper_source: str, entries: list[EntryPoint]) -> set[str]:
    """The entry points wrapper.cpp defines itself, which are not generated."""
    protos, _forwards, _dispatches = parse_wrapper(wrapper_source)
    # parse_wrapper cannot see a body that contains a nested block. Leaving such
    # a stub out would generate a second definition of it, so count the
    # definitions independently and refuse when the two disagree.
    stripped = strip_comments_keep_lines(wrapper_source, "wrapper.cpp")
    defined = len(re.findall(r'\bextern\s+"C"(?!\s*\{)', stripped))
    if defined != len(protos):
        raise AbiError(
            f'wrapper.cpp has {defined} extern "C" definitions but only {len(protos)}'
            " could be read. A stub body with a nested { } block is not supported;"
            " move that logic into a helper function"
        )
    declared = {entry.name for entry in entries}
    unknown = sorted(set(protos) - declared)
    if unknown:
        raise AbiError(
            "wrapper.cpp defines entry points that miopen.h does not declare: "
            + ", ".join(unknown)
        )
    return set(protos)


def check_dispatchable(entries: list[EntryPoint], hand_written: set[str]) -> None:
    for entry in entries:
        if entry.ret != STATUS_TYPE and entry.name not in hand_written:
            raise AbiError(
                f"miopen.h:{entry.line}: {entry.name} returns {entry.ret!r}, but a"
                f" generated stub's MIOPEN_WRAPPER_DISPATCH only works for {STATUS_TYPE}."
                f" Write its stub by hand in {WRAPPER_RELPATH}"
            )


# --------------------------------------------------------------------------
# Building the files. Unformatted text only: clang-format decides the layout.
# --------------------------------------------------------------------------


def render_rename_defines(entries: list[EntryPoint]) -> str:
    return "".join(f"#define {e.name} {e.name}_impl\n" for e in entries)


def render_rename_header(entries: list[EntryPoint]) -> str:
    return (
        MIOPEN_COPYRIGHT
        + BANNER
        + RENAME_EXPLANATION
        + "#pragma once\n\n"
        + "#ifndef MIOPEN_BUILDING_PRIVATE\n"
        + '#error "miopen_private_rename.h included outside of MIOpen Private build"\n'
        + "#endif\n\n"
        + render_rename_defines(entries)
    )


def render_provider_rename(entries: list[EntryPoint]) -> str:
    return (
        PROVIDER_COPYRIGHT
        + BANNER
        + PROVIDER_EXPLANATION
        + "#pragma once\n\n"
        + render_rename_defines(entries)
    )


def render_impl_declaration(entry: EntryPoint) -> str:
    params = ", ".join(p.declaration_text() for p in entry.params) or "void"
    return f'extern "C" {entry.ret} {entry.name}_impl({params});\n'


def render_impl_header(entries: list[EntryPoint]) -> str:
    return (
        MIOPEN_COPYRIGHT
        + BANNER
        + IMPL_EXPLANATION
        + "#pragma once\n\n"
        + "#ifdef MIOPEN_BUILDING_PRIVATE\n"
        + "#error \\\n"
        + '    "miopen_impl.h declares the _impl entry points for callers of the private'
        + ' library, not for the private library itself"\n'
        + "#endif\n\n"
        + "#include <miopen/miopen.h>\n\n"
        + "".join(render_impl_declaration(e) for e in entries)
    )


def render_stub(entry: EntryPoint) -> str:
    params = ", ".join(p.definition_text() for p in entry.params) or "void"
    args = ", ".join(p.name for p in entry.params)
    return (
        f'extern "C" {entry.ret} {entry.name}({params})\n'
        "{\n"
        f"    MIOPEN_WRAPPER_DISPATCH({entry.name});\n"
        f"    return {entry.name}_impl({args});\n"
        "}\n"
    )


def render_passthrough(entries: list[EntryPoint], hand_written: set[str]) -> str:
    stubs = [render_stub(e) for e in entries if e.name not in hand_written]
    return (
        MIOPEN_COPYRIGHT
        + BANNER
        + PASSTHROUGH_EXPLANATION
        + "\n"
        + PASSTHROUGH_GUARDS
        + "\n"
        + "\n".join(stubs)
    )


# --------------------------------------------------------------------------
# clang-format
# --------------------------------------------------------------------------

_VERSION_RE = re.compile(r"clang-format version (\d+\.\d+\.\d+)")
_PIN_RE = re.compile(r"mirrors-clang-format\s*\n\s*rev:\s*v?(\d+\.\d+\.\d+)")


def pinned_clang_format_version(pre_commit_config: str) -> str:
    match = _PIN_RE.search(pre_commit_config)
    if match is None:
        raise AbiError(
            f"no mirrors-clang-format 'rev:' line found in {PRE_COMMIT_CONFIG_RELPATH}"
        )
    return match.group(1)


def clang_format_version(binary: str) -> str | None:
    try:
        done = subprocess.run(
            [binary, "--version"], capture_output=True, check=False, text=True
        )
    except OSError:
        return None
    match = _VERSION_RE.search(done.stdout)
    return match.group(1) if match else None


def wrong_clang_format_message(pinned: str, found: list[str]) -> str:
    seen = "; ".join(found) if found else "none found"
    return (
        f"clang-format {pinned} is required (the version pinned in"
        f" {PRE_COMMIT_CONFIG_RELPATH}); other versions wrap lines differently, so"
        f" the generated files would not match what CI produces. Found: {seen}.\n"
        f"Get it with `pip install clang-format=={pinned}` and pass --clang-format"
        f" or set CLANG_FORMAT, or run the manual pre-commit hook, which installs"
        f" it for you:\n  {REGENERATE_HOOK_COMMAND}"
    )


def find_clang_format(explicit: str | None, pinned: str) -> str:
    requested = explicit or os.environ.get("CLANG_FORMAT")
    if requested:
        version = clang_format_version(requested)
        if version != pinned:
            found = [f"{requested}: {version or 'not runnable'}"]
            raise AbiError(wrong_clang_format_message(pinned, found))
        return requested

    found = []
    for candidate in (shutil.which("clang-format"), FALLBACK_CLANG_FORMAT):
        if not candidate or not Path(candidate).is_file():
            continue
        version = clang_format_version(candidate)
        if version == pinned:
            return candidate
        found.append(f"{candidate}: {version or 'unknown version'}")
    raise AbiError(wrong_clang_format_message(pinned, found))


def format_text(text: str, style: str, assume_filename: str, binary: str) -> str:
    """Format ``text`` as if it were the file ``assume_filename``.

    The style is passed as text rather than discovered from the file system:
    in a sparse checkout the directory that holds the output, and so its
    .clang-format, may not be on disk, and clang-format would then quietly fall
    back to its default style and report drift that is not there.
    """
    with tempfile.TemporaryDirectory() as tmp:
        style_path = Path(tmp) / ".clang-format"
        style_path.write_text(style, encoding="utf-8")
        done = subprocess.run(
            [
                binary,
                f"--style=file:{style_path}",
                f"--assume-filename={assume_filename}",
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            check=False,
        )
    if done.returncode != 0:
        raise AbiError(
            f"clang-format failed on {assume_filename}: "
            + done.stderr.decode("utf-8", "replace").strip()
        )
    return done.stdout.decode("utf-8")


# --------------------------------------------------------------------------
# The four outputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Output:
    relpath: str
    style_relpath: str
    # The provider lives in a sibling project that a MIOpen-only tree does not ship.
    optional: bool = False


def outputs_for(miopen_rel: str) -> dict[str, Output]:
    style = f"{miopen_rel}/{MIOPEN_CLANG_FORMAT_RELPATH}"
    return {
        "rename": Output(f"{miopen_rel}/{RENAME_RELPATH}", style),
        "impl": Output(f"{miopen_rel}/{IMPL_RELPATH}", style),
        "passthrough": Output(f"{miopen_rel}/{PASSTHROUGH_RELPATH}", style),
        "provider": Output(
            PROVIDER_RENAME_RELPATH, PROVIDER_CLANG_FORMAT_RELPATH, optional=True
        ),
    }


def build_texts(entries: list[EntryPoint], hand_written: set[str]) -> dict[str, str]:
    """The unformatted text of each output, keyed as in outputs_for()."""
    return {
        "rename": render_rename_header(entries),
        "impl": render_impl_header(entries),
        "passthrough": render_passthrough(entries, hand_written),
        "provider": render_provider_rename(entries),
    }


def unified_diff(committed: str, generated: str, relpath: str) -> str:
    lines = list(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=f"{relpath} (committed)",
            tofile=f"{relpath} (generated)",
        )
    )
    shown = lines[:MAX_DIFF_LINES]
    text = "".join(line if line.endswith("\n") else line + "\n" for line in shown)
    if len(lines) > len(shown):
        text += f"... {len(lines) - len(shown)} more diff lines not shown\n"
    return text


def run(args: argparse.Namespace) -> int:
    root = Path(args.source_root).resolve()
    repo_root = root.parent.parent
    miopen_rel = root.relative_to(repo_root).as_posix()
    outputs = outputs_for(miopen_rel)
    reasons: dict[str, str] = {}

    def read(rel: str, may_be_new: bool = False) -> str | None:
        """Read one file. ``may_be_new`` is for an output the first run creates."""
        path = repo_root / rel
        if args.check:
            return read_tracked_source(path, repo_root, reasons)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        if may_be_new and path.parent.is_dir():
            return None
        raise AbiError(
            f"{rel} is not on disk. Writing regenerates every file, so the whole"
            " tree must be checked out; in a sparse checkout, add the missing"
            " directory to the path list"
        )

    public_source = read(f"{miopen_rel}/{PUBLIC_HEADER_RELPATH}")
    wrapper_source = read(f"{miopen_rel}/{WRAPPER_RELPATH}")
    if public_source is None or wrapper_source is None:
        missing = [
            rel
            for rel, source in (
                (f"{miopen_rel}/{PUBLIC_HEADER_RELPATH}", public_source),
                (f"{miopen_rel}/{WRAPPER_RELPATH}", wrapper_source),
            )
            if source is None
        ]
        print("SKIP: MIOpen's own sources could not be read; nothing to generate from")
        for rel in missing:
            print(f"  {rel}: {reasons.get(rel, 'unreadable')}")
        return 0

    entries = parse_public_header(public_source)
    hand_written = hand_written_stubs(wrapper_source, entries)
    check_dispatchable(entries, hand_written)
    texts = build_texts(entries, hand_written)

    pre_commit_config = read(PRE_COMMIT_CONFIG_RELPATH)
    if pre_commit_config is None:
        raise AbiError(
            f"{PRE_COMMIT_CONFIG_RELPATH} is needed to learn which clang-format"
            f" version to use: {reasons.get(PRE_COMMIT_CONFIG_RELPATH, 'unreadable')}"
        )
    binary = find_clang_format(
        args.clang_format, pinned_clang_format_version(pre_commit_config)
    )

    generated: dict[str, str] = {}
    committed: dict[str, str | None] = {}
    skipped: list[str] = []
    for key, output in outputs.items():
        current = read(output.relpath, may_be_new=True)
        if current is None and output.optional and args.check:
            if args.require_provider:
                raise AbiError(
                    f"--require-provider was passed, but {output.relpath} is not part"
                    f" of this tree ({reasons.get(output.relpath, 'unreadable')}). Make"
                    " dnn-providers/miopen-provider part of the tree being checked, for"
                    " instance by adding it to the sparse checkout's path list. Drop"
                    " --require-provider only for a tree that genuinely has no provider."
                )
            skipped.append(output.relpath)
            continue
        style = read(output.style_relpath)
        if style is None:
            raise AbiError(
                f"{output.style_relpath} is not available, so {output.relpath} cannot"
                f" be formatted: {reasons.get(output.style_relpath, 'unreadable')}"
            )
        generated[key] = format_text(texts[key], style, output.relpath, binary)
        committed[key] = current

    if not args.check:
        for key, text in generated.items():
            path = repo_root / outputs[key].relpath
            if committed[key] == text:
                print(f"unchanged: {outputs[key].relpath}")
            else:
                path.write_bytes(text.encode("utf-8"))
                print(f"wrote: {outputs[key].relpath}")
        return 0

    drifted = False
    for key, text in generated.items():
        relpath = outputs[key].relpath
        current = committed[key]
        if current == text:
            print(f"PASS: {relpath} is up to date")
            continue
        drifted = True
        if current is None:
            print(f"FAIL: {relpath} does not exist")
        else:
            print(f"FAIL: {relpath} differs from what miopen.h generates:")
            print(unified_diff(current, text, relpath), end="")
    for relpath in skipped:
        print(
            f"SKIP: {relpath} is not part of this tree"
            f" ({reasons.get(relpath, 'unreadable')})"
        )
    if drifted:
        print(
            "\nThese files are generated from include/miopen/miopen.h and the stubs"
            " in src/private/wrapper.cpp. Do not edit them by hand. Regenerate and"
            f" commit the result:\n  {GENERATOR_COMMAND}\nor, without installing the"
            f" pinned clang-format yourself:\n  {REGENERATE_HOOK_COMMAND}"
        )
        return 1
    print("wrapper source generation check: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_wrapper_sources.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 and print a diff if a committed file differs",
    )
    parser.add_argument(
        "--require-provider",
        action="store_true",
        help="fail instead of skipping when the provider's rename header is not part "
        "of the tree, so a tree without it cannot report a pass for a file it never "
        "opened",
    )
    parser.add_argument(
        "--clang-format",
        help="clang-format binary to use; it must be the pinned version (default: "
        "$CLANG_FORMAT, else the first of clang-format on PATH and "
        f"{FALLBACK_CLANG_FORMAT} with the pinned version)",
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="MIOpen source root (default: the tree containing this script)",
    )
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except AbiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
