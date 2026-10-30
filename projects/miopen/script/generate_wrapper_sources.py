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
parse error, no usable clang-format, or a refused write (see below).

To give an entry point custom behaviour, move its stub from
wrapper_passthrough.cpp into wrapper.cpp, edit it, and rerun this script; it
skips every entry point wrapper.cpp already defines, but refuses one whose stub
signature differs from miopen.h.

wrapper.cpp also holds the stubs of the wrapper's own functions, declared in
include_wrapper/miopen/hipdnn_forwarding.h instead of miopen.h. Only a flag-on
build exports them and they have no _impl, so nothing is generated for them.
This script reads that header only to accept those stubs, and refuses a name
that both headers declare.

The output is formatted with clang-format, and releases can wrap lines
differently. The committed text is what the release pinned in
.pre-commit-config.yaml produces, so that release is used when it can be found.
Another release is accepted while its output matches the committed files, so a
moved pin or a lagging CI image breaks nothing as long as the wrapping is the
same. When its output differs, --check reports the drift with a note that the
release is not the pinned one, and a write is refused: a change in wrapping
cannot be told from a real change.

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
    AbiError,
    EntryPoint,
    parse_public_header,
    parse_wrapper,
    read_tracked_source,
    render_signature,
    strip_comments,
    stub_signature,
)

STATUS_TYPE = "miopenStatus_t"

PUBLIC_HEADER_RELPATH = "include/miopen/miopen.h"
WRAPPER_ONLY_HEADER_RELPATH = "include_wrapper/miopen/hipdnn_forwarding.h"
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
// wrapper_passthrough.cpp) and for the hipDNN code that reads MIOpen state
// through the _impl getters (hipdnn_conv.cpp, hipdnn_graph.cpp). The _impl names
// have C linkage, so a signature that differed from the definition would link
// cleanly and corrupt arguments at run time. Each declaration is generated from
// the miopen.h declaration the definition is compiled from, so the two cannot
// differ.
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
# Reading wrapper.cpp and the wrapper-only header. miopen.h is read by the
# shared parser in check_public_abi.py.
# --------------------------------------------------------------------------


def wrapper_only_names(source: str | None, entries: list[EntryPoint]) -> set[str]:
    """The functions hipdnn_forwarding.h declares; None means there is no header."""
    if source is None:
        return set()
    names = {
        e.name
        for e in parse_public_header(source, "hipdnn_forwarding.h", allow_empty=True)
    }
    both = sorted(names & {e.name for e in entries})
    if both:
        raise AbiError(
            "declared in both miopen.h and hipdnn_forwarding.h: "
            + ", ".join(both)
            + ". A function both builds export belongs in miopen.h alone"
        )
    return names


def hand_written_stubs(
    wrapper_source: str,
    entries: list[EntryPoint],
    wrapper_only: frozenset[str] | set[str] = frozenset(),
) -> set[str]:
    """The entry points wrapper.cpp defines itself, which are not generated.

    ``wrapper_only`` names the functions declared in hipdnn_forwarding.h, whose
    stubs are hand-written by definition.

    A stub whose signature differs from miopen.h is refused. Only a flag-on
    build compiles wrapper.cpp, so otherwise the mismatch would surface there,
    long after the miopen.h change merged.
    """
    protos, _forwards, _dispatches = parse_wrapper(wrapper_source)
    # parse_wrapper cannot see a body that contains a nested block. Leaving such
    # a stub out would generate a second definition of it, so count the
    # definitions independently and refuse when the two disagree.
    stripped = strip_comments(wrapper_source, "wrapper.cpp")
    defined = len(re.findall(r'\bextern\s+"C"(?!\s*\{)', stripped))
    if defined != len(protos):
        raise AbiError(
            f'wrapper.cpp has {defined} extern "C" definitions but only {len(protos)}'
            " could be read. A stub body with a nested { } block is not supported;"
            " move that logic into a helper function"
        )
    declared = {entry.name for entry in entries} | set(wrapper_only)
    unknown = sorted(set(protos) - declared)
    if unknown:
        raise AbiError(
            "wrapper.cpp defines entry points that neither miopen.h nor"
            " hipdnn_forwarding.h declares: " + ", ".join(unknown)
        )
    mismatched = []
    for entry in entries:
        if entry.name not in protos:
            continue
        expected = stub_signature(entry)
        if protos[entry.name] != expected:
            mismatched.append(
                f"  {entry.name}\n"
                f"      miopen.h:{entry.line}: {render_signature(expected)}\n"
                f"      wrapper.cpp: {render_signature(protos[entry.name])}"
            )
    if mismatched:
        raise AbiError(
            "hand-written stubs in wrapper.cpp do not match miopen.h. The generator"
            " does not write these, so edit each stub to match:\n"
            + "\n".join(mismatched)
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


def pinned_clang_format_version(pre_commit_config: str | None) -> str | None:
    """The release pinned for the clang-format hook, or None if it cannot be read.

    The pin only chooses a binary and words a note, so a config edit that hides it
    from the pattern must not fail the check.
    """
    match = _PIN_RE.search(pre_commit_config) if pre_commit_config else None
    return match.group(1) if match else None


def clang_format_version(binary: str) -> str | None:
    try:
        done = subprocess.run(
            [binary, "--version"], capture_output=True, check=False, text=True
        )
    except OSError:
        return None
    match = _VERSION_RE.search(done.stdout)
    return match.group(1) if match else None


def install_hint(pinned: str | None) -> str:
    package = (
        f"clang-format=={pinned}" if pinned else "clang-format==<the pinned release>"
    )
    return (
        f"Get the release pinned in {PRE_COMMIT_CONFIG_RELPATH} with `pip install"
        f" {package}` and pass --clang-format or set CLANG_FORMAT, or run the manual"
        f" pre-commit hook, which installs it for you:\n  {REGENERATE_HOOK_COMMAND}"
    )


def not_pinned_note(version: str, pinned: str | None) -> str | None:
    if version == pinned:
        return None
    if pinned is None:
        return (
            f"clang-format {version} was used, and the release pinned in"
            f" {PRE_COMMIT_CONFIG_RELPATH} could not be read, so a difference may"
            " be formatting only."
        )
    return (
        f"clang-format {version} was used, not the pinned {pinned}. Releases can wrap"
        " lines differently, so a difference may be formatting only."
    )


def find_clang_format(explicit: str | None, pinned: str | None) -> tuple[str, str]:
    """The binary to format with, and its release.

    A named binary is used as given. Otherwise prefer the pinned release but take
    any that runs: only a difference in output needs explaining.
    """
    requested = explicit or os.environ.get("CLANG_FORMAT")
    if requested:
        version = clang_format_version(requested)
        if version is None:
            raise AbiError(f"{requested} is not a runnable clang-format")
        return requested, version

    found: list[tuple[str, str]] = []
    for candidate in (shutil.which("clang-format"), FALLBACK_CLANG_FORMAT):
        if not candidate or not Path(candidate).is_file():
            continue
        version = clang_format_version(candidate)
        if version is not None:
            found.append((candidate, version))
    for candidate, version in found:
        if version == pinned:
            return candidate, version
    if found:
        return found[0]
    raise AbiError(
        f"no clang-format found on PATH or at {FALLBACK_CLANG_FORMAT}. "
        + install_hint(pinned)
    )


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

    def read(rel: str, may_be_new: bool = False, optional: bool = False) -> str | None:
        """Read one file. ``may_be_new`` is for an output the first run creates,
        ``optional`` for an input whose absence is meaningful."""
        path = repo_root / rel
        if args.check:
            return read_tracked_source(path, repo_root, reasons)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        if optional or (may_be_new and path.parent.is_dir()):
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
    wrapper_only = wrapper_only_names(
        read(f"{miopen_rel}/{WRAPPER_ONLY_HEADER_RELPATH}", optional=True), entries
    )
    hand_written = hand_written_stubs(wrapper_source, entries, wrapper_only)
    check_dispatchable(entries, hand_written)
    texts = build_texts(entries, hand_written)

    pinned = pinned_clang_format_version(read(PRE_COMMIT_CONFIG_RELPATH, optional=True))
    binary, version = find_clang_format(args.clang_format, pinned)
    note = not_pinned_note(version, pinned)

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
        changing = [
            outputs[key].relpath
            for key, text in generated.items()
            if committed[key] != text
        ]
        if changing and note:
            raise AbiError(
                f"{note} Nothing was written, because a change in wrapping cannot be"
                " told from a real change. These files would have changed:\n  "
                + "\n  ".join(changing)
                + "\n"
                + install_hint(pinned)
            )
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
        if note:
            print(f"\n{note}")
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
        help="clang-format binary to use (default: $CLANG_FORMAT, else the pinned "
        f"release on PATH or at {FALLBACK_CLANG_FORMAT}, else any release found there)",
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
