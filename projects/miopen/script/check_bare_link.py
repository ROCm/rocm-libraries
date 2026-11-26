#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check that a bare -lMIOpen link resolves every symbol the flag-off library exports.

With MIOPEN_ENABLE_HIPDNN_WRAPPER=ON, libMIOpen.so is a thin wrapper whose
implementation sits behind a DT_NEEDED entry on libMIOpen_private.so. A symbol
defined only in that indirect library is not visible to a consumer that names
just -lMIOpen: GNU ld has defaulted to --no-copy-dt-needed-entries since binutils
2.22, so the reference is undefined at link time. A CMake configure-time probe
such as MIGraphX's check_library_exists(MIOpen ...) fails the same way, and
silently turns a feature off.

check_public_abi.py check-wrapper compares the wrapper's exported symbol set with
the baseline, which says the symbols are exported. This links a program that
takes the address of every baseline symbol against -lMIOpen alone, which says a
consumer can use them. It never runs the program.

Linux only: a Windows import library already fails the link for a symbol the DLL
does not export, so the export-table comparison in check-wrapper covers it.

Exit codes: 0 every symbol resolved, 1 the link failed or a library is missing,
2 bad input (unreadable baseline, a named compiler that does not exist, no
libMIOpen.so in the library directory), 77 nothing to check here (Windows, or no
compiler found when none was named). 77 is the code ctest's SKIP_RETURN_CODE
property is set to.

    check_bare_link.py [--lib-dir <dir>] [--baseline <file>] [--compiler <c++>]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from check_public_abi import AbiError, read_symbol_list
from miopen_wrapper_libs import ON_WINDOWS, default_lib_dirs, resolve_pair

SCRIPT_DIR = Path(__file__).resolve().parent

SKIP = 77

# Tried in order, after $CXX, when --compiler is not given. Any C++ driver that
# runs the linker will do.
COMPILER_CANDIDATES = ("c++", "clang++", "g++")

# Linker messages: an undefined symbol, and a linker that rejects the GNU ld
# option.
UNDEFINED_RE = re.compile(r"undefined (?:reference to|symbol:?) [`'‘]?([A-Za-z_]\w*)")
UNKNOWN_OPTION_RE = re.compile(
    r"(unknown argument|unrecognized option|unknown option)[^\n]*"
    r"no-copy-dt-needed-entries",
    re.I,
)


def render_source(symbols):
    """A program that references every symbol, so each must resolve at link time.

    The table is a non-static global, so the references stay in the object file
    whatever the optimization level. Types do not matter to the linker: C linkage
    names only the symbol.
    """
    names = sorted(symbols)
    lines = ['extern "C" {']
    lines += [f"void {name}(void);" for name in names]
    lines += ["}", "", "typedef void (*Entry)(void);", "extern const Entry entries[];"]
    lines += ["const Entry entries[] = {"]
    lines += [f"    {name}," for name in names]
    lines += ["};", "", "int main() { return entries[0] == nullptr; }", ""]
    return "\n".join(lines)


def find_compiler(requested):
    if requested:
        found = shutil.which(requested)
        if found is None:
            raise AbiError(f"compiler not found: {requested}")
        return found
    for candidate in (os.environ.get("CXX"), *COMPILER_CANDIDATES):
        found = shutil.which(candidate) if candidate else None
        if found:
            return found
    return None


def link(compiler, source, output, lib_dir, copy_dt_needed_option):
    command = [
        compiler,
        str(source),
        "-o",
        str(output),
        f"-L{lib_dir}",
        "-lMIOpen",
        # MIOpen_private's own dependencies are not under test and may not be on
        # the link path; only this program's references count.
        "-Wl,--allow-shlib-undefined",
        f"-Wl,-rpath-link,{lib_dir}",
    ]
    if copy_dt_needed_option:
        command.append("-Wl,--no-copy-dt-needed-entries")
    print(f"+ {' '.join(command)}", flush=True)
    return subprocess.run(command, capture_output=True, text=True)


def check_bare_link(compiler, lib_dir, symbols):
    """Return (ok, unresolved symbols, linker output)."""
    if not (Path(lib_dir) / "libMIOpen.so").exists():
        raise AbiError(
            f"{lib_dir} has no libMIOpen.so, so -lMIOpen cannot be linked there"
        )
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "bare_link.cpp"
        source.write_text(render_source(symbols), encoding="utf-8")
        output = Path(work) / "bare_link"
        # GNU ld defaults to --no-copy-dt-needed-entries, but a distribution can
        # change that, so name it. A linker that rejects the option gets a retry
        # without it.
        result = link(compiler, source, output, lib_dir, copy_dt_needed_option=True)
        if result.returncode != 0 and UNKNOWN_OPTION_RE.search(
            result.stdout + result.stderr
        ):
            result = link(
                compiler, source, output, lib_dir, copy_dt_needed_option=False
            )
    text = result.stdout + result.stderr
    unresolved = sorted({m for m in UNDEFINED_RE.findall(text) if m in symbols})
    return result.returncode == 0, unresolved, text


REMEDY = """
A symbol the flag-off libMIOpen.so exports must also resolve from the flag-on
wrapper alone. Every entry point is exported by the wrapper under its original
name, and the private library exports it under the _impl name. An entry point
that is declared in neither include/miopen/miopen.h nor
src/include/miopen/miopen_internal.h gets no wrapper stub: declare it in one of
them and rerun projects/miopen/script/generate_wrapper_sources.py. Do not add
-lMIOpen_private to this check's link line to make it pass: consumers that link
a bare -lMIOpen, such as pkg-config users and configure-time probes, do not."""


def main(argv=None, windows=ON_WINDOWS):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lib-dir",
        help="directory holding libMIOpen.so and libMIOpen_private.so; defaults to "
        "the install prefix's lib*/, found from this script's location",
    )
    parser.add_argument("--baseline", default=SCRIPT_DIR / "public_symbols.baseline")
    parser.add_argument(
        "--compiler",
        help="C++ compiler driver to link with. When given it must exist; when not, "
        "$CXX, c++, clang++ and g++ are tried and the check is skipped if none is found",
    )
    args = parser.parse_args(argv)

    if windows:
        print(
            "SKIP: a Windows import library already rejects a symbol the DLL does not "
            "export; check_public_abi.py check-wrapper covers it"
        )
        return SKIP

    try:
        compiler = find_compiler(args.compiler)
        if compiler is None:
            print("SKIP: no C++ compiler found to link with")
            return SKIP
        lib_dirs = (
            [Path(args.lib_dir)] if args.lib_dir else default_lib_dirs(SCRIPT_DIR)
        )
        wrapper_lib, private_lib, problems = resolve_pair(lib_dirs)
        if problems:
            for problem in problems:
                print(f"FAIL: {problem}", flush=True)
            return 1
        print(f"libraries under test: {wrapper_lib}, {private_lib}", flush=True)
        symbols = read_symbol_list(args.baseline, "baseline symbols file")
        ok, unresolved, text = check_bare_link(compiler, wrapper_lib.parent, symbols)
    except AbiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if ok:
        print(
            f"PASS: a bare -lMIOpen link resolves all {len(symbols)} baseline symbols"
        )
        return 0
    print(f"FAIL: a bare -lMIOpen link could not resolve {len(unresolved)} symbols:")
    for name in unresolved:
        print(f"  {name}")
    if not unresolved:
        print("The link failed for another reason:")
        print(text)
    print(REMEDY)
    return 1


if __name__ == "__main__":
    sys.exit(main())
