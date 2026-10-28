#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

"""Check or synchronize documentation for hipSPARSE precision groups.

The full Doxygen block immediately before ``/**@{*/`` is the canonical
documentation for that group. Each declaration after the first one must carry
an exact literal copy so declaration-based tools such as libclang can see it.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Match, Tuple


DOC_PATTERN = r"/\*!(?:(?!\*/).)*\*/"
PREFIX_PATTERN = (
    r"(?:DEPRECATED_[A-Z0-9_]+\([^\n]*\)|"
    r"HIPSPARSE_DEPRECATED_MSG\([^\n]*\)|"
    r"HIPSPARSE_EXPORT)"
)
GROUP_PATTERN = re.compile(
    r"(?P<canonical>"
    + DOC_PATTERN
    + r")\n+"
    + r"(?P<open>/\*\*@\{\*/)\n"
    + r"(?P<body>.*?)"
    + r"(?P<close>/\*\*@\}\*/)",
    re.DOTALL,
)
GROUP_OPEN = "/**@{*/"
ENTRY_PATTERN = re.compile(
    r"(?:(?P<doc>" + DOC_PATTERN + r")(?P<separator>\s*))?"
    r"(?P<prefix>(?:" + PREFIX_PATTERN + r"\n)+)"
    r"(?=hipsparseStatus_t\s+(?P<name>hipsparse[A-Za-z0-9_]+)\s*\()",
    re.DOTALL,
)
DECL_NAME_PATTERN = re.compile(
    r"hipsparseStatus_t\s+(hipsparse[A-Za-z0-9_]+)\s*\("
)


class SyncError(RuntimeError):
    pass


def header_files(paths: Iterable[Path]) -> List[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(path.rglob("*.h"))
        elif path.is_file():
            files.append(path)
        else:
            raise SyncError(f"{path}: no such file or directory")
    return sorted(set(files))


def synchronize_group(match: Match[str], write: bool) -> Tuple[str, List[str]]:
    canonical = "\n".join(
        line.rstrip() for line in match.group("canonical").splitlines()
    )
    body = match.group("body")
    entries = list(ENTRY_PATTERN.finditer(body))
    declared = DECL_NAME_PATTERN.findall(body)
    stale = []

    if [entry.group("name") for entry in entries] != declared:
        unmatched = [
            name
            for name in declared
            if name not in {entry.group("name") for entry in entries}
        ]
        raise SyncError(
            "unable to parse grouped declarations: " + ", ".join(unmatched)
        )
    if not entries:
        raise SyncError("group without hipsparse declarations")

    if match.group("canonical") != canonical:
        stale.append(entries[0].group("name"))
    for entry in entries[1:]:
        if entry.group("doc") != canonical or entry.group("separator") != "\n":
            stale.append(entry.group("name"))

    for entry in reversed(entries[1:]):
        replacement = canonical + "\n" + entry.group("prefix")
        body = body[: entry.start()] + replacement + body[entry.end() :]

    replacement = (
        canonical
        + "\n"
        + match.group("open")
        + "\n"
        + body
        + match.group("close")
    )
    if replacement != match.group(0) and not stale:
        stale.append(entries[0].group("name"))
    return (replacement if write else match.group(0)), stale


def synchronize_file(path: Path, write: bool) -> List[str]:
    source = path.read_text(encoding="utf-8")
    stale = []

    groups = 0

    def replace(match: Match[str]) -> str:
        nonlocal groups
        groups += 1
        replacement, group_stale = synchronize_group(match, write)
        stale.extend(group_stale)
        return replacement

    try:
        updated = GROUP_PATTERN.sub(replace, source)
        markers = source.count(GROUP_OPEN)
        if groups != markers:
            raise SyncError(
                f"{markers - groups} of {markers} {GROUP_OPEN} group(s) do not "
                "directly follow a /*! ... */ block"
            )
    except SyncError as error:
        raise SyncError(f"{path}: {error}") from error

    if write and updated != source:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(updated)
    return stale


def main() -> int:
    default_path = (
        Path(__file__).resolve().parent.parent
        / "library"
        / "include"
        / "internal"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[default_path],
        help="header file or directory (defaults to library/include/internal)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace missing or stale copies with the canonical documentation",
    )
    args = parser.parse_args()

    stale_count = 0
    try:
        for path in header_files(args.paths):
            stale = synchronize_file(path, args.write)
            stale_count += len(stale)
            for function in stale:
                action = "synchronized" if args.write else "out of sync"
                print(f"{path}:{function}: {action}")
    except SyncError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if stale_count and not args.write:
        print(
            f"{stale_count} declaration(s) have missing or stale documentation."
            " Edit only the block before /**@{*/ and run"
            " projects/hipsparse/scripts/sync_group_docs.py --write.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
