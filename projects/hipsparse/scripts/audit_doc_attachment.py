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

"""Audit per-function Doxygen attachment as Clang sees it.

This is the same diagnostic used in AISPARSE-550: a function is documented
when Clang attaches a FullComment / raw_comment to the declaration. The
implementation shells out to clang so a Python libclang package is not
required.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


FUNCTION_RE = re.compile(
    r"FunctionDecl\b.*\b(?P<name>hipsparse[A-Za-z0-9_]+)\s+'"
)
COMMENT_RE = re.compile(r"`?-FullComment\b")

PREDEFINES = [
    "HIPSPARSE_EXPORT=",
    "ROCSPARSE_EXPORT=",
    "DEPRECATED_CUDA_9000(x)=",
    "DEPRECATED_CUDA_10000(x)=",
    "DEPRECATED_CUDA_11000(x)=",
    "DEPRECATED_CUDA_12000(x)=",
    "DEPRECATED_CUDA_13000(x)=",
    "HIPSPARSE_DEPRECATED_MSG(x)=",
    "HIPSPARSE_DEPRECATED=",
    "ROCSPARSE_DEPRECATED_MSG(x)=",
    "ROCSPARSE_DEPRECATED=",
    "__attribute__(x)=",
    "__HIP_PLATFORM_AMD__",
    "CUDART_VERSION=0",
]


def clang_binary() -> str:
    candidates = [
        os.environ.get("HIPSPARSE_CLANG"),
        "/opt/rocm/llvm/bin/clang",
        shutil.which("clang"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("clang is required to audit documentation attachment")


def header_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(path.rglob("*.h"))
        else:
            files.append(path)
    return sorted(set(files))


def parse_ast(dump: str) -> List[Tuple[str, bool]]:
    results: List[Tuple[str, bool]] = []
    current = None
    documented = False
    for line in dump.splitlines():
        match = FUNCTION_RE.search(line)
        if match:
            if current is not None:
                results.append((current, documented))
            current = match.group("name")
            documented = False
            continue
        if current is not None and COMMENT_RE.search(line):
            documented = True
    if current is not None:
        results.append((current, documented))
    return results


def audit_header(path: Path, clang: str) -> Sequence[Tuple[str, bool]]:
    args = [
        clang,
        "-x",
        "c",
        "-fsyntax-only",
        "-fparse-all-comments",
        "-Xclang",
        "-ast-dump",
        "-Wno-everything",
    ]
    args.extend(f"-D{define}" for define in PREDEFINES)
    args.append(str(path))
    completed = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return parse_ast(completed.stdout)


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
    args = parser.parse_args()

    clang = clang_binary()
    documented = undocumented = 0
    failed = 0
    for path in header_files(args.paths):
        print(f"\n=== {path} ===")
        results = audit_header(path, clang)
        if not results:
            continue
        for name, has_docs in results:
            if has_docs:
                documented += 1
                status = "documented"
            else:
                undocumented += 1
                failed += 1
                status = "UNDOCUMENTED"
            print(f"  {name:50s} -> {status}")

    print()
    print("=" * 60)
    print(f"Functions documented:   {documented}")
    print(f"Functions undocumented: {undocumented}")
    total = documented + undocumented
    if total:
        print(f"Coverage:               {100.0 * documented / total:.1f}%")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
