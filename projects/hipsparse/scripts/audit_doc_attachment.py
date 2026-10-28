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
when Clang attaches a FullComment to its declaration. hipsparse.h is parsed
as a single translation unit, the way a user or binding generator includes
it, and the results are filtered to the requested headers. CUDART_VERSION
is left undefined, as in the AMD build, so declarations that exist only for
older CUDA versions are not audited. The implementation shells out to clang
so a Python libclang package is not required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

PROTOTYPE_RE = re.compile(r"\bhipsparseStatus_t\s+hipsparse[A-Za-z0-9_]+\s*\(")

# Stand-ins for the headers CMake generates into the build tree.
GENERATED_HEADERS = {
    "hipsparse-config.h": "",
    "hipsparse-export.h": (
        '#define HIPSPARSE_EXPORT __attribute__((visibility("default")))\n'
        "#define HIPSPARSE_DEPRECATED __attribute__((__deprecated__))\n"
    ),
    "hipsparse-version.h": "",
}

Result = Tuple[Path, int, str, bool]


def clang_binary() -> str:
    candidates = [
        os.environ.get("HIPSPARSE_CLANG"),
        "/opt/rocm/llvm/bin/clang",
        shutil.which("clang"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "clang is required to audit documentation attachment"
    )


def header_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(path.rglob("*.h"))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"{path}: no such file or directory")
    return sorted(set(file.resolve() for file in files))


def write_generated_headers(directory: Path) -> None:
    for subdirectory in (directory, directory / "hipsparse"):
        subdirectory.mkdir(exist_ok=True)
        for name, contents in GENERATED_HEADERS.items():
            (subdirectory / name).write_text(contents, encoding="utf-8")


def json_objects(dump: str) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    position = 0
    while True:
        while position < len(dump) and dump[position].isspace():
            position += 1
        if position == len(dump):
            return
        node, position = decoder.raw_decode(dump, position)
        yield node


def location_file(location: dict) -> Tuple[Optional[str], Optional[int]]:
    # Each filtered declaration is dumped on its own, so the first location
    # printed (the declaration's own "loc") always carries the file name.
    if "expansionLoc" in location:
        spelling = location.get("spellingLoc", {})
        expansion = location["expansionLoc"]
        return (
            expansion.get("file", spelling.get("file")),
            expansion.get("line", spelling.get("line")),
        )
    return location.get("file"), location.get("line")


def parse_ast(dump: str) -> List[Result]:
    results: List[Result] = []
    for node in json_objects(dump):
        name = node.get("name", "")
        if node.get("kind") != "FunctionDecl" or not name.startswith(
            "hipsparse"
        ):
            continue
        file, line = location_file(node.get("loc", {}))
        if file is None or line is None:
            raise RuntimeError(
                f"no source location in the AST dump for {name}"
            )
        documented = any(
            child.get("kind") == "FullComment"
            for child in node.get("inner", [])
        )
        results.append((Path(file).resolve(), line, name, documented))
    return results


def audit(
    umbrella: Path, include_dirs: List[Path], rocm_path: Path, clang: str
) -> List[Result]:
    with tempfile.TemporaryDirectory() as generated:
        write_generated_headers(Path(generated))
        args = [
            clang,
            "-x",
            "c",
            "-fsyntax-only",
            "-D__HIP_PLATFORM_AMD__",
        ]
        args.extend(f"-I{directory}" for directory in include_dirs)
        args.extend(
            [
                f"-I{generated}",
                f"-I{umbrella.parent}",
                f"-I{rocm_path / 'include'}",
                "-Xclang",
                "-ast-dump=json",
                "-Xclang",
                "-ast-dump-filter=hipsparse",
                str(umbrella),
            ]
        )
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"clang exited with status {completed.returncode} parsing {umbrella}"
        )
    return parse_ast(completed.stdout)


def main() -> int:
    include_root = (
        Path(__file__).resolve().parent.parent / "library" / "include"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[include_root],
        help="header file or directory to report (defaults to library/include)",
    )
    parser.add_argument(
        "--umbrella",
        type=Path,
        default=include_root / "hipsparse.h",
        help="header parsed as the translation unit (defaults to hipsparse.h)",
    )
    parser.add_argument(
        "-I",
        dest="include_dirs",
        action="append",
        type=Path,
        default=[],
        help="extra include directory searched before the generated-header stand-ins",
    )
    parser.add_argument(
        "--rocm-path",
        type=Path,
        default=Path(os.environ.get("ROCM_PATH", "/opt/rocm")),
        help="ROCm installation providing hip/hip_runtime.h (defaults to $ROCM_PATH or /opt/rocm)",
    )
    args = parser.parse_args()

    try:
        headers = header_files(args.paths)
        results = audit(
            args.umbrella.resolve(),
            args.include_dirs,
            args.rocm_path,
            clang_binary(),
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    by_header: Dict[Path, List[Result]] = {header: [] for header in headers}
    for result in results:
        if result[0] in by_header:
            by_header[result[0]].append(result)

    documented = undocumented = 0
    missing: List[Path] = []
    for header, header_results in by_header.items():
        if not header_results:
            if PROTOTYPE_RE.search(header.read_text(encoding="utf-8")):
                missing.append(header)
            continue
        print(f"\n=== {header} ===")
        for _, line, name, has_docs in sorted(header_results):
            if has_docs:
                documented += 1
                status = "documented"
            else:
                undocumented += 1
                status = "UNDOCUMENTED"
            print(f"  {line:5d} {name:50s} -> {status}")

    print()
    print("=" * 60)
    print(f"Functions documented:   {documented}")
    print(f"Functions undocumented: {undocumented}")
    total = documented + undocumented
    if total:
        print(f"Coverage:               {100.0 * documented / total:.1f}%")

    for header in missing:
        print(
            f"error: {header} declares hipsparse functions but none were found "
            f"in the AST of {args.umbrella}",
            file=sys.stderr,
        )
    if missing:
        return 2
    if not total:
        print("error: no hipsparse functions found", file=sys.stderr)
        return 2
    return 1 if undocumented else 0


if __name__ == "__main__":
    sys.exit(main())
