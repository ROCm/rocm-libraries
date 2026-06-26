#!/usr/bin/env python3
###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ###############################################################################

"""Certify that the hipTensor stub source covers the whole public API.

The host-only stub (library/stub/hiptensor_stub.cpp) must implement every
public entry point so that, on architectures where the device library cannot be
built, downstream consumers still link and get a defined HIPTENSOR_STATUS_NOT_
SUPPORTED result instead of an unresolved symbol.

The stub is easy to forget when the API changes, so this test compares the two
source files directly:
  * the set of functions declared with HIPTENSOR_EXPORT in the public header
  * the set of functions defined in the stub source

If the header has a function the stub does not implement (or the stub defines a
function no longer in the header), the test fails and names the offenders. It is
a pure source-level check: no compiler, no built library, no GPU required, so it
runs as part of the normal test suite for any GPU_TARGETS.
"""

import argparse
import re
import sys
from pathlib import Path


def _strip_comments(text: str) -> str:
    """Remove C/C++ block and line comments so prose can't add phantom names."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def parse_declared_functions(header: Path) -> set[str]:
    """Return the function names declared with HIPTENSOR_EXPORT in the header."""
    text = _strip_comments(header.read_text())
    names: set[str] = set()
    # Each exported declaration looks like:
    #   HIPTENSOR_EXPORT <return type> hiptensorXxx( ... );
    for decl in re.findall(r"HIPTENSOR_EXPORT\b(.*?);", text, flags=re.DOTALL):
        match = re.search(r"(hiptensor[A-Za-z0-9_]*)\s*\(", decl)
        if match:
            names.add(match.group(1))
    return names


def parse_defined_functions(source: Path) -> set[str]:
    """Return the hiptensor* function names defined in the stub source.

    A definition is a `hiptensorXxx( ... ) {`; a trailing-semicolon declaration
    is not. Matching the brace avoids counting any forward declarations.
    """
    text = _strip_comments(source.read_text())
    names: set[str] = set()
    for match in re.finditer(r"\b(hiptensor[A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{", text):
        names.add(match.group(1))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--header", required=True, type=Path, help="Path to public hiptensor.h"
    )
    parser.add_argument(
        "--stub", required=True, type=Path, help="Path to hiptensor_stub.cpp"
    )
    args = parser.parse_args()

    declared = parse_declared_functions(args.header)
    if not declared:
        print(f"ERROR: no HIPTENSOR_EXPORT functions found in {args.header}")
        return 1

    defined = parse_defined_functions(args.stub)

    missing = sorted(declared - defined)  # in header, not in stub
    extra = sorted(defined - declared)  # in stub, not in header

    print(f"public API functions declared in header: {len(declared)}")
    print(f"functions defined in stub:               {len(defined)}")

    if missing:
        print("ERROR: stub is missing these public API functions:")
        for name in missing:
            print(f"  {name}")
    if extra:
        print("ERROR: stub defines functions not in the public API:")
        for name in extra:
            print(f"  {name}")

    if missing or extra:
        print(
            "The public API and the stub are out of sync. Update "
            f"{args.stub.name} so it implements exactly the public API."
        )
        return 1

    print("PASS: stub covers every public API function.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
