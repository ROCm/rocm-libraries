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

"""Generate hipTensor stub sources from the public header.

Two related outputs are produced from the same parse of the public API (every
function declared with HIPTENSOR_EXPORT), selected with --mode:

* coverage (default): a link-time completeness check. Emits a translation unit
  that takes the address of every public function. Linked against only the stub
  object, a public API function the stub does not implement becomes an unresolved
  external symbol, so the build fails.

* stub: the stub implementations themselves. Emits a definition returning
  HIPTENSOR_STATUS_NOT_SUPPORTED for every function whose return type is
  hiptensorStatus_t. Functions with any other return type (the informational
  helpers such as hiptensorGetErrorString / hiptensorGetVersion /
  hiptensorGetHiprtVersion) are intentionally skipped -- they carry real behavior
  and are hand-written in hiptensor_stub_special.cpp.

Because both outputs are regenerated from the header, a newly added public API is
handled automatically: coverage mode adds an address reference the stub must
satisfy, and stub mode emits its NOT_SUPPORTED body (for status-returning APIs).
"""

import argparse
import re
import sys
from pathlib import Path

# The return type that gets an auto-generated NOT_SUPPORTED body. Every other
# return type denotes an informational helper with real behavior, hand-written
# in hiptensor_stub_special.cpp.
_STATUS_RETURN_TYPE = "hiptensorStatus_t"


class ExportedFunction:
    """A function declared with HIPTENSOR_EXPORT in the public header."""

    def __init__(self, return_type: str, name: str, params: str):
        self.return_type = return_type
        self.name = name
        self.params = params

    def __repr__(self) -> str:
        return f"ExportedFunction({self.return_type!r}, {self.name!r}, {self.params!r})"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    # Drop preprocessor directives. The header guards the export macro with
    #   #if !defined(HIPTENSOR_EXPORT) / #define HIPTENSOR_EXPORT / #endif
    # whose tokens would otherwise be captured as part of the first declaration's
    # return type.
    text = re.sub(r"^[ \t]*#[^\n]*", "", text, flags=re.MULTILINE)
    return text


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_exported_declarations(header: Path) -> list[ExportedFunction]:
    """Parse every HIPTENSOR_EXPORT declaration into (return type, name, params).

    The function name is located by the '(' that follows it -- this avoids
    mistaking the return type 'hiptensorStatus_t' for the function name, since
    the return type is not followed by a parenthesis.
    """
    text = _strip_comments(header.read_text())

    functions: list[ExportedFunction] = []
    seen: set[str] = set()
    for decl in re.findall(r"HIPTENSOR_EXPORT\b(.*?);", text, flags=re.DOTALL):
        # ret (non-greedy) then the first `hiptensorXxx(` -- the '(' anchors the
        # name so the return type is whatever precedes it.
        match = re.search(
            r"(?P<ret>.*?)\b(?P<name>hiptensor[A-Za-z0-9_]*)\s*\((?P<params>.*)\)\s*$",
            decl.strip(),
            flags=re.DOTALL,
        )
        if not match:
            continue
        name = match.group("name")
        if name in seen:
            continue
        seen.add(name)
        functions.append(
            ExportedFunction(
                return_type=_normalize_ws(match.group("ret")),
                name=name,
                params=_normalize_ws(match.group("params")),
            )
        )
    functions.sort(key=lambda f: f.name)
    return functions


def parse_exported_functions(header: Path) -> list[str]:
    """Return the sorted list of function names declared with HIPTENSOR_EXPORT."""
    return [f.name for f in parse_exported_declarations(header)]


# ---------------------------------------------------------------------------
# coverage mode
# ---------------------------------------------------------------------------

_COVERAGE_TEMPLATE = """\
// Auto-generated by generate_api_stub.py. Do not edit.
//
// Link-time completeness check for the hipTensor stub. References the address of
// every HIPTENSOR_EXPORT function in the public header. Linked against only the
// stub object, any public API function the stub fails to implement becomes an
// unresolved symbol and the build fails. This guarantees, at build time and for
// any GPU_TARGETS, that the stub implements the entire public API.

#include <hiptensor/hiptensor.h>

namespace
{{
    // Taking each function's address forces the linker to resolve the symbol.
    void* const kPublicApiSymbols[] = {{
{symbol_rows}
    }};
}}

// Referenced from main so the table (and thus every symbol) is not stripped.
const void* const* hiptensor_stub_api_coverage_symbols()
{{
    return kPublicApiSymbols;
}}

int main()
{{
    return hiptensor_stub_api_coverage_symbols() != nullptr ? 0 : 1;
}}
"""


def render_coverage(functions: list[ExportedFunction]) -> str:
    rows = ",\n".join(
        f"        reinterpret_cast<void*>(&{f.name})" for f in functions
    )
    return _COVERAGE_TEMPLATE.format(symbol_rows=rows)


# ---------------------------------------------------------------------------
# stub mode
# ---------------------------------------------------------------------------

_STUB_HEADER = """\
// Auto-generated by generate_api_stub.py. Do not edit.
//
// Stub implementations of the hipTensor public API: every function returns
// HIPTENSOR_STATUS_NOT_SUPPORTED. Used as libhiptensor on architectures with no
// usable GPU target (HIPTENSOR_DISABLE_DEVICE=ON), so downstream consumers link
// and get a defined runtime error instead of an unresolved symbol.
//
// Only functions returning hiptensorStatus_t are generated here. Informational
// helpers with real behavior (hiptensorGetErrorString, hiptensorGetVersion,
// hiptensorGetHiprtVersion) are hand-written in hiptensor_stub_special.cpp.

#include <hiptensor/hiptensor.h>

// Parameters are reproduced verbatim from the header but unused in the stub.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

"""

_STUB_FOOTER = "\n#pragma clang diagnostic pop\n"


def render_stub(functions: list[ExportedFunction]) -> str:
    bodies = []
    for f in functions:
        if f.return_type != _STATUS_RETURN_TYPE:
            # Informational helper with real behavior; hand-written elsewhere.
            continue
        bodies.append(
            f"{f.return_type} {f.name}({f.params})\n"
            f"{{\n"
            f"    return HIPTENSOR_STATUS_NOT_SUPPORTED;\n"
            f"}}\n"
        )
    return _STUB_HEADER + "\n".join(bodies) + _STUB_FOOTER


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--header", required=True, type=Path, help="Path to public hiptensor.h"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path of the .cpp to generate"
    )
    parser.add_argument(
        "--mode",
        choices=("coverage", "stub"),
        default="coverage",
        help="coverage: link-time completeness check (default). "
        "stub: NOT_SUPPORTED implementations.",
    )
    args = parser.parse_args()

    functions = parse_exported_declarations(args.header)
    if not functions:
        print(
            f"ERROR: no HIPTENSOR_EXPORT functions found in {args.header}",
            file=sys.stderr,
        )
        return 1

    if args.mode == "coverage":
        content = render_coverage(functions)
        summary = f"referencing {len(functions)} public API functions"
    else:
        content = render_stub(functions)
        generated = sum(1 for f in functions if f.return_type == _STATUS_RETURN_TYPE)
        summary = f"implementing {generated} status-returning public API functions"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Generated {args.output} ({args.mode} mode) {summary}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
