#!/usr/bin/env python3
"""
Audit per-function doxygen comment attachment in hipSPARSE headers, as seen
by libclang. Reports whether each function declaration has a doc comment
lexically attached to it (the same diagnostic referenced in AISPARSE-550).

A function is reported as "documented" iff Clang's raw_comment is non-empty
for that cursor (i.e. there is a /** ... */, /*! ... */, or /// comment
immediately preceding the declaration in source).

Usage:
    python3 audit_doc_attachment.py path/to/header.h
    python3 audit_doc_attachment.py path/to/directory/
"""

import sys
from pathlib import Path

try:
    import clang.cindex
except ImportError:
    sys.stderr.write("ERROR: install with `pip install --user libclang`\n")
    sys.exit(2)

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

PARSE_ARGS = ["-x", "c", "-Wno-everything"] + [f"-D{d}" for d in PREDEFINES]


def audit(path: Path) -> tuple[int, int]:
    idx = clang.cindex.Index.create()
    tu = idx.parse(str(path), args=PARSE_ARGS)
    documented = undocumented = 0
    print(f"\n=== {path} ===")
    for c in tu.cursor.get_children():
        if c.kind != clang.cindex.CursorKind.FUNCTION_DECL:
            continue
        if c.location.file is None or not str(c.location.file).endswith(path.name):
            continue
        if c.raw_comment:
            print("raw_comment:", c.raw_comment)
            print(f"  {c.spelling:50s} -> documented")
            documented += 1
        if c.brief_comment:
            print("brief_comment:", c.brief_comment)
        else:
            print(f"  {c.spelling:50s} -> UNDOCUMENTED")
            undocumented += 1
    return documented, undocumented


def collect(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(
        f for f in target.rglob("*")
        if f.is_file() and f.suffix in {".h", ".hpp", ".hh", ".hxx", ".h++"}
    )


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <header-or-directory>\n")
        return 2
    target = Path(sys.argv[1])
    if not target.exists():
        sys.stderr.write(f"ERROR: {target} does not exist\n")
        return 2

    total_doc = total_undoc = 0
    for header in collect(target):
        d, u = audit(header)
        total_doc += d
        total_undoc += u

    print()
    print("=" * 60)
    print(f"Functions documented:   {total_doc}")
    print(f"Functions undocumented: {total_undoc}")
    if total_doc + total_undoc > 0:
        pct = 100.0 * total_doc / (total_doc + total_undoc)
        print(f"Coverage:               {pct:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
