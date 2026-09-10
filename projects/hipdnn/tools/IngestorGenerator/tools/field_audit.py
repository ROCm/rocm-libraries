#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Inventory schema accessor references, not native semantic correctness."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schema", type=Path, help="Exactly one FlatBuffer schema")
    parser.add_argument("sources", type=Path, nargs="+", help="Native source files")
    args = parser.parse_args(argv)
    try:
        if args.schema.suffix != ".fbs" or any(
            p.suffix == ".fbs" for p in args.sources
        ):
            raise ValueError("pass exactly one .fbs schema, followed by native sources")
        schema = args.schema.read_text(encoding="utf-8")
        # This is intentionally a lexical inventory. A reference can still be in
        # dead code or fail to implement the field's semantics.
        schema = re.sub(r"/\*.*?\*/|//[^\n]*", "", schema, flags=re.DOTALL)
        fields = sorted(
            set(re.findall(r"^\s*([a-z][a-z0-9_]*)\s*:", schema, re.MULTILINE))
        )
        if not fields:
            raise ValueError(f"parsed zero fields from {args.schema}")
        sources = [path.read_text(encoding="utf-8") for path in args.sources]
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    missing = [
        field
        for field in fields
        if not any(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}\s*\(", source)
            for source in sources
        )
    ]
    for field in missing:
        print(f"UNCHECKED: {field}")
    print(
        f"-- {len(fields) - len(missing)}/{len(fields)} fields referenced in {len(sources)} pack source(s)"
    )
    print(
        "Lexical reference inventory only; does not prove consumption, rejection, or native correctness."
    )
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
