# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Derive the integer engine id dnn-benchmark selects with, from the engine's name.

`dnn-benchmark --engine` takes integers, not names: an id is the FNV-1a 64-bit hash
of the engine name, reinterpreted as a signed int64 (data_sdk EngineNames.hpp,
StringUtil.hpp). A flow that wants to benchmark one named engine therefore has to
convert first, and doing it in the flow is what keeps the engine's identity written
down once instead of pasted as a magic number into a YAML file that nobody can check.

Pinning a single engine matters more than convenience. The report is a list of rows
and the flow's JSON extractor cannot select one by name, so a run that benchmarks
every discovered engine hands the first row's timings to whatever asserts next. One
`--engine <id>` makes `graphs[0].results[0]` deterministic, and bench_report.py then
re-checks the row's `engine_name` anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: FNV-1a 64-bit, the constants data_sdk/utilities/StringUtil.hpp uses.
FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def engine_id(name: str) -> int:
    """FNV-1a over the UTF-8 name, as a signed 64-bit integer."""
    h = FNV_OFFSET_BASIS
    for byte in name.encode("utf-8"):
        h ^= byte
        h = (h * FNV_PRIME) & _MASK64
    return h - (1 << 64) if h >= (1 << 63) else h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--engine-name", required=True, help="`<namespace>:<Local>` engine name"
    )
    ap.add_argument("--out", required=True, help="where to write the JSON")
    args = ap.parse_args()

    name = args.engine_name.strip()
    if not name:
        print("error: --engine-name is empty", file=sys.stderr)
        return 2

    ident = engine_id(name)
    result = {
        "engine_name": name,
        "engine_id": ident,
        "engine_id_unsigned": ident + (1 << 64) if ident < 0 else ident,
        "feedback": "",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
