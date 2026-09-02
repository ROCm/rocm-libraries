#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Single-shot exact-shape benchmark for gfx950 dense prefill (JSON to stdout/file)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

from builders.gfx950.attention.prefill.attention_dense_prefill import (  # noqa: E402
    make_spec_from_shape,
    run_benchmark,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape-json", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    shape = json.loads(args.shape_json.read_text())
    result = run_benchmark(
        make_spec_from_shape(shape),
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        check=not args.no_check,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
