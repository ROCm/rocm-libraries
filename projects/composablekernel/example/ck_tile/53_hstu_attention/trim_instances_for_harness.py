#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Keep a harness-focused jagged instance subset (bf16/fp16, maxk 64/128)."""

from __future__ import annotations

import re
from pathlib import Path

KEEP_PATTERNS = [
    re.compile(
        r"hstu_attention_jagged_forward_(fp16|bf16)_"
        r"(has_causal|no_causal)_softmax_false_no_bias_no_dropout_maxk_(64|128)\.cpp$"
    ),
]

INSTANCE_DIR = Path(__file__).parent / "instances"


def should_keep(name: str) -> bool:
    return any(p.search(name) for p in KEEP_PATTERNS)


def main() -> None:
    kept = []
    removed = 0
    for path in sorted(INSTANCE_DIR.glob("hstu_attention_*.cpp")):
        if should_keep(path.name):
            kept.append(path.name)
            continue
        path.unlink()
        removed += 1
    print(f"Kept {len(kept)} instance .cpp files, removed {removed}")
    for name in sorted(kept):
        print(f"  {name}")


if __name__ == "__main__":
    main()
