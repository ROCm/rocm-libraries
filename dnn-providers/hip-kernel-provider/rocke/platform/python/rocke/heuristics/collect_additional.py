#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Cross-platform (Windows + Linux) replacement for collect_additional.sh.
# Generates additional benchmark data for shapes NOT in the original log by
# running the CK Tile gemm_universal benchmark binaries and capturing the JSON
# block each prints. Output is streaming JSON parseable by data_pipeline.py.
#
# Point CK_TILE_BIN_DIR at your local CK Tile build (default: ./ck_tile/bin).

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

BIN_DIR = Path(os.environ.get("CK_TILE_BIN_DIR", "ck_tile/bin"))
OUT_LOG = Path("data/additional_shapes.log")
WARMUP = 3
REPEAT = 10

# Square powers-of-2 and common ML sizes not in the original DeepSeek set.
SHAPES = [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (1, 4096, 4096),
    (8, 4096, 4096),
    (32, 4096, 4096),
    (128, 4096, 4096),
    (1, 4096, 11008),
    (32, 4096, 11008),
    (1, 8192, 8192),
    (32, 8192, 8192),
    (1, 8192, 28672),
    (32, 8192, 28672),
    (256, 256, 8192),
    (8192, 8192, 256),
    (1024, 4096, 1024),
    (4096, 1024, 4096),
    (2048, 8192, 2048),
]

_JSON_BLOCK = re.compile(r"\{.*?\n\}", re.DOTALL)


def main() -> int:
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    exes = sorted(BIN_DIR.glob("benchmark_gemm_universal_fp8_rcr_*"))
    if not exes:
        print(
            f"no benchmark binaries under {BIN_DIR} " f"(set CK_TILE_BIN_DIR)",
            file=sys.stderr,
        )

    with OUT_LOG.open("w", encoding="utf-8") as log:
        log.write("CK Tile Additional Shapes Benchmark\n")
        log.write("GPU ID: 0\n")
        log.write("Implementation: gemm_universal\n\n")
        for idx, (m, n, k) in enumerate(SHAPES, 1):
            log.write("========================================\n")
            log.write(f"Shape {idx}: M={m} N={n} K={k} dtype=fp8 layout=rcr\n")
            log.write("========================================\n")
            for exe in exes:
                out = subprocess.run(
                    [
                        str(exe),
                        f"-m={m}",
                        f"-n={n}",
                        f"-k={k}",
                        f"-warmup={WARMUP}",
                        f"-repeat={REPEAT}",
                        "-verify=0",
                    ],
                    capture_output=True,
                    text=True,
                ).stdout
                for block in _JSON_BLOCK.findall(out):
                    log.write(block + "\n")
            log.write(f"Found {len(exes)} kernels\n")
            print(
                f"Completed shape {idx}: M={m} N={n} K={k} ({len(exes)} kernels)",
                file=sys.stderr,
            )
    print("Done generating additional data", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
