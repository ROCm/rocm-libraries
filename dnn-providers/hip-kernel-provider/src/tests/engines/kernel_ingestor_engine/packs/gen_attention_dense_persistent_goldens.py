#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Writes TestAttentionDensePersistentGoldens.inc, the rocKE side of the launch-grid
restatement in packs/AttentionDensePersistent.hpp, which TestAttentionDensePersistent.cpp
checks the C++ against:

  * calculate_magic_numbers for every divisor in 1..4096 and 512 seeded random divisors
    up to 2^20 (as the i32 the kernel's mul-hi reads);
  * the kernel argument names attention_dense_signature declares per grid and emit_lse;
  * per (grid, shape) over a grid of shapes: whether Gfx942AttentionDenseSpec construction
    (the decode's check) or fastdiv_kernargs (the 2^30 dividend bound) refuses it, and
    otherwise attention_dense_grid and the fast-division kernargs.

    cd dnn-providers/hip-kernel-provider/rocke
    PYTHONPATH=$PWD/platform/python:$PWD/library python \\
        ../src/tests/engines/kernel_ingestor_engine/packs/gen_attention_dense_persistent_goldens.py [--check]

--check writes nothing and exits 1 when the checked-in table differs from the rendered one;
IngestorGenerator's tests/test_attention_dense_goldens.py runs the same comparison.
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    attention_dense_grid,
    attention_dense_runtime_args,
    attention_dense_signature,
)
from rocke.helpers.magic_division import calculate_magic_numbers

OUT = Path(__file__).with_name("TestAttentionDensePersistentGoldens.inc")
NUM_PERSISTENT = 304
# (name, persistent, decode field value)
GRIDS = (
    ("qb_minor", False, "qb_minor"),
    ("bt_hkv_minor", False, "bt_hkv_minor"),
    ("hq_minor_swz", False, "hq_minor_swz"),
    ("persistent/qb_major", True, "qb_major"),
    ("persistent/hkv_major", True, "hkv_major"),
    ("persistent/bt_hkv_minor", True, "bt_hkv_minor"),
    ("persistent/hq_minor_swz", True, "hq_minor_swz"),
)


def spec(grid, *, batch, sq, skv, hq, hkv, bm, causal, window, lse=False):
    _, persistent, decode = grid
    kw = dict(
        batch=batch,
        seqlen_q=sq,
        seqlen_kv=skv,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=128,
        causal=causal,
        dtype="bf16",
        block_m=bm,
        block_n=64,
        sliding_window=window,
        emit_lse=lse,
        persistent=persistent,
        num_persistent=NUM_PERSISTENT,
    )
    if persistent:
        kw["persist_decode"] = decode
    else:
        kw["nonpersist_decode"] = decode
    return Gfx942AttentionDenseSpec(**kw)


def magic_rows():
    divisors = list(range(1, 4097))
    rng = random.Random(942)
    divisors += sorted(rng.randrange(4097, 1 << 20) for _ in range(512))
    rows = []
    for d in divisors:
        magic, shift = calculate_magic_numbers(d)
        rows.append((d, magic - (1 << 32) if magic >= (1 << 31) else magic, shift))
    return rows


def signature_rows():
    rows = []
    for grid in GRIDS:
        for lse in (False, True):
            s = spec(
                grid,
                batch=1,
                sq=512,
                skv=512,
                hq=8,
                hkv=8,
                bm=256,
                causal=True,
                window=0,
                lse=lse,
            )
            rows.append(
                (
                    grid[0],
                    lse,
                    " ".join(a["name"] for a in attention_dense_signature(s)),
                )
            )
    return rows


SHAPES = (
    # batch, sq, skv, hq, hkv, bm, causal, window
    *itertools.product(
        (1, 7),
        (256, 768, 4608),
        (512, 4608),
        ((8, 8), (12, 4), (16, 2), (40, 8), (28, 4), (32, 32)),
        (256,),
        (False, True),
        (0, 512),
    ),
)


def grid_rows():
    rows = []
    for batch, sq, skv, (hq, hkv), bm, causal, window in SHAPES:
        if window and not causal:
            continue
        shape = (batch, sq, skv, hq, hkv, bm, causal, window)
        for grid in GRIDS:
            rows.append((grid[0], shape, *verdict(grid, shape)))
    # The dividend bounds: bt_hkv_minor's gqa * Hq at extreme GQA, and the work count.
    for shape in (
        (1, 256, 256, 32768, 1, 256, True, 0),
        # gqa * Hq = 32760^2 is in (2^29, 2^30), so only a 2^30 bound admits it.
        (1, 256, 256, 32760, 1, 256, True, 0),
        (1, 256, 256, 16384, 1, 256, True, 0),
        (4096, 32768, 32768, 64, 64, 256, True, 0),
        (2048, 32768, 32768, 64, 64, 256, False, 0),
    ):
        for grid in GRIDS:
            rows.append((grid[0], shape, *verdict(grid, shape)))
    return rows


def verdict(grid, shape):
    batch, sq, skv, hq, hkv, bm, causal, window = shape
    try:
        s = spec(
            grid,
            batch=batch,
            sq=sq,
            skv=skv,
            hq=hq,
            hkv=hkv,
            bm=bm,
            causal=causal,
            window=window,
        )
        args = attention_dense_runtime_args(s)
    except ValueError:
        return False, (0, 0, 0), ()
    pairs = tuple(v for k, v in args.items() if k.endswith(("_magic", "_shift")))
    return True, attention_dense_grid(s), pairs


def render() -> str:
    """The .inc text for the current rocKE launch-grid helpers."""
    lines = [
        "// Copyright © Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier:  MIT",
        "",
        "// Generated by gen_attention_dense_persistent_goldens.py from rocKE. Do not edit.",
        "// clang-format off",
        "",
    ]
    magic = magic_rows()
    signatures = signature_rows()
    grids = grid_rows()
    lines.append(
        f"inline constexpr std::array<MagicGolden, {len(magic)}> MAGIC_GOLDENS{{{{"
    )
    rows = [f"{{{d}, {m}, {s}}}," for d, m, s in magic]
    lines += ["    " + " ".join(rows[i : i + 6]) for i in range(0, len(rows), 6)]
    lines += [
        "}};",
        "",
        f"inline constexpr std::array<SignatureGolden, {len(signatures)}> SIGNATURE_GOLDENS{{{{",
    ]
    lines += [
        f'    {{"{g}", {"true" if lse else "false"}, "{names}"}},'
        for g, lse, names in signatures
    ]
    lines += [
        "}};",
        "",
        f"inline constexpr std::array<GridGolden, {len(grids)}> GRID_GOLDENS{{{{",
    ]
    for g, shape, legal, dims, pairs in grids:
        batch, sq, skv, hq, hkv, bm, causal, window = shape
        padded = list(pairs) + [0] * (6 - len(pairs))
        lines.append(
            f'    {{"{g}", {batch}, {sq}, {skv}, {hq}, {hkv}, {bm}, '
            f'{"true" if causal else "false"}, {window}, {"true" if legal else "false"}, '
            f"{dims[0]}, {dims[1]}, {dims[2]}, {len(pairs)}, "
            f"{{{', '.join(str(v) for v in padded)}}}}},"
        )
    lines += ["}};", "", "// clang-format on", ""]
    return "\n".join(lines)


def main(argv=None):
    check = "--check" in (sys.argv[1:] if argv is None else argv)
    text = render()
    if check:
        if OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.name} is stale; rerun {Path(__file__).name}", file=sys.stderr)
            return 1
        print(f"{OUT.name} is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
