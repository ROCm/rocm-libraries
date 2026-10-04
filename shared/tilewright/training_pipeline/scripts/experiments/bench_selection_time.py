#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Write the inputs file of bench_selection_time.cpp.

The kernel pool is every solution of the config's library
(`hipblaslt.library_stem`) in `--library-dir`, in the order hipBLASLt builds
it, converted the way training and the parity check convert it. The problem
data types and layout come from the config's `hipblaslt` block, the hardware
from its `hardware` block or the KFD sysfs topology of `--device`.

    PROBLEM  a_dtype b_dtype c_dtype d_dtype mi_dtype transA transB
    HARDWARE n_cu lds_bytes l2_bytes
    SHAPES   n        then n lines: M N K batch
    CONFIGS  n        then n lines: mt_m mt_n mt_k mi_m mi_n mi_k occupancy
                      cache_hints_a cache_hints_b grvw_a grvw_b gwvw_d

Data types are tilewright::DataType values; transposes are 0 for T, 1 for N.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import dat, evaluate  # noqa: E402
from lib import features as fs  # noqa: E402
from lib.hardware import DATATYPE_VALUE, device_hardware  # noqa: E402

FIELDS = (
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "occupancy",
    "cache_hints_a",
    "cache_hints_b",
    "grvw_a",
    "grvw_b",
    "gwvw_d",
)


def read_shapes(path: Path) -> List[Tuple[int, int, int, int]]:
    out = []
    for line in path.read_text().splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if len(parts) not in (3, 4):
            raise SystemExit(f"{path}: bad shape line {line!r}")
        m, n, k = (int(v) for v in parts[:3])
        out.append((m, n, k, int(parts[3]) if len(parts) == 4 else 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-yaml", type=Path, required=True)
    ap.add_argument("--library-dir", type=Path, required=True)
    ap.add_argument(
        "--library-stem", default=None, help="default: hipblaslt.library_stem"
    )
    ap.add_argument(
        "--shapes-file",
        type=Path,
        default=Path(__file__).with_name("bench_default_shapes.txt"),
        help="one 'M N K [batch]' per line",
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with args.config_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    hbl = cfg.get("hipblaslt") or {}
    stem = args.library_stem or hbl.get("library_stem")
    if not stem:
        raise SystemExit(
            "no library stem: set hipblaslt.library_stem or --library-stem"
        )
    pool = evaluate.pool_from_kernels(dat.load_library_kernels(args.library_dir, stem))
    row = dict(hbl, m=1, n=1, k=1, batch_count=1)
    prob = fs.problem_kwargs_from_row(row)
    dtypes = [
        DATATYPE_VALUE[fs.normalize_dtype_name(prob[f])]
        for f in ("a_dtype", "b_dtype", "c_dtype", "d_dtype", "mi_dtype")
    ]
    trans = [0 if str(hbl[f]).upper() == "T" else 1 for f in ("transA", "transB")]
    hw = device_hardware(cfg, args.device)
    shapes = read_shapes(args.shapes_file)

    lines = [
        "PROBLEM " + " ".join(str(v) for v in dtypes + trans),
        f"HARDWARE {hw.n_cu} {hw.lds_bytes} {hw.l2_bytes}",
        f"SHAPES {len(shapes)}",
    ]
    lines += [f"{m} {n} {k} {b}" for m, n, k, b in shapes]
    lines.append(f"CONFIGS {len(pool)}")
    lines += [" ".join(str(int(c[f])) for f in FIELDS) for c in pool]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(
        f"{args.out}: {len(shapes)} shapes, {len(pool)} kernels of {stem}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
