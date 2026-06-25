# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""C++ <-> Python dispatcher selection-parity check for the ck-dsl norm family.

For every (rows, cols, kind) in the corpus this script runs the Python norm
dispatcher (``ck_dsl.dispatch.families.norm.dispatch_norm``) and compares its
selected candidate against the C++ pick read from the JSONL emitted by the
compiled ``cpp_select`` harness (which ran the REAL ``ck_dsl::Dispatcher`` over a
synthesized manifest-only norm bundle).

Norm kernels are not MMA-bound, so the shared structural identity is the
occupancy/vector tuple that determines the kernel:

    (block_size, vec, kind)

We compare on that. As with the GEMM check, the raw kernel_name prefixes differ
by construction, so we do NOT compare names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ck_dsl.dispatch.families.norm import NormRequest, dispatch_norm


def python_pick(rows: int, cols: int, kind: str, arch: str, dtype: str):
    try:
        r = dispatch_norm(
            NormRequest(rows=rows, cols=cols, kind=kind, arch=arch, dtype=dtype)
        )
    except ValueError as e:
        return None, str(e)
    return (
        {
            "selected": True,
            "kernel_name": r.spec.kernel_name(),
            "candidate": r.candidate.name,
            "spec_id": r.candidate.spec_id,
            "block_size": int(r.spec.block_size),
            "vec": int(r.spec.vec),
            "kind": kind,
        },
        None,
    )


def structural_key(block_size, vec, kind):
    return (int(block_size), int(vec), str(kind))


def parse_shapes(path):
    shapes = []
    with open(path) as f:
        for line in f:
            h = line.find("#")
            if h != -1:
                line = line[:h]
            parts = line.split()
            if len(parts) < 2:
                continue
            rows, cols = int(parts[0]), int(parts[1])
            kind = parts[2] if len(parts) >= 3 else "rmsnorm"
            shapes.append((rows, cols, kind))
    return shapes


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cpp-jsonl", default=str(here / "cpp_picks_norm.jsonl"))
    ap.add_argument("--shapes", default=str(here / "shapes_norm.txt"))
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    cpp_by_shape = {}
    with open(args.cpp_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cpp_by_shape[(rec["rows"], rec["cols"], rec["kind"])] = rec

    shapes = parse_shapes(args.shapes)

    total = matches = 0
    divergences = []
    rows_out = []

    for rows, cols, kind in shapes:
        cpp = cpp_by_shape.get((rows, cols, kind))
        total += 1
        if cpp is None:
            divergences.append(
                {"shape": [rows, cols, kind], "reason": "missing C++ pick"}
            )
            continue

        py, py_err = python_pick(rows, cols, kind, args.arch, args.dtype)
        cpp_selected = bool(cpp.get("selected"))
        py_selected = py is not None

        if not cpp_selected and not py_selected:
            matches += 1
            rows_out.append((rows, cols, kind, "MATCH(no-kernel)", "", ""))
            continue
        if cpp_selected != py_selected:
            divergences.append(
                {
                    "shape": [rows, cols, kind],
                    "reason": "selection-existence mismatch",
                    "cpp_selected": cpp_selected,
                    "py_selected": py_selected,
                    "py_error": py_err,
                    "cpp_kernel": cpp.get("kernel_name"),
                }
            )
            rows_out.append((rows, cols, kind, "DIVERGE(exists)", "", ""))
            continue

        ck = structural_key(cpp["block_m"], cpp["block_n"], kind)
        pk = structural_key(py["block_size"], py["vec"], py["kind"])
        if ck == pk:
            matches += 1
            rows_out.append(
                (
                    rows,
                    cols,
                    kind,
                    "MATCH",
                    f"b{cpp['block_m']}/v{cpp['block_n']}",
                    f"b{py['block_size']}/v{py['vec']}",
                )
            )
        else:
            divergences.append(
                {
                    "shape": [rows, cols, kind],
                    "reason": "structural-identity mismatch",
                    "cpp": {"block_size": cpp["block_m"], "vec": cpp["block_n"]},
                    "py": {
                        "candidate": py["candidate"],
                        "block_size": py["block_size"],
                        "vec": py["vec"],
                    },
                }
            )
            rows_out.append(
                (
                    rows,
                    cols,
                    kind,
                    "DIVERGE",
                    f"b{cpp['block_m']}/v{cpp['block_n']}",
                    f"b{py['block_size']}/v{py['vec']}",
                )
            )

    print("=" * 92)
    print(
        f"ck-dsl norm dispatcher C++<->Python selection parity "
        f"(arch={args.arch}, dtype={args.dtype})"
    )
    print("  identity compared: (block_size, vec, kind)")
    print("=" * 92)
    hdr = (
        f"{'rows':>6} {'cols':>6} {'kind':<10}  {'verdict':<16} "
        f"{'C++ pick':<16} {'Python pick':<16}"
    )
    print(hdr)
    print("-" * len(hdr))
    for rows, cols, kind, verdict, cpp_s, py_s in rows_out:
        print(f"{rows:>6} {cols:>6} {kind:<10}  {verdict:<16} {cpp_s:<16} {py_s:<16}")
    print("-" * len(hdr))
    rate = (matches / total * 100.0) if total else 0.0
    print(f"match rate: {matches}/{total} = {rate:.1f}%")
    if divergences:
        print("\nDIVERGENCES:")
        print(json.dumps(divergences, indent=2))
    else:
        print(
            "\nNo divergences. C++ and Python pick the same norm kernel for every shape."
        )
    return 0 if not divergences else 1


if __name__ == "__main__":
    sys.exit(main())
