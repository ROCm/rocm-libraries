# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""C++ <-> Python dispatcher selection-parity check for the fused-MoE family.

For every MoE problem in the corpus this runs the Python MoE dispatcher and
compares its selected element-path mega-kernel against the C++ pick. Identity
compared: (path, tile_m, tile_n_inter, tile_k_gu, atom_k).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ck_dsl.dispatch.families.moe import MoeRequest, _struct, dispatch_moe


def python_pick(p, arch):
    try:
        r = dispatch_moe(
            MoeRequest(
                num_tokens=p["num_tokens"],
                hidden=p["hidden"],
                intermediate=p["intermediate"],
                num_experts=p["num_experts"],
                top_k=p["top_k"],
                dtype=p["dtype"],
                arch=arch,
            )
        )
    except ValueError as e:
        return None, str(e)
    s = _struct(r.spec)
    s["selected"] = True
    return s, None


def key(p):
    return (
        p["num_tokens"],
        p["hidden"],
        p["intermediate"],
        p["num_experts"],
        p["top_k"],
        p["dtype"],
    )


def parse_shapes(path):
    out = []
    names = ["num_tokens", "hidden", "intermediate", "num_experts", "top_k"]
    with open(path) as f:
        for line in f:
            h = line.find("#")
            if h != -1:
                line = line[:h]
            parts = line.split()
            if len(parts) < 6:
                continue
            d = {n: int(parts[i]) for i, n in enumerate(names)}
            d["dtype"] = parts[5]
            out.append(d)
    return out


def struct_key(rec):
    return (
        str(rec["path"]),
        int(rec["tile_m"]),
        int(rec["tile_n_inter"]),
        int(rec["tile_k_gu"]),
        int(rec["atom_k"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cpp-jsonl", default=str(here / "cpp_picks_moe.jsonl"))
    ap.add_argument("--shapes", default=str(here / "shapes_moe.txt"))
    ap.add_argument("--arch", default="gfx950")
    args = ap.parse_args()

    cpp_by = {}
    with open(args.cpp_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cpp_by[key(r)] = r

    shapes = parse_shapes(args.shapes)
    total = matches = 0
    divergences = []
    rows = []

    for p in shapes:
        cpp = cpp_by.get(key(p))
        total += 1
        if cpp is None:
            divergences.append({"shape": p, "reason": "missing C++ pick"})
            continue
        py, py_err = python_pick(p, args.arch)
        cpp_sel = bool(cpp.get("selected"))
        py_sel = py is not None
        tag = f"t{p['num_tokens']}h{p['hidden']}i{p['intermediate']}/{p['dtype']}"
        if not cpp_sel and not py_sel:
            matches += 1
            rows.append((tag, "MATCH(no-kernel)", "", ""))
            continue
        if cpp_sel != py_sel:
            divergences.append(
                {
                    "shape": p,
                    "reason": "selection-existence mismatch",
                    "cpp_selected": cpp_sel,
                    "py_selected": py_sel,
                    "py_error": py_err,
                }
            )
            rows.append((tag, "DIVERGE(exists)", "", ""))
            continue
        if struct_key(cpp) == struct_key(py):
            matches += 1
            rows.append(
                (
                    tag,
                    "MATCH",
                    f"{cpp['path']}/m{cpp['tile_m']}n{cpp['tile_n_inter']}/ak{cpp['atom_k']}",
                    f"{py['path']}/m{py['tile_m']}n{py['tile_n_inter']}/ak{py['atom_k']}",
                )
            )
        else:
            divergences.append(
                {"shape": p, "reason": "structural mismatch", "cpp": cpp, "py": py}
            )
            rows.append((tag, "DIVERGE", str(struct_key(cpp)), str(struct_key(py))))

    print("=" * 92)
    print(f"ck-dsl MoE dispatcher C++<->Python selection parity (arch={args.arch})")
    print("  identity compared: (path, tile_m, tile_n_inter, tile_k_gu, atom_k)")
    print("=" * 92)
    hdr = f"{'shape':<32} {'verdict':<16} {'C++ pick':<22} {'Python pick':<22}"
    print(hdr)
    print("-" * len(hdr))
    for tag, verdict, cpp_s, py_s in rows:
        print(f"{tag:<32} {verdict:<16} {cpp_s:<22} {py_s:<22}")
    print("-" * len(hdr))
    rate = (matches / total * 100.0) if total else 0.0
    print(f"match rate: {matches}/{total} = {rate:.1f}%")
    if divergences:
        print("\nDIVERGENCES:")
        print(json.dumps(divergences, indent=2))
    else:
        print(
            "\nNo divergences. C++ and Python pick the same MoE kernel for every shape."
        )
    return 0 if not divergences else 1


if __name__ == "__main__":
    sys.exit(main())
