# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""C++ <-> Python dispatcher selection-parity check for the ck-dsl conv family.

For every conv problem in the corpus this runs the Python conv dispatcher
(``ck_dsl.dispatch.families.conv.dispatch_conv``) and compares its selected
candidate against the C++ pick read from the JSONL emitted by ``cpp_select``
(running the REAL ``ck_dsl::Dispatcher`` over a synthesized shape-generic conv
bundle). Identity compared: (block_m, block_n, block_k, pipeline, epilogue).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ck_dsl.dispatch.families.conv import ConvRequest, dispatch_conv


def python_pick(prob, arch, dtype):
    try:
        r = dispatch_conv(
            ConvRequest(
                N=prob["N"],
                C=prob["C"],
                K=prob["K"],
                Hi=prob["Hi"],
                Wi=prob["Wi"],
                Y=prob["Y"],
                X=prob["X"],
                pad_h=prob["pad_h"],
                pad_w=prob["pad_w"],
                stride_h=prob["stride_h"],
                stride_w=prob["stride_w"],
                arch=arch,
                dtype=dtype,
            )
        )
    except ValueError as e:
        return None, str(e)
    return (
        {
            "selected": True,
            "kernel_name": r.spec.kernel_name(),
            "candidate": r.candidate.name,
            "spec_id": r.candidate.spec_id,
            "block_m": r.spec.tile_m,
            "block_n": r.spec.tile_n,
            "block_k": r.spec.tile_k,
            "pipeline": r.spec.pipeline,
            "epilogue": r.spec.epilogue,
        },
        None,
    )


def structural_key(rec):
    return (
        int(rec["block_m"]),
        int(rec["block_n"]),
        int(rec["block_k"]),
        str(rec.get("pipeline", "")),
        str(rec.get("epilogue", "")),
    )


def parse_shapes(path):
    shapes = []
    fields = ["N", "C", "K", "Hi", "Wi", "Y", "X"]
    with open(path) as f:
        for line in f:
            h = line.find("#")
            if h != -1:
                line = line[:h]
            parts = line.split()
            if len(parts) < 7:
                continue
            d = {k: int(parts[i]) for i, k in enumerate(fields)}
            opt = [0, 0, 1, 1]  # pad_h pad_w stride_h stride_w
            for i in range(4):
                if len(parts) > 7 + i:
                    opt[i] = int(parts[7 + i])
            d["pad_h"], d["pad_w"], d["stride_h"], d["stride_w"] = opt
            shapes.append(d)
    return shapes


def keyof(p):
    return (
        p["N"],
        p["C"],
        p["K"],
        p["Hi"],
        p["Wi"],
        p["Y"],
        p["X"],
        p["pad_h"],
        p["pad_w"],
        p["stride_h"],
        p["stride_w"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cpp-jsonl", default=str(here / "cpp_picks_conv.jsonl"))
    ap.add_argument("--shapes", default=str(here / "shapes_conv.txt"))
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()

    cpp_by = {}
    with open(args.cpp_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cpp_by[keyof(r)] = r

    shapes = parse_shapes(args.shapes)
    total = matches = 0
    divergences = []
    rows = []

    for prob in shapes:
        cpp = cpp_by.get(keyof(prob))
        total += 1
        if cpp is None:
            divergences.append({"shape": prob, "reason": "missing C++ pick"})
            continue
        py, py_err = python_pick(prob, args.arch, args.dtype)
        cpp_sel = bool(cpp.get("selected"))
        py_sel = py is not None
        tag = f"N{prob['N']}C{prob['C']}K{prob['K']}Y{prob['Y']}"
        if not cpp_sel and not py_sel:
            matches += 1
            rows.append((tag, "MATCH(no-kernel)", "", ""))
            continue
        if cpp_sel != py_sel:
            divergences.append(
                {
                    "shape": prob,
                    "reason": "selection-existence mismatch",
                    "cpp_selected": cpp_sel,
                    "py_selected": py_sel,
                    "py_error": py_err,
                }
            )
            rows.append((tag, "DIVERGE(exists)", "", ""))
            continue
        if structural_key(cpp) == structural_key(py):
            matches += 1
            rows.append(
                (
                    tag,
                    "MATCH",
                    f"{cpp['block_m']}x{cpp['block_n']}x{cpp['block_k']}/{cpp['pipeline']}/{cpp['epilogue']}",
                    f"{py['block_m']}x{py['block_n']}x{py['block_k']}/{py['pipeline']}/{py['epilogue']}",
                )
            )
        else:
            divergences.append(
                {"shape": prob, "reason": "structural mismatch", "cpp": cpp, "py": py}
            )
            rows.append(
                (
                    tag,
                    "DIVERGE",
                    f"{cpp['block_m']}x{cpp['block_n']}x{cpp['block_k']}/{cpp['pipeline']}/{cpp['epilogue']}",
                    f"{py['block_m']}x{py['block_n']}x{py['block_k']}/{py['pipeline']}/{py['epilogue']}",
                )
            )

    print("=" * 92)
    print(
        f"ck-dsl conv dispatcher C++<->Python selection parity "
        f"(arch={args.arch}, dtype={args.dtype})"
    )
    print("  identity compared: (block_m, block_n, block_k, pipeline, epilogue)")
    print("=" * 92)
    hdr = f"{'shape':<22} {'verdict':<16} {'C++ pick':<28} {'Python pick':<28}"
    print(hdr)
    print("-" * len(hdr))
    for tag, verdict, cpp_s, py_s in rows:
        print(f"{tag:<22} {verdict:<16} {cpp_s:<28} {py_s:<28}")
    print("-" * len(hdr))
    rate = (matches / total * 100.0) if total else 0.0
    print(f"match rate: {matches}/{total} = {rate:.1f}%")
    if divergences:
        print("\nDIVERGENCES:")
        print(json.dumps(divergences, indent=2))
    else:
        print(
            "\nNo divergences. C++ and Python pick the same conv kernel for every shape."
        )
    return 0 if not divergences else 1


if __name__ == "__main__":
    sys.exit(main())
