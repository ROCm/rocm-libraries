# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""C++ <-> Python dispatcher selection-parity check for the attention family.

For every attention problem in the corpus this runs the Python attention
dispatcher (``ck_dsl.dispatch.families.attention.dispatch_attention``) and
compares its selected PATH against the C++ pick. Identity compared:
(path, head_size, block_size). The arch-tuned CTA geometry is deferred (see the
attention family module docstring) and is NOT part of this identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ck_dsl.dispatch.families.attention import AttentionRequest, dispatch_attention


def python_pick(p, arch, dtype):
    try:
        r = dispatch_attention(
            AttentionRequest(
                batch=p["batch"],
                nhead_q=p["nhead_q"],
                nhead_k=p["nhead_k"],
                seqlen_q=p["seqlen_q"],
                seqlen_k=p["seqlen_k"],
                hdim_q=p["hdim"],
                hdim_v=p["hdim"],
                sliding_window=p["sliding_window"],
                kv_block_size=p["block_kv"],
                num_sms=p["num_sms"],
                arch=arch,
                dtype=dtype,
            )
        )
    except ValueError as e:
        return None, str(e)
    return (
        {
            "selected": True,
            "path": r.spec.path,
            "head_size": r.spec.head_size,
            "block_size": r.spec.block_size,
        },
        None,
    )


def key(p):
    return (
        p["batch"],
        p["nhead_q"],
        p["nhead_k"],
        p["seqlen_q"],
        p["seqlen_k"],
        p["hdim"],
        p["sliding_window"],
        p["block_kv"],
        p["num_sms"],
    )


def parse_shapes(path):
    out = []
    names = ["batch", "nhead_q", "nhead_k", "seqlen_q", "seqlen_k", "hdim"]
    with open(path) as f:
        for line in f:
            h = line.find("#")
            if h != -1:
                line = line[:h]
            parts = line.split()
            if len(parts) < 6:
                continue
            d = {n: int(parts[i]) for i, n in enumerate(names)}
            opt = [0, 16, 120]  # sliding_window block_kv num_sms
            for i in range(3):
                if len(parts) > 6 + i:
                    opt[i] = int(parts[6 + i])
            d["sliding_window"], d["block_kv"], d["num_sms"] = opt
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cpp-jsonl", default=str(here / "cpp_picks_attention.jsonl"))
    ap.add_argument("--shapes", default=str(here / "shapes_attention.txt"))
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
        py, py_err = python_pick(p, args.arch, args.dtype)
        cpp_sel = bool(cpp.get("selected"))
        py_sel = py is not None
        tag = f"b{p['batch']}q{p['seqlen_q']}kv{p['seqlen_k']}hd{p['hdim']}sw{p['sliding_window']}"
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
        ck = (cpp["path"], int(cpp["head_size"]), int(cpp["block_size"]))
        pk = (py["path"], int(py["head_size"]), int(py["block_size"]))
        if ck == pk:
            matches += 1
            rows.append(
                (
                    tag,
                    "MATCH",
                    f"{cpp['path']}/hd{cpp['head_size']}",
                    f"{py['path']}/hd{py['head_size']}",
                )
            )
        else:
            divergences.append(
                {"shape": p, "reason": "path mismatch", "cpp": ck, "py": pk}
            )
            rows.append((tag, "DIVERGE", str(ck), str(pk)))

    print("=" * 92)
    print(
        f"ck-dsl attention dispatcher C++<->Python selection parity "
        f"(arch={args.arch}, dtype={args.dtype})"
    )
    print("  identity compared: (path, head_size, block_size)")
    print("=" * 92)
    hdr = f"{'shape':<34} {'verdict':<16} {'C++ pick':<18} {'Python pick':<18}"
    print(hdr)
    print("-" * len(hdr))
    for tag, verdict, cpp_s, py_s in rows:
        print(f"{tag:<34} {verdict:<16} {cpp_s:<18} {py_s:<18}")
    print("-" * len(hdr))
    rate = (matches / total * 100.0) if total else 0.0
    print(f"match rate: {matches}/{total} = {rate:.1f}%")
    if divergences:
        print("\nDIVERGENCES:")
        print(json.dumps(divergences, indent=2))
    else:
        print(
            "\nNo divergences. C++ and Python pick the same attention path for every shape."
        )
    return 0 if not divergences else 1


if __name__ == "__main__":
    sys.exit(main())
