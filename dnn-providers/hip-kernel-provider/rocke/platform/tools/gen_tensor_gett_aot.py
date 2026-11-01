#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write (or check) python/rocke/dispatch/families/tensor_gett_aot.json, the tensor GETT AOT route
table: per problem, the route and GEMM body of the product route ``plan_tensor_gett``, frozen for
ahead-of-time kernel packs (hipTensor's rocKE pack builds one launch program per entry).

  python tools/gen_tensor_gett_aot.py --add problems.json   # add problems (JSON list, see below)
  python tools/gen_tensor_gett_aot.py --refresh             # recompute every entry
  python tools/gen_tensor_gett_aot.py --check               # exit 1 when an entry is stale

A problem is ``{"modes_a": [...], "modes_b": [...], "modes_d": [...], "extents": {mode: n},
"dtype": "fp16" | "bf16", "bilinear": bool}`` with row-major mode lists (the last mode is the
fastest; a hipTensor column-major descriptor's mode list reversed) and packed tensors. Problems
with an entry in the tuned route table (``tensor_gett_tuned.json``) are left out: the product
route already takes the tuned route there, and consumers read both tables. Problems whose product
route runs no GEMM body (the ring-body fallback) are left out and reported.

Set ``ROCKE_LLVM_FLAVOR`` when no COMGR is loadable (no IR is lowered; the dispatcher only needs
the flavor name).
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

# the product route with its shipped defaults: no knob or selector override from the environment
# (the variables hipTensor's reference pack writer scrubs, and the GETT route policy)
for _k in list(os.environ):
    if _k.startswith(
        (
            "ROCKE_EXP_",
            "CK_",
            "ROCKE_FP8_",
            "ROCKE_GEMM_",
            "TENSOR_GETT_",
            "ROCKE_TENSOR_GETT_",
        )
    ):
        del os.environ[_k]

from rocke.dispatch.families import tensor_gett as TG  # noqa: E402
from rocke.instances.common.tensor import GettKnobs, TensorGettSpec  # noqa: E402

OUT = (
    HERE.parent / "python" / "rocke" / "dispatch" / "families" / TG.TENSOR_GETT_AOT_FILE
)


def request(problem: dict, arch: str) -> "TG.TensorGettRequest":
    spec = TensorGettSpec.make(
        problem["modes_a"],
        problem["modes_b"],
        problem["modes_d"],
        {m: int(e) for m, e in problem["extents"].items()},
        dtype="fp16" if problem["dtype"] in ("f16", "fp16") else problem["dtype"],
        knobs=GettKnobs(bilinear=bool(problem["bilinear"])),
    )
    return TG.TensorGettRequest(problem=spec, arch=arch)


def render(entries: dict) -> str:
    return (
        json.dumps({"entries": entries, "schema": 1}, indent=1, sort_keys=True) + "\n"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument(
        "--add", metavar="PROBLEMS_JSON", help="JSON list of problems to add"
    )
    ap.add_argument(
        "--refresh", action="store_true", help="recompute every existing entry"
    )
    ap.add_argument(
        "--check", action="store_true", help="exit 1 when an entry is stale"
    )
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    path = Path(a.out)
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    entries = {k: dict(v) for k, v in raw.get("entries", {}).items()}
    table = dict(entries.get(a.arch, {}))
    tuned = TG._SHIPPED_TUNED.get(a.arch, {})
    stale = 0
    if a.refresh or a.check:
        for key, ent in sorted(table.items()):
            got = TG.tensor_gett_aot_entry(request(ent["problem"], a.arch))
            if got is None or got[0] != key or got[1] != ent:
                stale += 1
                print(f"stale: {key}: {ent['route']} {ent['body']} -> {got and got[1]}")
                if a.refresh:
                    del table[key]
                    if got is not None and got[0] not in tuned:
                        table[got[0]] = got[1]
    if a.add:
        for prob in json.loads(Path(a.add).read_text(encoding="utf-8")):
            req = request(prob, a.arch)
            key = TG.tensor_gett_route_key(req)
            if key in tuned:
                print(f"skip {key}: in the tuned route table")
                continue
            got = TG.tensor_gett_aot_entry(req)
            if got is None:
                print(f"skip {key}: the product route runs no GEMM body")
                continue
            table[got[0]] = got[1]
    if a.check:
        print(f"{len(table)} entries, {stale} stale")
        return 1 if stale else 0
    entries[a.arch] = table
    text = render(entries)
    path.write_bytes(text.encode())
    print(f"wrote {path} ({len(table)} {a.arch} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
