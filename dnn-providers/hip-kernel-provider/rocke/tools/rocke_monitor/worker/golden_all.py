# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check every representative-IR harness case against the tree's golden, per flavor.
Usage: python golden_all.py <rocke_root> [flavors=llvm20,llvm22,llvm23]"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
flavors = (sys.argv[2] if len(sys.argv) > 2 else "llvm20,llvm22,llvm23").split(",")
sys.path[:0] = [
    str(root / "platform/python"),
    str(root / "library"),
    str(root / "platform/tests/instances"),
]
import rocke_ir_parity_harness as h  # noqa: E402

# One file, or one shard per family in a directory of that name; the tree's own loader merges them.
gdir = root / "platform/tests/golden/rocke_representative_ir_sha256"
if gdir.is_dir():
    from rocke.core.ir_golden import (
        load_golden,
    )  # the tree under test is first on sys.path

    golden = load_golden(gdir)
else:
    golden = json.loads(gdir.with_suffix(".json").read_text())
cases = h.cases()
print("cases:", len(cases), flush=True)
for fl in flavors:
    g = golden["flavors"][fl]["cases"]
    drift, ok, missing = [], 0, []
    for c in cases:
        cid = c["case_id"]
        try:
            rec, _ = h.lower_case(c, fl)
        except Exception as e:  # report, keep going
            drift.append(f"{cid}: {type(e).__name__}: {str(e)[:120]}")
            continue
        exp = g.get(cid)
        if exp is None:
            missing.append(cid)
        elif exp["sha256"] != rec["sha256"]:
            drift.append(f"{cid}: got {rec['sha256'][:12]} exp {exp['sha256'][:12]}")
        else:
            ok += 1
    stale = sorted(set(g) - {c["case_id"] for c in cases})
    print(
        f"{fl}: ok {ok} drift {len(drift)} missing {len(missing)} stale {len(stale)}",
        flush=True,
    )
    for d in drift[:40]:
        print("   DRIFT", d)
    for m in missing[:20]:
        print("   MISSING", m)
