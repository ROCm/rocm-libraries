#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare two runner --export-graphs directories (for example gfx942 and gfx950).

Reports two results per case_key and in total:
  raw        the .json and .meta.json files are byte-identical
  canonical  the graphs are equal after the import's UID canonicalisation by tensor name
             (with the meta inputs remapped the same way)
The frontend assigns UIDs to tensors that the graph builder leaves unset (the SDPA
outputs), so a raw difference with a canonical match is a UID difference only.
Also compares the outcome JSONL files when --outcomes A B is given.
Exit 0 only when every case matches canonically and both sides hold the same case_keys.
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "migration-scripts"))
from bundle_utils import (
    canon,
    canonical_uid_map_by_name,
    remap_graph,
    remap_meta_inputs,
)  # noqa: E402


def cases(d):
    return {
        p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.json").is_file()
    }


def canonical(d, key):
    g = json.loads((d / key / f"{key}.json").read_text())
    m = json.loads((d / key / f"{key}.meta.json").read_text())
    uid_map = canonical_uid_map_by_name(g)
    return canon(remap_graph(g, uid_map)), canon(remap_meta_inputs(m, uid_map))


def outcomes(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line) if line.strip() else {}
        if rec.get("record") == "export":
            out[rec["case_key"]] = rec["outcome"]
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--outcomes", nargs=2, type=Path, metavar=("A_JSONL", "B_JSONL"))
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    ka, kb = cases(args.a), cases(args.b)
    common = sorted(ka & kb)
    raw_diff, canon_diff, remap_errors = [], [], []
    for key in common:
        same_raw = all(
            (args.a / key / n).read_bytes() == (args.b / key / n).read_bytes()
            for n in (f"{key}.json", f"{key}.meta.json")
        )
        if not same_raw:
            raw_diff.append(key)
            try:
                same_canonical = canonical(args.a, key) == canonical(args.b, key)
            except ValueError as e:
                remap_errors.append(f"{key}: {e}")
                same_canonical = False
            if not same_canonical:
                canon_diff.append(key)
    report = {
        "a": str(args.a),
        "b": str(args.b),
        "cases_a": len(ka),
        "cases_b": len(kb),
        "only_in_a": sorted(ka - kb)[:50],
        "only_in_b": sorted(kb - ka)[:50],
        "compared": len(common),
        "raw_identical": len(common) - len(raw_diff),
        "raw_different": len(raw_diff),
        "canonical_identical": len(common) - len(canon_diff),
        "canonical_different": len(canon_diff),
        "raw_different_examples": raw_diff[:20],
        "canonical_different_examples": canon_diff[:20],
        "remap_errors": remap_errors[:20],
    }
    if args.outcomes:
        oa, ob = outcomes(args.outcomes[0]), outcomes(args.outcomes[1])
        report["outcome_differences"] = sorted(
            k for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)
        )[:50]
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    ok = not canon_diff and ka == kb and not report.get("outcome_differences")
    print(
        f"compare_exports: {len(common)} compared; raw identical {report['raw_identical']},"
        f" raw different {len(raw_diff)}; canonical different {len(canon_diff)}"
        f" ({len(remap_errors)} do not remap);"
        f" only in a {len(ka - kb)}, only in b {len(kb - ka)}; {'OK' if ok else 'DIFFERENT'}",
        file=sys.stderr,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
