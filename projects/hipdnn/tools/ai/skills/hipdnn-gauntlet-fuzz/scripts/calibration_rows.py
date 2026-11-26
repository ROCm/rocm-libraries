#!/usr/bin/env python3
"""Pick an oracle-calibration sample: the fewest pairwise rows that show every axis value of a space at least once.

Run from dnn-providers/integration-tests of the fuzz checkout:
    python calibration_rows.py --filter '^SdpaBackward/FlydslBwdSplit' --out <dir> [--seed 1] [--data-seed 12345]
Writes <dir>/<Space>/cal<i>.json rows ({"space", "axes", "data_seed", "case_id"}), replayable with
`run_gauntlet_fuzz.py --row <row.json>` (add `--dump-dir <dir>` to dump the tensors). Rows with `placement: under` are skipped: they only check that
the backend rejects a misaligned pointer and compare nothing. Pairwise rows already hold every value of every axis, so a greedy set
cover over (axis, value) gives the extremes (largest scale, widest domain, biggest head dim, ...) in a handful of rows.
"""
import argparse, json, os, re, sys

sys.path.insert(0, "gauntlet")
import pairwise  # noqa: E402
import space  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--filter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--data-seed", type=int, default=12345)
a = ap.parse_args()
for sp in space.load_spaces():
    if not re.search(a.filter, sp.space_id) or sp.space_id.endswith("Probe"):
        continue
    rows = [r for r in pairwise.build(sp, a.seed).rows if r.get("placement") != "under"]
    key = lambda r: {(k, json.dumps(v, sort_keys=True)) for k, v in r.items()}
    todo = set().union(*(key(r) for r in rows)) if rows else set()
    picked = []
    while todo:
        best = max(rows, key=lambda r: len(key(r) & todo))
        gain = key(best) & todo
        if not gain:
            break
        picked.append(best)
        todo -= gain
    d = os.path.join(a.out, sp.space_id.split("/")[-1])
    os.makedirs(d, exist_ok=True)
    for i, r in enumerate(picked):
        json.dump({"space": sp.space_id, "axes": r, "data_seed": a.data_seed, "case_id": f"cal{i}"},
                  open(os.path.join(d, f"cal{i}.json"), "w"), indent=1)
    print(f"{sp.space_id}: {len(picked)} rows cover every axis value ({len(rows)} executing pairwise rows)")
