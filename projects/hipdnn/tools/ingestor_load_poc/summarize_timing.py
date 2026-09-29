# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Collapse per-rep plugin_load_timing JSON lines into one row per tree.

Input lines are {"tree": ..., "rep": ..., <PluginLoadTiming fields>}. Best-of-N is the
minimum of each metric over the reps; the median is printed beside it so a noisy host is
visible rather than hidden. Fails if any rep disagrees on the admitted engine list or
reports a nonzero status.

Usage: python summarize_timing.py <raw.jsonl> > rows.json
"""
import json
import statistics
import sys
from collections import OrderedDict

METRICS = ("discover_ms", "create_ms", "hwm_create_mib")


def main(path):
    runs = OrderedDict()
    for line in open(path):
        line = line.strip()
        if line:
            rec = json.loads(line)
            runs.setdefault(rec["tree"], []).append(rec)
    rows = []
    for tree, reps in runs.items():
        names = {tuple(r["engine_names"]) for r in reps}
        if len(names) != 1:
            sys.exit(f"{tree}: reps disagree on the engine list: {names}")
        if any(r["rc_ids"] != 0 or r["rc_create"] != 0 for r in reps):
            sys.exit(f"{tree}: a rep reported a nonzero plugin status")
        row = {"tree": tree, "reps": len(reps), "engines": reps[0]["engines"]}
        for m in METRICS:
            values = [r[m] for r in reps]
            row[m] = round(min(values), 2)
            row[m + "_median"] = round(statistics.median(values), 2)
        row["startup_ms"] = round(row["discover_ms"] + row["create_ms"], 2)
        row["engine_names"] = list(next(iter(names)))
        rows.append(row)
    json.dump(rows, sys.stdout, indent=1)
    print()
    for row in rows:
        print(
            f"{row['tree']:>8}  discover {row['discover_ms']:>10.2f} ms"
            f" (med {row['discover_ms_median']:.2f})  create {row['create_ms']:>9.2f} ms"
            f"  startup {row['startup_ms']:>10.2f} ms  peak {row['hwm_create_mib']:>8.1f} MiB"
            f"  engines {row['engines']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main(sys.argv[1])
