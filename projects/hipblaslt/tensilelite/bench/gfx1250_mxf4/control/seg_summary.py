#!/usr/bin/env python3
"""Per-dispatch LDS counters for control/lds_segment*, joined with the probe's own
(dispatch, params...) stdout table. Usage: seg_summary.py <counter_collection.csv> <probe.log>"""
import collections
import csv
import sys

SEG = "TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES"
ACT = "TX_PERF_SEL_VMW_LDS_INPUT_ACTIVE"
BANK = "TX_PERF_SEL_VMW_LDS_BANK_CONFLICT"

counters = collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    counters[int(r["Dispatch_Id"])][r["Counter_Name"]] = float(r["Counter_Value"])

header, params = None, []
for line in open(sys.argv[2]):
    line = line.strip()
    if line.startswith("dispatch,"):
        header = line.split(",")[1:]
    elif header and line and line[0].isdigit() and line.count(",") == len(header):
        params.append(line.split(",")[1:])

ids = sorted(counters)
print(",".join(header + ["active", "bank", "seg", "seg/active"]))
for p, i in zip(params, ids):
    c = counters[i]
    act, seg = c.get(ACT, 0), c.get(SEG, 0)
    print(",".join(p + [f"{act:.0f}", f"{c.get(BANK, 0):.0f}", f"{seg:.0f}",
                        f"{seg / act:.3f}" if act else "nan"]))
