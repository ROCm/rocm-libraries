#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check the size of the harness files against their budgets, and estimate the tokens on common paths.

Usage:  python3 check-budget.py          (from anywhere)
Exit code: 0 = every file is within its budget, 1 = a file is over its budget.
Tokens are estimated at about 4 characters each. The budgets are listed in MAINTAINING.md.
"""

import pathlib
import sys

ST = pathlib.Path(__file__).resolve().parent.parent
HB = ST.parents[2] / "projects" / "hipblaslt" / "harness"
KB = 1024

# name -> (path, budget in bytes)
FILES = {
    "st/HARNESS.md": (ST / "HARNESS.md", int(4.5 * KB)),
    "hb/HARNESS.md": (HB / "HARNESS.md", 5 * KB),
    "st/ab.md": (ST / "ab.md", 5 * KB),
    "hb/ab.md": (HB / "ab.md", 5 * KB),
    "compare.md": (ST / "compare.md", 5 * KB),
    "state.md": (ST / "state.md", 5 * KB),
    "clone.md": (ST / "clone.md", 5 * KB),
    "DATABASE.md": (ST / "DATABASE.md", 7 * KB),
    "ENTRY-FORMAT.md": (ST / "ENTRY-FORMAT.md", 5 * KB),
    "PROFILING.md": (HB / "PROFILING.md", 5 * KB),
}

# a common path is the set of files an agent reads for one kind of task
PATHS = {
    "StinkyTofu: any task": ["st/HARNESS.md"],
    "StinkyTofu: verify a change": ["st/HARNESS.md", "st/ab.md", "compare.md"],
    "StinkyTofu: verify, database set": [
        "st/HARNESS.md",
        "st/ab.md",
        "compare.md",
        "DATABASE.md",
    ],
    "hipBLASLt: verify a change": ["hb/HARNESS.md", "hb/ab.md", "compare.md"],
    "hipBLASLt: verify, then perf A/B": [
        "hb/HARNESS.md",
        "hb/ab.md",
        "compare.md",
        "PROFILING.md",
    ],
}

over = 0
sizes = {}
print(f"{'file':18} {'bytes':>7} {'budget':>7} {'tokens':>7}")
for name, (path, budget) in FILES.items():
    if not path.exists():
        print(f"{name:18} missing: {path}")
        over += 1
        continue
    size = path.stat().st_size
    sizes[name] = size
    flag = "  OVER BUDGET" if size > budget else ""
    over += size > budget
    print(f"{name:18} {size:7d} {budget:7d} {size // 4:7d}{flag}")

print("\ncommon paths (estimated tokens, files read in full):")
for label, names in PATHS.items():
    total = sum(sizes.get(n, 0) for n in names)
    print(f"  {label:36} {total // 4:6d}")
sys.exit(1 if over else 0)
