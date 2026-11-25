# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ABI history invariant for rocke/abi.h: each counter's #define equals the highest ` * N:` history entry above it,
and no number appears twice. Prints `OK ...` or `FAIL ...`. Usage: python3 -I abi_check.py <abi.h>
"""
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
problems, info = [], []
for name in ("ROCKE_ABI_VERSION", "ROCKE_RECIPE_ABI"):
    m = re.search(rf"#define\s+{name}\s+(\d+)", text)
    if not m:
        problems.append(f"{name} not defined")
        continue
    v = int(m.group(1))
    block = text[: m.start()].rsplit("/*", 1)[-1]
    nums = [int(n) for n in re.findall(r"^\s*\*?\s*(\d+):", block, re.M)]
    dup = sorted({n for n in nums if nums.count(n) > 1})
    if dup:
        problems.append(f"{name}: history number(s) {dup} appear more than once")
    if nums and max(nums) != v:
        problems.append(f"{name} is {v} but the newest history entry is {max(nums)}")
    if v > 1 and v not in nums:
        problems.append(f"{name} {v} has no history line")
    info.append(f"{name}={v}")
print(("FAIL " + "; ".join(problems)) if problems else ("OK " + " ".join(info)))
