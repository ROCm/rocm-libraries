# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Usage: python faildiff2.py <A/out.txt> <B/out.txt>: the A-only, B-only and common failure IDs of two pytest runs.
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import rkenv  # noqa: E402

# Failure IDs are compared relative to the watched folder's parent (RKMON_ROCKE_PATH, the job's or rk env's): a path
# from any tree through that parent loses the parent and what comes before it, so the same failure reads the same in two
# trees. A folder at the repository's root has no parent to pass through: its paths read from the folder on.
FOLDER = rkenv.get("RKMON_ROCKE_PATH").strip("/")
PARENT = FOLDER.rpartition("/")[0]
THROUGH = re.compile(r"(?<!\S)(?:\S*?/)?" + re.escape(f"{PARENT or FOLDER}/"))
KEPT = "" if PARENT else f"{FOLDER}/"


def f(p):
    out = set()
    for l in open(p, encoding="utf-8", errors="replace").read().splitlines():
        if l.startswith(("FAILED", "ERROR", "SUBFAIL")):
            k = l.split(" - ")[0].replace("\\", "/")
            k = THROUGH.sub(KEPT, k)
            out.add(k)
    return out


a, b = f(sys.argv[1]), f(sys.argv[2])
print("A-only:", sorted(a - b))
print("B-only:", sorted(b - a))
print("common", len(a & b))
