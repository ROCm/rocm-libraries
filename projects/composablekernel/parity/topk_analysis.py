#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Task #19 (bridge side) — top-K fastest kernels per problem from a sweep CSV.

Reads the bridge sweep results and, for each problem shape, ranks kernels by
TFLOPS and prints the top-K. Two CSVs can be passed to compare top-K set
overlap (e.g. two independent bridge runs => determinism, or bridge vs a future
old-TE export). Jaccard overlap of the top-K kernel-name sets is reported.
"""
import argparse
import csv
from collections import defaultdict


def load(path):
    # returns {(M,N,K): [(kernel, tflops, non_zero), ...]}
    by_prob = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            tf = float(row["tflops"])
            nz = int(row.get("non_zero", 1))
            by_prob[key].append((row["kernel"], tf, nz))
    return by_prob


def topk(rows, k):
    valid = [(name, tf) for name, tf, nz in rows if nz > 0 and tf > 0]
    return sorted(valid, key=lambda x: -x[1])[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b", nargs="?", default=None)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    a = load(args.csv_a)
    b = load(args.csv_b) if args.csv_b else None

    for key in sorted(a.keys()):
        M, N, K = key
        ta = topk(a[key], args.k)
        print(f"\nProblem M={M} N={N} K={K}  (top-{args.k} by TFLOPS)")
        for rank, (name, tf) in enumerate(ta, 1):
            print(f"  {rank}. {tf:8.1f} TFLOPS  {name}")
        if b is not None and key in b:
            tb = topk(b[key], args.k)
            sa, sb = {n for n, _ in ta}, {n for n, _ in tb}
            inter, union = len(sa & sb), len(sa | sb)
            jac = inter / union if union else 1.0
            print(f"  top-{args.k} set overlap vs B: {inter}/{args.k}  Jaccard={jac:.2f}")
    print("\nTOPK ANALYSIS DONE")


if __name__ == "__main__":
    main()
