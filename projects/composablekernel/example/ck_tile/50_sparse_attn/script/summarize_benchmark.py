#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Summarize benchmark_sparse_attn.sh JSONL into markdown tables.

Dense (bf16) latency is the speedup denominator; TOPS uses full dense-equivalent FLOPs.
"""
import argparse, json, os, sys

D = 128  # hdim_q == hdim_v in these sweeps


def norm_mask(m):
    # normalize mask labels: 0/no/no_mask -> "no"; 1/causal/t -> "causal".
    s = str(m)
    if s in ("0", "no", "no_mask"):
        return "no"
    if s in ("1", "causal", "t"):
        return "causal"
    return s


def full_flops(b, h, s, mask):
    # 2 GEMMs x 2 flop/MAC x area x D (causal area = s(s+1)/2).
    area = s * s if mask == "no" else s * (s + 1) / 2.0
    return 4.0 * b * h * area * D


def tops(b, h, s, mask, ms):
    return full_flops(b, h, s, mask) / (ms * 1e-3) / 1e12


def load(outdir, name):
    rows = []
    path = os.path.join(outdir, name)
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("JSON "):
                line = line[5:]
            if line:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"warn: skipping malformed line in {name}: {line[:80]}", file=sys.stderr)
                    continue
                r["mask_type"] = norm_mask(r.get("mask_type"))
                rows.append(r)
    return rows


def key(r):
    return (int(r["seqlen_k"]), r["mask_type"])


# (token, display name, jsonl file). Token matches benchmark_sparse_attn.sh VARIANTS.
ALL_VARIANTS = [
    ("jenga",     "Jenga",     "jenga.jsonl"),
    ("vsa",       "VSA",       "vsa.jsonl"),
    ("sparge",    "Sparge",    "sparge.jsonl"),
    ("sage_int8", "Sage-INT8", "sage_int8.jsonl"),
    ("sage_fp8",  "Sage-FP8",  "sage_fp8.jsonl"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--variants", default=os.environ.get("VARIANTS", ""),
                    help="space-separated subset (default all): "
                         "jenga vsa sparge sage_int8 sage_fp8")
    args = ap.parse_args()

    sel = args.variants.split() or [t for t, _, _ in ALL_VARIANTS]
    chosen = [(disp, f) for tok, disp, f in ALL_VARIANTS if tok in sel]
    if not chosen:
        print(f"No known variants in '{args.variants}'.", file=sys.stderr)
        sys.exit(1)

    # dedup by (seqlen_k, mask): later rows overwrite earlier (Sweep B re-runs dense at 16384).
    dense = {key(r): r for r in load(args.outdir, "dense.jsonl")}
    variants = {disp: load(args.outdir, f) for disp, f in chosen}
    if not dense:
        print("No dense.jsonl rows found.", file=sys.stderr)
        sys.exit(1)

    # records[(s, mask, sparsity)][variant] = {"speedup":.., "tops":..}
    records = {}
    b = h = None
    for vname, rows in variants.items():
        for r in rows:
            try:
                s, mask = int(r["seqlen_k"]), r["mask_type"]
                b, h = int(r["batch"]), int(r["nhead"])
                sp = round(float(r["sparsity"]), 3)
                d = dense.get((s, mask))
                if d is None:
                    continue
                ms = float(r["latency_ms"])
                if ms <= 0.0 or float(d["latency_ms"]) <= 0.0:
                    print(f"warn: non-positive latency for s={r.get('seqlen_k')} mask={r.get('mask_type')}; skipping", file=sys.stderr)
                    continue
                speedup = float(d["latency_ms"]) / ms
                records.setdefault((s, mask, sp), {})[vname] = {
                    "speedup": speedup, "tops": tops(b, h, s, mask, ms)}
            except (KeyError, ValueError):
                print(f"warn: skipping row with missing/bad fields in {vname}: {str(r)[:80]}", file=sys.stderr)
                continue

    if b is None or h is None:
        print("No variant rows found; nothing to summarize.", file=sys.stderr)
        sys.exit(1)

    cols = [disp for disp, _ in chosen]

    def cell(rec, v):
        if v not in rec:
            return "—"
        return f"{rec[v]['speedup']:.2f}x / {rec[v]['tops']:.0f}"

    for mask in ("no", "causal"):
        print(f"\n### Table A — matrix (mask={mask}); cell = speedup x / TOPS\n")
        print("| seqlen | sparsity | " + " | ".join(cols) + " |")
        print("|---:|---:|" + "|".join(["---:"] * len(cols)) + "|")
        keys = sorted(k for k in records if k[1] == mask and k[0] in (8192, 16384, 32768))
        for (s, m, sp) in keys:
            rec = records[(s, m, sp)]
            print(f"| {s} | {sp} | " + " | ".join(cell(rec, c) for c in cols) + " |")

    for mask in ("no", "causal"):
        s = 16384
        print(f"\n### Table B — official curve (s={s}, mask={mask}); TOPS\n")
        d = dense.get((s, mask))
        dense_tops = tops(b, h, s, mask, float(d["latency_ms"])) if d else 0.0
        print(f"dense bf16 (example01 async) TOPS ≈ {dense_tops:.0f}\n")
        print("| sparsity | " + " | ".join(f"{c} TOPS" for c in cols) + " |")
        print("|---:|" + "|".join(["---:"] * len(cols)) + "|")
        sps = sorted({k[2] for k in records if k[0] == s and k[1] == mask})
        for sp in sps:
            rec = records.get((s, mask, sp), {})
            print(f"| {sp} | " + " | ".join(
                (f"{rec[c]['tops']:.0f}" if c in rec else "—") for c in cols) + " |")


if __name__ == "__main__":
    main()
