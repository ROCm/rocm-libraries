#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convert MIOpen LGBM heuristic assets from JSON/LightGBM-text to the compact
binary format the C++ runtime loads (see lgbm_binary.hpp / lgbm_forest.cpp).

Two outputs, both little-endian, self-describing (magic + version):

  lgbm_rank.bin   = "MIORANK1" u32:version  FOREST
  lgbm_pcfg.bin   = "MIOPCFG1" u32:version  u32:num_solvers
                    directory[num_solvers]{ u16 name_len; name; u64 off; u64 size }
                    per-solver section{
                        i32 feat_count; i32 prob_feat_count; i32 arg_count; u8 has_gfx_code
                        FOREST
                        u32 num_buckets
                        bucket{ u16 key_len; key; u32 num_cands
                                cand{ u16 desc_len; desc; f64[arg_count] args } }
                    }

FOREST (shared, mirrors LgbmForest::Tree/Node, decoded — no decision_type logic
left for the reader):
  u32 num_trees
  tree{ u32 num_nodes
        i32[n] split_feature; f64[n] threshold; i32[n] left; i32[n] right
        i32[n] cat_index; u8[n] default_left; u8[n] missing_type
        u32 num_leaves; f64[num_leaves] leaf_value
        u32 cat_bitset_len; u32[...] cat_bitset
        u32 cat_offsets_len; i64[...] cat_offsets }

Usage:
  convert_lgbm_binary.py --kernels-dir <dir> [--out-dir <dir>]
    reads <dir>/lgbm_rank_model.txt, <dir>/lgbm_pcfg_model_meta.json,
    <dir>/lgbm_pcfg_catalog.json, <dir>/lgbm_pcfg_<solver>_model.txt
"""
import argparse
import json
import math
import struct
from pathlib import Path

RANK_MAGIC = b"MIORANK1"
PCFG_MAGIC = b"MIOPCFG1"
VERSION = 1

# LightGBM decision_type bitmask (matches lgbm_forest.cpp).
CAT_MASK = 0x01
DEFAULT_LEFT_MASK = 0x02
MISSING_TYPE_SHIFT = 2
MISSING_TYPE_MASK = 0x03

# Must match kNumBaseProbFeatures in lgbm_pcfg_metadata.hpp.
NUM_BASE_PROB_FEATURES = 27


def _parse_line_values(text, conv):
    return [conv(tok) for tok in text.split()]


def parse_lightgbm_text(path):
    """Parse a LightGBM text dump into a list of decoded trees.

    Each tree is a dict with the exact arrays the FOREST block stores.
    Mirrors LgbmForest's text constructor (flush_tree) semantics.
    """
    trees = []
    cur = None

    def flush():
        nonlocal cur
        if cur is None:
            return
        sf = cur.get("split_feature", [])
        dt = cur.get("decision_type", [])
        lc = cur.get("left_child", [])
        rc = cur.get("right_child", [])
        th = cur.get("threshold", [])
        lv = cur.get("leaf_value", [])
        cat_bound = cur.get("cat_boundaries", [])
        cat_thr = cur.get("cat_threshold", [])

        n = len(sf)
        split_feature = []
        threshold = []
        left = []
        right = []
        cat_index = []
        default_left = []
        missing_type = []
        for i in range(n):
            d = dt[i] if i < len(dt) else 0
            split_feature.append(sf[i])
            default_left.append(1 if (d & DEFAULT_LEFT_MASK) else 0)
            missing_type.append((d >> MISSING_TYPE_SHIFT) & MISSING_TYPE_MASK)
            left.append(lc[i] if i < len(lc) else -1)
            right.append(rc[i] if i < len(rc) else -1)
            if d & CAT_MASK:
                cat_index.append(int(th[i]))
                threshold.append(0.0)
            else:
                cat_index.append(-1)
                threshold.append(th[i] if i < len(th) else 0.0)

        trees.append(
            {
                "split_feature": split_feature,
                "threshold": threshold,
                "left": left,
                "right": right,
                "cat_index": cat_index,
                "default_left": default_left,
                "missing_type": missing_type,
                "leaf_value": lv,
                "cat_bitset": cat_thr,
                "cat_offsets": cat_bound,
            }
        )
        cur = None

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                flush()
                continue
            eq = line.find("=")
            if eq < 0:
                continue
            key = line[:eq]
            val = line[eq + 1 :]
            if key == "Tree":
                flush()
                cur = {}
            elif cur is None:
                continue
            elif key == "split_feature":
                cur[key] = _parse_line_values(val, int)
            elif key == "decision_type":
                cur[key] = _parse_line_values(val, int)
            elif key == "left_child":
                cur[key] = _parse_line_values(val, int)
            elif key == "right_child":
                cur[key] = _parse_line_values(val, int)
            elif key == "threshold":
                cur[key] = _parse_line_values(val, float)
            elif key == "leaf_value":
                cur[key] = _parse_line_values(val, float)
            elif key == "cat_boundaries":
                cur[key] = _parse_line_values(val, int)
            elif key == "cat_threshold":
                cur[key] = _parse_line_values(val, lambda t: int(t) & 0xFFFFFFFF)
    flush()
    return trees


def _i32(v):
    return struct.pack("<i", v)


def _u32(v):
    return struct.pack("<I", v)


def _u64(v):
    return struct.pack("<Q", v)


def _u16(v):
    return struct.pack("<H", v)


def _arr(fmt, vals):
    return struct.pack("<%d%s" % (len(vals), fmt), *vals)


def _str(s):
    b = s.encode("utf-8")
    return _u16(len(b)) + b


def serialize_forest(trees):
    out = bytearray()
    out += _u32(len(trees))
    for t in trees:
        n = len(t["split_feature"])
        out += _u32(n)
        out += _arr("i", t["split_feature"])
        out += _arr("d", t["threshold"])
        out += _arr("i", t["left"])
        out += _arr("i", t["right"])
        out += _arr("i", t["cat_index"])
        out += _arr("B", t["default_left"])
        out += _arr("B", t["missing_type"])
        out += _u32(len(t["leaf_value"]))
        out += _arr("d", t["leaf_value"])
        out += _u32(len(t["cat_bitset"]))
        out += _arr("I", t["cat_bitset"])
        out += _u32(len(t["cat_offsets"]))
        out += _arr("q", t["cat_offsets"])
    return bytes(out)


def build_rank(kernels_dir):
    trees = parse_lightgbm_text(kernels_dir / "lgbm_rank_model.txt")
    out = bytearray()
    out += RANK_MAGIC
    out += _u32(VERSION)
    out += serialize_forest(trees)
    return bytes(out), len(trees)


def build_pcfg(kernels_dir):
    meta = json.load(open(kernels_dir / "lgbm_pcfg_model_meta.json"))
    catalog = json.load(open(kernels_dir / "lgbm_pcfg_catalog.json"))

    sections = []  # (name, bytes)
    for name, block in meta.items():
        prob_cols = block["prob_feat_cols"]
        n_prob = len(prob_cols)
        n_arg = len(block["arg_cols"])
        n_feat = len(block["feat_order"])
        has_gfx = n_prob == NUM_BASE_PROB_FEATURES + 1 and prob_cols[-1] == "gfx_code"
        base_ok = n_prob == NUM_BASE_PROB_FEATURES
        if not (base_ok or has_gfx) or n_feat != n_prob + n_arg:
            print(
                "skip %s (schema mismatch prob=%d arg=%d feat=%d)"
                % (name, n_prob, n_arg, n_feat)
            )
            continue
        model_file = kernels_dir / ("lgbm_pcfg_%s_model.txt" % name)
        if not model_file.exists():
            print("skip %s (missing %s)" % (name, model_file.name))
            continue
        trees = parse_lightgbm_text(model_file)
        if not trees:
            print("skip %s (no trees)" % name)
            continue

        sec = bytearray()
        sec += _i32(n_feat)
        sec += _i32(n_prob)
        sec += _i32(n_arg)
        sec += struct.pack("<B", 1 if has_gfx else 0)
        sec += serialize_forest(trees)

        buckets = catalog.get(name, {}).get("buckets", {})
        # Deterministic bucket order for reproducible output.
        bucket_items = sorted(buckets.items())
        sec += _u32(len(bucket_items))
        for key, cands in bucket_items:
            sec += _str(key)
            valid = []
            for c in cands:
                args = c["args"]
                if len(args) != n_arg:
                    continue  # mirror the C++ arg-count guard
                valid.append(c)
            sec += _u32(len(valid))
            for c in valid:
                sec += _str(c["desc"])
                vals = [(math.nan if a is None else float(a)) for a in c["args"]]
                sec += _arr("d", vals)
        sections.append((name, bytes(sec)))

    # Assemble header + directory + sections. Directory offsets are absolute.
    header = bytearray()
    header += PCFG_MAGIC
    header += _u32(VERSION)
    header += _u32(len(sections))

    dir_bytes = bytearray()
    for name, _ in sections:
        dir_bytes += _str(name)
        dir_bytes += _u64(0)  # offset placeholder
        dir_bytes += _u64(0)  # size placeholder

    body_start = len(header) + len(dir_bytes)
    # Recompute directory with real offsets.
    dir_bytes = bytearray()
    off = body_start
    for name, sec in sections:
        dir_bytes += _str(name)
        dir_bytes += _u64(off)
        dir_bytes += _u64(len(sec))
        off += len(sec)

    out = bytearray()
    out += header
    out += dir_bytes
    for _, sec in sections:
        out += sec
    return bytes(out), [n for n, _ in sections]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.kernels_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_bytes, n_trees = build_rank(args.kernels_dir)
    (out_dir / "lgbm_rank.bin").write_bytes(rank_bytes)
    print("wrote lgbm_rank.bin: %d trees, %d bytes" % (n_trees, len(rank_bytes)))

    pcfg_bytes, solvers = build_pcfg(args.kernels_dir)
    (out_dir / "lgbm_pcfg.bin").write_bytes(pcfg_bytes)
    print("wrote lgbm_pcfg.bin: %d solvers, %d bytes" % (len(solvers), len(pcfg_bytes)))


if __name__ == "__main__":
    main()
