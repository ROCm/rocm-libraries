#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convert MIOpen LGBM heuristic assets from JSON/LightGBM-text to the compact
binary format the C++ runtime loads (see lgbm_binary.hpp / lgbm_forest.cpp).

Two outputs, both little-endian, self-describing (magic + version):

  lgbm_rank.bin   = "MIORANK1" u32:version(2)  FOREST
  lgbm_pcfg.bin   = "MIOPCFG1" u32:version(5)  u32:num_solvers
                    directory[num_solvers]{ u16 name_len; name; u64 off; u64 size }
                    per-solver section{
                        i32 feat_count; i32 prob_feat_count; i32 arg_count; u8 has_gfx_code
                        [has_gfx_code only] u16 num_gfx; str[num_gfx] gfx_vocab
                        FOREST
                        CATALOG
                    }

gfx_vocab is the solver's gfx_code category order as trained (gfx_code of a
gfx_id == its index). It comes from the solver block's "gfx_code_vocab" list in
lgbm_pcfg_model_meta.json and is required for every solver whose
prob_feat_cols ends in "gfx_code".

CATALOG (v5; varint = unsigned LEB128, zigzag = signed varint, "packed" = fixed
width unsigned values, LSB first, padded to a byte):
  varint num_unique (<= 65536); varint num_templates
  strs templates; varint[num_templates] count (sum == num_unique)
  [num_unique > 0 only]
  COLUMN(num_unique) x arg_count              the args of every candidate
  per template, per SLOT_MARK in it: u8 ref; [ref == 0] COLUMN(count)
  varint num_buckets; strs keys; u8 idx_width
  bucket[num_buckets]{ varint n; packed(n, idx_width) candidate_index }

  strs       = { varint shared (bytes kept from the previous string); varint
                 tail_len; tail } repeated
  COLUMN(n)  = varint d (distinct values, 1..n); d == 1: zigzag value; else
                 zigzag delta of each distinct value in ascending order (the first
                 from 0), then packed(n, (d-1).bit_length()) index into them

Candidates are the distinct (desc, args) pairs, sorted by template, then args.
A descriptor is stored as a template (each decimal run of at most 18 digits and
no leading zero replaced by byte 0x01) plus those numbers, one per slot in
order. Slot numbers are COLUMNs over the template's candidates, or, when
ref != 0, equal to arg column ref - 1. Buckets keep their candidate order
(it breaks score ties) and refer to candidates by index.

Catalog args must be integers in [-2147483647, 2147483647]; null is stored as
INT32_MIN and is read back as NaN. ConvHipConv's arg_max_px uses 2147483647 for
"no limit". An arg column that no tree of the solver's model splits on is never
read by the runtime (LgbmForest::Score reads only split features), so it is
stored as the constant 0 unless --keep-unused-args is given.

FOREST (shared, mirrors LgbmForest::Tree/Node, decoded — no decision_type logic
left for the reader):
  u32 num_trees; u32 num_thresholds; f64[num_thresholds] thresholds (sorted,
  unique, bit-exact); u8 child_width (1|2|4, signed); u8 leaf_width (4|8)
  tree{ u16 n (internal nodes; leaves = n + 1)
        u8[n] split_feature
        u16[n] split_index (numeric: index into thresholds; categorical: cat index)
        child_width[n] left; child_width[n] right
        u8[n] flags (bit0 default_left, bits1-2 missing_type, bit3 categorical)
        leaf_width[n + 1] leaf_value
        u16 cat_bitset_len; u32[...] cat_bitset
        u16 cat_offsets_len; u16[...] cat_offsets }

Thresholds are never rounded, so every split takes the same branch as the
trained model. Leaf values are stored as f32 by default (--leaf-bytes 4); each
tree's leaf then differs by at most 2^-24 of its magnitude, so a score moves by
at most 2^-24 * sum over trees of max |leaf|. --leaf-bytes 8 keeps them exact.

Usage:
  convert_lgbm_binary.py --kernels-dir <dir> [--out-dir <dir>] [--leaf-bytes 4|8] [--keep-unused-args]
    reads <dir>/lgbm_rank_model.txt, <dir>/lgbm_pcfg_model_meta.json,
    <dir>/lgbm_pcfg_catalog.json, <dir>/lgbm_pcfg_<solver>_model.txt

After a retrain, run merge_lgbm_pcfg_develop_arch.py on the converter's
lgbm_pcfg.bin (and its golden file) if a solver must keep develop's ranking on
one architecture. The ship file does this for gfx1250
ConvHipImplicitGemm3DGroupBwdXdlops. The converter does not write that
section, so a retrain without the merge drops it. Run the merge's "verify"
subcommand on the result.
"""
import argparse
import json
import re
import struct
from pathlib import Path

RANK_MAGIC = b"MIORANK1"
PCFG_MAGIC = b"MIOPCFG1"
# Must match kRankFormatVersion / kPcfgFormatVersion in lgbm_binary.hpp.
RANK_VERSION = 2
PCFG_VERSION = 5

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


def _int_width(lo, hi):
    """Smallest signed little-endian width (bytes, struct code) holding [lo, hi]."""
    for width, code in ((1, "b"), (2, "h"), (4, "i")):
        bound = 1 << (8 * width - 1)
        if -bound <= lo and hi < bound:
            return width, code
    raise SystemExit("child index range [%d, %d] does not fit in i32" % (lo, hi))


def _check(cond, what):
    if not cond:
        raise SystemExit("cannot encode model: %s" % what)


def serialize_forest(trees, leaf_bytes=4):
    """Compact FOREST block. Thresholds are kept bit-exact in a shared f64 table;
    leaf values are narrowed to f32 when leaf_bytes == 4."""
    table = sorted(
        {th for t in trees for th, ci in zip(t["threshold"], t["cat_index"]) if ci < 0}
    )
    _check(len(table) <= 0xFFFF, "%d distinct thresholds (max 65535)" % len(table))
    index = {th: i for i, th in enumerate(table)}
    children = [c for t in trees for c in t["left"] + t["right"]]
    child_width, child_code = _int_width(
        min(children, default=0), max(children, default=0)
    )
    leaf_code = {4: "f", 8: "d"}[leaf_bytes]

    out = bytearray()
    out += _u32(len(trees))
    out += _u32(len(table))
    out += _arr("d", table)
    out += struct.pack("<BB", child_width, leaf_bytes)
    for t in trees:
        n = len(t["split_feature"])
        _check(n <= 0xFFFF, "tree with %d internal nodes (max 65535)" % n)
        _check(
            len(t["leaf_value"]) == n + 1,
            "tree with %d nodes and %d leaves" % (n, len(t["leaf_value"])),
        )
        _check(
            all(0 <= f <= 0xFF for f in t["split_feature"]),
            "split feature index above 255",
        )
        _check(
            all(0 <= ci <= 0xFFFF for ci in t["cat_index"] if ci >= 0),
            "categorical index above 65535",
        )
        _check(
            len(t["cat_bitset"]) <= 0xFFFF, "categorical bitset longer than 65535 words"
        )
        _check(len(t["cat_offsets"]) <= 0xFFFF, "more than 65535 categorical offsets")
        _check(
            all(0 <= o <= 0xFFFF for o in t["cat_offsets"]),
            "categorical offset above 65535",
        )
        split_index = [
            ci if ci >= 0 else index[th]
            for th, ci in zip(t["threshold"], t["cat_index"])
        ]
        flags = [
            (dl & 1) | ((mt & MISSING_TYPE_MASK) << 1) | ((1 if ci >= 0 else 0) << 3)
            for dl, mt, ci in zip(t["default_left"], t["missing_type"], t["cat_index"])
        ]
        out += _u16(n)
        out += _arr("B", t["split_feature"])
        out += _arr("H", split_index)
        out += _arr(child_code, t["left"])
        out += _arr(child_code, t["right"])
        out += _arr("B", flags)
        out += _arr(leaf_code, t["leaf_value"])
        out += _u16(len(t["cat_bitset"]))
        out += _arr("I", t["cat_bitset"])
        out += _u16(len(t["cat_offsets"]))
        out += _arr("H", t["cat_offsets"])
    return bytes(out)


# A catalog arg that is null/NaN is stored as this value and read back as NaN.
CAND_ARG_MISSING = -2147483648
CAND_ARG_MAX = 2147483647
# Marks a numeric slot in a descriptor template (see serialize_catalog).
SLOT_MARK = "\x01"
# A decimal run without a leading zero and of at most 18 digits (fits in i64).
_NUM_RE = re.compile(r"(?<![0-9])(0|[1-9][0-9]{0,17})(?![0-9])")
MAX_CANDIDATES = 0x10000


def _cand_arg(a):
    """Integer value of a catalog arg, or None for a null/NaN arg."""
    if a is None or a != a:
        return None
    _check(float(a) == int(a), "candidate arg %r is not an integer" % a)
    return int(a)


def _varint(v):
    """Unsigned LEB128."""
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if not v:
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)


def _zigzag(v):
    return (v << 1) if v >= 0 else ((-v) << 1) - 1


def _pack(vals, width):
    """Fixed-width unsigned values packed LSB first, padded to a whole byte."""
    out = bytearray()
    acc = nbits = 0
    for v in vals:
        acc |= v << nbits
        nbits += width
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def _int_column(vals):
    """varint d (distinct values); d == 1: one zigzag value; else the sorted
    distinct values as zigzag deltas from the previous (first from 0), then the
    values as bit-packed indices of (d - 1).bit_length() bits."""
    distinct = sorted(set(vals))
    out = bytearray(_varint(len(distinct)))
    if len(distinct) == 1:
        return bytes(out + _varint(_zigzag(distinct[0])))
    prev = 0
    for v in distinct:
        out += _varint(_zigzag(v - prev))
        prev = v
    index = {v: i for i, v in enumerate(distinct)}
    out += _pack([index[v] for v in vals], (len(distinct) - 1).bit_length())
    return bytes(out)


def _front_coded(strs):
    """Strings as {varint bytes shared with the previous string, varint tail
    length, tail}."""
    out = bytearray()
    prev = b""
    for s in strs:
        b = s.encode("utf-8")
        n = 0
        while n < min(len(b), len(prev)) and b[n] == prev[n]:
            n += 1
        out += _varint(n) + _varint(len(b) - n) + b[n:]
        prev = b
    return bytes(out)


def serialize_catalog(bucket_items, n_arg, used_args=None):
    """Candidate catalog (see the module docstring for the layout).

    Each distinct (desc, args) candidate is stored once, sorted by descriptor
    template, then args. A descriptor is stored as its template (digit runs
    replaced by SLOT_MARK) plus those numbers. The numbers and the args are
    stored column-wise. A number column that equals an arg column over the whole
    template group is stored as a reference to it. Arg columns not in used_args
    (None = all are used) are never read by the model; they are stored as the
    constant 0."""
    unique = {}
    buckets = []
    for key, cands in bucket_items:
        idx = []
        for c in cands:
            if len(c["args"]) != n_arg:
                continue  # mirror the C++ arg-count guard
            cand = (c["desc"], tuple(_cand_arg(a) for a in c["args"]))
            idx.append(unique.setdefault(cand, len(unique)))
        buckets.append((key, idx))
    _check(
        len(unique) <= MAX_CANDIDATES,
        "%d distinct candidates (max %d)" % (len(unique), MAX_CANDIDATES),
    )

    recs = []  # (template, args with missing as CAND_ARG_MISSING, numbers, old index)
    for old, (desc, args) in enumerate(unique):
        _check(
            SLOT_MARK not in desc, "descriptor %r contains the slot marker byte" % desc
        )
        _check(
            all(a is None or -CAND_ARG_MAX <= a <= CAND_ARG_MAX for a in args),
            "candidate arg outside [-%d, %d]" % (CAND_ARG_MAX, CAND_ARG_MAX),
        )
        recs.append(
            (
                _NUM_RE.sub(SLOT_MARK, desc),
                tuple(CAND_ARG_MISSING if a is None else a for a in args),
                [int(m) for m in _NUM_RE.findall(desc)],
                old,
            )
        )
    recs.sort(key=lambda r: r[:3])
    new_index = {r[3]: i for i, r in enumerate(recs)}
    groups = {}
    for r in recs:
        groups.setdefault(r[0], []).append(r)
    templates = list(groups)  # sorted: recs is sorted by template first

    out = bytearray()
    out += _varint(len(recs))
    out += _varint(len(templates))
    out += _front_coded(templates)
    for t in templates:
        out += _varint(len(groups[t]))
    if recs:
        for a in range(n_arg):
            keep = used_args is None or a in used_args
            out += _int_column([r[1][a] if keep else 0 for r in recs])
        for t in templates:
            grp = groups[t]
            for s in range(len(grp[0][2])):
                col = [r[2][s] for r in grp]
                ref = next(
                    (
                        a + 1
                        for a in range(n_arg)
                        if (used_args is None or a in used_args)
                        and all(r[1][a] == v for r, v in zip(grp, col))
                    ),
                    0,
                )
                _check(ref <= 0xFF, "more than 255 arg columns")
                out += bytes([ref])
                if not ref:
                    out += _int_column(col)

    width = max(1, (len(recs) - 1).bit_length())
    out += _varint(len(buckets))
    out += _front_coded([key for key, _ in buckets])
    out += bytes([width])
    for _, idx in buckets:
        out += _varint(len(idx))
        out += _pack([new_index[i] for i in idx], width)
    return bytes(out)


def build_rank(kernels_dir, leaf_bytes):
    trees = parse_lightgbm_text(kernels_dir / "lgbm_rank_model.txt")
    out = bytearray()
    out += RANK_MAGIC
    out += _u32(RANK_VERSION)
    out += serialize_forest(trees, leaf_bytes)
    return bytes(out), len(trees)


def build_pcfg(kernels_dir, leaf_bytes, keep_unused_args=False):
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
        if has_gfx:
            vocab = block.get("gfx_code_vocab")
            if (
                not isinstance(vocab, list)
                or not vocab
                or not all(isinstance(g, str) and g for g in vocab)
                or len(set(vocab)) != len(vocab)
            ):
                # Without the trained category order the runtime cannot encode
                # gfx_code; guessing it would silently corrupt every prediction.
                raise SystemExit(
                    "%s: gfx_code solver needs a non-empty, unique "
                    '"gfx_code_vocab" string list in lgbm_pcfg_model_meta.json' % name
                )
            sec += _u16(len(vocab))
            for g in vocab:
                sec += _str(g)
        sec += serialize_forest(trees, leaf_bytes)

        buckets = catalog.get(name, {}).get("buckets", {})
        # Feature f of a row is arg column f - n_prob. Only split features are
        # ever read (LgbmForest::Score), so the others need not be stored.
        used_args = None
        if not keep_unused_args:
            used_args = {
                f - n_prob for t in trees for f in t["split_feature"] if f >= n_prob
            }
        # Deterministic bucket order for reproducible output.
        sec += serialize_catalog(sorted(buckets.items()), n_arg, used_args)
        sections.append((name, bytes(sec)))

    # Assemble header + directory + sections. Directory offsets are absolute.
    header = bytearray()
    header += PCFG_MAGIC
    header += _u32(PCFG_VERSION)
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
    ap.add_argument(
        "--leaf-bytes",
        type=int,
        choices=(4, 8),
        default=4,
        help="leaf value width: 4 = f32 (default), 8 = f64 (bit-exact scores)",
    )
    ap.add_argument(
        "--keep-unused-args",
        action="store_true",
        help="store every catalog arg column, including those no tree splits on",
    )
    args = ap.parse_args()
    out_dir = args.out_dir or args.kernels_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_bytes, n_trees = build_rank(args.kernels_dir, args.leaf_bytes)
    (out_dir / "lgbm_rank.bin").write_bytes(rank_bytes)
    print("wrote lgbm_rank.bin: %d trees, %d bytes" % (n_trees, len(rank_bytes)))

    pcfg_bytes, solvers = build_pcfg(
        args.kernels_dir, args.leaf_bytes, args.keep_unused_args
    )
    (out_dir / "lgbm_pcfg.bin").write_bytes(pcfg_bytes)
    print("wrote lgbm_pcfg.bin: %d solvers, %d bytes" % (len(solvers), len(pcfg_bytes)))


if __name__ == "__main__":
    main()
