#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Keep develop's ranking for one architecture of one solver in a retrained
lgbm_pcfg.bin.

Run this after convert_lgbm_binary.py. The converter writes one forest per
solver, trained on all architectures. For some (solver, architecture) pairs the
retrained model is not wanted yet and develop's model must keep ranking those
candidates, for example gfx1250 ConvHipImplicitGemm3DGroupBwdXdlops. The
runtime loads one forest per solver and the model has no architecture feature,
so copying develop's catalog buckets alone would still rank them with the new
forest. This script builds a section that holds both forests behind a gate on
one catalog arg column that neither model splits on:

  * develop's buckets whose key starts with "<arch>|" replace the new ones, and
    their candidates carry MARK in arg column --mark-col;
  * every other candidate carries 0 there (the converter stores unused
    columns as 0);
  * each new tree gets a root split "arg[mark-col] <= GATE_THRESHOLD ? 0.0 :
    tree", each develop tree gets "arg[mark-col] <= GATE_THRESHOLD ? tree :
    0.0". A candidate's score is therefore the sum of one forest plus exact
    zeros;
  * leaves of the section are f64, so develop's leaves stay exact (the new
    forest's f32 leaves widen exactly).

There is no format or C++ change. Every other section is copied unchanged.

Subcommands (all take --solver and --arch; --mark-col defaults to 16):

  build   --new NEW.bin --develop DEVELOP.bin --out OUT.bin
          Writes OUT.bin, then runs the same checks as verify on it.
          NEW.bin is the converter's output (pcfg v5). DEVELOP.bin is
          develop's lgbm_pcfg.bin before it was converted (pcfg v2 or v5) with
          no gate on --mark-col.
  verify  --merged OUT.bin --new NEW.bin --develop DEVELOP.bin [--golden G.json]
          Exits non-zero unless: every other section is byte-identical to
          NEW.bin; the target section keeps NEW.bin's header fields and the
          bucket keys of NEW.bin plus develop's <arch> keys; buckets outside
          <arch> equal NEW.bin's candidates and score bitwise like NEW.bin's
          forest; <arch> buckets equal develop's candidates, rank in the same
          order and score bitwise like develop's forest. Scores are compared on
          seeded probe rows whose features are drawn from the models' split
          thresholds and their neighbours. With --golden, every golden row of
          the solver must also score within 1e-6 of its expected value.
  golden  --new-golden NEW.json --develop-golden DEVELOP.json --out OUT.json
          --cu-count N --wave-size N
          Copies develop's golden rows of the solver for the architecture
          (rows with that cu_count and wave_size, problem features 19 and 20)
          into the new golden file with arg column --mark-col set to MARK, so
          CPU_LgbmPcfgForest_NONE.MatchesGoldenVectors scores them with
          develop's forest (their expected values are develop's).

Example (gfx1250 ConvHipImplicitGemm3DGroupBwdXdlops, run from this directory
after convert_lgbm_binary.py wrote lgbm_pcfg.bin to build/):

  S=ConvHipImplicitGemm3DGroupBwdXdlops
  git show origin/develop:projects/miopen/src/kernels/lgbm_pcfg.bin > dev.bin
  merge_lgbm_pcfg_develop_arch.py build --solver $S --arch gfx1250 \\
      --new build/lgbm_pcfg.bin --develop dev.bin \\
      --out ../src/kernels/lgbm_pcfg.bin
  merge_lgbm_pcfg_develop_arch.py golden --solver $S --arch gfx1250 \\
      --cu-count 256 --wave-size 32 --new-golden new_golden.json \\
      --develop-golden dev_golden.json --out ../src/kernels/lgbm_pcfg_golden.json
"""
import argparse
import json
import math
import random
import struct
import sys
from pathlib import Path

import convert_lgbm_binary as cv

# Value of the gate column for candidates ranked by develop's forest. Within
# the range the catalog stores (cv.CAND_ARG_MAX) and never produced by a trained
# model.
MARK = -2147483647
# Root split threshold of the gate. Every stored column value is either 0 (new
# forest) or MARK (develop forest) or NaN, which is coerced to 0 (missing_type
# None); -2**30 separates them.
GATE_THRESHOLD = -1073741824.0
DEFAULT_MARK_COL = 16
# Golden rows: problem features that identify the hardware.
GOLDEN_CU_COUNT_FEATURE = 19
GOLDEN_WAVE_SIZE_FEATURE = 20
# CPU_LgbmPcfgForest_NONE.MatchesGoldenVectors tolerance.
GOLDEN_TOLERANCE = 1e-6


def fail(msg):
    raise SystemExit("error: " + msg)


# ---------------------------------------------------------------------------
# Reader for lgbm_pcfg.bin v2 (develop's original layout) and v5 (the layout
# convert_lgbm_binary.py writes). Mirrors the C++ loaders.
# ---------------------------------------------------------------------------


class Reader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def take(self, n):
        if n > len(self.buf) - self.pos:
            fail("truncated file at byte %d (+%d)" % (self.pos, n))
        v = self.buf[self.pos : self.pos + n]
        self.pos += n
        return v

    def scalar(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def array(self, fmt, n):
        if n == 0:
            return []
        return list(
            struct.unpack("<%d%s" % (n, fmt), self.take(struct.calcsize(fmt) * n))
        )

    def string(self):
        return self.take(self.scalar("H")).decode("utf-8")

    def varint(self):
        v = shift = 0
        while True:
            b = self.scalar("B")
            v |= (b & 0x7F) << shift
            if not b & 0x80:
                return v
            shift += 7

    def zigzag(self):
        u = self.varint()
        return (u >> 1) ^ -(u & 1)

    def packed(self, n, width):
        acc = int.from_bytes(self.take((n * width + 7) // 8), "little")
        mask = (1 << width) - 1
        return [(acc >> (i * width)) & mask for i in range(n)]


def read_forest_v2(r):
    trees = []
    for _ in range(r.scalar("I")):
        n = r.scalar("I")
        t = {
            "split_feature": r.array("i", n),
            "threshold": r.array("d", n),
            "left": r.array("i", n),
            "right": r.array("i", n),
            "cat_index": r.array("i", n),
            "default_left": r.array("B", n),
            "missing_type": r.array("B", n),
        }
        t["leaf_value"] = r.array("d", r.scalar("I"))
        t["cat_bitset"] = r.array("I", r.scalar("I"))
        t["cat_offsets"] = r.array("q", r.scalar("I"))
        trees.append(t)
    return trees


def read_forest_v5(r):
    num_trees = r.scalar("I")
    table = r.array("d", r.scalar("I"))
    child_code = {1: "b", 2: "h", 4: "i"}[r.scalar("B")]
    leaf_bytes = r.scalar("B")
    leaf_code = {4: "f", 8: "d"}[leaf_bytes]
    trees = []
    for _ in range(num_trees):
        n = r.scalar("H")
        split_feature = r.array("B", n)
        split_index = r.array("H", n)
        left = r.array(child_code, n)
        right = r.array(child_code, n)
        flags = r.array("B", n)
        t = {
            "split_feature": split_feature,
            "threshold": [
                0.0 if f & 8 else table[i] for f, i in zip(flags, split_index)
            ],
            "cat_index": [i if f & 8 else -1 for f, i in zip(flags, split_index)],
            "left": left,
            "right": right,
            "default_left": [f & 1 for f in flags],
            "missing_type": [(f >> 1) & 3 for f in flags],
            "leaf_value": r.array(leaf_code, n + 1),
        }
        t["cat_bitset"] = r.array("I", r.scalar("H"))
        t["cat_offsets"] = r.array("H", r.scalar("H"))
        trees.append(t)
    return trees, leaf_bytes


def read_int_column(r, n):
    distinct = r.varint()
    if not 1 <= distinct <= n:
        fail("bad column of %d values with %d distinct" % (n, distinct))
    values = []
    prev = 0
    for _ in range(distinct):
        prev += r.zigzag()
        values.append(prev)
    if distinct == 1:
        return values * n
    return [values[i] for i in r.packed(n, (distinct - 1).bit_length())]


def read_strings(r, count):
    out = []
    prev = b""
    for _ in range(count):
        shared = r.varint()
        prev = prev[:shared] + r.take(r.varint())
        out.append(prev.decode("utf-8"))
    return out


def read_catalog_v5(r, arg_count):
    num_unique = r.varint()
    templates = read_strings(r, r.varint())
    counts = [r.varint() for _ in templates]
    descs = [None] * num_unique
    args = []
    if num_unique:
        cols = [read_int_column(r, num_unique) for _ in range(arg_count)]
        first = 0
        for tpl, count in zip(templates, counts):
            slots = []
            for _ in range(tpl.count(cv.SLOT_MARK)):
                ref = r.scalar("B")
                slots.append(
                    read_int_column(r, count)
                    if ref == 0
                    else cols[ref - 1][first : first + count]
                )
            parts = tpl.split(cv.SLOT_MARK)
            for k in range(count):
                desc = parts[0]
                for s, tail in enumerate(parts[1:]):
                    desc += str(slots[s][k]) + tail
                descs[first + k] = desc
            first += count
        args = [
            [math.nan if c[i] == cv.CAND_ARG_MISSING else float(c[i]) for c in cols]
            for i in range(num_unique)
        ]
    keys = read_strings(r, r.varint())
    width = r.scalar("B")
    buckets = {}
    for key in keys:
        idx = r.packed(r.varint(), width)
        buckets[key] = [(descs[i], args[i]) for i in idx]
    return buckets


def read_catalog_v2(r, arg_count):
    buckets = {}
    for _ in range(r.scalar("I")):
        key = r.string()
        cands = []
        for _ in range(r.scalar("I")):
            desc = r.string()
            cands.append((desc, r.array("d", arg_count)))
        buckets[key] = cands
    return buckets


def load_pcfg(path):
    buf = Path(path).read_bytes()
    r = Reader(buf)
    if r.take(8) != cv.PCFG_MAGIC:
        fail("%s is not an lgbm_pcfg.bin" % path)
    version = r.scalar("I")
    if version not in (2, 5):
        fail("%s: unsupported pcfg version %d" % (path, version))
    directory = []
    for _ in range(r.scalar("I")):
        name = r.string()
        directory.append((name, r.scalar("Q"), r.scalar("Q")))
    solvers = {}
    for name, off, size in directory:
        raw = buf[off : off + size]
        s = Reader(raw)
        sec = {
            "feat_count": s.scalar("i"),
            "prob_feat_count": s.scalar("i"),
            "arg_count": s.scalar("i"),
            "has_gfx_code": s.scalar("B") != 0,
        }
        sec["gfx_vocab"] = []
        if sec["has_gfx_code"]:
            sec["gfx_vocab"] = [s.string() for _ in range(s.scalar("H"))]
        sec["leaf_bytes"] = None
        if version == 2:
            sec["trees"] = read_forest_v2(s)
            sec["buckets"] = read_catalog_v2(s, sec["arg_count"])
        else:
            sec["trees"], sec["leaf_bytes"] = read_forest_v5(s)
            sec["buckets"] = read_catalog_v5(s, sec["arg_count"])
        if s.pos != len(raw):
            fail(
                "%s: section %s has %d trailing bytes" % (path, name, len(raw) - s.pos)
            )
        sec["raw"] = raw
        solvers[name] = sec
    return {
        "version": version,
        "directory": directory,
        "solvers": solvers,
        "bytes": buf,
    }


# ---------------------------------------------------------------------------
# Scoring (mirrors LgbmForest::ScoreTree)
# ---------------------------------------------------------------------------


def is_missing(v):
    return v is None or v != v


def score_tree(t, row):
    if not t["split_feature"]:
        return t["leaf_value"][0] if t["leaf_value"] else 0.0
    node = 0
    while True:
        v = row[t["split_feature"][node]]
        missing = is_missing(v)
        cat = t["cat_index"][node]
        if cat >= 0:
            go_left = False
            if not missing and int(v) >= 0:
                c = int(v)
                base = t["cat_offsets"][cat]
                if base + (c >> 5) < t["cat_offsets"][cat + 1]:
                    go_left = (t["cat_bitset"][base + (c >> 5)] >> (c & 31)) & 1 != 0
        else:
            value = 0.0 if missing else v
            mt = t["missing_type"][node]
            if (missing and mt == 2) or (mt == 1 and value == 0.0):
                go_left = bool(t["default_left"][node])
            else:
                go_left = value <= t["threshold"][node]
        child = t["left"][node] if go_left else t["right"][node]
        if child < 0:
            return t["leaf_value"][~child]
        node = child


def score_row(trees, row):
    total = 0.0
    for t in trees:
        total += score_tree(t, row)
    return total


def same_bits(a, b):
    return struct.pack("<d", a) == struct.pack("<d", b)


def same_arg(a, b):
    return (a != a and b != b) or a == b


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def used_arg_columns(trees, prob_feat_count):
    return {
        f - prob_feat_count
        for t in trees
        for f in t["split_feature"]
        if f >= prob_feat_count
    }


def encode_section(sec, trees, buckets, leaf_bytes, used_args):
    out = bytearray()
    out += cv._i32(sec["feat_count"]) + cv._i32(sec["prob_feat_count"])
    out += cv._i32(sec["arg_count"])
    out += struct.pack("<B", 1 if sec["has_gfx_code"] else 0)
    if sec["has_gfx_code"]:
        out += cv._u16(len(sec["gfx_vocab"]))
        for g in sec["gfx_vocab"]:
            out += cv._str(g)
    out += cv.serialize_forest(trees, leaf_bytes)
    items = [
        (
            key,
            [
                {"desc": desc, "args": [None if a != a else a for a in args]}
                for desc, args in cands
            ],
        )
        for key, cands in sorted(buckets.items())
    ]
    out += cv.serialize_catalog(items, sec["arg_count"], used_args)
    return bytes(out)


def assemble(sections):
    header = cv.PCFG_MAGIC + cv._u32(cv.PCFG_VERSION) + cv._u32(len(sections))
    off = len(header) + sum(2 + len(n.encode()) + 16 for n, _ in sections)
    directory = bytearray()
    body = bytearray()
    for name, data in sections:
        directory += cv._str(name) + cv._u64(off) + cv._u64(len(data))
        off += len(data)
        body += data
    return bytes(header + directory + body)


def gate_tree(t, feature, develop_side):
    """Prepends the root split on `feature`: <= GATE_THRESHOLD takes the develop
    side. The other side returns 0.0, a new leaf appended to the tree."""
    n = len(t["split_feature"])
    if len(t["leaf_value"]) != n + 1:
        fail("tree with %d nodes and %d leaves" % (n, len(t["leaf_value"])))
    zero_leaf = ~(n + 1)
    original = 1 if n else ~0

    def shift(c):
        return c + 1 if c >= 0 else c

    return {
        "split_feature": [feature] + t["split_feature"],
        "threshold": [GATE_THRESHOLD] + t["threshold"],
        "cat_index": [-1] + t["cat_index"],
        "left": [original if develop_side else zero_leaf]
        + [shift(c) for c in t["left"]],
        "right": [zero_leaf if develop_side else original]
        + [shift(c) for c in t["right"]],
        "default_left": [0] + t["default_left"],
        "missing_type": [0] + t["missing_type"],
        "leaf_value": t["leaf_value"] + [0.0],
        "cat_bitset": t["cat_bitset"],
        "cat_offsets": t["cat_offsets"],
    }


def check_inputs(new, develop, solver, mark_col):
    if new["version"] != 5:
        fail(
            "--new must be pcfg v5 (the converter's output), found v%d" % new["version"]
        )
    if solver not in new["solvers"]:
        fail("solver %s is not in --new" % solver)
    if solver not in develop["solvers"]:
        fail("solver %s is not in --develop" % solver)
    s = new["solvers"][solver]
    d = develop["solvers"][solver]
    for field in (
        "feat_count",
        "prob_feat_count",
        "arg_count",
        "has_gfx_code",
        "gfx_vocab",
    ):
        if s[field] != d[field]:
            fail(
                "%s differs between --new (%r) and --develop (%r)"
                % (field, s[field], d[field])
            )
    if not 0 <= mark_col < s["arg_count"]:
        fail("--mark-col %d outside the %d arg columns" % (mark_col, s["arg_count"]))
    feature = s["prob_feat_count"] + mark_col
    if feature > 0xFF:
        fail("gate feature %d does not fit the u8 split feature" % feature)
    if any(c >= 0 for t in s["trees"] + d["trees"] for c in t["cat_index"]):
        fail(
            "%s has categorical splits, which this script has not been checked with"
            % solver
        )
    used = {f for t in s["trees"] + d["trees"] for f in t["split_feature"]}
    if feature in used:
        fail(
            "arg column %d is split on by a model (a gate from an earlier merge?)"
            % mark_col
        )
    return s, d, feature


def build(args):
    new = load_pcfg(args.new)
    develop = load_pcfg(args.develop)
    s, d, feature = check_inputs(new, develop, args.solver, args.mark_col)
    p = s["prob_feat_count"]

    # Re-encoding the decoded sections must reproduce them byte for byte, or the
    # decoder and the encoder disagree and the output cannot be trusted.
    used_new = used_arg_columns(s["trees"], p)
    if (
        encode_section(s, s["trees"], s["buckets"], s["leaf_bytes"], used_new)
        != s["raw"]
    ):
        fail("re-encoding the %s section of --new is not byte-identical" % args.solver)
    if (
        assemble([(n, new["solvers"][n]["raw"]) for n, _, _ in new["directory"]])
        != new["bytes"]
    ):
        fail("re-assembling --new is not byte-identical")

    trees = [gate_tree(t, feature, False) for t in s["trees"]]
    trees += [gate_tree(t, feature, True) for t in d["trees"]]
    buckets = dict(s["buckets"])
    prefix = args.arch + "|"
    develop_keys = sorted(k for k in d["buckets"] if k.startswith(prefix))
    if not develop_keys:
        fail("--develop has no %s* bucket for %s" % (prefix, args.solver))
    for key in develop_keys:
        buckets[key] = [
            (
                desc,
                [MARK if c == args.mark_col else a[c] for c in range(d["arg_count"])],
            )
            for desc, a in d["buckets"][key]
        ]
    section = encode_section(s, trees, buckets, 8, used_arg_columns(trees, p))
    sections = [
        (n, section if n == args.solver else new["solvers"][n]["raw"])
        for n, _, _ in new["directory"]
    ]
    data = assemble(sections)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print("replaced buckets: %s" % ", ".join(develop_keys))
    print(
        "trees: %d new + %d develop; section %d -> %d bytes; file %d bytes"
        % (len(s["trees"]), len(d["trees"]), len(s["raw"]), len(section), len(data))
    )
    return verify_files(args, out)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def probe_rows(trees, prob_feat_count, count, seed):
    """Problem-feature prefixes drawn per feature from the split thresholds of
    `trees`, their neighbours, small integers and NaN."""
    pools = [{0.0, 1.0, -1.0, 2.0, 3.0} for _ in range(prob_feat_count)]
    for t in trees:
        for f, th, cat in zip(t["split_feature"], t["threshold"], t["cat_index"]):
            if f < prob_feat_count and cat < 0:
                pools[f].update(
                    (th, math.nextafter(th, -math.inf), math.nextafter(th, math.inf))
                )
    pools = [sorted(pool) + [math.nan] for pool in pools]
    rng = random.Random(seed)
    return [[rng.choice(pool) for pool in pools] for _ in range(count)]


def rank_bucket(trees, prefix, cands):
    scores = [score_row(trees, prefix + list(args)) for _, args in cands]
    order = sorted(range(len(cands)), key=lambda i: -scores[i])
    return [cands[i][0] for i in order], scores


def verify_files(args, merged_path):
    merged = load_pcfg(merged_path)
    new = load_pcfg(args.new)
    develop = load_pcfg(args.develop)
    s, d, _ = check_inputs(new, develop, args.solver, args.mark_col)
    errors = []

    def check(ok, what):
        print("%s %s" % ("PASS" if ok else "FAIL", what))
        if not ok:
            errors.append(what)

    check(merged["version"] == 5, "merged file is pcfg v5")
    check(
        [n for n, _, _ in merged["directory"]] == [n for n, _, _ in new["directory"]],
        "directory lists the same solvers in the same order as --new",
    )
    others = [n for n, _, _ in new["directory"] if n != args.solver]
    differing = [
        n
        for n in others
        if n not in merged["solvers"]
        or merged["solvers"][n]["raw"] != new["solvers"][n]["raw"]
    ]
    check(
        not differing,
        "%d other sections byte-identical to --new%s"
        % (len(others), "".join(" (differs: %s)" % n for n in differing)),
    )
    if args.solver not in merged["solvers"]:
        check(False, "solver %s is in the merged file" % args.solver)
        return report(errors)
    m = merged["solvers"][args.solver]
    p = m["prob_feat_count"]
    check(
        all(
            m[f] == s[f]
            for f in (
                "feat_count",
                "prob_feat_count",
                "arg_count",
                "has_gfx_code",
                "gfx_vocab",
            )
        ),
        "section header fields equal --new",
    )
    prefix_key = args.arch + "|"
    develop_keys = {k for k in d["buckets"] if k.startswith(prefix_key)}
    check(
        set(m["buckets"]) == set(s["buckets"]) | develop_keys,
        "bucket keys are --new's plus develop's %s* keys" % prefix_key,
    )
    check(len(develop_keys) > 0, "develop has %s* buckets" % prefix_key)
    check(
        len(m["trees"]) == len(s["trees"]) + len(d["trees"]),
        "section holds %d new + %d develop trees" % (len(s["trees"]), len(d["trees"])),
    )

    used = sorted(used_arg_columns(m["trees"], p))

    def args_equal(a, b):
        return all(same_arg(a[c], b[c]) for c in used)

    # Candidates.
    bad = [
        k
        for k in s["buckets"]
        if k not in develop_keys
        and (
            k not in m["buckets"]
            or len(m["buckets"][k]) != len(s["buckets"][k])
            or any(
                x[0] != y[0] or not args_equal(x[1], y[1])
                for x, y in zip(m["buckets"][k], s["buckets"][k])
            )
        )
    ]
    check(
        not bad,
        "buckets outside %s* equal --new (descriptors, order, used args)" % prefix_key,
    )
    bad = []
    for k in develop_keys:
        mine = m["buckets"].get(k, [])
        theirs = d["buckets"][k]
        if len(mine) != len(theirs):
            bad.append(k)
            continue
        for (md, ma), (dd, da) in zip(mine, theirs):
            want = [MARK if c == args.mark_col else da[c] for c in range(len(da))]
            if md != dd or not args_equal(ma, want):
                bad.append(k)
                break
    check(
        not bad,
        "%s* buckets equal develop's (descriptors, order, used args, gate column %d = %d)"
        % (prefix_key, args.mark_col, MARK),
    )

    # Scores and rankings on probe rows.
    probes = probe_rows(m["trees"] + s["trees"] + d["trees"], p, args.probes, args.seed)
    n_scores = 0
    mismatch_dev = []
    for key in sorted(develop_keys):
        cands = m["buckets"][key]
        for prefix in probes:
            got_order, got = rank_bucket(m["trees"], prefix, cands)
            want_order, want = rank_bucket(d["trees"], prefix, d["buckets"][key])
            n_scores += len(got)
            if got_order != want_order or not all(
                same_bits(a, b) for a, b in zip(got, want)
            ):
                mismatch_dev.append(key)
                break
    check(
        not mismatch_dev,
        "%s* rankings and scores bitwise equal develop's (%d buckets x %d probes, %d scores)"
        % (prefix_key, len(develop_keys), len(probes), n_scores),
    )
    other_probes = probes[: max(1, args.probes // 4)]
    n_scores = 0
    mismatch_new = []
    for key in sorted(set(s["buckets"]) - develop_keys):
        cands = m["buckets"][key]
        for prefix in other_probes:
            got_order, got = rank_bucket(m["trees"], prefix, cands)
            want_order, want = rank_bucket(s["trees"], prefix, s["buckets"][key])
            n_scores += len(got)
            if got_order != want_order or not all(
                same_bits(a, b) for a, b in zip(got, want)
            ):
                mismatch_new.append(key)
                break
    check(
        not mismatch_new,
        "other buckets' rankings and scores bitwise equal --new's (%d buckets x %d probes, %d scores)"
        % (len(set(s["buckets"]) - develop_keys), len(other_probes), n_scores),
    )

    if args.golden:
        golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))["solvers"][
            args.solver
        ]
        worst = 0.0
        bad_rows = 0
        for row, want in zip(golden["rows"], golden["expected"]):
            got = score_row(m["trees"], [math.nan if v is None else v for v in row])
            err = abs(got - want)
            worst = max(worst, err)
            bad_rows += err > GOLDEN_TOLERANCE
        check(
            bad_rows == 0,
            "%d golden rows within %g of the merged forest (max abs error %.3g, %d outside)"
            % (len(golden["rows"]), GOLDEN_TOLERANCE, worst, bad_rows),
        )
    return report(errors)


def report(errors):
    if errors:
        print("verify FAILED: %d check(s)" % len(errors))
        return 1
    print("verify OK")
    return 0


def verify(args):
    return verify_files(args, args.merged)


# ---------------------------------------------------------------------------
# Golden file
# ---------------------------------------------------------------------------


def merge_golden(args):
    raw = Path(args.new_golden).read_text(encoding="utf-8")
    new = json.loads(raw)
    if json.dumps(new) != raw:
        fail("--new-golden does not round-trip through json.dumps")
    develop = json.loads(Path(args.develop_golden).read_text(encoding="utf-8"))
    dg = develop["solvers"][args.solver]
    ng = new["solvers"][args.solver]
    if dg["feature_count"] != ng["feature_count"]:
        fail("feature_count differs between the golden files")
    feature = args.prob_feat_count + args.mark_col
    added = 0
    for row, expected in zip(dg["rows"], dg["expected"]):
        if (
            row[GOLDEN_CU_COUNT_FEATURE] == args.cu_count
            and row[GOLDEN_WAVE_SIZE_FEATURE] == args.wave_size
        ):
            r = list(row)
            if r[feature] == MARK:
                fail("a develop golden row already carries the gate value")
            r[feature] = MARK
            ng["rows"].append(r)
            ng["expected"].append(expected)
            added += 1
    if not added:
        fail(
            "no develop golden row has cu_count %d and wave_size %d"
            % (args.cu_count, args.wave_size)
        )
    note = (
        " %s rows whose arg column %d (feature %d) is %d are develop's %s rows:"
        " lgbm_pcfg.bin scores those candidates with develop's trees, and their"
        " expected values come from develop's golden file."
        % (args.solver, args.mark_col, feature, MARK, args.arch)
    )
    if note in new["_note"]:
        fail("--new-golden already carries the merge note")
    new["_note"] += note
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(new))
    print("added %d rows; %s now has %d rows" % (added, args.solver, len(ng["rows"])))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument(
            "--solver",
            required=True,
            help="solver section, e.g. ConvHipImplicitGemm3DGroupBwdXdlops",
        )
        p.add_argument(
            "--arch",
            required=True,
            help="architecture prefix of the bucket keys, e.g. gfx1250",
        )
        p.add_argument(
            "--mark-col",
            type=int,
            default=DEFAULT_MARK_COL,
            help="arg column the gate reads; no model may split on it (default %(default)s)",
        )

    def checks(p):
        p.add_argument(
            "--probes",
            type=int,
            default=64,
            help="probe rows per bucket (default %(default)s)",
        )
        p.add_argument(
            "--seed", type=int, default=0, help="probe row seed (default %(default)s)"
        )

    b = sub.add_parser(
        "build",
        help="merge develop's <arch> ranking into the converter's lgbm_pcfg.bin",
    )
    common(b)
    checks(b)
    b.add_argument(
        "--new",
        required=True,
        type=Path,
        help="lgbm_pcfg.bin from convert_lgbm_binary.py",
    )
    b.add_argument(
        "--develop", required=True, type=Path, help="develop's lgbm_pcfg.bin"
    )
    b.add_argument("--out", required=True, type=Path)
    b.add_argument(
        "--golden", type=Path, help="golden file (merged) to score with the output"
    )
    b.set_defaults(run=build)

    v = sub.add_parser("verify", help="check a merged lgbm_pcfg.bin")
    common(v)
    checks(v)
    v.add_argument("--merged", required=True, type=Path)
    v.add_argument("--new", required=True, type=Path)
    v.add_argument("--develop", required=True, type=Path)
    v.add_argument(
        "--golden",
        type=Path,
        help="golden file (merged) to score with the merged forest",
    )
    v.set_defaults(run=verify)

    g = sub.add_parser(
        "golden", help="add develop's <arch> golden rows to the new golden file"
    )
    common(g)
    g.add_argument("--new-golden", required=True, type=Path)
    g.add_argument("--develop-golden", required=True, type=Path)
    g.add_argument("--out", required=True, type=Path)
    g.add_argument(
        "--cu-count",
        required=True,
        type=int,
        help="cu_count of <arch> (problem feature 19)",
    )
    g.add_argument(
        "--wave-size",
        required=True,
        type=int,
        help="wave_size of <arch> (problem feature 20)",
    )
    g.add_argument(
        "--prob-feat-count",
        type=int,
        default=cv.NUM_BASE_PROB_FEATURES,
        help="problem features of the solver (default %(default)s)",
    )
    g.set_defaults(run=merge_golden)

    args = ap.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
