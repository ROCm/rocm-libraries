#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
###############################################################################
"""Train, export and evaluate the rocKE backend's kernel selection models.

Input: JSON lines written by hipTensor's measured selection (HIPTENSOR_ROCKE_SELECT_DUMP) with
every candidate of every problem timed (HIPTENSOR_ROCKE_AUTOTUNE_TOPK=all). The features are the
library's own integer features (no re-implementation, so no drift between training and
inference).

  train_select.py train --dumps 'GLOB' --arch gfx950 --models OUTDIR --report FILE.json
  train_select.py eval  --dumps 'GLOB' --models OUTDIR/gfx950 --report FILE.json

Target per candidate: log2(time / best time of the problem), clipped at CLIP. One LightGBM L2
regressor per cell (contraction.<h|s|c|d|z>, reduction.<s|d>), deterministic training. Selection
quality is measured on held-out problems (GroupKFold by problem): efficiency = best time /
time of the pick, per problem.

Model text (route table value select.model.<cell>, read by rocke_select.cpp):
  v1;schema=1;features=<names>;trees=<n>|<tree>|...
  tree: <leaves>;<split features>;<thresholds>;<left>;<right>;<leaf values>
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

SCHEMA = 1
CLIP = 4.0
GRID = [(150, 15), (300, 15), (300, 31)]


def load(pattern):
    rows = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["tag"], r["key"])
                rows.setdefault(key, []).append(r)
    # one row per (problem, candidate): the median time of repeated runs
    out = []
    for key, rs in rows.items():
        r = dict(rs[-1])
        r["us"] = float(np.median([x["us"] for x in rs]))
        out.append(r)
    return out


def by_problem(rows):
    probs = defaultdict(list)
    for r in rows:
        probs[r["tag"]].append(r)
    return {t: rs for t, rs in probs.items() if len(rs) >= 2}


def efficiencies(probs, pick):
    """pick(rows) -> row; per problem best / picked time."""
    eff = []
    for tag, rs in sorted(probs.items()):
        best = min(r["us"] for r in rs)
        eff.append(best / pick(rs)["us"])
    return np.array(eff)


def summary(eff):
    if len(eff) == 0:
        return {}
    return dict(
        n=int(len(eff)),
        geomean=float(np.exp(np.mean(np.log(eff)))),
        mean=float(np.mean(eff)),
        p10=float(np.percentile(eff, 10)),
        min=float(np.min(eff)),
        top1=float(np.mean(eff >= 0.999)),
        ge95=float(np.mean(eff >= 0.95)),
    )


def static_pick(rs):
    return min(rs, key=lambda r: r["static_rank"])


def matrix(rows, names):
    return np.array([[r["f"][n] for n in names] for r in rows], dtype=np.float64)


def target(rows, probs):
    best = {t: min(r["us"] for r in rs) for t, rs in probs.items()}
    return np.array([min(CLIP, math.log2(r["us"] / best[r["tag"]])) for r in rows])


def fit(X, y, trees, leaves):
    import lightgbm as lgb

    m = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=trees,
        num_leaves=leaves,
        learning_rate=0.05,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        deterministic=True,
        force_col_wise=True,
        num_threads=1,
        random_state=7,
        verbose=-1,
    )
    m.fit(X, y)
    return m


def pick_by(scores):
    def pick(rs):
        return min(rs, key=lambda r: (scores[id(r)], r["static_rank"]))

    return pick


def cv(rows, probs, names, trees, leaves, folds=5):
    from sklearn.model_selection import GroupKFold

    X = matrix(rows, names)
    y = target(rows, probs)
    groups = np.array([r["tag"] for r in rows])
    pred = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        m = fit(X[tr], y[tr], trees, leaves)
        pred[te] = m.predict(X[te])
    scores = {id(r): p for r, p in zip(rows, pred)}
    return efficiencies(probs, pick_by(scores)), scores


# ---------------------------------------------------------------------------------- export


def export(booster, names, compact=True):
    """Model text from a LightGBM booster; checks the tree layout the C++ walker assumes.

    compact: thresholds become floor(t) + 0.5 (the same decisions for the integer features) and
    leaf values keep 9 significant digits; the exported text is then the model (the C++ walker and
    predict() below agree with it exactly), and train checks that it picks what LightGBM picks.
    """
    dump = booster.dump_model()
    # split_feature indices are columns of the training matrix, in the order of `names`
    assert len(dump["feature_names"]) == len(names), "feature count"
    trees = []
    for t in dump["tree_info"]:
        nodes = {}
        leaves = {}

        def walk(n):
            if "leaf_index" in n:
                leaves[n["leaf_index"]] = n["leaf_value"]
                return ~n["leaf_index"]
            if "split_index" not in n:  # one-leaf tree
                leaves[0] = n["leaf_value"]
                return ~0
            i = n["split_index"]
            assert n["decision_type"] == "<=", n["decision_type"]
            assert n.get("missing_type", "None") in ("None", "NaN"), n.get(
                "missing_type"
            )
            left = walk(n["left_child"])
            right = walk(n["right_child"])
            nodes[i] = (n["split_feature"], float(n["threshold"]), left, right)
            return i

        walk(t["tree_structure"])
        nl = t["num_leaves"]
        assert len(leaves) == nl and len(nodes) == nl - 1
        for i, (f, th, l, r) in nodes.items():
            for c in (l, r):
                assert (c >= 0 and c > i and c < nl - 1) or (
                    c < 0 and ~c < nl
                ), "layout"
        feats = [nodes[i][0] for i in range(nl - 1)]
        ths = [nodes[i][1] for i in range(nl - 1)]
        ls = [nodes[i][2] for i in range(nl - 1)]
        rs = [nodes[i][3] for i in range(nl - 1)]
        vals = [leaves[i] for i in range(nl)]
        if compact:
            th_text = " ".join("%.1f" % (math.floor(x) + 0.5) for x in ths)
            leaf_text = " ".join("%.9g" % float(x) for x in vals)
        else:
            th_text = " ".join(repr(x) for x in ths)
            leaf_text = " ".join(repr(float(x)) for x in vals)
        trees.append(
            "%d;%s;%s;%s;%s;%s"
            % (
                nl,
                " ".join(str(x) for x in feats),
                th_text,
                " ".join(str(x) for x in ls),
                " ".join(str(x) for x in rs),
                leaf_text,
            )
        )
    head = "v1;schema=%d;features=%s;trees=%d" % (SCHEMA, ",".join(names), len(trees))
    return "|".join([head] + trees)


def parse(text):
    parts = text.split("|")
    head = parts[0].split(";")
    names = head[2][len("features=") :].split(",")
    trees = []
    for p in parts[1:]:
        f = p.split(";")
        lst = lambda s, conv: (
            [conv(x) for x in s.split()] if s.strip() else []
        )  # noqa: E731
        trees.append(
            (
                lst(f[1], int),
                lst(f[2], float),
                lst(f[3], int),
                lst(f[4], int),
                lst(f[5], float),
            )
        )
    return names, trees


def predict(model, x):
    """The C++ walker (rocke_select.cpp Model::predict), on one feature dict."""
    names, trees = model
    v = [float(x[n]) for n in names]
    s = 0.0
    for feat, th, left, right, leaf in trees:
        if not feat:
            s += leaf[0]
            continue
        n = 0
        while n >= 0:
            n = left[n] if v[feat[n]] <= th[n] else right[n]
        s += leaf[~n]
    return s


# ---------------------------------------------------------------------------------- commands


def cmd_train(a):
    rows = load(a.dumps)
    cells = defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r)
    report = {"dumps": a.dumps, "rows": len(rows), "cells": {}}
    out_dir = os.path.join(a.models, a.arch)
    os.makedirs(out_dir, exist_ok=True)
    for cell, crow in sorted(cells.items()):
        probs = by_problem(crow)
        crow = [r for r in crow if r["tag"] in probs]
        names = sorted(crow[0]["f"].keys(), key=list(crow[0]["f"].keys()).index)
        st = summary(efficiencies(probs, static_pick))
        entry = {"problems": len(probs), "rows": len(crow), "static": st, "grid": {}}
        if len(probs) < a.min_problems:
            entry["skipped"] = "too few problems"
            report["cells"][cell] = entry
            print(cell, entry)
            continue
        best = None
        for trees, leaves in GRID:
            eff, _ = cv(crow, probs, names, trees, leaves)
            s = summary(eff)
            entry["grid"]["%dx%d" % (trees, leaves)] = s
            key = (round(s["geomean"], 4), round(s["p10"], 4))
            if best is None or key > best[0]:
                best = (key, trees, leaves, s)
        _, trees, leaves, s = best
        entry["chosen"] = "%dx%d" % (trees, leaves)
        entry["model_cv"] = s
        X = matrix(crow, names)
        y = target(crow, probs)
        m = fit(X, y, trees, leaves)
        assert np.array_equal(X, np.floor(X)), "features must be integers"
        ref = m.predict(X)
        # the full-precision export reproduces LightGBM bit for bit
        full = parse(export(m.booster_, names, compact=False))
        mine = np.array([predict(full, r["f"]) for r in crow])
        exact = bool(np.array_equal(ref, mine))
        entry["export_exact"] = exact
        assert exact, float(np.max(np.abs(ref - mine)))
        # the compact export (the shipped model) picks what LightGBM picks
        text = export(m.booster_, names)
        model = parse(text)
        small = np.array([predict(model, r["f"]) for r in crow])
        entry["compact_max_abs_diff"] = float(np.max(np.abs(ref - small)))
        s_ref = {id(r): v for r, v in zip(crow, ref)}
        s_small = {id(r): v for r, v in zip(crow, small)}
        same = sum(
            1 for rs in probs.values() if pick_by(s_ref)(rs) is pick_by(s_small)(rs)
        )
        entry["compact_same_pick"] = "%d/%d" % (same, len(probs))
        assert entry["compact_max_abs_diff"] < 1e-6 and same >= 0.999 * len(
            probs
        ), entry
        path = os.path.join(out_dir, cell + ".model")
        with open(path, "w", newline="\n") as f:
            f.write(text + "\n")
        entry["model_bytes"] = len(text)
        report["cells"][cell] = entry
        print(
            cell,
            json.dumps(
                {
                    k: entry[k]
                    for k in (
                        "problems",
                        "static",
                        "model_cv",
                        "chosen",
                        "export_exact",
                        "compact_same_pick",
                    )
                }
            ),
        )
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1)


def cmd_eval(a):
    """Dumps of plans made with the models in the pack: the selector's pick (rank 0 of the
    library's order) vs the best timed candidate; and the C++ scores vs this file's walker.
    """
    rows = load(a.dumps)
    cells = defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r)
    report = {"dumps": a.dumps, "cells": {}}
    for cell, crow in sorted(cells.items()):
        probs = by_problem(crow)
        sel = summary(efficiencies(probs, lambda rs: min(rs, key=lambda r: r["rank"])))
        st = summary(efficiencies(probs, static_pick))
        entry = {"problems": len(probs), "selector": sel, "static": st}
        sources = defaultdict(int)
        for r in crow:
            sources[r["source"]] += 1
        entry["sources"] = dict(sources)
        path = os.path.join(a.models, cell + ".model")
        if os.path.exists(path):
            model = parse(open(path).read().strip())
            scored = [r for r in crow if "score" in r]
            same = sum(1 for r in scored if predict(model, r["f"]) == r["score"])
            entry["score_parity"] = "%d/%d" % (same, len(scored))
        report["cells"][cell] = entry
        print(cell, json.dumps(entry))
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--models", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--min-problems", type=int, default=40)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
