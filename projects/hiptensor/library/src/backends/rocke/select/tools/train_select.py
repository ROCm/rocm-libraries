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

Input: JSON lines written by hipTensor (HIPTENSOR_ROCKE_SELECT_DUMP): the library's own integer
features of every candidate of a problem (no re-implementation, so no drift between training and
inference), with the time of the measured selection ("us", HIPTENSOR_ROCKE_AUTOTUNE_TOPK=all) or
without a time (plans that are not measured). --cold adds the cold times of sweep_driver --cold
(a cache-evicting read before every timed call), which then replace the dump's times; a candidate
whose output disagrees with the reference candidate's (rel_dev > --max-dev, or elements not
finite) or that did not run is dropped and reported.

  train_select.py train   --dumps 'GLOB' [--cold 'GLOB'] [--calib 'GLOB'] --arch gfx950
                          --models OUTDIR --report FILE.json
  train_select.py eval    --dumps 'GLOB' [--cold 'GLOB'] --models OUTDIR/gfx950 --report FILE.json
  train_select.py propose --dumps 'GLOB' [--cold 'GLOB'] --pool 'GLOB' --n N --out IDS
                          [--prev-models DIR] --report FILE.json

Target per candidate: log2(time / best time of the problem), clipped at CLIP. One LightGBM model
per cell (contraction.<h|s|c|d|z>[.<f32 recipe>], reduction.<s|d>), deterministic training. The
objective and the model size are chosen per cell on held-out problems (GroupKFold by problem;
efficiency = best time / time of the pick, per problem) from GRID: rank (LambdaRank) and xendcg
rank the candidates of a problem (relevance levels of 1/REL_STEPS log2 slowdown below the best
candidate, the problems as query groups), l2 regresses the slowdown; --objective fixes one.
A ranking model's scores are exported negated, so that the lowest prediction wins in either case.
The plan's mode is a feature (jit_mode, schema 3): a plan created with HIPTENSOR_JIT_MODE_DEFAULT
specializes its elementwise launches with the JIT, which makes the candidates of the GEMM routes
(their operand packing and epilogue launches) faster than in a HIPTENSOR_JIT_MODE_NONE plan. A
corpus is swept in both modes: the JIT-mode sweep (sweep_driver --jit) under the problems' ids, the
NONE-mode sweep under "<id>#none" (sweep_driver --tag-suffix '#none'), each with the times of its
own mode. A problem with a JIT candidate and no NONE-mode sweep also trains as a problem "<tag>#none"
of its other candidates with the JIT-mode times (the earlier approximation; --no-none-copies turns
it off).

--config objective:TREESxLEAVES (with --margin) trains that configuration without the search and
the cross-validation (a retraining whose configuration was chosen before; judge it on held-out
problems). --ensemble K averages K models of the chosen configuration, each trained on a bootstrap sample of the
problems (their trees concatenated, leaf values divided by K): a lower-variance ranking where routes
trade places near the decision boundary.

--calib names the dumps of the problems that stand for the problems the library will see (a
space-filling round): the model size, the objective and the margin are chosen on the held-out
folds' predictions of those problems only, while every round trains the models (the later,
model-guided rounds oversample the decision boundaries, where rocKE's order is weaker than on
typical problems).

propose (model-guided sweep rounds): bagged models (bootstrap over problems) score every
candidate of every problem of an untimed pool dump; the problems where the bags disagree on the
pick, then those whose two best predictions are closest (the decision boundary), are chosen for
the next timing round, with a share of random pool problems for exploration. With --prev-models
it also reports how many pool picks changed against the previous round's models (stop when the
boundary is stable).

Model text (route table value select.model.<cell>, read by rocke_select_model.cpp):
  v3;schema=3;routes=<route>,...;margin=<m>;gemm=<sha256|->;features=<names>;trees=<n>|<tree>|...
  tree: <leaves>;<split features>;<thresholds>;<left>;<right>;<leaf values>
routes: the Route values (rocke_plan.hpp) of the training rows; the library scores exactly those
routes with the model and orders the others by its rules. margin: the model's best candidate
replaces the first candidate of the static order only when its prediction is lower by more than
the margin (chosen per cell on the held-out folds: where rocKE's order is already good, a model
pick that is not clearly better keeps it). gemm: when the features include
lg_pred_ns (rocKE's tensor GETT planner's predicted time), the SHA-256 of the index of the GEMM
selection models the planner read when the dumps were written (--gemm-digest: the digest of
<pack dir>/gemm_models/index.txt of the build that wrote them); the library does not use the model
with other ones, and the pack build refuses the pair.

Next to each model, <cell>.check holds CHECK_ROWS feature vectors of the training rows (every route
the model scores among them) with the exported model's prediction (this file's walker) and
LightGBM's: the library's CPU test (test/00_unit/rocke_select_model_test.cpp) checks that its walk of
the shipped model gives the first exactly and the second within the compact export's rounding.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

SCHEMA = 3
CLIP = 4.0
# feature vectors per model in its .check file
CHECK_ROWS = 96
GRID = [("rank", 150, 15), ("rank", 300, 31), ("xendcg", 300, 31), ("l2", 300, 31)]
# ranking relevance: REL_TOP for the best candidate, one level less per 1/REL_STEPS log2 slowdown
REL_STEPS = 16
REL_TOP = 30
OBJECTIVE = "rank"
ROUTE_NAMES = {
    0: "elementwise",
    1: "reduction",
    2: "tuned_gett",
    3: "h16",
    4: "x6",
    5: "cf32",
    6: "gett64",
    7: "ring_gett",
    8: "direct_gett",
    9: "gett32",
    10: "tuned_reduction",
    11: "jit_gett",
}


def read_lines(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line of a killed job


def load(pattern, cold=None, max_dev=0.05, report=None, timed=True):
    """One row per (problem, candidate) with features and a time ("us").

    Without `cold`: the dump's measured times (median of repeated runs). With `cold`: the dump's
    features (timed or not) and the cold times; disagreeing / failed candidates are dropped.
    timed=False: every candidate with features (no time needed; a pool)."""
    rows = {}
    for r in read_lines(pattern):
        key = (r["tag"], r["key"])
        rows.setdefault(key, []).append(r)
    out = []
    dropped = defaultdict(int)
    if cold is None:
        for key, rs in rows.items():
            r = dict(rs[-1])
            us = [x["us"] for x in rs if "us" in x]
            if us:
                r["us"] = float(np.median(us))
            elif timed:
                continue
            out.append(r)
    else:
        times = {}
        for c in read_lines(cold):
            times.setdefault((c["tag"], c["key"]), []).append(c)
        for key, rs in rows.items():
            r = dict(rs[-1])
            cs = times.get(key)
            if not cs:
                dropped["no cold time"] += 1
                continue
            c = cs[-1]
            r["cold"] = c
            if c.get("status") != "HIPTENSOR_STATUS_SUCCESS" or c["us"] <= 0:
                dropped["did not run"] += 1
                continue
            if (
                c.get("bad", 0) > 0
                or c.get("rel_dev", -1) < 0
                or c["rel_dev"] > max_dev
            ):
                dropped["output disagrees with the reference"] += 1
                if report is not None:
                    report.setdefault("disagreeing", []).append(
                        {k: c.get(k) for k in ("tag", "key", "rel_dev", "bad", "ref")}
                    )
                continue
            r["us"] = float(c["us"])
            out.append(r)
    if report is not None:
        report["dropped"] = dict(dropped)
        devs = defaultdict(list)
        for r in out:
            if "cold" in r and r["cold"]["key"] != r["cold"].get("ref"):
                devs[ROUTE_NAMES.get(r["f"]["route"], r["f"]["route"])].append(
                    r["cold"]["rel_dev"]
                )
        report["rel_dev_by_route"] = {
            k: {"n": len(v), "p50": float(np.median(v)), "max": float(np.max(v))}
            for k, v in sorted(devs.items())
        }
    return out


JIT_ROUTE = 11


def with_none_copies(rows):
    """`rows` plus, for every problem with a JIT candidate that has no NONE-mode sweep of its
    own ("<tag>#none"), its other candidates as problem "<tag>#none" (the selection of a plan
    without the JIT, with the JIT-mode times)."""
    tags = set(r["tag"] for r in rows)
    jit = set(
        r["tag"]
        for r in rows
        if r["f"]["route"] == JIT_ROUTE and r["tag"] + "#none" not in tags
    )
    extra = [
        dict(r, tag=r["tag"] + "#none")
        for r in rows
        if r["tag"] in jit and r["f"]["route"] != JIT_ROUTE
    ]
    return rows + extra


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
    slow = 1.0 / eff
    return dict(
        n=int(len(eff)),
        geomean=float(np.exp(np.mean(np.log(eff)))),
        mean=float(np.mean(eff)),
        p10=float(np.percentile(eff, 10)),
        min=float(np.min(eff)),
        top1=float(np.mean(eff >= 0.999)),
        ge95=float(np.mean(eff >= 0.95)),
        # the same as slowdowns against the best candidate (pick / best)
        slow_gm=float(np.exp(np.mean(np.log(slow)))),
        slow_p90=float(np.percentile(slow, 90)),
        slow_max=float(np.max(slow)),
    )


def static_pick(rs):
    return min(rs, key=lambda r: r["static_rank"])


def library_pick(rs):
    """The candidate the library's selection put first when the dump was written (the pack's
    models and rules of that build)."""
    return min(rs, key=lambda r: r["rank"])


def matrix(rows, names):
    return np.array([[r["f"][n] for n in names] for r in rows], dtype=np.float64)


def target(rows, probs):
    best = {t: min(r["us"] for r in rs) for t, rs in probs.items()}
    return np.array([min(CLIP, math.log2(r["us"] / best[r["tag"]])) for r in rows])


def relevance(y):
    return np.clip(REL_TOP - np.floor(y * REL_STEPS), 0, REL_TOP).astype(int)


class Ranked:
    """A LambdaRank model whose predict() is the negated ranking score (lower is better)."""

    def __init__(self, ranker):
        self.ranker = ranker
        self.booster_ = ranker.booster_

    def predict(self, X):
        return -self.ranker.predict(X)


def fit(X, y, trees, leaves, seed=7, groups=None, objective=None):
    """groups: the problem of every row (the ranking objectives' query groups); objective:
    rank, xendcg or l2 (default OBJECTIVE)."""
    import lightgbm as lgb

    objective = objective or OBJECTIVE
    if objective in ("rank", "xendcg"):
        assert groups is not None, "the ranking objective needs the problems"
        order = sorted(range(len(groups)), key=lambda i: groups[i])
        sizes = []
        last = None
        for i in order:
            if groups[i] != last:
                sizes.append(0)
                last = groups[i]
            sizes[-1] += 1
        m = lgb.LGBMRanker(
            objective="lambdarank" if objective == "rank" else "rank_xendcg",
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
            random_state=seed,
            verbose=-1,
            lambdarank_truncation_level=3,
        )
        m.fit(X[order], relevance(y[order]), group=sizes)
        return Ranked(m)
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
        random_state=seed,
        verbose=-1,
    )
    m.fit(X, y)
    return m


class Bagged:
    """The mean of several models' predictions."""

    def __init__(self, models):
        self.models = models
        self.boosters = [m.booster_ for m in models]

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def fit_bagged(X, y, trees, leaves, groups, objective, bags, seed=7):
    """`bags` models, each on a bootstrap sample of the problems (a copy of a problem is a group
    of its own)."""
    if bags <= 1:
        return fit(X, y, trees, leaves, seed=seed, groups=groups, objective=objective)
    tags = sorted(set(groups))
    index = defaultdict(list)
    for i, t in enumerate(groups):
        index[t].append(i)
    models = []
    for b in range(bags):
        rng = random.Random(seed * 1000 + b)
        idx, bgroups = [], []
        for c, t in enumerate(tags[rng.randrange(len(tags))] for _ in tags):
            idx += index[t]
            bgroups += ["%s#%d" % (t, c)] * len(index[t])
        models.append(
            fit(
                X[idx],
                y[idx],
                trees,
                leaves,
                seed=seed + b,
                groups=np.array(bgroups),
                objective=objective,
            )
        )
    return Bagged(models)


def pick_by(scores, margin=0.0):
    """The library's pick (rocke_select.cpp): the lowest prediction, or the first candidate of the
    static order when the lowest prediction does not beat its own by more than `margin`.
    """

    def pick(rs):
        best = min(rs, key=lambda r: (scores[id(r)], r["static_rank"]))
        anchor = min(rs, key=lambda r: r["static_rank"])
        if margin > 0.0 and scores[id(anchor)] - scores[id(best)] <= margin:
            return anchor
        return best

    return pick


def choose_margin(probs, scores):
    """The margin that selects best on the held-out folds' predictions: 0 or a quantile of the
    problems' gaps between the static order's first candidate and the model's best (ties: the
    smaller margin). Returned as the text the model carries and its value."""
    gaps = []
    for rs in probs.values():
        anchor = min(rs, key=lambda r: r["static_rank"])
        gaps.append(scores[id(anchor)] - min(scores[id(r)] for r in rs))
    pos = np.array([g for g in gaps if g > 0])
    options = [0.0]
    if len(pos):
        options += sorted(
            set(float("%.6g" % np.percentile(pos, q)) for q in range(5, 100, 5))
        )
    best = None
    for m in options:
        eff = efficiencies(probs, pick_by(scores, m))
        key = round(float(np.exp(np.mean(np.log(eff)))), 4)
        if best is None or key > best[0]:
            best = (key, m, eff)
    return best[1], best[2]


def cv(rows, probs, names, trees, leaves, folds=5, objective=None, bags=1):
    from sklearn.model_selection import GroupKFold

    X = matrix(rows, names)
    y = target(rows, probs)
    groups = np.array([r["tag"] for r in rows])
    pred = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        m = fit_bagged(X[tr], y[tr], trees, leaves, groups[tr], objective, bags)
        pred[te] = m.predict(X[te])
    scores = {id(r): p for r, p in zip(rows, pred)}
    return efficiencies(probs, pick_by(scores)), scores


def route_shares(probs, pick):
    """Share of problems whose pick (or best candidate) is of each route."""
    n = defaultdict(int)
    for rs in probs.values():
        n[ROUTE_NAMES.get(pick(rs)["f"]["route"], "?")] += 1
    total = max(1, len(probs))
    return {k: round(v / total, 4) for k, v in sorted(n.items())}


def best_pick(rs):
    return min(rs, key=lambda r: r["us"])


# ---------------------------------------------------------------------------------- export


def export(
    booster,
    names,
    routes,
    compact=True,
    sign=1.0,
    margin=0.0,
    gemm="-",
):
    """Model text from a LightGBM booster; checks the tree layout the C++ walker assumes.

    compact: thresholds become floor(t) + 0.5 (the same decisions for the integer features) and
    leaf values keep 9 significant digits; the exported text is then the model (the C++ walker and
    predict() below agree with it exactly), and train checks that it picks what LightGBM picks.
    """
    boosters = booster if isinstance(booster, list) else [booster]
    sign = sign / len(boosters)
    infos = []
    for b in boosters:
        dump = b.dump_model()
        # split_feature indices are columns of the training matrix, in the order of `names`
        assert len(dump["feature_names"]) == len(names), "feature count"
        infos += dump["tree_info"]
    trees = []
    for t in infos:
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
        vals = [sign * leaves[i] for i in range(nl)]
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
    if "lg_pred_ns" not in names:
        gemm = "-"
    assert gemm == "-" or (
        len(gemm) == 64 and all(c in "0123456789abcdef" for c in gemm)
    ), "the model reads lg_pred_ns: --gemm-digest <sha256 of gemm_models/index.txt>"
    head = "v3;schema=%d;routes=%s;margin=%.6g;gemm=%s;features=%s;trees=%d" % (
        SCHEMA,
        ",".join(str(r) for r in sorted(routes)),
        margin,
        gemm,
        ",".join(names),
        len(trees),
    )
    return "|".join([head] + trees)


def parse(text):
    """(feature names, trees, routes, margin) of a model text (v1: routes None = the legacy set,
    margin 0)."""
    parts = text.split("|")
    head = parts[0].split(";")
    routes = None
    margin = 0.0
    if head[0] == "v3":
        head = head[:4] + head[5:]
    if head[0] in ("v2", "v3"):
        routes = set(int(x) for x in head[2][len("routes=") :].split(","))
        margin = float(head[3][len("margin=") :])
        head = head[:2] + head[4:]
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
    return names, trees, routes, margin


def predict(model, x):
    """The C++ walker (rocke_select.cpp Model::predict), on one feature dict."""
    names, trees = model[0], model[1]
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


def feature_names(rows):
    return list(rows[0]["f"].keys())


# ---------------------------------------------------------------------------------- commands


def cells_of(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r)
    return cells


def search(a, entry, crow, probs, cprobs, names):
    """The configuration (objective, size) of GRID that selects best on the held-out folds of the
    calibration problems, and its margin; the cross-validated results go to `entry`."""
    best = None
    for objective, trees, leaves in GRID:
        if a.objective != "auto" and objective != a.objective:
            continue
        eff, scores = cv(crow, probs, names, trees, leaves, objective=objective)
        s = summary(eff)
        sc = summary(efficiencies(cprobs, pick_by(scores)))
        entry["grid"]["%s:%dx%d" % (objective, trees, leaves)] = {
            "all": s,
            "calib": sc,
        }
        key = (round(sc["geomean"], 4), round(sc["p10"], 4))
        if best is None or key > best[0]:
            best = (key, objective, trees, leaves, s, scores)
    _, objective, trees, leaves, s, cv_scores = best
    entry["chosen"] = "%s:%dx%d" % (objective, trees, leaves)
    entry["model_cv_no_margin"] = s
    if a.ensemble > 1:
        # the ensemble of the chosen configuration, its margin on its own held-out folds
        eff, cv_scores = cv(
            crow, probs, names, trees, leaves, objective=objective, bags=a.ensemble
        )
        entry["bags"] = a.ensemble
        entry["model_cv_single_no_margin"] = s
        s = summary(eff)
        entry["model_cv_no_margin"] = s
    margin, _ = choose_margin(cprobs, cv_scores)
    s = summary(efficiencies(probs, pick_by(cv_scores, margin)))
    entry["margin"] = margin
    entry["calib_model_cv"] = summary(efficiencies(cprobs, pick_by(cv_scores, margin)))
    entry["model_cv"] = s
    entry["model_cv_routes"] = route_shares(probs, pick_by(cv_scores, margin))
    # per plan mode (HIPTENSOR_JIT_MODE_NONE problems are tagged "<id>#none"), and the worst
    # held-out-fold picks
    pick = pick_by(cv_scores, margin)
    for mode, sel in (("jit", lambda t: not t.endswith("#none")), ("none", lambda t: t.endswith("#none"))):
        sub = {t: rs for t, rs in probs.items() if sel(t)}
        if sub:
            entry.setdefault("model_cv_by_mode", {})[mode] = summary(efficiencies(sub, pick))
            entry.setdefault("static_by_mode", {})[mode] = summary(efficiencies(sub, static_pick))
    worst = sorted(
        ((pick(rs)["us"] / min(r["us"] for r in rs), t, pick(rs)["key"], best_pick(rs)["key"]) for t, rs in probs.items()),
        reverse=True,
    )[:12]
    entry["model_cv_worst"] = [{"slow": round(w[0], 3), "tag": w[1], "pick": w[2], "best": w[3]} for w in worst]
    return objective, trees, leaves, margin


def check_rows(crow, routes, n=CHECK_ROWS, seed=11):
    """n training rows for the .check file: each route the model scores, then a deterministic
    sample of the rest (distinct feature vectors)."""
    rng = random.Random(seed)
    seen, out = set(), []
    by_route = defaultdict(list)
    for r in crow:
        by_route[int(r["f"]["route"])].append(r)
    pool = list(crow)
    rng.shuffle(pool)
    order = []
    for rt in sorted(by_route):
        rs = list(by_route[rt])
        rng.shuffle(rs)
        order += rs[:4]
    for r in order + pool:
        key = tuple(sorted(r["f"].items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= n:
            break
    return out


def write_check(path, model, names, rows, ref):
    """<cell>.check: '# features <names>' then per row '<walker> <lightgbm> <values>'."""
    with open(path, "w", newline="\n") as f:
        f.write("# features %s\n" % ",".join(names))
        for r, lg in zip(rows, ref):
            f.write(
                "%s %s %s\n"
                % (
                    repr(float(predict(model, r["f"]))),
                    repr(float(lg)),
                    " ".join(str(int(r["f"][n])) for n in names),
                )
            )


def cmd_train(a):
    report = {"dumps": a.dumps, "cold": a.cold, "cells": {}}
    rows = load(a.dumps, a.cold, a.max_dev, report)
    report["rows"] = len(rows)
    calib = None
    if a.calib:
        calib = set(r["tag"] for r in read_lines(a.calib))
        report["calib"] = {"dumps": a.calib, "problems": len(calib)}
    out_dir = os.path.join(a.models, a.arch)
    os.makedirs(out_dir, exist_ok=True)
    only = set(a.cells.split(",")) if a.cells else None
    for cell, crow in sorted(cells_of(rows).items()):
        if only is not None and cell not in only:
            continue
        if a.none_copies:
            n0 = len(set(r["tag"] for r in crow))
            crow = with_none_copies(crow)
            report.setdefault("none_copies", {})[cell] = (
                len(set(r["tag"] for r in crow)) - n0
            )
        probs = by_problem(crow)
        crow = [r for r in crow if r["tag"] in probs]
        names = feature_names(crow)
        routes = sorted(set(int(r["f"]["route"]) for r in crow))
        st = summary(efficiencies(probs, static_pick))
        # the training domain is what the API admits: no contraction has a batch mode (a mode of
        # A, B and D; hiptensorCreateContraction refuses them)
        assert not any(r["f"].get("modes_l", 0) for r in crow), "a row with a batch mode"
        entry = {
            "problems": len(probs),
            "rows": len(crow),
            "routes": [ROUTE_NAMES.get(r, r) for r in routes],
            "static": st,
            "library_at_sweep": summary(efficiencies(probs, library_pick)),
            "oracle_routes": route_shares(probs, best_pick),
            "grid": {},
        }
        if len(probs) < a.min_problems:
            entry["skipped"] = "too few problems"
            report["cells"][cell] = entry
            print(cell, entry)
            continue
        # the problems the choices are made on (--calib), and their static order's selection
        cprobs = (
            probs
            if calib is None
            else {t: rs for t, rs in probs.items() if t.split("#")[0] in calib}
        )
        if len(cprobs) < a.min_problems:
            cprobs = probs
        entry["calib_problems"] = len(cprobs)
        entry["calib_static"] = summary(efficiencies(cprobs, static_pick))
        if a.config:
            objective, size = a.config.split(":")
            trees, leaves = (int(x) for x in size.split("x"))
            margin = a.margin
            entry["chosen"] = a.config + " (given: no search, no cross-validation)"
            entry["margin"] = margin
            if a.ensemble > 1:
                entry["bags"] = a.ensemble
        else:
            objective, trees, leaves, margin = search(
                a, entry, crow, probs, cprobs, names
            )
        X = matrix(crow, names)
        y = target(crow, probs)
        groups = np.array([r["tag"] for r in crow])
        m = fit_bagged(X, y, trees, leaves, groups, objective, a.ensemble)
        boosters = m.boosters if isinstance(m, Bagged) else m.booster_
        sign = -1.0 if objective in ("rank", "xendcg") else 1.0
        entry["objective"] = objective
        assert np.array_equal(X, np.floor(X)), "features must be integers"
        ref = m.predict(X)
        # the full-precision export reproduces LightGBM bit for bit (an ensemble: up to the
        # order of the sums)
        full = parse(
            export(
                boosters,
                names,
                routes,
                compact=False,
                sign=sign,
                margin=margin,
                gemm=a.gemm_digest or "-",
            )
        )
        mine = np.array([predict(full, r["f"]) for r in crow])
        exact = bool(np.array_equal(ref, mine))
        entry["export_exact"] = exact
        entry["export_max_abs_diff"] = float(np.max(np.abs(ref - mine)))
        assert exact or (a.ensemble > 1 and entry["export_max_abs_diff"] < 1e-9), entry[
            "export_max_abs_diff"
        ]
        # the compact export (the shipped model) picks what LightGBM picks
        text = export(
            boosters,
            names,
            routes,
            sign=sign,
            margin=margin,
            gemm=a.gemm_digest or "-",
        )
        model = parse(text)
        small = np.array([predict(model, r["f"]) for r in crow])
        entry["compact_max_abs_diff"] = float(np.max(np.abs(ref - small)))
        s_ref = {id(r): v for r, v in zip(crow, ref)}
        s_small = {id(r): v for r, v in zip(crow, small)}
        same = sum(
            1
            for rs in probs.values()
            if pick_by(s_ref, margin)(rs) is pick_by(s_small, margin)(rs)
        )
        entry["compact_same_pick"] = "%d/%d" % (same, len(probs))
        assert entry["compact_max_abs_diff"] < 1e-6 and same >= 0.999 * len(
            probs
        ), entry
        path = os.path.join(out_dir, cell + ".model")
        with open(path, "w", newline="\n") as f:
            f.write(text + "\n")
        entry["model_bytes"] = len(text)
        # the check vectors of the library's test: the exported model's and LightGBM's predictions
        crows = check_rows(crow, routes)
        cref = m.predict(matrix(crows, names))
        write_check(os.path.join(out_dir, cell + ".check"), model, names, crows, cref)
        entry["check_rows"] = len(crows)
        report["cells"][cell] = entry
        print(
            cell,
            json.dumps(
                {
                    k: entry.get(k)
                    for k in (
                        "problems",
                        "routes",
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
    """Plans made with the models in the pack: the library's pick (the candidate the plan took,
    "pick", else rank 0 of the library's order) vs the best timed candidate; the C++ scores vs
    this file's walker; slowdown distributions per cell and per route of the pick.
    """
    report = {"dumps": a.dumps, "cold": a.cold, "cells": {}}
    rows = load(a.dumps, a.cold, a.max_dev, report)

    def lib_pick(rs):
        picked = [r for r in rs if r.get("pick") == 1]
        return picked[0] if picked else min(rs, key=lambda r: r["rank"])

    for cell, crow in sorted(cells_of(rows).items()):
        probs = by_problem(crow)
        # problems whose picked candidate has a time (it may have been dropped)
        probs = {
            t: rs
            for t, rs in probs.items()
            if any(r.get("pick") == 1 for r in rs) or not any("pick" in r for r in rs)
        }
        sel = summary(efficiencies(probs, lib_pick))
        st = summary(efficiencies(probs, static_pick))
        entry = {
            "problems": len(probs),
            "selector": sel,
            "static": st,
            "oracle_routes": route_shares(probs, best_pick),
            "pick_routes": route_shares(probs, lib_pick),
        }
        sources = defaultdict(int)
        for rs in probs.values():
            sources[lib_pick(rs)["source"]] += 1
        entry["sources"] = dict(sources)
        worst = sorted(
            (
                (
                    lib_pick(rs)["us"] / min(r["us"] for r in rs),
                    t,
                    lib_pick(rs)["key"],
                    best_pick(rs)["key"],
                )
                for t, rs in probs.items()
            ),
            reverse=True,
        )[:10]
        entry["worst"] = [
            {"slow": round(w[0], 4), "tag": w[1], "pick": w[2], "best": w[3]}
            for w in worst
        ]
        path = os.path.join(a.models, cell + ".model")
        if os.path.exists(path):
            model = parse(open(path).read().strip())
            scored = [r for r in crow if "score" in r]
            same = sum(1 for r in scored if predict(model, r["f"]) == r["score"])
            entry["score_parity"] = "%d/%d" % (same, len(scored))
        report["cells"][cell] = entry
        print(
            cell,
            json.dumps(
                {
                    k: entry[k]
                    for k in (
                        "problems",
                        "selector",
                        "static",
                        "pick_routes",
                        "oracle_routes",
                    )
                }
            ),
        )
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1)


def distinct(rs, names):
    """The candidates of a problem with distinct features (the model cannot tell candidates with
    equal features apart: one of them stands for all)."""
    seen = {}
    for r in rs:
        seen.setdefault(tuple(r["f"][n] for n in names), r)
    return list(seen.values())


def cmd_propose(a):
    """The next sweep round's problems from an untimed pool (see the top of the file)."""
    report = {"dumps": a.dumps, "cold": a.cold, "pool": a.pool, "cells": {}}
    rows = load(a.dumps, a.cold, a.max_dev, report)
    pool = load(a.pool, timed=False)
    seen = set(r["tag"] for r in rows)
    pool_cells = cells_of([r for r in pool if r["tag"] not in seen])
    train_cells = cells_of(rows)
    rng = random.Random(a.seed)
    ranked_by_cell = {}
    for cell, prow in sorted(pool_cells.items()):
        pprobs = by_problem(prow)
        entry = {"pool_problems": len(pprobs)}
        crow = train_cells.get(cell, [])
        probs = by_problem(crow)
        crow = [r for r in crow if r["tag"] in probs]
        ranked = []
        if len(probs) >= a.min_problems:
            names = feature_names(crow)
            pprobs = {t: distinct(rs, names) for t, rs in pprobs.items()}
            pprobs = {t: rs for t, rs in pprobs.items() if len(rs) >= 2}
            entry["pool_problems_with_a_choice"] = len(pprobs)
            X = matrix(crow, names)
            y = target(crow, probs)
            tags = sorted(probs)
            picks = defaultdict(list)
            scores = defaultdict(lambda: defaultdict(list))
            for b in range(a.bags):
                brng = random.Random(a.seed * 1000 + b)
                boot = [tags[brng.randrange(len(tags))] for _ in tags]
                weight = defaultdict(int)
                for t in boot:
                    weight[t] += 1
                idx = [i for i, r in enumerate(crow) for _ in range(weight[r["tag"]])]
                # a bootstrap copy of a problem is a group of its own
                copy = defaultdict(int)
                bgroups = []
                last = None
                for i in idx:
                    t = crow[i]["tag"]
                    if t != last:
                        copy[t] += 1
                        last = t
                    bgroups.append("%s#%d" % (t, copy[t]))
                m = fit(
                    X[idx],
                    y[idx],
                    150,
                    15,
                    seed=7 + b,
                    groups=np.array(bgroups),
                    objective="rank",
                )
                for t, rs in pprobs.items():
                    pred = m.predict(matrix(rs, names))
                    for r, p in zip(rs, pred):
                        scores[t][r["key"]].append(p)
                    picks[t].append(rs[int(np.argmin(pred))]["key"])
            for t, rs in pprobs.items():
                counts = defaultdict(int)
                for k in picks[t]:
                    counts[k] += 1
                agree = max(counts.values()) / a.bags
                mean = sorted(np.mean(v) for v in scores[t].values())
                margin = mean[1] - mean[0]
                ranked.append((agree, margin, t))
            ranked.sort()
            entry["disagreeing"] = sum(1 for x in ranked if x[0] < 1.0)
            entry["margin_p10"] = (
                float(np.percentile([x[1] for x in ranked], 10)) if ranked else None
            )
            if a.prev_models:
                path = os.path.join(a.prev_models, cell + ".model")
                if os.path.exists(path):
                    # the pool problems whose pick (by its features) differs between the previous
                    # round's model and a model of every round's data
                    # a pick changes when the new model predicts the previous pick slower than
                    # its own by more than 2 % (equal or near-equal candidates do not count; an
                    # L2 model of the slowdown judges, whatever the objective)
                    prev = parse(open(path).read().strip())
                    m = fit(X, y, 150, 15, objective="l2")
                    changed = 0
                    for t, rs in pprobs.items():
                        pred = m.predict(matrix(rs, names))
                        old = int(np.argmin([predict(prev, r["f"]) for r in rs]))
                        new = int(np.argmin(pred))
                        changed += pred[old] - pred[new] > math.log2(1.02)
                    entry["picks_changed_vs_prev"] = round(
                        changed / max(1, len(pprobs)), 4
                    )
        else:
            entry["note"] = "too few trained problems: random pool problems"
            entry["disagreeing"] = len(pprobs)
        ranked_by_cell[cell] = (ranked, sorted(pprobs))
        report["cells"][cell] = entry
    # the round's problems go to the cells in proportion to their disagreeing problems (at least a
    # tenth of an equal share each)
    total = sum(max(e["disagreeing"], 1) for e in report["cells"].values())
    floor = a.n / max(1, len(ranked_by_cell)) / 10
    chosen = []
    for cell, (ranked, tags) in ranked_by_cell.items():
        entry = report["cells"][cell]
        quota = int(round(max(floor, a.n * max(entry["disagreeing"], 1) / total)))
        quota = min(quota, len(tags))
        n_explore = int(round(quota * a.explore)) if ranked else quota
        take = [t for _, _, t in ranked[: quota - n_explore]]
        taken = set(take)
        rest = [t for t in tags if t not in taken]
        rng.shuffle(rest)
        take += rest[: quota - len(take)]
        entry["chosen"] = len(take)
        chosen += take
        print(cell, json.dumps(entry))
    with open(a.out, "w", newline="\n") as f:
        f.write("\n".join(chosen) + "\n")
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval", "propose"))
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--cold", default=None, help="sweep_driver --cold output (glob)")
    ap.add_argument(
        "--calib", default="", help="train: dumps of the calibration problems (glob)"
    )
    ap.add_argument("--max-dev", type=float, default=0.05)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--models", default="")
    ap.add_argument("--cells", default="", help="train only these cells (comma list)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--min-problems", type=int, default=40)
    ap.add_argument("--pool", default="")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bags", type=int, default=8)
    ap.add_argument("--explore", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--prev-models", default="")
    ap.add_argument(
        "--objective", choices=("auto", "rank", "xendcg", "l2"), default="auto"
    )
    ap.add_argument(
        "--ensemble", type=int, default=1, help="train: models averaged per cell"
    )
    ap.add_argument(
        "--config", default="", help="train: objective:TREESxLEAVES, no search"
    )
    ap.add_argument("--margin", type=float, default=0.0, help="train: with --config")
    ap.add_argument(
        "--gemm-digest",
        default="",
        help="train: sha256 of the dumps' build's gemm_models/index.txt (models that read "
        "lg_pred_ns name it)",
    )
    ap.add_argument(
        "--no-none-copies",
        dest="none_copies",
        action="store_false",
        help="train: no copies of the JIT problems without the JIT candidate",
    )
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "propose": cmd_propose}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
