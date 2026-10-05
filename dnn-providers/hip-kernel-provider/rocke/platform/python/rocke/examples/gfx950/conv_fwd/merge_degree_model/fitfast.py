#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Vectorised fit of the analytic degree model. This is where the eight
constants in model.py come from.

WHY IT LOOKS LIKE THIS. The objective -- geometric mean over shapes of the
*realised fraction* the modelled pick achieves -- is **piecewise constant** in
the parameters. The model's output is an ``argmin`` over seven candidate
degrees, so moving a constant changes nothing at all until it moves far enough
to flip some shape's winner, at which point the score jumps. There is no
gradient anywhere, so gradient descent, Newton, and every smooth optimiser are
unusable; what works is brute force. That means a draw only counts if it lands
in the right cell of a 8-dimensional piecewise-constant surface, which in turn
means the draw count has to be large.

So every Gm-independent quantity is precomputed into dense ``(n_shapes, 7)``
tables once, and a whole batch of parameter draws is then scored with a single
set of numpy ops -- 10^6 draws is routine. The search is random sampling over
log-uniform ranges followed by multiplicative coordinate refinement, restarted
from several seeds because a single run lands in whatever cell it happened to
sample.

Structure under test (see model.py for the derivation and for what each constant
means):

    compute = Kpad * (1 + B/vw)
    memory  = Kpad * tile_m * esize * W / (u**A * reuse**r)
    brake   = (1 + footprint/F)**q
    per_cta = combine(compute, memory) * brake + D        # brake="all"
    cost    = ceil(CTAs/P) * per_cta

Two modelling choices are left open and *fitted*, rather than assumed:

  ``combine``  max (the two buckets overlap, i.e. loads hide under MFMA) or sum
               (they serialise). Which one this kernel does is an empirical
               question, so both are fitted and cross-validation decides.
  ``brake``    where the working-set penalty acts -- "mem" on the DRAM bucket
               only, "all" on the whole per-CTA cost. See ``evaluate()``.

What ships is ``combine="sum"``, ``brake="all"``.

``MODELS`` below is the honesty check: it pins terms to their neutral values to
build models strictly nested inside the full one, so "is this constant earning
its place?" can be answered by cross-validation rather than by argument.

Usage:
    python3 fitfast.py --cv            # select the form by CV, then fit
    python3 fitfast.py                 # fit at the shipped form
"""

import argparse

import numpy as np

import corpus_fwd
from model import DEGREES, LINE_BYTES, VEC_BYTES, admissible

FIELDS = (
    "b_aload_issue",
    "w_memory",
    "r_reuse",
    "d_cta_fixed",
    "cupar",
    "c_footprint",
    "a_util",
    "q_brake",
)


def tabulate(
    cfgs, tile_m=64, tile_n=64, tile_k=64, max_degree=64, form=("taps", "taps")
):
    """Dense (n, 7) tables of everything the cost needs, plus the payoff."""
    n, d = len(cfgs), len(DEGREES)
    z = lambda: np.zeros((n, d))
    ctas, kpad, vw, u, fp, base, scale = (z() for _ in range(7))
    realised = np.zeros((n, d))
    ok = np.zeros((n, d), dtype=bool)
    oracle = np.array([c.oracle for c in cfgs])

    for i, c in enumerate(cfgs):
        cands = set(admissible(c.groups, tile_n, c.y, c.x, max_degree))
        m, es = c.m(), c.esize
        for j, gm in enumerate(DEGREES):
            r = c.realised(gm)
            if gm not in cands or r != r:
                continue
            ok[i, j] = True
            realised[i, j] = max(r, 1e-6)
            ctas[i, j] = -(-m // tile_m) * (c.groups // gm)
            kpad[i, j] = tile_k * -(-(c.y * c.x * gm) // tile_k)
            vw[i, j] = min(gm, VEC_BYTES // es)
            u[i, j] = min(1.0, gm * es / LINE_BYTES)
            # Two independent modelling choices, crossed:
            #
            # footprint -- what the CTA must hold resident. "box" is the real
            #   input window: a tile_m run of M covers min(tile_m, Wo) output
            #   columns and ceil(tile_m/Wo) rows, each reaching s plus the
            #   (X-1)*d dilated tap span. "taps" is the naive tile_m*Y*X, which
            #   is stride- and dilation-blind and so over-charges exactly the
            #   strided large-filter shapes whose windows barely overlap.
            # reuse -- how often a resident line is hit again. "taps" counts the
            #   W-direction tap overlap only; "box" is the full ratio of reads
            #   to distinct elements.
            cols = min(tile_m, c.wo) * c.stride + (c.x - 1) * c.dilation
            rows = -(-tile_m // c.wo) * c.stride + (c.y - 1) * c.dilation
            distinct = cols * rows
            fp[i, j] = (distinct if form[0] == "box" else tile_m * c.y * c.x) * gm * es
            base[i, j] = (
                tile_m * c.y * c.x / distinct
                if form[1] == "box"
                else 1.0 + (c.x - 1) / c.stride
            )
            scale[i, j] = tile_m * es
    return dict(
        ctas=ctas,
        kpad=kpad,
        vw=vw,
        u=u,
        fp=fp,
        base=base,
        scale=scale,
        realised=realised,
        ok=ok,
        oracle=oracle,
    )


def evaluate(T, p, combine="max", brake="mem"):
    """p: (batch, 8) parameters -> (picked realised (batch, n), picked gm).

    ``brake`` decides where the capacity penalty (1 + footprint/F)**q acts.
    "mem" divides the DRAM bucket only -- physically the narrow reading, that a
    working set spilling cache costs extra *traffic*. "all" multiplies the whole
    per-CTA cost -- the reading that an oversized K working set also costs issue
    slots and occupancy. Whether the top degree still pays turns out to be a
    clean monotone function of Y*X alone, so the penalty has to have enough
    leverage to overturn a halved CTA count; under "mem" it never does,
    because the fitted DRAM bucket is an order of magnitude below compute.
    """
    B, W, r, D, P, F, A, q = (p[:, k][:, None, None] for k in range(8))
    ok = T["ok"][None]
    pen = np.power(1.0 + T["fp"][None] / F, q)
    reuse = np.power(T["base"][None], r) * (pen if brake == "mem" else 1.0) ** -1.0
    compute = T["kpad"][None] * (1.0 + B / np.maximum(T["vw"][None], 1e-9))
    memory = (
        T["kpad"][None]
        * T["scale"][None]
        * W
        / np.maximum(np.power(T["u"][None], A) * reuse, 1e-30)
    )
    per = np.maximum(compute, memory) if combine == "max" else compute + memory
    if brake == "all":
        per = per * pen
    per = per + D
    cost = np.ceil(T["ctas"][None] / P) * per
    cost = np.where(ok, cost, np.inf)
    # ties -> smaller degree: DEGREES is ascending and argmin takes the first.
    j = np.argmin(cost, axis=2)
    got = np.take_along_axis(
        T["realised"][None].repeat(p.shape[0], 0), j[..., None], 2
    )[..., 0]
    return got, j


def summarise(got, j, T):
    lg = np.log(np.maximum(got, 1e-9))
    geo = np.exp(lg.mean(axis=1))
    worst = got.min(axis=1)
    deg = np.array(DEGREES)[j]
    exact = (deg == T["oracle"][None]).sum(axis=1)
    below = (got < 0.95).sum(axis=1)
    return geo, worst, exact, below


# Neutral value per field: the setting at which that term drops out of the
# cost entirely. Fixing a field here is what makes the model *nested* inside
# the full one, so a CV comparison measures the term's worth, not a reparam.
NEUTRAL = dict(
    b_aload_issue=0.0,
    w_memory=0.0,
    r_reuse=0.0,
    d_cta_fixed=0.0,
    cupar=1.0,
    c_footprint=1e18,
    a_util=0.0,
    q_brake=1.0,
)

MODELS = {
    # name: fields left free; everything else pinned to NEUTRAL
    "pad": ("cupar",),
    "vec": ("b_aload_issue", "cupar"),
    "vecD": ("b_aload_issue", "cupar", "d_cta_fixed"),
    "vecFP": ("b_aload_issue", "cupar", "w_memory", "c_footprint"),
    "vecFPq": ("b_aload_issue", "cupar", "w_memory", "c_footprint", "q_brake"),
    "vecFPr": ("b_aload_issue", "cupar", "w_memory", "c_footprint", "r_reuse"),
    # The shipped constants have a small B and a very small W next to a large D,
    # which *looks* like the fit collapsed to pure wave-counting -- "waves x
    # (K-extent, braked by footprint)" -- with the vector-width and DRAM terms
    # discarded. These two spell that corner out explicitly so the suspicion can
    # be tested rather than inferred from the magnitudes of constants that are
    # only meaningful as ratios. It does not hold: both "wave" and "waveB" score
    # materially below "full" under cross-validation, so neither dropped term is
    # dead weight. See the case study for the comparison.
    "wave": ("cupar", "c_footprint", "q_brake", "d_cta_fixed"),
    "waveB": ("cupar", "c_footprint", "q_brake", "d_cta_fixed", "b_aload_issue"),
    "full": FIELDS,
}


def sample(rng, batch, free=FIELDS):
    cols = [
        10 ** rng.uniform(-2, 3, batch),  # b_aload_issue
        10 ** rng.uniform(-5, 2, batch),  # w_memory
        rng.uniform(0.0, 3.0, batch),  # r_reuse
        10 ** rng.uniform(0, 8, batch),  # d_cta_fixed
        2 ** rng.uniform(4, 16, batch),  # cupar
        10 ** rng.uniform(2, 9, batch),  # c_footprint
        rng.uniform(0.0, 3.0, batch),  # a_util
        rng.uniform(0.0, 3.0, batch),  # q_brake
    ]
    for k, f in enumerate(FIELDS):
        if f not in free:
            cols[k] = np.full(batch, NEUTRAL[f])
    return np.stack(cols, axis=1)


def search(
    T, draws=1_000_000, seed=0, combine="max", batch=20000, free=FIELDS, brake="mem"
):
    rng = np.random.default_rng(seed)
    best_p, best_key = None, (-1.0, -1.0)
    for _ in range(max(1, draws // batch)):
        p = sample(rng, batch, free)
        got, j = evaluate(T, p, combine, brake)
        geo, worst, _, _ = summarise(got, j, T)
        k = int(np.lexsort((worst, geo))[-1])
        key = (float(geo[k]), float(worst[k]))
        if key > best_key:
            best_p, best_key = p[k].copy(), key

    # coordinate refinement: all one-field multiplicative moves, in parallel
    mults = np.array([0.5, 0.7, 0.85, 0.93, 1.07, 1.18, 1.4, 2.0])
    for _ in range(40):
        cand = [best_p]
        for f in range(len(FIELDS)):
            if FIELDS[f] not in free:
                continue
            for mu in mults:
                q = best_p.copy()
                # the two exponents are additive-domain, not multiplicative
                q[f] = (q[f] + (mu - 1.0) * 0.5) if f in (2, 6, 7) else q[f] * mu
                if q[f] >= 0:
                    cand.append(q)
        p = np.stack(cand)
        got, j = evaluate(T, p, combine, brake)
        geo, worst, _, _ = summarise(got, j, T)
        k = int(np.lexsort((worst, geo))[-1])
        key = (float(geo[k]), float(worst[k]))
        if key <= best_key:
            break
        best_p, best_key = p[k].copy(), key
    return best_p, best_key


def report(label, cfgs, p, combine, max_degree=64, form=("taps", "taps"), brake="mem"):
    if not cfgs:
        return
    T = tabulate(cfgs, max_degree=max_degree, form=form)
    got, j = evaluate(T, p[None], combine, brake)
    geo, worst, exact, below = summarise(got, j, T)
    print(
        f"{label:<17} exact {exact[0]:>3}/{len(cfgs):<3} geomean {geo[0]:.4f}  "
        f"worst {worst[0]:.4f}  below95 {below[0]}"
    )
    return float(geo[0])


def split(configs, frac=0.3, seed=11):
    import random

    by = {}
    for c in configs:
        by.setdefault(c.oracle, []).append(c)
    train, test = [], []
    for o in sorted(by):
        grp = sorted(by[o], key=lambda c: c.name)
        random.Random(seed + o).shuffle(grp)
        cut = int(round(len(grp) * frac))
        test += grp[:cut]
        train += grp[cut:]
    return train, test


def restarts(T, draws, seeds, combine, free=FIELDS, brake="mem"):
    """Best of several independent searches.

    The objective is piecewise constant, so a single random-search run lands in
    whatever cell it happened to sample; without restarts the spread between
    seeds is comparable to the difference between model forms, which would make
    any form comparison meaningless.
    """
    best, best_key = None, (-1.0, -1.0)
    for sd in seeds:
        p, key = search(T, draws, sd, combine, free=free, brake=brake)
        if key > best_key:
            best, best_key = p, key
    return best, best_key


def cross_validate(
    cfgs,
    form,
    combine,
    draws,
    seeds,
    folds=5,
    max_degree=64,
    free=FIELDS,
    fold_seed=1234,
    brake="mem",
):
    """Mean held-out geomean over k folds, stratified on the oracle degree."""
    import random

    by = {}
    for c in cfgs:
        by.setdefault(c.oracle, []).append(c)
    assign = {}
    for o in sorted(by):
        grp = sorted(by[o], key=lambda c: c.name)
        random.Random(fold_seed + o).shuffle(grp)
        for i, c in enumerate(grp):
            assign[c.name, c.dtype, c.stride] = i % folds
    out = []
    for f in range(folds):
        key = lambda c: assign[c.name, c.dtype, c.stride]
        tr = [c for c in cfgs if key(c) != f]
        va = [c for c in cfgs if key(c) == f]
        p, _ = restarts(
            tabulate(tr, max_degree=max_degree, form=form),
            draws,
            seeds,
            combine,
            free=free,
            brake=brake,
        )
        Tv = tabulate(va, max_degree=max_degree, form=form)
        got, j = evaluate(Tv, p[None], combine, brake)
        out.append(float(summarise(got, j, Tv)[0][0]))
    return sum(out) / len(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-degree", type=int, default=64)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--cv", action="store_true", help="pick the form by CV on TRAIN")
    ap.add_argument("--fp", default="taps", choices=("taps", "box"))
    ap.add_argument("--reuse", default="taps", choices=("taps", "box"))
    ap.add_argument("--combine", default="sum", choices=("max", "sum"))
    ap.add_argument("--brake", default="all", choices=("mem", "all"))
    ap.add_argument(
        "--model",
        default="full",
        choices=sorted(MODELS),
        help="nested model to fit; everything outside it is pinned to NEUTRAL",
    )
    args = ap.parse_args()

    measured = corpus_fwd.measured()
    if not measured:
        raise SystemExit("no measured corpus present — see README.md")
    # TEST is split off first and never touched again until the final report;
    # form selection below runs entirely inside TRAIN.
    tr, te = split(measured, args.test_frac)
    print(f"measured {len(measured)} -> train {len(tr)} test {len(te)}\n")
    seeds = list(range(args.seeds))
    free = MODELS[args.model]

    form, combine = (args.fp, args.reuse), args.combine
    if args.cv:
        print("5-fold CV on TRAIN only (TEST untouched):")
        best = None
        for fo in (("taps", "taps"), ("box", "taps"), ("taps", "box"), ("box", "box")):
            for co in ("max", "sum"):
                mu, fs = cross_validate(
                    tr,
                    fo,
                    co,
                    args.draws,
                    seeds,
                    max_degree=args.max_degree,
                    free=free,
                    brake=args.brake,
                )
                print(
                    f"  fp={fo[0]:<4} reuse={fo[1]:<4} combine={co:<3}  "
                    f"CV geomean {mu:.4f}   " + " ".join(f"{v:.3f}" for v in fs)
                )
                if best is None or mu > best[0]:
                    best = (mu, fo, co)
        _, form, combine = best
        print(f"  -> selected fp={form[0]} reuse={form[1]} combine={combine}\n")

    p, key = restarts(
        tabulate(tr, max_degree=args.max_degree, form=form),
        args.draws,
        seeds,
        combine,
        free=free,
        brake=args.brake,
    )
    print(
        f"--- final fit: model={args.model} fp={form[0]} reuse={form[1]} "
        f"combine={combine} brake={args.brake} (train {key[0]:.4f}) ---"
    )
    print("  " + "  ".join(f"{f}={v:.5g}" for f, v in zip(FIELDS, p)))
    for label, cfgs in (
        ("measured TRAIN", tr),
        ("measured TEST", te),
        ("measured ALL", measured),
    ):
        report(label, cfgs, p, combine, args.max_degree, form, args.brake)


if __name__ == "__main__":
    main()
