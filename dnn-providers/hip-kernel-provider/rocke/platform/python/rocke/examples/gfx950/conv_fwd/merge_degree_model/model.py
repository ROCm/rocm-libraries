#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Analytic cost model for the forward depthwise group-merge degree.

This file is the readable reference for the heuristic that
``library/dispatch/grouped_convolution.py`` ships as
``fwd_group_merge_for_geometry()``. Read this one; the dispatch copy is the same
arithmetic inlined so that ``dispatch`` keeps importing nothing.

-------------------------------------------------------------------------------
WHAT PROBLEM IS BEING SOLVED
-------------------------------------------------------------------------------

Depthwise convolution (``cpg == kpg == 1``: one input channel per group, one
output channel per group) is a pathological case for an implicit-GEMM kernel.
Each group is its own tiny GEMM with ``N_gemm = 1`` and ``K_gemm = Y*X``, so:

  * the N dimension is 1 against a tile that is 16..128 wide -- almost the whole
    MFMA output tile is discarded;
  * K is the filter area, typically 9, against a K tile of 16..64 -- so the
    K loop runs one mostly-padded iteration;
  * the A (activation) load walks the channel axis one element at a time, which
    in NHWC is one element per 128-byte cache line.

*Merged groups* fixes this by folding ``Gm`` consecutive conv groups into a
single GEMM. The arrangement puts ``Gm`` on GemmN *and* on GemmK:

    M_gemm = N*Ho*Wo            (unchanged by Gm)
    N_gemm = Gm                 (one grid.y tile as long as Gm <= tile_n)
    K_gemm = Y*X*Gm             (k decodes to (y, x, g_k), g_k innermost)
    grid.z = G / Gm

Because a k index now carries a group field, the B (weight) operand must be
masked on the diagonal ``g_k == g_n`` -- group i's weights may only reach group
i's output column. That mask is why B loads at ``vector_size_b = 1``; it is the
fixed price of the whole scheme and does not vary with Gm.

Net effect of the launch geometry:

    CTAs(Gm) = ceil(M/tile_m) * G/Gm           -- halves on every doubling
    Kpad(Gm) = tile_k * ceil(Y*X*Gm / tile_k)  -- grows roughly linearly

So Gm trades CTA count against per-CTA work, and the exchange rate depends on
the shape. Too small and the kernel keeps the degenerate N=1 GEMM; too large and
each CTA drags a working set that no longer fits. **The question this file
answers is: given only what dispatch knows at selection time (G, M, Y, X,
stride, the tile, the element size), which degree?**

The alternative -- a sweep, or a tuning table -- is what the benchmark does and
what dispatch cannot do: dispatch must answer in microseconds, from geometry
alone, for shapes nobody measured.

-------------------------------------------------------------------------------
THE FOUR MECHANISMS THAT MOVE WITH Gm
-------------------------------------------------------------------------------

Three of them pull towards a larger degree, one pulls back. Every fitted
constant below exists to weigh exactly one of these against the others.

  pad(Gm)   K-padding waste, already folded into ``Kpad`` above. This is pure
            arithmetic, no constant needed. At Y*X=9 against tile_k=64, Gm=1
            issues a 64-wide K tile to do 9 columns of work -- a factor of about
            7 thrown away -- while Gm=8 brings Y*X*Gm to 72 and the waste to
            about 1.1x. At Y*X=961 the first tile is already full, there is
            nothing to recover, and this term stops arguing for merging at all.
            That is the first reason very large filters never want a high
            degree.

  vw(Gm)    A-load vector width, ``min(Gm, 16/esize)``. The k index's innermost
            field is the channel, and in NHWC consecutive channels are
            consecutive in memory, so merging Gm groups makes Gm elements
            contiguous. Saturates at dwordx4 (16 B), i.e. at Gm=8 for 16-bit.

  u(Gm)     A-load cache-line utilisation, ``min(1, Gm*esize/128)``. Distinct
            from vw and it keeps improving past the vector-width knee: at Gm=8
            and 16-bit a 128-byte line delivers 16 useful bytes. This is why
            large shapes keep gaining from 8 -> 16 -> 32 after the vector width
            has stopped moving.

  fp(Gm)    The brake. A CTA's A working set is ``tile_m x Y*X*Gm`` elements --
            linear in the degree *and* in the filter area. This is the only term
            that argues against merging, and it has to be strong enough to
            overturn a halved CTA count, because that is what it is competing
            with.

**Occupancy is the obvious candidate for the brake and it is the wrong one.**
Shapes with tens of thousands of CTAs -- where the machine cannot possibly run
dry, so launch granularity cannot be what limits them -- still turn over at
Gm=8..16, and they turn over sooner the larger Y*X is. Across the corpus,
whether Gm=64 still beats Gm=32 is a clean monotone function of Y*X alone, with
no dependence on the CTA count (see the case study). Occupancy
cannot produce that. Working-set size can. Occupancy survives in the model only
as the wave-quantisation term ``ceil(CTAs/CUPAR)``.

The penalty is therefore charged against the *whole* per-CTA cost rather than
only the DRAM term: an oversized K working set costs issue slots and resident
state, not just traffic. fitfast.py calls the two readings ``brake="all"`` and
``brake="mem"``; "all" wins under repeated cross-validation, and "mem" never
gets enough leverage to overturn a halved CTA count because the fitted DRAM
bucket sits well below the compute bucket.

-------------------------------------------------------------------------------
THE COST FUNCTION
-------------------------------------------------------------------------------

Per CTA, in arbitrary units -- only ratios between degrees are ever used, so the
overall scale is free and the MFMA coefficient is pinned at 1:

    compute(Gm) = Kpad * (1 + B/vw)
    memory(Gm)  = Kpad * tile_m*esize * W / (u**A * reuse**r)
    per_cta(Gm) = (compute + memory) * (1 + fp/F)**q + D
    cost(Gm)    = ceil(CTAs/P) * per_cta

and the answer is ``argmin`` over the admissible degrees.

That leaves eight free constants, documented one by one on ``Constants`` below.
They are *not* individually interpretable as hardware quantities -- B and W are
ratios against a large D, and several trade against each other -- but each one
is load-bearing: nested-model cross-validation (``fitfast.MODELS``) pins every
term to its neutral value in turn, and removing the vector-width term or the
memory bucket both cost real accuracy. See ``../fwd_merged_groups_case_study.md``
for the measured comparison.

-------------------------------------------------------------------------------
SCOPE
-------------------------------------------------------------------------------

Fitted on gfx950, 16-bit (fp16/bf16), forward direction, at the single tile
dispatch currently ships (64x64x64). It is a *degree* model, not a tile model:
Finding 5 of the case study scores it at eight different tiles and shows the
accuracy is tile-local, because tile_m enters the cost in three places and was
constant throughout the fit. If dispatch ever widens its tile selection, these
constants must be refitted -- see fit_tiles.py, which already does the pooled
refit.
"""

import math
from dataclasses import dataclass

# The kernel can only build power-of-two degrees (fwd_group_merge_available()).
# Ascending order matters: ``pick()`` takes the first strict improvement, so an
# exact tie resolves to the *smaller* degree.
DEGREES = (1, 2, 4, 8, 16, 32, 64)

# gfx950 cache line. Sets where A-load utilisation saturates: once Gm*esize
# reaches this, a line is fully consumed and ``u`` stops improving.
LINE_BYTES = 128

# Widest single load (dwordx4). Sets where the A-load vector width saturates.
VEC_BYTES = 16


@dataclass(frozen=True)
class Constants:
    """The eight fitted constants, and what each one is actually weighing.

    Fitted by random search plus coordinate refinement (fitfast.search) on the
    training split of the measured corpus -- the objective is the geometric mean
    of the *realised fraction* (what share of a shape's measured best the single
    modelled pick keeps), which is piecewise constant in the parameters, so there
    is no gradient to follow and the search is deliberately brute force.

    None of these is a hardware constant read off a datasheet. They are the
    exchange rates between the four mechanisms above, in a unit system where the
    MFMA coefficient is 1. Signs and rough magnitudes are interpretable; exact
    values are not, and two of them (B, W) are only meaningful relative to D.

    Refitting is a mechanical operation -- see README.md -- and is required if
    the arch, the element size, or the dispatch tile changes.
    """

    # --- the three terms that argue FOR merging --------------------------------

    b_aload_issue: float = 0.014175
    """Weight of the A-load issue cost, charged as ``B/vw`` per K element.

    This is the vector-width mechanism. A narrow load issues more instructions
    for the same bytes, so the cost falls as ``vw`` rises and stops falling once
    ``vw`` saturates at dwordx4. Small relative to the pinned MFMA coefficient of
    1, which is the right order: at these tiles the kernel is not issue-bound,
    the A loads are a correction on top of the MFMA pipeline rather than the main
    term. It is nonetheless the single most load-bearing constant in the model --
    pinning it to zero is the largest accuracy loss of any one-term ablation,
    because without it nothing distinguishes Gm=1 from Gm=2 beyond padding.
    """

    w_memory: float = 0.00030657
    """Weight of the DRAM bucket, i.e. how much a byte of A traffic costs
    relative to a unit of compute.

    Multiplies ``Kpad * tile_m * esize``, the bytes of A a CTA pulls per K tile.
    Small, which says the kernel is compute-side at these tiles -- but not
    negligible: this is the only term that can see ``u`` and ``reuse``, so
    without it the model is blind to cache-line utilisation and to the fact that
    neighbouring output columns share input. Removing it is the second largest
    one-term accuracy loss.
    """

    a_util: float = 2.7279
    """Exponent on cache-line utilisation ``u`` in the memory term.

    ``u`` is the fraction of a fetched 128-byte line the kernel actually
    consumes, and the DRAM cost divides by ``u**A``. A=1 would mean traffic is
    exactly inversely proportional to utilisation -- the naive accounting. The
    fit lands well above 1, i.e. poor utilisation is *superlinearly* expensive.
    That is the expected shape once partially-used lines start displacing lines
    that would have been reused: you pay once for the waste and again for the
    eviction. This exponent is what keeps the model recommending 16 and 32 on
    large shapes after the vector width has already saturated at 8.
    """

    r_reuse: float = 0.4249
    """Exponent on the W-direction tap-overlap factor in the memory term.

    Adjacent output columns read input columns ``stride`` apart, so with
    stride=1 an X-tap filter re-reads X-1 of the X columns its neighbour already
    pulled; at stride=2 roughly half that. The raw overlap count is
    ``1 + (X-1)/stride``, and the exponent discounts it: a fraction below 1 says
    the caches realise part, not all, of the theoretically available reuse.
    This term is what separates otherwise identical shapes that differ only in
    stride, which the measured corpus contains deliberately.
    """

    # --- the term that argues AGAINST merging ----------------------------------

    c_footprint: float = 130.64
    """Scale of the working-set brake: the knee of ``(1 + fp/C)**q``.

    ``fp = tile_m * Y*X*Gm * esize`` is the A working set a CTA must keep live.
    Below C the brake is inert; above it the penalty takes hold. In the fitted
    unit system C is small relative to typical footprints, which is the point --
    for the filter sizes that matter the brake is always in its active region,
    and what the model is really using is the exponent's slope, not the location
    of the knee. Together with ``q_brake`` this pair is the entire reason the
    model ever says "stop merging."
    """

    q_brake: float = 1.6812
    """Exponent of the working-set brake.

    Above 1, so the penalty is superlinear in the footprint. It has to be: the
    brake competes against ``ceil(CTAs/P)``, which halves on every doubling of
    Gm, and a linear penalty can never overturn a halving. This exponent is what
    makes the turnover point depend on Y*X -- a 3x3 filter reaches the painful
    part of the curve four degrees later than a 7x7 -- and reproducing that
    Y*X-ordered turnover was the main thing the functional form had to get right.
    """

    # --- the two launch-geometry terms -----------------------------------------

    cupar: float = 558.42
    """Effective CTA parallelism: the divisor in ``ceil(CTAs/P)``.

    This is the wave-quantisation term and the only surviving trace of
    occupancy. It is deliberately *not* pinned to (CU count x waves per CU):
    nothing guarantees the shipped kernel reaches its theoretical occupancy, and
    what the model needs is the width at which adding CTAs stops costing time,
    which is an empirical quantity. The ``ceil`` is the load-bearing part -- it is
    what makes a shape with few CTAs care about the degree completely differently
    from a shape with many -- and the fitted value lands in a plausible range for
    this device, which is a weak but real sanity check.
    """

    d_cta_fixed: float = 1.9548e6
    """Per-CTA fixed cost, added after the brake.

    Launch overhead, prologue, epilogue, C writeback -- everything a CTA pays
    that does not scale with K. Large relative to B and W, which is why those two
    read as small: they are ratios against this. Its real job is to make CTA
    count matter *on its own*, independent of per-CTA work; without it, halving
    the CTA count while doubling per-CTA work would be exactly free and the model
    would be nearly indifferent to the degree on small shapes.
    """

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


DEFAULT = Constants()


def admissible(groups, tile_n, y, x, max_degree=64):
    """Degrees the kernel can actually build for this shape.

    Mirrors ``fwd_group_merge_available()``: power of two, divides G evenly, and
    fits inside the N tile (the whole point is one grid.y tile, so Gm > tile_n
    would spill). Pointwise (Y==X==1) is excluded by the gate entirely -- with no
    filter there is no K to amortise and the diagonal mask is pure loss -- so
    there is nothing to choose and the caller gets [1].
    """
    if y * x <= 1:
        return [1]
    return [gm for gm in DEGREES if gm <= min(tile_n, max_degree) and groups % gm == 0]


def cost(gm, *, groups, m, y, x, stride, tile_m, tile_n, tile_k, esize=2, k=DEFAULT):
    """Modelled time for one degree, in arbitrary units (only ratios are used).

    ``tile_n`` is accepted for signature symmetry with ``admissible()``; the cost
    itself does not use it, because N_gemm is Gm and the unused part of the N
    tile costs the same whatever Gm is.
    """
    # Launch geometry: CTAs halve on every doubling of Gm, Kpad grows.
    ctas = -(-m // tile_m) * (groups // gm)
    kpad = tile_k * -(-(y * x * gm) // tile_k)

    # A-load quality, both saturating in Gm but at different points.
    vw = min(gm, VEC_BYTES // esize)  # vector width, saturates at dwordx4
    u = min(1.0, gm * esize / LINE_BYTES)  # line utilisation, saturates later

    # The brake: elements of A a CTA must hold live, times the element size.
    footprint = tile_m * y * x * gm * esize

    # Available W-direction reuse: adjacent output columns read input columns
    # `stride` apart, so X-1 of the X taps land on data a neighbour already
    # pulled when stride==1, and about half that at stride 2. Gm-independent, so
    # it does not move the argmin on its own -- it sets how much the memory term
    # is allowed to matter relative to compute for this shape.
    reuse = (1.0 + (x - 1) / stride) ** k.r_reuse

    compute = kpad * (1.0 + k.b_aload_issue / vw)
    memory = kpad * tile_m * esize * k.w_memory / max(u**k.a_util * reuse, 1e-30)

    # Charged against the whole per-CTA cost, not just DRAM -- see the module
    # docstring on why occupancy is not the brake.
    brake = (1.0 + footprint / k.c_footprint) ** k.q_brake
    per_cta = (compute + memory) * brake + k.d_cta_fixed

    # ceil(), not a plain divide: the wave quantisation is the point.
    return math.ceil(ctas / k.cupar) * per_cta


def pick(
    groups,
    m,
    y,
    x,
    stride,
    tile_m,
    tile_n,
    tile_k=64,
    esize=2,
    k=DEFAULT,
    max_degree=64,
):
    """Argmin of the cost over admissible degrees.

    Ties go to the smaller degree -- DEGREES is ascending and the comparison
    requires a strict improvement. That bias is deliberate: a smaller degree
    carries less padding risk on shapes the model cannot see (odd Ho*Wo tails,
    G just above a power of two), so when the model is indifferent the safer of
    the two choices wins.
    """
    best, best_cost = 1, None
    for gm in admissible(groups, tile_n, y, x, max_degree):
        c = cost(
            gm,
            groups=groups,
            m=m,
            y=y,
            x=x,
            stride=stride,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            esize=esize,
            k=k,
        )
        if best_cost is None or c < best_cost * (1.0 - 1e-9):
            best, best_cost = gm, c
    return best


def selfcheck():
    """Agree with fitfast's vectorised evaluate() on every measured config.

    The two implementations exist for different reasons -- this one to be read
    and ported, that one to be run a hundred thousand times a second -- so they
    are kept honest against each other rather than one being derived from the
    other. Requires a locally generated degrees.csv (see README.md); with no
    corpus present there is nothing to check and this trivially passes.
    """
    import numpy as np

    import corpus_fwd
    import fitfast

    cfgs = corpus_fwd.measured()
    if not cfgs:
        print("selfcheck: no measured corpus present (see README.md) — skipped")
        return True
    p = np.array([[getattr(DEFAULT, f) for f in Constants.FIELDS]])
    T = fitfast.tabulate(cfgs, form=("taps", "taps"))
    _, j = fitfast.evaluate(T, p, "sum", "all")
    bad = 0
    for i, c in enumerate(cfgs):
        ref = fitfast.DEGREES[int(j[0, i])]
        got = pick(c.groups, c.m(), c.y, c.x, c.stride, 64, 64, 64, c.esize)
        if ref != got:
            bad += 1
            print(f"  MISMATCH {c.name}: fitfast={ref} model={got}")
    print(f"selfcheck: {len(cfgs) - bad}/{len(cfgs)} agree")
    return bad == 0


if __name__ == "__main__":
    raise SystemExit(0 if selfcheck() else 1)
