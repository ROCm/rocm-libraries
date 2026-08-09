# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The measured tuning record for the fp8 fused-MoE mega kernel.

Single source of truth for three things that used to be declared in more than
one place: the shape every measurement was taken on, the token bands and the
tile geometry each one selects, and the knob sets those measurements produced.

WHY IT LIVES HERE
-----------------
Beside the kernel, because the two modules that need it are both *consumers* of
it and neither is upstream of the other: :mod:`rocke.dispatch.families.moe`
turns a band into a selected spec, and the ``fused_mega_moe`` benchmark turns a
config into a spec for the same builders. When each held its own copy they
could disagree, and the failure mode was silent -- a band boundary moved in one
place routes traffic the other place never measured.

Deliberately import-light: plain data and two lookups, no numpy, no IR layer,
and no environment reads, so importing it costs a dispatcher nothing.

WHAT IS *NOT* HERE
------------------
This module states what was measured. It does not state routing policy -- which
candidate is allowed to answer ``auto``, or which scheduling knobs the
dispatcher pins over the instance defaults -- because those are decisions about
selection rather than results of a sweep, and they belong with the dispatcher
that makes them.

THE SWEEP
---------
T = 1..4096, minimum of 3 runs per point, one XCD of MI355X, microseconds end
to end, on :data:`TUNED_SHAPE`::

    T      fused tm16   coop tm16   coop tm32   coop tm64
    1            38.8        41.7        41.6        46.2
    8           206.1       209.2       228.5       287.6
    64          498.5       466.3       539.0       703.0
    256         705.3       556.6       558.5       723.2
    512        1064.0       808.4       717.1       741.8
    4096       5337.6      4484.5      3313.3      2863.6
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TunedShape:
    """The MoE geometry a tuning record was measured against.

    The dims themselves are runtime kernel arguments -- the kernel stays
    correct at any of them -- so this is a statement about where the numbers
    below were *observed*, not about what the kernel can run.
    """

    hidden: int
    intermediate: int
    num_experts: int
    top_k: int


#: The shape every band boundary and knob value below was measured on. Named
#: for its dimensions rather than for a model: the cohort is
#: ``(2048, 768, 128, 8)``, so any deployment landing on those dims qualifies
#: and the same model under a different tensor-parallel split does not.
TUNED_SHAPE = TunedShape(hidden=2048, intermediate=768, num_experts=128, top_k=8)

#: Largest token count any band claims. A measurement bound, not a kernel one:
#: the sweep stopped at 4096, so a request past it would be served by a
#: crossover nobody observed. Above this the cohort gate should decline and let
#: the request fall to the untuned generic candidate, which is the honest
#: answer -- it is a configuration, not an extrapolation dressed as one.
MAX_TUNED_TOKENS = 4096

#: Geometry shared by every candidate in the cohort. Everything here was
#: measured on :data:`TUNED_SHAPE`; none of it is a class default. The class
#: defaults on ``FusedMegaKernelSpecFp8`` are the best config for the shape it
#: was originally tuned on (I=7168); this cohort is the *narrow* intermediate --
#: I=768 leaves three N-tiles -- and wants a different geometry entirely, so
#: the base is stated rather than inherited.
BASE_KNOBS: dict = {
    "tile_n_inter": 128,
    "tile_n_down": 128,
    "tile_k_gu": 32,
    "tile_k_down": 64,
    "warp_n": 1,
    "gate_up_k": 128,
    "down_k": 128,
    "use_dtla": False,
    "use_fused_kloop": True,
    "dtla_depth": 2,
    "window_group": 2,
    "down_fused_cells": True,
    "down_depth": 2,
    "down_group": 1,
    "window_sched": "barrier",
    "hidden_group_k": 128,
    "swizzle_gu": True,
    "swizzle_down": True,
}

#: The three levers the cooperative configs need together. ``coop_b_lds`` is
#: the point; the other two are what pay for it. ``mfma_vgpr_form`` frees the
#: registers the shared tile's read-back needs, ``static_inter_scale`` frees the
#: LDS, and ``lds_pad`` must be 0 because a 384 pad widens the epilogue staging
#: past what a wide ``tile_m`` can afford alongside a 32 KB B tile.
COOP_LEVERS: dict = {
    "coop_b_lds": True,
    "mfma_vgpr_form": True,
    "static_inter_scale": True,
    "lds_pad": 0,
}

#: The single launch keeps the padded staging it was tuned with; it has no
#: shared B tile competing for the LDS budget.
FUSED_LDS_PAD = 384

#: ``(min_tokens, max_tokens, band_id, tile_m, warp_m)``, ascending, closed
#: intervals, contiguous, and jointly covering ``1 .. MAX_TUNED_TOKENS``. Held
#: as ONE table because the boundary and the geometry it selects are the same
#: decision: when they lived in two structures a band could be moved without
#: its tile widening with it, and nothing failed.
#:
#: A band id names the kernel *structure and geometry* it selects, never the
#: shape or the model it was measured on -- the shape is already carried by
#: :data:`TUNED_SHAPE` here and by the capability the dispatcher declares, and
#: a third copy inside an identifier is the one nothing keeps honest.
#:
#: ``warp_m`` splits the token rows across waves so the per-wave accumulator
#: count stays at the tile_m=16 value while the workgroup's tile widens.
#:
#: WHERE EACH BOUNDARY COMES FROM, in the terms of the sweep above:
#:
#: * ``8 | 9`` -- the fused/split crossover. A single launch keeps the
#:   intermediate in LDS and pays no HBM round trip for it, which dominates
#:   while there are only a handful of token rows to amortize the split's
#:   extra launch and 1.4 MB re-read over. Fused wins at T<=8 and loses from
#:   T=16 up; the measured margin at T=32 is ~1.5%, inside this shape's
#:   run-to-run spread, so the boundary is placed at the last point where the
#:   fused kernel wins outright rather than at the last point where it is not
#:   clearly behind.
#: * ``256 | 257`` -- the tile_m 16->32 crossover. Widening ``tile_m`` cuts
#:   per-workgroup weight traffic (the shared B tile is read once, not once
#:   per wave) and adds padded rows; with top-8 over 128 experts an active
#:   expert first has enough real rows to pay for a 32-row tile here.
#: * ``512 | 513`` -- the same trade one step further, 32->64.
TOKEN_BANDS: tuple[tuple[int, int, str, int, int], ...] = (
    (1, 8, "fused_tm16", 16, 1),
    (9, 256, "split_coop_tm16", 16, 1),
    (257, 512, "split_coop_tm32", 32, 2),
    (513, MAX_TUNED_TOKENS, "split_coop_tm64", 64, 4),
)

#: The one band served by a single launch. Named rather than spelled out at
#: each test against it, because "is this the fused band?" is asked in several
#: places and a string literal in most of them is a rename away from silently
#: meaning "no".
FUSED_BAND = "fused_tm16"

#: ``band_id -> (tile_m, warp_m)`` and ``band_id -> (min_tokens, max_tokens)``,
#: both projections of :data:`TOKEN_BANDS` so neither can disagree with it.
BAND_GEOMETRY: dict[str, tuple[int, int]] = {
    band_id: (tile_m, warp_m) for _, _, band_id, tile_m, warp_m in TOKEN_BANDS
}
BAND_RANGE: dict[str, tuple[int, int]] = {
    band_id: (lo, hi) for lo, hi, band_id, _, _ in TOKEN_BANDS
}


def band_for(num_tokens: int) -> str | None:
    """The band serving ``num_tokens``, or ``None`` if no band was measured.

    Total over the integers by construction: the table is contiguous from 1 to
    :data:`MAX_TUNED_TOKENS` and anything outside it returns ``None`` rather
    than the nearest band. An earlier form ended in an open-ended sentinel,
    which made every token count above the sweep silently the widest tile -- a
    claim about hardware behaviour that no measurement backs.
    """
    tokens = int(num_tokens)
    for lo, hi, band_id, _, _ in TOKEN_BANDS:
        if lo <= tokens <= hi:
            return band_id
    return None


def matches_tuned_shape(
    *, hidden: int, intermediate: int, num_experts: int, top_k: int
) -> tuple[bool, str]:
    """Whether these dims are the shape the bands were measured on.

    Deliberately exact. A band boundary is only meaningful for the geometry it
    was measured against -- ``tile_m`` trades weight traffic against row
    padding, and both terms move with intermediate width and expert count -- so
    a near-miss shape should get the untuned default candidate rather than a
    crossover extrapolated from a shape it does not share.

    Takes plain dims rather than a request object so that nothing here depends
    on a dispatcher's request type, and so the benchmark can ask the same
    question without constructing one.
    """
    for field, got in (
        ("hidden", int(hidden)),
        ("intermediate", int(intermediate)),
        ("num_experts", int(num_experts)),
        ("top_k", int(top_k)),
    ):
        want = getattr(TUNED_SHAPE, field)
        if got != want:
            return False, f"{field}={got} outside the tuned cohort ({want})"
    return True, "ok"
