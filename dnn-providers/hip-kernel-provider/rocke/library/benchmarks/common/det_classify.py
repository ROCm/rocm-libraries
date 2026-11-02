# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which cached conv kernels (and launches) reduce deterministically.

A launch is *deterministic* when its output is bitwise identical across
repeated launches and across GPUs: no atomic accumulates into the output or
into any scratch whose summation order depends on scheduling, and every
reduction runs in a fixed order. ``benchmark_implicit_gemm_conv.py
--deterministic`` keeps only the kernels/launches this module accepts.

The classification is read off the kernel identity (and the launch's split-K
degree) and is conservative: anything not known to be deterministic is
rejected. Prove a new reduction with ``benchmarks/common/conv_det_check.py``
(bitwise, >= 20 launches) before adding it here.

Extending it (other tracks): a new opt-in identity field that makes a split
binary deterministic goes into the ``*_DET_SPLIT_FIELDS`` tuple of its
direction -- a split binary is accepted when any listed field is truthy on
its identity and none of the ``*_ATOMIC_FIELDS`` is.
"""

from __future__ import annotations

from typing import Optional, Tuple

# ---- fwd -----------------------------------------------------------------
# Every fwd split-K binary today reduces through packed bf16 atomics into D
# (split_tail: the tail tiles; split_flag_store / split_zero_first: slice 0
# stores, the rest atomic-add; split_streamk / split_hybrid: atomics on the
# shared tiles). Identity fields of deterministic fwd split reductions:
# split_det (det-fwd): owner-ordered f32 fold (1, 2) / two-stage slabs (3).
FWD_DET_SPLIT_FIELDS: Tuple[str, ...] = ("split_det",)

# ---- dgrad ---------------------------------------------------------------
# split_k > 1 (incl. pp_boxes box split degrees, which need split_k > 1)
# atomic-adds into dX; pp_streamk without pp_sk_mode bit 1 atomic-adds;
# pp_sk_mode bit 1 (fix-up by the last-arriving contributor) adds the last
# arriver's own partial at a scheduling-dependent place -- rejected;
# tapk_pp_persist 2 (stream-K tail) atomic-adds. Deterministic split fields:
# pp_box_det (det-dgrad): split box slices store per-block slots, a second
# kernel folds them in slice order.
DGRAD_DET_SPLIT_FIELDS: Tuple[str, ...] = ("pp_box_det",)

# ---- wgrad ---------------------------------------------------------------
# Two-stage fields that atomically accumulate (into the scratch or dW).
WGRAD_ATOMIC_FIELDS: Tuple[str, ...] = ("ws_fuse", "ws_lc", "ws_gred")
# Split binaries that reduce in a fixed order: ws_store (one plain-stored
# scratch slab per slice, Stage 2 folds the slabs in index order).
WGRAD_DET_SPLIT_FIELDS: Tuple[str, ...] = ("ws_store",)

# ---- low-precision partials ------------------------------------------------
# (field, value) pairs of deterministic reductions that round every partial
# sum below f32 before the fold (less accurate than an f32 reduction):
# ``--deterministic f32`` rejects them as well. ws_store 2 = bf16 slabs.
LOWP_PARTIAL_FIELDS: Tuple[Tuple[str, int], ...] = (("ws_store", 2), ("pp_box_det", 2))


# (det-fwd) Reason of the two-way bf16-atomic fwd rule below: deterministic,
# but each partial is rounded to bf16 -> rejected by ``--deterministic f32``.
TWO_WAY_BF16_REASON = "two-way split: commutative bf16 add"


def _any(ident, fields) -> bool:
    return any(getattr(ident, f, 0) for f in fields)


def classify(direction: str, ident, split_k: Optional[int] = None) -> Tuple[bool, str]:
    """``(deterministic, reason)`` of ``ident`` launched at ``split_k``.

    ``split_k`` None means "any degree the binary can be launched at" (the
    binary-level answer: True only when every launch is deterministic).
    """
    sk_bin = max(1, int(getattr(ident, "split_k", 1) or 1))
    if direction == "wgrad":
        streamk = getattr(ident, "streamk", "off") or "off"
        if streamk != "off":
            red = getattr(ident, "streamk_reduction", "") or "linear"
            if red in ("linear", "tree"):
                return True, f"stream-K {red}: ordered flag fold"
            return False, f"stream-K {red}: atomics"
        if sk_bin <= 1:
            return True, "unsplit: direct store"
        if not getattr(ident, "two_stage", False):
            return False, "split-K packed atomics into dW"
        if _any(ident, WGRAD_ATOMIC_FIELDS):
            return False, "two-stage with scratch atomics (ws_fuse/ws_lc/ws_gred)"
        if _any(ident, WGRAD_DET_SPLIT_FIELDS):
            reps = int(getattr(ident, "ws_replicas", 0) or 0)
            if split_k is not None and reps and int(split_k) > reps:
                return False, f"ws_store needs split_k <= ws_replicas ({split_k} > {reps})"
            return True, "two-stage ws_store: one slab per slice, ordered fold"
        return False, "two-stage f32 atomics into shared slabs"
    if direction == "fwd":
        if sk_bin <= 1:
            return True, "unsplit: direct store"
        if _any(ident, FWD_DET_SPLIT_FIELDS):
            return True, "deterministic split field"
        if (
            split_k is not None
            and int(split_k) == 2
            and not getattr(ident, "split_streamk", False)
            and not getattr(ident, "split_hybrid", 0)
        ):
            # (det-fwd) Two contributions into a zeroed (or slice-0-stored)
            # tile: bf16(a) then round(bf16(a) + bf16(b)) -- IEEE addition
            # commutes, so either order gives the same bits (fwd s3 record,
            # 2/full1824, checked bitwise). Three or more do not.
            return True, TWO_WAY_BF16_REASON
        return False, "split-K atomics into D"
    if direction == "dgrad":
        if getattr(ident, "pp_streamk", 0):
            if _any(ident, DGRAD_DET_SPLIT_FIELDS):
                return True, "deterministic split field"
            return False, "pp_streamk: atomics / last-arriver fix-up"
        if getattr(ident, "tapk_pp_persist", 0) == 2:
            return False, "persistent stream-K tail atomics"
        if sk_bin <= 1:
            return True, "unsplit: disjoint stores"
        if _any(ident, DGRAD_DET_SPLIT_FIELDS):
            return True, "deterministic split field"
        return False, "split-K atomics into dX"
    return False, f"unknown direction {direction!r}"


def is_deterministic(
    direction: str, ident, split_k: Optional[int] = None, mode: str = "any"
) -> bool:
    """``mode`` "any": any bitwise-deterministic reduction; "f32": also no
    partial sums rounded below f32 (LOWP_PARTIAL_FIELDS)."""
    ok, reason = classify(direction, ident, split_k)
    if ok and mode == "f32":
        ok = not any(getattr(ident, f, None) == v for f, v in LOWP_PARTIAL_FIELDS)
        ok = ok and reason != TWO_WAY_BF16_REASON
    return ok
