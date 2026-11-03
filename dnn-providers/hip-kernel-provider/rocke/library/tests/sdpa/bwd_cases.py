# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared SDPA backward case table, materialisation, oracle and comparison.

Backward cases reuse the forward case model (:class:`SdpaCase`) and its
validity rules, and are kept in a table of their own so the forward table, its
ids and its tier budgets are untouched. Every backward case id starts with
``bwd_`` and carries the ``bwd`` tag and coverage item 15 (item 16 for the
ragged THD layout; :data:`~.helpers.COVERAGE_ITEMS`). Excluded by
construction: head dim 256, bias, dropout.

Tiers mirror the forward table (``smoke`` <= ``full`` <= ``exhaustive``);
``smoke`` is sized for a few minutes of device time across both CDNA arches.
Cases whose every query length is 1 also carry ``decode_bwd`` so a capability
predicate can include or decline them as one group.

A materialised backward problem holds Q, K, V (as in the forward), plus
``O`` (the float64 oracle forward rounded to the case dtype), ``LSE`` (the
oracle forward statistic stored as fp32, natural log, ``-inf`` on fully masked
rows) and a random ``dO``; every one of them sits in a strided buffer with
finite garbage outside the attended region. The oracle is
:func:`rocke.numeric.sdpa_reference.sdpa_reference_bwd` evaluated on exactly
those stored values.

Tolerance (proposed; see :data:`TOLERANCE_BWD`): per gradient tensor, over the
valid rows only, ``max|got - ref| <= atol + rtol * g * max|ref|`` with
``g = max(1, sqrt(kv_len_max / 1024))``. The metric is norm-relative: a
backward gradient is a sum over ``s_kv`` (dQ) or ``s_q * group`` (dK, dV)
products, so a per-element relative bound is meaningless where the true value
cancels to near zero, while the error of a correct fp32-accumulate kernel
scales with the tensor's magnitude. ``atol`` is a floor for tensors whose
reference is (near) zero. Calibrated with :func:`simulate_lowp_bwd`: over the
whole backward table and over extra key lengths up to 8192 at d = 64 / 128 the
simulated ratio ``max|err| / max|ref|`` stayed within the half-ulp of the
stored dtype (no measurable growth with ``s_kv`` or ``d``), so ``g`` only
widens the bound for key lengths beyond the calibrated range.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, replace

import numpy as np

from rocke.numeric.sdpa_reference import sdpa_reference_bwd

from .cases import _mask, _mk, _pairwise_cover
from .helpers import (
    CDNA_ARCHS,
    DTYPES,
    TIERS,
    SdpaInputs,
    TensorDesc,
    _view,
    case_seed,
    evaluate,
    layout_tensors,
    lse_shape,
    materialize,
    round_to,
    validate_case,
)

BWD_HEAD_DIMS = (64, 128)
BWD_OPTIONAL_HEAD_DIMS = (32,)
BWD_REQ = 15
BWD_RAGGED_REQ = 16

# Proposed backward tolerances (norm-relative, see the module docstring).
# Worst simulated ratio over the whole table: fp16 about 7e-4, bf16 about 6e-3
# (the half-ulp of the stored gradient dominates). rtol leaves about 6x (fp16)
# and 4x (bf16) headroom for kernels whose summation order, exp implementation
# or P / dS rounding points differ from the simulation.
TOLERANCE_BWD = {
    "fp16": {"rtol": 4e-3, "atol": 1e-4},
    "bf16": {"rtol": 2.5e-2, "atol": 1e-3},
}
GRADS = ("dq", "dk", "dv")

# Per-row check on top of the tensor-wide bound: a row (query row of dq, key
# row of dk / dv) is measured against its own magnitude, floored at this
# fraction of the tensor-wide max so that near-zero rows do not demand
# better-than-rounding accuracy. Catches errors confined to one row (a
# dropped tail key zeroes a whole dk / dv row) that the global bound hides.
ROW_FLOOR_BWD = 0.05


def kv_growth(case):
    """Tolerance growth factor ``max(1, sqrt(kv_len_max / 1024))``."""
    _, lk = case.lens()
    return max(1.0, math.sqrt(max(lk) / 1024.0))


# ---------------------------------------------------------------------------
# case construction
# ---------------------------------------------------------------------------


def _mkb(cid, *, reqs, tags=(), **kw):
    """One backward case: ``bwd_`` id prefix, coverage item 15 (and 16 when
    ragged), tags."""
    ragged = kw.get("length_mode") == "ragged"
    items = set(reqs) | {BWD_REQ} | ({BWD_RAGGED_REQ} if ragged else set())
    case = _mk(f"bwd_{cid}", reqs=items, tags=tuple(tags) + ("bwd",), **kw)
    if case.decode:
        case = replace(case, tags=case.tags | {"decode_bwd"})
    return case


def _dtype_headdim_cases():
    out = []
    add = out.append
    for dt in DTYPES:
        for d in BWD_HEAD_DIMS:
            add(
                _mkb(
                    f"dtype_{dt}_d{d}_mha",
                    reqs=(1, 9),
                    tier=(
                        "smoke" if (dt, d) in {("fp16", 128), ("bf16", 64)} else "full"
                    ),
                    dtype=dt,
                    d=d,
                    rdna=True,
                )
            )
            add(
                _mkb(
                    f"dtype_{dt}_d{d}_causal_tl",
                    reqs=(1, 5, 9),
                    tier="smoke" if (dt, d) == ("bf16", 128) else "full",
                    dtype=dt,
                    d=d,
                    s_q=256,
                    rdna=True,
                    **_mask("tl"),
                )
            )
            add(
                _mkb(
                    f"headdim_{dt}_d{d}_br_gqa",
                    reqs=(3, 6, 9),
                    tier="smoke" if (dt, d) == ("fp16", 64) else "full",
                    dtype=dt,
                    d=d,
                    h_k=2,
                    s_q=64,
                    s_kv=192,
                    **_mask("br"),
                )
            )
    for dt in DTYPES:
        for d in BWD_OPTIONAL_HEAD_DIMS:
            add(
                _mkb(
                    f"headdim_optional_{dt}_d{d}",
                    reqs=(9,),
                    tier="exhaustive",
                    tags=("optional",),
                    dtype=dt,
                    d=d,
                    s_q=96,
                    rdna=True,
                    **_mask("tl"),
                )
            )
    return out


def _layout_cases():
    out = []
    add = out.append
    for layout in ("bhsd", "bshd", "strided_bhsd", "strided_bshd", "mixed"):
        for dt in DTYPES:
            smoke = dt == "fp16" and layout in ("bshd", "strided_bshd", "mixed")
            add(
                _mkb(
                    f"layout_{layout}_{dt}_causal_tl",
                    reqs=(2, 5),
                    tier="smoke" if smoke else "full",
                    dtype=dt,
                    layout=layout,
                    s_q=192,
                    rdna=layout in ("bhsd", "bshd"),
                    **_mask("tl"),
                )
            )
        add(
            _mkb(
                f"layout_{layout}_gqa_br",
                reqs=(2, 3, 6),
                layout=layout,
                h_k=2,
                s_q=48,
                s_kv=160,
                **_mask("br"),
            )
        )
    for d in BWD_HEAD_DIMS:
        add(
            _mkb(
                f"layout_packed_qkv_mha_d{d}",
                reqs=(2, 9, 5),
                tier="smoke" if d == 128 else "full",
                layout="packed_qkv",
                d=d,
                s_q=160,
                **_mask("tl"),
            )
        )
    add(
        _mkb(
            "layout_packed_qkv_gqa_bf16",
            reqs=(2, 3, 5),
            dtype="bf16",
            layout="packed_qkv",
            h_k=2,
            s_q=96,
            **_mask("tl"),
        )
    )
    add(
        _mkb(
            "layout_packed_qkv_noncausal", reqs=(2,), layout="packed_qkv", s_q=72, d=64
        )
    )
    add(
        _mkb(
            "layout_strided_single_batch",
            reqs=(2,),
            layout="strided_bshd",
            b=1,
            s_q=100,
        )
    )
    return out


def _gqa_scale_cases():
    out = []
    add = out.append
    for hq, hk, hv, tag in (
        (8, 8, 8, "mha"),
        (8, 4, 4, "gqa2"),
        (8, 2, 2, "gqa4"),
        (8, 1, 1, "mqa"),
        (32, 8, 8, "gqa4_h32"),
        (6, 2, 2, "gqa3_odd_ratio"),
        (64, 8, 8, "gqa8_h64"),
        (8, 4, 2, "kv_heads_differ"),
        (8, 2, 8, "k_fewer_than_v"),
    ):
        optional = tag in ("kv_heads_differ", "k_fewer_than_v")
        add(
            _mkb(
                f"gqa_{tag}_causal_tl",
                reqs=(3, 5),
                tier=(
                    "exhaustive"
                    if optional
                    else ("smoke" if tag in ("gqa4", "mqa") else "full")
                ),
                tags=("optional",) if optional else (),
                h_q=hq,
                h_k=hk,
                h_v=hv,
                b=1 if hq >= 32 else 2,
                s_q=96,
                rdna=tag in ("mha", "gqa4", "mqa"),
                **_mask("tl"),
            )
        )
    for tag, sc in (("half", 0.5), ("one", 1.0), ("tiny", 0.01), ("default", None)):
        add(
            _mkb(
                f"scale_{tag}",
                reqs=(4,),
                tier="smoke" if tag == "half" else "full",
                scale=sc,
                s_q=96,
                rdna=tag == "half",
                **_mask("tl"),
            )
        )
    add(
        _mkb(
            "scale_explicit_equals_default",
            reqs=(4,),
            scale=1.0 / math.sqrt(64),
            d=64,
            s_q=64,
        )
    )
    return out


def _mask_lse_cases():
    out = []
    add = out.append
    rels = {"eq": (128, 128), "q_lt_kv": (64, 192), "q_gt_kv": (192, 64)}
    smoke = {("tl", "eq"), ("br", "q_lt_kv"), ("br", "q_gt_kv")}
    for rel, (sq, sk) in rels.items():
        for kind in ("tl", "br"):
            req = 5 if kind == "tl" else 6
            add(
                _mkb(
                    f"causal_{kind}_{rel}",
                    reqs=(req, 7),
                    tier="smoke" if (kind, rel) in smoke else "full",
                    s_q=sq,
                    s_kv=sk,
                    rdna=rel == "eq",
                    **_mask(kind),
                )
            )
            add(
                _mkb(
                    f"causal_{kind}_{rel}_bf16",
                    reqs=(req, 7),
                    dtype="bf16",
                    s_q=sq,
                    s_kv=sk,
                    **_mask(kind),
                )
            )
    add(_mkb("mask_none_q_lt_kv", reqs=(7,), s_q=64, s_kv=200))
    add(_mkb("mask_none_q_gt_kv", reqs=(7,), s_q=200, s_kv=64))
    for dt in DTYPES:
        for kind in ("none", "tl", "br"):
            add(
                _mkb(
                    f"lse_{dt}_{kind}_odd_lengths",
                    reqs=(7, 8)
                    + ((5,) if kind == "tl" else (6,) if kind == "br" else ()),
                    tier="smoke" if (dt, kind) == ("bf16", "br") else "full",
                    dtype=dt,
                    s_q=77,
                    s_kv=133,
                    **_mask(kind),
                )
            )
    add(
        _mkb(
            "lse_fully_masked_rows_neg_inf",
            reqs=(7, 6),
            tier="smoke",
            s_q=96,
            s_kv=32,
            **_mask("br"),
        )
    )
    add(
        _mkb(
            "lse_peaked_softmax_scale_one",
            reqs=(7, 4),
            scale=1.0,
            d=64,
            s_q=96,
            **_mask("tl"),
        )
    )
    return out


def _seqlen_cases():
    out = []
    add = out.append
    pairs = [
        (17, 17),
        (33, 33),
        (72, 72),
        (100, 100),
        (17, 100),
        (100, 17),
        (33, 72),
        (72, 33),
        (127, 129),
        (129, 127),
        (255, 257),
        (257, 255),
        (1000, 1000),
        (1023, 1025),
        (1025, 1023),
    ]
    smoke_pairs = {(33, 72), (100, 17)}
    for i, (sq, sk) in enumerate(pairs):
        for kind in ("none", "tl", "br"):
            if (sq, sk) in smoke_pairs:
                tier = "smoke" if kind == "br" else "full"
            elif kind == ("none", "tl", "br")[i % 3]:
                tier = "full"
            else:
                tier = "exhaustive"
            big = max(sq, sk) > 600
            add(
                _mkb(
                    f"seqlen_{sq}x{sk}_{kind}",
                    reqs=(8,)
                    + ((5,) if kind == "tl" else (6,) if kind == "br" else ()),
                    tier=tier,
                    b=1 if big else 2,
                    h_q=4 if big else 8,
                    d=64,
                    s_q=sq,
                    s_kv=sk,
                    rdna=sq < 300 and kind == "tl",
                    **_mask(kind),
                )
            )
    for dt in DTYPES:
        add(
            _mkb(
                f"seqlen_odd_{dt}_d128_br",
                reqs=(8, 9, 6),
                dtype=dt,
                d=128,
                s_q=129,
                s_kv=257,
                **_mask("br"),
            )
        )
    return out


def _window_cases():
    out = []
    add = out.append
    lefts = (0, 1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129)
    smoke = {("tl", 1), ("br", 64)}
    for al in ("tl", "br"):
        for w in lefts:
            if (al, w) in smoke:
                tier = "smoke"
            else:
                tier = "full" if w in (0, 1, 63, 64, 65, 129) else "exhaustive"
            sq, sk = (192, 192) if al == "tl" else (96, 224)
            add(
                _mkb(
                    f"window_{al}_w{w}",
                    reqs=(11, 5 if al == "tl" else 6),
                    tier=tier,
                    s_q=sq,
                    s_kv=sk,
                    diagonal="top_left" if al == "tl" else "bottom_right",
                    left_bound=w,
                    right_bound=0,
                    rdna=al == "tl" and w in (1, 64),
                )
            )
    add(
        _mkb("window_wider_than_seq", reqs=(11,), s_q=64, left_bound=500, right_bound=0)
    )
    add(_mkb("window_left_only_tl", reqs=(11,), s_q=160, left_bound=20))
    add(
        _mkb(
            "window_left_only_br",
            reqs=(11, 6),
            s_q=64,
            s_kv=190,
            left_bound=20,
            diagonal="bottom_right",
        )
    )
    add(_mkb("window_right_only_lookahead", reqs=(11,), s_q=160, right_bound=3))
    add(
        _mkb(
            "window_right_only_br_q_gt_kv",
            reqs=(11, 6),
            s_q=150,
            s_kv=70,
            right_bound=5,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "window_symmetric_tl",
            reqs=(11,),
            tier="smoke",
            s_q=160,
            left_bound=9,
            right_bound=5,
        )
    )
    add(
        _mkb(
            "window_symmetric_br_q_lt_kv",
            reqs=(11, 6),
            s_q=60,
            s_kv=180,
            left_bound=9,
            right_bound=5,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "window_diagonal_only_tl", reqs=(11,), s_q=100, left_bound=0, right_bound=0
        )
    )
    add(
        _mkb(
            "window_diagonal_only_br",
            reqs=(11, 6),
            s_q=40,
            s_kv=100,
            left_bound=0,
            right_bound=0,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "window_tl_q_gt_kv_fully_masked",
            reqs=(11, 7),
            tier="smoke",
            s_q=160,
            s_kv=64,
            left_bound=7,
            right_bound=0,
        )
    )
    add(
        _mkb(
            "window_br_q_gt_kv_fully_masked",
            reqs=(11, 6, 7),
            s_q=160,
            s_kv=64,
            left_bound=7,
            right_bound=0,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "window_band_tl_q_gt_kv_rows_masked",
            reqs=(11,),
            s_q=160,
            s_kv=64,
            left_bound=2,
            right_bound=3,
        )
    )
    for d in BWD_HEAD_DIMS:
        add(
            _mkb(
                f"window_d{d}_gqa_br",
                reqs=(11, 3, 9, 6),
                d=d,
                h_k=2,
                s_q=64,
                s_kv=192,
                left_bound=31,
                right_bound=0,
                diagonal="bottom_right",
            )
        )
    add(
        _mkb(
            "window_bf16_strided_bshd",
            reqs=(11, 2),
            dtype="bf16",
            layout="strided_bshd",
            s_q=128,
            left_bound=31,
            right_bound=0,
        )
    )
    return out


def _padding_cases():
    out = []
    add = out.append
    sets = {
        "eq": ((45, 19, 31), (45, 19, 31)),
        "q_lt_kv": ((17, 33, 5), (70, 77, 41)),
        "q_gt_kv": ((77, 50, 9), (33, 40, 21)),
    }
    smoke = {("q_lt_kv", "tl"), ("q_gt_kv", "br")}
    for tag, (ql, kl) in sets.items():
        for kind in ("none", "tl", "br"):
            add(
                _mkb(
                    f"padding_{tag}_{kind}",
                    reqs=(10,)
                    + ((5,) if kind == "tl" else (6,) if kind == "br" else ()),
                    tier="smoke" if (tag, kind) in smoke else "full",
                    b=3,
                    d=64,
                    q_lens=ql,
                    kv_lens=kl,
                    rdna=(tag, kind) == ("eq", "tl"),
                    **_mask(kind),
                )
            )
    add(
        _mkb(
            "padding_some_batches_full_length",
            reqs=(10, 6),
            b=4,
            q_lens=(64, 64, 64, 30),
            kv_lens=(64, 40, 64, 64),
            d=64,
            **_mask("br"),
        )
    )
    add(
        _mkb(
            "padding_kv_only_shorter",
            reqs=(10,),
            tier="smoke",
            b=3,
            q_lens=(64, 64, 64),
            kv_lens=(64, 37, 9),
        )
    )
    add(
        _mkb(
            "padding_q_only_shorter",
            reqs=(10, 5),
            b=3,
            q_lens=(64, 20, 3),
            kv_lens=(64, 64, 64),
            **_mask("tl"),
        )
    )
    add(
        _mkb(
            "padding_single_key_batches",
            reqs=(10, 8),
            b=3,
            q_lens=(33, 33, 33),
            kv_lens=(33, 1, 2),
            d=64,
        )
    )
    add(
        _mkb(
            "padding_window_br",
            reqs=(10, 11, 6),
            b=3,
            q_lens=(50, 30, 12),
            kv_lens=(90, 60, 70),
            d=64,
            left_bound=15,
            right_bound=0,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "padding_gqa_strided_bshd_bf16",
            reqs=(10, 3, 2, 5),
            dtype="bf16",
            layout="strided_bshd",
            b=3,
            h_k=2,
            q_lens=(45, 19, 31),
            kv_lens=(45, 19, 31),
            d=64,
            **_mask("tl"),
        )
    )
    add(
        _mkb(
            "padding_packed_qkv",
            reqs=(10, 2, 5),
            layout="packed_qkv",
            b=3,
            q_lens=(45, 19, 31),
            kv_lens=(45, 19, 31),
            d=64,
            **_mask("tl"),
        )
    )
    add(
        _mkb(
            "padding_tails_d128_br",
            reqs=(10, 8, 9, 6),
            b=4,
            q_lens=(17, 33, 72, 100),
            kv_lens=(100, 72, 33, 17),
            **_mask("br"),
        )
    )
    return out


def _ragged_cases():
    out = []
    add = out.append
    sets = {
        "eq": ((45, 19, 31), (45, 19, 31)),
        "q_lt_kv": ((17, 33, 5), (70, 77, 41)),
        "q_gt_kv": ((77, 50, 9), (33, 40, 21)),
        "tails": ((17, 33, 72, 100), (100, 72, 33, 17)),
        "tile_edges": ((64, 65, 63), (128, 129, 127)),
    }
    smoke = {("q_lt_kv", "br"), ("q_gt_kv", "tl")}
    for tag, (ql, kl) in sets.items():
        for kind in ("none", "tl", "br"):
            if (tag, kind) in smoke:
                tier = "smoke"
            elif tag == "tile_edges" and kind == "none":
                tier = "exhaustive"
            else:
                tier = "full"
            add(
                _mkb(
                    f"ragged_{tag}_{kind}",
                    reqs=((5,) if kind == "tl" else (6,) if kind == "br" else ())
                    + (8,),
                    tier=tier,
                    b=len(ql),
                    d=64,
                    q_lens=ql,
                    kv_lens=kl,
                    length_mode="ragged",
                    rdna=(tag, kind) == ("eq", "tl"),
                    **_mask(kind),
                )
            )
    add(
        _mkb(
            "ragged_gqa4_br",
            reqs=(3, 6),
            b=3,
            h_k=2,
            q_lens=(17, 33, 5),
            kv_lens=(70, 77, 41),
            length_mode="ragged",
            **_mask("br"),
        )
    )
    add(
        _mkb(
            "ragged_mqa_bf16_tl",
            reqs=(3, 5, 1),
            dtype="bf16",
            b=4,
            h_k=1,
            q_lens=(40, 9, 128, 1),
            kv_lens=(40, 9, 128, 1),
            length_mode="ragged",
            **_mask("tl"),
        )
    )
    add(
        _mkb(
            "ragged_window_br",
            reqs=(11, 6),
            b=3,
            d=64,
            q_lens=(30, 12, 50),
            kv_lens=(90, 60, 70),
            length_mode="ragged",
            left_bound=15,
            right_bound=0,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "ragged_single_sequence",
            reqs=(5,),
            b=1,
            d=64,
            q_lens=(100,),
            kv_lens=(100,),
            length_mode="ragged",
            **_mask("tl"),
        )
    )
    add(
        _mkb(
            "ragged_many_short_sequences",
            reqs=(8, 6),
            b=16,
            h_q=4,
            d=64,
            q_lens=tuple(range(1, 17)),
            kv_lens=tuple(range(1, 17)),
            length_mode="ragged",
            **_mask("br"),
        )
    )
    add(
        _mkb(
            "ragged_long_and_short_d128",
            reqs=(9, 5),
            b=4,
            h_k=2,
            q_lens=(300, 1, 513, 77),
            kv_lens=(300, 1, 513, 77),
            length_mode="ragged",
            **_mask("tl"),
        )
    )
    return out


def _decode_cases():
    """``s_q == 1`` backward: tagged ``decode_bwd``; never in the smoke tier."""
    out = []
    add = out.append
    for skv, kind, tier in (
        (17, "br", "full"),
        (128, "br", "exhaustive"),
        (1000, "br", "full"),
        (17, "tl", "exhaustive"),
        (300, "none", "exhaustive"),
    ):
        add(
            _mkb(
                f"decode_skv{skv}_{kind}",
                reqs=(13, 8) + ((5,) if kind == "tl" else (6,) if kind == "br" else ()),
                tier=tier,
                b=4,
                h_k=2,
                s_q=1,
                s_kv=skv,
                **_mask(kind),
            )
        )
    add(
        _mkb(
            "decode_window_w64",
            reqs=(13, 11, 6),
            tier="exhaustive",
            h_k=2,
            s_q=1,
            s_kv=300,
            left_bound=64,
            right_bound=0,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "decode_padded_kv_lens",
            reqs=(13, 10, 6),
            b=4,
            h_k=2,
            q_lens=(1, 1, 1, 1),
            kv_lens=(300, 17, 1, 129),
            **_mask("br"),
        )
    )
    add(
        _mkb(
            "decode_ragged_kv_lens",
            reqs=(13, 6),
            b=4,
            h_k=2,
            q_lens=(1, 1, 1, 1),
            kv_lens=(300, 17, 1, 129),
            length_mode="ragged",
            **_mask("br"),
        )
    )
    return out


# ---------------------------------------------------------------------------
# cross-product: GQA ratio x length mode x mask x q/kv relation
# ---------------------------------------------------------------------------

_BX_RATIOS = {"mha": (8, 8), "gqa4": (8, 2), "mqa": (8, 1)}
_BX_LENGTHS = ("fixed", "padded", "ragged")
_BX_MASKS = ("none", "tl", "br", "tl_win", "br_win")
_BX_REL = ("lt", "eq", "gt")
_BX_DTYPE_D = (("fp16", 64), ("bf16", 64), ("fp16", 128), ("bf16", 128))
_BX_LAYOUTS = ("bhsd", "bshd", "strided_bshd", "strided_bhsd")
_BX_FIXED = {"lt": (33, 100), "eq": (72, 72), "gt": (100, 33)}
_BX_LENS = {
    "lt": ((17, 33, 5), (70, 77, 41)),
    "eq": ((45, 19, 31), (45, 19, 31)),
    "gt": ((77, 50, 9), (33, 40, 21)),
}
_BX_HAND_SMOKE = {
    ("gqa4", "ragged", "br_win", "gt"),
    ("gqa4", "padded", "br", "lt"),
    ("mqa", "fixed", "tl_win", "eq"),
}


def _cross_case(idx, combo, tier):
    ratio, lens, mask, rel = combo
    dt, d = _BX_DTYPE_D[idx % 4]
    hq, hk = _BX_RATIOS[ratio]
    kw = {}
    reqs = {3, {"fixed": 8, "padded": 10, "ragged": 8}[lens]}
    if mask in ("tl", "br"):
        kw.update(_mask(mask))
        reqs.add(5 if mask == "tl" else 6)
    elif mask in ("tl_win", "br_win"):
        kw.update(
            diagonal="top_left" if mask == "tl_win" else "bottom_right",
            left_bound=15,
            right_bound=0,
        )
        reqs.update((11, 5 if mask == "tl_win" else 6))
    if lens == "fixed":
        sq, sk = _BX_FIXED[rel]
        shape = {"s_q": sq, "s_kv": sk, "layout": _BX_LAYOUTS[idx % 4]}
    else:
        ql, kl = _BX_LENS[rel]
        shape = {"q_lens": ql, "kv_lens": kl, "length_mode": lens}
        if lens == "padded":
            shape["layout"] = _BX_LAYOUTS[idx % 4]
    return _mkb(
        f"x_{ratio}_{lens}_{mask}_{rel}",
        reqs=reqs,
        tier=tier,
        tags=("cross",),
        dtype=dt,
        d=d,
        b=3,
        h_q=hq,
        h_k=hk,
        **shape,
        **kw,
    )


def _cross_cases():
    import itertools

    combos = list(itertools.product(_BX_RATIOS, _BX_LENGTHS, _BX_MASKS, _BX_REL))
    rows = [c + (_BX_DTYPE_D[i % 4], _BX_LAYOUTS[i % 4]) for i, c in enumerate(combos)]
    picked = set(_pairwise_cover(rows))
    out = []
    for i, c in enumerate(combos):
        tier = (
            "smoke"
            if c in _BX_HAND_SMOKE
            else ("full" if i in picked else "exhaustive")
        )
        out.append(_cross_case(i, c, tier))
    return out


def _intersection_cases():
    out = []
    add = out.append
    add(
        _mkb(
            "x_gqa_ragged_br_window_q_gt_kv_bf16",
            reqs=(3, 6, 11, 1, 7),
            tags=("cross",),
            dtype="bf16",
            b=3,
            h_k=2,
            q_lens=(77, 50, 9),
            kv_lens=(33, 40, 21),
            length_mode="ragged",
            left_bound=7,
            right_bound=0,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "x_packed_qkv_gqa_padded_br",
            reqs=(2, 3, 10, 6),
            tier="smoke",
            tags=("cross",),
            layout="packed_qkv",
            b=3,
            h_k=2,
            q_lens=(45, 19, 31),
            kv_lens=(45, 19, 31),
            d=64,
            **_mask("br"),
        )
    )
    add(
        _mkb(
            "x_mixed_gqa_padded_window_tl_bf16",
            reqs=(2, 3, 10, 11, 5, 1),
            tags=("cross",),
            dtype="bf16",
            layout="mixed",
            b=3,
            h_k=4,
            q_lens=(72, 33, 17),
            kv_lens=(72, 33, 17),
            left_bound=16,
            right_bound=0,
        )
    )
    add(
        _mkb(
            "x_strided_bhsd_padded_tails_scale_br",
            reqs=(2, 4, 8, 10, 6),
            tags=("cross",),
            layout="strided_bhsd",
            b=4,
            q_lens=(17, 33, 72, 100),
            kv_lens=(33, 72, 100, 17),
            scale=0.3,
            **_mask("br"),
        )
    )
    add(
        _mkb(
            "x_mqa_padded_q_gt_kv_window_fully_masked",
            reqs=(3, 10, 11, 6, 7),
            tags=("cross",),
            b=3,
            h_k=1,
            d=64,
            q_lens=(100, 72, 33),
            kv_lens=(17, 33, 9),
            left_bound=4,
            right_bound=0,
            diagonal="bottom_right",
        )
    )
    add(
        _mkb(
            "x_d32_gqa_ragged_br",
            reqs=(9, 3, 6),
            tier="exhaustive",
            tags=("cross", "optional"),
            d=32,
            b=3,
            h_k=2,
            q_lens=(17, 33, 5),
            kv_lens=(70, 77, 41),
            length_mode="ragged",
            **_mask("br"),
        )
    )
    return out


_BWD_BUILDERS = (
    _dtype_headdim_cases,
    _layout_cases,
    _gqa_scale_cases,
    _mask_lse_cases,
    _seqlen_cases,
    _window_cases,
    _padding_cases,
    _ragged_cases,
    _decode_cases,
    _intersection_cases,
    _cross_cases,
)
_BWD_CACHE = None


def backward_cases():
    """The deterministic, ordered backward table."""
    global _BWD_CACHE
    if _BWD_CACHE is None:
        _BWD_CACHE = tuple(c for fn in _BWD_BUILDERS for c in fn())
    return _BWD_CACHE


def select_bwd(tag="full", *, arch=None, reqs=None, max_cost=None, include_decode=True):
    """Backward cases carrying ``tag``, optionally filtered.

    ``include_decode=False`` drops the ``decode_bwd`` group (``s_q == 1``).
    """
    out = []
    for c in backward_cases():
        if tag is not None and tag not in c.tags:
            continue
        if arch is not None and arch not in c.archs:
            continue
        if reqs is not None and not set(reqs) <= set(c.reqs):
            continue
        if max_cost is not None and c.cost > max_cost:
            continue
        if not include_decode and "decode_bwd" in c.tags:
            continue
        out.append(c)
    return out


def get_bwd_case(case_id):
    for c in backward_cases():
        if c.id == case_id:
            return c
    raise KeyError(case_id)


def bwd_table_summary():
    cases = backward_cases()
    out = {
        "total": len(cases),
        "decode_bwd": sum("decode_bwd" in c.tags for c in cases),
    }
    for t in TIERS:
        out[t] = sum(t in c.tags for c in cases)
    return out


def validate_bwd_case(case):
    """Forward validity rules plus the backward scope rules."""
    e = list(validate_case(case))
    if not case.id.startswith("bwd_"):
        e.append("backward case ids start with bwd_")
    if "bwd" not in case.tags:
        e.append("backward case lacks the bwd tag")
    if BWD_REQ not in case.reqs:
        e.append("backward case lacks coverage item 15 (bwd)")
    if (case.length_mode == "ragged") != (BWD_RAGGED_REQ in case.reqs):
        e.append("coverage item 16 (bwd_ragged) iff ragged")
    if case.d not in BWD_HEAD_DIMS + BWD_OPTIONAL_HEAD_DIMS:
        e.append(f"head dim {case.d} is outside the backward scope")
    if case.d in BWD_OPTIONAL_HEAD_DIMS and "optional" not in case.tags:
        e.append("optional head dims must be tagged optional")
    if case.bias is not None:
        e.append("bias is out of scope for backward")
    if case.emit_lse:
        e.append("backward cases consume LSE; emit_lse stays False")
    if case.decode != ("decode_bwd" in case.tags):
        e.append("decode_bwd tag must match the decode flag")
    if case.decode and "smoke" in case.tags:
        e.append("decode backward is kept out of the smoke tier")
    return e


# ---------------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------------


def layout_tensors_bwd(case):
    """Descriptors and buffer sizes for q, k, v, o, do, dq, dk, dv.

    ``o`` and ``do`` share the output layout; ``dq``/``dk``/``dv`` mirror the
    ``q``/``k``/``v`` layout in their own buffers (one ``dqkv`` buffer for
    packed QKV).
    """
    descs, sizes = layout_tensors(case)
    descs = dict(descs)
    sizes = dict(sizes)
    od = descs["o"]
    descs["do"] = TensorDesc("do", od.offset, od.dims, od.strides)
    sizes["do"] = sizes["o"]
    for n in "qkv":
        src = descs[n]
        buf = "dqkv" if src.buffer == "qkv" else f"d{n}"
        descs[f"d{n}"] = TensorDesc(buf, src.offset, src.dims, src.strides)
        sizes[buf] = sizes[src.buffer]
    return descs, sizes


@dataclass
class SdpaBwdInputs:
    """Materialised backward problem (the forward inputs plus O, dO, LSE)."""

    case: object
    fwd: SdpaInputs
    buffers: dict  # q/k/v (or qkv), o, do -> float32 values rounded to the dtype; lse -> fp32
    descs: dict  # q, k, v, o, do, dq, dk, dv -> TensorDesc
    sizes: dict  # includes the gradient buffers the consumer allocates
    o: np.ndarray  # logical float64 views (dense [B,H,S,D] or ragged [T,H,D])
    do: np.ndarray
    lse: np.ndarray  # fp32 values as float64, lse_shape(case); garbage on padded rows
    valid_q: np.ndarray  # bool, lse_shape(case)
    valid_kv: np.ndarray  # bool, [B, S_kv] (dense) or [T_kv] (ragged)

    def __getattr__(self, name):
        # forward fields (q, k, v, scale, seq_len_*, offsets) come from ``fwd``
        if name in (
            "q",
            "k",
            "v",
            "scale",
            "seq_len_q",
            "seq_len_kv",
            "q_offsets",
            "kv_offsets",
        ):
            return getattr(self.fwd, name)
        raise AttributeError(name)


def _bwd_seed(case, seed):
    base = case_seed(case) if seed is None else int(seed)
    return (base ^ zlib.crc32(b"backward")) & 0xFFFFFFFF


def _valid_masks(case):
    lq, lk = case.lens()
    if case.length_mode == "ragged":
        return np.ones((sum(lq), case.h_q), dtype=bool), np.ones(sum(lk), dtype=bool)
    vq = np.arange(case.s_q)[None, None, :] < np.array(lq)[:, None, None]
    vq = np.broadcast_to(vq, (case.b, case.h_q, case.s_q)).copy()
    vk = np.arange(case.s_kv)[None, :] < np.array(lk)[:, None]
    return vq, vk


def materialize_bwd(case, seed=None):
    """Deterministic backward inputs for ``case``.

    Q/K/V come from :func:`materialize`; ``O`` and ``LSE`` come from the
    float64 oracle forward (``O`` rounded to the dtype, ``LSE`` to fp32);
    ``dO`` is random in the dtype. Padded rows of O, dO and LSE and every
    stride gap hold finite garbage.
    """
    if case.bias is not None:
        raise ValueError("backward cases carry no bias")
    fwd = materialize(case, seed)
    fref = evaluate(fwd)
    rng = np.random.default_rng(_bwd_seed(case, seed))
    descs, sizes = layout_tensors_bwd(case)
    buffers = dict(fwd.buffers)
    vq, vk = _valid_masks(case)
    rows = vq[..., None]
    logical = {}
    for n, values in (("o", fref.o), ("do", None)):
        dsc = descs[n]
        buf = round_to(rng.standard_normal(sizes[n]) * 4.0, case.dtype)
        view = _view(buf, dsc)
        if values is None:
            values = rng.standard_normal(dsc.dims)
        view[...] = np.where(rows, round_to(values, case.dtype), view)
        buffers[n] = buf
        logical[n] = view.astype(np.float64)
    lse = fref.lse.astype(np.float32)
    lse = np.where(vq, lse, (rng.standard_normal(lse.shape) * 4.0).astype(np.float32))
    buffers["lse"] = np.ascontiguousarray(lse, dtype=np.float32)
    return SdpaBwdInputs(
        case=case,
        fwd=fwd,
        buffers=buffers,
        descs=descs,
        sizes=sizes,
        o=logical["o"],
        do=logical["do"],
        lse=buffers["lse"].astype(np.float64),
        valid_q=vq,
        valid_kv=vk,
    )


def to_torch_bwd(inputs, device="cuda"):
    """Torch tensors for a backward launch (lazy torch import).

    q, k, v, o, do are strided views of device buffers holding the dtype
    values; dq, dk, dv are zero-filled strided views; ``lse`` is fp32 with
    :func:`lse_shape`; plus the length tensors / offsets and the scale.
    """
    import torch

    case = inputs.case
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[case.dtype]
    dev = {
        n: torch.from_numpy(a).to(device=device, dtype=tdt)
        for n, a in inputs.buffers.items()
        if n != "lse"
    }
    for n, size in inputs.sizes.items():
        if n not in dev:
            dev[n] = torch.zeros(size, dtype=tdt, device=device)
    out = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        dsc = inputs.descs[n]
        out[n] = torch.as_strided(dev[dsc.buffer], dsc.dims, dsc.strides, dsc.offset)
    out["lse"] = torch.from_numpy(inputs.buffers["lse"]).to(device)
    for name in ("seq_len_q", "seq_len_kv", "q_offsets", "kv_offsets"):
        a = getattr(inputs.fwd, name)
        out[name] = None if a is None else torch.from_numpy(a).to(device)
    out["scale"] = inputs.fwd.scale
    return out


# ---------------------------------------------------------------------------
# oracle, comparison, low-precision simulation
# ---------------------------------------------------------------------------


@dataclass
class SdpaBwdReference:
    dq: np.ndarray
    dk: np.ndarray
    dv: np.ndarray
    delta: np.ndarray
    valid_q: np.ndarray
    valid_kv: np.ndarray
    dead_q: np.ndarray  # valid query rows that are fully masked (LSE = -inf)


def _oracle_kwargs(inputs):
    case = inputs.case
    left, right, tl = case.band
    kw = dict(scale=inputs.fwd.scale, left_bound=left, right_bound=right, top_left=tl)
    if case.length_mode == "ragged":
        kw.update(q_offsets=inputs.fwd.q_offsets, kv_offsets=inputs.fwd.kv_offsets)
    elif case.length_mode == "padded":
        kw.update(
            seq_len_q=inputs.fwd.seq_len_q.reshape(-1),
            seq_len_kv=inputs.fwd.seq_len_kv.reshape(-1),
        )
    return kw


def evaluate_bwd(inputs):
    """Run the float64 backward oracle on exactly the stored inputs."""
    f = inputs.fwd
    res = sdpa_reference_bwd(
        f.q, f.k, f.v, inputs.o, inputs.do, inputs.lse, **_oracle_kwargs(inputs)
    )
    dead = inputs.valid_q & np.isneginf(inputs.lse)
    return SdpaBwdReference(
        dq=res.dq,
        dk=res.dk,
        dv=res.dv,
        delta=res.delta,
        valid_q=inputs.valid_q,
        valid_kv=inputs.valid_kv,
        dead_q=dead,
    )


def _row_masks(case, ref):
    """Element masks (broadcastable to dq / dk / dv) of the compared rows."""
    mq = ref.valid_q[..., None]
    if case.length_mode == "ragged":
        mk = ref.valid_kv[:, None, None]
    else:
        mk = ref.valid_kv[:, None, :, None]
    return {"dq": mq, "dk": mk, "dv": mk}


def compare_bwd(case, dq, dk, dv, ref, *, tol=None):
    """Compare kernel gradients to the oracle; returns a list of problems.

    Only valid rows are compared (padding storage is the caller's). For each
    tensor: no NaN/Inf, ``max|got - ref| <= atol + rtol * g * max|ref|``
    (``g`` = :func:`kv_growth`), and per row (query rows of dq, key rows of
    dk / dv, reduced over the head dim)
    ``max_D|got - ref| <= atol + rtol * g * max(max_D|ref|, f * max|ref|)``
    with ``f`` = :data:`ROW_FLOOR_BWD`. Fully masked query rows must have
    ``|dq| <= atol``.
    """
    t = tol or TOLERANCE_BWD[case.dtype]
    rtol, atol = t["rtol"], t["atol"]
    g = kv_growth(case)
    masks = _row_masks(case, ref)
    probs = []
    for name, got in zip(GRADS, (dq, dk, dv)):
        want = getattr(ref, name)
        got = np.asarray(got, dtype=np.float64)
        if got.shape != want.shape:
            probs.append(f"{name} shape {got.shape} != {want.shape}")
            continue
        rows = np.broadcast_to(masks[name][..., 0], got.shape[:-1])
        gv, wv = got[rows], want[rows]  # [n_rows, D]
        if gv.size == 0:
            continue
        if not np.isfinite(gv).all():
            probs.append(f"non-finite values in valid rows of {name}")
            continue
        diff = np.abs(gv - wv)
        err = float(diff.max())
        mx = float(np.abs(wv).max())
        bound = atol + rtol * g * mx
        if err > bound:
            probs.append(
                f"{name} mismatch: max abs err {err:.3g} > {bound:.3g} (max |ref| {mx:.3g})"
            )
            continue
        row_err = diff.max(axis=-1)
        row_bound = atol + rtol * g * np.maximum(
            np.abs(wv).max(axis=-1), ROW_FLOOR_BWD * mx
        )
        bad = row_err > row_bound
        if bad.any():
            i = int(np.argmax(row_err / row_bound))
            probs.append(
                f"{name} row mismatch in {int(bad.sum())} of {bad.size} rows: worst row err "
                f"{row_err[i]:.3g} > {row_bound[i]:.3g}"
            )
    if ref.dead_q.any():
        dqa = np.asarray(dq, dtype=np.float64)
        if dqa.shape == ref.dq.shape:
            dead = dqa[ref.dead_q]
            if not np.isfinite(dead).all() or np.abs(dead).max() > atol:
                probs.append("fully masked rows of dq are not zero")
    return probs


def error_ratios(case, dq, dk, dv, ref):
    """Calibration helper: per tensor, ``name`` = ``max|got - ref| / (g * max|ref|)``
    and ``name_row`` = the worst per-row ``max_D|got - ref| / (g * max(max_D|ref|,
    f * max|ref|))`` (the quantities :func:`compare_bwd` bounds by ``rtol``)."""
    g = kv_growth(case)
    masks = _row_masks(case, ref)
    out = {}
    for name, got in zip(GRADS, (dq, dk, dv)):
        want = getattr(ref, name)
        rows = np.broadcast_to(masks[name][..., 0], want.shape[:-1])
        if not rows.any():
            out[name] = out[name + "_row"] = 0.0
            continue
        wv = want[rows]
        diff = np.abs(np.asarray(got, dtype=np.float64)[rows] - wv)
        mx = float(np.abs(wv).max())
        err = float(diff.max())
        out[name] = err / (g * mx) if mx > 0 else err
        den = g * np.maximum(np.abs(wv).max(axis=-1), ROW_FLOOR_BWD * mx)
        rerr = diff.max(axis=-1)
        out[name + "_row"] = float(
            np.max(np.where(den > 0, rerr / np.where(den > 0, den, 1.0), rerr))
        )
    return out


def _lowp_head(q, k, v, o, do, lse, scale, band, dtype):
    f32 = np.float32
    q, k, v, o, do = (a.astype(f32) for a in (q, k, v, o, do))
    lse = lse.astype(f32)
    live = np.isfinite(lse)
    keepm = band & live[:, None]
    delta = (do * o).sum(axis=1, dtype=f32)
    s = (q @ k.T) * f32(scale)
    p = np.where(
        keepm, np.exp(np.where(keepm, s - np.where(live, lse, 0)[:, None], 0)), 0
    )
    p = p.astype(f32)
    p_lo = round_to(p, dtype)
    dv = p_lo.T @ do
    dp = do @ v.T
    ds = (p * (dp - delta[:, None])).astype(f32)
    ds_lo = round_to(ds, dtype)
    dq = (ds_lo @ k) * f32(scale)
    dk = (ds_lo.T @ q) * f32(scale)
    return dq, dk, dv


def simulate_lowp_bwd(inputs, *, round_head_partials=False):
    """fp32 simulation of a fp32-accumulate low-precision backward kernel.

    Mirrors the usual flash-attention backward data flow: inputs are already
    in the case dtype, ``delta`` and ``P = exp(S - LSE)`` are fp32, ``P`` and
    ``dS`` are rounded to the dtype before the dV / dQ / dK matrix products
    (fp32 accumulate), and the gradients are rounded to the dtype on store.
    ``round_head_partials=True`` also rounds each query head's dK / dV
    contribution to the dtype before the GQA group sum (a kernel that stores
    per-query-head partials in the dtype and reduces them afterwards).
    Used to calibrate :data:`TOLERANCE_BWD` and as a must-pass self-check.
    """
    case = inputs.case
    f = inputs.fwd
    left, right, tl = case.band
    lq, lk = case.lens()
    dq = np.zeros(f.q.shape)
    dk = np.zeros(f.k.shape)
    dv = np.zeros(f.v.shape)
    ragged = case.length_mode == "ragged"
    qo = np.concatenate([[0], np.cumsum(lq)])
    ko = np.concatenate([[0], np.cumsum(lk)])
    gk, gv = case.h_q // case.h_k, case.h_q // case.h_v
    for b in range(case.b):
        sq, skv = lq[b], lk[b]
        qi = np.arange(sq)[:, None]
        ki = np.arange(skv)[None, :]
        off = 0 if tl else skv - sq
        band = np.ones((sq, skv), dtype=bool)
        if right >= 0:
            band &= ki <= qi + off + right
        if left >= 0:
            band &= ki >= qi + off - left
        for h in range(case.h_q):
            hk, hv = h // gk, h // gv
            if ragged:
                qs, ks = slice(qo[b], qo[b + 1]), slice(ko[b], ko[b + 1])
                args = (
                    f.q[qs, h],
                    f.k[ks, hk],
                    f.v[ks, hv],
                    inputs.o[qs, h],
                    inputs.do[qs, h],
                )
                l_ = inputs.lse[qs, h]
            else:
                args = (
                    f.q[b, h, :sq],
                    f.k[b, hk, :skv],
                    f.v[b, hv, :skv],
                    inputs.o[b, h, :sq],
                    inputs.do[b, h, :sq],
                )
                l_ = inputs.lse[b, h, :sq]
            a, c, e = _lowp_head(*args, l_, f.scale, band, case.dtype)
            if round_head_partials:
                c = round_to(c, case.dtype).astype(np.float64)
                e = round_to(e, case.dtype).astype(np.float64)
            if ragged:
                dq[qs, h] = a
                dk[ks, hk] += c
                dv[ks, hv] += e
            else:
                dq[b, h, :sq] = a
                dk[b, hk, :skv] += c
                dv[b, hv, :skv] += e
    rnd = lambda x: round_to(x, case.dtype).astype(np.float64)  # noqa: E731
    return rnd(dq), rnd(dk), rnd(dv)


def reference_bwd_for(case, seed=None):
    """Materialise ``case`` (a case or id) and evaluate the backward oracle."""
    case = get_bwd_case(case) if isinstance(case, str) else case
    inp = materialize_bwd(case, seed)
    return inp, evaluate_bwd(inp)


__all__ = [
    "BWD_HEAD_DIMS",
    "BWD_OPTIONAL_HEAD_DIMS",
    "CDNA_ARCHS",
    "GRADS",
    "ROW_FLOOR_BWD",
    "SdpaBwdInputs",
    "SdpaBwdReference",
    "TOLERANCE_BWD",
    "backward_cases",
    "bwd_table_summary",
    "compare_bwd",
    "error_ratios",
    "evaluate_bwd",
    "get_bwd_case",
    "kv_growth",
    "layout_tensors_bwd",
    "lse_shape",
    "materialize_bwd",
    "reference_bwd_for",
    "select_bwd",
    "simulate_lowp_bwd",
    "to_torch_bwd",
    "validate_bwd_case",
]
