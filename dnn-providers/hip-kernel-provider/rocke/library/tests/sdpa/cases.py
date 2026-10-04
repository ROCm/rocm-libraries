# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared SDPA forward case table.

Kernel-facing tests of the attention forward surface draw their problems from
:func:`all_cases` so that every stream agrees on what "covered" means. The
case model, validity rules, input materialisation and oracle helpers are in
``helpers.py`` and re-exported here.

Tiers (every case carries the tags of every tier it belongs to)::

    smoke       small; a few minutes of device time across both CDNA arches
    full        smoke + every single-feature case + a pairwise cover of the
                feature cross-product
    exhaustive  full + the complete factorial of the cross-product and the
                remaining tile-edge sweeps

Other tags: ``cross`` (feature intersections), ``optional`` (outside the
required matrix), ``deprecated_form``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

from .helpers import *  # noqa: F401,F403
from .helpers import (
    ALL_ARCHS,
    CDNA_ARCHS,
    DTYPES,
    TIERS,
    BiasSpec,
    SdpaCase,
    has_fully_masked_rows,
)

_TIER_TAGS = {
    "smoke": frozenset(TIERS),
    "full": frozenset(("full", "exhaustive")),
    "exhaustive": frozenset(("exhaustive",)),
}


def _mk(
    cid,
    *,
    reqs,
    tier="full",
    tags=(),
    dtype="fp16",
    b=2,
    h_q=8,
    h_k=None,
    h_v=None,
    s_q=128,
    s_kv=None,
    d=128,
    rdna=False,
    q_lens=None,
    kv_lens=None,
    length_mode=None,
    layout=None,
    **kw,
):
    """Build one case; ``rdna=True`` also applies it to the RDNA targets (d != 256)."""
    h_k = h_q if h_k is None else h_k
    h_v = h_k if h_v is None else h_v
    if q_lens is not None:
        s_q, s_kv = max(q_lens), max(kv_lens)
    s_kv = s_q if s_kv is None else s_kv
    if length_mode is None:
        length_mode = "fixed" if q_lens is None else "padded"
    if layout is None:
        layout = "thd" if length_mode == "ragged" else "bhsd"
    case = SdpaCase(
        id=cid,
        archs=ALL_ARCHS if (rdna and d != 256) else CDNA_ARCHS,
        dtype=dtype,
        b=b,
        h_q=h_q,
        h_k=h_k,
        h_v=h_v,
        s_q=s_q,
        s_kv=s_kv,
        d=d,
        layout=layout,
        length_mode=length_mode,
        q_lens=None if q_lens is None else tuple(q_lens),
        kv_lens=None if kv_lens is None else tuple(kv_lens),
        reqs=tuple(sorted(set(reqs))),
        tags=_TIER_TAGS[tier] | frozenset(tags),
        **kw,
    )
    lq, _ = case.lens()
    return replace(
        case,
        decode=all(n == 1 for n in lq),
        fully_masked_rows=has_fully_masked_rows(case),
    )


def _mask(kind):
    """Bound/alignment kwargs for a named causal mask."""
    return {
        "none": {},
        "tl": {"diagonal": "top_left", "right_bound": 0},
        "br": {"diagonal": "bottom_right", "right_bound": 0},
    }[kind]


# ---------------------------------------------------------------------------
# single-feature groups
# ---------------------------------------------------------------------------


def _dtype_headdim_cases():
    out = []
    add = out.append
    smoke = {("fp16", 128), ("bf16", 64), ("bf16", 256)}
    for dt in DTYPES:
        for d in (64, 128, 256):
            add(
                _mk(
                    f"dtype_{dt}_d{d}_mha",
                    reqs=(1, 9),
                    tier="smoke" if (dt, d) in smoke else "full",
                    dtype=dt,
                    d=d,
                    rdna=True,
                )
            )
            add(
                _mk(
                    f"dtype_{dt}_d{d}_causal_tl_lse",
                    reqs=(1, 5, 7, 9),
                    dtype=dt,
                    d=d,
                    s_q=256,
                    emit_lse=True,
                    rdna=True,
                    **_mask("tl"),
                )
            )
            add(
                _mk(
                    f"headdim_{dt}_d{d}_br_gqa_lse",
                    reqs=(3, 6, 7, 9),
                    dtype=dt,
                    d=d,
                    h_k=2,
                    s_q=64,
                    s_kv=192,
                    emit_lse=True,
                    **_mask("br"),
                )
            )
    for d in (32, 72, 96, 192):
        add(
            _mk(
                f"headdim_optional_d{d}",
                reqs=(9,),
                tier="exhaustive",
                tags=("optional",),
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
                _mk(
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
            _mk(
                f"layout_{layout}_gqa_br_lse",
                reqs=(2, 3, 6, 7),
                layout=layout,
                h_k=2,
                s_q=48,
                s_kv=160,
                emit_lse=True,
                **_mask("br"),
            )
        )
    for d in (64, 128, 256):
        add(
            _mk(
                f"layout_packed_qkv_mha_d{d}",
                reqs=(2, 9),
                tier="smoke" if d == 128 else "full",
                layout="packed_qkv",
                d=d,
                s_q=160,
                **_mask("tl"),
            )
        )
    add(
        _mk(
            "layout_packed_qkv_gqa_bf16",
            reqs=(2, 3),
            dtype="bf16",
            layout="packed_qkv",
            h_k=2,
            s_q=96,
            **_mask("tl"),
        )
    )
    add(
        _mk(
            "layout_packed_qkv_noncausal_lse",
            reqs=(2, 7),
            layout="packed_qkv",
            s_q=64,
            d=64,
            emit_lse=True,
        )
    )
    add(
        _mk(
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
        (12, 4, 4, "gqa3_h12"),
        (64, 8, 8, "gqa8_h64"),
        (8, 4, 2, "kv_heads_differ"),
        (8, 2, 8, "k_fewer_than_v"),
    ):
        optional = tag in ("kv_heads_differ", "k_fewer_than_v")
        add(
            _mk(
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
            _mk(
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
        _mk(
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
                _mk(
                    f"causal_{kind}_{rel}",
                    reqs=(req,),
                    tier="smoke" if (kind, rel) in smoke else "full",
                    s_q=sq,
                    s_kv=sk,
                    rdna=rel == "eq",
                    **_mask(kind),
                )
            )
            add(
                _mk(
                    f"causal_{kind}_{rel}_bf16_lse",
                    reqs=(req, 7),
                    dtype="bf16",
                    s_q=sq,
                    s_kv=sk,
                    emit_lse=True,
                    **_mask(kind),
                )
            )
    for kind in ("top_left", "bottom_right"):
        add(
            _mk(
                f"causal_bool_{kind}_overrides_bounds",
                reqs=(5, 6),
                tier="exhaustive",
                tags=("deprecated_form",),
                s_q=80,
                s_kv=144,
                causal_bool=kind,
                left_bound=7,
                right_bound=3,
                diagonal="bottom_right" if kind == "top_left" else "top_left",
            )
        )
    add(
        _mk("mask_none_q_lt_kv", reqs=(5, 6), s_q=64, s_kv=200, diagonal="bottom_right")
    )
    add(_mk("mask_none_q_gt_kv", reqs=(5,), s_q=200, s_kv=64))
    for dt in DTYPES:
        for kind in ("none", "tl", "br"):
            add(
                _mk(
                    f"lse_{dt}_{kind}_odd_lengths",
                    reqs=(7, 8),
                    tier="smoke" if (dt, kind) == ("fp16", "br") else "full",
                    dtype=dt,
                    s_q=77,
                    s_kv=133,
                    emit_lse=True,
                    **_mask(kind),
                )
            )
    add(_mk("lse_off_base", reqs=(7,), s_q=64))
    add(
        _mk(
            "lse_fully_masked_rows_neg_inf",
            reqs=(7, 6),
            tier="smoke",
            s_q=96,
            s_kv=32,
            emit_lse=True,
            **_mask("br"),
        )
    )
    return out


def _seqlen_cases():
    out = []
    add = out.append
    pairs = [
        (1, 1),
        (1, 7),
        (7, 7),
        (13, 100),
        (31, 33),
        (33, 31),
        (65, 65),
        (127, 129),
        (129, 127),
        (191, 191),
        (255, 257),
        (257, 255),
        (513, 513),
        (1000, 1000),
        (1023, 1025),
        (1025, 1023),
    ]
    smoke_pairs = {(127, 129), (257, 255)}
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
                _mk(
                    f"seqlen_{sq}x{sk}_{kind}",
                    reqs=(8,),
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
    for dt, d in (("bf16", 128), ("fp16", 256), ("bf16", 256)):
        add(
            _mk(
                f"seqlen_odd_{dt}_d{d}_br",
                reqs=(8, 9),
                dtype=dt,
                d=d,
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
                _mk(
                    f"window_{al}_w{w}",
                    reqs=(11, 5 if al == "tl" else 6),
                    tier=tier,
                    s_q=sq,
                    s_kv=sk,
                    emit_lse=w in (0, 1, 64),
                    diagonal="top_left" if al == "tl" else "bottom_right",
                    left_bound=w,
                    right_bound=0,
                    rdna=al == "tl" and w in (1, 64),
                )
            )
    add(_mk("window_wider_than_seq", reqs=(11,), s_q=64, left_bound=500, right_bound=0))
    add(_mk("window_left_only_tl", reqs=(11,), s_q=160, left_bound=20))
    add(
        _mk(
            "window_left_only_br",
            reqs=(11, 6),
            s_q=64,
            s_kv=190,
            left_bound=20,
            diagonal="bottom_right",
        )
    )
    add(_mk("window_right_only_lookahead", reqs=(11,), s_q=160, right_bound=3))
    add(
        _mk(
            "window_symmetric_tl",
            reqs=(11,),
            tier="smoke",
            s_q=160,
            left_bound=9,
            right_bound=5,
        )
    )
    add(
        _mk(
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
        _mk("window_diagonal_only_tl", reqs=(11,), s_q=100, left_bound=0, right_bound=0)
    )
    add(
        _mk(
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
        _mk(
            "window_tl_q_gt_kv_fully_masked",
            reqs=(11, 7),
            tier="smoke",
            s_q=160,
            s_kv=64,
            left_bound=7,
            right_bound=0,
            emit_lse=True,
        )
    )
    add(
        _mk(
            "window_br_q_gt_kv_fully_masked",
            reqs=(11, 6, 7),
            s_q=160,
            s_kv=64,
            left_bound=7,
            right_bound=0,
            diagonal="bottom_right",
            emit_lse=True,
        )
    )
    add(
        _mk(
            "window_band_tl_q_gt_kv_rows_masked",
            reqs=(11,),
            s_q=160,
            s_kv=64,
            left_bound=2,
            right_bound=3,
        )
    )
    for d in (64, 256):
        add(
            _mk(
                f"window_d{d}_gqa_br",
                reqs=(11, 3, 9),
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
        _mk(
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
                _mk(
                    f"padding_{tag}_{kind}",
                    reqs=(10,)
                    + ((5,) if kind == "tl" else (6,) if kind == "br" else ()),
                    tier="smoke" if (tag, kind) in smoke else "full",
                    b=3,
                    d=64,
                    q_lens=ql,
                    kv_lens=kl,
                    emit_lse=True,
                    **_mask(kind),
                )
            )
    add(
        _mk(
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
        _mk(
            "padding_kv_only_shorter",
            reqs=(10,),
            tier="smoke",
            b=3,
            q_lens=(64, 64, 64),
            kv_lens=(64, 37, 9),
        )
    )
    add(
        _mk(
            "padding_q_only_shorter",
            reqs=(10, 5),
            b=3,
            q_lens=(64, 20, 3),
            kv_lens=(64, 64, 64),
            **_mask("tl"),
        )
    )
    add(
        _mk(
            "padding_single_key_batches",
            reqs=(10, 8),
            b=3,
            q_lens=(33, 33, 33),
            kv_lens=(33, 1, 2),
            d=64,
            emit_lse=True,
        )
    )
    add(
        _mk(
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
        _mk(
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
        _mk(
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
        _mk(
            "padding_d256_br",
            reqs=(10, 9, 6),
            d=256,
            q_lens=(50, 20),
            kv_lens=(90, 60),
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
        "tiny": ((1, 1, 1), (3, 1, 9)),
        "tile_edges": ((64, 65, 63), (128, 129, 127)),
    }
    for tag, (ql, kl) in sets.items():
        for kind in ("none", "tl", "br"):
            if (tag, kind) == ("q_lt_kv", "br"):
                tier = "smoke"
            elif tag == "tile_edges" and kind == "none":
                tier = "exhaustive"
            else:
                tier = "full"
            add(
                _mk(
                    f"ragged_{tag}_{kind}",
                    reqs=(12,)
                    + ((5,) if kind == "tl" else (6,) if kind == "br" else ()),
                    tier=tier,
                    b=3,
                    d=64,
                    q_lens=ql,
                    kv_lens=kl,
                    length_mode="ragged",
                    emit_lse=True,
                    **_mask(kind),
                )
            )
    add(
        _mk(
            "ragged_gqa4_br_lse",
            reqs=(12, 3, 6, 7),
            b=3,
            h_k=2,
            q_lens=(17, 33, 5),
            kv_lens=(70, 77, 41),
            length_mode="ragged",
            emit_lse=True,
            **_mask("br"),
        )
    )
    add(
        _mk(
            "ragged_mqa_bf16_tl",
            reqs=(12, 3, 5),
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
        _mk(
            "ragged_window_br",
            reqs=(12, 11, 6),
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
        _mk(
            "ragged_d256_br",
            reqs=(12, 9, 6),
            d=256,
            q_lens=(20, 50),
            kv_lens=(60, 90),
            length_mode="ragged",
            **_mask("br"),
        )
    )
    add(
        _mk(
            "ragged_single_sequence",
            reqs=(12, 5),
            b=1,
            d=64,
            q_lens=(100,),
            kv_lens=(100,),
            length_mode="ragged",
            **_mask("tl"),
        )
    )
    add(
        _mk(
            "ragged_many_short_sequences",
            reqs=(12, 8, 6),
            b=16,
            h_q=4,
            d=64,
            q_lens=tuple(range(1, 17)),
            kv_lens=tuple(range(1, 17)),
            length_mode="ragged",
            **_mask("br"),
        )
    )
    return out


def _decode_cases():
    out = []
    add = out.append
    for skv in (1, 2, 17, 128, 1000, 1024, 4096):
        for kind in ("none", "tl", "br"):
            if (skv, kind) == (1000, "br"):
                tier = "smoke"
            elif kind == "br" or skv in (1, 1024):
                tier = "full"
            else:
                tier = "exhaustive"
            add(
                _mk(
                    f"decode_skv{skv}_{kind}",
                    reqs=(13, 8)
                    + ((5,) if kind == "tl" else (6,) if kind == "br" else ()),
                    tier=tier,
                    b=4,
                    h_k=2,
                    s_q=1,
                    s_kv=skv,
                    emit_lse=kind == "br",
                    rdna=skv in (128, 1024) and kind == "br",
                    **_mask(kind),
                )
            )
    add(
        _mk(
            "decode_skv8192_gqa8",
            reqs=(13, 3, 6),
            h_q=16,
            h_k=2,
            s_q=1,
            s_kv=8192,
            **_mask("br"),
        )
    )
    for hq, hk in ((8, 8), (8, 4), (8, 1), (32, 8), (6, 2)):
        add(
            _mk(
                f"decode_gqa_h{hq}_{hk}",
                reqs=(13, 3, 6),
                tier="smoke" if (hq, hk) == (8, 1) else "full",
                b=4,
                h_q=hq,
                h_k=hk,
                s_q=1,
                s_kv=500,
                **_mask("br"),
            )
        )
    for dt in DTYPES:
        for d in (64, 128, 256):
            add(
                _mk(
                    f"decode_{dt}_d{d}",
                    reqs=(13, 9, 1, 6, 7),
                    tier="smoke" if (dt, d) == ("fp16", 256) else "full",
                    dtype=dt,
                    d=d,
                    h_k=2,
                    s_q=1,
                    s_kv=777,
                    emit_lse=True,
                    **_mask("br"),
                )
            )
    for w in (0, 1, 63, 64, 65):
        add(
            _mk(
                f"decode_window_w{w}",
                reqs=(13, 11, 6, 7),
                tier="smoke" if w == 64 else "full",
                h_k=2,
                s_q=1,
                s_kv=300,
                left_bound=w,
                right_bound=0,
                diagonal="bottom_right",
                emit_lse=True,
            )
        )
    add(
        _mk(
            "decode_padded_kv_lens",
            reqs=(13, 10, 6, 7),
            tier="smoke",
            b=4,
            h_k=2,
            q_lens=(1, 1, 1, 1),
            kv_lens=(300, 17, 1, 129),
            emit_lse=True,
            **_mask("br"),
        )
    )
    add(
        _mk(
            "decode_ragged_kv_lens",
            reqs=(13, 12, 6, 7),
            b=4,
            h_k=2,
            q_lens=(1, 1, 1, 1),
            kv_lens=(300, 17, 1, 129),
            length_mode="ragged",
            emit_lse=True,
            **_mask("br"),
        )
    )
    for layout in ("bshd", "strided_bshd", "strided_bhsd", "mixed"):
        add(
            _mk(
                f"decode_cache_layout_{layout}",
                reqs=(13, 2, 6),
                tier="smoke" if layout == "strided_bshd" else "full",
                layout=layout,
                b=3,
                h_k=2,
                s_q=1,
                s_kv=257,
                **_mask("br"),
            )
        )
    add(
        _mk(
            "decode_bias_key_row",
            reqs=(13, 14, 6),
            b=3,
            h_k=2,
            s_q=1,
            s_kv=200,
            bias=BiasSpec((3, 1, 1, 200)),
            **_mask("br"),
        )
    )
    add(
        _mk(
            "decode_bias_neginf_mask",
            reqs=(13, 14),
            b=2,
            h_k=2,
            s_q=1,
            s_kv=200,
            bias=BiasSpec((2, 8, 1, 200), kind="mask_neginf"),
        )
    )
    add(
        _mk(
            "decode_neginf_bias_hides_only_visible_key",
            reqs=(13, 14, 7, 11),
            h_q=4,
            s_q=1,
            s_kv=5,
            bias=BiasSpec((1, 1, 1, 5), kind="mask_neginf"),
            left_bound=0,
            right_bound=0,
            diagonal="bottom_right",
            emit_lse=True,
        )
    )
    return out


def _bias_cases():
    out = []
    add = out.append
    B, H, SQ, SK = 2, 8, 96, 160
    shapes = {
        "full": (B, H, SQ, SK),
        "h_bcast": (B, 1, SQ, SK),
        "b_bcast": (1, H, SQ, SK),
        "bh_bcast": (1, 1, SQ, SK),
        "key_row": (1, 1, 1, SK),
        "b_key_row": (B, 1, 1, SK),
        "h_key_row": (1, H, 1, SK),
        "bh_key_row": (B, H, 1, SK),
        "query_col": (1, 1, SQ, 1),
        "scalar": (1, 1, 1, 1),
        "rank3": (H, SQ, SK),
        "rank2": (SQ, SK),
        "rank1": (SK,),
    }
    smoke = {"full", "bh_bcast", "b_key_row"}
    optional = {"rank3", "rank1"}
    for tag, shp in shapes.items():
        add(
            _mk(
                f"bias_{tag}",
                reqs=(14,),
                tier=(
                    "smoke"
                    if tag in smoke
                    else ("exhaustive" if tag in optional else "full")
                ),
                tags=("optional",) if tag in optional else (),
                b=B,
                s_q=SQ,
                s_kv=SK,
                bias=BiasSpec(shp),
                emit_lse=tag in ("full", "bh_key_row"),
                rdna=tag in ("full", "bh_bcast"),
            )
        )
    for lay in ("expanded", "row_padded"):
        add(
            _mk(
                f"bias_layout_{lay}",
                reqs=(14, 2),
                tier="smoke" if lay == "expanded" else "full",
                b=B,
                s_q=SQ,
                s_kv=SK,
                bias=BiasSpec(
                    (1, 1, SQ, SK) if lay == "expanded" else (B, 1, SQ, SK), layout=lay
                ),
            )
        )
    add(
        _mk(
            "bias_row_padded_odd_key_len",
            reqs=(14, 8),
            b=B,
            s_q=33,
            s_kv=77,
            bias=BiasSpec((1, H, 33, 77), layout="row_padded"),
        )
    )
    add(
        _mk(
            "bias_dtype_same_as_query_bf16",
            reqs=(14, 1),
            tier="exhaustive",
            tags=("optional",),
            dtype="bf16",
            b=B,
            s_q=SQ,
            s_kv=SK,
            bias=BiasSpec((1, 1, SQ, SK), dtype="same"),
        )
    )
    add(
        _mk(
            "bias_after_scale_small_scale",
            reqs=(14, 4),
            tier="smoke",
            b=B,
            s_q=SQ,
            s_kv=SK,
            scale=0.05,
            bias=BiasSpec((B, H, SQ, SK)),
        )
    )
    for kind, tier in (("mask_neginf", "smoke"), ("neginf_rows", "full")):
        add(
            _mk(
                f"bias_{kind}",
                reqs=(14, 7),
                tier=tier,
                b=B,
                s_q=SQ,
                s_kv=SK,
                bias=BiasSpec((1, 1, SQ, SK), kind=kind),
                emit_lse=True,
            )
        )
    add(
        _mk(
            "bias_neginf_rows_br",
            reqs=(14, 6, 7),
            b=B,
            s_q=64,
            s_kv=160,
            bias=BiasSpec((B, 1, 64, 160), kind="neginf_rows"),
            emit_lse=True,
            **_mask("br"),
        )
    )
    for kind in ("tl", "br"):
        add(
            _mk(
                f"bias_causal_{kind}_q_gt_kv",
                reqs=(14, 5 if kind == "tl" else 6),
                tier="smoke" if kind == "br" else "full",
                b=B,
                s_q=160,
                s_kv=96,
                bias=BiasSpec((B, 1, 160, 96)),
                emit_lse=True,
                **_mask(kind),
            )
        )
    for d in (64, 256):
        add(
            _mk(
                f"bias_d{d}_gqa_br",
                reqs=(14, 3, 9, 6),
                d=d,
                h_k=2,
                s_q=64,
                s_kv=192,
                bias=BiasSpec((2, 1, 64, 192)),
                **_mask("br"),
            )
        )
    add(
        _mk(
            "bias_window_tl",
            reqs=(14, 11),
            s_q=160,
            left_bound=20,
            right_bound=0,
            bias=BiasSpec((1, 1, 160, 160)),
        )
    )
    add(
        _mk(
            "bias_bf16_strided_bshd",
            reqs=(14, 2, 5),
            dtype="bf16",
            layout="strided_bshd",
            bias=BiasSpec((1, 1, 1, 128)),
            **_mask("tl"),
        )
    )
    add(
        _mk(
            "bias_ragged_gqa_br",
            reqs=(14, 12, 3, 6),
            b=3,
            h_k=2,
            d=64,
            q_lens=(17, 33, 5),
            kv_lens=(70, 77, 41),
            length_mode="ragged",
            bias=BiasSpec((3, 1, 33, 77)),
            **_mask("br"),
        )
    )
    return out


# ---------------------------------------------------------------------------
# cross-product: GQA ratio x length mode x mask x bias x q/kv relation
# ---------------------------------------------------------------------------

_X_RATIOS = {"mha": (8, 8), "gqa4": (8, 2), "mqa": (8, 1)}
_X_LENGTHS = ("fixed", "padded", "ragged")
_X_MASKS = ("none", "tl", "br", "tl_win", "br_win")
_X_BIAS = ("none", "key_row", "full")
_X_REL = ("lt", "eq", "gt")
_X_DTYPE_D = (("fp16", 64), ("bf16", 64), ("fp16", 128), ("bf16", 128))
_X_LAYOUTS = ("bhsd", "bshd", "strided_bshd")
_X_FIXED = {"lt": (33, 77), "eq": (45, 45), "gt": (77, 33)}
_X_LENS = {
    "lt": ((17, 33, 5), (70, 77, 41)),
    "eq": ((45, 19, 31), (45, 19, 31)),
    "gt": ((77, 50, 9), (33, 40, 21)),
}
_X_HAND_SMOKE = {
    ("gqa4", "ragged", "br", "full", "lt"),
    ("gqa4", "padded", "br_win", "full", "gt"),
    ("gqa4", "ragged", "br", "key_row", "gt"),
}


def _cross_case(idx, combo, tier):
    ratio, lens, mask, bias, rel = combo
    dt, d = _X_DTYPE_D[idx % 4]
    hq, hk = _X_RATIOS[ratio]
    kw = {}
    reqs = {3, {"fixed": 8, "padded": 10, "ragged": 12}[lens]}
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
        sq, sk = _X_FIXED[rel]
        shape = {"s_q": sq, "s_kv": sk, "layout": _X_LAYOUTS[idx % 3]}
    else:
        ql, kl = _X_LENS[rel]
        sq, sk = max(ql), max(kl)
        shape = {"q_lens": ql, "kv_lens": kl, "length_mode": lens}
        if lens == "padded":
            shape["layout"] = _X_LAYOUTS[idx % 3]
    spec = None
    if bias == "key_row":
        spec = BiasSpec((3, 1, 1, sk))
    elif bias == "full":
        spec = BiasSpec((3, hq, sq, sk))
    if spec is not None:
        reqs.add(14)
    return _mk(
        f"x_{ratio}_{lens}_{mask}_{bias}_{rel}",
        reqs=reqs,
        tier=tier,
        tags=("cross",),
        dtype=dt,
        d=d,
        b=3,
        h_q=hq,
        h_k=hk,
        bias=spec,
        emit_lse=idx % 2 == 0,
        **shape,
        **kw,
    )


def _pairwise_cover(rows):
    """Deterministic greedy all-pairs cover; returns chosen row indices."""
    n = len(rows[0])
    pairs = list(itertools.combinations(range(n), 2))
    remaining = {(i, j, r[i], r[j]) for r in rows for i, j in pairs}
    chosen = []
    while remaining:
        best, best_gain = -1, -1
        for idx, r in enumerate(rows):
            gain = sum((i, j, r[i], r[j]) in remaining for i, j in pairs)
            if gain > best_gain:
                best, best_gain = idx, gain
        chosen.append(best)
        r = rows[best]
        for i, j in pairs:
            remaining.discard((i, j, r[i], r[j]))
    return chosen


def _cross_cases():
    combos = list(itertools.product(_X_RATIOS, _X_LENGTHS, _X_MASKS, _X_BIAS, _X_REL))
    rows = [c + (_X_DTYPE_D[i % 4], _X_LAYOUTS[i % 3]) for i, c in enumerate(combos)]
    picked = set(_pairwise_cover(rows))
    out = []
    for i, c in enumerate(combos):
        tier = (
            "smoke" if c in _X_HAND_SMOKE else ("full" if i in picked else "exhaustive")
        )
        out.append(_cross_case(i, c, tier))
    return out


def _intersection_cases():
    """Hand-picked intersections beyond the cross grid."""
    out = []
    add = out.append
    add(
        _mk(
            "x_gqa_varlen_br_bias_d256",
            reqs=(3, 12, 6, 14, 9),
            tags=("cross",),
            d=256,
            h_k=2,
            q_lens=(20, 50),
            kv_lens=(60, 90),
            length_mode="ragged",
            bias=BiasSpec((2, 1, 50, 90)),
            emit_lse=True,
            **_mask("br"),
        )
    )
    add(
        _mk(
            "x_gqa_padded_window_bias_lse_bf16",
            reqs=(3, 10, 11, 14, 7, 6),
            tags=("cross",),
            dtype="bf16",
            b=3,
            h_k=2,
            q_lens=(50, 30, 12),
            kv_lens=(90, 60, 70),
            left_bound=15,
            right_bound=0,
            diagonal="bottom_right",
            bias=BiasSpec((1, 8, 50, 90)),
            emit_lse=True,
        )
    )
    add(
        _mk(
            "x_packed_qkv_gqa_padded_br_lse",
            reqs=(2, 3, 10, 6, 7),
            tags=("cross",),
            layout="packed_qkv",
            b=3,
            h_k=2,
            q_lens=(45, 19, 31),
            kv_lens=(45, 19, 31),
            d=64,
            emit_lse=True,
            **_mask("br"),
        )
    )
    add(
        _mk(
            "x_decode_gqa_window_bias_ragged",
            reqs=(13, 3, 11, 14, 12, 6, 7),
            tier="smoke",
            tags=("cross",),
            b=3,
            h_k=2,
            q_lens=(1, 1, 1),
            kv_lens=(300, 70, 5),
            length_mode="ragged",
            d=64,
            left_bound=31,
            right_bound=0,
            diagonal="bottom_right",
            bias=BiasSpec((3, 1, 1, 300)),
            emit_lse=True,
        )
    )
    add(
        _mk(
            "x_window_zero_fully_masked_bias_gqa",
            reqs=(11, 7, 14, 3, 6),
            tags=("cross",),
            h_k=2,
            s_q=96,
            s_kv=40,
            left_bound=0,
            right_bound=0,
            diagonal="bottom_right",
            bias=BiasSpec((1, 1, 96, 40), kind="mask_neginf"),
            emit_lse=True,
        )
    )
    add(
        _mk(
            "x_odd_lengths_strided_bhsd_padded_bias_br",
            reqs=(8, 2, 14, 6),
            tags=("cross",),
            layout="strided_bhsd",
            d=64,
            s_q=61,
            s_kv=131,
            bias=BiasSpec((2, 8, 61, 131), layout="row_padded"),
            **_mask("br"),
        )
    )
    return out


# ---------------------------------------------------------------------------
# table access
# ---------------------------------------------------------------------------

_BUILDERS = (
    _dtype_headdim_cases,
    _layout_cases,
    _gqa_scale_cases,
    _mask_lse_cases,
    _seqlen_cases,
    _window_cases,
    _padding_cases,
    _ragged_cases,
    _decode_cases,
    _bias_cases,
    _intersection_cases,
    _cross_cases,
)
_CACHE = None


def all_cases():
    """The deterministic, ordered table."""
    global _CACHE
    if _CACHE is None:
        _CACHE = tuple(c for fn in _BUILDERS for c in fn())
    return _CACHE


def select(tag="full", *, arch=None, reqs=None, max_cost=None):
    """Cases carrying ``tag`` (a tier or group tag), optionally filtered."""
    out = []
    for c in all_cases():
        if tag is not None and tag not in c.tags:
            continue
        if arch is not None and arch not in c.archs:
            continue
        if reqs is not None and not set(reqs) <= set(c.reqs):
            continue
        if max_cost is not None and c.cost > max_cost:
            continue
        out.append(c)
    return out


def get_case(case_id):
    for c in all_cases():
        if c.id == case_id:
            return c
    raise KeyError(case_id)


def reference_for(case, seed=None):
    """Materialise ``case`` (a case or id) and evaluate the oracle."""
    from .helpers import evaluate, materialize

    case = get_case(case) if isinstance(case, str) else case
    return evaluate(materialize(case, seed))


def table_summary():
    """Counts per tier, requirement item and arch."""
    cases = all_cases()
    out = {"total": len(cases)}
    for t in TIERS:
        out[t] = sum(t in c.tags for c in cases)
    return out
