# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Backward kernel test selection: feature stages, requests, configurations.

Shared by the general-family backward kernel tests (plan, IR, numerics,
mutations). Nothing here touches a device.

* :func:`stage_of` ranks a case by the latest functional feature group it
  uses (1 base path, 2 masks and stats, 3 grouped heads / packed QKV,
  4 ``h_k != h_v``, 5 padding, 6 ragged THD); a kernel test at stage ``k``
  runs every case with ``stage_of(case) <= k``.
* :func:`request_from_case` is the capability-check view of a case.
* :data:`SUPPLEMENTARY_CASES` adds cases the table does not hold: the
  unmasked base path on every dense layout (the table's layout cases are
  masked), head dim 32 unmasked and under a band, the MHA ``s_q = 1``
  cell (unmasked and bottom-right), and grouped heads the table leaves out
  (an unmasked GQA group, two-sided windows on GQA and MQA, head dim 32 GQA,
  a packed QKV GQA view under a window), and ``h_k != h_v`` (atomic dK / dV)
  cells: the batched parts of the ``h_q = 8, h_k = 4, h_v = 2`` matrix
  (top-left and bottom-right, fp16 and bf16; the table's two ``h_k != h_v``
  cases are exhaustive-tier only), K heads fewer than V heads, and a unit of
  one query head (``gcd(h_q / h_k, h_q / h_v) = 1``) under a window; and
  padded cells (per-batch SEQ_LEN counts) the table leaves out: atomic dK /
  dV, head dim 32 under a window, and a packed GQA group; and ragged (THD)
  cells: the THD parts of the ``h_q = 8, h_k = 4, h_v = 2`` matrix, an
  unmasked GQA group and head dim 32 under a window.
* :data:`ZERO_LENGTH_CASES`: padded batches with ``len_q = 0`` and / or
  ``len_kv = 0``; :data:`THD_ZERO_LENGTH_CASES`: THD sequences of length 0
  (the last q and kv sequences, middle ones, all but one). Both are outside
  the table, whose validator requires positive lengths.
* :data:`THD_HK_NE_HV_CASES`: the THD ``h_k != h_v`` supplementary ids.
* :data:`WORKLIST_SKEW_CASES` / :data:`WORKLIST_CASES`: skewed THD batches
  (outside the table) and the ids the work-list test runs.
* :data:`THD_STORAGE_FORMS` / :data:`THD_STORAGE_CASES`: non-compact THD
  storages (tokens before the first sequence, gaps with SEQ_LEN counts
  shorter than the allocated segments, tails, padded tokens, token-unit
  offsets) run with and without a caller ``max_total``;
  :func:`thd_storage_fuzz` draws further seeded storages of the same kinds
  (:func:`thd_fuzz_count` per tier).
* :data:`HK_NE_HV_TABLE_CASES`: the table's ``h_k != h_v`` case ids.
* :data:`HEAD_PACK_CASES`: short query spans of GQA groups for the
  head-packing test (``s_q`` in 1..3, group ratios 2..8; batched and THD).
* :data:`NARROW_CASES`: the narrow-access cells (``stage_vec = 1``): a case,
  the view it is re-laid out on (BHSD or BSHD order with odd S strides and H
  strides that are not multiples of 8, or a THD storage with an odd token
  stride) and the byte offset of every tensor base from a 16-byte boundary
  (2, 4 or 8); :data:`NARROW_TWIN_CASES`: case ids whose ``stage_vec = 8``
  run and ``stage_vec = 1`` twin must agree bit for bit. No table case needs
  narrow access, so these cells are the stage 7 cases.
* :func:`selected_cases` applies the ``ROCKE_BWD_TIER`` / ``ROCKE_BWD_STAGE``
  / ``ROCKE_BWD_SHARD`` environment knobs; :func:`selected_configs` applies
  ``ROCKE_BWD_CONFIG``; the ``custom`` configuration takes its knobs per
  (head size, ``stage_vec``) from ``ROCKE_BWD_CUSTOM_KNOBS``
  (:func:`custom_rules`), so a swept configuration runs the same suite.
"""

from __future__ import annotations

import json
import math
import os
import zlib
from dataclasses import replace

from kernels.common.attention_bwd import AttnBwdSpec
from kernels.common.attention_bwd_plan import (
    AUX_KNOBS,
    DQ_KNOBS,
    RUNTIME_KNOBS,
    AttnBwdPolicy,
    AttnBwdRequest,
    attn_bwd_stage_vec,
)

from .bwd_cases import _mkb, layout_tensors_bwd, select_bwd, validate_bwd_case
from .bwd_tile_region_sets import region_knobs
from .cases import _mask
from .helpers import has_fully_masked_rows

__all__ = [
    "BUILT_STAGE",
    "CONFIGS",
    "CUSTOM_CONFIG",
    "CUSTOM_KNOBS_ENV",
    "HEAD_PACK_CASES",
    "HK_NE_HV_TABLE_CASES",
    "KNOB_SMOKE_CASES",
    "KNOB_SMOKE_REFUSED",
    "COMPOSE_REFUSED",
    "KNOB_SMOKE_SETS",
    "COMPOSE_TWIN_SETS",
    "LAYOUT_TWIN_REFUSED",
    "LAYOUT_TWIN_SETS",
    "NARROW_CASES",
    "NARROW_TWIN_CASES",
    "ROW_XOR_SWIZZLE",
    "SUPPLEMENTARY_CASES",
    "THD_HK_NE_HV_CASES",
    "THD_STORAGE_CASES",
    "THD_STORAGE_FORMS",
    "THD_ZERO_LENGTH_CASES",
    "WORKLIST_CASES",
    "WORKLIST_SKEW_CASES",
    "ZERO_LENGTH_CASES",
    "config_knobs",
    "config_policy",
    "custom_knobs_for",
    "custom_rules",
    "knob_smoke_params",
    "knob_smoke_rules_json",
    "compose_twin_params",
    "layout_twin_params",
    "perf_prune_cases",
    "request_from_case",
    "selected_cases",
    "selected_configs",
    "shard_of",
    "stage_of",
    "resolve_storage_form",
    "thd_fuzz_count",
    "thd_storage_fuzz",
]

# Highest functional stage the general main kernel builds today (8: the
# forward-to-backward chain and the negative controls; stages 7 and 8 add no
# table case, see NARROW_CASES and test_attention_bwd_fwd_chain.py).
BUILT_STAGE = 8

CONFIGS = ("debug", "default", "multi")
# A caller-given configuration (a swept candidate): spec and runtime knobs per
# (head size, stage_vec) from the JSON in CUSTOM_KNOBS_ENV. Selected only by
# name (``ROCKE_BWD_CONFIG=custom``), never by ``both``.
CUSTOM_CONFIG = "custom"
CUSTOM_KNOBS_ENV = "ROCKE_BWD_CUSTOM_KNOBS"
# Spec fields that come from the request, not from a configuration.
_REQUEST_SPEC_FIELDS = (
    "head_size",
    "dtype",
    "seq_mode",
    "dkv_mode",
    "stage_vec",
    "mask_class",
    "name",
)
_CUSTOM_MATCH = ("head_size", "stage_vec")
_CDNA = ("gfx942", "gfx950")
_RDNA = ("gfx1151", "gfx1201")

# One-wave 16 x 16 correctness configuration (the RDNA default geometry).
_DEBUG_CDNA = dict(
    waves=1,
    block_m=16,
    block_n=16,
    block_k4=16,
    warp_grid_g4=(1, 1),
    atom_g02="16x16x16",
    atom_g13="16x16x16",
    atom_g4="16x16x16",
    kv_residency="lds",
    kt_source="lds",
    transpose_source="lds_plain",
    pt_route="relabel",
    sched="none",
)
# Two-wave WMMA tile (RDNA): LDS accumulators, LDS K / V, LDS transposes.
_MULTI_RDNA = dict(waves=2, block_m=16, block_n=32, block_k4=32, warp_grid_g4=(1, 2))


def stage_of(case) -> int:
    """1-based rank of the latest functional feature group ``case`` uses."""
    rank = 1
    if (
        case.left_bound != -1
        or case.right_bound != -1
        or case.causal_bool is not None
        or case.scale is not None
        or case.d == 32
        or case.s_q == 1
        or case.fully_masked_rows
    ):
        rank = max(rank, 2)
    if (case.h_k == case.h_v and case.h_k != case.h_q) or case.layout == "packed_qkv":
        rank = max(rank, 3)
    if case.h_k != case.h_v:
        rank = max(rank, 4)
    if case.length_mode == "padded":
        rank = max(rank, 5)
    if case.length_mode == "ragged" or case.layout == "thd":
        rank = max(rank, 6)
    return rank


def request_from_case(case) -> AttnBwdRequest:
    """The capability-check view of a backward table case."""
    descs, _ = layout_tensors_bwd(case)
    strides, align = {}, 256
    for name, desc in descs.items():
        if case.layout == "thd":
            t, h, d = desc.strides
            strides[name] = (0, h, t, d)
        else:
            strides[name] = tuple(desc.strides)
        off = desc.offset * 2
        if off:
            align = min(align, off & -off)
    padded = case.length_mode == "padded"
    return AttnBwdRequest(
        b=case.b,
        h_q=case.h_q,
        h_k=case.h_k,
        h_v=case.h_v,
        s_q=case.s_q,
        s_kv=case.s_kv,
        d_qk=case.d,
        d_v=case.d,
        dtype=case.dtype,
        strides=strides,
        layout="thd" if case.layout == "thd" else "dense",
        left_bound=case.left_bound,
        right_bound=case.right_bound,
        bottom_right=case.diagonal == "bottom_right",
        causal=case.causal_bool == "top_left",
        causal_bottom_right=case.causal_bool == "bottom_right",
        padding=padded,
        has_seq_len_q=padded,
        has_seq_len_kv=padded,
        tensor_alignment=align,
        scale=case.scale,
    )


def _supplementary():
    out = []
    for layout in ("bshd", "strided_bhsd", "strided_bshd", "mixed"):
        for dt, d in (("fp16", 64), ("bf16", 128)):
            out.append(
                _mkb(
                    f"kern_base_{layout}_{dt}_d{d}",
                    reqs=(1, 2),
                    tier="smoke",
                    dtype=dt,
                    d=d,
                    b=2,
                    h_q=4,
                    s_q=80,
                    s_kv=136,
                    layout=layout,
                    rdna=True,
                )
            )
    for dt in ("fp16", "bf16"):
        out.append(
            _mkb(
                f"kern_base_bhsd_{dt}_d32",
                reqs=(1, 9),
                tier="smoke",
                tags=("optional",),
                dtype=dt,
                d=32,
                b=2,
                h_q=4,
                s_q=72,
                s_kv=100,
                rdna=True,
            )
        )
    # Masks and stats: the MHA s_q = 1 cell (unmasked and bottom-right; full
    # tier, as decode shapes stay out of smoke) and head dim 32 under a band
    # (the table's d32 cells are unmasked).
    for name, dt, d, sq, skv, mask in (
        ("sq1_mha_none", "fp16", 64, 1, 100, {}),
        ("sq1_mha_br", "bf16", 128, 1, 72, _mask("br")),
        ("d32_br", "fp16", 32, 72, 100, _mask("br")),
        ("d32_window_tl", "bf16", 32, 100, 72, dict(_mask("tl"), left_bound=15)),
    ):
        out.append(
            _mkb(
                f"kern_{name}_{dt}_d{d}",
                reqs=(1, 9),
                tier="full" if sq == 1 else "smoke",
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=2,
                h_q=4,
                s_q=sq,
                s_kv=skv,
                rdna=True,
                **mask,
            )
        )
    # Grouped heads (direct mode, on-chip group sum): the table's GQA / MQA
    # cells are all one-sided causal bands; these add an unmasked group, two-
    # sided windows (right > 0, both diagonals), head dim 32 and a packed QKV
    # view under a window.
    for name, dt, d, h_q, h_k, sq, skv, layout, mask, tier in (
        ("gqa4_none", "bf16", 64, 8, 2, 80, 136, None, {}, "smoke"),
        (
            "gqa2_win2_tl", "fp16", 64, 8, 4, 100, 72, None,
            dict(_mask("tl"), left_bound=20, right_bound=9), "smoke",
        ),
        (
            "mqa_win2_br", "bf16", 128, 8, 1, 64, 150, None,
            dict(_mask("br"), left_bound=7, right_bound=3), "full",
        ),
        ("gqa3_d32_br", "fp16", 32, 6, 2, 72, 100, None, _mask("br"), "full"),
        (
            "packed_qkv_gqa4_win", "fp16", 64, 8, 2, 96, 96, "packed_qkv",
            dict(_mask("tl"), left_bound=15, right_bound=4), "full",
        ),
    ):  # fmt: skip
        out.append(
            _mkb(
                f"kern_{name}_{dt}_d{d}",
                reqs=(3, 9),
                tier=tier,
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=2,
                h_q=h_q,
                h_k=h_k,
                s_q=sq,
                s_kv=skv,
                layout=layout,
                rdna=True,
                **mask,
            )
        )
    # h_k != h_v (atomic dK / dV through the workspace and two converts): the
    # batched h_q = 8, h_k = 4, h_v = 2 matrix (TL / BR x fp16 / bf16), K heads
    # fewer than V heads, and a unit of one query head (h_q = 6, h_k = 3,
    # h_v = 2: gk = 2, gv = 3, G = 1) under a two-sided window.
    for name, dt, d, h_q, h_k, h_v, sq, skv, mask, tier in (
        ("hkv_842_tl", "fp16", 64, 8, 4, 2, 80, 136, _mask("tl"), "smoke"),
        ("hkv_842_br", "bf16", 128, 8, 4, 2, 72, 100, _mask("br"), "smoke"),
        ("hkv_842_br", "fp16", 64, 8, 4, 2, 100, 72, _mask("br"), "full"),
        ("hkv_842_tl", "bf16", 128, 8, 4, 2, 96, 96, _mask("tl"), "full"),
        ("hkv_824_none", "bf16", 64, 8, 2, 4, 80, 100, {}, "full"),
        (
            "hkv_632_win2_br", "fp16", 32, 6, 3, 2, 72, 100,
            dict(_mask("br"), left_bound=9, right_bound=4), "full",
        ),
    ):  # fmt: skip
        out.append(
            _mkb(
                f"kern_{name}_{dt}_d{d}",
                reqs=(3, 9),
                tier=tier,
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=2,
                h_q=h_q,
                h_k=h_k,
                h_v=h_v,
                s_q=sq,
                s_kv=skv,
                rdna=True,
                **mask,
            )
        )
    # Padding (per-batch SEQ_LEN counts): the table's padded cells are all
    # direct mode with head dim 64 / 128 and query spans the plan never packs;
    # these add atomic dK / dV (h_k != h_v), head dim 32 under a window, and a
    # GQA group whose short query spans the plan packs onto one q tile.
    for name, dt, d, h_q, h_k, h_v, ql, kl, mask, tier in (
        (
            "pad_hkv_842_br", "fp16", 64, 8, 4, 2, (80, 33, 7), (17, 136, 100),
            _mask("br"), "smoke",
        ),
        (
            "pad_hkv_842_tl", "bf16", 128, 8, 4, 2, (72, 100, 5), (100, 40, 72),
            _mask("tl"), "full",
        ),
        (
            "pad_d32_win_br", "fp16", 32, 4, 4, 4, (72, 9, 50), (100, 33, 64),
            dict(_mask("br"), left_bound=11), "full",
        ),
        (
            "pad_sq2_gqa4_br", "bf16", 64, 8, 2, 2, (2, 1, 2), (33, 100, 1),
            _mask("br"), "full",
        ),
    ):  # fmt: skip
        out.append(
            _mkb(
                f"kern_{name}_{dt}_d{d}",
                reqs=(3, 10),
                tier=tier,
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=len(ql),
                h_q=h_q,
                h_k=h_k,
                h_v=h_v,
                q_lens=ql,
                kv_lens=kl,
                rdna=True,
                **mask,
            )
        )
    # Ragged THD sequences (offset tables): the table's ragged cells hold no
    # h_k != h_v cell and no head dim 32 cell; these add the THD parts of the
    # h_q = 8, h_k = 4, h_v = 2 matrix (top-left and bottom-right, fp16 and
    # bf16; atomic dK / dV), an unmasked GQA group and head dim 32 under a
    # window.
    for name, dt, d, h_q, h_k, h_v, ql, kl, mask, tier in (
        (
            "thd_hkv_842_tl", "fp16", 64, 8, 4, 2, (40, 72, 9), (100, 33, 64),
            _mask("tl"), "smoke",
        ),
        (
            "thd_hkv_842_br", "bf16", 128, 8, 4, 2, (72, 17, 50), (33, 100, 72),
            _mask("br"), "full",
        ),
        (
            "thd_hkv_842_br", "fp16", 64, 8, 4, 2, (9, 100, 33), (64, 40, 100),
            _mask("br"), "full",
        ),
        (
            "thd_hkv_842_tl", "bf16", 128, 8, 4, 2, (50, 33, 80), (50, 72, 17),
            _mask("tl"), "full",
        ),
        (
            "thd_gqa4_none", "bf16", 64, 8, 2, 2, (80, 1, 45), (136, 17, 45),
            {}, "full",
        ),
        (
            "thd_d32_win_br", "fp16", 32, 4, 4, 4, (72, 9, 50), (100, 33, 64),
            dict(_mask("br"), left_bound=11), "full",
        ),
    ):  # fmt: skip
        out.append(
            _mkb(
                f"kern_{name}_{dt}_d{d}",
                reqs=(3, 9),
                tier=tier,
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=len(ql),
                h_q=h_q,
                h_k=h_k,
                h_v=h_v,
                q_lens=ql,
                kv_lens=kl,
                length_mode="ragged",
                rdna=True,
                **mask,
            )
        )
    _check_cases(out)
    return tuple(out)


# The case table's h_k != h_v cells (exhaustive tier; test_hk_ne_hv runs them by id).
HK_NE_HV_TABLE_CASES = (
    "bwd_gqa_kv_heads_differ_causal_tl",
    "bwd_gqa_k_fewer_than_v_causal_tl",
)


def _check_cases(cases) -> None:
    for c in cases:
        problems = validate_bwd_case(c)
        if problems:
            raise AssertionError(f"{c.id}: {problems}")


SUPPLEMENTARY_CASES = _supplementary()

# The THD h_k != h_v supplementary cells (atomic dK / dV over ragged sequences).
THD_HK_NE_HV_CASES = tuple(
    c.id for c in SUPPLEMENTARY_CASES if c.length_mode == "ragged" and c.h_k != c.h_v
)


def _head_pack_cases():
    """Short query spans of GQA groups (``s_q`` in 1..3, ratios 2..8).

    The plan packs the group's query heads onto the M rows of one q tile when
    ``G * s_q <= kM0``; larger products fall back to the unpacked loop on the
    smaller tiles, so both paths run. The THD cells pack ragged sequences of
    at most three query tokens (a GQA group, an ``h_k != h_v`` unit with
    atomic dK / dV, and an MQA decode group).
    """
    out = []
    for sq, h_q, h_k, kind, dt, d, skv in (
        (1, 8, 4, "none", "fp16", 64, 100),
        (1, 8, 1, "br", "bf16", 128, 72),
        (1, 6, 2, "tl", "fp16", 32, 33),
        (2, 6, 2, "br", "fp16", 64, 17),
        (2, 8, 2, "none", "bf16", 64, 100),
        (3, 8, 4, "win", "bf16", 64, 72),
        (3, 8, 2, "br", "fp16", 128, 33),
        (3, 8, 1, "br", "fp16", 64, 100),
    ):
        mask = dict(_mask("br"), left_bound=7) if kind == "win" else _mask(kind)
        out.append(
            _mkb(
                f"kern_pack_sq{sq}_g{h_q // h_k}_{kind}_{dt}_d{d}",
                reqs=(3, 13),
                tier="full",
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=2,
                h_q=h_q,
                h_k=h_k,
                s_q=sq,
                s_kv=skv,
                rdna=True,
                **mask,
            )
        )
    for name, dt, d, h_q, h_k, h_v, ql, kl, kind in (
        ("g4", "fp16", 64, 8, 2, 2, (2, 1, 2), (33, 100, 17), "br"),
        ("hkv_842", "bf16", 64, 8, 4, 2, (3, 1, 3), (72, 5, 40), "br"),
        ("g8", "bf16", 128, 8, 1, 1, (1, 1, 1, 1), (100, 1, 17, 72), "tl"),
    ):
        out.append(
            _mkb(
                f"kern_pack_thd_sq{max(ql)}_{name}_{kind}_{dt}_d{d}",
                reqs=(3, 13),
                tier="full",
                dtype=dt,
                d=d,
                b=len(ql),
                h_q=h_q,
                h_k=h_k,
                h_v=h_v,
                q_lens=ql,
                kv_lens=kl,
                length_mode="ragged",
                rdna=True,
                **_mask(kind),
            )
        )
    _check_cases(out)
    return tuple(out)


HEAD_PACK_CASES = _head_pack_cases()


def _zero_length(cid, *, q_lens, kv_lens, **kw):
    """A padded (or ragged) case with zero-length batches (sequences).

    The case table requires positive lengths, so the case is validated with
    every zero length raised to 1, then the zero lengths are put back and the
    derived flags (decode, fully masked rows) recomputed.
    """
    template = _mkb(
        cid,
        q_lens=tuple(max(n, 1) for n in q_lens),
        kv_lens=tuple(max(n, 1) for n in kv_lens),
        **kw,
    )
    _check_cases([template])
    if max(q_lens) != template.s_q or max(kv_lens) != template.s_kv:
        raise AssertionError(f"{cid}: the longest batch must not be zero-length")
    case = replace(template, q_lens=tuple(q_lens), kv_lens=tuple(kv_lens))
    decode = all(n == 1 for n in q_lens)
    tags = case.tags if decode else case.tags - {"decode_bwd"}
    return replace(
        case,
        decode=decode,
        fully_masked_rows=has_fully_masked_rows(case),
        tags=tags,
    )


def _zero_length_cases():
    """Padded batches with ``len_q = 0`` and / or ``len_kv = 0``.

    Zero lengths in the middle and in the last batch, a batch with no keys
    but live query rows (fully masked: dQ exactly 0), both lengths zero, every
    batch but one empty, under no mask, top-left and bottom-right diagonals
    and a window; MHA, a GQA group (direct), ``h_k != h_v`` (atomic dK / dV)
    and a packed GQA decode group.
    """
    out = []
    for name, dt, d, h_q, h_k, h_v, ql, kl, mask in (
        ("mha_br", "fp16", 64, 4, 4, 4, (40, 0, 33, 0), (72, 50, 0, 90), _mask("br")),
        ("gqa4_tl", "bf16", 128, 8, 2, 2, (0, 45, 17), (45, 0, 30), _mask("tl")),
        ("mha_none_one_live", "fp16", 32, 4, 4, 4, (0, 72, 0), (0, 100, 0), {}),
        (
            "hkv_842_win_br", "fp16", 64, 8, 4, 2, (33, 0, 50), (0, 80, 41),
            dict(_mask("br"), left_bound=9),
        ),
        ("sq1_gqa4_br", "bf16", 64, 8, 2, 2, (1, 0, 1), (100, 33, 0), _mask("br")),
    ):  # fmt: skip
        out.append(
            _zero_length(
                f"kern_zero_len_{name}_{dt}_d{d}",
                reqs=(3, 10),
                tier="full",
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=len(ql),
                h_q=h_q,
                h_k=h_k,
                h_v=h_v,
                q_lens=ql,
                kv_lens=kl,
                rdna=True,
                **mask,
            )
        )
    return tuple(out)


# Padded zero-length batches (outside the case table, whose validator
# requires positive lengths).
ZERO_LENGTH_CASES = _zero_length_cases()


def _thd_zero_length_cases():
    """THD sequences of length 0.

    The last q and the last kv sequence empty (no token after the last
    offset), empty middle sequences, a sequence with no keys but live query
    tokens (fully masked: dQ exactly 0), every sequence but one empty; MHA,
    a GQA group (direct), ``h_k != h_v`` (atomic dK / dV) and a packed GQA
    decode group.
    """
    out = []
    for name, dt, d, h_q, h_k, h_v, ql, kl, mask in (
        (
            "last_mha_br", "fp16", 64, 4, 4, 4, (40, 33, 72, 0), (72, 50, 90, 0),
            _mask("br"),
        ),
        (
            "mid_gqa4_tl", "bf16", 128, 8, 2, 2, (45, 0, 17), (45, 30, 0),
            _mask("tl"),
        ),
        ("one_live_none", "fp16", 32, 4, 4, 4, (0, 72, 0), (0, 100, 0), {}),
        (
            "hkv_842_win_br", "fp16", 64, 8, 4, 2, (33, 50, 0), (0, 80, 41),
            dict(_mask("br"), left_bound=9),
        ),
        (
            "sq1_gqa4_br", "bf16", 64, 8, 2, 2, (1, 1, 0), (100, 0, 33),
            _mask("br"),
        ),
    ):  # fmt: skip
        out.append(
            _zero_length(
                f"kern_thd_zero_len_{name}_{dt}_d{d}",
                reqs=(3, 8),
                tier="full",
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=len(ql),
                h_q=h_q,
                h_k=h_k,
                h_v=h_v,
                q_lens=ql,
                kv_lens=kl,
                length_mode="ragged",
                rdna=True,
                **mask,
            )
        )
    return tuple(out)


THD_ZERO_LENGTH_CASES = _thd_zero_length_cases()


def _worklist_skew_cases():
    """Skewed THD batches for the work list (outside the case table).

    A GQA group with two equally long query spans (the index breaks the tie)
    among short sequences under a bottom-right band; an ``h_k != h_v`` unit
    (atomic dK / dV) with the longest kv span not on the longest query span,
    top-left; and 150 short sequences around two long ones, unmasked, so the
    prep role's sort runs several passes of its CTA (more sequences than
    threads) with ties across passes.
    """
    lq150 = [1 + (37 * i) % 19 for i in range(150)]
    lk150 = [1 + (53 * i) % 23 for i in range(150)]
    lq150[37], lk150[37] = 260, 300
    lq150[140], lk150[140] = 300, 41
    out = []
    for name, dt, d, h_q, h_k, h_v, ql, kl, mask in (
        (
            "gqa4_br", "fp16", 64, 8, 2, 2, (3, 5, 1, 7, 300, 2, 9, 4, 300),
            (17, 40, 1, 9, 600, 3, 30, 5, 280), _mask("br"),
        ),
        (
            "hkv_842_tl", "bf16", 128, 8, 4, 2, (1, 2, 400, 3, 5, 200),
            (9, 3, 260, 1, 70, 400), _mask("tl"),
        ),
        ("mha_none_b150", "fp16", 32, 4, 4, 4, tuple(lq150), tuple(lk150), {}),
    ):  # fmt: skip
        out.append(
            _mkb(
                f"kern_thd_skew_{name}_{dt}_d{d}",
                reqs=(3, 8),
                tier="full",
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=len(ql),
                h_q=h_q,
                h_k=h_k,
                h_v=h_v,
                q_lens=ql,
                kv_lens=kl,
                length_mode="ragged",
                rdna=True,
                **mask,
            )
        )
    _check_cases(out)
    return tuple(out)


# Skewed THD batches of the work-list test (not table cases).
WORKLIST_SKEW_CASES = _worklist_skew_cases()
# Every case of the work-list test: the skewed batches, the table's ragged
# single sequence, many short sequences, long-and-short and tile-edge cells,
# the THD zero-length sequences, a THD h_k != h_v cell, a head-packed THD
# decode group and a decode with ragged kv lengths.
WORKLIST_CASES = (
    tuple(c.id for c in WORKLIST_SKEW_CASES)
    + (
        "bwd_ragged_single_sequence",
        "bwd_ragged_many_short_sequences",
        "bwd_ragged_long_and_short_d128",
        "bwd_ragged_tile_edges_br",
        "bwd_kern_thd_hkv_842_tl_fp16_d64",
        "bwd_kern_pack_thd_sq2_g4_br_fp16_d64",
        "bwd_decode_ragged_kv_lens",
    )
    + tuple(c.id for c in THD_ZERO_LENGTH_CASES)
)


# Non-compact THD storages, as keyword arguments of the numeric harness's
# ``ThdStorage`` (``bound``: "none" never passes a caller max_total, "storage"
# passes the storage token count, the end of the last segment). Every form is
# a storage hipDNN may hand over: offsets that do not start at 0, SEQ_LEN
# counts shorter than the allocated segments, unused tokens after the last
# sequence, padded tokens. A ``lead`` / ``gap`` of "B*S" / "S" is resolved per
# case (:func:`resolve_storage_form`) to ``B * S_max`` / ``S_max`` tokens, so
# the storage tokens of the later sequences lie past the ``B * S_max`` rows of
# a workspace without a caller bound (the defect class these forms pin).
THD_STORAGE_FORMS = {
    "lead_none": dict(lead="B*S", bound="none"),
    "lead_small_tail_pad_none": dict(lead=5, tail=7, token_pad=8, bound="none"),
    "short_counts_none": dict(gap="S", seq_len_counts=True, bound="none"),
    "lead_short_counts_int64_none": dict(
        lead=3, gap="S", tail=1, seq_len_counts=True, offset_dtype="int64",
        bound="none",
    ),  # fmt: skip
    "tail_none": dict(tail=9, bound="none"),
    "lead_tokens_units_none": dict(
        lead="B*S", uniform=True, multiplier=0, bound="none"
    ),
    "lead_short_counts_storage": dict(
        lead="B*S", gap=3, seq_len_counts=True, bound="storage"
    ),
}
# A GQA group (direct dK / dV), an h_k != h_v cell (atomic dK / dV: kv rows of
# the workspace), an empty last segment and a head-packed decode group.
THD_STORAGE_CASES = (
    "bwd_ragged_gqa4_br",
    "bwd_kern_thd_hkv_842_tl_fp16_d64",
    "bwd_kern_thd_zero_len_last_mha_br_fp16_d64",
    "bwd_kern_pack_thd_sq2_g4_br_fp16_d64",
)
_FUZZ_COUNTS = {"smoke": 2, "full": 6, "exhaustive": 16}


def resolve_storage_form(case, kw: dict) -> dict:
    """``kw`` with a "B*S" lead / "S" gap replaced by the case's token counts
    (``S_max = max(max(len_q), max(len_kv), 1)``)."""
    lq, lk = case.lens()
    s_max = max(max(lq, default=0), max(lk, default=0), 1)
    out = dict(kw)
    for key in ("lead", "gap", "tail"):
        if out.get(key) == "B*S":
            out[key] = len(lq) * s_max
        elif out.get(key) == "S":
            out[key] = s_max
    return out


def thd_fuzz_count(tier: str | None = None) -> int:
    """Seeded storages per THD storage case at ``tier``."""
    return _FUZZ_COUNTS[tier or os.environ.get("ROCKE_BWD_TIER", "smoke")]


def thd_storage_fuzz(seed: int, n: int) -> list[tuple[str, dict]]:
    """``n`` seeded non-compact THD storages ``(name, ThdStorage kwargs)``.

    Lead tokens, gaps (always with SEQ_LEN counts, so the counts end each
    sequence before the next offset), tails, token padding, int32 / int64
    offsets, element / token / unit-8 offsets, and no caller bound or the
    storage token count as the bound (both sides). Deterministic in
    ``(seed, index)``.
    """
    import random

    out = []
    for i in range(n):
        rng = random.Random(seed * 1000003 + i)
        kw = dict(
            lead=rng.choice((0, 1, 3, 17, "B*S")),
            tail=rng.choice((0, 2, 11)),
            token_pad=rng.choice((0, 0, 8, 16)),
            offset_dtype=rng.choice(("int32", "int64")),
            bound=rng.choice(("none", "none", "storage")),
        )
        if rng.random() < 0.5:
            kw.update(gap=rng.choice((1, 4, "S")), seq_len_counts=True)
        units = rng.choice(("elements", "tokens", "units8"))
        if units == "tokens":
            kw.update(uniform=True, multiplier=0, token_pad=0)
        elif units == "units8":
            kw.update(multiplier=8, token_pad=kw["token_pad"] or 8)
        if not (kw["lead"] or kw.get("gap") or kw["tail"] or kw["token_pad"]):
            kw["lead"] = 5  # always non-compact
        out.append((f"fuzz{seed}_{i}", kw))
    return out


def shard_of(case_id: str, n: int) -> int:
    """Stable shard index of a case id (crc32, independent of the process)."""
    return zlib.crc32(case_id.encode()) % n


def _env_shard() -> tuple[int, int]:
    raw = os.environ.get("ROCKE_BWD_SHARD", "0/1")
    i, n = (int(x) for x in raw.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"ROCKE_BWD_SHARD={raw!r} is not i/n with 0 <= i < n")
    return i, n


def selected_cases(*, tier: str | None = None, stage: int | None = None) -> list:
    """Table plus supplementary cases of ``tier`` with ``stage_of <= stage``.

    Defaults come from ``ROCKE_BWD_TIER`` (``smoke``), ``ROCKE_BWD_STAGE``
    (:data:`BUILT_STAGE`) and ``ROCKE_BWD_SHARD`` (``i/n``, default ``0/1``).
    Head dim 32 supplementary cases ride with stage 1 (the base path covers
    every head dim) although their feature rank is 2.
    """
    tier = tier or os.environ.get("ROCKE_BWD_TIER", "smoke")
    if stage is None:
        stage = int(os.environ.get("ROCKE_BWD_STAGE", str(BUILT_STAGE)))
    i, n = _env_shard()
    out = []
    for c in list(select_bwd(tier)) + [
        s for s in SUPPLEMENTARY_CASES if tier in s.tags
    ]:
        rank = stage_of(c)
        if c in SUPPLEMENTARY_CASES and c.d == 32 and rank == 2:
            rank = 1
        if rank <= stage and shard_of(c.id, n) == i:
            out.append(c)
    return out


def config_knobs(config: str, arch: str) -> dict | None:
    """Spec knob overrides of a correctness configuration (None: not on arch).

    The ``custom`` configuration has no single knob set (its knobs depend on
    the request's head size and ``stage_vec``); use :func:`config_policy`.
    """
    if config == CUSTOM_CONFIG:
        raise ValueError(
            "the custom configuration's knobs depend on the request; "
            "use config_policy"
        )
    if config not in CONFIGS:
        raise ValueError(f"unknown backward test configuration {config!r}")
    if config == "default":
        return {}
    if config == "debug":
        return dict(_DEBUG_CDNA) if arch in _CDNA else {}
    return dict(_MULTI_RDNA) if arch in _RDNA else None


def custom_rules(raw: str | None = None) -> tuple[tuple[dict, dict], ...] | None:
    """Rules of the ``custom`` configuration: ``((match, knobs), ...)``.

    ``raw`` (default: the ``ROCKE_BWD_CUSTOM_KNOBS`` environment variable) is
    a JSON list of objects ``{"head_size": d, "stage_vec": v, "knobs": {...}}``
    whose ``head_size`` and ``stage_vec`` are optional (absent: any value).
    ``knobs`` holds spec knobs (lists become tuples), runtime plan knobs, the
    aux kernels' tuning values (``AUX_KNOBS``) and, with ``dq_mode =
    "split"``, the dQ kernel's (``DQ_KNOBS``). A ``g_split`` value is a cap:
    each request runs ``gcd(g_split, G)`` with ``G = gcd(h_q / h_k, h_q /
    h_v)`` (so MHA cases keep ``g_split = 1``); a test's own ``overrides``
    are applied after it.
    ``None`` when the text is unset or empty; malformed text raises
    ``ValueError``.
    """
    raw = os.environ.get(CUSTOM_KNOBS_ENV, "") if raw is None else raw
    if not raw.strip():
        return None
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as ex:
        raise ValueError(f"{CUSTOM_KNOBS_ENV} is not JSON: {ex}") from None
    if not isinstance(items, list) or not items:
        raise ValueError(f"{CUSTOM_KNOBS_ENV} must be a non-empty JSON list")
    spec_knobs = set(AttnBwdSpec.__dataclass_fields__) - set(_REQUEST_SPEC_FIELDS)
    allowed = spec_knobs | set(RUNTIME_KNOBS) | set(AUX_KNOBS) | set(DQ_KNOBS)
    out = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("knobs"), dict):
            raise ValueError(f"custom rule {item!r} needs a 'knobs' object")
        extra = set(item) - set(_CUSTOM_MATCH) - {"knobs"}
        if extra:
            raise ValueError(f"custom rule keys {sorted(extra)} not in {_CUSTOM_MATCH}")
        unknown = set(item["knobs"]) - allowed
        if unknown:
            raise ValueError(f"custom rule knobs {sorted(unknown)} are not knobs")
        knobs = {
            k: tuple(v) if isinstance(v, list) else v for k, v in item["knobs"].items()
        }
        match = {k: int(item[k]) for k in _CUSTOM_MATCH if k in item}
        out.append((match, knobs))
    return tuple(out)


def custom_knobs_for(rules, head_size: int, stage_vec: int) -> dict:
    """Knobs of the first rule matching ``(head_size, stage_vec)``, else ``{}``
    (the arch default)."""
    key = {"head_size": head_size, "stage_vec": stage_vec}
    for match, knobs in rules:
        if all(key[k] == v for k, v in match.items()):
            return dict(knobs)
    return {}


# Knob value sets that the sweep builds and enumerates but that had no device
# run: name -> (archs, head sizes (None: any), spec knobs, dK / dV bit for bit
# equal to the default configuration in direct mode). Each set runs as a
# custom configuration on KNOB_SMOKE_CASES (test_knob_value_sets of the device
# numerics; knob_smoke_rules_json gives the ROCKE_BWD_CUSTOM_KNOBS text for
# the whole suite). The CPU test plans every (set, arch, case) the device test
# runs and checks that the resolved spec carries every knob of the set.
# Every row-major LDS buffer of the planner with the line XOR swizzle (a
# buffer a configuration does not have is ignored).
ROW_XOR_SWIZZLE = (
    "dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,"
    "v_stage=xor"
)
_W2_TILE = dict(
    waves=2,
    block_m=16,
    block_n=32,
    warp_grid_g4=(1, 2),
    atom_g02="16x16x16",
    atom_g13="16x16x16",
    atom_g4="16x16x16",
)
KNOB_SMOKE_SETS = {
    # G0/G2 and G1/G3 warp grids (2, W/2) (legal only with pt_route = lds)
    "warp_grid_2xw": (
        _CDNA,
        None,
        dict(block_m=32, pt_route="lds", warp_grid_g02=(2, 4), warp_grid_g13=(2, 4)),
        False,
    ),
    # P / dS through LDS on CDNA
    "pt_route_lds": (_CDNA, None, dict(pt_route="lds"), False),
    # mixed G0/G2 and G4 atom pairs (gfx950: the only arch with two atoms)
    "atom_g02_narrow": (("gfx950",), None, dict(atom_g02="16x16x16"), False),
    "atom_g4_narrow": (("gfx950",), None, dict(atom_g4="16x16x16"), False),
    # the wide G1 / G3 atom (K-permuted B through the stacked transpose read;
    # block_m >= 32: the d64 start point, a 32-row d128 tile)
    "atom_g13_wide": (("gfx950",), (64,), dict(atom_g13="16x16x32"), False),
    "atom_g13_wide_d128": (
        ("gfx950",),
        (128,),
        dict(atom_g13="16x16x32", block_m=32),
        False,
    ),
    # occupancy hints 2 and 3 (the small-tile region runs 3 and 4 on both
    # CDNA targets: test_attention_bwd_tiles_numeric)
    "waves_per_eu_2": (_CDNA, None, dict(waves_per_eu=2), True),
    "waves_per_eu_3": (("gfx942",), (64,), dict(_W2_TILE, waves_per_eu=3), False),
    # LDS accumulators on an MFMA target
    "acc_in_lds_mfma": (
        _CDNA,
        None,
        dict(_W2_TILE, acc_in_lds=True, agpr_alloc=(0, 0)),
        False,
    ),
    # the four machine-scheduler strategies without a device run
    "sched_max_ilp": (_CDNA, None, dict(scheduler_strategy="max-ilp"), True),
    "sched_max_memory_clause": (
        _CDNA,
        None,
        dict(scheduler_strategy="max-memory-clause"),
        True,
    ),
    "sched_iterative_ilp": (
        _CDNA,
        None,
        dict(scheduler_strategy="iterative-ilp"),
        True,
    ),
    "sched_iterative_maxocc": (
        _CDNA,
        None,
        dict(scheduler_strategy="iterative-maxocc"),
        True,
    ),
    # the backend's own AGPR split
    "agpr_alloc_backend": (_CDNA, None, dict(agpr_alloc="backend"), True),
    # kM0 = 64 (a 4-wave 64 x 64 tile; the 8-wave start points exceed the
    # register budget at kM0 = 64 on gfx950 and on gfx942 d128)
    # (gfx942 only: on gfx950 the resource gate refuses the tile on every
    # device case, see RESOURCE_REFUSED)
    "block_m_64": (
        ("gfx942",),
        None,
        dict(block_m=64, block_n=64, waves=4, warp_grid_g4=(2, 2)),
        False,
    ),
    # the schedule values (placement only: dK / dV bit for bit; the deferred
    # dQ atomics of stage_table_atomics add the same values one step later)
    **{
        f"sched_{s}{'_setprio' if p else ''}": (
            _CDNA + _RDNA,
            None,
            dict(sched=s, setprio=p),
            True,
        )
        for s, p in (
            ("iglp0", False),
            ("iglp1", False),
            ("fences_only", False),
            ("stage_table", False),
            ("stage_table", True),
            ("stage_table_atomics", False),
            ("stage_table_atomics", True),
        )
    },
    # G4 dS key slicing (same MMAs in the same K order: bit for bit)
    "g4_slice": (_CDNA, None, dict(g4_slicing="slice", block_k4=64), True),
    "g4_slice_db": (_CDNA, None, dict(g4_slicing="slice_db", block_k4=32), True),
    "g4_slice_db_g3": (_CDNA, None, dict(g4_slicing="slice_db_g3", block_k4=64), True),
    # asm_g13 with s_in_agpr: G0 / G2 as AGPR-form asm clusters too
    "asm_g13_s_in_agpr": (
        _CDNA,
        None,
        dict(agpr_alloc=(128, 128), agpr_route="asm_g13", s_in_agpr=True),
        True,
    ),
    # layout levers (the LDS layout only): the XT transposes and the line
    # xor of every row-major buffer, bit for bit the default configuration
    "xt_lds": (_CDNA + _RDNA, None, dict(transpose_source="xt_lds"), True),
    "lds_swizzle_xor": (_CDNA + _RDNA, None, dict(lds_swizzle=ROW_XOR_SWIZZLE), True),
}

# Layout levers on top of other knob sets: name -> (archs, head sizes (None:
# any), knobs, twin knobs). The set must give dK / dV bit for bit equal to its
# twin (direct mode) and pass the oracle; only the LDS layout differs (the XT
# images and their swizzles, the row-major xor). Device test:
# test_layout_twin_sets of the device numerics, on KNOB_SMOKE_CASES.
_XT = dict(transpose_source="xt_lds")
_XT_IMAGES_XOR = "dot=xor,kt_res=xor,kt_stage=xor,qt=xor"
# canonical (sorted) spelling: the launched spec carries it
_ALL_XOR = ",".join(sorted(f"{_XT_IMAGES_XOR},{ROW_XOR_SWIZZLE}".split(",")))
LAYOUT_TWIN_SETS = {
    # P^T / dS^T through LDS (plain K-outer reads of pt / ds), XT for the rest
    "xt_pt_route_lds": (_CDNA, None, dict(_XT, pt_route="lds"), dict(pt_route="lds")),
    # the K^T image resident in LDS (read every step by G4)
    "xt_kt_lds": (_CDNA, None, dict(_XT, kt_source="lds"), dict(kt_source="lds")),
    # LDS accumulators on an MFMA target (G1 / G3 update the fp32 images)
    "xt_acc_in_lds": (
        _CDNA,
        None,
        dict(_W2_TILE, acc_in_lds=True, agpr_alloc=(0, 0), **_XT),
        dict(_W2_TILE, acc_in_lds=True, agpr_alloc=(0, 0)),
    ),
    # the per-step images and the resident K^T image padded instead of the
    # predictor's choice (a padded register-K^T stage image overflows the
    # 64 KiB stage-0 phase of the gfx942 d128 tile: d64 only, next set)
    "xt_padded_images": (
        _CDNA + _RDNA,
        None,
        dict(_XT, lds_swizzle="dot=pad16,kt_res=pad32,qt=pad8"),
        {},
    ),
    "xt_padded_kt_stage": (
        _CDNA + _RDNA,
        (64,),
        dict(_XT, lds_swizzle="dot=pad16,kt_stage=pad32,qt=pad8"),
        {},
    ),
    # every LDS buffer XOR swizzled (images and row-major buffers)
    "xt_all_xor": (
        _CDNA + _RDNA,
        None,
        dict(_XT, lds_swizzle=_ALL_XOR),
        {},
    ),
    # gfx950 16x16x32 G1 / G3 under the relabel route: K-permuted B reads of
    # the 16-byte-chunk images (tr_read does not build it; lds_plain does)
    "xt_atom_g13_x32": (
        ("gfx950",),
        (64,),
        dict(_XT, atom_g13="16x16x32"),
        dict(transpose_source="lds_plain", atom_g13="16x16x32"),
    ),
}


def layout_twin_params(arch: str) -> list[tuple[str, str]]:
    """``(set name, case id)`` pairs of the layout twin sets on ``arch``."""
    out = []
    for name, (archs, sizes, _knobs, _twin) in LAYOUT_TWIN_SETS.items():
        if arch not in archs:
            continue
        for cid, d in KNOB_SMOKE_CASES:
            if sizes is None or d in sizes:
                out.append((name, cid))
    return out


# Compositions of levers that were built separately: name -> (archs, head
# sizes (None: any), knobs, twin knobs). Each set puts a lever on a structure
# it was not built against (the split-dQ main kernel, the prefetch and merged
# rings, the counted waits, the small-tile grids, the XT images, the VALU
# levers, the G4 key slicing, the asm G0 / G2 clusters, the schedule values);
# the twin is the same set without that lever (for the asm G0 / G2 clusters
# without their AGPR reservation too: the structure with a (128, 128)
# reservation and the VGPR route fails the resource gate on every device case,
# AGPR copies in the hot loop, and is refused). Both pass the oracle with
# every knob in the launched spec (a ``dq_*`` knob in the dQ kernel's), and
# dK / dV are bit for bit equal in direct mode: none of these levers changes
# a dK / dV matrix-op chain (the VALU levers are in both); under split dQ in
# both, dQ is bit for bit equal too (one fixed-order register sum, which no
# schedule value changes). Device test: test_compose_twin_sets of the device
# numerics; the CPU test plans every (set, arch, case) the device runs.
_VALU = dict(bwd_exp2="fast", stats_fma=True, stats_vec=4)
_ASM_SAGPR = dict(agpr_alloc=(128, 128), agpr_route="asm_g13", s_in_agpr=True)
_SMALL = {k: v for k, v in region_knobs("rep16x16_w4").items() if v is not None}
_SPLIT = dict(dq_mode="split")
_XT = dict(transpose_source="xt_lds")
_DMA1 = dict(global_path="dma", ring_depth=1)
_DMA2 = dict(global_path="dma", ring_depth=2)
_DMA3 = dict(global_path="dma", ring_depth=3)
_RING2 = dict(ring_depth=2)
_RING3 = dict(ring_depth=3)
_MERGE = dict(barrier_merge="on", ring_depth=2)
_COUNTED = dict(loop_sync="counted")
# schedule values composed with each ring: (name tag, knobs); the stage
# tables on every ring, fences_only and iglp1 on the DMA and register rings
# of depth 2
_RING_SCHEDS = (
    ("st", dict(sched="stage_table")),
    ("stap", dict(sched="stage_table_atomics", setprio=True)),
    ("fen", dict(sched="fences_only")),
    ("ig1", dict(sched="iglp1")),
)
# (the split-dQ main kernel has no dQ atomics to defer: the plain table with
# and without the priority raise)
_SPLIT_SCHEDS = (
    ("st", dict(sched="stage_table")),
    ("stp", dict(sched="stage_table", setprio=True)),
    ("fen", dict(sched="fences_only")),
    ("ig1", dict(sched="iglp1")),
)
COMPOSE_TWIN_SETS = {
    # the split-dQ main kernel (no G4)
    "split_ring2": (_CDNA, None, dict(_SPLIT, ring_depth=2), _SPLIT),
    "split_dma": (("gfx950",), None, dict(_SPLIT, global_path="dma"), _SPLIT),
    "split_xt": (_CDNA, None, dict(_SPLIT, **_XT), _SPLIT),
    "split_valu": (_CDNA, None, dict(_SPLIT, **_VALU), _VALU),
    "split_asm_sagpr": (_CDNA, None, dict(_SPLIT, **_ASM_SAGPR), _SPLIT),
    "split_small": (_CDNA, None, dict(_SMALL, **_SPLIT), _SMALL),
    # the merged barrier (2-slot ring) and the prefetch rings
    "merge_valu": (_CDNA, None, dict(barrier_merge="on", **_VALU), _VALU),
    # (gfx950 only: on gfx942 the set's tile fails the resource gate on every
    # device case, hot-loop AGPR copies, and is refused)
    "merge_asm_sagpr": (
        ("gfx950",), None, dict(barrier_merge="on", **_ASM_SAGPR),
        dict(barrier_merge="on"),
    ),
    "ring2_valu": (_CDNA, None, dict(ring_depth=2, **_VALU), _VALU),
    "ring2_asm_sagpr": (  # (gfx950 only, as merge_asm_sagpr)
        ("gfx950",), None, dict(ring_depth=2, **_ASM_SAGPR), dict(ring_depth=2)
    ),
    "ring2_small": (_CDNA, None, dict(_SMALL, ring_depth=2), _SMALL),
    "dma_valu": (("gfx950",), None, dict(global_path="dma", **_VALU), _VALU),
    # the counted waits
    "counted_valu": (_CDNA, None, dict(loop_sync="counted", **_VALU), _VALU),
    "counted_asm_sagpr": (
        _CDNA, None, dict(loop_sync="counted", **_ASM_SAGPR),
        dict(loop_sync="counted"),
    ),
    "counted_small": (_CDNA, None, dict(_SMALL, loop_sync="counted"), _SMALL),
    # the XT images
    "xt_valu": (_CDNA, None, dict(_XT, **_VALU), _VALU),
    "xt_small": (_CDNA, None, dict(_SMALL, **_XT), _SMALL),
    "xt_iglp": (_CDNA, None, dict(_XT, sched="iglp1"), _XT),
    # (K^T from registers: the key slices of an LDS K^T image are not built)
    "xt_g4slice": (
        _CDNA, None, dict(_XT, kt_source="reg", g4_slicing="slice_db", block_k4=32),
        dict(_XT, kt_source="reg"),
    ),
    # the schedule values and the G4 key slicing
    "sta_valu": (_CDNA, None, dict(sched="stage_table_atomics", **_VALU), _VALU),
    "small_iglp": (_CDNA, None, dict(_SMALL, sched="iglp1"), _SMALL),
    # the schedule values on the rings and the counted waits (the twin is the
    # same ring without a schedule value)
    **{
        f"{ring}_{tag}": (archs, None, dict(rk, **sk), rk)
        for ring, archs, rk in (
            ("dma1", ("gfx950",), _DMA1), ("dma2", ("gfx950",), _DMA2),
            ("dma3", ("gfx950",), _DMA3), ("ring2", _CDNA, _RING2),
            ("ring3", ("gfx950",), _RING3), ("merge", _CDNA, _MERGE),
            ("mergelate", _CDNA, dict(_MERGE, ring_issue="late")),
            ("counted", _CDNA, _COUNTED),
            ("countedlate", _CDNA, dict(_COUNTED, counted_issue="late")),
        )
        for tag, sk in _RING_SCHEDS
        if not (ring == "countedlate" and sk["sched"] == "stage_table_atomics")
        and (tag in ("st", "stap") or ring in ("dma2", "ring2"))
    },
    "g4slice_valu": (
        _CDNA, None, dict(g4_slicing="slice_db", block_k4=32, **_VALU), _VALU
    ),
    # the schedule values on the split-dQ main kernel's bodies (no G4: its
    # stages A..C; stage_table_atomics is refused there), the twin the same
    # split body without a schedule value: the stage tables on every body,
    # fences_only and iglp1 on the synchronous body and the depth-2 rings
    **{
        f"split_{body}_{tag}": (archs, None, dict(_SPLIT, **bk, **sk),
                                dict(_SPLIT, **bk))
        for body, archs, bk in (
            ("sync", _CDNA, {}), ("ring2", _CDNA, _RING2),
            ("ring3", ("gfx950",), _RING3), ("dma1", ("gfx950",), _DMA1),
            ("dma2", ("gfx950",), _DMA2), ("dma3", ("gfx950",), _DMA3),
        )
        for tag, sk in _SPLIT_SCHEDS
        if tag in ("st", "stp") or body in ("sync", "ring2", "dma2")
    },
    # the dQ kernel's stage table on top of the main kernel's (dQ bit for bit)
    "split_st_dqst": (
        _CDNA, None, dict(_SPLIT, sched="stage_table", dq_sched="stage_table"),
        _SPLIT,
    ),
    "split_dma2_st_dqstp": (
        ("gfx950",), None,
        dict(_SPLIT, **_DMA2, sched="stage_table", dq_sched="stage_table_setprio"),
        dict(_SPLIT, **_DMA2),
    ),
}  # fmt: skip


def compose_twin_params(arch: str) -> list[tuple[str, str]]:
    """``(set name, case id)`` pairs of the composition twin sets on ``arch``."""
    out = []
    for name, (archs, sizes, _knobs, _twin) in COMPOSE_TWIN_SETS.items():
        if arch not in archs:
            continue
        for cid, d in KNOB_SMOKE_CASES:
            if sizes is None or d in sizes:
                out.append((name, cid))
    return out


# An MHA band (d64, direct), a GQA group (d128, direct) and h_k != h_v (d64,
# atomic dK / dV): the cases of the runtime-variant test.
KNOB_SMOKE_CASES = (
    ("bwd_seqlen_33x72_br", 64),
    ("bwd_gqa_gqa4_causal_tl", 128),
    ("bwd_kern_hkv_842_tl_fp16_d64", 64),
)


# (set, arch, case id) cells whose main kernel the validator refuses: its
# production-comgr compile fails the resource gate or crashes the compiler
# (RESOURCE_REFUSED, test_attention_bwd_refusals.py), or it is a recorded XT /
# XOR layout spill (LAYOUT_SPILLING_COMBOS). The device test skips exactly
# these cells with the validator's reason; the knob values stay (lever
# retention) and the other cells of each set still run.
_GQA4, _BR33, _HKV = (
    "bwd_gqa_gqa4_causal_tl",
    "bwd_seqlen_33x72_br",
    "bwd_kern_hkv_842_tl_fp16_d64",
)
KNOB_SMOKE_REFUSED = {
    ("warp_grid_2xw", "gfx942", _GQA4): "scratch+vgpr_spill",
    ("sched_max_memory_clause", "gfx942", _BR33): "sgpr_spill",
    ("sched_max_memory_clause", "gfx942", _HKV): "sgpr_spill",
    ("sched_iterative_ilp", "gfx942", _BR33): "compiler crash",
    ("sched_iterative_ilp", "gfx942", _HKV): "compiler crash",
    ("sched_iterative_maxocc", "gfx942", _BR33): "sgpr_spill",
    ("sched_iterative_maxocc", "gfx942", _HKV): "sgpr_spill",
    ("block_m_64", "gfx942", _GQA4): "agpr_copy_traffic",
    ("block_m_64", "gfx942", _HKV): "scratch+vgpr_spill",
    ("sched_iterative_ilp", "gfx950", _BR33): "sgpr_spill",
    ("sched_iterative_ilp", "gfx950", _HKV): "sgpr_spill",
    ("asm_g13_s_in_agpr", "gfx950", _BR33): "agpr_copy_traffic",
    ("asm_g13_s_in_agpr", "gfx950", _HKV): "agpr_copy_traffic",
    ("sched_iglp1", "gfx1151", _GQA4): "scratch+vgpr_spill",
    ("sched_stage_table", "gfx1151", _GQA4): "scratch+vgpr_spill",
    ("sched_stage_table_setprio", "gfx1151", _GQA4): "scratch+vgpr_spill",
    ("sched_stage_table_atomics", "gfx1151", _GQA4): "scratch+vgpr_spill",
    ("sched_stage_table_atomics_setprio", "gfx1151", _GQA4): "scratch+vgpr_spill",
    ("xt_lds", "gfx1151", _BR33): "LAYOUT_SPILLING_COMBOS",
    ("xt_lds", "gfx1151", _HKV): "scratch+vgpr_spill",
    ("sched_iglp1", "gfx1201", _GQA4): "scratch+vgpr_spill",
    ("sched_stage_table_atomics", "gfx1201", _GQA4): "scratch+vgpr_spill",
    ("sched_stage_table_atomics_setprio", "gfx1201", _GQA4): "scratch+vgpr_spill",
}


# (set, arch, case id) cells of the layout twin sets whose main kernel, or its
# twin's, the validator refuses (RESOURCE_REFUSED: scratch / VGPR spills of
# the XT images on the one-wave gfx1151 tile, production compiler). The device
# test skips exactly these cells with the reason;
# test_attention_bwd_kernel_cases.py pins the table to the planner.
LAYOUT_TWIN_REFUSED = {
    ("xt_all_xor", "gfx1151", _GQA4): "set: scratch+vgpr_spill",
    ("xt_padded_kt_stage", "gfx1151", _HKV): "set: scratch+vgpr_spill",
}


# (set, arch, case id) cells of the composition twin sets whose main kernel,
# or its twin's, the validator refuses (RESOURCE_REFUSED: AGPR copies in the
# hot loop above the budget, or scratch / VGPR spills, on the production
# compiler). The device test skips exactly these cells with the reason;
# test_attention_bwd_kernel_cases.py pins the table to the planner.
COMPOSE_REFUSED = {
    ('counted_asm_sagpr', 'gfx942', 'bwd_kern_hkv_842_tl_fp16_d64'): "set: agpr_copy_traffic",
    ('counted_asm_sagpr', 'gfx942', 'bwd_seqlen_33x72_br'): "set: agpr_copy_traffic",
    ('counted_asm_sagpr', 'gfx950', 'bwd_kern_hkv_842_tl_fp16_d64'): "set: scratch+vgpr_spill+agpr_copy_traffic",
    ('counted_asm_sagpr', 'gfx950', 'bwd_seqlen_33x72_br'): "set: scratch+vgpr_spill+agpr_copy_traffic",
    ('merge_asm_sagpr', 'gfx950', 'bwd_kern_hkv_842_tl_fp16_d64'): "set: agpr_copy_traffic",
    ('merge_asm_sagpr', 'gfx950', 'bwd_seqlen_33x72_br'): "set: agpr_copy_traffic",
    ('ring2_asm_sagpr', 'gfx950', 'bwd_kern_hkv_842_tl_fp16_d64'): "set: agpr_copy_traffic",
    ('ring2_asm_sagpr', 'gfx950', 'bwd_seqlen_33x72_br'): "set: agpr_copy_traffic",
    ('split_dma2_ig1', 'gfx950', 'bwd_seqlen_33x72_br'): "set: sgpr_spill, twin: sgpr_spill",
    # the split twin without a schedule value (DMA R = 2, d64) spills SGPRs on
    # its own: refused exactly (RESOURCE_REFUSED, rows splitsched)
    ('split_dma2_fen', 'gfx950', 'bwd_seqlen_33x72_br'): "twin: sgpr_spill",
    ('split_dma2_st', 'gfx950', 'bwd_seqlen_33x72_br'): "twin: sgpr_spill",
    ('split_dma2_st_dqstp', 'gfx950', 'bwd_seqlen_33x72_br'): "twin: sgpr_spill",
    ('split_dma2_stp', 'gfx950', 'bwd_seqlen_33x72_br'): "twin: sgpr_spill",
}  # fmt: skip


def knob_smoke_params(arch: str) -> list[tuple[str, str]]:
    """``(set name, case id)`` pairs of the knob smoke sets that apply on
    ``arch`` (arch and head size of the set)."""
    out = []
    for name, (archs, sizes, _knobs, _bitwise) in KNOB_SMOKE_SETS.items():
        if arch not in archs:
            continue
        for cid, d in KNOB_SMOKE_CASES:
            if sizes is None or d in sizes:
                out.append((name, cid))
    return out


def knob_smoke_rules_json(name: str) -> str:
    """``ROCKE_BWD_CUSTOM_KNOBS`` text that runs knob set ``name`` as the
    custom configuration (restricted to its head sizes)."""
    _archs, sizes, knobs, _bitwise = KNOB_SMOKE_SETS[name]
    plain = {k: list(v) if isinstance(v, tuple) else v for k, v in knobs.items()}
    if sizes is None:
        return json.dumps([{"knobs": plain}], sort_keys=True)
    return json.dumps([{"head_size": d, "knobs": plain} for d in sizes], sort_keys=True)


def selected_configs(arch: str, raw: str | None = None) -> list[str]:
    """Configurations from ``ROCKE_BWD_CONFIG`` (``debug``, ``default``,
    ``multi`` or ``both`` = every distinct one on ``arch``; ``custom`` only
    by name, with ``ROCKE_BWD_CUSTOM_KNOBS`` set)."""
    raw = raw or os.environ.get("ROCKE_BWD_CONFIG", "both")
    if raw == "both":
        names = list(CONFIGS)
    else:
        names = [x.strip() for x in raw.split(",") if x.strip()]
    out, seen = [], []
    for name in names:
        if name == CUSTOM_CONFIG:
            if custom_rules() is None:
                raise ValueError(f"config {CUSTOM_CONFIG!r} needs {CUSTOM_KNOBS_ENV}")
            if name not in out:
                out.append(name)
            continue
        knobs = config_knobs(name, arch)
        if knobs is None or knobs in seen:
            continue
        seen.append(knobs)
        out.append(name)
    return out


def config_policy(
    config: str, arch: str, overrides: dict | None = None
) -> AttnBwdPolicy:
    """Every functional group declared, unverified waived, the config's knobs.

    ``overrides`` adds or replaces spec knobs (for example
    ``{"edge_tiles": True}``) on top of the configuration. The ``custom``
    configuration picks its knobs per request (:func:`custom_knobs_for`).
    """
    from kernels.common.attention_bwd_plan import FEATURE_GROUPS

    if config == CUSTOM_CONFIG:
        rules = custom_rules()
        if rules is None:
            raise ValueError(f"config {CUSTOM_CONFIG!r} needs {CUSTOM_KNOBS_ENV}")

        def tuning(req, arch_, route, perf_class):
            knobs = custom_knobs_for(rules, req.d_qk, attn_bwd_stage_vec(req))
            if "g_split" in knobs:
                # a rule's head split is a cap: the largest split dividing
                # both the rule's value and the request's atomic head group
                # (an MHA case runs g_split = 1), so one rule serves the suite
                group = math.gcd(req.h_q // req.h_k, req.h_q // req.h_v)
                knobs["g_split"] = math.gcd(int(knobs["g_split"]), group)
            knobs.update(overrides or {})
            return knobs

        return AttnBwdPolicy(
            declared_groups=frozenset(FEATURE_GROUPS),
            waive_unverified=True,
            tuning=tuning,
        )

    knobs = dict(config_knobs(config, arch) or {})
    knobs.update(overrides or {})

    def tuning(req, arch_, route, perf_class):
        return dict(knobs)

    return AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS),
        waive_unverified=True,
        tuning=tuning,
    )


def _narrow_cases():
    """Narrow-access cells: ``(case, view, base_bytes)``.

    fp16 / bf16, d64 / d128 (and d32 on the narrow 32 x 64 tile), top-left /
    bottom-right diagonals (and a window), MHA / GQA (and ``h_k != h_v`` with
    atomic dK / dV), fixed and padded batches on BHSD and BSHD views, and THD
    sequences with an odd token stride; every tensor base 2, 4 or 8 bytes off
    a 16-byte boundary.
    """
    out = []
    for name, dt, d, h_q, h_k, h_v, sq, skv, mask, view, base, tier in (
        ("mha_tl", "fp16", 64, 4, 4, 4, 80, 136, "tl", "bhsd", 2, "smoke"),
        ("gqa4_br", "bf16", 128, 8, 2, 2, 72, 100, "br", "bshd", 4, "smoke"),
        ("gqa2_br", "fp16", 128, 4, 2, 2, 100, 72, "br", "bhsd", 8, "full"),
        ("mha_tl", "bf16", 64, 4, 4, 4, 64, 150, "tl", "bshd", 8, "full"),
        ("mha_none", "bf16", 128, 4, 4, 4, 72, 72, "none", "bhsd", 4, "full"),
        ("hkv_842_br", "fp16", 64, 8, 4, 2, 72, 100, "br", "bhsd", 2, "full"),
        ("d32_win_br", "fp16", 32, 6, 2, 2, 72, 100, "win", "bshd", 4, "full"),
    ):  # fmt: skip
        m = dict(_mask("br"), left_bound=11) if mask == "win" else _mask(mask)
        out.append(
            (
                _mkb(
                    f"kern_narrow_{view}_{name}_{dt}_d{d}",
                    reqs=(2, 9),
                    tier=tier,
                    tags=("optional",) if d == 32 else (),
                    dtype=dt, d=d, b=2, h_q=h_q, h_k=h_k, h_v=h_v,
                    s_q=sq, s_kv=skv, rdna=True, **m,
                ),
                view,
                base,
            )
        )  # fmt: skip
    # padded batches (per-batch SEQ_LEN counts) on a narrow view
    out.append(
        (
            _mkb(
                "kern_narrow_bshd_padded_gqa2_br_fp16_d64",
                reqs=(2, 10),
                tier="full",
                dtype="fp16", d=64, b=3, h_q=4, h_k=2, h_v=2,
                q_lens=(40, 72, 9), kv_lens=(100, 33, 64), length_mode="padded",
                rdna=True, **_mask("br"),
            ),
            "bshd",
            2,
        )
    )  # fmt: skip
    for name, dt, d, h_q, h_k, h_v, ql, kl, mask, base, tier in (
        ("gqa4_br", "bf16", 64, 8, 2, 2, (40, 72, 9), (100, 33, 64), "br", 4, "full"),
        ("mha_tl", "fp16", 128, 4, 4, 4, (50, 33, 80), (50, 72, 17), "tl", 2, "smoke"),
        (
            "hkv_842_tl", "bf16", 64, 8, 4, 2, (72, 17, 50), (33, 100, 72), "tl", 8,
            "full",
        ),
    ):  # fmt: skip
        out.append(
            (
                _mkb(
                    f"kern_narrow_thd_{name}_{dt}_d{d}",
                    reqs=(2, 8),
                    tier=tier,
                    dtype=dt, d=d, b=len(ql), h_q=h_q, h_k=h_k, h_v=h_v,
                    q_lens=ql, kv_lens=kl, length_mode="ragged", rdna=True,
                    **_mask(mask),
                ),
                "thd",
                base,
            )
        )  # fmt: skip
    _check_cases([c for c, _v, _b in out])
    return tuple(out)


NARROW_CASES = _narrow_cases()

# stage_vec = 8 runs and their stage_vec = 1 twins (same tile, same storage,
# the narrow instance forced by a request alignment of 8 bytes): direct mode,
# so dK / dV must agree bit for bit. An MHA band and a GQA group.
NARROW_TWIN_CASES = ("bwd_seqlen_33x72_br", "bwd_gqa_gqa4_causal_tl")


# ---------------------------------------------------------------------------
# pruning set of the benchmark correctness gate
# ---------------------------------------------------------------------------

# Odd tails of the pruning set (query or key lengths that are not tile
# multiples, up to one past a power of two).
PRUNE_TAILS = (17, 33, 100, 1025)
# The grouped-query ratio of the pruning set's GQA cases (h_q / h_k).
_PRUNE_GROUP = 4
# Every head group (h_q / h_k) of the benchmark census: each one other than
# the set's own ratio and MHA gets a grouped-query case of its own in every
# class whose ``g_split`` divides it, so the runtime head-group arithmetic of a
# group the census times (5 and 8 too) runs in the gate whatever the split.
# test_attention_bwd_perf_gate pins this to the census.
PRUNE_HEAD_GROUPS = (1, 4, 5, 8)


def _prune_mask(kind, mask_class):
    if mask_class == "none":
        return {}
    if kind == "window":
        return dict(_mask("tl"), left_bound=7)
    return _mask(kind)


def perf_prune_cases(
    *,
    head_size: int,
    dtype: str = "bf16",
    mask_class: str = "band",
    seq_mode: str = "batched",
    dkv_mode: str = "direct",
    stage_vec: int = 8,
    g_split: int = 1,
) -> tuple:
    """The pruning set run before any configuration of this problem class is timed.

    About a dozen small cases through the float64 oracle, all of the main
    kernel class being timed (``head_size``, ``dtype``, ``mask_class``,
    ``seq_mode``, ``dkv_mode``; the caller re-lays them out for
    ``stage_vec = 1``): the odd tails of :data:`PRUNE_TAILS`, top-left and
    bottom-right diagonals with ``s_q != s_kv``, a window and a bottom-right
    diagonal with fully masked rows (band classes), a grouped-query ratio of
    4 throughout, one MHA case, ``h_k != h_v`` when ``dkv_mode`` is atomic, and
    a THD set with a zero-length last sequence when ``seq_mode`` is ``thd``.
    One grouped-query case per further head group of the census
    (:data:`PRUNE_HEAD_GROUPS`, ids ``gqa<G>``), where ``g_split`` divides it.
    ``g_split``: a grouped-query case whose head group ``G = gcd(h_q / h_k,
    h_q / h_v)`` it does not divide runs with ``h_q = h_k * g_split`` instead
    (a head group of exactly ``g_split``, ``h_k != h_v`` kept; ids carry
    ``_hq<h_q>``), so every legal ``g_split`` value (a divisor of a head group
    of the census: 5 and 8 too) has a full pruning set; MHA cases and the
    ``gqa<G>`` cases it does not divide are left out (the plan would refuse
    them). The sets of ``g_split`` 1 and 2 carry no rescaled case.
    """
    if seq_mode not in ("batched", "thd") or dkv_mode not in ("direct", "atomic"):
        raise ValueError(f"unknown class {seq_mode!r} / {dkv_mode!r}")
    if stage_vec not in (8, 1):
        raise ValueError(f"stage_vec={stage_vec} not in (8, 1)")
    band = mask_class == "band"
    t17, t33, t100, t1025 = PRUNE_TAILS
    # (name, h_q, h_k, h_v, q lengths, kv lengths, mask kind)
    rows = [
        ("tail17_tl", 8, 2, 2, (t17, t17), (t17, t17), "tl"),
        ("tail33_br", 8, 2, 2, (t33, t33), (t100, t100), "br"),
        ("tail100_tl", 8, 2, 2, (t100,), (t1025,), "tl"),
        ("tail1025_br", 8, 2, 2, (t1025,), (t100,), "br"),
        ("mha_tl", 4, 4, 4, (72, 72), (72, 72), "tl"),
    ]
    if band:
        rows.append(("window_dead", 8, 2, 2, (t100, t100), (40, 40), "window"))
    # the census's other head groups (G = 5, 8): a bottom-right diagonal with
    # s_q != s_kv and odd tails at h_k = 2
    groups = [g for g in PRUNE_HEAD_GROUPS if g not in (1, _PRUNE_GROUP)]
    rows += [(f"gqa{g}_br", 2 * g, 2, 2, (t33, 72), (t100, t100), "br") for g in groups]
    census = {f"gqa{g}_br" for g in groups}
    if seq_mode == "thd":
        rows = [
            (n, hq, hk, hv, ql[:1] + (max(1, ql[0] // 2),), kl[:1] + (kl[0],), m)
            for n, hq, hk, hv, ql, kl, m in rows
        ]
        rows.append(("zero_last", 8, 2, 2, (40, t33, 72, 0), (72, 50, 90, 0), "br"))
    if dkv_mode == "atomic":
        if g_split == 1:
            # atomic dK / dV without a head split means h_k != h_v
            rows = [
                (n, hq, hk, max(1, hk // 2), ql, kl, m)
                for n, hq, hk, hv, ql, kl, m in rows
            ]
        rows.append(("hkv_842_br", 8, 4, 2, (72, 72), (t100, t100), "br"))
    elif g_split != 1:
        raise ValueError("g_split > 1 makes dK / dV atomic (dkv_mode='atomic')")
    out = []
    for name, hq, hk, hv, ql, kl, kind in rows:
        atomic = hk != hv or g_split > 1
        if atomic != (dkv_mode == "atomic"):
            continue  # another main kernel (the dK / dV mode is a kernel switch)
        group = math.gcd(hq // hk, hq // hv) if atomic else hq // hk
        rescaled = ""
        if group % g_split and hq != hk and name not in census:
            # a head split this case's group does not take: the case with a
            # head group of exactly g_split (h_v divides h_k: G = h_q / h_k)
            hq, group = hk * g_split, g_split
            rescaled = f"_hq{hq}"
        if group % g_split:
            continue  # the plan refuses a head split that does not divide G
        cid = (
            f"kern_prune_{name}{rescaled}_{dtype}_d{head_size}_{mask_class}_"
            f"{seq_mode}_{dkv_mode}"
        )
        kw = dict(
            reqs=(2, 9),
            tier="full",
            tags=("perf_prune",) + (("optional",) if head_size == 32 else ()),
            dtype=dtype,
            d=head_size,
            h_q=hq,
            h_k=hk,
            h_v=hv,
            rdna=True,
            **_prune_mask(kind, mask_class),
        )
        if seq_mode == "thd":
            if 0 in ql or 0 in kl:
                case = _zero_length(
                    cid,
                    b=len(ql),
                    q_lens=ql,
                    kv_lens=kl,
                    length_mode="ragged",
                    **kw,
                )
            else:
                case = _mkb(
                    cid,
                    b=len(ql),
                    q_lens=ql,
                    kv_lens=kl,
                    length_mode="ragged",
                    **kw,
                )
                _check_cases([case])
        else:
            case = _mkb(cid, b=len(ql), s_q=ql[0], s_kv=kl[0], **kw)
            _check_cases([case])
        out.append(case)
    return tuple(out)
