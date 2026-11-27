# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The host admission predicate of the dense attention strides, pinned by a table.

``kernels/common/_attention_dense_strides.py`` decides whether four dense
Q/K/V/O tensors (any ``[B, S, H, D]`` view with unit head-dim stride) may be
launched, and with which canonical strides (``dense_strides_admit``). The
predicate has a C++ twin in the hipDNN engines. Both are pinned to one file,
``platform/tests/golden/attention_dense_stride_vectors.json`` (platform owns
it: it is the contract of the twin):

* this module holds a hand-written table of cases. The expected verdict of every
  row is written by hand from the rules in the docstring of
  ``dense_strides_admit``, never computed by calling it: the table is the
  independent oracle. The edge values are derived by plain arithmetic
  (``test_edge_values_are_the_hand_derived_ones``), also without the predicate;
* ``python test_attention_dense_strided_admission.py --regenerate`` renders the
  table to the JSON file (the only thing that writes it). The canonical strides
  of the ``admit`` rows are the one value taken from the module
  (``canonical_dense_strides``); ``test_admit_rows_record_the_hand_canonical_strides``
  recomputes them by hand;
* the tests below require that the file on disk is what the table renders, and
  that the current Python gives every recorded verdict. The C++ twin reads the
  same file and must give the same ones.

JSON layout. Header: ``elem_bytes``, ``stride_multiple``, ``pointer_align``,
``slab_slack_tiles`` (the constants of the rules), ``reasons`` (the decline
reasons in the order the checks run), ``check_order``. Then ``cases``, one per
row: ``name``; ``problem`` (``batch``, ``seqlen_q``, ``seqlen_kv``,
``num_query_heads``, ``num_kv_heads``, ``head_size``, ``block_n``); the four
operands ``q``, ``k``, ``v``, ``o``, each ``dims`` = ``[B, H, S, D]``,
``strides`` = ``[batch, head, token, head-dim]`` in elements, ``align`` (the
power of two, in bytes, that divides the base address), ``virtual``,
``ragged``; ``expect`` (``"admit"`` or one decline reason); and, on ``admit``
rows only, ``strides``: the 12 canonical strides in the order
``stride_{q,k,v,o}_{batch,head,token}``. Integers only. Each row is built around
one defect, but some also break a rule a later check enforces (a head-dim stride
of 2 also overlaps the head axis; a zero or negative stride also overlaps): the
first check wins, so those rows pin the check order as well. The ``order_*``
rows have two defects on purpose and pin which reason wins.

The tests after the table cover what the table cannot: reading views, the raw
addresses and views of callers without torch, the check of O against Q, K and V
at launch, and the slab and parameter helpers the kernels call.

Pure host code -- no torch, no GPU, no compiler.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernels.common import _attention_dense_strides as ds
from kernels.common._attention_dense_strides import (
    DENSE_STRIDE_NAMES,
    DENSE_STRIDE_REASONS,
    DenseOperand,
    admit_dense_launch_strides,
    append_dense_stride_signature,
    canonical_dense_strides,
    declare_dense_stride_params,
    dense_kv_reach_rows,
    dense_operand_dims,
    dense_operand_from_view,
    dense_slab_bytes,
    dense_strides_admit,
    emit_dense_slab_ptr,
    packed_dense_operand,
    packed_launch_strides,
)
from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
from kernels.gfx950.attention_dense import Gfx950AttentionDenseSpec

# Source tree: rocke/platform/tests/golden. Installed tree (library tests under
# tests/library/tests, platform tests under tests/): tests/golden.
_ROOT = Path(__file__).resolve().parents[2]
VECTORS = next(
    (
        p
        for p in (
            _ROOT
            / "platform"
            / "tests"
            / "golden"
            / "attention_dense_stride_vectors.json",
            _ROOT / "golden" / "attention_dense_stride_vectors.json",
        )
        if p.is_file()
    ),
    _ROOT / "platform" / "tests" / "golden" / "attention_dense_stride_vectors.json",
)

TENSORS = ("q", "k", "v", "o")

# The constants of the rules, written out here and pinned against the module by
# test_header_constants_are_the_modules.
_ELEM_BYTES = 2
_STRIDE_MULTIPLE = 8
_POINTER_ALIGN = 16
_SLAB_SLACK_TILES = 2
_REASONS = [
    "operand",
    "head_dim_stride",
    "stride_alignment",
    "overlap",
    "pointer_alignment",
    "size_limit",
]
_CHECK_ORDER = (
    "checks run in the order of reasons; inside a check the operands go q, k, v, o; "
    "the first failure is the verdict"
)

# --------------------------------------------------------------------------- #
# Row constructors
# --------------------------------------------------------------------------- #


def _prob(**changes) -> dict:
    """A problem (the spec fields the predicate reads); the default is a small
    GQA batch of 2: Hq=8, Hkv=2, D=128, S=64, one 64-row tile."""
    p = {
        "batch": 2,
        "seqlen_q": 64,
        "seqlen_kv": 64,
        "num_query_heads": 8,
        "num_kv_heads": 2,
        "head_size": 128,
        "block_n": 64,
    }
    p.update(changes)
    return p


def _dims(p: dict, t: str) -> list[int]:
    """``[B, H, S, D]`` of tensor ``t``."""
    if t in ("q", "o"):
        return [p["batch"], p["num_query_heads"], p["seqlen_q"], p["head_size"]]
    return [p["batch"], p["num_kv_heads"], p["seqlen_kv"], p["head_size"]]


def _op(dims, strides, align=16, virtual=False, ragged=False) -> dict:
    return {
        "dims": list(dims),
        "strides": list(strides),
        "align": align,
        "virtual": virtual,
        "ragged": ragged,
    }


def packed(p: dict, t: str) -> dict:
    """Packed ``[B, S, H, D]``: strides (S*H*D, D, H*D, 1) for (batch, head, token,
    dim)."""
    B, H, S, D = _dims(p, t)
    return _op([B, H, S, D], [S * H * D, D, H * D, 1])


def with_(op: dict, **changes) -> dict:
    """``op`` with changes: ``dims``/``strides`` whole, ``align``/``virtual``/
    ``ragged``, or one stride by axis: ``sB`` (batch), ``sH`` (head), ``sS``
    (token), ``sD`` (head dim)."""
    out = dict(op, strides=list(op["strides"]), dims=list(op["dims"]))
    for key, value in changes.items():
        if key in ("dims", "strides"):
            out[key] = list(value)
        elif key in ("align", "virtual", "ragged"):
            out[key] = value
        else:
            out["strides"][{"sB": 0, "sH": 1, "sS": 2, "sD": 3}[key]] = value
    return out


def mod(**changes):
    """``op -> with_(op, **changes)``, to pass as a tensor override of ``_case``."""
    return lambda op: with_(op, **changes)


def to_bhsd(op: dict) -> dict:
    """``op`` laid out as packed ``[B, H, S, D]``: strides (H*S*D, S*D, D, 1)."""
    _, H, S, D = op["dims"]
    return with_(op, strides=[H * S * D, S * D, D, 1])


def then(*steps):
    def apply(op):
        for step in steps:
            op = step(op)
        return op

    return apply


def _case(name: str, expect: str, p: dict | None = None, **over) -> dict:
    """One row: ``p`` (default problem when None), packed operands, each
    overridden by ``over[t]`` (a function ``op -> op``)."""
    assert set(over) <= set(TENSORS), over
    p = _prob() if p is None else p
    row = {"name": name, "problem": p}
    for t in TENSORS:
        op = packed(p, t)
        row[t] = over[t](op) if t in over else op
    row["expect"] = expect
    return row


# --------------------------------------------------------------------------- #
# The edges, derived by hand (plain arithmetic, no call into the predicate)
# --------------------------------------------------------------------------- #
#
# K/V reach edge. Problem: seqlen_kv=64, block_n=64, B=1, Hkv=1, D=128, fp16.
#   reach rows R = 64 (padded length) + 2*64 (two slack tiles) = 192
#   the furthest byte a load may start at is 2*((R-1)*sS + D - 1), below 2**31:
#     191*sS + 127 < 2**30 = 1073741824  <=>  sS <= 5621684
#   largest multiple of 8: 5621680 (191*5621680 + 127 = 1073741007 < 1073741824)
#   next multiple of 8:    5621688 (191*5621688 + 127 = 1073742535 >= 1073741824)
KV_EDGE_ADMIT = 5621680
KV_EDGE_REFUSED = 5621688
# Without the 2*block_n slack the reach is 64 rows: 63*sS + 127 < 2**30 holds up to
# sS ~ 17.0 million. sS = 8000000 passes that (63*8000000 + 127 = 504000127) and fails
# with the slack (191*8000000 + 127 = 1528000127 >= 1073741824).
KV_SLACK_ONLY_REFUSED = 8000000
# The padded length counts: seqlen_kv=65 pads to 128 rows, R = 128 + 128 = 256,
#   255*sS + 127 < 2**30  <=>  sS <= 4210751; multiples of 8: 4210744 / 4210752.
KV_PADDED_EDGE_ADMIT = 4210744
KV_PADDED_EDGE_REFUSED = 4210752
# block_n=128 with seqlen_kv=64: padded 128, R = 128 + 256 = 384, 383*sS + 127 < 2**30
# <=> sS <= 2803503, so the edge sS = 5621680 of block_n=64 is refused.
# The rule is "the last load STARTS below 2**31", so a reach of exactly 2**31 bytes is
# admitted (2*((R-1)*sS + D - 1) = 2**31 - 2). With sS = D = 128 and B = Hkv = 1:
#   seqlen_kv = 2**23 - 128 = 8388480 is a multiple of 64, so it pads to itself,
#   R = 8388480 + 2*64 = 2**23 = 8388608, (R-1)*128 + 128 = 2**23 * 128 = 2**30,
#   reach = 2 * 2**30 = 2**31 bytes: admit. One more row (seqlen_kv = 8388481 pads to
#   8388544, R = 8388672, reach = 2*8388672*128 = 2147500032 bytes) is refused.
KV_REACH_EXACT_SEQLEN = 8388480
KV_REACH_PAST_SEQLEN = 8388481
#
# Q/O edge. Problem: seqlen_q=256, D=128, B=1, Hq=1: (256-1)*sS + 128 < 2**31
#   255*8421496 + 128 = 2147481608 < 2147483648 (= 2**31): admit
#   255*8421504 + 128 = 2147483648 (= 2**31):              refused
Q_EDGE_ADMIT = 8421496
Q_EDGE_REFUSED = 8421504
#
# Batch stride (B=2): every canonical stride is below 2**31, so a batch stride of
# 2**31 is refused and 2**31 - 8 (the largest multiple of 8) is admitted.
INT32 = 2**31
#
# 64-bit slab base (B = 2**32, Hq=Hkv=1, S=8, D=64, fp16). The base byte offset plus
# the slab, 2*((B-1)*sB + (H-1)*sH) + 2*((S-1)*sS + D) with sS = 64 (slab 1024 bytes),
# stays below 2**63:
#   sB = 2**30:     2*(2**32 - 1)*2**30 + 1024 = 2**63 - 2**31 + 1024 < 2**63: admit
#   sB = 2**30 + 8: 2*(2**32 - 1)*(2**30 + 8) = 2**63 + 2**36 - 2**31 - 16 >= 2**63
#   sB = 2**31 - 8: about 2**64: refused
BIG_B = 2**32
BASE_EDGE_ADMIT = 2**30
BASE_EDGE_REFUSED = 2**30 + 8

# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

P0 = _prob()
P_B1 = _prob(batch=1)
P_D64 = _prob(head_size=64)
P_D64_B1 = _prob(batch=1, head_size=64)
P_MHA = _prob(num_kv_heads=8)  # Hq == Hkv, for packed QKV
P_MHA_B1 = _prob(batch=1, num_kv_heads=8)
P_MQA = _prob(num_kv_heads=1)
P_H1 = _prob(num_query_heads=1, num_kv_heads=1)
P_DECODE = _prob(batch=4, seqlen_q=1, seqlen_kv=256, num_query_heads=32, num_kv_heads=8)
P_KV1 = _prob(seqlen_kv=1)
P_K2_B1 = _prob(batch=1, num_kv_heads=2)
# K/V reach edge problems: one KV head, one batch.
P_KVEDGE = _prob(batch=1, num_kv_heads=1)
P_KVPAD = _prob(batch=1, seqlen_kv=65, num_kv_heads=1)
P_KVBN128 = _prob(batch=1, num_kv_heads=1, block_n=128)
P_KVREACH = _prob(batch=1, seqlen_kv=KV_REACH_EXACT_SEQLEN, num_kv_heads=1)
P_KVREACH_PAST = _prob(batch=1, seqlen_kv=KV_REACH_PAST_SEQLEN, num_kv_heads=1)
# Q/O edge problem: 256 query rows, one head, one batch.
P_QEDGE = _prob(batch=1, seqlen_q=256, num_query_heads=1, num_kv_heads=1)
P_B1H1 = _prob(batch=1, num_query_heads=1, num_kv_heads=1)
P_BATCH = _prob(num_query_heads=1, num_kv_heads=1)
P_BIG = _prob(
    batch=BIG_B,
    seqlen_q=8,
    seqlen_kv=8,
    num_query_heads=1,
    num_kv_heads=1,
    head_size=64,
)
P_BIGH = _prob(
    batch=1,
    seqlen_q=8,
    seqlen_kv=8,
    num_query_heads=2**32,
    num_kv_heads=1,
    head_size=64,
)

_CASES: tuple[dict, ...] = (
    # ---- admit: layouts ----
    _case("admit_packed_bshd_gqa_d128_batch2", "admit"),
    _case("admit_packed_bshd_gqa_d128_batch1", "admit", P_B1),
    _case("admit_packed_bshd_gqa_d64_batch2", "admit", P_D64),
    _case("admit_packed_bshd_gqa_d64_batch1", "admit", P_D64_B1),
    _case(
        "admit_bhsd_all_four",
        "admit",
        q=to_bhsd,
        k=to_bhsd,
        v=to_bhsd,
        o=to_bhsd,
    ),
    _case(
        "admit_bhsd_all_four_batch1",
        "admit",
        P_B1,
        q=to_bhsd,
        k=to_bhsd,
        v=to_bhsd,
        o=to_bhsd,
    ),
    _case("admit_bhsd_kv_only", "admit", k=to_bhsd, v=to_bhsd),
    _case("admit_bhsd_q_and_o_only", "admit", q=to_bhsd, o=to_bhsd),
    # Q, K and V are views of one [B, S, 3, H, D] buffer (Hq == Hkv): token stride
    # 3*H*D = 3072, head stride D, batch stride S*3*H*D = 64*3072 = 196608.
    _case(
        "admit_packed_qkv_one_buffer",
        "admit",
        P_MHA,
        q=mod(sB=196608, sH=128, sS=3072),
        k=mod(sB=196608, sH=128, sS=3072),
        v=mod(sB=196608, sH=128, sS=3072),
    ),
    _case(
        "admit_packed_qkv_one_buffer_batch1",
        "admit",
        P_MHA_B1,
        q=mod(sB=196608, sH=128, sS=3072),
        k=mod(sB=196608, sH=128, sS=3072),
        v=mod(sB=196608, sH=128, sS=3072),
    ),
    # K and V are views of one [B, S, 2, Hkv, D] buffer: token stride 2*2*128 = 512,
    # batch stride 64*512 = 32768.
    _case(
        "admit_packed_kv_one_buffer_gqa",
        "admit",
        k=mod(sB=32768, sS=512),
        v=mod(sB=32768, sS=512),
    ),
    _case("admit_bhsd_head_stride_padded_k", "admit", k=mod(sB=24576, sH=9216, sS=128)),
    _case("admit_batch_stride_larger_than_slab_q", "admit", q=mod(sB=131072)),
    _case("admit_token_stride_padded_by_8_q", "admit", q=mod(sB=66048, sS=1032)),
    _case(
        "admit_token_stride_padded_by_8_all",
        "admit",
        q=mod(sB=66048, sS=1032),
        k=mod(sB=16896, sS=264),
        v=mod(sB=16896, sS=264),
        o=mod(sB=66048, sS=1032),
    ),
    _case(
        "admit_pointer_align_32",
        "admit",
        q=mod(align=32),
        k=mod(align=32),
        v=mod(align=32),
        o=mod(align=32),
    ),
    _case(
        "admit_pointer_align_256",
        "admit",
        q=mod(align=256),
        k=mod(align=256),
        v=mod(align=256),
        o=mod(align=256),
    ),
    _case("admit_one_head_token_stride_equals_head_size", "admit", P_H1),
    # ---- admit: extent-1 axes are exempt and canonicalized ----
    _case(
        "admit_mqa_kv_head_stride_12345",
        "admit",
        P_MQA,
        k=mod(sH=12345),
        v=mod(sH=12345),
    ),
    _case("admit_mqa_kv_head_stride_0", "admit", P_MQA, k=mod(sH=0), v=mod(sH=0)),
    _case("admit_mqa_kv_head_stride_neg7", "admit", P_MQA, k=mod(sH=-7), v=mod(sH=-7)),
    _case(
        "admit_mqa_kv_head_stride_2p40",
        "admit",
        P_MQA,
        k=mod(sH=2**40),
        v=mod(sH=2**40),
    ),
    _case("admit_batch1_q_batch_stride_0", "admit", P_B1, q=mod(sB=0)),
    _case("admit_batch1_k_batch_stride_neg1", "admit", P_B1, k=mod(sB=-1)),
    _case("admit_batch1_v_batch_stride_2p40", "admit", P_B1, v=mod(sB=2**40)),
    _case(
        "admit_batch1_all_batch_strides_garbage",
        "admit",
        P_B1,
        q=mod(sB=0),
        k=mod(sB=-1),
        v=mod(sB=2**40),
        o=mod(sB=12345),
    ),
    _case(
        "admit_one_head_all_head_strides_garbage",
        "admit",
        P_H1,
        q=mod(sH=0),
        k=mod(sH=12345),
        v=mod(sH=-7),
        o=mod(sH=2**40),
    ),
    _case("admit_decode_seqlen_q1_packed", "admit", P_DECODE),
    _case(
        "admit_decode_seqlen_q1_garbage_token_stride",
        "admit",
        P_DECODE,
        q=mod(sS=-5),
        o=mod(sS=0),
    ),
    _case(
        "admit_seqlen_kv1_garbage_token_stride",
        "admit",
        P_KV1,
        k=mod(sS=-5),
        v=mod(sS=0),
    ),
    # ---- operand ----
    _case(
        "operand_q_rank3_dims",
        "operand",
        q=mod(dims=[2, 8, 64], strides=[65536, 128, 1024]),
    ),
    _case(
        "operand_k_rank5_dims",
        "operand",
        k=mod(dims=[2, 2, 64, 128, 1], strides=[16384, 128, 256, 1, 1]),
    ),
    _case("operand_v_rank3_strides", "operand", v=mod(strides=[16384, 128, 256])),
    _case("operand_k_zero_dim", "operand", k=mod(dims=[2, 2, 0, 128])),
    _case("operand_q_negative_dim", "operand", q=mod(dims=[2, 8, -64, 128])),
    _case("operand_zero_dim_agrees_with_problem", "operand", _prob(seqlen_q=0)),
    _case("operand_k_seqlen_off_by_one", "operand", k=mod(dims=[2, 2, 63, 128])),
    _case("operand_q_head_count_mismatch", "operand", q=mod(dims=[2, 4, 64, 128])),
    _case("operand_o_head_size_mismatch", "operand", o=mod(dims=[2, 8, 64, 64])),
    _case("operand_v_batch_mismatch", "operand", v=mod(dims=[3, 2, 64, 128])),
    _case("operand_q_dims_in_bshd_order", "operand", q=mod(dims=[2, 64, 8, 128])),
    _case("operand_q_virtual", "operand", q=mod(virtual=True)),
    _case("operand_o_virtual", "operand", o=mod(virtual=True)),
    _case("operand_k_ragged", "operand", k=mod(ragged=True)),
    _case("operand_v_ragged", "operand", v=mod(ragged=True)),
    # ---- head_dim_stride ----
    _case("head_dim_stride_q_2", "head_dim_stride", q=mod(sD=2)),
    _case("head_dim_stride_v_0", "head_dim_stride", v=mod(sD=0)),
    _case("head_dim_stride_k_negative", "head_dim_stride", k=mod(sD=-1)),
    _case("head_dim_stride_o_8", "head_dim_stride", o=mod(sD=8)),
    # ---- stride_alignment ----
    _case(
        "stride_alignment_q_token_pad4", "stride_alignment", q=mod(sB=65792, sS=1028)
    ),
    _case("stride_alignment_q_token_odd", "stride_alignment", q=mod(sB=65600, sS=1025)),
    _case(
        "stride_alignment_k_batch_not_multiple_of_8",
        "stride_alignment",
        k=mod(sB=16388),
    ),
    _case(
        "stride_alignment_q_head_132",
        "stride_alignment",
        q=mod(sB=67584, sH=132, sS=1056),
    ),
    # Token stride below D with the stride a multiple of 8 and positive, so only the
    # token >= D rule fires (the row also overlaps; the alignment check runs first).
    _case(
        "stride_alignment_q_token_64_below_head_size_128",
        "stride_alignment",
        P_H1,
        q=mod(sS=64),
    ),
    _case(
        "stride_alignment_q_token_120_below_head_size_128",
        "stride_alignment",
        P_H1,
        q=mod(sS=120),
    ),
    _case(
        "stride_alignment_v_token_56_below_head_size_64",
        "stride_alignment",
        _prob(num_query_heads=1, num_kv_heads=1, head_size=64),
        v=mod(sS=56),
    ),
    _case("stride_alignment_k_batch_zero_broadcast", "stride_alignment", k=mod(sB=0)),
    _case("stride_alignment_q_head_zero_broadcast", "stride_alignment", q=mod(sH=0)),
    _case("stride_alignment_v_head_negative", "stride_alignment", v=mod(sH=-128)),
    _case("stride_alignment_q_batch_negative", "stride_alignment", q=mod(sB=-65536)),
    _case("stride_alignment_o_token_zero", "stride_alignment", o=mod(sS=0)),
    # ---- overlap ----
    _case(
        "overlap_k_head_and_token_stride_equal_head_size",
        "overlap",
        k=mod(sH=128, sS=128),
    ),
    _case("overlap_o_head_stride_64_below_head_size", "overlap", o=mod(sH=64)),
    _case("overlap_q_token_stride_inside_head_span", "overlap", q=mod(sS=512)),
    _case("overlap_q_batch_stride_below_slab", "overlap", q=mod(sB=8192)),
    _case("overlap_q_batch_stride_equals_token_stride", "overlap", q=mod(sB=1024)),
    # ---- pointer_alignment ----
    _case("pointer_alignment_q_8", "pointer_alignment", q=mod(align=8)),
    _case("pointer_alignment_v_2", "pointer_alignment", v=mod(align=2)),
    _case("pointer_alignment_o_4", "pointer_alignment", o=mod(align=4)),
    _case("pointer_alignment_k_1", "pointer_alignment", k=mod(align=1)),
    # ---- size_limit: K/V reach edge (reach rows R = padded seqlen_kv + 2*block_n) ----
    _case(
        "admit_k_token_stride_at_reach_edge", "admit", P_KVEDGE, k=mod(sS=KV_EDGE_ADMIT)
    ),
    _case(
        "size_limit_k_token_stride_past_reach_edge",
        "size_limit",
        P_KVEDGE,
        k=mod(sS=KV_EDGE_REFUSED),
    ),
    _case(
        "admit_v_token_stride_at_reach_edge", "admit", P_KVEDGE, v=mod(sS=KV_EDGE_ADMIT)
    ),
    _case(
        "size_limit_v_token_stride_past_reach_edge",
        "size_limit",
        P_KVEDGE,
        v=mod(sS=KV_EDGE_REFUSED),
    ),
    _case(
        "size_limit_k_token_stride_refused_only_by_slack",
        "size_limit",
        P_KVEDGE,
        k=mod(sS=KV_SLACK_ONLY_REFUSED),
    ),
    _case(
        "admit_k_token_stride_at_padded_reach_edge",
        "admit",
        P_KVPAD,
        k=mod(sS=KV_PADDED_EDGE_ADMIT),
    ),
    _case(
        "size_limit_k_token_stride_past_padded_reach_edge",
        "size_limit",
        P_KVPAD,
        k=mod(sS=KV_PADDED_EDGE_REFUSED),
    ),
    _case(
        "size_limit_k_token_stride_unpadded_edge_refused_when_padded",
        "size_limit",
        P_KVPAD,
        k=mod(sS=KV_EDGE_ADMIT),
    ),
    _case(
        "size_limit_k_token_stride_edge_refused_at_block_n_128",
        "size_limit",
        P_KVBN128,
        k=mod(sS=KV_EDGE_ADMIT),
    ),
    # K and V packed with one KV head: token stride D = 128, reach exactly 2**31 bytes.
    _case("admit_kv_reach_exactly_2p31_bytes", "admit", P_KVREACH),
    _case("size_limit_kv_reach_past_2p31_bytes", "size_limit", P_KVREACH_PAST),
    # ---- size_limit: Q/O edge ----
    _case("admit_q_token_stride_at_edge", "admit", P_QEDGE, q=mod(sS=Q_EDGE_ADMIT)),
    _case(
        "size_limit_q_token_stride_past_edge",
        "size_limit",
        P_QEDGE,
        q=mod(sS=Q_EDGE_REFUSED),
    ),
    _case("admit_o_token_stride_at_edge", "admit", P_QEDGE, o=mod(sS=Q_EDGE_ADMIT)),
    _case(
        "size_limit_o_token_stride_past_edge",
        "size_limit",
        P_QEDGE,
        o=mod(sS=Q_EDGE_REFUSED),
    ),
    # ---- size_limit: strides below 2**31 ----
    _case("admit_q_batch_stride_2p31_minus_8", "admit", P_BATCH, q=mod(sB=INT32 - 8)),
    _case("size_limit_q_batch_stride_2p31", "size_limit", P_BATCH, q=mod(sB=INT32)),
    _case("admit_k_batch_stride_2p31_minus_8", "admit", P_BATCH, k=mod(sB=INT32 - 8)),
    _case("size_limit_k_batch_stride_2p31", "size_limit", P_BATCH, k=mod(sB=INT32)),
    _case("size_limit_v_batch_stride_2p31", "size_limit", P_BATCH, v=mod(sB=INT32)),
    _case("size_limit_o_batch_stride_2p31", "size_limit", P_BATCH, o=mod(sB=INT32)),
    _case("admit_k_head_stride_2p31_minus_8", "admit", P_K2_B1, k=mod(sH=INT32 - 8)),
    _case("size_limit_k_head_stride_2p31", "size_limit", P_K2_B1, k=mod(sH=INT32)),
    _case("size_limit_q_token_stride_2p31", "size_limit", P_B1H1, q=mod(sS=INT32)),
    # ---- size_limit: the slab base offset is 64-bit ----
    _case("admit_huge_batch_packed", "admit", P_BIG),
    _case(
        "admit_huge_batch_q_batch_stride_at_base_edge",
        "admit",
        P_BIG,
        q=mod(sB=BASE_EDGE_ADMIT),
    ),
    _case(
        "size_limit_huge_batch_q_batch_stride_past_base_edge",
        "size_limit",
        P_BIG,
        q=mod(sB=BASE_EDGE_REFUSED),
    ),
    _case(
        "size_limit_huge_batch_k_batch_stride_past_base_edge",
        "size_limit",
        P_BIG,
        k=mod(sB=BASE_EDGE_REFUSED),
    ),
    _case(
        "size_limit_huge_batch_o_batch_stride_past_base_edge",
        "size_limit",
        P_BIG,
        o=mod(sB=BASE_EDGE_REFUSED),
    ),
    _case(
        "size_limit_huge_batch_q_batch_stride_2p31_minus_8",
        "size_limit",
        P_BIG,
        q=mod(sB=INT32 - 8),
    ),
    _case("admit_huge_head_count_bhsd", "admit", P_BIGH, q=to_bhsd, o=to_bhsd),
    _case(
        "size_limit_huge_head_count_q_head_stride_2p31_minus_8",
        "size_limit",
        P_BIGH,
        q=then(to_bhsd, mod(sH=INT32 - 8)),
        o=to_bhsd,
    ),
    # ---- check order: two defects, the earlier check wins ----
    _case(
        "order_operand_o_beats_head_dim_q", "operand", o=mod(virtual=True), q=mod(sD=2)
    ),
    _case(
        "order_head_dim_v_beats_stride_alignment_q",
        "head_dim_stride",
        v=mod(sD=2),
        q=mod(sB=65792, sS=1028),
    ),
    _case(
        "order_stride_alignment_k_beats_pointer_alignment_q",
        "stride_alignment",
        k=mod(sB=16640, sS=260),
        q=mod(align=8),
    ),
    _case(
        "order_stride_alignment_k_beats_overlap_v",
        "stride_alignment",
        k=mod(sB=0),
        v=mod(sH=128, sS=128),
    ),
    _case(
        "order_stride_alignment_beats_size_limit_same_operand",
        "stride_alignment",
        P_BATCH,
        q=mod(sB=INT32 + 4),
    ),
    _case(
        "order_overlap_v_beats_pointer_alignment_q",
        "overlap",
        v=mod(sH=128, sS=128),
        q=mod(align=8),
    ),
    _case(
        "order_overlap_beats_pointer_alignment_same_operand",
        "overlap",
        k=mod(sH=128, sS=128, align=8),
    ),
    _case(
        "order_pointer_alignment_q_beats_size_limit_k",
        "pointer_alignment",
        P_KVEDGE,
        q=mod(align=8),
        k=mod(sS=KV_EDGE_REFUSED),
    ),
)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _dump_case(c: dict) -> str:
    """One case over six lines: the verdict row stays readable and diffs stay local."""
    j = json.dumps
    lines = [f'  {{"name": {j(c["name"])}, "problem": {j(c["problem"])},']
    lines += [f'   "{t}": {j(c[t])},' for t in TENSORS]
    tail = f'   "expect": {j(c["expect"])}'
    if "strides" in c:
        tail += f', "strides": {j(c["strides"])}'
    lines.append(tail + "}")
    return "\n".join(lines)


def _record(c: dict) -> dict:
    """The JSON form of a table row; an ``admit`` row also records the 12
    canonical strides (the one value taken from the module)."""
    row = {k: v for k, v in c.items()}
    if c["expect"] == "admit":
        raw = {
            f"stride_{t}_{axis}": c[t]["strides"][i]
            for t in TENSORS
            for i, axis in enumerate(("batch", "head", "token"))
        }
        canon = canonical_dense_strides(SimpleNamespace(**c["problem"]), raw)
        row["strides"] = [canon[name] for name in DENSE_STRIDE_NAMES]
    return row


def _build() -> dict:
    return {
        "_comment": (
            "Generated by library/tests/test_attention_dense_strided_admission.py "
            "--regenerate from its hand-written case table. Read by that test and "
            "by the C++ twin of dense_strides_admit. Do not edit by hand."
        ),
        "elem_bytes": _ELEM_BYTES,
        "stride_multiple": _STRIDE_MULTIPLE,
        "pointer_align": _POINTER_ALIGN,
        "slab_slack_tiles": _SLAB_SLACK_TILES,
        "reasons": _REASONS,
        "check_order": _CHECK_ORDER,
        "cases": [_record(c) for c in _CASES],
    }


def _render(data: dict) -> str:
    """Header keys one per line, then every case over six lines."""
    lines = ["{"]
    items = list(data.items())
    for i, (key, value) in enumerate(items):
        tail = "," if i + 1 < len(items) else ""
        if key == "cases":
            rows = ",\n".join(_dump_case(c) for c in value)
            lines.append(f' "{key}": [\n{rows}\n ]{tail}')
        else:
            lines.append(f' "{key}": {json.dumps(value)}{tail}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _load() -> dict:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def _spec(problem: dict) -> SimpleNamespace:
    return SimpleNamespace(**problem)


def _operands(case: dict) -> dict[str, DenseOperand]:
    return {
        t: DenseOperand(
            dims=tuple(case[t]["dims"]),
            strides=tuple(case[t]["strides"]),
            align=case[t]["align"],
            virtual=case[t]["virtual"],
            ragged=case[t]["ragged"],
        )
        for t in TENSORS
    }


def _hand_canonical(case: dict) -> list[int]:
    """The canonical strides, by hand: an extent-1 axis gives 0 (batch, head) or
    D (token); every other stride is kept."""
    out = []
    for t in TENSORS:
        B, H, S, D = case[t]["dims"]
        sB, sH, sS, _ = case[t]["strides"]
        out += [sB if B > 1 else 0, sH if H > 1 else 0, sS if S > 1 else D]
    return out


def _stale(path: str, test: str) -> str:
    """The message of a vectors file that the table no longer renders."""
    return (
        f"{path} is not what the case table of {test} renders. If the table or the "
        f"rules changed on purpose, run `python {test} --regenerate`, review the diff "
        "and mirror the change in the C++ twin of dense_strides_admit."
    )


# --------------------------------------------------------------------------- #
# a) the file on disk is the table
# --------------------------------------------------------------------------- #


def test_vectors_file_is_what_the_table_renders():
    want = _render(_build())
    got = VECTORS.read_text(encoding="utf-8")
    assert got == want, _stale(VECTORS.name, Path(__file__).name)


def test_header_constants_are_the_modules():
    assert ds.DENSE_ELEM_BYTES == _ELEM_BYTES == 2
    assert ds.DENSE_STRIDE_MULTIPLE == _STRIDE_MULTIPLE == 8
    assert ds.DENSE_POINTER_ALIGN == _POINTER_ALIGN == 16
    assert ds.DENSE_SLAB_SLACK_TILES == _SLAB_SLACK_TILES == 2
    assert list(DENSE_STRIDE_REASONS) == _REASONS
    data = _load()
    for key, want in (
        ("elem_bytes", _ELEM_BYTES),
        ("stride_multiple", _STRIDE_MULTIPLE),
        ("pointer_align", _POINTER_ALIGN),
        ("slab_slack_tiles", _SLAB_SLACK_TILES),
        ("reasons", _REASONS),
        ("check_order", _CHECK_ORDER),
    ):
        assert data[key] == want, key


# --------------------------------------------------------------------------- #
# b) every row
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_case_gets_its_hand_written_verdict(case):
    verdict = dense_strides_admit(_spec(case["problem"]), _operands(case))
    if case["expect"] == "admit":
        assert verdict.ok, (verdict.reason, verdict.detail)
        got = [verdict.strides[n] for n in DENSE_STRIDE_NAMES]
        assert got == _hand_canonical(case)
        assert got == _record(case)["strides"]
    else:
        assert not verdict.ok, "admitted"
        assert verdict.reason == case["expect"], verdict.detail
        assert verdict.detail
        assert verdict.strides is None


def test_json_cases_give_the_recorded_verdicts():
    """What the C++ twin does: read the file, rebuild each row, compare."""
    cases = _load()["cases"]
    assert len(cases) == len(_CASES)
    for c in cases:
        verdict = dense_strides_admit(_spec(c["problem"]), _operands(c))
        assert (verdict.ok, verdict.reason) == (
            c["expect"] == "admit",
            "" if c["expect"] == "admit" else c["expect"],
        ), c["name"]
        if c["expect"] == "admit":
            assert [verdict.strides[n] for n in DENSE_STRIDE_NAMES] == c["strides"]
        else:
            assert "strides" not in c


def test_admit_rows_record_the_hand_canonical_strides():
    admits = [c for c in _CASES if c["expect"] == "admit"]
    for c in admits:
        assert _record(c)["strides"] == _hand_canonical(c), c["name"]
    named = {c["name"]: c for c in admits}
    # An extent-1 axis: garbage in, 0 (batch, head) or D (token) out.
    mqa = named["admit_mqa_kv_head_stride_12345"]
    assert _record(mqa)["strides"][4] == _record(mqa)["strides"][7] == 0
    b1 = named["admit_batch1_all_batch_strides_garbage"]
    assert _record(b1)["strides"][0::3] == [0, 0, 0, 0]
    dec = named["admit_decode_seqlen_q1_garbage_token_stride"]
    assert _record(dec)["strides"][2] == _record(dec)["strides"][11] == 128
    kv1 = named["admit_seqlen_kv1_garbage_token_stride"]
    assert _record(kv1)["strides"][5] == _record(kv1)["strides"][8] == 128
    # An axis of extent > 1 keeps its stride.
    assert _record(named["admit_bhsd_all_four"])["strides"] == [
        65536,
        8192,
        128,
        16384,
        8192,
        128,
        16384,
        8192,
        128,
        65536,
        8192,
        128,
    ]


def test_table_is_well_formed_and_covers_every_reason():
    names = [c["name"] for c in _CASES]
    assert len(set(names)) == len(names)
    assert all(_NAME_RE.match(n) for n in names)
    assert len(_CASES) >= 45
    for c in _CASES:
        assert list(c) == ["name", "problem", *TENSORS, "expect"], c["name"]
        assert c["expect"] == "admit" or c["expect"] in DENSE_STRIDE_REASONS
        assert list(c["problem"]) == [
            "batch",
            "seqlen_q",
            "seqlen_kv",
            "num_query_heads",
            "num_kv_heads",
            "head_size",
            "block_n",
        ]
        for t in TENSORS:
            op = c[t]
            assert list(op) == ["dims", "strides", "align", "virtual", "ragged"]
            assert op["align"] > 0 and op["align"] & (op["align"] - 1) == 0
            assert all(type(n) is int for n in op["dims"] + op["strides"])
    counts = {r: sum(c["expect"] == r for c in _CASES) for r in DENSE_STRIDE_REASONS}
    assert all(n >= 2 for n in counts.values()), counts
    assert sum(c["expect"] == "admit" for c in _CASES) >= 10
    assert [r for r in DENSE_STRIDE_REASONS if counts[r] == 0] == []


def test_edge_values_are_the_hand_derived_ones():
    """The size_limit edges, by plain arithmetic (see the comments above the table)."""
    # K/V reach: R = 64 + 2*64 = 192 rows; 2*((R-1)*sS + D - 1) < 2**31.
    assert 64 + 2 * 64 == 192
    assert 2 * (191 * KV_EDGE_ADMIT + 128 - 1) < 2**31
    assert 2 * (191 * KV_EDGE_REFUSED + 128 - 1) >= 2**31
    assert KV_EDGE_REFUSED == KV_EDGE_ADMIT + 8 and KV_EDGE_ADMIT % 8 == 0
    # The slack: sS passes without it (63 rows of stride) and fails with it.
    assert 2 * (63 * KV_SLACK_ONLY_REFUSED + 127) < 2**31
    assert 2 * (191 * KV_SLACK_ONLY_REFUSED + 127) >= 2**31
    assert KV_SLACK_ONLY_REFUSED % 8 == 0
    # seqlen_kv = 65 pads to 128 rows: R = 128 + 128 = 256.
    assert 2 * (255 * KV_PADDED_EDGE_ADMIT + 127) < 2**31
    assert 2 * (255 * KV_PADDED_EDGE_REFUSED + 127) >= 2**31
    assert KV_PADDED_EDGE_REFUSED == KV_PADDED_EDGE_ADMIT + 8
    assert 2 * (255 * KV_EDGE_ADMIT + 127) >= 2**31
    # block_n = 128, seqlen_kv = 64: R = 128 + 256 = 384.
    assert 2 * (383 * KV_EDGE_ADMIT + 127) >= 2**31
    # Reach of exactly 2**31 bytes (the last load starts at 2**31 - 2): admit.
    assert KV_REACH_EXACT_SEQLEN == 2**23 - 128 and KV_REACH_EXACT_SEQLEN % 64 == 0
    assert KV_REACH_EXACT_SEQLEN + 2 * 64 == 2**23
    assert 2 * ((2**23 - 1) * 128 + 128) == 2**31
    assert 2 * ((2**23 - 1) * 128 + 128 - 1) < 2**31
    # One row more pads to 8388544 rows: R = 8388544 + 128 = 8388672.
    assert 2 * ((8388672 - 1) * 128 + 128 - 1) >= 2**31
    # Q/O: (256 - 1)*sS + 128 < 2**31, exactly at 8421504.
    assert 255 * Q_EDGE_ADMIT + 128 == 2147481608 < 2**31
    assert 255 * Q_EDGE_REFUSED + 128 == 2**31
    assert Q_EDGE_REFUSED == Q_EDGE_ADMIT + 8 and Q_EDGE_ADMIT % 8 == 0
    # 64-bit base: B = 2**32, S = 8, D = 64, sS = 64: slab = 2*(7*64 + 64) = 1024 B.
    slab = 2 * (7 * 64 + 64)
    assert slab == 1024
    assert 2 * (BIG_B - 1) * BASE_EDGE_ADMIT + slab < 2**63
    assert 2 * (BIG_B - 1) * BASE_EDGE_REFUSED + slab >= 2**63
    assert BASE_EDGE_REFUSED == BASE_EDGE_ADMIT + 8
    assert 2 * (BIG_B - 1) * (INT32 - 8) >= 2**63
    # ... and with BIG_B the base is nowhere near 2**63 for packed strides:
    assert 2 * (BIG_B - 1) * (8 * 1 * 64) < 2**63


def test_first_failing_operand_is_named_in_the_detail():
    """Inside a check the operands go q, k, v, o: with the same defect on two
    operands the detail names the first of them."""
    defects = {
        "operand": mod(virtual=True),
        "head_dim_stride": mod(sD=2),
        "stride_alignment": mod(sB=0),
        "overlap": mod(sB=8),
        "pointer_alignment": mod(align=8),
        "size_limit": mod(sB=INT32),
    }
    assert list(defects) == list(DENSE_STRIDE_REASONS)
    for reason, defect in defects.items():
        for first, others in (("q", ("o",)), ("k", ("v", "o")), ("v", ("o",))):
            row = _case("x", reason, **{t: defect for t in (first, *others)})
            verdict = dense_strides_admit(_spec(row["problem"]), _operands(row))
            assert verdict.reason == reason, (reason, first, verdict.detail)
            assert verdict.detail.split()[0] == first, (reason, verdict.detail)


def test_a_rank_decline_says_which_count_is_wrong():
    """Wrong dims and wrong strides give different details, the C++ twin's."""
    for t, override, detail in (
        ("q", mod(dims=[2, 8, 64], strides=[65536, 128, 1024]), "q has rank 3, not 4"),
        ("v", mod(strides=[16384, 128, 256]), "v has 3 strides for its 4 dims, not 4"),
        (
            "k",
            mod(dims=[2, 2, 64, 128, 1], strides=[16384, 128, 256, 1, 1]),
            "k has rank 5, not 4",
        ),
    ):
        row = _case("x", "operand", **{t: override})
        verdict = dense_strides_admit(_spec(row["problem"]), _operands(row))
        assert (verdict.reason, verdict.detail) == ("operand", detail)


# --------------------------------------------------------------------------- #
# c) names and packed strides of real specs
# --------------------------------------------------------------------------- #


def test_stride_names_are_the_twelve_in_order():
    assert DENSE_STRIDE_NAMES == (
        "stride_q_batch",
        "stride_q_head",
        "stride_q_token",
        "stride_k_batch",
        "stride_k_head",
        "stride_k_token",
        "stride_v_batch",
        "stride_v_head",
        "stride_v_token",
        "stride_o_batch",
        "stride_o_head",
        "stride_o_token",
    )


def _named(q, k, v, o) -> dict[str, int]:
    out = {}
    for t, triple in zip(TENSORS, (q, k, v, o)):
        out.update(
            dict(zip((f"stride_{t}_{a}" for a in ("batch", "head", "token")), triple))
        )
    return out


@pytest.mark.parametrize("cls", [Gfx950AttentionDenseSpec, Gfx942AttentionDenseSpec])
def test_packed_strides_of_a_real_spec_are_contiguous_bshd(cls):
    spec = cls(
        batch=2,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=128,
    )
    # Q/O [2, 512, 8, 128]: batch 512*8*128, head 128, token 8*128.
    # K/V [2, 512, 2, 128]: batch 512*2*128, head 128, token 2*128.
    want = _named(
        (524288, 128, 1024),
        (131072, 128, 256),
        (131072, 128, 256),
        (524288, 128, 1024),
    )
    got = ds.packed_dense_strides(spec)
    assert got == want
    assert list(got) == list(DENSE_STRIDE_NAMES)
    assert dense_operand_dims(spec, "q") == (2, 8, 512, 128)
    assert dense_operand_dims(spec, "k") == (2, 2, 512, 128)
    assert packed_dense_operand(spec, "k") == DenseOperand(
        dims=(2, 2, 512, 128), strides=(131072, 128, 256, 1)
    )
    ops = {t: packed_dense_operand(spec, t) for t in TENSORS}
    verdict = dense_strides_admit(spec, ops)
    assert verdict.ok, verdict.detail
    assert dict(verdict.strides) == want


# --------------------------------------------------------------------------- #
# d) views
# --------------------------------------------------------------------------- #


class _View:
    """A tensor view: shape, stride(), data_ptr(), is_contiguous()."""

    def __init__(self, shape, strides, ptr=0x1000):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self._ptr = ptr

    def stride(self):
        return self._strides

    def data_ptr(self):
        return self._ptr

    def is_contiguous(self):
        expected = 1
        for size, stride in reversed(list(zip(self.shape, self._strides))):
            if size != 1:
                if stride != expected:
                    return False
                expected *= size
        return True


class _StridesOnly:
    """The view of a caller without torch: shape, stride() and data_ptr(), and no
    is_contiguous()."""

    def __init__(self, shape, strides, ptr):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self._ptr = ptr

    def stride(self):
        return self._strides

    def data_ptr(self):
        return self._ptr


def _bshd_view(p, t, ptr=0x1000):
    B, H, S, D = _dims(p, t)
    return _View((B, S, H, D), (S * H * D, H * D, D, 1), ptr)


def _bhsd_view(p, t, ptr=0x1000):
    """Storage [B, H, S, D] seen as the logical [B, S, H, D] (a permute)."""
    B, H, S, D = _dims(p, t)
    return _View((B, S, H, D), (H * S * D, D, S * D, 1), ptr)


# One sentinel base per tensor, 256 MiB apart, so that the tensors of a launch never
# share a byte (a P0 tensor is at most 256 KiB): the launch refuses an O that meets
# Q, K or V.
_BASES = {"q": 0x1000_0000, "k": 0x2000_0000, "v": 0x3000_0000, "o": 0x4000_0000}

# P0 = batch 2, Sq = Skv = 64, Hq = 8, Hkv = 2, D = 128, by hand.
_P0_BSHD = _named(
    (65536, 128, 1024), (16384, 128, 256), (16384, 128, 256), (65536, 128, 1024)
)
_P0_BHSD = _named(
    (65536, 8192, 128), (16384, 8192, 128), (16384, 8192, 128), (65536, 8192, 128)
)


def test_operand_strides_of_bshd_and_bhsd_views():
    """A [B, S, H, D] view becomes (B, H, S, D) dims and strides, as given."""
    spec = _spec(P0)
    for make, want in ((_bshd_view, _P0_BSHD), (_bhsd_view, _P0_BHSD)):
        for t in TENSORS:
            op = dense_operand_from_view(make(P0, t))
            assert op.dims == dense_operand_dims(spec, t)
            triple = tuple(want[f"stride_{t}_{a}"] for a in ("batch", "head", "token"))
            assert op.strides == triple + (1,), (make.__name__, t)
    # Not validated and not canonicalized there: the predicate declines these.
    odd = dense_operand_from_view(_View((2, 64, 8, 128), (-3, 7, 0, 1)))
    assert odd.strides == (-3, 0, 7, 1)
    ops = {t: packed_dense_operand(spec, t) for t in TENSORS}
    verdict = dense_strides_admit(spec, dict(ops, q=odd))
    assert (verdict.ok, verdict.reason) == (False, "stride_alignment")


def test_operand_from_a_view():
    q = _bhsd_view(P0, "q")
    assert dense_operand_from_view(q) == DenseOperand(
        dims=(2, 8, 64, 128), strides=(65536, 8192, 128, 1), align=256
    )
    # Alignment: the largest power of two dividing the address, at most 256.
    for ptr, align in (
        (0x1010, 16),
        (0x1008, 8),
        (0x1002, 2),
        (0x1001, 1),
        (0, 256),
        (0x300, 256),
        (0x1200, 256),
    ):
        assert dense_operand_from_view(_bshd_view(P0, "q", ptr)).align == align, hex(
            ptr
        )
    # No data_ptr(): a DeviceMem's ptr() gives the address; with neither, the base
    # is assumed to be 16-byte aligned.
    shape, strides = (2, 64, 8, 128), (65536, 1024, 128, 1)
    bare = SimpleNamespace(shape=shape, stride=lambda: strides)
    assert dense_operand_from_view(bare).align == 16
    mem = SimpleNamespace(shape=shape, stride=lambda: strides, ptr=lambda: 0x1008)
    assert dense_operand_from_view(mem).align == 8
    # Another rank keeps its own dims, which the predicate then declines.
    flat = _View((2, 64, 1024), (65536, 1024, 1))
    op = dense_operand_from_view(flat)
    assert (op.dims, op.strides) == ((2, 64, 1024), (65536, 1024, 1))
    ops = {t: packed_dense_operand(_spec(P0), t) for t in TENSORS}
    verdict = dense_strides_admit(_spec(P0), dict(ops, q=op))
    assert (verdict.ok, verdict.reason) == (False, "operand")


# --------------------------------------------------------------------------- #
# e) what a runner launches with
# --------------------------------------------------------------------------- #


def _launch_spec(p=None, **extra):
    fields = dict(_prob() if p is None else p, kv_segments=1, varlen=False, paged=False)
    fields.update(extra)
    return SimpleNamespace(**fields)


def _views(p, make=_bshd_view, **replace):
    """The four views of problem ``p``, each at its own sentinel base, with
    ``replace`` swapped in."""
    views = {t: make(p, t, _BASES[t]) for t in TENSORS}
    views.update(replace)
    return views


@pytest.mark.parametrize("serves", [False, True])
def test_launch_packed_views_give_the_packed_values(serves):
    spec = _launch_spec()
    got = admit_dense_launch_strides(spec, **_views(P0), serves_strides=serves)
    assert got == _P0_BSHD
    assert got == packed_launch_strides(spec)
    # Raw addresses (no .stride()) are taken as packed.
    bare = admit_dense_launch_strides(spec, **_BASES, serves_strides=serves)
    assert bare == _P0_BSHD


def test_launch_takes_an_object_without_stride_as_packed():
    """A raw address, an object with only an address, or one with no address at
    all is a packed operand; the 12 values come back in name order."""
    got = admit_dense_launch_strides(
        _launch_spec(),
        q=_BASES["q"],
        k=_bshd_view(P0, "k", _BASES["k"]),
        v=object(),
        o=SimpleNamespace(data_ptr=lambda: _BASES["o"]),
        serves_strides=False,
    )
    assert got == _P0_BSHD
    assert list(got) == list(DENSE_STRIDE_NAMES)


def test_launch_refuses_a_view_that_is_not_4d():
    flat = _View((2, 64, 1024), (65536, 1024, 1), _BASES["q"])
    with pytest.raises(ValueError, match=r"declined \(operand\): q has rank 3"):
        admit_dense_launch_strides(
            _launch_spec(), **_views(P0, q=flat), serves_strides=True
        )


def test_launch_a_batch_of_one_canonicalizes_the_unused_stride():
    p = _prob(batch=1)
    spec = _launch_spec(p)
    views = _views(p, q=_View((1, 64, 8, 128), (99999, 1024, 128, 1), _BASES["q"]))
    got = admit_dense_launch_strides(spec, **views, serves_strides=False)
    assert got["stride_q_batch"] == 0
    assert got["stride_k_batch"] == got["stride_v_batch"] == got["stride_o_batch"] == 0
    assert (got["stride_q_head"], got["stride_q_token"]) == (128, 1024)


def test_launch_bhsd_views_need_a_body_that_serves_strides():
    spec = _launch_spec()
    views = _views(P0, _bhsd_view)
    with pytest.raises(ValueError, match="not packed"):
        admit_dense_launch_strides(spec, **views, serves_strides=False)
    got = admit_dense_launch_strides(spec, **views, serves_strides=True)
    assert got == _P0_BHSD
    # One BHSD view is enough to refuse a body that reads packed storage only.
    one = _views(P0, k=_bhsd_view(P0, "k", _BASES["k"]))
    with pytest.raises(ValueError, match="k is not packed"):
        admit_dense_launch_strides(spec, **one, serves_strides=False)


def test_launch_kv_split_needs_a_packed_o():
    spec = _launch_spec(kv_segments=2)
    views = _views(P0, _bhsd_view)
    with pytest.raises(ValueError, match="combine kernel"):
        admit_dense_launch_strides(spec, **views, serves_strides=True)
    # Q, K and V may be strided; O packed.
    ok = _views(P0, _bhsd_view, o=_bshd_view(P0, "o", _BASES["o"]))
    got = admit_dense_launch_strides(spec, **ok, serves_strides=True)
    assert got == _named(
        (65536, 8192, 128), (16384, 8192, 128), (16384, 8192, 128), (65536, 128, 1024)
    )
    # A single segment writes O itself.
    one = _launch_spec(kv_segments=1)
    assert admit_dense_launch_strides(one, **views, serves_strides=True) == _P0_BHSD


def test_launch_a_declined_view_raises_with_its_reason():
    spec = _launch_spec()
    views = _views(P0, k=_bshd_view(P0, "k", _BASES["k"] + 8))
    with pytest.raises(ValueError, match="pointer_alignment"):
        admit_dense_launch_strides(spec, **views, serves_strides=True)
    views = _views(P0, v=_View((2, 64, 2, 128), (16384, 256, 128, 2), _BASES["v"]))
    with pytest.raises(ValueError, match="head_dim_stride"):
        admit_dense_launch_strides(spec, **views, serves_strides=False)


@pytest.mark.parametrize("serves", [False, True])
def test_launch_measures_the_alignment_of_a_raw_address(serves):
    """A raw address, or an object with only data_ptr() or ptr(), is checked at
    the alignment of its address, not assumed to be 16-byte aligned."""
    spec = _launch_spec()
    for t, misaligned, align in (
        ("q", _BASES["q"] + 2, 2),
        ("k", SimpleNamespace(data_ptr=lambda: _BASES["k"] + 8), 8),
        ("v", SimpleNamespace(ptr=lambda: _BASES["v"] + 4), 4),
        ("o", _BASES["o"] + 1, 1),
    ):
        views = dict(_BASES, **{t: misaligned})
        with pytest.raises(
            ValueError, match=rf"\(pointer_alignment\): {t} base is {align}-byte"
        ):
            admit_dense_launch_strides(spec, **views, serves_strides=serves)
    # 16-byte aligned addresses, in every form, are admitted.
    aligned = {
        "q": _BASES["q"] + 16,
        "k": SimpleNamespace(data_ptr=lambda: _BASES["k"] + 32),
        "v": SimpleNamespace(ptr=lambda: _BASES["v"] + 48),
        "o": _BASES["o"],
    }
    got = admit_dense_launch_strides(spec, **aligned, serves_strides=serves)
    assert got == _P0_BSHD


@pytest.mark.parametrize("flag", ["varlen", "paged"])
def test_launch_varlen_and_paged_read_contiguous_storage_only(flag):
    spec = _launch_spec(**{flag: True})
    strided = _views(P0, _bhsd_view)
    with pytest.raises(ValueError, match="not packed"):
        admit_dense_launch_strides(spec, **strided, serves_strides=True)
    # Contiguous views, and raw addresses: the packed values, even for a view the
    # predicate would decline (packed [total_tokens, H, D] has rank 3; the kernel
    # does not read the strides).
    tokens = _View((128, 8, 128), (1024, 128, 1), _BASES["q"])
    for views in (_views(P0), dict(_BASES), _views(P0, q=tokens)):
        got = admit_dense_launch_strides(spec, **views, serves_strides=False)
        assert got == packed_launch_strides(spec) == _P0_BSHD


@pytest.mark.parametrize("flag", ["varlen", "paged"])
def test_launch_varlen_and_paged_check_the_alignment_of_every_address(flag):
    """The varlen and paged bodies claim 16-byte aligned Q/K/V/O pointers too: a
    view, a raw address, or an object with only data_ptr() or ptr() is checked at
    the alignment of its address, as on every other body. Address 0 and an
    object with no address are not refused."""
    spec = _launch_spec(**{flag: True})
    for t, misaligned, align in (
        ("q", _BASES["q"] + 2, 2),
        ("k", SimpleNamespace(data_ptr=lambda: _BASES["k"] + 8), 8),
        ("v", SimpleNamespace(ptr=lambda: _BASES["v"] + 4), 4),
        ("o", _BASES["o"] + 1, 1),
        ("q", _bshd_view(P0, "q", _BASES["q"] + 8), 8),
        ("o", _View((128, 8, 128), (1024, 128, 1), _BASES["o"] + 4), 4),
    ):
        for views in (dict(_BASES), _views(P0)):
            views[t] = misaligned
            with pytest.raises(
                ValueError,
                match=rf"declined \(pointer_alignment\): {t} base is {align}-byte "
                r"aligned, needs 16",
            ):
                admit_dense_launch_strides(spec, **views, serves_strides=False)
    for views in (
        {
            "q": _BASES["q"] + 16,
            "k": SimpleNamespace(data_ptr=lambda: _BASES["k"] + 32),
            "v": SimpleNamespace(ptr=lambda: _BASES["v"] + 48),
            "o": _bshd_view(P0, "o", _BASES["o"] + 64),
        },
        {"q": 0, "k": object(), "v": _BASES["v"], "o": _BASES["o"]},
    ):
        got = admit_dense_launch_strides(spec, **views, serves_strides=False)
        assert got == _P0_BSHD


@pytest.mark.parametrize("flag", ["varlen", "paged"])
def test_launch_varlen_and_paged_judge_a_view_by_its_strides(flag):
    """A view with shape and stride() but no is_contiguous() (a caller without
    torch) is packed only when its strides are the row-major ones of its shape;
    an extent-1 axis may have any stride."""
    spec = _launch_spec(**{flag: True})
    B, S, H, D = 2, 64, 8, 128
    base = _BASES["q"]
    for q in (
        _StridesOnly((B, S, H, D), (H * S * D, D, S * D, 1), base),  # [B, H, S, D]
        _StridesOnly((100, H, D), (H * D + 64, D, 1), base),  # padded tokens
    ):
        with pytest.raises(ValueError, match="q is not packed"):
            admit_dense_launch_strides(spec, **_views(P0, q=q), serves_strides=False)
    for q in (
        _StridesOnly((B, S, H, D), (S * H * D, H * D, D, 1), base),
        _StridesOnly((100, H, D), (H * D, D, 1), base),  # varlen [tokens, H, D]
        _StridesOnly((1, S, H, D), (12345, H * D, D, 1), base),
    ):
        got = admit_dense_launch_strides(spec, **_views(P0, q=q), serves_strides=False)
        assert got == _P0_BSHD


@pytest.mark.parametrize("varlen", [False, True])
def test_launch_refuses_a_non_contiguous_view_without_strides(varlen):
    """A view that says it is not contiguous but has no stride() has a layout
    that cannot be read: refused on every body. One that says it is contiguous
    is packed."""
    spec = _launch_spec(varlen=varlen)
    gap = SimpleNamespace(is_contiguous=lambda: False, data_ptr=lambda: _BASES["v"])
    with pytest.raises(ValueError, match="v is not packed"):
        admit_dense_launch_strides(spec, **_views(P0, v=gap), serves_strides=True)
    dense = SimpleNamespace(is_contiguous=lambda: True, data_ptr=lambda: _BASES["v"])
    got = admit_dense_launch_strides(spec, **_views(P0, v=dense), serves_strides=True)
    assert got == _P0_BSHD


# --------------------------------------------------------------------------- #
# e2) O may not share storage with Q, K or V
# --------------------------------------------------------------------------- #


def _fused_qkv(p: dict, base: int) -> dict[str, _View]:
    """Q, K and V as views of one ``[B, S, Hq + 2 Hkv, D]`` buffer at ``base``."""
    B, Hq, S, D = _dims(p, "q")
    Hkv = p["num_kv_heads"]
    width = Hq + 2 * Hkv
    strides = (S * width * D, width * D, D, 1)
    return {
        "q": _View((B, S, Hq, D), strides, base),
        "k": _View((B, S, Hkv, D), strides, base + _ELEM_BYTES * Hq * D),
        "v": _View((B, S, Hkv, D), strides, base + _ELEM_BYTES * (Hq + Hkv) * D),
    }


def test_launch_refuses_an_o_inside_the_fused_qkv_buffer():
    spec = _launch_spec()
    qkv = _fused_qkv(P0, _BASES["q"])
    # O in its own buffer: admitted, Q, K and V at the fused strides
    # (batch 64*12*128, head 128, token 12*128).
    o = _bshd_view(P0, "o", _BASES["o"])
    got = admit_dense_launch_strides(spec, **qkv, o=o, serves_strides=True)
    fused = (98304, 128, 1536)
    assert got == _named(fused, fused, fused, (65536, 128, 1024))
    # O written over Q (in place), or carved out of the same buffer at V's offset:
    # refused before anything launches.
    B, Hq, S, D = _dims(P0, "o")
    carved = _View((B, S, Hq, D), qkv["q"].stride(), qkv["v"].data_ptr())
    for o in (qkv["q"], carved):
        with pytest.raises(ValueError, match="o overlaps q:"):
            admit_dense_launch_strides(spec, **qkv, o=o, serves_strides=True)


@pytest.mark.parametrize("serves", [False, True])
def test_launch_refuses_an_o_that_shares_bytes_with_an_input(serves):
    """Packed tensors: an O at the storage of Q, K or V is refused, naming that
    input; an O that starts right after an input's last byte is admitted."""
    spec = _launch_spec()
    for t in ("q", "k", "v"):
        views = _views(P0, o=_bshd_view(P0, "o", _BASES[t]))
        with pytest.raises(ValueError, match=f"o overlaps {t}:"):
            admit_dense_launch_strides(spec, **views, serves_strides=serves)
    # Q is 2*64*8*128 elements of 2 bytes: bytes [base, base + 0x40000).
    q_end = _BASES["q"] + 2 * 64 * 8 * 128 * _ELEM_BYTES
    assert q_end == _BASES["q"] + 0x40000
    touching = _views(P0, o=_bshd_view(P0, "o", q_end))
    got = admit_dense_launch_strides(spec, **touching, serves_strides=serves)
    assert got == _P0_BSHD
    inside = _views(P0, o=_bshd_view(P0, "o", q_end - 16))
    with pytest.raises(ValueError, match="o overlaps q:"):
        admit_dense_launch_strides(spec, **inside, serves_strides=serves)
    # Raw addresses are spanned as packed tensors.
    raw = dict(_BASES, o=_BASES["k"] + 0x100)
    with pytest.raises(ValueError, match="o overlaps k:"):
        admit_dense_launch_strides(spec, **raw, serves_strides=serves)


@pytest.mark.parametrize("flag", ["varlen", "paged"])
def test_launch_varlen_and_paged_refuse_an_o_over_an_input(flag):
    spec = _launch_spec(**{flag: True})
    views = _views(P0, o=_bshd_view(P0, "o", _BASES["v"]))
    with pytest.raises(ValueError, match="o overlaps v:"):
        admit_dense_launch_strides(spec, **views, serves_strides=False)


def test_launch_does_not_span_a_zero_or_unknown_address():
    """Synthetic launches keep working: a base of 0, or a tensor with no address,
    is not checked against the others."""
    spec = _launch_spec()
    no_address = SimpleNamespace(
        shape=(2, 64, 8, 128), stride=lambda: (65536, 1024, 128, 1)
    )
    for o in (0, no_address):
        got = admit_dense_launch_strides(
            spec, q=0, k=_BASES["k"], v=_BASES["v"], o=o, serves_strides=False
        )
        assert got == _P0_BSHD


# --------------------------------------------------------------------------- #
# f) one slab rule
# --------------------------------------------------------------------------- #


def test_slab_bytes_int_form():
    f = dense_slab_bytes
    assert f(0, 256, head_size=128, elem_bytes=2) == 0
    assert f(-3, 256, head_size=128, elem_bytes=2) == 0
    assert f(1, 256, head_size=128, elem_bytes=2) == 2 * 128
    assert f(1, 99999, head_size=128, elem_bytes=2) == 2 * 128
    # 3 rows of stride 256: 2 strides + the last row of D elements, 2 bytes each.
    assert f(3, 256, head_size=128, elem_bytes=2) == 2 * (2 * 256 + 128) == 1280
    assert f(3, 256, head_size=128, elem_bytes=1) == 2 * 256 + 128 == 640
    assert f(10**6, 2**20, head_size=64, elem_bytes=2) == 2 * ((10**6 - 1) * 2**20 + 64)


def test_kv_reach_rows():
    assert dense_kv_reach_rows(64, 64) == 64 + 2 * 64 == 192
    assert dense_kv_reach_rows(65, 64) == 128 + 2 * 64 == 256
    assert dense_kv_reach_rows(1, 64) == 64 + 128 == 192
    assert dense_kv_reach_rows(64, 128) == 128 + 2 * 128 == 384
    assert dense_kv_reach_rows(100, 32) == 128 + 2 * 32 == 192


def _packed_p0_operands() -> dict[str, DenseOperand]:
    case = _CASES[0]
    assert case["name"] == "admit_packed_bshd_gqa_d128_batch2"
    return _operands(case)


def test_predicate_goes_through_the_shared_slab_function(monkeypatch):
    calls = []
    real = ds.dense_slab_bytes

    def spy(rows, token_stride, *, head_size, elem_bytes, b=None):
        calls.append((rows, token_stride, head_size, elem_bytes))
        return real(rows, token_stride, head_size=head_size, elem_bytes=elem_bytes, b=b)

    monkeypatch.setattr(ds, "dense_slab_bytes", spy)
    verdict = dense_strides_admit(_spec(P0), _packed_p0_operands())
    assert verdict.ok
    # K and V: the reach rows (64 + 2*64 = 192), token stride 2*128, 2 bytes.
    assert [c for c in calls if c[0] == 192] == [(192, 256, 128, 2)] * 2
    # Q and O: seqlen_q = 64 rows of token stride 8*128, counted in elements.
    assert calls.count((64, 1024, 128, 1)) == 2


def test_slab_function_decides_the_verdict(monkeypatch):
    """The reach is not re-derived anywhere else: a different slab function gives
    a different verdict."""
    ops = _packed_p0_operands()
    real = ds.dense_slab_bytes
    # Every slab is 2**31 elements: q, the first operand checked, declines.
    monkeypatch.setattr(ds, "dense_slab_bytes", lambda *a, **k: 2**31)
    verdict = dense_strides_admit(_spec(P0), ops)
    assert (verdict.ok, verdict.reason) == (False, "size_limit")
    assert verdict.detail.split()[0] == "q"
    # Only the K/V reach (192 rows) is too big: k declines.
    monkeypatch.setattr(
        ds,
        "dense_slab_bytes",
        lambda rows, *a, **k: 2**32 if rows == 192 else real(rows, *a, **k),
    )
    verdict = dense_strides_admit(_spec(P0), ops)
    assert (verdict.ok, verdict.reason) == (False, "size_limit")
    assert verdict.detail.split()[0] == "k"


def test_slab_function_that_returns_zero_admits_an_edge_row(monkeypatch):
    case = next(
        c for c in _CASES if c["name"] == "size_limit_k_token_stride_past_reach_edge"
    )
    monkeypatch.setattr(ds, "dense_slab_bytes", lambda *a, **k: 0)
    assert dense_strides_admit(_spec(case["problem"]), _operands(case)).ok


# --------------------------------------------------------------------------- #
# g) the IR form
# --------------------------------------------------------------------------- #


def test_slab_bytes_ir_form_returns_an_ir_value():
    from rocke.core.ir import I32, IRBuilder, Value

    b = IRBuilder("t")
    n = b.param("n", I32)
    s = b.param("s", I32)
    out = dense_slab_bytes(n, s, head_size=128, elem_bytes=2, b=b)
    assert isinstance(out, Value) and not isinstance(out, int)


def _run_ir(b, result, **bindings: int) -> int:
    """Evaluate the straight-line integer IR of builder b (the ops the slab
    function emits) with the named parameters bound; returns result."""
    env = dict(bindings)
    for op in b.kernel.body.ops:
        args = [env[v.name] for v in op.operands]
        out = op.results[0].name
        if op.name == "arith.constant":
            env[out] = op.attrs["value"]
        elif op.name == "arith.add":
            env[out] = args[0] + args[1]
        elif op.name == "arith.sub":
            env[out] = args[0] - args[1]
        elif op.name == "arith.mul":
            env[out] = args[0] * args[1]
        elif op.name == "arith.cmp" and op.attrs["pred"] == "gt":
            env[out] = int(args[0] > args[1])
        elif op.name == "arith.select":
            env[out] = args[1] if args[0] else args[2]
        else:
            raise AssertionError(f"the slab IR has an op this test cannot run: {op}")
    return env[result.name]


def test_slab_bytes_ir_form_agrees_with_the_int_form():
    from rocke.core.ir import I32, IRBuilder

    b = IRBuilder("t")
    n = b.param("n", I32)
    s = b.param("s", I32)
    out = dense_slab_bytes(n, s, head_size=128, elem_bytes=2, b=b)
    for rows in (-1, 0, 1, 2, 3, 64, 192):
        for stride in (128, 256, 1024, 5000):
            ir = _run_ir(b, out, **{n.name: rows, s.name: stride})
            want = dense_slab_bytes(rows, stride, head_size=128, elem_bytes=2)
            assert ir == want, (rows, stride)
    # By hand: rows 3, stride 256, D 128, 2 bytes: 2 * (2*256 + 128) = 1280; rows 0: 0.
    assert _run_ir(b, out, **{n.name: 3, s.name: 256}) == 1280
    assert _run_ir(b, out, **{n.name: 0, s.name: 256}) == 0


def test_slab_pointer_zero_extends_before_it_multiplies():
    """The one slab-pointer emitter of both arches: each i32 operand is
    zero-extended to i64 before its multiply, so the offset cannot wrap at 2^31,
    then one scale by the element size and one pointer add. The op order is the
    one the IR goldens of both arches pin."""
    from rocke.core.ir import F16, I32, I64, IRBuilder, PtrType

    b = IRBuilder("t")
    ptr = b.param("ptr", PtrType(F16, "global"))
    batch, head = b.param("bt", I32), b.param("h", I32)
    strides = declare_dense_stride_params(b)
    out = emit_dense_slab_ptr(b, strides, ptr, batch, head, "k")
    ops = b.kernel.body.ops
    assert [op.name for op in ops] == [
        "arith.zext",
        "arith.zext",
        "arith.mul",
        "arith.zext",
        "arith.zext",
        "arith.mul",
        "arith.add",
        "arith.constant",
        "arith.mul",
        "tile.global_ptr_add",
    ]
    zext_inputs = [op.operands[0].name for op in ops if op.name == "arith.zext"]
    assert zext_inputs == ["%bt", "%stride_k_batch", "%h", "%stride_k_head"]
    assert all(op.results[0].type == I64 for op in ops[:-1])
    assert ops[7].attrs["value"] == ds.DENSE_ELEM_BYTES == 2
    assert [v.name for v in ops[-1].operands] == ["%ptr", ops[8].results[0].name]
    assert ops[-1].results[0] is out


def test_declare_params_adds_the_twelve_strides_last():
    from rocke.core.ir import I32, IRBuilder, Value

    b = IRBuilder("t")
    b.param("n", I32)
    params = declare_dense_stride_params(b)
    assert list(params) == list(DENSE_STRIDE_NAMES)
    assert all(isinstance(v, Value) for v in params.values())
    tail = b.kernel.params[-12:]
    assert [p.name for p in tail] == list(DENSE_STRIDE_NAMES)
    assert all(p.type == I32 for p in tail)
    assert [p.name for p in b.kernel.params[:-12]] == ["n"]


def test_signature_lists_the_twelve_strides_as_i32_scalars():
    from rocke.helpers.spec import SignatureBuilder

    sig = append_dense_stride_signature(
        SignatureBuilder().ptr("Q", "fp16").scalar("n", "i32")
    )
    built = sig.build()
    tail = built[-12:]
    assert [e["name"] for e in tail] == list(DENSE_STRIDE_NAMES)
    assert all(e["type"] == "i32" for e in tail)
    assert len(built) == 2 + 12
    assert [e["name"] for e in built[:2]] == ["Q", "n"]
    only = append_dense_stride_signature(SignatureBuilder()).build()
    assert only == [{"name": n, "type": "i32"} for n in DENSE_STRIDE_NAMES]


# --------------------------------------------------------------------------- #
# f) K and V of an fp8 K/V-cache spec are one byte per element
# --------------------------------------------------------------------------- #
#
# The shared vectors count 2 bytes per element (the C++ twin ships no fp8 dense
# kernel), so these rows are Python only. A spec that declares ``kv_elem_bytes = 1``
# gets the K/V reach and the 64-bit base counted in 1-byte elements; Q and O stay at 2.
# Hand-derived edge, B = Hkv = 1, D = 128, sS = 128, block_n = 64, R = padded + 128:
#   1 byte: the last load starts at (R - 1)*128 + 127 < 2**31 <=> R <= 2**24:
#           seqlen_kv = 2**24 - 128 = 16777088 pads to itself, R = 2**24: admit;
#           one row more pads to 16777152, R = 16777280: refused.
#   2 bytes: the same two rows are both refused (2x the reach), and the 2-byte edge
#           rows KV_REACH_EXACT_SEQLEN / KV_REACH_PAST_SEQLEN above are 8388480 /
#           8388481.
FP8_KV_EDGE_ADMIT = 2**24 - 128
FP8_KV_EDGE_REFUSED = FP8_KV_EDGE_ADMIT + 1


def _fp8_kv(problem: dict) -> SimpleNamespace:
    return SimpleNamespace(**problem, kv_elem_bytes=1)


def _packed_operands(spec) -> dict[str, DenseOperand]:
    return {t: packed_dense_operand(spec, t) for t in TENSORS}


def test_the_fp8_kv_edge_is_the_hand_derived_one():
    assert FP8_KV_EDGE_ADMIT % 64 == 0 and FP8_KV_EDGE_ADMIT + 2 * 64 == 2**24
    assert (2**24 - 1) * 128 + 128 - 1 < 2**31 <= (2**24 + 1 - 1) * 128 + 128 - 1 + 1
    assert -(-FP8_KV_EDGE_REFUSED // 64) * 64 + 2 * 64 == 16777280
    assert (16777280 - 1) * 128 + 128 - 1 >= 2**31


def test_an_fp8_kv_spec_counts_the_kv_reach_in_one_byte_elements():
    # Past the 2-byte edge (refused there, size_limit) and under the 1-byte one.
    p = P_KVREACH_PAST
    two_byte = dense_strides_admit(_spec(p), _packed_operands(_spec(p)))
    assert (two_byte.ok, two_byte.reason) == (False, "size_limit")
    assert dense_strides_admit(_fp8_kv(p), _packed_operands(_spec(p))).ok
    # The 1-byte edge: exactly admitted, one row more refused; a 2-byte spec refuses
    # both.
    for seqlen, ok in ((FP8_KV_EDGE_ADMIT, True), (FP8_KV_EDGE_REFUSED, False)):
        q = dict(P_KVREACH, seqlen_kv=seqlen)
        verdict = dense_strides_admit(_fp8_kv(q), _packed_operands(_spec(q)))
        assert verdict.ok is ok, (seqlen, verdict)
        if not ok:
            assert verdict.reason == "size_limit"
        assert dense_strides_admit(_spec(q), _packed_operands(_spec(q))).reason == (
            "size_limit"
        )


def test_the_fp8_kv_element_size_does_not_touch_q_and_o():
    # Q/O at 2**31 elements are refused whatever the K/V element size is.
    p = _prob(batch=1, seqlen_q=256, num_query_heads=1)
    ops = _packed_operands(_spec(p))
    ops["q"] = DenseOperand(dims=ops["q"].dims, strides=(0, 0, Q_EDGE_REFUSED, 1))
    for spec in (_spec(p), _fp8_kv(p)):
        verdict = dense_strides_admit(spec, ops)
        assert (verdict.ok, verdict.reason) == (False, "size_limit")


if __name__ == "__main__":
    if sys.argv[1:] == ["--regenerate"]:
        VECTORS.write_text(_render(_build()), encoding="utf-8")
        print(f"wrote {VECTORS}")
    else:
        sys.exit("usage: test_attention_dense_strided_admission.py --regenerate")
