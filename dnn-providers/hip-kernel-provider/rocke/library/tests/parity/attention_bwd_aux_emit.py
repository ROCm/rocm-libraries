#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_aux_emit.py -- Python reference emitter for the
# attention backward aux kernels (prep, convert, reduce, page_zero,
# page_convert of kernels/common/attention_bwd_aux.py). Selects one config by
# argv[1], builds the AttnBwdAuxSpec and the kernel of that stage for the
# config's arch and prints the .ll, to be byte-compared with the C emitter
# attention_bwd_aux_emit.c.
#
# Configs cover gfx942 and gfx950, the shipped defaults, the tuned-table
# entries of dispatch/attention_bwd that reach the aux kernels
# (convert_rows_per_cta "4x", convert_vec 16, use_worklist on gfx950), every
# compile-time field of the spec, wave32 (gfx1151), and the specs both
# engines must reject.
#
# Config index map (keep in sync with attention_bwd_aux_emit.c):
#    0  prep         gfx950  shipped default
#    1  prep         gfx942  shipped default
#    2  prep         gfx950  THD
#    3  prep         gfx950  tuned gfx950 thd entry: use_worklist
#    4  prep         gfx942  token-major workspace
#    5  prep         gfx950  scalar user-tensor access
#    6  prep         gfx942  narrow_vec 2
#    7  prep         gfx950  unified family prep (paged kv)
#    8  prep         gfx942  narrow_vec 4, explicit rows, work list
#    9  convert      gfx950  shipped default
#   10  convert      gfx942  shipped default
#   11  convert      gfx950  tuned gfx950 d64 bf16 entry: convert_rows_per_cta 4x
#   12  convert      gfx950  tuned gfx950 d64 bf16 thd entry: convert_vec 16
#   13  convert      gfx942  convert_vec 4, token-major
#   14  convert      gfx950  narrow_vec 2
#   15  convert      gfx942  scalar stores
#   16  reduce       gfx950  dkv_sink slices, default
#   17  reduce       gfx942  convert_vec 16
#   18  reduce       gfx950  convert_vec 4, token-major, narrow_vec 4
#   19  page_zero    gfx950  unified default
#   20  page_zero    gfx942  unified default
#   21  page_zero    gfx950  explicit rows
#   22  page_convert gfx950  unified default
#   23  page_convert gfx942  scalar stores
#   24  page_convert gfx942  d32
#   25  convert      gfx950  rejected: worklist on the convert
#   26  page_zero    gfx950  rejected: page role without paged kv
#   27  prep         gfx950  rejected: convert_vec on the prep
#   28  convert      gfx950  rejected: rows that do not fill a wave
#   29  prep         gfx942  custom kernel-name stem
#   30  convert      gfx950  rejected: explicit rows_per_cta 0
#   31  prep         gfx942  rejected: negative rows_per_cta
#   32  convert      gfx950  rejected: rows over 1024 threads
#   33  prep         gfx000  rejected: unknown arch
#   34  prep         gfx950  batched, token-major workspace
#   35  prep         gfx942  THD, scalar user-tensor access
#   36  convert      gfx950  d128, convert_vec 16
#   37  convert      gfx942  d128, convert_vec 4, THD
#   38  reduce       gfx950  THD, convert_vec 8
#   39  reduce       gfx942  explicit rows, d128
#   40  page_zero    gfx950  rejected: token-major page workspace
#   41  page_convert gfx942  rejected: token-major page workspace
#   42  page_convert gfx950  rejected: convert_vec 16
#   43  page_zero    gfx942  rejected: convert_vec 4
#   44  page_convert gfx950  narrow_vec 2
#   45  page_zero    gfx950  scalar, narrow_vec 4
#   46  convert      gfx1151 wave32 default rows
#   47  convert      gfx1151 wave32, rows filling one wave
#   48  prep         gfx1151 wave32, THD work list
from kernels.common.attention_bwd_aux import (
    AttnBwdAuxSpec,
    build_attn_bwd_convert,
    build_attn_bwd_page_convert,
    build_attn_bwd_page_zero,
    build_attn_bwd_prep,
    build_attn_bwd_reduce,
)

from _emit_common import run_emit

_BUILDERS = {
    "prep": build_attn_bwd_prep,
    "convert": build_attn_bwd_convert,
    "reduce": build_attn_bwd_reduce,
    "page_zero": build_attn_bwd_page_zero,
    "page_convert": build_attn_bwd_page_convert,
}

# (stage, arch, AttnBwdAuxSpec fields)
_CONFIGS = (
    ("prep", "gfx950", {"head_size": 128}),
    ("prep", "gfx942", {"head_size": 128}),
    ("prep", "gfx950", {"head_size": 64, "dtype": "bf16", "seq_mode": "thd"}),
    (
        "prep",
        "gfx950",
        {"head_size": 64, "dtype": "bf16", "seq_mode": "thd", "worklist": True},
    ),
    (
        "prep",
        "gfx942",
        {"head_size": 32, "seq_mode": "thd", "ws_layout": "token_major"},
    ),
    ("prep", "gfx950", {"head_size": 128, "dtype": "bf16", "stage_vec": 1}),
    (
        "prep",
        "gfx942",
        {"head_size": 64, "stage_vec": 1, "narrow_vec": 2, "ws_layout": "token_major"},
    ),
    ("prep", "gfx950", {"head_size": 128, "seq_mode": "thd", "kv_target": "paged"}),
    (
        "prep",
        "gfx942",
        {
            "head_size": 64,
            "dtype": "bf16",
            "seq_mode": "thd",
            "stage_vec": 1,
            "narrow_vec": 4,
            "rows_per_cta": 16,
            "worklist": True,
        },
    ),
    ("convert", "gfx950", {"head_size": 128}),
    ("convert", "gfx942", {"head_size": 128}),
    ("convert", "gfx950", {"head_size": 64, "dtype": "bf16", "rows_per_cta": 32}),
    (
        "convert",
        "gfx950",
        {"head_size": 64, "dtype": "bf16", "seq_mode": "thd", "convert_vec": 16},
    ),
    (
        "convert",
        "gfx942",
        {
            "head_size": 32,
            "seq_mode": "thd",
            "convert_vec": 4,
            "ws_layout": "token_major",
        },
    ),
    (
        "convert",
        "gfx950",
        {"head_size": 128, "dtype": "bf16", "stage_vec": 1, "narrow_vec": 2},
    ),
    ("convert", "gfx942", {"head_size": 64, "seq_mode": "thd", "stage_vec": 1}),
    ("reduce", "gfx950", {"head_size": 128}),
    (
        "reduce",
        "gfx942",
        {"head_size": 64, "dtype": "bf16", "seq_mode": "thd", "convert_vec": 16},
    ),
    (
        "reduce",
        "gfx950",
        {
            "head_size": 32,
            "dtype": "bf16",
            "convert_vec": 4,
            "ws_layout": "token_major",
            "stage_vec": 1,
            "narrow_vec": 4,
        },
    ),
    (
        "page_zero",
        "gfx950",
        {"head_size": 128, "seq_mode": "thd", "kv_target": "paged"},
    ),
    (
        "page_zero",
        "gfx942",
        {"head_size": 64, "dtype": "bf16", "seq_mode": "thd", "kv_target": "paged"},
    ),
    (
        "page_zero",
        "gfx950",
        {"head_size": 32, "kv_target": "paged", "rows_per_cta": 32},
    ),
    (
        "page_convert",
        "gfx950",
        {"head_size": 128, "dtype": "bf16", "seq_mode": "thd", "kv_target": "paged"},
    ),
    ("page_convert", "gfx942", {"head_size": 64, "kv_target": "paged", "stage_vec": 1}),
    (
        "page_convert",
        "gfx942",
        {"head_size": 32, "dtype": "bf16", "kv_target": "paged"},
    ),
    ("convert", "gfx950", {"head_size": 64, "seq_mode": "thd", "worklist": True}),
    ("page_zero", "gfx950", {"head_size": 64}),
    ("prep", "gfx950", {"head_size": 64, "convert_vec": 16}),
    ("convert", "gfx950", {"head_size": 64, "rows_per_cta": 3}),
    ("prep", "gfx942", {"head_size": 128, "name": "custom_bwd"}),
    ("convert", "gfx950", {"head_size": 64, "rows_per_cta": 0}),
    ("prep", "gfx942", {"head_size": 64, "rows_per_cta": -1}),
    ("convert", "gfx950", {"head_size": 64, "rows_per_cta": 256}),
    ("prep", "gfx000", {"head_size": 64}),
    ("prep", "gfx950", {"head_size": 128, "ws_layout": "token_major"}),
    (
        "prep",
        "gfx942",
        {"head_size": 128, "dtype": "bf16", "seq_mode": "thd", "stage_vec": 1},
    ),
    ("convert", "gfx950", {"head_size": 128, "convert_vec": 16}),
    (
        "convert",
        "gfx942",
        {"head_size": 128, "dtype": "bf16", "seq_mode": "thd", "convert_vec": 4},
    ),
    ("reduce", "gfx950", {"head_size": 64, "seq_mode": "thd"}),
    ("reduce", "gfx942", {"head_size": 128, "dtype": "bf16", "rows_per_cta": 16}),
    (
        "page_zero",
        "gfx950",
        {"head_size": 64, "kv_target": "paged", "ws_layout": "token_major"},
    ),
    (
        "page_convert",
        "gfx942",
        {"head_size": 64, "kv_target": "paged", "ws_layout": "token_major"},
    ),
    (
        "page_convert",
        "gfx950",
        {"head_size": 64, "kv_target": "paged", "convert_vec": 16},
    ),
    ("page_zero", "gfx942", {"head_size": 64, "kv_target": "paged", "convert_vec": 4}),
    (
        "page_convert",
        "gfx950",
        {
            "head_size": 64,
            "dtype": "bf16",
            "kv_target": "paged",
            "stage_vec": 1,
            "narrow_vec": 2,
        },
    ),
    (
        "page_zero",
        "gfx950",
        {"head_size": 128, "kv_target": "paged", "stage_vec": 1, "narrow_vec": 4},
    ),
    ("convert", "gfx1151", {"head_size": 64}),
    ("convert", "gfx1151", {"head_size": 64, "rows_per_cta": 4}),
    (
        "prep",
        "gfx1151",
        {"head_size": 64, "dtype": "bf16", "seq_mode": "thd", "worklist": True},
    ),
)


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    stage, arch, fields = _CONFIGS[idx]
    # A (stage, spec) pair; the spec is built in _build so a spec the
    # dataclass rejects fails like the C side (build error, not unknown idx).
    return (stage, fields), arch


def _build(selected, *, arch: str):
    stage, fields = selected
    return _BUILDERS[stage](AttnBwdAuxSpec(**fields), arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: attention_bwd_aux_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
