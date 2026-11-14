#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_spec_extra_emit.py -- more specs for the C port
# of the attention backward main kernel's spec and decisions. Same dump as
# attention_bwd_spec_emit.py (imported from it), over the configs that do not
# fit that family's 256-config cap: the remaining shipped defaults on gfx942 /
# gfx950 (fp16 sv1, d32 no-mask and split dQ), the general tuned-table GQA
# entries in atomic dK / dV mode, stats_vec 2 / 4 on the RDNA atoms (the C
# lane-map evaluation and its message) and Latin-1 names (str.isalnum and
# repr escapes). The C twin attention_bwd_spec_extra_emit.c includes
# attention_bwd_spec_emit.c with this table; regenerate it with
# `attention_bwd_spec_extra_emit.py --print-c-configs`.
#
# argv[2] selects the output mode; only "ll" (the dump) is supported.
import sys

from attention_bwd_spec_emit import _parse, dump

# (comment, config)
_CONFIGS = (
    ("default gfx942 d32 fp16 sv1", "gfx942|head_size:32;dtype:fp16;stage_vec:1"),
    ("default gfx942 d64 fp16 sv1", "gfx942|head_size:64;dtype:fp16;stage_vec:1"),
    ("default gfx942 d128 fp16 sv1", "gfx942|head_size:128;dtype:fp16;stage_vec:1"),
    (
        "default gfx942 d32 no mask, runtime scale",
        "gfx942|head_size:32;mask_class:none;static_scale:runtime",
    ),
    ("default gfx942 d32 split dQ", "gfx942|head_size:32;dq_mode:split"),
    ("default gfx950 d32 fp16 sv1", "gfx950|head_size:32;dtype:fp16;stage_vec:1"),
    ("default gfx950 d64 fp16 sv1", "gfx950|head_size:64;dtype:fp16;stage_vec:1"),
    ("default gfx950 d128 fp16 sv1", "gfx950|head_size:128;dtype:fp16;stage_vec:1"),
    (
        "default gfx950 d32 no mask, runtime scale",
        "gfx950|head_size:32;mask_class:none;static_scale:runtime",
    ),
    ("default gfx950 d32 split dQ", "gfx950|head_size:32;dq_mode:split"),
    (
        "tuned general/128/bf16/causal_tl/gqa/unequal/fixed/8/t16/full (gfx942) atomic",
        "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:32;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    ),
    (
        "tuned general/64/bf16/causal_br/gqa/square/fixed/8/t64/full (gfx942) atomic",
        "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:head_major;lb_mirror:True;dkv_mode:atomic",
    ),
    (
        "tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t16/full (gfx942) atomic",
        "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:head_major;lb_mirror:True;dkv_mode:atomic",
    ),
    (
        "tuned general/128/bf16/causal_br/gqa/square/fixed/8/t16/over (gfx950) atomic",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    ),
    (
        "tuned general/128/bf16/causal_br/gqa/unequal/fixed/8/t16/over (gfx950) atomic",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;step_order:head_outer;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    ),
    (
        "tuned general/128/bf16/none/gqa/square/fixed/8/t64/over (gfx950) atomic",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    ),
    (
        "tuned general/128/bf16/none/gqa/unequal/fixed/8/t64/over (gfx950) atomic",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    ),
    (
        "tuned general/128/fp16/causal_br/gqa/unequal/fixed/8/t64/over (gfx950) atomic",
        "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;lb_mirror:True;dkv_mode:atomic",
    ),
    (
        "tuned general/128/fp16/causal_tl/gqa/square/fixed/8/t4/full (gfx950) atomic",
        "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_interleave:True;dkv_mode:atomic",
    ),
    (
        "tuned general/64/bf16/causal_br/gqa/unequal/fixed/8/t16/full (gfx950) atomic",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;static_scale:convert;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;lds_order:phase_low;lds_swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_interleave:True;dkv_mode:atomic",
    ),
    (
        "tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t4/full (gfx950) atomic",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_mirror:True;dkv_mode:atomic",
    ),
    ("stats_vec 2 on gfx1151 (C lane map)", "gfx1151|head_size:64;stats_vec:2"),
    ("stats_vec 4 on gfx1151 (C lane map)", "gfx1151|head_size:64;stats_vec:4"),
    ("stats_vec 2 on gfx1201 (C lane map)", "gfx1201|head_size:64;stats_vec:2"),
    ("stats_vec 4 on gfx1201 (C lane map)", "gfx1201|head_size:64;stats_vec:4"),
    ("Latin-1 name", "gfx942|head_size:64;name:café_bwd"),
    ("Latin-1 numeric name", "gfx942|head_size:64;name:x²_½"),
    ("rejected: Latin-1 sign in name", "gfx942|head_size:64;name:a×b"),
    ("rejected: no-break space in name", "gfx942|head_size:64;name:a\xa0b"),
    ("rejected: soft hyphen in name", "gfx942|head_size:64;name:a\xadb"),
    ("rejected: C1 control in name", "gfx942|head_size:64;name:a\x85b"),
)


def _c_literal(text: str) -> str:
    out = ""
    for byte in text.encode("utf-8"):
        if byte < 0x80:
            out += chr(byte)
        else:
            out += f'\\x{byte:02x}""'
    return out


def _print_c_configs() -> None:
    for comment, config in _CONFIGS:
        c = _c_literal(comment).replace("*/", "* /")
        print(f'    /* {c} */\n    "{_c_literal(config)}",')


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--print-c-configs":
        _print_c_configs()
        return 0
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: attention_bwd_spec_extra_emit.py <config_index> [ll]\n"
        )
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode != "ll":
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    arch, fields = _parse(_CONFIGS[idx][1])
    sys.stdout.write("".join(line + "\n" for line in dump(arch, fields)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
