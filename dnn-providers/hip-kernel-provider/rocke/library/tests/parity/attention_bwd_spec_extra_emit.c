/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_spec_extra_emit.c -- C-side emitter for the
 * extra attention backward spec configs (attention_bwd_spec_extra_emit.py):
 * the dump of attention_bwd_spec_emit.c over this table, which does not fit
 * that family's 256-config cap. Regenerate the table with
 * `attention_bwd_spec_extra_emit.py --print-c-configs`.
 *
 * run_diff fingerprints this file only: after editing
 * attention_bwd_spec_emit.c, use a fresh --build-root (check_byte_identity
 * does) so the cached emitter is rebuilt.
 *
 * Only mode "ll" (the dump) is supported.
 */
static const char* const CONFIGS[] = {
    /* default gfx942 d32 fp16 sv1 */
    "gfx942|head_size:32;dtype:fp16;stage_vec:1",
    /* default gfx942 d64 fp16 sv1 */
    "gfx942|head_size:64;dtype:fp16;stage_vec:1",
    /* default gfx942 d128 fp16 sv1 */
    "gfx942|head_size:128;dtype:fp16;stage_vec:1",
    /* default gfx942 d32 no mask, runtime scale */
    "gfx942|head_size:32;mask_class:none;static_scale:runtime",
    /* default gfx942 d32 split dQ */
    "gfx942|head_size:32;dq_mode:split",
    /* default gfx950 d32 fp16 sv1 */
    "gfx950|head_size:32;dtype:fp16;stage_vec:1",
    /* default gfx950 d64 fp16 sv1 */
    "gfx950|head_size:64;dtype:fp16;stage_vec:1",
    /* default gfx950 d128 fp16 sv1 */
    "gfx950|head_size:128;dtype:fp16;stage_vec:1",
    /* default gfx950 d32 no mask, runtime scale */
    "gfx950|head_size:32;mask_class:none;static_scale:runtime",
    /* default gfx950 d32 split dQ */
    "gfx950|head_size:32;dq_mode:split",
    /* tuned general/128/bf16/causal_tl/gqa/unequal/fixed/8/t16/full (gfx942) atomic */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:32;edge_tiles:"
    "False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;"
    "ws_layout:head_major;dkv_mode:atomic",
    /* tuned general/64/bf16/causal_br/gqa/square/fixed/8/t64/full (gfx942) atomic */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;"
    "atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_"
    "path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_"
    "plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True;dkv_mode:atomic",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t16/full (gfx942) atomic */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;"
    "atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_"
    "path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_"
    "plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True;dkv_mode:atomic",
    /* tuned general/128/bf16/causal_br/gqa/square/fixed/8/t16/over (gfx950) atomic */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    /* tuned general/128/bf16/causal_br/gqa/unequal/fixed/8/t16/over (gfx950) atomic */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;step_order:head_outer;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    /* tuned general/128/bf16/none/gqa/square/fixed/8/t64/over (gfx950) atomic */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    /* tuned general/128/bf16/none/gqa/unequal/fixed/8/t64/over (gfx950) atomic */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;dkv_mode:atomic",
    /* tuned general/128/fp16/causal_br/gqa/unequal/fixed/8/t64/over (gfx950) atomic */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True;dkv_mode:atomic",
    /* tuned general/128/fp16/causal_tl/gqa/square/fixed/8/t4/full (gfx950) atomic */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:"
    "False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_"
    "layout:head_major;lb_interleave:True;dkv_mode:atomic",
    /* tuned general/64/bf16/causal_br/gqa/unequal/fixed/8/t16/full (gfx950) atomic */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;static_scale:convert;acc_in_lds:False;agpr_alloc:"
    "None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;"
    "edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;lds_order:phase_low;lds_"
    "swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:"
    "None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_"
    "grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_interleave:True;"
    "dkv_mode:atomic",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t4/full (gfx950) atomic */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;"
    "atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_"
    "path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_"
    "read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True;dkv_mode:atomic",
    /* stats_vec 2 on gfx1151 (C lane map) */
    "gfx1151|head_size:64;stats_vec:2",
    /* stats_vec 4 on gfx1151 (C lane map) */
    "gfx1151|head_size:64;stats_vec:4",
    /* stats_vec 2 on gfx1201 (C lane map) */
    "gfx1201|head_size:64;stats_vec:2",
    /* stats_vec 4 on gfx1201 (C lane map) */
    "gfx1201|head_size:64;stats_vec:4",
    /* Latin-1 name */
    "gfx942|head_size:64;name:caf\xc3"
    "\xa9"
    "_bwd",
    /* Latin-1 numeric name */
    "gfx942|head_size:64;name:x\xc2"
    "\xb2"
    "_\xc2"
    "\xbd"
    "",
    /* rejected: Latin-1 sign in name */
    "gfx942|head_size:64;name:a\xc3"
    "\x97"
    "b",
    /* rejected: no-break space in name */
    "gfx942|head_size:64;name:a\xc2"
    "\xa0"
    "b",
    /* rejected: soft hyphen in name */
    "gfx942|head_size:64;name:a\xc2"
    "\xad"
    "b",
    /* rejected: C1 control in name */
    "gfx942|head_size:64;name:a\xc2"
    "\x85"
    "b",
};

#define ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS 1
#include "attention_bwd_spec_emit.c"
