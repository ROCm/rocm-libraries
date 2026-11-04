/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_refusals_emit.c -- C-side emitter for the
 * attention backward resource refusals (attention_bwd_refusals_emit.py):
 * spec_digest, RESOURCE_REFUSED and DQ_RESOURCE_REFUSED through
 * rocke/instance_attention_bwd.h and rocke/instance_attention_bwd_dq.h, with
 * the same text dump as the Python emitter. The main-spec parser comes from
 * attention_bwd_spec_emit.c and the dQ one from attention_bwd_dq_emit.c (both
 * included in their parse-only modes). Regenerate the table with
 * `attention_bwd_refusals_emit.py --print-c-configs`.
 *
 * run_diff fingerprints this file only: after editing an included emitter,
 * use a fresh --build-root (check_byte_identity does) so the cached emitter
 * is rebuilt.
 *
 * Only mode "ll" (the dump) is supported.
 */
static const char* const CONFIGS[] = {
    /* default gfx942 d64 */
    "main gfx942|head_size:64",
    /* default gfx950 d128 bf16 */
    "main gfx950|head_size:128;dtype:bf16",
    /* default gfx1151 d64 */
    "main gfx1151|head_size:64",
    /* default gfx1201 d32 */
    "main gfx1201|head_size:32",
    /* knobs gfx950 d64: grids, agpr pair, swizzle, ptr_noalias */
    "main "
    "gfx950|head_size:64;waves:4;block_m:32;block_n:128;warp_grid_g4:1x4;agpr_alloc:128x128;lds_"
    "swizzle:ds=pad32;ptr_noalias:all;lb_mirror:True;dkv_mode:atomic",
    /* rejected spec: head_size 48 */
    "main gfx942|head_size:48",
    /* rejected by validate before the table: barrier_merge on gfx1201 */
    "main gfx1201|head_size:64;barrier_merge:on",
    /* refused gfx942 sgpr_spill: ir row */
    "main "
    "gfx942|head_size:64;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:lds;kt_source:lds;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x1;warp_grid_g13:1x1;step_order:head_outer",
    /* refused gfx942 scratch+vgpr_spill+sgpr_spill: ir row */
    "main "
    "gfx942|head_size:32;dkv_mode:atomic;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:"
    "2x2;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:True;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;"
    "warp_grid_g02:1x4;warp_grid_g13:4x1;loop_sync:lds_only;step_order:head_outer",
    /* refused gfx942 scratch+vgpr_spill: set row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:2x4;warp_grid_g13:2x4",
    /* refused gfx942 compiler crash (comgr segfault): set row */
    "main "
    "gfx942|head_size:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;"
    "scheduler_strategy:iterative-ilp;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* refused gfx942 agpr_copy_traffic: set row */
    "main "
    "gfx942|head_size:32;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:backend;s_in_agpr:False;"
    "warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* refused gfx950 sgpr_spill: ir row */
    "main "
    "gfx950|head_size:32;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;step_order:head_outer",
    /* refused gfx950 scratch+vgpr_spill+sgpr_spill: ir row */
    "main "
    "gfx950|head_size:32;dkv_mode:atomic;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:"
    "2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:True;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;"
    "warp_grid_g02:1x4;warp_grid_g13:4x1;loop_sync:lds_only;step_order:head_outer",
    /* refused gfx950 agpr_copy_traffic: set row */
    "main "
    "gfx950|head_size:32;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:backend;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1",
    /* refused gfx1151 sgpr_spill: ir row */
    "main "
    "gfx1151|head_size:32;dtype:bf16;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;"
    "warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:"
    "lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;"
    "sched:none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;warp_"
    "grid_g02:1x1;warp_grid_g13:1x1;loop_sync:lds_only;step_order:head_outer",
    /* refused gfx1151 scratch+vgpr_spill+sgpr_spill: ir row */
    "main "
    "gfx1151|head_size:64;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:"
    "1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_"
    "residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;warp_grid_"
    "g02:1x1;warp_grid_g13:1x1;loop_sync:lds_only;step_order:head_outer",
    /* refused gfx1151 scratch+vgpr_spill: ir row */
    "main "
    "gfx1151|head_size:128;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:"
    "1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_"
    "residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;scheduler_"
    "strategy:iterative-minreg;warp_grid_g02:1x1;warp_grid_g13:1x1;loop_sync:lds_only;step_order:"
    "head_outer",
    /* refused gfx1201 scratch+vgpr_spill: ir row */
    "main "
    "gfx1201|head_size:64;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:"
    "1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_"
    "residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;warp_grid_"
    "g02:1x1;warp_grid_g13:1x1;loop_sync:lds_only;step_order:head_outer",
    /* refused gfx1201 sgpr_margin: ir row */
    "main "
    "gfx1201|head_size:32;dkv_mode:atomic;waves:2;block_m:16;block_n:32;block_k4:32;warp_grid_g4:"
    "1x2;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_"
    "residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;warp_grid_"
    "g02:1x2;warp_grid_g13:2x1;loop_sync:lds_only;step_order:head_outer",
    /* refused gfx950 scratch+vgpr_spill: mcell row */
    "main "
    "gfx950|head_size:128;dtype:bf16;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:lds;kt_"
    "source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:stage_table_atomics;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;scheduler_strategy:iterative-minreg;warp_grid_g02:1x1;warp_grid_g13:1x1;lb_"
    "mirror:True",
    /* refused gfx950 scratch+vgpr_spill+agpr_copy_traffic: compose row */
    "main "
    "gfx950|head_size:64;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:True;agpr_"
    "route:asm_g13;warp_grid_g02:1x8;warp_grid_g13:8x1;loop_sync:counted",
    /* refused gfx1151 sgpr_margin: mirror row */
    "main "
    "gfx1151|head_size:32;seq_mode:thd;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;"
    "atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:"
    "lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:"
    "False;edge_tiles:False;head_pack:False;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x1;warp_"
    "grid_g13:1x1;dkv_slab_hoist:True;lb_mirror:True",
    /* refused gfx1201 sgpr_spill: mirror row */
    "main "
    "gfx1201|head_size:32;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:"
    "wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_"
    "source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "edge_tiles:True;head_pack:False;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_"
    "g13:1x1;step_order:head_outer;lb_mirror:True",
    /* refused gfx942 scratch+vgpr_spill+agpr_copy_traffic: spec row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:True;agpr_"
    "route:asm_g13;asm_cluster:gemm;warp_grid_g02:2x4;warp_grid_g13:8x1",
    /* refused gfx942 sgpr_spill+agpr_copy_traffic: spec row */
    "main "
    "gfx942|head_size:128;dtype:bf16;seq_mode:thd;waves:4;block_m:64;block_n:64;block_k4:32;warp_"
    "grid_g4:2x2;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* refused gfx950 compiler crash (comgr segfault): spec row */
    "main "
    "gfx950|head_size:128;dtype:bf16;seq_mode:thd;dkv_mode:atomic;waves:8;block_m:16;block_n:128;"
    "block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:"
    "relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:vgpr;"
    "sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;"
    "agpr_alloc:0x0;s_in_agpr:False;scheduler_strategy:iterative-ilp;warp_grid_g02:1x8;warp_grid_"
    "g13:8x1;barrier_merge:on",
    /* refused gfx950 sgpr_spill+agpr_copy_traffic: spec row */
    "main "
    "gfx950|head_size:128;dtype:bf16;waves:4;block_m:64;block_n:64;block_k4:32;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;s_in_agpr:False;warp_grid_g02:"
    "1x4;warp_grid_g13:4x1",
    /* refused gfx950 vgpr_spill+agpr_copy_traffic: spec row */
    "main "
    "gfx950|head_size:64;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:"
    "2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:"
    "False;agpr_route:asm_g13;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* refused gfx942 vgpr_spill+agpr_copy_traffic: spec row */
    "main "
    "gfx942|head_size:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:192x192;s_in_agpr:True;agpr_"
    "route:asm_g13;asm_cluster:gemm;asm_load_hoist:True;warp_grid_g02:1x8;warp_grid_g13:2x4",
    /* refused gfx942 sgpr_spill: counted row */
    "main "
    "gfx942|head_size:32;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:lds;kt_source:lds;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x1;warp_grid_g13:1x1;loop_sync:counted;counted_issue:late;lb_mirror:True",
    /* refused gfx942 sgpr_spill: mirror row */
    "main "
    "gfx942|head_size:128;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:lds;kt_source:lds;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;xcd_remap:True;agpr_alloc:0x0;s_in_"
    "agpr:False;scheduler_strategy:iterative-minreg;warp_grid_g02:1x1;warp_grid_g13:1x1;lb_mirror:"
    "True",
    /* refused gfx942 sgpr_spill: mcell row */
    "main "
    "gfx942|head_size:64;dq_mode:split;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;lb_mirror:True",
    /* refused gfx942 sgpr_spill: mcell row */
    "main "
    "gfx942|head_size:64;dtype:bf16;dkv_mode:atomic;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;g4_slicing:slice_db;lb_mirror:"
    "True",
    /* refused gfx942 sgpr_spill: ccell row */
    "main "
    "gfx942|head_size:64;mask_class:none;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:"
    "2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:True;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;"
    "warp_grid_g02:1x8;warp_grid_g13:8x1;loop_sync:counted;q_rotate:True",
    /* refused gfx942 sgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x8;warp_grid_g13:8x1;dkv_store_vec:1",
    /* refused gfx942 sgpr_spill: spec row */
    "main "
    "gfx942|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;stats_vec:4",
    /* refused gfx942 sgpr_spill: spec row */
    "main "
    "gfx942|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;bf16_cast:rne_noguard;bwd_exp2:"
    "fast;stats_fma:True;stats_vec:4",
    /* refused gfx942 sgpr_spill: access row */
    "main "
    "gfx942|head_size:64;dkv_mode:atomic;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:"
    "2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;dkv_sink:slices;ws_bounds:buffer",
    /* refused gfx942 sgpr_spill: access row */
    "main "
    "gfx942|head_size:64;dtype:bf16;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;slab_carry:True",
    /* refused gfx942 scratch+vgpr_spill+sgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;dtype:bf16;seq_mode:thd;dkv_mode:atomic;waves:8;block_m:32;block_n:128;"
    "block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:"
    "lds;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;"
    "sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;"
    "agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:2x4;worklist:True",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;seq_mode:thd;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:dkv_out=xor,"
    "do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,v_stage=xor;sched:none;setprio:"
    "False;waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_"
    "agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;seq_mode:thd;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:"
    "reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_"
    "per_eu:1;edge_tiles:True;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;seq_mode:thd;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:"
    "reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_"
    "per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;ws_layout:token_major;agpr_alloc:"
    "0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x8;warp_grid_g13:2x4;dkv_epilogue:lds_staged_pair;dkv_store_vec:2",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;static_scale:runtime;waves:8;block_m:32;block_n:128;block_k4:32;warp_"
    "grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:"
    "reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:"
    "False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_"
    "agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select;lb_interleave:True",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x8;warp_grid_g13:2x4;stage_order:g4g1g3",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select;dq_issue:fragment_regs",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select;ptr_noalias:workspace",
    /* refused gfx942 scratch+vgpr_spill: spec row */
    "main "
    "gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select;lds_order:phase_low",
    /* refused gfx950 sgpr_spill: ir row */
    "main "
    "gfx950|head_size:64;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x8;warp_grid_g13:8x1;qdo_stage:joint",
    /* refused gfx950 sgpr_spill: spec row */
    "main "
    "gfx950|head_size:64;dkv_mode:atomic;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:"
    "2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;interior_clamp:True",
    /* refused gfx950 sgpr_spill: ringsched row */
    "main "
    "gfx950|head_size:32;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:tr_read;ring_depth:2;global_path:vgpr;sched:iglp1;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;barrier_merge:on;ring_issue:late",
    /* refused gfx950 agpr_copy_traffic: spec row */
    "main "
    "gfx950|head_size:64;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:True;agpr_"
    "route:asm_g13;asm_tail_nop:12;asm_cluster:gemm;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* refused gfx1151 scratch+vgpr_spill: ir row */
    "main "
    "gfx1151|head_size:64;seq_mode:thd;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;"
    "warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:"
    "lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;"
    "sched:none;setprio:False;edge_tiles:False;head_pack:False;acc_in_lds:True;s_in_agpr:False;"
    "warp_grid_g02:1x1;warp_grid_g13:1x1;stats_load:step_top",
    /* refused gfx1151 scratch+vgpr_spill: sched row */
    "main "
    "gfx1151|head_size:128;stage_vec:1;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;"
    "atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:"
    "lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:iglp1;setprio:"
    "False;edge_tiles:False;head_pack:False;acc_in_lds:True;s_in_agpr:False;scheduler_strategy:"
    "iterative-minreg;warp_grid_g02:1x1;warp_grid_g13:1x1",
    /* default dq gfx942 d64 */
    "dq gfx942|head_size:64",
    /* default dq gfx950 d128 thd */
    "dq gfx950|head_size:128;seq_mode:thd",
    /* dq of a main spec: gfx942 d128 bf16 split */
    "dq main=gfx942|head_size:128;dtype:bf16;dq_mode:split#kv_prefetch:True",
    /* rejected dq: bad name */
    "dq gfx942|head_size:64;name:bad-name",
    /* refused dq gfx942 compiler crash: ('gfx942', 'default', 'bf16', 64, 'batched', 'band', 1, 'edge') */
    "dq "
    "gfx942|head_size:64;dtype:bf16;stage_vec:1;edge_tiles:True;scheduler_strategy:iterative-ilp",
    /* refused dq gfx950 compiler crash: ('gfx950', 'default', 'bf16', 64, 'batched', 'band', 1, 'prefetch') */
    "dq "
    "gfx950|head_size:64;dtype:bf16;stage_vec:1;kv_prefetch:True;scheduler_strategy:iterative-ilp",
    /* refused dq gfx950 compiler crash: ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 1, 'edge') */
    "dq "
    "gfx950|head_size:128;dtype:bf16;stage_vec:1;waves:8;block_m:128;block_n:32;edge_tiles:True;kv_"
    "pad:8;scheduler_strategy:iterative-ilp",
    /* refused dq gfx942 scratch+vgpr_spill */
    "dq "
    "gfx942|head_size:64;dtype:bf16;waves:2;block_m:32;block_n:128;atom_g02:16x16x16;atom_g4:"
    "16x16x16;transpose_source:lds_plain;waves_per_eu:2;agpr_alloc:0x0;scheduler_strategy:"
    "iterative-maxocc",
    /* refused dq gfx942 scratch+vgpr_spill+sgpr_spill */
    "dq "
    "gfx942|head_size:128;dtype:bf16;mask_class:none;waves:2;block_m:32;block_n:128;atom_g02:"
    "16x16x16;atom_g4:16x16x16;transpose_source:lds_plain;waves_per_eu:2;agpr_alloc:0x0;scheduler_"
    "strategy:iterative-maxocc",
    /* refused dq gfx950 scratch+vgpr_spill */
    "dq "
    "gfx950|head_size:128;dtype:bf16;seq_mode:thd;waves:2;block_m:32;block_n:128;atom_g02:16x16x32;"
    "atom_g4:16x16x16;transpose_source:lds_plain;waves_per_eu:2;agpr_alloc:0x0;scheduler_strategy:"
    "iterative-maxocc",
    /* refused dq gfx950 compiler crash (comgr segfault) */
    "dq "
    "gfx950|head_size:128;waves:8;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:16x16x16;"
    "transpose_source:lds_plain;kv_pad:8;agpr_alloc:0x0;scheduler_strategy:iterative-ilp",
};

#define ROCKE_ATTN_BWD_DQ_EMIT_CONFIGS 1
#define ROCKE_ATTN_BWD_DQ_EMIT_PARSE_ONLY 1
#include "attention_bwd_dq_emit.c"

static Lines dump_main(const std::string& rest)
{
    Lines lines = {"kind: main"};
    std::string arch;
    rocke_attn_bwd_spec_t spec;
    rocke_attn_bwd_err_t err;
    if(!parse_config(dq_keep(rest), &arch, &spec))
        return {};
    if(rocke_attn_bwd_spec_init(&spec, &err) != ROCKE_OK)
    {
        lines.push_back("spec: " + exc(err));
        return lines;
    }
    lines.push_back("arch: " + arch);
    char digest[ROCKE_ATTN_BWD_DIGEST_CAP];
    rocke_attn_bwd_spec_digest(&spec, arch.c_str(), digest, &err);
    lines.push_back(std::string("digest: ") + digest);
    rocke_attn_bwd_spec_t r;
    if(rocke_attn_bwd_spec_resolved(&spec, arch.c_str(), &r, &err) != ROCKE_OK)
    {
        lines.push_back("resolved: " + exc(err));
        return lines;
    }
    rocke_attn_bwd_spec_digest(&r, arch.c_str(), digest, &err);
    lines.push_back(std::string("resolved digest: ") + digest);
    const char* reason = NULL;
    rocke_attn_bwd_resource_refusal(&r, arch.c_str(), &reason, &err);
    lines.push_back(std::string("refusal: ") + (reason ? reason : "None"));
    for(bool flag : {true, false})
    {
        rocke_attn_bwd_spec_t v;
        std::string verdict
            = rocke_attn_bwd_validate_spec(&spec, arch.c_str(), flag, &v, &err) == ROCKE_OK
                  ? std::string("ok")
                  : exc(err);
        lines.push_back(std::string("validate main_kernel_refusals=") + (flag ? "True" : "False")
                        + ": " + verdict);
    }
    return lines;
}

static Lines dump_dq(const std::string& rest)
{
    Lines lines = {"kind: dq"};
    std::string arch, msg;
    rocke_attn_bwd_dq_spec_t spec;
    rocke_attn_bwd_err_t err;
    int rc = dq_config_spec(dq_keep(rest), &arch, &spec, &msg);
    if(rc == 2)
        return {};
    if(rc != 0)
    {
        lines.push_back("spec: ValueError: " + msg);
        return lines;
    }
    lines.push_back("arch: " + arch);
    char digest[ROCKE_ATTN_BWD_DIGEST_CAP];
    rocke_attn_bwd_dq_spec_digest(&spec, arch.c_str(), digest, &err);
    lines.push_back(std::string("digest: ") + digest);
    rocke_attn_bwd_dq_spec_t r;
    if(rocke_attn_bwd_dq_spec_resolved(&spec, arch.c_str(), &r, &err) != ROCKE_OK)
    {
        lines.push_back("resolved: " + exc(err));
        return lines;
    }
    rocke_attn_bwd_dq_spec_digest(&r, arch.c_str(), digest, &err);
    lines.push_back(std::string("resolved digest: ") + digest);
    const char* reason = NULL;
    rocke_attn_bwd_dq_resource_refusal(&r, arch.c_str(), &reason, &err);
    lines.push_back(std::string("refusal: ") + (reason ? reason : "None"));
    rocke_attn_bwd_dq_spec_t v;
    lines.push_back("validate: "
                    + (rocke_attn_bwd_validate_dq_spec(&spec, arch.c_str(), &v, &err) == ROCKE_OK
                           ? std::string("ok")
                           : exc(err)));
    return lines;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= N_CONFIGS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    std::string config(CONFIGS[idx]);
    size_t sp = config.find(' ');
    std::string kind = config.substr(0, sp), rest = config.substr(sp + 1);
    Lines lines = kind == "main" ? dump_main(rest) : kind == "dq" ? dump_dq(rest) : Lines();
    if(lines.empty())
    {
        fprintf(stderr, "bad config %s\n", CONFIGS[idx]);
        return 1;
    }
    for(const std::string& line : lines)
    {
        fputs(line.c_str(), stdout);
        fputc('\n', stdout);
    }
    return 0;
}
