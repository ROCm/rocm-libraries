/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_sched_params_emit.c -- C-side emitter for the
 * direct, edge and rejected stage-table configs
 * (attention_bwd_sched_params_emit.py): the dump of attention_bwd_sched_emit.c
 * over this table. Regenerate the table with
 * `attention_bwd_sched_params_emit.py --print-c-configs`.
 *
 * run_diff fingerprints this file only: after editing
 * attention_bwd_sched_emit.c, use a fresh --build-root (check_byte_identity
 * does) so the cached emitter is rebuilt.
 *
 * Only mode "ll" (the dump) is supported.
 */
static const char* const CONFIGS[] = {
    /* dq default gfx942 d32 fp16 sv8 */
    "dqgeom|gfx942|head_size:32;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:fp16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx942 d32 fp16 sv1 */
    "dqgeom|gfx942|head_size:32;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:fp16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx942 d32 bf16 sv8 */
    "dqgeom|gfx942|head_size:32;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:bf16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx942 d32 bf16 sv1 */
    "dqgeom|gfx942|head_size:32;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:bf16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx942 d64 fp16 sv8 */
    "dqgeom|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:fp16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx942 d64 fp16 sv1 */
    "dqgeom|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:fp16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx942 d64 bf16 sv8 */
    "dqgeom|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:bf16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx942 d64 bf16 sv1 */
    "dqgeom|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:bf16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx942 d128 fp16 sv8 */
    "dqgeom|gfx942|head_size:128;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:fp16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx942 d128 fp16 sv1 */
    "dqgeom|gfx942|head_size:128;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:fp16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx942 d128 bf16 sv8 */
    "dqgeom|gfx942|head_size:128;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:bf16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx942 d128 bf16 sv1 */
    "dqgeom|gfx942|head_size:128;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:bf16;"
    "transpose_source:lds_plain;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx950 d32 fp16 sv8 */
    "dqgeom|gfx950|head_size:32;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx950 d32 fp16 sv1 */
    "dqgeom|gfx950|head_size:32;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx950 d32 bf16 sv8 */
    "dqgeom|gfx950|head_size:32;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx950 d32 bf16 sv1 */
    "dqgeom|gfx950|head_size:32;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx950 d64 fp16 sv8 */
    "dqgeom|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx950 d64 fp16 sv1 */
    "dqgeom|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx950 d64 bf16 sv8 */
    "dqgeom|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx950 d64 bf16 sv1 */
    "dqgeom|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx950 d128 fp16 sv8 */
    "dqgeom|gfx950|head_size:128;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx950 d128 fp16 sv1 */
    "dqgeom|gfx950|head_size:128;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq default gfx950 d128 bf16 sv8 */
    "dqgeom|gfx950|head_size:128;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq default gfx950 d128 bf16 sv1 */
    "dqgeom|gfx950|head_size:128;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:1",
    /* dq tuned general/64/bf16/none/mha/square/fixed/8/t16/full (gfx950) */
    "dqgeom|gfx950|head_size:64;k_m0:128;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq tuned general/64/fp16/none/mha/square/fixed/8/t16/full (gfx950) */
    "dqgeom|gfx950|head_size:64;k_m0:128;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read;sched:none;kv_pad:0;kv_prefetch:False;stage_vec:8",
    /* dq tuned dense 1024/4096/32/8/128/bf16/bshd/causal_br (gfx950) */
    "dqgeom|gfx950|head_size:128;k_m0:128;k_n0:32;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:16;kv_prefetch:False;stage_vec:8",
    /* dq tuned dense 2048/2048/12/12/64/bf16/bshd/none (gfx950) */
    "dqgeom|gfx950|head_size:64;k_m0:128;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:none;kv_pad:32;kv_prefetch:False;stage_vec:8",
    /* sync default sched */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4",
    /* sync stage_table_atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:stage_table_atomics",
    /* sync fences_only */
    "table|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:fences_only",
    /* sync iglp0 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:iglp0",
    /* sync iglp1 */
    "table|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:iglp1",
    /* sync none */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;sched:none",
    /* sync setprio */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;sched:stage_table;setprio:True",
    /* sync bf16 wide atoms gfx950 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;dtype:bf16;atom_g02:32;atom_g13:"
    "32;atom_g4:32",
    /* sync wide atoms split on gfx942 */
    "table|gfx942|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atom_g02:32;atom_g13:32;atom_g4:"
    "32",
    /* sync f16 alias */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dtype:f16",
    /* sync step_top stats */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;stats_load:step_top",
    /* sync dq_issue fragment */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;dq_issue:fragment;kt_source:lds;"
    "kv_residency:lds",
    /* sync dq_issue fragment_regs */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;dq_issue:fragment_regs;kv_"
    "residency:lds;kt_source:lds",
    /* sync dq_issue fragment_regs reg K */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_issue:fragment_regs;sched:fences_"
    "only",
    /* sync LDS residency tr_read */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "transpose_source:tr_read",
    /* sync LDS residency xt_lds */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;kv_residency:lds;kt_source:lds;"
    "transpose_source:xt_lds;sched:fences_only",
    /* sync LDS residency lds_plain none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;kv_residency:lds;kt_source:lds;"
    "transpose_source:lds_plain;sched:none",
    /* sync kt lds kv reg rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;kv_residency:lds",
    /* sync kt lds, kv reg */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;kt_source:lds",
    /* sync acc_in_lds */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;acc_in_lds:True",
    /* sync acc_in_lds lds route */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;acc_in_lds:True;pt_route:lds;sched:"
    "fences_only",
    /* sync pt_route lds */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pt_route:lds",
    /* sync pt_route lds none */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;pt_route:lds;sched:none;transpose_"
    "source:tr_read",
    /* sync g4 slice_db_g3 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;g4_slicing:slice_db_g3",
    /* sync g4 slice */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;g4_slicing:slice",
    /* sync row pads */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;lds_swizzle:q=pad8,do=pad16,k_stage="
    "pad32,ds=pad8",
    /* sync row pads LDS residency */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "lds_swizzle:q=pad16,do=pad8,k_res=pad8,v_res=pad32,ds=xor",
    /* sync row pads spaced */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;lds_swizzle: q = pad8 , "
    "do=none,,junk",
    /* sync negative pad */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;lds_swizzle:q=pad-8",
    /* sync pad without width */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;lds_swizzle:q=pad",
    /* sync pad bad width */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;lds_swizzle:ds=padx",
    /* sync split dQ */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split",
    /* sync split dQ lds route */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;dq_mode:split;pt_route:lds;sched:"
    "fences_only",
    /* sync split dQ none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:none",
    /* sync stage_vec 1 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;stage_vec:1",
    /* sync one wave */
    "table|gfx942|head_size:32;k_m0:16;k_n0:16;k_k4:16;waves:1",
    /* sync explicit grids */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g4:1x4;grid_g02:1x4;grid_g13:"
    "4x1",
    /* sync grid g4 2x2 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g4:2x2",
    /* sync replicated G0 grid, sched none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g02:1x2;sched:none;pt_route:"
    "lds",
    /* sync small tile, sched none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:32;k_k4:16;waves:4;grid_g13:2x2;grid_g02:1x2;sched:"
    "none",
    /* sync bf16 kK4 64 */
    "table|gfx950|head_size:128;k_m0:64;k_n0:128;k_k4:64;waves:8;dtype:bf16",
    /* dma ring R1 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring",
    /* dma ring R2 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:2",
    /* dma ring R3 atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;global_path:dma;pipeline:ring;ring_"
    "depth:3;sched:stage_table_atomics",
    /* dma ring R2 none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;global_path:dma;pipeline:ring;ring_"
    "depth:2;sched:none",
    /* dma ring R3 split */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:3;dq_mode:split",
    /* dma ring 2 waves */
    "table|gfx942|head_size:64;k_m0:16;k_n0:32;k_k4:16;waves:2;global_path:dma;pipeline:ring;ring_"
    "depth:2",
    /* dma ring 8 waves */
    "table|gfx950|head_size:128;k_m0:64;k_n0:128;k_k4:64;waves:8;global_path:dma;pipeline:ring",
    /* dma ring block_m 128 */
    "table|gfx950|head_size:64;k_m0:128;k_n0:64;k_k4:16;waves:4;global_path:dma;pipeline:ring;"
    "sched:none",
    /* dma ring partial piece */
    "table|gfx942|head_size:16;k_m0:16;k_n0:16;k_k4:16;waves:1;global_path:dma;pipeline:ring;sched:"
    "none",
    /* dma ring wave32 */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;global_path:dma;pipeline:ring;"
    "sched:none",
    /* dma ring stage_vec 1 rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;global_path:dma;pipeline:ring;stage_"
    "vec:1",
    /* dma ring head_pack rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;global_path:dma;pipeline:ring;head_"
    "pack:True",
    /* register ring R2 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:2;pipeline:ring",
    /* register ring R3 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;ring_depth:3;pipeline:ring",
    /* register ring R2 none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:2;pipeline:ring;sched:"
    "none",
    /* register ring R2 fences */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;ring_depth:2;pipeline:ring;sched:"
    "fences_only",
    /* register ring head_pack */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:2;pipeline:ring;head_"
    "pack:True",
    /* register ring joint pass */
    "table|gfx942|head_size:32;k_m0:16;k_n0:32;k_k4:16;waves:2;ring_depth:2;pipeline:ring",
    /* register ring stage_vec 1 pads */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:2;pipeline:ring;stage_"
    "vec:1;lds_swizzle:q=pad8,do=pad8",
    /* register ring split */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:2;pipeline:ring;dq_mode:"
    "split;sched:none",
    /* register ring step_top rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:2;pipeline:ring;stats_"
    "load:step_top",
    /* merged early */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged",
    /* merged late */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;ring_issue:late",
    /* merged none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:none",
    /* merged atomics pads */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:stage_table_atomics;lds_swizzle:q=pad8",
    /* merged R3 rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:3;"
    "pipeline:merged",
    /* merged dma rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "global_path:dma;pipeline:merged",
    /* merged split rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;dq_mode:split",
    /* counted early */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted",
    /* counted late */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;counted_issue:"
    "late",
    /* counted early atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;sched:stage_table_"
    "atomics",
    /* counted late atomics rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:stage_table_atomics",
    /* counted late fences */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:fences_only",
    /* counted none */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;sched:none;"
    "stage_vec:1",
    /* counted on a ring rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:2;pipeline:ring;loop_"
    "sync:counted",
    /* counted split rejected */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;dq_mode:split",
    /* lds_only */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:lds_only",
    /* ck default */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck",
    /* ck atomics */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;pipeline:ck;sched:stage_table_"
    "atomics",
    /* ck xt_lds */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;transpose_source:xt_lds",
    /* ck xt_lds 8 waves */
    "table|gfx950|head_size:128;k_m0:64;k_n0:128;k_k4:64;waves:8;pipeline:ck;transpose_source:xt_"
    "lds",
    /* ck lds route residency acc */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;pipeline:ck;pt_route:lds;kv_"
    "residency:lds;kt_source:lds;acc_in_lds:True;transpose_source:tr_read",
    /* ck dma */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;pipeline:ck;global_path:dma",
    /* ck slice_db_g3 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;g4_slicing:slice_db_g3",
    /* ck split */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;dq_mode:split",
    /* ck split lds route */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;dq_mode:split;pt_route:"
    "lds;sched:stage_table_atomics",
    /* ck ignores staging */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;ring_depth:3;barrier_"
    "merge:on;loop_sync:counted",
    /* ck wmma lds */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;pipeline:ck;transpose_source:wmma_"
    "lds;pt_route:lds",
    /* ck stage_vec 1 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;stage_vec:1;transpose_"
    "source:lds_plain",
    /* gfx1151 sync */
    "table|gfx1151|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:wmma_lds",
    /* gfx1201 lds route */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;pt_route:lds;transpose_source:lds_"
    "plain;sched:fences_only",
    /* gfx1201 wide atom split */
    "table|gfx1201|head_size:64;k_m0:32;k_n0:64;k_k4:32;waves:2;atom_g02:32;atom_g13:32;atom_g4:32",
    /* gfx1151 register ring */
    "table|gfx1151|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;ring_depth:2;pipeline:ring",
    /* gfx1250 K = 32 atoms only (no 16x16x16 atom for c_frag_len) */
    "table|gfx1250|head_size:64;k_m0:32;k_n0:64;k_k4:32;waves:4;atom_g02:32;atom_g13:32;atom_g4:32;"
    "transpose_source:lds_plain",
    /* gfx942 atom_g4 32 split into K = 16 instructions */
    "table|gfx942|head_size:64;k_m0:32;k_n0:64;k_k4:32;waves:4;atom_g4:32",
    /* bad sched */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:bogus",
    /* sched None */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:None",
    /* setprio outside a table */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:none;setprio:True",
    /* bad pipeline */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:async",
    /* pipeline ring on the sync body */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ring",
    /* pipeline sync on a ring */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:2",
    /* ring_depth 4 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:4;pipeline:ring",
    /* ring_depth 0 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:0",
    /* bad barrier_merge */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:maybe",
    /* bad ring_issue */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_issue:never",
    /* bad loop_sync */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:none",
    /* bad counted_issue */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;counted_issue:mid",
    /* bad stats_load */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;stats_load:never",
    /* bad dq_issue */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_issue:row",
    /* bad transpose_source */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;transpose_source:dma",
    /* bad kt_source */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;kt_source:vmem",
    /* bad kv_residency */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;kv_residency:vmem;kt_source:lds",
    /* bad global_path */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;global_path:tdm",
    /* bad pt_route */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pt_route:agpr",
    /* bad dq_mode */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:none",
    /* bad stage_vec */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;stage_vec:4",
    /* bad g4_slicing */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;g4_slicing:half",
    /* G4 grid holds the wrong waves */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g4:2x4",
    /* G0 grid does not divide */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g02:1x3;sched:none",
    /* G0 grid of zero waves */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g02:0x4",
    /* G1 grid does not hold */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g13:2x1",
    /* stage table on a replicated grid */
    "table|gfx942|head_size:64;k_m0:16;k_n0:32;k_k4:16;waves:4;grid_g02:1x2",
    /* kN0 not split over the waves */
    "table|gfx942|head_size:64;k_m0:16;k_n0:32;k_k4:16;waves:4",
    /* kM0 not 16 */
    "table|gfx942|head_size:64;k_m0:24;k_n0:64;k_k4:16;waves:4",
    /* D not over g4c */
    "table|gfx942|head_size:48;k_m0:16;k_n0:64;k_k4:16;waves:4",
    /* kM0 not over g4r */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g4:2x2",
    /* kN0 not over kK4 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:48;waves:4",
    /* kK4 not over atom_g4 */
    "table|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;atom_g4:32",
    /* D not over atom_g02 */
    "table|gfx950|head_size:16;k_m0:16;k_n0:64;k_k4:16;waves:1;atom_g02:32",
    /* kM0 not over atom_g13 */
    "table|gfx950|head_size:64;k_m0:48;k_n0:64;k_k4:16;waves:4;atom_g13:32;grid_g4:1x4",
    /* atom_g13 32 with kM0 16 */
    "table|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;atom_g13:32",
    /* kM0 not over g0r */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g02:2x2;sched:none;grid_g4:1x4",
    /* kN0 not over g1r */
    "table|gfx942|head_size:64;k_m0:16;k_n0:32;k_k4:16;waves:4;sched:none",
    /* D not over g1c */
    "table|gfx942|head_size:16;k_m0:64;k_n0:64;k_k4:16;waves:4;grid_g4:4x1;grid_g13:2x2;grid_g02:"
    "1x1;sched:none",
    /* several checks fail */
    "table|gfx942|head_size:40;k_m0:20;k_n0:60;k_k4:16;waves:4",
    /* atom K 48 */
    "table|gfx942|head_size:96;k_m0:48;k_n0:64;k_k4:16;waves:2;atom_g02:48",
    /* atom K 8 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;atom_g4:8",
    /* unsupported dtype */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dtype:fp32",
    /* dtype None */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dtype:None",
    /* unknown arch */
    "table|gfx000|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4",
    /* negative waves, sched none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:-4;grid_g02:1x1;grid_g4:1x4;sched:"
    "none",
    /* geometry with the minimum attributes */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4",
    /* geometry with None atoms and grids */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atoms:None;grid_g4:None;grid_g0:"
    "None;grid_g1:None;ring_depth:None",
    /* geometry keyword arguments */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;kw_dtype:bf16;kw_stage_vec:1;kw_dq_"
    "mode:split;kw_acc_in_lds:True",
    /* geometry attributes over keywords */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;stage_vec:8;dq_mode:atomic;acc_in_"
    "lds:False;kw_stage_vec:1;kw_dq_mode:split;kw_acc_in_lds:True",
    /* geometry ring_depth 0 is 1 */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:0",
    /* geometry dma ring */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;atoms:32/16/32",
    /* geometry merged */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;ring_"
    "issue:late",
    /* geometry merged on R3 */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:3",
    /* geometry counted late */
    "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:late;"
    "setprio:True",
    /* geometry with one absent atom (None, K 16) */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atoms:32/None/32",
    /* dq default gfx950 */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4",
    /* dq default gfx942 lds_plain */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;transpose_source:lds_plain",
    /* dq setprio */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:stage_table_setprio",
    /* dq fences */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;sched:fences_only;transpose_source:lds_plain",
    /* dq iglp1 */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:iglp1",
    /* dq prefetch */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;kv_prefetch:True",
    /* dq prefetch sv1 pad8 */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;kv_prefetch:True;stage_vec:1;kv_pad:8;"
    "transpose_source:lds_plain",
    /* dq pad16 wide atom */
    "dq|gfx950|head_size:128;k_m0:128;k_n0:32;waves:4;kv_pad:16;atom_g02:32;dtype:bf16",
    /* dq pad32 prefetch */
    "dq|gfx950|head_size:128;k_m0:128;k_n0:64;waves:8;kv_pad:32;kv_prefetch:True",
    /* dq wide atom split on gfx942 */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:32;transpose_source:lds_"
    "plain",
    /* dq wave32 */
    "dq|gfx1201|head_size:64;k_m0:32;k_n0:32;waves:2;transpose_source:lds_plain;kv_prefetch:True",
    /* dq bad sched */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:stage_table_atomics",
    /* dq bad transpose */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;transpose_source:xt_lds",
    /* dq bad kv_pad */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;kv_pad:4",
    /* dq bad stage_vec */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;stage_vec:2",
    /* dq kM0 not over the waves */
    "dq|gfx950|head_size:64;k_m0:32;k_n0:64;waves:4",
    /* dq kN0 not 16 */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:24;waves:4",
    /* dq D not 16 */
    "dq|gfx950|head_size:40;k_m0:64;k_n0:64;waves:4",
    /* dq D not over atom_g02 */
    "dq|gfx950|head_size:16;k_m0:64;k_n0:64;waves:4;atom_g02:32",
    /* dq kN0 not over atom_g4 */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:16;waves:4;atom_g4:32",
    /* dq atom 48 */
    "dq|gfx950|head_size:96;k_m0:64;k_n0:96;waves:4;atom_g02:48",
    /* dq unknown arch */
    "dq|gfx000|head_size:64;k_m0:64;k_n0:64;waves:4",
    /* dq bad dtype */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;dtype:fp8",
    /* dq geometry with absent knobs */
    "dqgeom|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:32;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read",
    /* dq geometry rejected */
    "dqgeom|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;atom_g02:16;atom_g4:16;dtype:fp16;"
    "transpose_source:tr_read;kv_pad:8",
    /* groups spread */
    "groups|-|mfma:4;mem:32=3,256=5,512=2,64=1",
    /* groups lead and spans */
    "groups|-|mfma:6;mem:32=4,256=7,512=6,64=3;lead:32=2,256=3;spans:512=2,64=1",
    /* groups lead None, spans {} */
    "groups|-|mfma:3;mem:256=2;spans:",
    /* groups no MFMA */
    "groups|-|mfma:0;mem:32=2,256=0,512=3,64=1;lead:32=9",
    /* groups negative MFMA */
    "groups|-|mfma:-1;mem:32=2,256=1",
    /* groups repeated class */
    "groups|-|mfma:2;mem:256=2,256=4,32=1",
    /* groups repeated lead keeps the last */
    "groups|-|mfma:2;mem:256=4;lead:256=1,256=3",
    /* groups lead past its count */
    "groups|-|mfma:2;mem:256=4;lead:256=5",
    /* groups lead of an absent class */
    "groups|-|mfma:2;mem:256=4;lead:32=1",
    /* groups zero lead of an absent class */
    "groups|-|mfma:2;mem:256=4;lead:32=0",
    /* groups negative lead */
    "groups|-|mfma:2;mem:256=4;lead:256=-1",
    /* groups span 0 */
    "groups|-|mfma:2;mem:256=4;spans:256=0",
    /* groups span past the MFMAs */
    "groups|-|mfma:2;mem:256=4;spans:512=3",
    /* groups negative count */
    "groups|-|mfma:3;mem:256=-2,32=1",
    /* groups single MFMA */
    "groups|-|mfma:1;mem:32=1,256=1,512=1,64=1;lead:256=1",
    /* counts census */
    "counts|-|stage:B;mfma:4;vmem_read:2;ds_read:6;ds_write:4;lead:256=2,32=1;spans:512=2;fixed:"
    "512=3,32=1",
    /* counts census bad class */
    "counts|-|stage:D;mfma:2;ds_read:2;fixed:1024=1",
    /* counts repeated lead */
    "counts|-|stage:A;mfma:3;ds_read:6;lead:256=1,256=2",
    /* counts bad lead */
    "counts|-|stage:A;mfma:3;ds_read:1;lead:256=2",
    /* counts five-plus pairs per lead / spans / fixed */
    "counts|-|stage:B;mfma:8;vmem_read:4;vmem_write:3;ds_read:10;ds_write:6;lead:256=1,32=1,256=2,"
    "512=1,64=1;spans:512=2,64=1,512=3,256=1,512=1;fixed:256=1,256=1,32=1,512=1,64=1,8=1",
    /* atoms gfx942 */
    "atoms|gfx942|dtype:fp16",
    /* atoms gfx950 bf16 */
    "atoms|gfx950|dtype:bf16",
    /* atoms gfx1151 */
    "atoms|gfx1151|dtype:f16",
    /* atoms gfx1201 bf16 */
    "atoms|gfx1201|dtype:bf16",
    /* atoms gfx1250 (K 32 only) */
    "atoms|gfx1250|dtype:fp16",
    /* atoms gfx90a bf16 */
    "atoms|gfx90a|dtype:bf16",
    /* atoms bad dtype */
    "atoms|gfx942|dtype:fp32",
    /* atoms unknown arch */
    "atoms|gfx000|dtype:fp16",
    /* dma 16x128 4 waves R1 */
    "dma|-|block_m:16;head_size:128;waves:4;wave_size:64;ring_depth:1;n_qd:2;n_a:8",
    /* dma 32x128 4 waves R2 */
    "dma|-|block_m:32;head_size:128;waves:4;wave_size:64;ring_depth:2;n_qd:3;n_a:16",
    /* dma 64x64 8 waves R3 */
    "dma|-|block_m:64;head_size:64;waves:8;wave_size:64;ring_depth:3;n_qd:1;n_a:4",
    /* dma 16x32 3 waves */
    "dma|-|block_m:16;head_size:32;waves:3;wave_size:64;ring_depth:3;n_qd:60;n_a:10",
    /* dma one wave clamp */
    "dma|-|block_m:32;head_size:64;waves:1;wave_size:64;ring_depth:2;n_qd:0;n_a:70",
    /* dma wave32 */
    "dma|-|block_m:16;head_size:64;waves:2;wave_size:32;ring_depth:4;n_qd:1;n_a:1",
    /* dma partial piece */
    "dma|-|block_m:16;head_size:16;waves:2;wave_size:64;ring_depth:2;n_qd:-1;n_a:0",
    /* dma block_m 128 */
    "dma|-|block_m:128;head_size:64;waves:4;wave_size:64;ring_depth:1;n_qd:0;n_a:-1",
    /* dma block_m 48 */
    "dma|-|block_m:48;head_size:64;waves:4;wave_size:64;ring_depth:0;n_qd:0;n_a:0",
    /* dma zero waves */
    "dma|-|block_m:16;head_size:64;waves:0;wave_size:64;ring_depth:3;n_qd:63;n_a:63",
};

#define ROCKE_ATTN_BWD_SCHED_EMIT_CONFIGS 1
#include "attention_bwd_sched_emit.c"
