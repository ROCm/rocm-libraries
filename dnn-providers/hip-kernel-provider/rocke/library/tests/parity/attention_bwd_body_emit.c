/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_body_emit.c -- C-side emitter for the shared
 * inner body of the attention backward main kernel. Builds, identically to
 * the Python emitter attention_bwd_body_emit.py, a probe kernel that drives
 * helper_rocke.instances.common._attention_bwd_body.h the way the main
 * kernel's _kv_tile does (staging, a few straight-line q steps, the deferred
 * dQ flush, finish and one epilogue reader), lowers it via
 * rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout so
 * the two outputs can be byte-compared. The spec parser comes from
 * attention_bwd_spec_emit.c (included with ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS
 * and ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY); regenerate the table with
 * `attention_bwd_body_emit.py --print-c-configs`.
 *
 * argv[1] is the config index. Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <map>
#include <string>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.instances.common._attention_bwd_body.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define PROBE_NAME "attn_bwd_body_probe"

static const char* const CONFIGS[] = {
    /* default gfx942 d32 fp16 sv8 */
    "gfx942|head_size:32;dtype:fp16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx942 d32 bf16 sv8 */
    "gfx942|head_size:32;dtype:bf16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx942 d32 bf16 sv1 */
    "gfx942|head_size:32;dtype:bf16;stage_vec:1;global_path:vgpr;ring_depth:1",
    /* default gfx942 d32 thd atomic */
    "gfx942|head_size:32;seq_mode:thd;dkv_mode:atomic;global_path:vgpr;ring_depth:1",
    /* default gfx942 d64 fp16 sv8 */
    "gfx942|head_size:64;dtype:fp16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx942 d64 bf16 sv8 */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx942 d64 bf16 sv1 */
    "gfx942|head_size:64;dtype:bf16;stage_vec:1;global_path:vgpr;ring_depth:1",
    /* default gfx942 d64 thd atomic */
    "gfx942|head_size:64;seq_mode:thd;dkv_mode:atomic;global_path:vgpr;ring_depth:1",
    /* default gfx942 d64 no mask, runtime scale */
    "gfx942|head_size:64;mask_class:none;static_scale:runtime;global_path:vgpr;ring_depth:1",
    /* default gfx942 d64 split dQ */
    "gfx942|head_size:64;dq_mode:split;global_path:vgpr;ring_depth:1",
    /* default gfx942 d128 fp16 sv8 */
    "gfx942|head_size:128;dtype:fp16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx942 d128 bf16 sv8 */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx942 d128 bf16 sv1 */
    "gfx942|head_size:128;dtype:bf16;stage_vec:1;global_path:vgpr;ring_depth:1",
    /* default gfx942 d128 thd atomic */
    "gfx942|head_size:128;seq_mode:thd;dkv_mode:atomic;global_path:vgpr;ring_depth:1",
    /* default gfx942 d128 no mask, runtime scale */
    "gfx942|head_size:128;mask_class:none;static_scale:runtime;global_path:vgpr;ring_depth:1",
    /* default gfx942 d128 split dQ */
    "gfx942|head_size:128;dq_mode:split;global_path:vgpr;ring_depth:1",
    /* default gfx950 d32 fp16 sv8 */
    "gfx950|head_size:32;dtype:fp16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx950 d32 bf16 sv8 */
    "gfx950|head_size:32;dtype:bf16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx950 d32 bf16 sv1 */
    "gfx950|head_size:32;dtype:bf16;stage_vec:1;global_path:vgpr;ring_depth:1",
    /* default gfx950 d32 thd atomic */
    "gfx950|head_size:32;seq_mode:thd;dkv_mode:atomic;global_path:vgpr;ring_depth:1",
    /* default gfx950 d64 fp16 sv8 */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx950 d64 bf16 sv8 */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx950 d64 bf16 sv1 */
    "gfx950|head_size:64;dtype:bf16;stage_vec:1;global_path:vgpr;ring_depth:1",
    /* default gfx950 d64 thd atomic */
    "gfx950|head_size:64;seq_mode:thd;dkv_mode:atomic;global_path:vgpr;ring_depth:1",
    /* default gfx950 d64 no mask, runtime scale */
    "gfx950|head_size:64;mask_class:none;static_scale:runtime;global_path:vgpr;ring_depth:1",
    /* default gfx950 d64 split dQ */
    "gfx950|head_size:64;dq_mode:split;global_path:vgpr;ring_depth:1",
    /* default gfx950 d128 fp16 sv8 */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx950 d128 bf16 sv8 */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;global_path:vgpr;ring_depth:1",
    /* default gfx950 d128 bf16 sv1 */
    "gfx950|head_size:128;dtype:bf16;stage_vec:1;global_path:vgpr;ring_depth:1",
    /* default gfx950 d128 thd atomic */
    "gfx950|head_size:128;seq_mode:thd;dkv_mode:atomic;global_path:vgpr;ring_depth:1",
    /* default gfx950 d128 no mask, runtime scale */
    "gfx950|head_size:128;mask_class:none;static_scale:runtime;global_path:vgpr;ring_depth:1",
    /* default gfx950 d128 split dQ */
    "gfx950|head_size:128;dq_mode:split;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/causal_tl/gqa/unequal/fixed/8/t16/full */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:32;edge_tiles:"
    "False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;"
    "ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t16/over */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t64/over */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/none/mha/square/fixed/8/t16/over */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "192x192;agpr_route:asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;"
    "block_m:16;block_n:128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;"
    "warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over */
    "gfx942|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/causal_br/gqa/square/fixed/8/t64/full */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;"
    "atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_"
    "path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_"
    "plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t16/full */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;"
    "atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_"
    "path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_"
    "plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/none/mha/square/thd/8/t16/over */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_"
    "n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;"
    "waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/fp16/none/mha/square/thd/8/t64/over */
    "gfx942|head_size:64;dtype:fp16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_"
    "n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;"
    "waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/causal_br/gqa/square/fixed/8/t16/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/causal_br/gqa/unequal/fixed/8/t16/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;step_order:head_outer;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t4/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:"
    "False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_"
    "layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/none/gqa/square/fixed/8/t64/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/none/gqa/unequal/fixed/8/t64/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/none/mha/square/fixed/8/t16/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "192x192;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:16;block_n:192;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/fp16/causal_br/gqa/unequal/fixed/8/t64/over */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True;global_path:vgpr;ring_depth:1",
    /* tuned general/128/fp16/causal_tl/gqa/square/fixed/8/t4/full */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:"
    "False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_"
    "layout:head_major;lb_interleave:True;global_path:vgpr;ring_depth:1",
    /* tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:"
    "192;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:iterative-ilp;setprio:False;stats_fma:"
    "True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:"
    "1x4;waves:4;waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/causal_br/gqa/unequal/fixed/8/t16/full */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;static_scale:convert;acc_in_lds:False;agpr_alloc:"
    "None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;"
    "edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;lds_order:phase_low;lds_"
    "swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:"
    "None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_"
    "grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_interleave:True;"
    "global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t4/full */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;"
    "atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_"
    "path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_"
    "read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t16/full */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t16/over */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t4/over */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t64/full */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:2x2;waves:4;waves_per_eu:2;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t64/over */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/bf16/none/mha/square/thd/8/t4/over */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_"
    "n:192;bwd_exp2:fast;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;lds_"
    "swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:"
    "None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_"
    "grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major;worklist:True;"
    "global_path:vgpr;ring_depth:1",
    /* tuned general/64/fp16/none/mha/square/fixed/8/t16/full */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_"
    "grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/fp16/none/mha/square/fixed/8/t4/full */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/64/fp16/none/mha/square/thd/8/t64/over */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_"
    "n:192;bwd_exp2:fast;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:"
    "relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_"
    "fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_"
    "g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major;global_path:vgpr;ring_depth:1",
    /* tuned general/128/bf16/causal_tl/gqa/unequal/fixed/8/t16/full (ring) */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:32;edge_tiles:"
    "False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;"
    "ws_layout:head_major",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t16/over (ring) */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t64/over (ring) */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    /* tuned general/128/bf16/none/mha/square/fixed/8/t16/over (ring) */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "192x192;agpr_route:asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;"
    "block_m:16;block_n:128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;"
    "warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over (ring) */
    "gfx942|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    /* tuned general/64/bf16/causal_br/gqa/square/fixed/8/t64/full (ring) */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;"
    "atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_"
    "path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_"
    "plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t16/full (ring) */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;"
    "atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_"
    "path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_"
    "plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True",
    /* tuned general/64/bf16/none/mha/square/thd/8/t16/over (ring) */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_"
    "n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;"
    "waves_per_eu:1;ws_layout:head_major",
    /* tuned general/64/fp16/none/mha/square/thd/8/t64/over (ring) */
    "gfx942|head_size:64;dtype:fp16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_"
    "n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;"
    "waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/causal_br/gqa/square/fixed/8/t16/over (ring) */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/causal_br/gqa/unequal/fixed/8/t16/over (ring) */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;step_order:head_outer;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t4/over (ring) */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:"
    "False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_"
    "layout:head_major",
    /* tuned general/128/bf16/none/gqa/square/fixed/8/t64/over (ring) */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/none/gqa/unequal/fixed/8/t64/over (ring) */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/none/mha/square/fixed/8/t16/over (ring) */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "192x192;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:16;block_n:192;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/fp16/causal_br/gqa/unequal/fixed/8/t64/over (ring) */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    /* tuned general/128/fp16/causal_tl/gqa/square/fixed/8/t4/full (ring) */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:"
    "False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_"
    "layout:head_major;lb_interleave:True",
    /* tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over (ring) */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:"
    "192;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:iterative-ilp;setprio:False;stats_fma:"
    "True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:"
    "1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/64/bf16/causal_br/gqa/unequal/fixed/8/t16/full (ring) */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;static_scale:convert;acc_in_lds:False;agpr_alloc:"
    "None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;"
    "edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;lds_order:phase_low;lds_"
    "swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:"
    "None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_"
    "grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_interleave:True",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t4/full (ring) */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;"
    "atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_"
    "path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_"
    "read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t16/full (ring) */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t16/over (ring) */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t4/over (ring) */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t64/full (ring) */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:2x2;waves:4;waves_per_eu:2;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t64/over (ring) */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/thd/8/t4/over (ring) */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_"
    "n:192;bwd_exp2:fast;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;lds_"
    "swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:"
    "None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_"
    "grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major;worklist:True",
    /* tuned general/64/fp16/none/mha/square/fixed/8/t16/full (ring) */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_"
    "grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/64/fp16/none/mha/square/fixed/8/t4/full (ring) */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major",
    /* tuned general/64/fp16/none/mha/square/thd/8/t64/over (ring) */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_"
    "n:192;bwd_exp2:fast;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:"
    "relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_"
    "fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_"
    "g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major",
    /* steps ei gfx942 d64 */
    "gfx942|head_size:64;probe_steps:ei",
    /* steps iee gfx950 d128 */
    "gfx950|head_size:128;probe_steps:iee",
    /* steps ie gfx942 d64 edge tiles */
    "gfx942|head_size:64;edge_tiles:True;probe_steps:ie",
    /* stage_order g3g1g4 gfx942 d64 */
    "gfx942|head_size:64;stage_order:g3g1g4;probe_steps:ei",
    /* stage_order g4g1g3 gfx942 d64 */
    "gfx942|head_size:64;stage_order:g4g1g3;probe_steps:ei",
    /* stage_order g1g4g3 gfx942 d64 */
    "gfx942|head_size:64;stage_order:g1g4g3;probe_steps:ei",
    /* stage_order g3g4g1 gfx942 d64 */
    "gfx942|head_size:64;stage_order:g3g4g1;probe_steps:ei",
    /* stage_order g4g3g1 gfx942 d64 */
    "gfx942|head_size:64;stage_order:g4g3g1;probe_steps:ei",
    /* stage_order g4g3g1 gfx950 lds route */
    "gfx950|head_size:128;stage_order:g4g3g1;pt_route:lds",
    /* stage_order g3g1g4 split dQ */
    "gfx942|head_size:64;stage_order:g3g1g4;dq_mode:split",
    /* qdo_stage loads_first gfx942 */
    "gfx942|head_size:64;qdo_stage:loads_first;probe_steps:ei",
    /* qdo_stage joint gfx942 */
    "gfx942|head_size:64;qdo_stage:joint",
    /* qdo_stage joint gfx950 d64 */
    "gfx950|head_size:64;qdo_stage:joint",
    /* qdo_stage loads_first gfx950 head pack */
    "gfx950|head_size:128;qdo_stage:loads_first;head_pack:True",
    /* qdo_stage joint gfx942 block_m 32 */
    "gfx942|head_size:64;qdo_stage:joint;block_m:32",
    /* g4 slice gfx942 d64 */
    "gfx942|head_size:64;g4_slicing:slice;block_k4:32",
    /* g4 slice_db gfx942 d64 */
    "gfx942|head_size:64;g4_slicing:slice_db;block_k4:32",
    /* g4 slice_db gfx950 d128 */
    "gfx950|head_size:128;g4_slicing:slice_db;block_k4:64",
    /* g4 slice_db_g3 gfx950 d128 */
    "gfx950|head_size:128;g4_slicing:slice_db_g3",
    /* g4 slice_db_g3 g3g4g1 gfx942 */
    "gfx942|head_size:64;g4_slicing:slice_db_g3;stage_order:g3g4g1;block_k4:32",
    /* g4 slice gfx950 lds route */
    "gfx950|head_size:128;g4_slicing:slice;pt_route:lds;block_k4:64",
    /* acc_in_lds gfx942 d32 */
    "gfx942|head_size:32;acc_in_lds:True;probe_steps:ei",
    /* acc_in_lds gfx950 d32 kt lds */
    "gfx950|head_size:32;acc_in_lds:True;kt_source:lds",
    /* acc_in_lds gfx950 d32 kv lds */
    "gfx950|head_size:32;acc_in_lds:True;kv_residency:lds;kt_source:lds",
    /* acc_in_lds gfx942 image epilogue */
    "gfx942|head_size:32;acc_in_lds:True;probe_epi:image",
    /* acc_in_lds atomic pair epilogue */
    "gfx942|head_size:32;acc_in_lds:True;dkv_mode:atomic;dkv_epilogue:lds_staged_pair;probe_epi:"
    "slots",
    /* dq_issue fragment gfx942 */
    "gfx942|head_size:64;dq_issue:fragment",
    /* dq_issue fragment_regs gfx950 */
    "gfx950|head_size:128;dq_issue:fragment_regs",
    /* s_in_agpr gfx950 d32 asm gemm hoist */
    "gfx950|head_size:32;agpr_alloc:128x128;agpr_route:asm_g13;s_in_agpr:True;asm_tail_nop:8;asm_"
    "cluster:gemm;asm_load_hoist:True",
    /* s_in_agpr gfx942 d64 asm */
    "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;s_in_agpr:True",
    /* asm_g13 gfx942 tail nop 7 */
    "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:7",
    /* asm_g13 gfx942 d128 gemm cluster */
    "gfx942|head_size:128;agpr_alloc:192x192;agpr_route:asm_g13;asm_cluster:gemm",
    /* asm_g13 hoist g3g1g4 */
    "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;asm_load_hoist:True;stage_order:"
    "g3g1g4",
    /* agpr_fence gfx950 d128 */
    "gfx950|head_size:128;agpr_alloc:128x128;agpr_route:agpr_fence;probe_steps:ei",
    /* vgpr_form gfx942 d128 */
    "gfx942|head_size:128;agpr_alloc:192x192;agpr_route:vgpr_form",
    /* stats_load step_top gfx942 */
    "gfx942|head_size:64;stats_load:step_top;probe_steps:ei",
    /* stats_load step_top gfx950 head pack */
    "gfx950|head_size:128;stats_load:step_top;head_pack:True",
    /* ws_bounds buffer gfx942 */
    "gfx942|head_size:64;ws_bounds:buffer;probe_steps:ei",
    /* stats_vec 4 stats_fma gfx950 */
    "gfx950|head_size:128;stats_vec:4;stats_fma:True",
    /* stats_vec 2 gfx942 */
    "gfx942|head_size:64;stats_vec:2",
    /* token-major workspace gfx942 */
    "gfx942|head_size:64;ws_layout:token_major;probe_steps:ei",
    /* stage_table_atomics gfx942 */
    "gfx942|head_size:64;sched:stage_table_atomics;probe_steps:eie",
    /* stage_table_atomics gfx950 head pack */
    "gfx950|head_size:128;sched:stage_table_atomics;head_pack:True;probe_steps:ee",
    /* stage_table gfx942 setprio */
    "gfx942|head_size:64;sched:stage_table;setprio:True;probe_steps:ei",
    /* fences_only gfx950 kt lds */
    "gfx950|head_size:128;sched:fences_only;kt_source:lds",
    /* iglp0 gfx942 */
    "gfx942|head_size:64;sched:iglp0",
    /* iglp1 gfx950 */
    "gfx950|head_size:128;sched:iglp1",
    /* stage_table split dQ gfx942 */
    "gfx942|head_size:64;sched:stage_table;dq_mode:split",
    /* counted early gfx942 */
    "gfx942|head_size:64;loop_sync:counted;probe_steps:ei",
    /* counted late gfx950 */
    "gfx950|head_size:128;loop_sync:counted;counted_issue:late;probe_steps:ee",
    /* counted early deferred gfx950 */
    "gfx950|head_size:128;loop_sync:counted;sched:stage_table_atomics;probe_steps:ee",
    /* counted gfx942 head pack */
    "gfx942|head_size:64;loop_sync:counted;head_pack:True",
    /* loop_sync lds_only gfx942 */
    "gfx942|head_size:64;loop_sync:lds_only;probe_steps:ei",
    /* barrier merge early gfx942 */
    "gfx942|head_size:64;barrier_merge:on;ring_depth:2;probe_steps:eie",
    /* barrier merge late gfx950 */
    "gfx950|head_size:128;barrier_merge:on;ring_depth:2;ring_issue:late;probe_steps:ee",
    /* barrier merge stage_table gfx942 */
    "gfx942|head_size:64;barrier_merge:on;ring_depth:2;sched:stage_table;probe_steps:ee",
    /* barrier merge deferred gfx950 */
    "gfx950|head_size:128;barrier_merge:on;ring_depth:2;sched:stage_table_atomics;probe_steps:ee",
    /* barrier merge head pack gfx942 */
    "gfx942|head_size:64;barrier_merge:on;ring_depth:2;head_pack:True",
    /* ring vgpr R2 gfx942 */
    "gfx942|head_size:64;ring_depth:2;probe_steps:eie",
    /* ring vgpr R3 gfx950 d64 */
    "gfx950|head_size:64;ring_depth:3;probe_steps:eiee",
    /* ring vgpr R3 one step gfx950 */
    "gfx950|head_size:128;ring_depth:3",
    /* ring vgpr R2 head pack gfx942 */
    "gfx942|head_size:64;ring_depth:2;head_pack:True;probe_steps:ee",
    /* ring vgpr R3 head pack gfx950 */
    "gfx950|head_size:128;ring_depth:3;head_pack:True;probe_steps:ee",
    /* ring vgpr R2 stage_table gfx942 */
    "gfx942|head_size:64;ring_depth:2;sched:stage_table;probe_steps:ee",
    /* ring vgpr R3 deferred gfx950 */
    "gfx950|head_size:128;ring_depth:3;sched:stage_table_atomics;probe_steps:eee",
    /* ring vgpr R2 lds route gfx950 */
    "gfx950|head_size:64;ring_depth:2;pt_route:lds;probe_steps:ie",
    /* ring dma R1 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:1;probe_steps:ei",
    /* ring dma R2 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;probe_steps:eie",
    /* ring dma R3 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:3;probe_steps:eiee",
    /* ring dma R2 d64 bf16 gfx950 */
    "gfx950|head_size:64;dtype:bf16;global_path:dma;ring_depth:2;probe_steps:ee",
    /* ring dma R2 stage_table gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;sched:stage_table;probe_steps:ee",
    /* ring dma R2 deferred gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;sched:stage_table_atomics;probe_steps:ee",
    /* ring dma R3 split dQ gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:3;dq_mode:split;probe_steps:ee",
    /* ring dma R2 d32 short rows gfx950 */
    "gfx950|head_size:32;global_path:dma;ring_depth:2;probe_steps:ee",
    /* ring dma R2 one wave 16x16 gfx950 */
    "gfx950|head_size:32;global_path:dma;ring_depth:2;waves:1;block_m:16;block_n:16;warp_grid_g4:"
    "1x1;warp_grid_g02:1x1;warp_grid_g13:1x1;block_k4:16;atom_g4:16x16x16;atom_g02:16x16x16",
    /* ring vgpr R2 d32 joint gfx942 */
    "gfx942|head_size:32;ring_depth:2;probe_steps:ee",
    /* ring vgpr R3 d32 joint gfx950 */
    "gfx950|head_size:32;ring_depth:3;probe_steps:eee",
    /* one wave 16x16 gfx942 */
    "gfx942|head_size:64;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;warp_grid_g02:"
    "1x1;warp_grid_g13:1x1;probe_steps:ei",
    /* rejected: dma ring with head pack */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;head_pack:True",
    /* xt_lds gfx942 d64 */
    "gfx942|head_size:64;transpose_source:xt_lds;probe_steps:ei",
    /* xt_lds gfx942 d128 kt lds */
    "gfx942|head_size:128;transpose_source:xt_lds;kt_source:lds",
    /* xt_lds gfx950 d128 split dQ */
    "gfx950|head_size:128;transpose_source:xt_lds;dq_mode:split",
    /* xt_lds gfx950 lds route */
    "gfx950|head_size:64;transpose_source:xt_lds;pt_route:lds",
    /* lds_plain gfx950 d64 */
    "gfx950|head_size:64;transpose_source:lds_plain",
    /* pt_route lds gfx942 d64 */
    "gfx942|head_size:64;pt_route:lds;probe_steps:ei",
    /* kv_residency lds gfx942 */
    "gfx942|head_size:64;kv_residency:lds;kt_source:lds",
    /* kt_source lds gfx950 d64 */
    "gfx950|head_size:64;kt_source:lds",
    /* replicated G0 grid gfx942 */
    "gfx942|head_size:64;pt_route:lds;waves:4;block_n:64;warp_grid_g02:1x2;warp_grid_g4:1x4",
    /* no mask gfx942 runtime scale */
    "gfx942|head_size:64;mask_class:none;static_scale:runtime;probe_steps:ei",
    /* no mask gfx950 dq before atomic */
    "gfx950|head_size:128;mask_class:none;static_scale:dq_before_atomic",
    /* bf16 gfx942 rne_noguard */
    "gfx942|head_size:64;dtype:bf16;bf16_cast:rne_noguard",
    /* bf16 gfx950 fast exp2 */
    "gfx950|head_size:128;dtype:bf16;bwd_exp2:fast",
    /* thd atomic gfx942 */
    "gfx942|head_size:64;seq_mode:thd;dkv_mode:atomic;probe_epi:slots",
    /* thd gfx950 head pack interior clamp */
    "gfx950|head_size:128;seq_mode:thd;head_pack:True;interior_clamp:True;probe_steps:ei",
    /* interior clamp gfx942 */
    "gfx942|head_size:64;interior_clamp:True;probe_steps:i",
    /* stage_vec 1 gfx942 */
    "gfx942|head_size:64;stage_vec:1;narrow_vec:2",
    /* head pack gfx942 d64 */
    "gfx942|head_size:64;head_pack:True;probe_steps:ee",
    /* epilogue rows gfx942 */
    "gfx942|head_size:64;probe_epi:rows",
    /* epilogue quad gfx950 */
    "gfx950|head_size:128;probe_epi:quad",
    /* epilogue half gfx942 */
    "gfx942|head_size:64;probe_epi:half",
    /* epilogue half gfx950 g13 grid 2x2 */
    "gfx950|head_size:64;pt_route:lds;waves:4;warp_grid_g13:2x2;warp_grid_g4:2x2;probe_epi:half",
    /* epilogue slots gfx950 */
    "gfx950|head_size:128;probe_epi:slots",
    /* epilogue none gfx942 */
    "gfx942|head_size:64;probe_epi:none",
    /* packed copy helper gfx942 d64 */
    "gfx942|head_size:64;probe_extra:copy",
    /* packed copy helper gfx950 d128 bf16 */
    "gfx950|head_size:128;dtype:bf16;probe_extra:copy",
    /* epilogue rows acc_in_lds runtime scale */
    "gfx942|head_size:32;acc_in_lds:True;static_scale:runtime;probe_epi:rows",
    /* epilogue rows vec 4 gfx950 */
    "gfx950|head_size:128;dkv_store_vec:4;static_scale:runtime;probe_epi:rows",
    /* epilogue half runtime scale gfx942 */
    "gfx942|head_size:64;static_scale:runtime;probe_epi:half",
    /* epilogue quad runtime scale gfx942 */
    "gfx942|head_size:64;static_scale:runtime;probe_epi:quad",
    /* barrier merge gfx950 d128 block_m 32 */
    "gfx950|head_size:128;barrier_merge:on;ring_depth:2;block_m:32;probe_steps:ei",
    /* rejected: unknown stage order */
    "gfx942|head_size:64;stage_order:g1g1g4",
    /* rejected: g4 slicing with acc_in_lds */
    "gfx942|head_size:32;g4_slicing:slice;acc_in_lds:True",
    /* rejected: quad epilogue with acc_in_lds */
    "gfx942|head_size:32;acc_in_lds:True;probe_epi:quad",
    /* rejected: image epilogue without acc_in_lds */
    "gfx942|head_size:64;probe_epi:image",
    /* rdna gfx1201 d64 */
    "gfx1201|head_size:64;probe_steps:ei",
    /* rdna gfx1151 d64 bf16 */
    "gfx1151|head_size:64;dtype:bf16",
    /* sweep stage_order g1g3g4 gfx950 lds route */
    "gfx950|head_size:128;pt_route:lds;stage_order:g1g3g4;probe_steps:ie",
    /* sweep stage_order g3g1g4 gfx950 lds route */
    "gfx950|head_size:128;pt_route:lds;stage_order:g3g1g4;probe_steps:ie",
    /* sweep stage_order g4g1g3 gfx950 lds route */
    "gfx950|head_size:128;pt_route:lds;stage_order:g4g1g3;probe_steps:ie",
    /* sweep stage_order g1g4g3 gfx950 lds route */
    "gfx950|head_size:128;pt_route:lds;stage_order:g1g4g3;probe_steps:ie",
    /* sweep stage_order g3g4g1 gfx950 lds route */
    "gfx950|head_size:128;pt_route:lds;stage_order:g3g4g1;probe_steps:ie",
    /* sweep stage_order g4g3g1 gfx950 lds route */
    "gfx950|head_size:128;pt_route:lds;stage_order:g4g3g1;probe_steps:ie",
    /* sweep qdo_stage split gfx942 head pack False */
    "gfx942|head_size:64;qdo_stage:split;head_pack:False;probe_steps:ee",
    /* sweep qdo_stage split gfx942 head pack True */
    "gfx942|head_size:64;qdo_stage:split;head_pack:True;probe_steps:ee",
    /* sweep qdo_stage loads_first gfx942 head pack False */
    "gfx942|head_size:64;qdo_stage:loads_first;head_pack:False;probe_steps:ee",
    /* sweep qdo_stage loads_first gfx942 head pack True */
    "gfx942|head_size:64;qdo_stage:loads_first;head_pack:True;probe_steps:ee",
    /* sweep qdo_stage joint gfx942 head pack False */
    "gfx942|head_size:64;qdo_stage:joint;head_pack:False;probe_steps:ee",
    /* sweep qdo_stage joint gfx942 head pack True */
    "gfx942|head_size:64;qdo_stage:joint;head_pack:True;probe_steps:ee",
    /* sweep g4_slicing step gfx950 d64 */
    "gfx950|head_size:64;g4_slicing:step;block_k4:32;stage_order:g1g3g4",
    /* sweep g4_slicing slice gfx950 d64 */
    "gfx950|head_size:64;g4_slicing:slice;block_k4:32;stage_order:g1g3g4",
    /* sweep g4_slicing slice_db gfx950 d64 */
    "gfx950|head_size:64;g4_slicing:slice_db;block_k4:32;stage_order:g1g3g4",
    /* sweep g4_slicing slice_db_g3 gfx950 d64 */
    "gfx950|head_size:64;g4_slicing:slice_db_g3;block_k4:32;stage_order:g1g3g4",
    /* sweep sched none gfx950 d128 bf16 */
    "gfx950|head_size:128;dtype:bf16;sched:none;probe_steps:ee",
    /* sweep sched iglp0 gfx950 d128 bf16 */
    "gfx950|head_size:128;dtype:bf16;sched:iglp0;probe_steps:ee",
    /* sweep sched iglp1 gfx950 d128 bf16 */
    "gfx950|head_size:128;dtype:bf16;sched:iglp1;probe_steps:ee",
    /* sweep sched fences_only gfx950 d128 bf16 */
    "gfx950|head_size:128;dtype:bf16;sched:fences_only;probe_steps:ee",
    /* sweep sched stage_table gfx950 d128 bf16 */
    "gfx950|head_size:128;dtype:bf16;sched:stage_table;probe_steps:ee",
    /* sweep sched stage_table_atomics gfx950 d128 bf16 */
    "gfx950|head_size:128;dtype:bf16;sched:stage_table_atomics;probe_steps:ee",
    /* sweep loop_sync full gfx950 d64 kt lds */
    "gfx950|head_size:64;kt_source:lds;loop_sync:full;probe_steps:ei",
    /* sweep loop_sync lds_only gfx950 d64 kt lds */
    "gfx950|head_size:64;kt_source:lds;loop_sync:lds_only;probe_steps:ei",
    /* sweep loop_sync counted gfx950 d64 kt lds */
    "gfx950|head_size:64;kt_source:lds;loop_sync:counted;probe_steps:ei",
    /* sweep dq_issue block acc_in_lds False gfx942 d32 */
    "gfx942|head_size:32;dq_issue:block;acc_in_lds:False",
    /* sweep dq_issue block acc_in_lds True gfx942 d32 */
    "gfx942|head_size:32;dq_issue:block;acc_in_lds:True",
    /* sweep dq_issue fragment acc_in_lds False gfx942 d32 */
    "gfx942|head_size:32;dq_issue:fragment;acc_in_lds:False",
    /* sweep dq_issue fragment acc_in_lds True gfx942 d32 */
    "gfx942|head_size:32;dq_issue:fragment;acc_in_lds:True",
    /* sweep dq_issue fragment_regs acc_in_lds False gfx942 d32 */
    "gfx942|head_size:32;dq_issue:fragment_regs;acc_in_lds:False",
    /* sweep dq_issue fragment_regs acc_in_lds True gfx942 d32 */
    "gfx942|head_size:32;dq_issue:fragment_regs;acc_in_lds:True",
    /* sweep mask band edge_tiles False gfx942 d128 */
    "gfx942|head_size:128;mask_class:band;edge_tiles:False;probe_steps:ei",
    /* sweep mask band edge_tiles True gfx942 d128 */
    "gfx942|head_size:128;mask_class:band;edge_tiles:True;probe_steps:ei",
    /* sweep mask none edge_tiles False gfx942 d128 */
    "gfx942|head_size:128;mask_class:none;edge_tiles:False;probe_steps:ei",
    /* sweep mask none edge_tiles True gfx942 d128 */
    "gfx942|head_size:128;mask_class:none;edge_tiles:True;probe_steps:ei",
    /* sweep dtype fp16 transpose tr_read gfx950 d64 */
    "gfx950|head_size:64;dtype:fp16;transpose_source:tr_read",
    /* sweep dtype fp16 transpose lds_plain gfx950 d64 */
    "gfx950|head_size:64;dtype:fp16;transpose_source:lds_plain",
    /* sweep dtype fp16 transpose xt_lds gfx950 d64 */
    "gfx950|head_size:64;dtype:fp16;transpose_source:xt_lds",
    /* sweep dtype bf16 transpose tr_read gfx950 d64 */
    "gfx950|head_size:64;dtype:bf16;transpose_source:tr_read",
    /* sweep dtype bf16 transpose lds_plain gfx950 d64 */
    "gfx950|head_size:64;dtype:bf16;transpose_source:lds_plain",
    /* sweep dtype bf16 transpose xt_lds gfx950 d64 */
    "gfx950|head_size:64;dtype:bf16;transpose_source:xt_lds",
    /* qdo_stage loads_first full rounds gfx942 d128 block_m 32 */
    "gfx942|head_size:128;qdo_stage:loads_first;block_m:32;probe_steps:ei",
    /* qdo_stage loads_first full rounds gfx942 d64 four waves */
    "gfx942|head_size:64;qdo_stage:loads_first;block_m:64;waves:4;warp_grid_g4:1x4;warp_grid_g02:"
    "1x4;warp_grid_g13:4x1",
    /* qdo_stage loads_first full rounds gfx950 d128 head pack */
    "gfx950|head_size:128;qdo_stage:loads_first;block_m:32;head_pack:True;probe_steps:ee",
    /* barrier merge g3g1g4 gfx942 */
    "gfx942|head_size:64;barrier_merge:on;ring_depth:2;stage_order:g3g1g4;probe_steps:ei",
    /* rejected: unknown arch gfx1100 */
    "gfx1100|head_size:64",
    /* rejected: unknown transpose_source */
    "gfx942|head_size:64;transpose_source:bogus",
    /* rejected: counted with g4 slice */
    "gfx942|head_size:64;loop_sync:counted;g4_slicing:slice;block_k4:32",
    /* rejected: counted with loads_first */
    "gfx942|head_size:64;loop_sync:counted;qdo_stage:loads_first",
    /* rejected: barrier merge ring depth 3 */
    "gfx942|head_size:64;barrier_merge:on;ring_depth:3",
};

#define ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS 1
#define ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY 1
#include "attention_bwd_spec_emit.c"

typedef rocke_value_t* V;
typedef rocke_attn_bwd_iv_t IV;

static IV iv(V v)
{
    return rocke_attn_bwd_iv_val(v);
}

/* The config without its probe_* keys (into *probe). */
static std::string split_probe(const char* config, std::map<std::string, std::string>* probe)
{
    std::string c(config);
    size_t bar = c.find('|');
    std::string out = c.substr(0, bar + 1);
    std::string rest = c.substr(bar + 1);
    size_t start = 0;
    bool first = true;
    while(start <= rest.size())
    {
        size_t semi = rest.find(';', start);
        std::string item
            = rest.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if(item.compare(0, 6, "probe_") == 0)
        {
            size_t colon = item.find(':');
            (*probe)[item.substr(0, colon)] = item.substr(colon + 1);
        }
        else if(!item.empty())
        {
            out += (first ? "" : ";") + item;
            first = false;
        }
        if(semi == std::string::npos)
            break;
        start = semi + 1;
    }
    return out;
}

static void fail(rocke_ir_builder_t* b, const rocke_attn_bwd_err_t& e)
{
    rocke_i_set_err_msg(b, e.code, e.msg);
}

/* attention_bwd._decode_side */
static rocke_attn_bwd_seq_info_t side(
    rocke_ir_builder_t* b, const rocke_attn_bwd_spec_t& s, int* seq_guards, V bz, const char* which)
{
    bool q = strcmp(which, "q") == 0;
    rocke_attn_bwd_decode_args_t args = rocke_attn_bwd_decode_args_default();
    args.s_max = iv(rocke_b_get_param(b, q ? "S_q_max" : "S_kv_max"));
    args.seq_ptr = rocke_b_get_param(b, q ? "SEQ_Q" : "SEQ_KV");
    args.has_len = iv(rocke_b_get_param(b, "has_len"));
    args.len_stride = iv(rocke_b_get_param(b, "len_stride"));
    if(strcmp(s.seq_mode, "thd") == 0)
    {
        args.off_ptr = rocke_b_get_param(b, q ? "OFF_Q" : "OFF_KV");
        args.off64 = iv(rocke_b_get_param(b, "off64"));
        args.mult = iv(rocke_b_get_param(b, q ? "q_mult" : "kv_mult"));
        args.div = iv(rocke_b_get_param(b, q ? "q_div" : "kv_div"));
        args.ws_seg = iv(rocke_b_get_param(b, "ws_seg"));
        return rocke_attn_bwd_decode_thd(b, seq_guards, bz, &args);
    }
    return rocke_attn_bwd_decode_batched(b, seq_guards, bz, &args);
}

struct Probe
{
    rocke_ir_builder_t* b;
    const rocke_attn_bwd_spec_t* s;
    rocke_attn_bwd_tile_step_t* step;
    std::string steps;
    V qt_start, h0, n_h, per, rows_per_head;
    const rocke_attn_bwd_strided_tensor_addr_t* q_addr;
    const rocke_attn_bwd_strided_tensor_addr_t* do_addr;
    /* storage of the packed steps (inputs() returns a pointer into it) */
    rocke_attn_bwd_packed_step_t packed[64];
    int n_packed;
};

/* inputs(i, edge=, slabs=) */
static rocke_attn_bwd_q_step_inputs_t inputs(Probe* p, int i, bool edge, bool slabs)
{
    rocke_ir_builder_t* b = p->b;
    const rocke_attn_bwd_spec_t& s = *p->s;
    V qt = p->qt_start;
    if(i != 0)
    {
        V ci = rocke_b_const_i32(b, i);
        qt = rocke_b_add(b, p->qt_start, ci);
    }
    V ckm = rocke_b_const_i32(b, s.block_m);
    V q0 = rocke_b_mul(b, qt, ckm);
    rocke_attn_bwd_q_step_inputs_t in;
    memset(&in, 0, sizeof(in));
    in.q0 = q0;
    in.h = p->h0;
    if(s.head_pack == 1)
    {
        rocke_attn_bwd_packed_step_t* ps = &p->packed[p->n_packed++ % 64];
        ps->rows.head0 = p->h0;
        ps->rows.n_heads = rocke_b_smin(b, p->per, p->n_h);
        ps->rows.rows_per_head = p->rows_per_head;
        ps->rows.row0 = q0;
        ps->q_addr = p->q_addr;
        ps->do_addr = p->do_addr;
        in.packed = ps;
        return in;
    }
    if(!slabs)
        return in;
    bool clamp = edge || s.interior_clamp;
    in.has_slabs = true;
    in.q_slab = rocke_attn_bwd_strided_tensor_addr_slab(p->q_addr, iv(p->h0), iv(q0), clamp);
    in.do_slab = rocke_attn_bwd_strided_tensor_addr_slab(p->do_addr, iv(p->h0), iv(q0), clamp);
    return in;
}

static bool step_loads(Probe* p, int i, rocke_attn_bwd_staged_step_t* out)
{
    bool edge = p->steps[i] == 'e';
    rocke_attn_bwd_q_step_inputs_t t = inputs(p, i, edge, true);
    return rocke_attn_bwd_tile_step_counted_loads(p->step,
                                                  p->b,
                                                  t.h,
                                                  t.q0,
                                                  t.has_slabs ? &t.q_slab : NULL,
                                                  t.has_slabs ? &t.do_slab : NULL,
                                                  t.packed,
                                                  edge,
                                                  out);
}

struct NextLoads
{
    Probe* p;
    int i;
};

static bool next_loads_fn(void* ctx, rocke_ir_builder_t* b, rocke_attn_bwd_staged_step_t* out)
{
    (void)b;
    NextLoads* n = (NextLoads*)ctx;
    return step_loads(n->p, n->i, out);
}

/* Build the probe kernel of config `idx` into `b`. 0 on success. */
static int build(int idx, rocke_ir_builder_t* b)
{
    std::map<std::string, std::string> probe;
    std::string config = split_probe(CONFIGS[idx], &probe);
    std::string arch_s;
    rocke_attn_bwd_spec_t spec;
    if(!parse_config(config.c_str(), &arch_s, &spec))
        return -1;
    const char* arch = keep(arch_s);
    std::string steps = probe.count("probe_steps") ? probe["probe_steps"] : "e";
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_spec_init(&spec, &e) != ROCKE_OK)
    {
        fail(b, e);
        return 1;
    }
    static rocke_attn_bwd_spec_t s;
    if(rocke_attn_bwd_validate_spec(&spec, arch, &s, &e) != ROCKE_OK)
    {
        fail(b, e);
        return 1;
    }
    std::string epi
        = probe.count("probe_epi") ? probe["probe_epi"] : (s.acc_in_lds == 1 ? "rows" : "cells");
    static rocke_attn_bwd_tile_geometry_t geom;
    if(!rocke_attn_bwd_tile_geometry_from_spec(b, &s, arch, &geom))
        return 1;
    static rocke_attn_bwd_lds_plan_t plan;
    if(rocke_attn_bwd_lds_plan(&s, arch, &plan, &e) != ROCKE_OK)
    {
        fail(b, e);
        return 1;
    }
    const rocke_type_t* io = rocke_attn_bwd_ir_type(b, s.dtype);
    int D = s.head_size, km = s.block_m, kn = s.block_n;

    /* _declare_main(b, io) */
    const char* names[ROCKE_ATTN_BWD_PARAMS_MAX];
    const char* formats[ROCKE_ATTN_BWD_PARAMS_MAX];
    int n_params = 0;
    if(rocke_attn_bwd_params("main", names, formats, &n_params, &e) != ROCKE_OK)
    {
        fail(b, e);
        return 1;
    }
    static const char* const readonly[] = {"Q",
                                           "K",
                                           "V",
                                           "dO",
                                           "WS_LSE2",
                                           "WS_DSUM",
                                           "SEQ_Q",
                                           "SEQ_KV",
                                           "OFF_Q",
                                           "OFF_KV",
                                           "WORKLIST"};
    std::map<std::string, V> a;
    for(int i = 0; i < n_params; ++i)
    {
        const char* n = names[i];
        const char* f = formats[i];
        V v;
        if(strcmp(f, "Q") == 0)
        {
            const rocke_type_t* pointee;
            static const char* const io_ptrs[] = {"Q", "K", "V", "dO", "dK", "dV"};
            bool is_io = false;
            for(const char* x : io_ptrs)
                is_io = is_io || strcmp(n, x) == 0;
            if(is_io)
                pointee = io;
            else if(strncmp(n, "WS_", 3) == 0)
                pointee = rocke_f32();
            else
                pointee = rocke_i32();
            bool ro = false;
            for(const char* x : readonly)
                ro = ro || strcmp(n, x) == 0;
            rocke_param_opts_t opts;
            memset(&opts, 0, sizeof(opts));
            opts.readonly = true;
            opts.readonly_set = true;
            v = rocke_b_param(b, n, rocke_ptr_type(b, pointee, "global"), ro ? &opts : NULL);
        }
        else if(strcmp(f, "q") == 0)
            v = rocke_b_param(b, n, rocke_i64(), NULL);
        else if(strcmp(f, "f") == 0)
            v = rocke_b_param(b, n, rocke_f32(), NULL);
        else
            v = rocke_b_param(b, n, rocke_i32(), NULL);
        a[n] = v;
    }
    unsigned unit = 0;
    if(rocke_attn_bwd_static_unit_mults(s.static_scale, s.dkv_mode, &unit, &e) != ROCKE_OK)
    {
        fail(b, e);
        return 1;
    }
    if(unit & ROCKE_ATTN_BWD_UNIT_DS_MULT)
        a["ds_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DK_MULT)
        a["dk_mult"] = NULL;
    if(unit & ROCKE_ATTN_BWD_UNIT_DQ_MULT)
        a["dq_mult"] = NULL;

    V tid, lane, wave;
    rocke_attn_bwd_lane_and_wave(
        b, rocke_attn_bwd_tile_geometry_wave_size(&geom), NULL, &tid, &lane, &wave);
    V kt = rocke_b_block_id_x(b);
    V hy = rocke_b_block_id_y(b);
    V bz = rocke_b_block_id_z(b);
    int seq_guards = 0;
    rocke_attn_bwd_seq_info_t seq_q = side(b, s, &seq_guards, bz, "q");
    rocke_attn_bwd_seq_info_t seq_kv = side(b, s, &seq_guards, bz, "kv");
    V hk = hy;
    V h0 = rocke_b_mul(b, hy, a["G"]);
    V n_h = a["G"];

    static rocke_attn_bwd_strided_tensor_addr_t addrs[4];
    const char* tnames[4] = {"Q", "dO", "K", "V"};
    const char* prefixes[4] = {"q", "do", "k", "v"};
    for(int i = 0; i < 4; ++i)
    {
        std::string p = prefixes[i];
        IV strides[3] = {iv(a[p + "_b"]), iv(a[p + "_h"]), iv(a[p + "_t"])};
        if(!rocke_attn_bwd_strided_tensor_addr_init(&addrs[i],
                                                    b,
                                                    a[tnames[i]],
                                                    io,
                                                    strides,
                                                    i < 2 ? &seq_q : &seq_kv,
                                                    s.stage_vec,
                                                    s.narrow_vec))
            return 1;
    }
    bool token_major = strcmp(s.ws_layout, "token_major") == 0;
    rocke_attn_bwd_ws_layout_t layout
        = token_major ? ROCKE_ATTN_BWD_WS_TOKEN_MAJOR : ROCKE_ATTN_BWD_WS_HEAD_MAJOR;
    IV nh = iv(a["h_q"]);
    auto ws = [&](const char* name, int width, rocke_attn_bwd_workspace_addr_t* out) {
        return rocke_attn_bwd_workspace_addr_init(out,
                                                  b,
                                                  a[name],
                                                  iv(a["ws_rows_q"]),
                                                  width,
                                                  seq_q.ws_row0,
                                                  layout,
                                                  token_major ? &nh : NULL);
    };
    rocke_attn_bwd_workspace_addr_t ws_lse2, ws_dsum, ws_dq;
    if(!ws("WS_LSE2", 1, &ws_lse2) || !ws("WS_DSUM", 1, &ws_dsum))
        return 1;
    static rocke_attn_bwd_stats_view_t stats;
    if(!rocke_attn_bwd_stats_view_init(
           &stats, b, &ws_lse2, &ws_dsum, s.stats_vec != 1 ? a["h_q"] : NULL, s.ws_bounds, km))
        return 1;
    static rocke_attn_bwd_dq_sink_t sink;
    if(strcmp(s.dq_mode, "split") == 0)
        rocke_attn_bwd_dq_sink_init_null(&sink);
    else
    {
        if(!ws("WS_DQ", D, &ws_dq))
            return 1;
        bool ok = s.head_pack == 1
                      ? rocke_attn_bwd_dq_sink_init_packed_atomic(
                            &sink, b, &ws_dq, &seq_q, a["dq_mult"])
                      : rocke_attn_bwd_dq_sink_init_atomic(&sink, b, &ws_dq, &seq_q, a["dq_mult"]);
        if(!ok)
            return 1;
    }
    V zero = rocke_b_const_i32(b, 0);
    static rocke_attn_bwd_mask_policy_t mask;
    if(strcmp(s.mask_class, "band") == 0)
    {
        V ne = rocke_b_cmp_ne(b, a["bottom_right"], zero);
        V diff = rocke_b_sub(b, seq_kv.length, seq_q.length);
        V off = rocke_b_select(b, ne, diff, zero);
        rocke_attn_runtime_bounds_t bounds = rocke_attn_runtime_bounds_make(
            rocke_iov_value(seq_q.length), rocke_iov_value(seq_kv.length));
        bounds.diag_offset = rocke_iov_value(off);
        bounds.left_bound = rocke_iov_value(a["left"]);
        bounds.right_bound = rocke_iov_value(a["right"]);
        if(rocke_attn_bwd_runtime_band_init(&mask, b, &bounds, km, kn) != ROCKE_OK)
            return 1;
    }
    else if(rocke_attn_bwd_no_mask_ktail_init(
                &mask, b, rocke_iov_value(seq_q.length), rocke_iov_value(seq_kv.length), km, kn)
            != ROCKE_OK)
        return 1;
    static rocke_attn_bwd_bwd_lds_t lds;
    if(!rocke_attn_bwd_bwd_lds_init(&lds, b, &plan, s.dtype))
        return 1;
    static rocke_attn_bwd_tile_step_t step;
    if(!rocke_attn_bwd_tile_step_init(&step,
                                      b,
                                      &geom,
                                      &mask,
                                      &stats,
                                      &sink,
                                      &lds,
                                      tid,
                                      lane,
                                      wave,
                                      a["scale_log2"],
                                      a["ds_mult"],
                                      true))
        return 1;
    static rocke_attn_bwd_q_ring_t ring;
    bool has_ring = !(strcmp(s.global_path, "vgpr") == 0 && s.ring_depth == 1)
                    && strcmp(s.barrier_merge, "on") != 0;
    if(has_ring)
    {
        /* _cta_setup: the Q / dO prefetch ring */
        int n_atomics = km * D / (geom.waves * rocke_attn_bwd_tile_geometry_wave_size(&geom));
        if(!rocke_attn_bwd_q_ring_init(&ring,
                                       b,
                                       &geom,
                                       &lds,
                                       tid,
                                       lane,
                                       wave,
                                       &stats,
                                       strcmp(s.dq_mode, "atomic") == 0 ? n_atomics : 0)
           || !rocke_attn_bwd_q_ring_check_step(&ring, b, &step.sched))
            return 1;
    }

    /* _kv_tile: the staging, then the q steps */
    V one = rocke_b_const_i32(b, 1);
    V ckn = rocke_b_const_i32(b, kn);
    V k0 = rocke_b_mul(b, kt, ckn);
    V live = rocke_b_cmp_lt(b, k0, seq_kv.length);
    rocke_attn_bwd_row_slab_t k_slab
        = rocke_attn_bwd_strided_tensor_addr_slab(&addrs[2], iv(hk), iv(k0), true);
    rocke_attn_bwd_row_slab_t v_slab
        = rocke_attn_bwd_strided_tensor_addr_slab(&addrs[3], iv(hk), iv(k0), true);
    if(!rocke_attn_bwd_tile_step_stage_kv(&step, b, &k_slab, &v_slab, k0, live))
        return 1;
    if(probe.count("probe_extra"))
    {
        if(probe["probe_extra"] != "copy")
            return -1;
        rocke_attn_bwd_packed_rows_t rows;
        rows.head0 = h0;
        rows.n_heads = n_h;
        rows.rows_per_head = rocke_b_const_i32(b, km);
        rows.row0 = zero;
        rocke_attn_bwd_lds_view_t vq;
        if(!rocke_attn_bwd_bwd_lds_view(b, &lds, "q", D, 0, &vq)
           || !rocke_attn_bwd_copy_packed_rows_to_lds(b,
                                                      &addrs[0],
                                                      &vq,
                                                      &rows,
                                                      km,
                                                      D,
                                                      tid,
                                                      rocke_attn_bwd_tile_geometry_threads(&geom)
                                                          + 64,
                                                      8,
                                                      false))
            return 1;
    }
    static Probe p;
    p = Probe{};
    p.b = b;
    p.s = &s;
    p.step = &step;
    p.steps = steps;
    p.h0 = h0;
    p.n_h = n_h;
    p.q_addr = &addrs[0];
    p.do_addr = &addrs[1];
    if(s.head_pack == 1)
    {
        V pk = a["pack_heads"];
        V pack_on = rocke_b_cmp_ne(b, pk, zero);
        p.per = rocke_b_select(b, pack_on, pk, one);
        V mx = rocke_b_smax(b, seq_q.length, one);
        V ckm = rocke_b_const_i32(b, km);
        p.rows_per_head = rocke_b_select(b, pack_on, mx, ckm);
    }
    V n_qt;
    if(rocke_attn_bwd_mask_q_tile_range(&mask, b, k0, &p.qt_start, &n_qt) != ROCKE_OK)
        return 1;
    rocke_attn_bwd_dkdv_acc_t acc;
    if(!rocke_attn_bwd_tile_step_zero_acc(&step, b, &acc))
        return 1;
    rocke_attn_bwd_pending_dq_t pend;
    bool has_pend = strcmp(s.sched, "stage_table_atomics") == 0;
    if(has_pend
       && !rocke_attn_bwd_tile_step_zero_pending_dq(&step,
                                                    b,
                                                    h0,
                                                    s.head_pack == 1 ? one : NULL,
                                                    s.head_pack == 1 ? p.rows_per_head : NULL,
                                                    &pend))
        return 1;

    int n = (int)steps.size();
    bool counted = strcmp(s.loop_sync, "counted") == 0;
    bool merge = strcmp(s.barrier_merge, "on") == 0;
    rocke_attn_bwd_staged_step_t staged;
    memset(&staged, 0, sizeof(staged));
    rocke_value_t** carried = NULL;
    int n_carried = 0;
    /* PrefetchTiles live as long as the kernel build (the steps borrow them) */
    static rocke_attn_bwd_prefetch_tile_t tiles[256];
    int n_tiles_used = 0;
    auto ring_tile = [&](int i) -> rocke_attn_bwd_prefetch_tile_t* {
        rocke_attn_bwd_q_step_inputs_t t = inputs(&p, i, steps[i] == 'e', true);
        rocke_attn_bwd_prefetch_tile_t* pt = &tiles[n_tiles_used++ % 256];
        pt->q0 = t.q0;
        pt->h = t.h;
        pt->has_slabs = t.has_slabs;
        pt->q_slab = t.q_slab;
        pt->do_slab = t.do_slab;
        pt->packed = t.packed;
        return pt;
    };
    if(has_ring)
    {
        rocke_attn_bwd_prefetch_tile_t pro[2];
        int n_pro = 0;
        pro[n_pro++] = *ring_tile(0);
        if(ring.depth == 3)
            pro[n_pro++] = *ring_tile(n > 1 ? 1 : 0);
        if(ring.has_dma && ring.depth == 1)
            n_pro = 0;
        if(!rocke_attn_bwd_q_ring_prologue(
               &ring, b, n_qt, pro, n_pro, "bwd_q", &carried, &n_carried))
            return 1;
    }
    if(merge)
    {
        rocke_attn_bwd_q_step_inputs_t first = inputs(&p, 0, steps[0] == 'e', true);
        if(!rocke_attn_bwd_tile_step_ring_prologue(&step, b, &first)
           || !rocke_attn_bwd_tile_step_ring_publish(&step, b))
            return 1;
    }
    if(counted)
    {
        /* (the main kernel's guard loop starts from these zeros) */
        int nz = rocke_attn_bwd_tile_step_counted_zero_len(&step);
        rocke_value_t** zs = (rocke_value_t**)calloc(nz ? nz : 1, sizeof(rocke_value_t*));
        bool zok = rocke_attn_bwd_tile_step_counted_zero(&step, b, zs);
        free(zs);
        if(!zok)
            return 1;
        if(!step_loads(&p, 0, &staged))
            return 1;
        rocke_b_s_waitcnt(b, 0, -1, -1);
    }
    for(int i = 0; i < n; ++i)
    {
        bool edge = steps[i] == 'e';
        int nxt_i = i + 1 < n ? i + 1 : i;
        if(!has_pend && !rocke_attn_bwd_dq_sink_begin_head(&sink, b, h0))
            return 1;
        rocke_attn_bwd_q_step_opts_t o = rocke_attn_bwd_q_step_opts_default();
        o.edge = edge;
        o.pending = has_pend ? &pend : NULL;
        rocke_attn_bwd_q_step_result_t res;
        if(counted)
        {
            rocke_attn_bwd_q_step_inputs_t here = inputs(&p, i, edge, false);
            V qt = p.qt_start;
            if(i != 0)
            {
                V ci = rocke_b_const_i32(b, i);
                qt = rocke_b_add(b, p.qt_start, ci);
            }
            NextLoads nl = {&p, nxt_i};
            o.packed = here.packed;
            o.staged = &staged;
            o.next_loads = next_loads_fn;
            o.next_loads_ctx = &nl;
            if(!rocke_attn_bwd_tile_step_run_q_step(
                   &step, b, &acc, NULL, NULL, qt, here.h, k0, &o, &res))
                return 1;
            staged = res.staged;
        }
        else
        {
            rocke_attn_bwd_q_step_inputs_t here = inputs(&p, i, edge, true);
            V qt = p.qt_start;
            if(i != 0)
            {
                V ci = rocke_b_const_i32(b, i);
                qt = rocke_b_add(b, p.qt_start, ci);
            }
            rocke_attn_bwd_ring_step_t merged;
            if(merge)
            {
                V ci = rocke_b_const_i32(b, i);
                V c2 = rocke_b_const_i32(b, 2);
                merged.slot = rocke_b_mod(b, ci, c2);
                merged.nxt = inputs(&p, nxt_i, steps[nxt_i] == 'e', true);
                o.ring = &merged;
            }
            rocke_attn_bwd_prefetch_step_t pf;
            if(has_ring)
            {
                V it = rocke_b_const_i32(b, i);
                rocke_attn_bwd_prefetch_tile_t* tile = &tiles[n_tiles_used++ % 256];
                tile->q0 = here.q0;
                tile->h = here.h;
                tile->has_slabs = here.has_slabs;
                tile->q_slab = here.q_slab;
                tile->do_slab = here.do_slab;
                tile->packed = here.packed;
                rocke_attn_bwd_prefetch_tile_t* ahead = tile;
                int k = ring.schedule.ahead;
                if(k)
                    ahead = ring_tile(i + k < n - 1 ? i + k : n - 1);
                pf = rocke_attn_bwd_prefetch_step_make(&ring, it, tile, ahead, carried, n_carried);
                o.prefetch = &pf;
            }
            o.packed = here.packed;
            if(!rocke_attn_bwd_tile_step_run_q_step(&step,
                                                    b,
                                                    &acc,
                                                    here.has_slabs ? &here.q_slab : NULL,
                                                    here.has_slabs ? &here.do_slab : NULL,
                                                    qt,
                                                    here.h,
                                                    k0,
                                                    &o,
                                                    &res))
                return 1;
            if(has_ring)
            {
                carried = pf.yields;
                n_carried = pf.n_yields;
            }
        }
        acc = res.acc;
        if(has_pend)
            pend = res.pending;
        if(!counted && acc.frags_m * acc.frags_n > 0)
        {
            /* the loop-carried form (no IR) */
            int nf = rocke_attn_bwd_dkdv_acc_flat_len(&acc);
            rocke_value_t** flat = (rocke_value_t**)calloc(nf, sizeof(rocke_value_t*));
            rocke_attn_bwd_dkdv_acc_flat(&acc, flat);
            bool fok
                = rocke_attn_bwd_dkdv_acc_from_flat(b, flat, nf, acc.frags_m, acc.frags_n, &acc);
            free(flat);
            if(!fok)
                return 1;
        }
    }
    if(has_ring && !rocke_attn_bwd_q_ring_drain(&ring, b))
        return 1;
    if(has_pend && !rocke_attn_bwd_tile_step_flush_dq(&step, b, &pend))
        return 1;
    if(!rocke_attn_bwd_tile_step_finish(&step, b, &acc, &acc))
        return 1;
    V mult = a["dk_mult"];
    rocke_attn_bwd_dkv_row_t* rows;
    int n_rows;
    if(epi == "cells")
    {
        if(!rocke_attn_bwd_tile_step_dkv_cells(&step, b, "dk", &acc, mult, &rows, &n_rows)
           || !rocke_attn_bwd_tile_step_dkv_cells(&step, b, "dv", &acc, NULL, &rows, &n_rows))
            return 1;
    }
    else if(epi == "rows")
    {
        rocke_attn_bwd_dkv_reader_t rd;
        if(!rocke_attn_bwd_tile_step_dkv_rows(
               &step, b, "dk", &acc, mult, s.dkv_store_vec, 0, true, &rd))
            return 1;
        V c0 = rocke_b_const_i32(b, 0);
        if(rocke_attn_bwd_dkv_reader_read(b, &rd, tid, c0) == NULL)
            return 1;
    }
    else if(epi == "slots")
    {
        if(!rocke_attn_bwd_tile_step_dkv_row_slots(&step, b, "dk", &acc, &rows, &n_rows)
           || !rocke_attn_bwd_tile_step_dkv_row_slots(&step, b, "dv", &acc, &rows, &n_rows))
            return 1;
    }
    else if(epi == "quad")
    {
        rocke_attn_bwd_dkv_quad_t* q;
        int nq;
        if(!rocke_attn_bwd_tile_step_dkv_quad_rows(&step, b, "dk", &acc, mult, &q, &nq))
            return 1;
    }
    else if(epi == "half")
    {
        for(int half = 0; half < 2; ++half)
        {
            rocke_attn_bwd_lds_view_t v;
            if(!rocke_attn_bwd_tile_step_dkv_f32_part(&step, b, "dv", &acc, mult, half, 0, &v))
                return 1;
            rocke_b_sync_fence(b, ROCKE_ATTN_BWD_BARRIER_FENCE);
        }
    }
    else if(epi == "image")
    {
        rocke_attn_bwd_lds_view_t v;
        if(!rocke_attn_bwd_tile_step_dkv_f32_image(&step, b, "dk", &v))
            return 1;
        if(rocke_attn_bwd_lds_view_load(b, &v, iv(tid), rocke_attn_bwd_iv_int(0), rocke_f32(), 1)
           == NULL)
            return 1;
    }
    else if(epi != "none")
        return -1;
    rocke_attn_bwd_tile_step_free(&step);
    rocke_attn_bwd_dq_sink_free(&sink);
    rocke_b_ret(b);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= N_CONFIGS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    std::map<std::string, std::string> probe;
    std::string config = split_probe(CONFIGS[idx], &probe);
    std::string arch = config.substr(0, config.find('|'));

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, PROBE_NAME) != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* A strict-mode builder error throws; the guard turns it back into the
     * sticky error, like the guarded entry points of the C++ kernels. */
    int rc = 1;
    ckc::guard_builder(&b, [&]() -> void* {
        rc = build(idx, &b);
        return &b;
    });
    if(rc != 0 || !rocke_ir_builder_ok(&b))
    {
        const char* msg = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed for config %d: %s\n", idx, msg ? msg : "");
        rocke_ir_builder_free(&b);
        return 1;
    }
    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch.c_str(), &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(text, stdout);
        }
        free(text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(t, stdout);
        }
        free(t);
    }
    else
    {
        rocke_diag_t* d = NULL;
        size_t nd = 0;
        rocke_verify(kernel, &d, &nd);
        for(size_t i = 0; i < nd; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, nd);
    }
    rocke_ir_builder_free(&b);
    return ret;
}
