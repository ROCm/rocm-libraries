/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_spec_emit.c -- C-side emitter for the attention
 * backward main kernel's spec and decisions (rocke/instance_attention_bwd.h).
 * Selects one config by argv[1] (the same table as _CONFIGS of
 * attention_bwd_spec_emit.py; regenerate it with
 * `attention_bwd_spec_emit.py --print-c-configs`), fills the spec, and prints
 * the same text dump of every derived decision as the Python emitter, so
 * run_diff can byte-compare the two (the dump stands in for a .ll: these
 * decisions emit no IR). Compiled as C++20 against librocke_core.a.
 *
 * Only mode "ll" (the dump) is supported.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_tiles.h"
#include "rocke/instance_attention_bwd.h"
#include "rocke/instance_attention_bwd_aux.h"

/* attention_bwd_spec_extra_emit.c defines its own CONFIGS and includes this
 * file with ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS set. */
#ifndef ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS
static const char* const CONFIGS[] = {
    /* default gfx942 d32 fp16 sv8 */
    "gfx942|head_size:32;dtype:fp16;stage_vec:8",
    /* default gfx942 d32 bf16 sv8 */
    "gfx942|head_size:32;dtype:bf16;stage_vec:8",
    /* default gfx942 d32 bf16 sv1 */
    "gfx942|head_size:32;dtype:bf16;stage_vec:1",
    /* default gfx942 d32 thd atomic */
    "gfx942|head_size:32;seq_mode:thd;dkv_mode:atomic",
    /* default gfx942 d64 fp16 sv8 */
    "gfx942|head_size:64;dtype:fp16;stage_vec:8",
    /* default gfx942 d64 bf16 sv8 */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8",
    /* default gfx942 d64 bf16 sv1 */
    "gfx942|head_size:64;dtype:bf16;stage_vec:1",
    /* default gfx942 d64 thd atomic */
    "gfx942|head_size:64;seq_mode:thd;dkv_mode:atomic",
    /* default gfx942 d64 no mask, runtime scale */
    "gfx942|head_size:64;mask_class:none;static_scale:runtime",
    /* default gfx942 d64 split dQ */
    "gfx942|head_size:64;dq_mode:split",
    /* default gfx942 d128 fp16 sv8 */
    "gfx942|head_size:128;dtype:fp16;stage_vec:8",
    /* default gfx942 d128 bf16 sv8 */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8",
    /* default gfx942 d128 bf16 sv1 */
    "gfx942|head_size:128;dtype:bf16;stage_vec:1",
    /* default gfx942 d128 thd atomic */
    "gfx942|head_size:128;seq_mode:thd;dkv_mode:atomic",
    /* default gfx942 d128 no mask, runtime scale */
    "gfx942|head_size:128;mask_class:none;static_scale:runtime",
    /* default gfx942 d128 split dQ */
    "gfx942|head_size:128;dq_mode:split",
    /* default gfx950 d32 fp16 sv8 */
    "gfx950|head_size:32;dtype:fp16;stage_vec:8",
    /* default gfx950 d32 bf16 sv8 */
    "gfx950|head_size:32;dtype:bf16;stage_vec:8",
    /* default gfx950 d32 bf16 sv1 */
    "gfx950|head_size:32;dtype:bf16;stage_vec:1",
    /* default gfx950 d32 thd atomic */
    "gfx950|head_size:32;seq_mode:thd;dkv_mode:atomic",
    /* default gfx950 d64 fp16 sv8 */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8",
    /* default gfx950 d64 bf16 sv8 */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8",
    /* default gfx950 d64 bf16 sv1 */
    "gfx950|head_size:64;dtype:bf16;stage_vec:1",
    /* default gfx950 d64 thd atomic */
    "gfx950|head_size:64;seq_mode:thd;dkv_mode:atomic",
    /* default gfx950 d64 no mask, runtime scale */
    "gfx950|head_size:64;mask_class:none;static_scale:runtime",
    /* default gfx950 d64 split dQ */
    "gfx950|head_size:64;dq_mode:split",
    /* default gfx950 d128 fp16 sv8 */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8",
    /* default gfx950 d128 bf16 sv8 */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8",
    /* default gfx950 d128 bf16 sv1 */
    "gfx950|head_size:128;dtype:bf16;stage_vec:1",
    /* default gfx950 d128 thd atomic */
    "gfx950|head_size:128;seq_mode:thd;dkv_mode:atomic",
    /* default gfx950 d128 no mask, runtime scale */
    "gfx950|head_size:128;mask_class:none;static_scale:runtime",
    /* default gfx950 d128 split dQ */
    "gfx950|head_size:128;dq_mode:split",
    /* default gfx1151 d32 */
    "gfx1151|head_size:32",
    /* default gfx1151 d64 */
    "gfx1151|head_size:64",
    /* default gfx1151 d128 */
    "gfx1151|head_size:128",
    /* default gfx1151 d64 bf16 thd atomic sv1 */
    "gfx1151|head_size:64;dtype:bf16;seq_mode:thd;dkv_mode:atomic;stage_vec:1",
    /* default gfx1201 d32 */
    "gfx1201|head_size:32",
    /* default gfx1201 d64 */
    "gfx1201|head_size:64",
    /* default gfx1201 d128 */
    "gfx1201|head_size:128",
    /* default gfx1201 d64 bf16 thd atomic sv1 */
    "gfx1201|head_size:64;dtype:bf16;seq_mode:thd;dkv_mode:atomic;stage_vec:1",
    /* tuned general/128/bf16/causal_tl/gqa/unequal/fixed/8/t16/full */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:32;edge_tiles:"
    "False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;"
    "ws_layout:head_major",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t16/over */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t64/over */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    /* tuned general/128/bf16/none/mha/square/fixed/8/t16/over */
    "gfx942|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "192x192;agpr_route:asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;"
    "block_m:16;block_n:128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;"
    "warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over */
    "gfx942|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:"
    "128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    /* tuned general/64/bf16/causal_br/gqa/square/fixed/8/t64/full */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;"
    "atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_"
    "path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_"
    "plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t16/full */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;"
    "atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_"
    "path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_"
    "plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True",
    /* tuned general/64/bf16/none/mha/square/thd/8/t16/over */
    "gfx942|head_size:64;dtype:bf16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_"
    "n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;"
    "waves_per_eu:1;ws_layout:head_major",
    /* tuned general/64/fp16/none/mha/square/thd/8/t64/over */
    "gfx942|head_size:64;dtype:fp16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_"
    "n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;"
    "waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/causal_br/gqa/square/fixed/8/t16/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/causal_br/gqa/unequal/fixed/8/t16/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;step_order:head_outer;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t4/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:"
    "False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_"
    "layout:head_major",
    /* tuned general/128/bf16/none/gqa/square/fixed/8/t64/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/none/gqa/unequal/fixed/8/t64/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/bf16/none/mha/square/fixed/8/t16/over */
    "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:"
    "192x192;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:16;block_n:192;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/128/fp16/causal_br/gqa/unequal/fixed/8/t64/over */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_"
    "vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;"
    "waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    /* tuned general/128/fp16/causal_tl/gqa/square/fixed/8/t4/full */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:"
    "False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_"
    "source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_"
    "layout:head_major;lb_interleave:True",
    /* tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over */
    "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:"
    "192;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;scheduler_strategy:iterative-ilp;setprio:False;stats_fma:"
    "True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:"
    "1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/64/bf16/causal_br/gqa/unequal/fixed/8/t16/full */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;static_scale:convert;acc_in_lds:False;agpr_alloc:"
    "None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;"
    "edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;lds_order:phase_low;lds_"
    "swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:"
    "None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_"
    "grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_interleave:True",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t4/full */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;"
    "atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_"
    "path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_"
    "read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:"
    "head_major;lb_mirror:True",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t16/full */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t16/over */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t4/over */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t64/full */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_"
    "grid_g4:2x2;waves:4;waves_per_eu:2;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t64/over */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major",
    /* tuned general/64/bf16/none/mha/square/thd/8/t4/over */
    "gfx950|head_size:64;dtype:bf16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_"
    "n:192;bwd_exp2:fast;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;lds_"
    "swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:"
    "None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_"
    "grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major;worklist:True",
    /* tuned general/64/fp16/none/mha/square/fixed/8/t16/full */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_"
    "exp2:fast;dq_mode:split;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_"
    "route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;"
    "stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_"
    "grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major",
    /* tuned general/64/fp16/none/mha/square/fixed/8/t4/full */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_"
    "tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;"
    "transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_"
    "per_eu:1;ws_layout:head_major",
    /* tuned general/64/fp16/none/mha/square/thd/8/t64/over */
    "gfx950|head_size:64;dtype:fp16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_"
    "alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_"
    "n:192;bwd_exp2:fast;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:"
    "relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_"
    "fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_"
    "g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major",
    /* xt_lds gfx942 d64 (predictor swizzle) */
    "gfx942|head_size:64;transpose_source:xt_lds",
    /* xt_lds gfx942 d128 K^T resident image */
    "gfx942|head_size:128;transpose_source:xt_lds;kt_source:lds",
    /* xt_lds gfx942 d32 explicit XT swizzles */
    "gfx942|head_size:32;transpose_source:xt_lds;lds_swizzle:qt=pad16,dot=none,kt_stage=pad8",
    /* xt_lds gfx950 d64 wide G4 atom */
    "gfx950|head_size:64;transpose_source:xt_lds",
    /* xt_lds gfx950 d128 split dQ */
    "gfx950|head_size:128;transpose_source:xt_lds;dq_mode:split",
    /* xt_lds gfx950 d32 bf16 kt lds */
    "gfx950|head_size:32;dtype:bf16;transpose_source:xt_lds;kt_source:lds",
    /* xt_lds gfx1201 d64 (no wave64 predictor) */
    "gfx1201|head_size:64;transpose_source:xt_lds",
    /* row xor buffers gfx942 d64 */
    "gfx942|head_size:64;lds_swizzle:q=xor,do=xor,ds=xor,k_stage=xor,v_stage=xor,dkv_out=xor",
    /* row pads gfx950 d64 */
    "gfx950|head_size:64;lds_swizzle: dkv_out = pad16 , ds=pad8,q=pad32,unused=none ",
    /* rejected: ring-slotted row xor */
    "gfx942|head_size:64;ring_depth:2;lds_swizzle:q=xor",
    /* ring 3 gfx950 d64 */
    "gfx950|head_size:64;ring_depth:3",
    /* rejected: ring 2 with counted */
    "gfx942|head_size:64;ring_depth:2;loop_sync:counted",
    /* counted gfx942 d64 */
    "gfx942|head_size:64;loop_sync:counted",
    /* counted gfx950 d128 late issue */
    "gfx950|head_size:128;loop_sync:counted;counted_issue:late",
    /* rejected: counted late with stage_table_atomics */
    "gfx950|head_size:128;loop_sync:counted;counted_issue:late;sched:stage_table_atomics",
    /* counted early with stage_table_atomics */
    "gfx950|head_size:128;loop_sync:counted;sched:stage_table_atomics",
    /* rejected: counted with loads_first */
    "gfx942|head_size:64;loop_sync:counted;qdo_stage:loads_first",
    /* rejected: counted on gfx1151 */
    "gfx1151|head_size:64;loop_sync:counted",
    /* rejected: counted grouped head_outer */
    "gfx942|head_size:64;loop_sync:counted;step_order:head_outer;edge_tiles:True",
    /* rejected: counted with split dQ */
    "gfx950|head_size:64;loop_sync:counted;dq_mode:split",
    /* rejected: counted with barrier merge */
    "gfx950|head_size:64;loop_sync:counted;barrier_merge:on",
    /* rejected: counted atomics past vmcnt */
    "gfx942|head_size:128;block_m:64;block_n:32;waves:2;warp_grid_g4:1x2;block_k4:32;loop_sync:"
    "counted",
    /* stats_vec 2 gfx942 d64 */
    "gfx942|head_size:64;stats_vec:2",
    /* stats_vec 4 gfx950 d128 bf16 */
    "gfx950|head_size:128;dtype:bf16;stats_vec:4",
    /* rejected: stats_vec 4 on gfx1201 */
    "gfx1201|head_size:64;stats_vec:4",
    /* rejected: stats_vec token-major */
    "gfx942|head_size:64;stats_vec:2;ws_layout:token_major",
    /* dkv_cell_vec 4 gfx950 direct epilogue */
    "gfx950|head_size:64;dkv_epilogue:direct;dkv_cell_vec:4",
    /* rejected: dkv_cell_vec 4 on gfx1201 */
    "gfx1201|head_size:64;dkv_cell_vec:4",
    /* dkv_cell_vec 4 gfx942 atomic slices */
    "gfx942|head_size:32;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;dkv_cell_vec:4",
    /* g4 slice_db gfx942 d64 */
    "gfx942|head_size:64;g4_slicing:slice_db",
    /* g4 slice -> step (kK4 = atom K) */
    "gfx942|head_size:64;g4_slicing:slice;block_k4:16",
    /* g4 slice_db -> slice (kK4 = kN0) */
    "gfx942|head_size:32;g4_slicing:slice_db;block_n:64;block_k4:64",
    /* g4 slice_db_g3 gfx950 d128 kt lds */
    "gfx950|head_size:128;g4_slicing:slice_db_g3",
    /* rejected: slice_db_g3 with g1g4g3 */
    "gfx942|head_size:64;g4_slicing:slice_db_g3;stage_order:g1g4g3",
    /* rejected: g4 slicing on gfx1151 (resolve) */
    "gfx1151|head_size:64;g4_slicing:slice",
    /* rejected: g4 slicing with fragment issue */
    "gfx942|head_size:64;g4_slicing:slice;dq_issue:fragment",
    /* rejected: g4 slicing with acc_in_lds */
    "gfx942|head_size:64;g4_slicing:slice;acc_in_lds:True",
    /* rejected: g4 slicing with ring */
    "gfx950|head_size:64;g4_slicing:slice;ring_depth:2",
    /* rejected: g4 slicing xt kt lds */
    "gfx942|head_size:64;g4_slicing:slice;transpose_source:xt_lds;kt_source:lds",
    /* asm_g13 gfx942 tail nop 7 */
    "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:7",
    /* rejected: asm tail nop below the ISA floor */
    "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:5",
    /* asm_g13 tail nop 15 -> None, gemm -> k_step */
    "gfx942|head_size:128;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:15;asm_cluster:gemm",
    /* asm_g13 s_in_agpr gfx950 tail 8 gemm d32 */
    "gfx950|head_size:32;agpr_alloc:128x128;agpr_route:asm_g13;s_in_agpr:True;asm_tail_nop:8;asm_"
    "cluster:gemm;asm_load_hoist:True",
    /* rejected: asm tail 6 below the gfx950 floor */
    "gfx950|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:6",
    /* rejected: vgpr_form with s_in_agpr */
    "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:vgpr_form;s_in_agpr:True",
    /* rejected: agpr_fence without reservation */
    "gfx942|head_size:64;agpr_route:agpr_fence",
    /* agpr_fence gfx950 d128 */
    "gfx950|head_size:128;agpr_alloc:128x128;agpr_route:agpr_fence",
    /* vgpr_form gfx942 d128 */
    "gfx942|head_size:128;agpr_alloc:192x192;agpr_route:vgpr_form",
    /* agpr backend gfx950 d128 */
    "gfx950|head_size:128;agpr_alloc:backend",
    /* backend with s_in_agpr */
    "gfx942|head_size:64;agpr_alloc:backend;s_in_agpr:True",
    /* rejected: 256 AGPRs on eight waves */
    "gfx942|head_size:64;agpr_alloc:256x256",
    /* rejected: reservation below the estimate */
    "gfx942|head_size:128;waves:4;block_n:128;warp_grid_g4:1x4;agpr_alloc:128x128",
    /* policy None: four-wave d128 */
    "gfx942|head_size:128;waves:4;block_n:128;warp_grid_g4:1x4",
    /* rejected: agpr_alloc on WMMA */
    "gfx1151|head_size:64;agpr_alloc:0x0",
    /* acc_in_lds gfx942 d32 */
    "gfx942|head_size:32;acc_in_lds:True",
    /* rejected: acc_in_lds d64 LDS plan */
    "gfx942|head_size:64;acc_in_lds:True",
    /* rejected: acc_in_lds with direct epilogue */
    "gfx950|head_size:64;acc_in_lds:True;dkv_epilogue:direct",
    /* rejected: acc_in_lds with reservation */
    "gfx942|head_size:64;acc_in_lds:True;agpr_alloc:128x128",
    /* acc_in_lds atomic pair epilogue */
    "gfx942|head_size:32;acc_in_lds:True;dkv_mode:atomic;dkv_epilogue:lds_staged_pair",
    /* small tile lds route gfx942 d64 16x16x4 */
    "gfx942|head_size:64;waves:4;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x4;warp_grid_g02:"
    "1x1;warp_grid_g13:1x4;pt_route:lds",
    /* small tile lds route gfx950 d32 32x32x2 */
    "gfx950|head_size:32;waves:2;block_m:32;block_n:32;block_k4:32;warp_grid_g4:1x2;warp_grid_g02:"
    "2x1;warp_grid_g13:2x1;pt_route:lds",
    /* (2, W/2) grids gfx942 d64 */
    "gfx942|head_size:64;warp_grid_g02:2x4;warp_grid_g13:2x4;pt_route:lds",
    /* rejected: non-natural grid on relabel */
    "gfx942|head_size:64;warp_grid_g02:2x4",
    /* rejected: grid not a CTA grid */
    "gfx942|head_size:64;warp_grid_g13:4x4;pt_route:lds",
    /* rejected: grid does not divide */
    "gfx942|head_size:32;warp_grid_g13:1x8;pt_route:lds",
    /* rejected: natural next to small grid does not divide */
    "gfx942|head_size:64;waves:8;block_m:16;block_n:64;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_"
    "grid_g4:1x8;pt_route:lds;block_k4:32",
    /* rejected: one wave on 32 x 32 */
    "gfx942|head_size:64;waves:1;block_m:32;block_n:32;warp_grid_g4:1x1",
    /* one wave d128 debug tile gfx942 (scheduler) */
    "gfx942|head_size:128;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1",
    /* rejected: 8 waves on gfx1151 */
    "gfx1151|head_size:64;waves:8",
    /* rejected: wide atom on gfx942 */
    "gfx942|head_size:64;atom_g02:16x16x32",
    /* rejected: tr_read on gfx942 */
    "gfx942|head_size:64;transpose_source:tr_read",
    /* rejected: dma on gfx942 */
    "gfx942|head_size:64;global_path:dma",
    /* rejected: dma with head_pack */
    "gfx950|head_size:64;global_path:dma;head_pack:True",
    /* rejected: dma with lds_only */
    "gfx950|head_size:64;global_path:dma;loop_sync:lds_only",
    /* dma gfx950 d32 ring 3 */
    "gfx950|head_size:32;global_path:dma;ring_depth:3",
    /* dma gfx950 tr_read q xor */
    "gfx950|head_size:64;global_path:dma;lds_swizzle:q=xor,ds=pad16",
    /* rejected: dma tr_read q pad */
    "gfx950|head_size:64;global_path:dma;lds_swizzle:q=pad8",
    /* rejected: ring 3 on gfx942 */
    "gfx942|head_size:64;ring_depth:3",
    /* rejected: split dQ on gfx1151 */
    "gfx1151|head_size:64;dq_mode:split",
    /* rejected: kK4 above kN0 */
    "gfx942|head_size:64;block_n:32;block_k4:64;waves:2;warp_grid_g4:1x2",
    /* rejected: warp_grid_g4 shape */
    "gfx942|head_size:64;warp_grid_g4:4x2",
    /* rejected: warp_grid_g4 does not divide D */
    "gfx942|head_size:32;warp_grid_g4:1x8",
    /* rejected: block_n not a multiple of 16 W */
    "gfx942|head_size:64;block_n:64",
    /* rejected: sched stage_table with replicated grid */
    "gfx942|head_size:64;waves:4;block_m:16;block_n:64;block_k4:16;warp_grid_g4:1x4;warp_grid_g02:"
    "1x2;warp_grid_g13:4x1;pt_route:lds;sched:stage_table",
    /* ptr_noalias workspace */
    "gfx942|head_size:64;ptr_noalias:workspace",
    /* ptr_noalias all atomic */
    "gfx950|head_size:64;ptr_noalias:all;dkv_mode:atomic",
    /* rejected: ptr_noalias all direct */
    "gfx950|head_size:64;ptr_noalias:all",
    /* rejected: kv reg on WMMA */
    "gfx1201|head_size:64;kv_residency:reg;kt_source:lds",
    /* rejected: relabel on WMMA */
    "gfx1201|head_size:64;pt_route:relabel",
    /* lds_only gfx942 d64 */
    "gfx942|head_size:64;loop_sync:lds_only",
    /* qdo joint gfx942 d32 */
    "gfx942|head_size:32;qdo_stage:joint",
    /* rejected: qdo joint too many chunks */
    "gfx942|head_size:64;waves:4;block_n:128;warp_grid_g4:2x2;qdo_stage:joint",
    /* rejected: loads_first with ring */
    "gfx942|head_size:64;ring_depth:2;qdo_stage:loads_first",
    /* stats step_top gfx950 */
    "gfx950|head_size:128;stats_load:step_top;qdo_stage:loads_first",
    /* barrier merge gfx950 d64 */
    "gfx950|head_size:64;barrier_merge:on;ring_issue:late",
    /* barrier merge gfx942 d128 g3g1g4 */
    "gfx942|head_size:128;barrier_merge:on;stage_order:g3g1g4",
    /* rejected: barrier merge on gfx1151 */
    "gfx1151|head_size:64;barrier_merge:on",
    /* rejected: barrier merge with dma */
    "gfx950|head_size:64;barrier_merge:on;global_path:dma",
    /* rejected: barrier merge G4 not last */
    "gfx942|head_size:64;barrier_merge:on;stage_order:g1g4g3",
    /* rejected: barrier merge split dQ */
    "gfx950|head_size:64;barrier_merge:on;dq_mode:split",
    /* rejected: barrier merge acc_in_lds */
    "gfx942|head_size:64;barrier_merge:on;acc_in_lds:True",
    /* rejected: ring_issue without merge */
    "gfx950|head_size:64;ring_issue:late",
    /* rejected: grouped head_outer with ring */
    "gfx950|head_size:64;step_order:head_outer;edge_tiles:True;ring_depth:2",
    /* head_outer edge tiles gfx942 */
    "gfx942|head_size:64;step_order:head_outer;edge_tiles:True",
    /* rejected: setprio without stage table */
    "gfx942|head_size:64;setprio:True",
    /* stage_table setprio gfx942 */
    "gfx942|head_size:64;sched:stage_table;setprio:True",
    /* rejected: stage order with stage table */
    "gfx942|head_size:64;sched:stage_table;stage_order:g3g1g4",
    /* rejected: stage_table_atomics split dQ */
    "gfx950|head_size:64;sched:stage_table_atomics;dq_mode:split",
    /* rejected: xt_lds with ring */
    "gfx942|head_size:64;transpose_source:xt_lds;ring_depth:2",
    /* rejected: xt_lds with fences */
    "gfx942|head_size:64;transpose_source:xt_lds;sched:fences_only",
    /* rejected: waves_per_eu 3 on WMMA */
    "gfx1151|head_size:64;waves_per_eu:3",
    /* rejected: waves_per_eu 4 on d128 */
    "gfx942|head_size:128;waves_per_eu:4",
    /* rejected: waves_per_eu 2 RDNA d128 */
    "gfx1201|head_size:128;waves_per_eu:2",
    /* waves_per_eu 2 gfx1201 d32 */
    "gfx1201|head_size:32;waves_per_eu:2",
    /* rejected: spill combo gfx942 d128 edge row xor */
    "gfx942|head_size:128;edge_tiles:True;lds_swizzle:dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage="
    "xor,pt=xor,q=xor,v_res=xor,v_stage=xor",
    /* rejected: spill combo gfx1151 xt_lds */
    "gfx1151|head_size:64;transpose_source:xt_lds",
    /* rejected: bf16_cast on fp16 */
    "gfx942|head_size:64;bf16_cast:rne_noguard",
    /* bf16_cast rne_noguard bf16, exp2 fast, stats_fma */
    "gfx950|head_size:64;dtype:bf16;bf16_cast:rne_noguard;bwd_exp2:fast;stats_fma:True",
    /* ws_bounds buffer */
    "gfx942|head_size:64;ws_bounds:buffer",
    /* rejected: narrow_vec on sv8 */
    "gfx942|head_size:64;narrow_vec:2",
    /* narrow_vec 4 sv1 gfx950 */
    "gfx950|head_size:128;stage_vec:1;narrow_vec:4",
    /* rejected: dkv_store_vec on direct epilogue */
    "gfx942|head_size:64;dkv_epilogue:direct;dkv_store_vec:4",
    /* lds_staged_pair gfx950 d64 store vec 4 */
    "gfx950|head_size:64;dkv_epilogue:lds_staged_pair;dkv_store_vec:4",
    /* atomic pair planned as staged */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_epilogue:lds_staged_pair;dkv_atomic_guard:select",
    /* atomic slices direct epilogue */
    "gfx950|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct",
    /* last_cta sink (validated, not built) */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:last_cta",
    /* phase_low kv lds gfx942 d64 */
    "gfx942|head_size:64;kv_residency:lds;kt_source:lds;lds_order:phase_low",
    /* phase_low gfx950 d128 (kt lds default) */
    "gfx950|head_size:128;lds_order:phase_low",
    /* kt lds kv reg gfx942 lds_plain (k_res copy) */
    "gfx942|head_size:64;kt_source:lds",
    /* kv lds tr_read gfx950 d64 */
    "gfx950|head_size:64;kv_residency:lds;kt_source:lds",
    /* head_pack + edge tiles gfx950 */
    "gfx950|head_size:64;head_pack:True;edge_tiles:True",
    /* rejected: slab_carry with q_rotate */
    "gfx942|head_size:64;slab_carry:True;q_rotate:True",
    /* rejected: slab_carry with ring */
    "gfx950|head_size:64;slab_carry:True;ring_depth:2",
    /* rejected: slab_carry with counted */
    "gfx942|head_size:64;slab_carry:True;loop_sync:counted",
    /* slab_carry gfx942 */
    "gfx942|head_size:64;slab_carry:True;interior_clamp:True;edge_tiles:True",
    /* rejected: worklist batched */
    "gfx950|head_size:64;worklist:True",
    /* rejected: worklist with lb_mirror */
    "gfx950|head_size:64;seq_mode:thd;worklist:True;lb_mirror:True",
    /* worklist thd + flags */
    "gfx950|head_size:64;seq_mode:thd;dkv_mode:atomic;worklist:True;xcd_remap:True;q_rotate:True;"
    "lb_interleave:True;dkv_slab_hoist:True",
    /* rejected: counted_issue without counted */
    "gfx942|head_size:64;counted_issue:late",
    /* rejected: asm_tail_nop without asm */
    "gfx942|head_size:64;asm_tail_nop:8",
    /* rejected: asm_tail_nop out of range */
    "gfx942|head_size:64;asm_tail_nop:16",
    /* rejected: asm_cluster without asm */
    "gfx942|head_size:64;asm_cluster:gemm",
    /* rejected: asm with iglp */
    "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;sched:iglp1",
    /* rejected: lds_swizzle entry without value */
    "gfx942|head_size:64;lds_swizzle:q",
    /* rejected: lds_swizzle bad value */
    "gfx942|head_size:64;lds_swizzle:q=pad4",
    /* rejected: lds_swizzle names a buffer twice */
    "gfx942|head_size:64;lds_swizzle:q=xor,q=pad8",
    /* rejected: lds_swizzle name not an identifier */
    "gfx942|head_size:64;lds_swizzle:1q=xor",
    /* empty lds_swizzle */
    "gfx942|head_size:64;lds_swizzle:",
    /* rejected: scheduler strategy */
    "gfx942|head_size:64;scheduler_strategy:fastest",
    /* scheduler strategy max-ilp */
    "gfx950|head_size:64;scheduler_strategy:max-ilp",
    /* custom name */
    "gfx942|head_size:64;name:my_bwd2",
    /* rejected: empty name */
    "gfx942|head_size:64;name:",
    /* rejected: agpr_alloc (64, 64) */
    "gfx942|head_size:64;agpr_alloc:64x64",
    /* rejected: warp grid 0 x 4 */
    "gfx942|head_size:64;warp_grid_g02:0x4",
    /* rejected: head_size 48 */
    "gfx942|head_size:48",
    /* rejected: waves 0 (not None) */
    "gfx942|head_size:64;waves:0",
    /* rejected: asm_tail_nop 0 without asm (not None) */
    "gfx942|head_size:64;asm_tail_nop:0",
    /* rejected: atom bad */
    "gfx942|head_size:64;atom_g4:32x32x8",
    /* rejected: kt reg with kv lds */
    "gfx942|head_size:64;kt_source:reg;kv_residency:lds",
    /* rejected: atom_g13 wide on block_m 16 */
    "gfx950|head_size:64;atom_g13:16x16x32;block_m:16",
    /* wide atom_g13 gfx950 block_m 32 */
    "gfx950|head_size:64;atom_g13:16x16x32",
    /* rejected: s_in_agpr with VGPR form policy */
    "gfx942|head_size:32;s_in_agpr:True",
    /* rejected: LDS plan too big */
    "gfx942|head_size:128;kv_residency:lds;kt_source:lds;block_n:256;block_m:64;waves:8;warp_grid_"
    "g4:2x4",
    /* rejected: arch-VGPR estimate */
    "gfx942|head_size:128;waves:2;block_m:64;block_n:64;warp_grid_g4:1x2;agpr_alloc:0x0",
    /* rejected: workgroup share */
    "gfx942|head_size:128;waves:8;block_m:64;block_n:256;warp_grid_g4:2x4;agpr_alloc:backend",
    /* rejected: unknown arch */
    "gfx90a|head_size:64",
    /* rejected: WMMA register accumulators */
    "gfx1151|head_size:64;acc_in_lds:False",
    /* RDNA d128 one wave kv lds */
    "gfx1201|head_size:128;waves:1;kv_residency:lds",
    /* gfx1201 d128 one wave defaults + phase_low */
    "gfx1201|head_size:128;lds_order:phase_low",
    /* dq_issue fragment gfx942 d32 runtime */
    "gfx942|head_size:32;dq_issue:fragment;static_scale:dq_before_atomic",
    /* static_scale convert atomic */
    "gfx950|head_size:64;static_scale:convert;dkv_mode:atomic",
    /* token-major thd gfx942 */
    "gfx942|head_size:128;seq_mode:thd;ws_layout:token_major;mask_class:none",
    /* rejected: ws_bounds token-major */
    "gfx942|head_size:64;ws_bounds:buffer;ws_layout:token_major",
    /* rejected: ptr_noalias value */
    "gfx942|head_size:64;ptr_noalias:both",
    /* rejected: barrier merge ring 3 */
    "gfx950|head_size:64;barrier_merge:on;ring_depth:3",
    /* rejected: dma with stage_vec 1 */
    "gfx950|head_size:64;global_path:dma;stage_vec:1",
    /* rejected: dma with xt_lds */
    "gfx950|head_size:64;global_path:dma;transpose_source:xt_lds",
    /* agpr_route attr spelled None */
    "gfx942|head_size:64;agpr_route:attr;agpr_alloc:128x128",
    /* rejected: asm_load_hoist without asm */
    "gfx942|head_size:64;asm_load_hoist:True",
    /* rejected: kK4 below the wide atom */
    "gfx950|head_size:64;block_k4:16",
    /* rejected: warp_grid_g4 does not divide kM0 */
    "gfx942|head_size:128;warp_grid_g4:2x4",
    /* rejected: natural G1/G3 grid next to a small G0/G2 grid */
    "gfx942|head_size:64;waves:4;block_m:16;block_n:32;block_k4:16;warp_grid_g4:1x4;warp_grid_g02:"
    "1x2;pt_route:lds",
    /* rejected: VGPR form within the prune slack */
    "gfx942|head_size:64;block_m:64;block_n:128;waves:4;warp_grid_g4:1x4;block_k4:16;kv_residency:"
    "lds;kt_source:lds;atom_g02:16x16x16;atom_g4:16x16x16;agpr_alloc:0x0",
    /* rejected: AGPR estimate past the slack */
    "gfx942|head_size:64;block_m:16;block_n:256;waves:2;warp_grid_g4:1x2;block_k4:16",
    /* rejected: workgroup share of eight waves */
    "gfx942|head_size:128;block_m:16;block_n:256;waves:8;warp_grid_g4:1x8;block_k4:16",
    /* rejected: RDNA register estimate */
    "gfx1201|head_size:64;block_m:64;block_n:32;waves:2;warp_grid_g4:1x2;block_k4:16",
};
#endif

static const int N_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

static const int NONE = ROCKE_ATTN_BWD_NONE;

typedef std::vector<std::string> Lines;

/* ---------------------------------------------------------------- reprs */

static std::string repr_s(const char* s)
{
    if(s == NULL)
        return "None";
    return rocke_attn_bwd::py_repr(s);
}

static std::string repr_b(bool v)
{
    return v ? "True" : "False";
}

static std::string repr_oi(int v)
{
    return v == NONE ? std::string("None") : std::to_string(v);
}

static std::string repr_ob(int v)
{
    return v == NONE ? std::string("None") : repr_b(v != 0);
}

static std::string repr_grid(const rocke_attn_bwd_opt_grid_t& g)
{
    if(!g.has)
        return "None";
    return "(" + std::to_string(g.rows) + ", " + std::to_string(g.cols) + ")";
}

static std::string repr_agpr(const rocke_attn_bwd_agpr_alloc_t& a)
{
    return rocke_attn_bwd::agpr_alloc_repr(a);
}

static std::string exc_name(rocke_status_t code)
{
    switch(code)
    {
    case ROCKE_ERR_VALUE:
        return "ValueError";
    case ROCKE_ERR_KEY:
        return "KeyError";
    case ROCKE_ERR_TYPE:
        return "TypeError";
    case ROCKE_ERR_NOTIMPL:
        return "NotImplementedError";
    case ROCKE_ERR_OOM:
        return "MemoryError";
    default:
        return "Error";
    }
}

static std::string exc(const rocke_attn_bwd_err_t& e)
{
    return exc_name(e.code) + ": " + e.msg;
}

/* --------------------------------------------------------------- config */

static bool parse_pair(const std::string& text, int* r, int* c)
{
    size_t x = text.find('x');
    if(x == std::string::npos || x == 0 || x + 1 >= text.size())
        return false;
    for(size_t i = 0; i < text.size(); ++i)
        if(i != x && !(text[i] >= '0' && text[i] <= '9'))
            return false;
    *r = atoi(text.substr(0, x).c_str());
    *c = atoi(text.substr(x + 1).c_str());
    return true;
}

/* Interned copies of the config strings (the spec borrows them). */
static const char* keep(const std::string& s)
{
    static std::vector<std::string*> store;
    store.push_back(new std::string(s));
    return store.back()->c_str();
}

static bool set_field(rocke_attn_bwd_spec_t* s, const std::string& name, const std::string& text)
{
    bool none = text == "None";
    const char* str = none ? NULL : keep(text);
    int iv = none ? NONE : atoi(text.c_str());
    int bv = none ? NONE : (text == "True" ? 1 : 0);
    if(!none && (text == "True" || text == "False"))
        bv = text == "True";
#define F_STR(f)     \
    if(name == #f)   \
    {                \
        s->f = str;  \
        return true; \
    }
#define F_INT(f)     \
    if(name == #f)   \
    {                \
        s->f = iv;   \
        return true; \
    }
#define F_BOOL(f)       \
    if(name == #f)      \
    {                   \
        s->f = bv == 1; \
        return true;    \
    }
#define F_OBOOL(f)   \
    if(name == #f)   \
    {                \
        s->f = bv;   \
        return true; \
    }
#define F_GRID(f)                                                \
    if(name == #f)                                               \
    {                                                            \
        s->f.has = !none;                                        \
        return none || parse_pair(text, &s->f.rows, &s->f.cols); \
    }
    F_INT(head_size)
    F_STR(dtype)
    F_STR(seq_mode)
    F_STR(dkv_mode)
    F_INT(stage_vec)
    F_STR(mask_class)
    F_STR(dq_mode)
    F_STR(static_scale)
    F_INT(waves)
    F_INT(block_m)
    F_INT(block_n)
    F_INT(block_k4)
    F_GRID(warp_grid_g4)
    F_STR(atom_g02)
    F_STR(atom_g13)
    F_STR(atom_g4)
    F_STR(pt_route)
    F_STR(kv_residency)
    F_STR(kt_source)
    F_STR(transpose_source)
    F_INT(ring_depth)
    F_STR(global_path)
    if(name == "lds_swizzle")
    {
        rocke_attn_bwd_err_t err;
        return rocke_attn_bwd_spec_set_lds_swizzle(s, str, &err) == ROCKE_OK;
    }
    F_STR(sched)
    F_OBOOL(setprio)
    F_INT(waves_per_eu)
    F_OBOOL(edge_tiles)
    F_OBOOL(head_pack)
    F_OBOOL(acc_in_lds)
    F_STR(ws_layout)
    F_BOOL(xcd_remap)
    if(name == "agpr_alloc")
    {
        int r, c;
        memset(&s->agpr_alloc, 0, sizeof(s->agpr_alloc));
        if(none)
            s->agpr_alloc.kind = ROCKE_ATTN_BWD_AGPR_NONE;
        else if(parse_pair(text, &r, &c))
        {
            s->agpr_alloc.kind = ROCKE_ATTN_BWD_AGPR_PAIR;
            s->agpr_alloc.a = r;
            s->agpr_alloc.b = c;
        }
        else
        {
            s->agpr_alloc.kind = ROCKE_ATTN_BWD_AGPR_STR;
            s->agpr_alloc.str = str;
        }
        return true;
    }
    F_OBOOL(s_in_agpr)
    F_STR(agpr_route)
    F_INT(asm_tail_nop)
    F_STR(asm_cluster)
    F_BOOL(asm_load_hoist)
    F_STR(scheduler_strategy)
    F_GRID(warp_grid_g02)
    F_GRID(warp_grid_g13)
    F_STR(stage_order)
    F_STR(loop_sync)
    F_STR(step_order)
    F_STR(dkv_epilogue)
    F_INT(dkv_store_vec)
    F_STR(dkv_atomic_guard)
    F_STR(dkv_sink)
    F_STR(lds_order)
    F_STR(qdo_stage)
    F_STR(stats_load)
    F_STR(dq_issue)
    F_STR(g4_slicing)
    if(name == "ptr_noalias")
    {
        s->ptr_noalias = (none || text == "False") ? NULL : str;
        return true;
    }
    F_BOOL(q_rotate)
    F_BOOL(lb_interleave)
    F_BOOL(dkv_slab_hoist)
    F_BOOL(interior_clamp)
    F_STR(barrier_merge)
    F_STR(ring_issue)
    F_STR(counted_issue)
    F_BOOL(lb_mirror)
    F_STR(bf16_cast)
    F_STR(bwd_exp2)
    F_BOOL(stats_fma)
    F_INT(stats_vec)
    F_BOOL(worklist)
    F_STR(ws_bounds)
    F_INT(dkv_cell_vec)
    F_INT(narrow_vec)
    F_BOOL(slab_carry)
    F_STR(name)
    return false;
}

static bool parse_config(const char* config, std::string* arch, rocke_attn_bwd_spec_t* s)
{
    std::string c(config);
    size_t bar = c.find('|');
    *arch = c.substr(0, bar);
    *s = rocke_attn_bwd_spec_default();
    std::string rest = c.substr(bar + 1);
    size_t start = 0;
    while(start < rest.size())
    {
        size_t semi = rest.find(';', start);
        std::string item
            = rest.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if(!item.empty())
        {
            size_t colon = item.find(':');
            if(!set_field(s, item.substr(0, colon), item.substr(colon + 1)))
                return false;
        }
        if(semi == std::string::npos)
            break;
        start = semi + 1;
    }
    return true;
}

/* ------------------------------------------------------------------ dump */

static void field_lines(Lines& out, const char* prefix, const rocke_attn_bwd_spec_t& s)
{
    auto add = [&](const char* name, const std::string& v) {
        out.push_back(std::string(prefix) + name + "=" + v);
    };
    add("head_size", std::to_string(s.head_size));
    add("dtype", repr_s(s.dtype));
    add("seq_mode", repr_s(s.seq_mode));
    add("dkv_mode", repr_s(s.dkv_mode));
    add("stage_vec", std::to_string(s.stage_vec));
    add("mask_class", repr_s(s.mask_class));
    add("dq_mode", repr_s(s.dq_mode));
    add("static_scale", repr_s(s.static_scale));
    add("waves", repr_oi(s.waves));
    add("block_m", repr_oi(s.block_m));
    add("block_n", repr_oi(s.block_n));
    add("block_k4", repr_oi(s.block_k4));
    add("warp_grid_g4", repr_grid(s.warp_grid_g4));
    add("atom_g02", repr_s(s.atom_g02));
    add("atom_g13", repr_s(s.atom_g13));
    add("atom_g4", repr_s(s.atom_g4));
    add("pt_route", repr_s(s.pt_route));
    add("kv_residency", repr_s(s.kv_residency));
    add("kt_source", repr_s(s.kt_source));
    add("transpose_source", repr_s(s.transpose_source));
    add("ring_depth", repr_oi(s.ring_depth));
    add("global_path", repr_s(s.global_path));
    add("lds_swizzle", s.has_lds_swizzle ? repr_s(s.lds_swizzle) : std::string("None"));
    add("sched", repr_s(s.sched));
    add("setprio", repr_ob(s.setprio));
    add("waves_per_eu", repr_oi(s.waves_per_eu));
    add("edge_tiles", repr_ob(s.edge_tiles));
    add("head_pack", repr_ob(s.head_pack));
    add("acc_in_lds", repr_ob(s.acc_in_lds));
    add("ws_layout", repr_s(s.ws_layout));
    add("xcd_remap", repr_b(s.xcd_remap));
    add("agpr_alloc", repr_agpr(s.agpr_alloc));
    add("s_in_agpr", repr_ob(s.s_in_agpr));
    add("agpr_route", repr_s(s.agpr_route));
    add("asm_tail_nop", repr_oi(s.asm_tail_nop));
    add("asm_cluster", repr_s(s.asm_cluster));
    add("asm_load_hoist", repr_b(s.asm_load_hoist));
    add("scheduler_strategy", repr_s(s.scheduler_strategy));
    add("warp_grid_g02", repr_grid(s.warp_grid_g02));
    add("warp_grid_g13", repr_grid(s.warp_grid_g13));
    add("stage_order", repr_s(s.stage_order));
    add("loop_sync", repr_s(s.loop_sync));
    add("step_order", repr_s(s.step_order));
    add("dkv_epilogue", repr_s(s.dkv_epilogue));
    add("dkv_store_vec", std::to_string(s.dkv_store_vec));
    add("dkv_atomic_guard", repr_s(s.dkv_atomic_guard));
    add("dkv_sink", repr_s(s.dkv_sink));
    add("lds_order", repr_s(s.lds_order));
    add("qdo_stage", repr_s(s.qdo_stage));
    add("stats_load", repr_s(s.stats_load));
    add("dq_issue", repr_s(s.dq_issue));
    add("g4_slicing", repr_s(s.g4_slicing));
    add("ptr_noalias", s.ptr_noalias ? repr_s(s.ptr_noalias) : std::string("False"));
    add("q_rotate", repr_b(s.q_rotate));
    add("lb_interleave", repr_b(s.lb_interleave));
    add("dkv_slab_hoist", repr_b(s.dkv_slab_hoist));
    add("interior_clamp", repr_b(s.interior_clamp));
    add("barrier_merge", repr_s(s.barrier_merge));
    add("ring_issue", repr_s(s.ring_issue));
    add("counted_issue", repr_s(s.counted_issue));
    add("lb_mirror", repr_b(s.lb_mirror));
    add("bf16_cast", repr_s(s.bf16_cast));
    add("bwd_exp2", repr_s(s.bwd_exp2));
    add("stats_fma", repr_b(s.stats_fma));
    add("stats_vec", std::to_string(s.stats_vec));
    add("worklist", repr_b(s.worklist));
    add("ws_bounds", repr_s(s.ws_bounds));
    add("dkv_cell_vec", std::to_string(s.dkv_cell_vec));
    add("narrow_vec", std::to_string(s.narrow_vec));
    add("slab_carry", repr_b(s.slab_carry));
    add("name", repr_s(s.name));
}

/* BwdLdsRequest.canonical_swizzle(text) as the tuple repr. */
static std::string swizzle_tuple_repr(const char* text)
{
    std::vector<std::pair<std::string, std::string>> items;
    if(text != NULL)
    {
        std::string v(text);
        size_t start = 0;
        while(true)
        {
            size_t comma = v.find(',', start);
            std::string part
                = v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if(!part.empty())
            {
                size_t eq = part.find('=');
                items.emplace_back(part.substr(0, eq),
                                   eq == std::string::npos ? "" : part.substr(eq + 1));
            }
            if(comma == std::string::npos)
                break;
            start = comma + 1;
        }
    }
    std::sort(items.begin(), items.end());
    std::string out = "(";
    for(size_t i = 0; i < items.size(); ++i)
    {
        if(i)
            out += ", ";
        out += "(" + repr_s(items[i].first.c_str()) + ", " + repr_s(items[i].second.c_str()) + ")";
    }
    if(items.size() == 1)
        out += ",";
    return out + ")";
}

static void request_lines(Lines& out, const rocke_attn_bwd_lds_request_t& r)
{
    auto add = [&](const char* name, const std::string& v) {
        out.push_back(std::string("lds_request ") + name + "=" + v);
    };
    add("head_size", std::to_string(r.head_size));
    add("k_m0", std::to_string(r.k_m0));
    add("k_n0", std::to_string(r.k_n0));
    add("waves", std::to_string(r.waves));
    add("wave_size", std::to_string(r.wave_size));
    add("dtype", repr_s(r.dtype));
    add("kv_residency", repr_s(r.kv_residency));
    add("kt_source", repr_s(r.kt_source));
    add("transpose_source", repr_s(r.transpose_source));
    add("ring_depth", std::to_string(r.ring_depth));
    add("global_path", repr_s(r.global_path));
    add("pt_route", repr_s(r.pt_route));
    add("acc_in_lds", repr_b(r.acc_in_lds));
    add("dq_mode", repr_s(r.dq_mode));
    add("lds_swizzle", swizzle_tuple_repr(r.lds_swizzle));
    add("dkv_epilogue", repr_s(r.dkv_epilogue));
    add("lds_order", repr_s(r.lds_order));
    add("ds_slots", std::to_string(r.ds_slots));
    add("atom_k_g13", std::to_string(r.atom_k_g13));
    add("atom_k_g4", std::to_string(r.atom_k_g4));
}

static std::string layout_repr(const rocke_attn_bwd_lds_layout_t& l)
{
    char buf[256];
    switch(l.kind)
    {
    case ROCKE_ATTN_BWD_LAYOUT_XT:
        snprintf(buf,
                 sizeof(buf),
                 "XtLayout(rows=%d, cols=%d, elem_bytes=%d, swizzle=%s, chunk_bytes=%d)",
                 l.xt.rows,
                 l.xt.cols,
                 l.xt.elem_bytes,
                 repr_s(l.xt.swizzle).c_str(),
                 l.xt.chunk_bytes);
        return buf;
    case ROCKE_ATTN_BWD_LAYOUT_ROW_XOR:
        snprintf(buf,
                 sizeof(buf),
                 "RowXorLayout(rows=%d, cols=%d, elem_bytes=%d, chunk_bytes=%d)",
                 l.row_xor.rows,
                 l.row_xor.cols,
                 l.row_xor.elem_bytes,
                 l.row_xor.chunk_bytes);
        return buf;
    case ROCKE_ATTN_BWD_LAYOUT_DMA_SLAB:
        snprintf(buf,
                 sizeof(buf),
                 "DmaSlabLayout(rows=%d, cols=%d, elem_bytes=%d)",
                 l.dma.rows,
                 l.dma.cols,
                 l.dma.elem_bytes);
        return buf;
    default:
        return repr_s(l.desc);
    }
}

static std::string phases_text(unsigned phases)
{
    std::string out;
    for(int i = 0; i < 4; ++i)
        if(phases & (1u << i))
        {
            if(!out.empty())
                out += ",";
            out += rocke_attn_bwd_lds_phase_name((rocke_attn_bwd_lds_phase_t)(1u << i));
        }
    return out;
}

static void plan_lines(Lines& out, const rocke_attn_bwd_lds_plan_t& p)
{
    out.push_back("plan: total=" + std::to_string(p.total_bytes)
                  + " resident=" + std::to_string(p.resident_bytes)
                  + " capacity=" + std::to_string(p.capacity_bytes)
                  + " fits=" + repr_b(rocke_attn_bwd_lds_plan_fits(&p))
                  + " phase_offset=" + std::to_string(rocke_attn_bwd_lds_plan_phase_offset(&p))
                  + " unaliased=" + std::to_string(rocke_attn_bwd_lds_plan_unaliased_bytes(&p)));
    std::string ph = "plan phases:";
    for(int i = 0; i < 4; ++i)
        ph += std::string(" ")
              + rocke_attn_bwd_lds_phase_name((rocke_attn_bwd_lds_phase_t)(1u << i)) + "="
              + std::to_string(p.phase_bytes[i]);
    out.push_back(ph);
    for(int i = 0; i < p.n_buffers; ++i)
    {
        const rocke_attn_bwd_lds_buffer_t& b = p.buffers[i];
        out.push_back(std::string("plan buffer: ") + b.name + "[" + std::to_string(b.slot)
                      + "] offset=" + std::to_string(b.offset)
                      + " nbytes=" + std::to_string(b.nbytes) + " phases=" + phases_text(b.phases)
                      + " layout=" + b.layout + " resident=" + repr_b(b.resident));
    }
    for(int i = 0; i < p.n_layouts; ++i)
        out.push_back(std::string("plan layout: ") + p.layouts[i].name + " = "
                      + layout_repr(p.layouts[i]));
    std::vector<std::string> names;
    for(int i = 0; i < p.n_buffers; ++i)
        if(std::find(names.begin(), names.end(), p.buffers[i].name) == names.end())
            names.push_back(p.buffers[i].name);
    for(const std::string& n : names)
    {
        rocke_attn_bwd_err_t err;
        int stride = rocke_attn_bwd_lds_plan_ring_stride(&p, n.c_str(), &err);
        if(err.code == ROCKE_OK)
            out.push_back("plan ring_stride: " + n + " = " + std::to_string(stride));
        else
            out.push_back("plan ring_stride: " + n + " = " + exc(err));
    }
}

static Lines dump(const std::string& arch, rocke_attn_bwd_spec_t spec)
{
    Lines lines;
    lines.push_back("arch: " + arch);
    rocke_attn_bwd_err_t err;
    if(rocke_attn_bwd_spec_init(&spec, &err) != ROCKE_OK)
    {
        lines.push_back("spec: " + exc(err));
        return lines;
    }
    lines.push_back("spec: ok");
    char name[512];
    for(const char* stage : {"main", "dq"})
    {
        rocke_attn_bwd_spec_kernel_name(&spec, stage, name, sizeof(name), &err);
        lines.push_back(std::string("kernel_name ") + stage + ": " + name);
    }
    if(rocke_attn_bwd_spec_kernel_name(&spec, "bwd", name, sizeof(name), &err) == ROCKE_OK)
        lines.push_back(std::string("kernel_name bwd: ") + name);
    else
        lines.push_back("kernel_name bwd: " + exc(err));
    {
        rocke_attn_bwd_lds_request_t ureq;
        if(rocke_attn_bwd_lds_request(&spec, arch.c_str(), &ureq, &err) == ROCKE_OK)
            lines.push_back("unresolved lds_request: ok");
        else
            lines.push_back("unresolved lds_request: " + exc(err));
    }
    rocke_attn_bwd_spec_t geo;
    if(rocke_attn_bwd_default_geometry(
           arch.c_str(), spec.head_size, spec.dtype, spec.stage_vec, spec.seq_mode, &geo, &err)
       == ROCKE_OK)
    {
        std::string g = "default_geometry: ";
        g += "acc_in_lds=" + repr_ob(geo.acc_in_lds);
        g += ", atom_g02=" + repr_s(geo.atom_g02);
        g += ", atom_g13=" + repr_s(geo.atom_g13);
        g += ", atom_g4=" + repr_s(geo.atom_g4);
        g += ", block_k4=" + repr_oi(geo.block_k4);
        g += ", block_m=" + repr_oi(geo.block_m);
        g += ", block_n=" + repr_oi(geo.block_n);
        g += ", edge_tiles=" + repr_ob(geo.edge_tiles);
        g += ", global_path=" + repr_s(geo.global_path);
        g += ", head_pack=" + repr_ob(geo.head_pack);
        g += ", kt_source=" + repr_s(geo.kt_source);
        g += ", kv_residency=" + repr_s(geo.kv_residency);
        g += ", pt_route=" + repr_s(geo.pt_route);
        g += ", ring_depth=" + repr_oi(geo.ring_depth);
        g += ", s_in_agpr=" + repr_ob(geo.s_in_agpr);
        g += ", sched=" + repr_s(geo.sched);
        g += ", scheduler_strategy=" + repr_s(geo.scheduler_strategy);
        g += ", setprio=" + repr_ob(geo.setprio);
        g += ", transpose_source=" + repr_s(geo.transpose_source);
        g += ", warp_grid_g4=" + repr_grid(geo.warp_grid_g4);
        g += ", waves=" + repr_oi(geo.waves);
        g += ", waves_per_eu=" + repr_oi(geo.waves_per_eu);
        lines.push_back(g);
    }
    else
        lines.push_back("default_geometry: " + exc(err));
    rocke_attn_bwd_spec_t r;
    if(rocke_attn_bwd_spec_resolved(&spec, arch.c_str(), &r, &err) != ROCKE_OK)
    {
        lines.push_back("resolved: " + exc(err));
        return lines;
    }
    field_lines(lines, "resolved ", r);
    for(const char* stage : {"main", "dq"})
    {
        rocke_attn_bwd_spec_kernel_name(&r, stage, name, sizeof(name), &err);
        lines.push_back(std::string("resolved kernel_name ") + stage + ": " + name);
    }
    rocke_attn_bwd_spec_t v;
    if(rocke_attn_bwd_validate_spec(&spec, arch.c_str(), &v, &err) == ROCKE_OK)
        lines.push_back("validate: ok");
    else
        lines.push_back("validate: " + exc(err));
    rocke_attn_bwd_lds_request_t req;
    if(rocke_attn_bwd_lds_request(&r, arch.c_str(), &req, &err) == ROCKE_OK)
    {
        request_lines(lines, req);
        static rocke_attn_bwd_lds_plan_t plan;
        if(rocke_attn_bwd_lds_plan(&r, arch.c_str(), &plan, &err) == ROCKE_OK)
            plan_lines(lines, plan);
        else
            lines.push_back("lds: " + exc(err));
    }
    else
        lines.push_back("lds: " + exc(err));
    int vg = 0, ag = 0;
    if(rocke_attn_bwd_register_estimates(&r, arch.c_str(), &vg, &ag, &err) == ROCKE_OK)
        lines.push_back("register_estimates: arch_vgpr=" + std::to_string(vg)
                        + " agpr=" + std::to_string(ag));
    else
        lines.push_back("register_estimates: " + exc(err));
    int rd = 0, wr = 0, mv = 0;
    if(rocke_attn_bwd_agpr_copy_budget(&r, arch.c_str(), &rd, &wr, &mv, &err) == ROCKE_OK)
        lines.push_back("agpr_copy_budget: reads=" + std::to_string(rd)
                        + " writes=" + std::to_string(wr) + " moves=" + std::to_string(mv));
    else
        lines.push_back("agpr_copy_budget: " + exc(err));
    if(rocke_attn_bwd_check_counted_sync(&r, arch.c_str(), &err) == ROCKE_OK)
        lines.push_back("check_counted_sync: ok");
    else
        lines.push_back("check_counted_sync: " + exc(err));
    rocke_attn_bwd_occupancy_t occ;
    if(rocke_attn_bwd_occupancy(&r, arch.c_str(), &occ, &err) == ROCKE_OK)
        lines.push_back(
            "occupancy: by_registers=" + std::to_string(occ.by_registers)
            + " by_lds=" + std::to_string(occ.by_lds) + " requested=" + repr_oi(occ.requested)
            + " waves=" + std::to_string(rocke_attn_bwd_occupancy_waves(&occ))
            + " hint_reachable=" + repr_b(rocke_attn_bwd_occupancy_hint_reachable(&occ)));
    else
        lines.push_back("occupancy: " + exc(err));
    rocke_attn_bwd_agpr_alloc_t pol;
    if(rocke_attn_bwd_policy_agpr_alloc(&r, arch.c_str(), &pol, &err) == ROCKE_OK)
        lines.push_back("policy_agpr_alloc: " + repr_agpr(pol));
    else
        lines.push_back("policy_agpr_alloc: " + exc(err));
    unsigned mask = 0;
    rocke_attn_bwd_static_unit_mults(r.static_scale, r.dkv_mode, &mask, &err);
    std::string mults = "static_unit_mults: ";
    std::vector<std::string> m;
    if(mask & ROCKE_ATTN_BWD_UNIT_DK_MULT)
        m.push_back("dk_mult");
    if(mask & ROCKE_ATTN_BWD_UNIT_DQ_MULT)
        m.push_back("dq_mult");
    if(mask & ROCKE_ATTN_BWD_UNIT_DS_MULT)
        m.push_back("ds_mult");
    for(size_t i = 0; i < m.size(); ++i)
        mults += (i ? "," : "") + m[i];
    lines.push_back(mults);
    const struct
    {
        const char* label;
        const char* stage;
        int convert_vec;
    } aux_stages[] = {{"prep", "prep", 8}, {"convert", "convert", 16}, {"prep cv16", "prep", 16}};
    for(const auto& as : aux_stages)
    {
        rocke_attn_bwd_aux_spec_t aux;
        char aux_name[512];
        if(rocke_attn_bwd_spec_aux_spec(&r, as.stage, NONE, as.convert_vec, &aux, &err) == ROCKE_OK
           && rocke_attn_bwd_aux_kernel_name(&aux, as.stage, aux_name, sizeof(aux_name))
                  == ROCKE_OK)
            lines.push_back(std::string("aux ") + as.label + ": " + aux_name);
        else
            lines.push_back(std::string("aux ") + as.label + ": " + exc_name(err.code));
    }
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
    std::string arch;
    rocke_attn_bwd_spec_t spec;
    if(!parse_config(CONFIGS[idx], &arch, &spec))
    {
        fprintf(stderr, "bad config %s\n", CONFIGS[idx]);
        return 1;
    }
    for(const std::string& line : dump(arch, spec))
    {
        fputs(line.c_str(), stdout);
        fputc('\n', stdout);
    }
    return 0;
}
