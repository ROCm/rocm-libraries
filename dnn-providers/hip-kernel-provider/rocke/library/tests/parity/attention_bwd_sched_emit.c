/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_sched_emit.c -- C-side emitter for the attention
 * backward stage tables (rocke/helper_rocke.instances.common._attention_bwd_sched.h
 * and the DmaBatch / dma_wait_counts facts of
 * helper_rocke.instances.common._attention_bwd_ring.h). Selects one config by
 * argv[1] (the same table as _CONFIGS of attention_bwd_sched_emit.py;
 * regenerate it with `attention_bwd_sched_emit.py --print-c-configs`) and
 * prints the same text dump as the Python emitter, so run_diff can
 * byte-compare the two (the dump stands in for a .ll: the counts emit no IR).
 * Compiled as C++20 against librocke_core.a.
 *
 * attention_bwd_sched_params_emit.c defines its own CONFIGS and includes this
 * file with ROCKE_ATTN_BWD_SCHED_EMIT_CONFIGS set.
 *
 * Only mode "ll" (the dump) is supported.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sched.h"

#ifndef ROCKE_ATTN_BWD_SCHED_EMIT_CONFIGS
static const char* const CONFIGS[] = {
    /* default gfx942 d32 fp16 sv8 batched atomic */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d32 fp16 sv8 batched split */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d32 fp16 sv1 batched atomic */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d32 fp16 sv1 batched split */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d32 bf16 sv8 batched atomic */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d32 bf16 sv8 batched split */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d32 bf16 sv1 batched atomic */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d32 bf16 sv1 batched split */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d64 fp16 sv8 batched atomic */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d64 fp16 sv8 batched split */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d64 fp16 sv1 batched atomic */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d64 fp16 sv1 batched split */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d64 bf16 sv8 batched atomic */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d64 bf16 sv8 batched split */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d64 bf16 sv1 batched atomic */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d64 bf16 sv1 batched split */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d128 fp16 sv8 batched atomic */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d128 fp16 sv8 batched split */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d128 fp16 sv1 batched atomic */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d128 fp16 sv1 batched split */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx942 d128 bf16 sv8 batched atomic */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d128 bf16 sv8 batched split */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d128 bf16 sv1 batched atomic */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx942 d128 bf16 sv1 batched split */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d32 fp16 sv8 batched atomic */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d32 fp16 sv8 batched split */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d32 fp16 sv1 batched atomic */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d32 fp16 sv1 batched split */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d32 bf16 sv8 batched atomic */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d32 bf16 sv8 batched split */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d32 bf16 sv1 batched atomic */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d32 bf16 sv1 batched split */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d64 fp16 sv8 batched atomic */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d64 fp16 sv8 batched split */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d64 fp16 sv1 batched atomic */
    "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d64 fp16 sv1 batched split */
    "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d64 bf16 sv8 batched atomic */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d64 bf16 sv8 batched split */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d64 bf16 sv1 batched atomic */
    "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d64 bf16 sv1 batched split */
    "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d128 fp16 sv8 batched atomic */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d128 fp16 sv8 batched split */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d128 fp16 sv1 batched atomic */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d128 fp16 sv1 batched split */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* default gfx950 d128 bf16 sv8 batched atomic */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d128 bf16 sv8 batched split */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d128 bf16 sv1 batched atomic */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* default gfx950 d128 bf16 sv1 batched split */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/128/bf16/causal_tl/gqa/unequal/fixed/8/t16/full (gfx942) */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:32;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t16/over (gfx942) */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:"
    "1x4;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over (gfx942) */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:"
    "1x4;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* tuned general/64/bf16/causal_br/gqa/square/fixed/8/t64/full (gfx942) */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/64/fp16/none/mha/square/thd/8/t64/over (gfx942) */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* tuned general/128/bf16/causal_br/gqa/square/fixed/8/t16/over (gfx950) */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:"
    "1x4;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/128/bf16/causal_tl/mha/square/fixed/8/t4/over (gfx950) */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "2x4;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/128/bf16/none/mha/square/fixed/8/t16/over (gfx950) */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:192;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:"
    "1x4;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/128/fp16/causal_br/gqa/unequal/fixed/8/t64/over (gfx950) */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:"
    "1x4;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* tuned general/128/fp16/causal_tl/gqa/square/fixed/8/t4/full (gfx950) */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "2x4;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over (gfx950) */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:192;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:"
    "1x4;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* tuned general/64/bf16/causal_br/gqa/unequal/fixed/8/t16/full (gfx950) */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:ds=pad32;sched:none;setprio:False;head_pack:False;acc_in_lds:"
    "False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_"
    "merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t4/full (gfx950) */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t16/full (gfx950) */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t16/over (gfx950) */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/64/bf16/none/mha/square/fixed/8/t64/full (gfx950) */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/64/bf16/none/mha/square/thd/8/t4/over (gfx950) */
    "geom|gfx950|head_size:64;k_m0:16;k_n0:192;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:ds=pad32;sched:none;setprio:False;head_pack:False;acc_in_lds:"
    "False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_"
    "merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned general/64/fp16/none/mha/square/fixed/8/t16/full (gfx950) */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* tuned general/64/fp16/none/mha/square/fixed/8/t4/full (gfx950) */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* tuned general/64/fp16/none/mha/square/thd/8/t64/over (gfx950) */
    "geom|gfx950|head_size:64;k_m0:16;k_n0:192;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* tuned dense 1024/4096/32/8/128/bf16/bshd/causal_br (gfx950) */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:"
    "1x4;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* tuned dense 2048/2048/12/12/64/bf16/bshd/none (gfx950) */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* spec: xt_lds gfx942 d64 (predictor swizzle) */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: xt_lds gfx942 d128 K^T resident image */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: xt_lds gfx942 d32 explicit XT swizzles */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:dot=none,kt_stage=pad8,qt=pad16;sched:none;setprio:False;head_"
    "pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_"
    "issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:"
    "fp16",
    /* spec: xt_lds gfx950 d64 wide G4 atom */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: xt_lds gfx950 d128 split dQ */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: xt_lds gfx950 d32 bf16 kt lds */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    /* spec: row xor buffers gfx942 d64 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:dkv_out=xor,do=xor,ds=xor,k_stage=xor,q=xor,v_stage=xor;sched:"
    "none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;"
    "stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_"
    "slicing:step;kw_dtype:fp16",
    /* spec: row pads gfx950 d64 */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:dkv_out=pad16,ds=pad8,q=pad32,unused=none;sched:none;setprio:"
    "False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:"
    "at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;"
    "kw_dtype:fp16",
    /* spec: rejected: ring-slotted row xor */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:q=xor;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: ring 3 gfx950 d64 */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:3;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: counted gfx942 d64 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:"
    "off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: counted gfx950 d128 late issue */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:"
    "off;ring_issue:early;counted_issue:late;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: counted late with stage_table_atomics */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:stage_table_atomics;setprio:False;head_pack:False;acc_"
    "in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;"
    "barrier_merge:off;ring_issue:early;counted_issue:late;g4_slicing:step;kw_dtype:fp16",
    /* spec: counted early with stage_table_atomics */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:stage_table_atomics;setprio:False;head_pack:False;acc_"
    "in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;"
    "barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: counted with split dQ */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:"
    "off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: counted with barrier merge */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:"
    "on;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: counted atomics past vmcnt */
    "geom|gfx942|head_size:128;k_m0:64;k_n0:32;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:"
    "off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: g4 slice_db gfx942 d64 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:slice_db;kw_dtype:fp16",
    /* spec: g4 slice -> step (kK4 = atom K) */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:16;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: g4 slice_db -> slice (kK4 = kN0) */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:slice;kw_dtype:fp16",
    /* spec: g4 slice_db_g3 gfx950 d128 kt lds */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:slice_db_g3;kw_dtype:fp16",
    /* spec: rejected: g4 slicing with ring */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: g4 slicing xt kt lds */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:slice;kw_dtype:fp16",
    /* spec: rejected: reservation below the estimate */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:"
    "1x4;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: acc_in_lds gfx942 d32 */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:True;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: acc_in_lds d64 LDS plan */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:True;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: acc_in_lds with direct epilogue */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:True;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: small tile lds route gfx942 d64 16x16x4 */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:16;k_k4:16;waves:4;grid_g0:1x1;grid_g1:1x4;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_"
    "path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:"
    "atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: small tile lds route gfx950 d32 32x32x2 */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:32;k_k4:32;waves:2;grid_g0:2x1;grid_g1:2x1;grid_g4:1x2;"
    "atoms:32/16/"
    "32;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_"
    "path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:"
    "atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: (2, W/2) grids gfx942 d64 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:2x4;grid_g1:2x4;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_"
    "path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:"
    "atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: non-natural grid on relabel */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:2x4;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: grid not a CTA grid */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:4x4;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_"
    "path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:"
    "atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: grid does not divide */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:1x8;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_"
    "path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:"
    "atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: natural next to small grid does not divide */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;"
    "atoms:16/16/"
    "16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_"
    "path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:"
    "atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: one wave on 32 x 32 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:32;k_k4:32;waves:1;grid_g0:1x1;grid_g1:1x1;grid_g4:1x1;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: one wave d128 debug tile gfx942 (scheduler) */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:16;k_k4:16;waves:1;grid_g0:1x1;grid_g1:1x1;grid_g4:1x1;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: wide atom on gfx942 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: tr_read on gfx942 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: dma on gfx942 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: dma gfx950 d32 ring 3 */
    "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:3;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: dma gfx950 tr_read q xor */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:dma;lds_swizzle:ds=pad16,q=xor;sched:none;setprio:False;head_pack:False;acc_in_"
    "lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_"
    "merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: dma tr_read q pad */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:dma;lds_swizzle:q=pad8;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: ring 3 on gfx942 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:3;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: kK4 above kN0 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:32;k_k4:64;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: warp_grid_g4 shape */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:4x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: warp_grid_g4 does not divide D */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x8;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: block_n not a multiple of 16 W */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:64;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: sched stage_table with replicated grid */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g0:1x2;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_"
    "path:vgpr;lds_swizzle:None;sched:stage_table;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: lds_only gfx942 d64 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:lds_only;stats_load:at_use;dq_issue:block;barrier_merge:"
    "off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: qdo joint too many chunks */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: loads_first with ring */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: stats step_top gfx950 */
    "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:step_top;dq_issue:block;barrier_merge:"
    "off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: barrier merge gfx950 d64 */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;"
    "ring_issue:late;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: barrier merge gfx942 d128 g3g1g4 */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: barrier merge with dma */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: barrier merge G4 not last */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: barrier merge split dQ */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: barrier merge acc_in_lds */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:True;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: stage_table setprio gfx942 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:stage_table;setprio:True;head_pack:False;acc_in_lds:"
    "False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_"
    "merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: stage_table_atomics split dQ */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:stage_table_atomics;setprio:False;head_pack:False;acc_"
    "in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;"
    "barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: xt_lds with ring */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:2;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: xt_lds with fences */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:fences_only;setprio:False;head_pack:False;acc_in_lds:"
    "False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_"
    "merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: spill combo gfx942 d128 edge row xor */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_"
    "res=xor,v_stage=xor;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;"
    "stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:"
    "early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: phase_low kv lds gfx942 d64 */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: kt lds kv reg gfx942 lds_plain (k_res copy) */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: kv lds tr_read gfx950 d64 */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: head_pack + edge tiles gfx950 */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:True;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: empty lds_swizzle */
    "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_"
    "mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: wide atom_g13 gfx950 block_m 32 */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/32/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: LDS plan too big */
    "geom|gfx942|head_size:128;k_m0:64;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "2x4;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: arch-VGPR estimate */
    "geom|gfx942|head_size:128;k_m0:64;k_n0:64;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: workgroup share */
    "geom|gfx942|head_size:128;k_m0:64;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "2x4;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: dq_issue fragment gfx942 d32 runtime */
    "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:fragment;barrier_merge:"
    "off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: kK4 below the wide atom */
    "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:16;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;"
    "atoms:32/16/"
    "32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: warp_grid_g4 does not divide kM0 */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "2x4;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: natural G1/G3 grid next to a small G0/G2 grid */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:32;k_k4:16;waves:4;grid_g0:1x2;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_"
    "path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:"
    "atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_"
    "issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: VGPR form within the prune slack */
    "geom|gfx942|head_size:64;k_m0:64;k_n0:128;k_k4:16;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: AGPR estimate past the slack */
    "geom|gfx942|head_size:64;k_m0:16;k_n0:256;k_k4:16;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;"
    "atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    /* spec: rejected: workgroup share of eight waves */
    "geom|gfx942|head_size:128;k_m0:16;k_n0:256;k_k4:16;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:"
    "1x8;atoms:16/16/"
    "16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;"
    "global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;"
    "dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;"
    "ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
};
#endif

static const int N_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

static const int NONE = ROCKE_ATTN_BWD_NONE;

typedef std::vector<std::string> Lines;
typedef std::map<std::string, std::string> Fields;
typedef rocke_attn_bwd_sched_pair_t Pair;
typedef std::vector<Pair> Pairs;

/* --------------------------------------------------------------- config */

static const char* keep(const std::string& s)
{
    static std::vector<std::string*> store;
    store.push_back(new std::string(s));
    return store.back()->c_str();
}

static bool parse_config(const char* config, std::string* kind, std::string* arch, Fields* f)
{
    std::string c = config;
    size_t a = c.find('|');
    size_t b = a == std::string::npos ? a : c.find('|', a + 1);
    if(b == std::string::npos)
        return false;
    *kind = c.substr(0, a);
    *arch = c.substr(a + 1, b - a - 1);
    std::string rest = c.substr(b + 1);
    size_t start = 0;
    while(start <= rest.size())
    {
        size_t semi = rest.find(';', start);
        std::string item
            = rest.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if(!item.empty())
        {
            size_t colon = item.find(':');
            if(colon == std::string::npos)
                (*f)[item] = "";
            else
                (*f)[item.substr(0, colon)] = item.substr(colon + 1);
        }
        if(semi == std::string::npos)
            break;
        start = semi + 1;
    }
    return true;
}

static bool has(const Fields& f, const char* k)
{
    return f.count(k) != 0;
}

static int ival(const Fields& f, const char* k)
{
    return atoi(f.at(k).c_str());
}

static bool bval(const Fields& f, const char* k)
{
    return f.at(k) == "True";
}

static const char* sval(const Fields& f, const char* k)
{
    const std::string& v = f.at(k);
    return v == "None" ? NULL : keep(v);
}

static void grid(const std::string& text, int g[2])
{
    size_t x = text.find('x');
    g[0] = atoi(text.substr(0, x).c_str());
    g[1] = atoi(text.substr(x + 1).c_str());
}

/* ------------------------------------------------------------- builders */

typedef rocke_attn_bwd_stage_table_t Table;
typedef rocke_attn_bwd_dq_stage_table_t DqTable;
typedef rocke_attn_bwd_stage_counts_t Counts;

static void set_table_field(Table* t, const std::string& k, const Fields& f)
{
    const char* key = k.c_str();
#define T_INT(n)             \
    if(k == #n)              \
    {                        \
        t->n = ival(f, key); \
        return;              \
    }
#define T_STR(n)             \
    if(k == #n)              \
    {                        \
        t->n = sval(f, key); \
        return;              \
    }
#define T_BOOL(n)            \
    if(k == #n)              \
    {                        \
        t->n = bval(f, key); \
        return;              \
    }
#define T_GRID(n)            \
    if(k == #n)              \
    {                        \
        grid(f.at(k), t->n); \
        return;              \
    }
    T_INT(head_size)
    T_INT(k_m0)
    T_INT(k_n0)
    T_INT(k_k4)
    T_INT(waves)
    T_GRID(grid_g4)
    T_INT(atom_g02)
    T_INT(atom_g13)
    T_INT(atom_g4)
    T_STR(dtype)
    T_STR(transpose_source)
    T_STR(kt_source)
    T_STR(global_path)
    T_STR(pt_route)
    T_INT(stage_vec)
    T_STR(dq_mode)
    T_STR(sched)
    T_BOOL(setprio)
    T_STR(kv_residency)
    T_BOOL(acc_in_lds)
    T_GRID(grid_g02)
    T_GRID(grid_g13)
    T_STR(pipeline)
    T_STR(stats_load)
    T_STR(dq_issue)
    T_STR(lds_swizzle)
    T_STR(g4_slicing)
    T_INT(ring_depth)
    T_STR(barrier_merge)
    T_STR(ring_issue)
    T_BOOL(head_pack)
    T_STR(loop_sync)
    T_STR(counted_issue)
    fprintf(stderr, "unknown table field %s\n", key);
    exit(1);
}

static void set_dq_field(DqTable* t, const std::string& k, const Fields& f)
{
    const char* key = k.c_str();
    T_INT(head_size)
    T_INT(k_m0)
    T_INT(k_n0)
    T_INT(waves)
    T_INT(atom_g02)
    T_INT(atom_g4)
    T_STR(dtype)
    T_STR(transpose_source)
    T_INT(kv_pad)
    T_BOOL(kv_prefetch)
    T_INT(stage_vec)
    T_STR(sched)
    fprintf(stderr, "unknown dq field %s\n", key);
    exit(1);
}
#undef T_INT
#undef T_STR
#undef T_BOOL
#undef T_GRID

static rocke_attn_bwd_sched_geom_t make_geom(const Fields& f)
{
    rocke_attn_bwd_sched_geom_t g = rocke_attn_bwd_sched_geom_empty();
    for(const auto& kv : f)
    {
        const std::string& k = kv.first;
        const char* key = k.c_str();
        if(k.rfind("kw_", 0) == 0)
            continue;
        if(k == "head_size")
            g.head_size = ival(f, key);
        else if(k == "k_m0")
            g.k_m0 = ival(f, key);
        else if(k == "k_n0")
            g.k_n0 = ival(f, key);
        else if(k == "k_k4")
            g.k_k4 = ival(f, key);
        else if(k == "waves")
            g.waves = ival(f, key);
        else if(k == "atoms")
        {
            if(kv.second != "None")
            {
                g.has_atoms = true;
                /* "a/b/c", each an int or None (an absent atom). */
                int* dst[3] = {&g.atom_g02, &g.atom_g13, &g.atom_g4};
                std::string rest = kv.second;
                for(int i = 0; i < 3; ++i)
                {
                    size_t cut = rest.find('/');
                    std::string part = rest.substr(0, cut);
                    *dst[i] = part == "None" ? NONE : atoi(part.c_str());
                    rest = cut == std::string::npos ? "" : rest.substr(cut + 1);
                }
            }
        }
        else if(k == "grid_g4" || k == "grid_g0" || k == "grid_g1")
        {
            if(kv.second == "None")
                continue;
            int* dst = k == "grid_g4" ? g.grid_g4 : (k == "grid_g0" ? g.grid_g0 : g.grid_g1);
            bool* h = k == "grid_g4" ? &g.has_grid_g4
                                     : (k == "grid_g0" ? &g.has_grid_g0 : &g.has_grid_g1);
            grid(kv.second, dst);
            *h = true;
        }
        else if(k == "ring_depth")
            g.ring_depth = kv.second == "None" ? NONE : ival(f, key);
        else if(k == "global_path")
            g.global_path = sval(f, key);
        else if(k == "barrier_merge")
            g.barrier_merge = sval(f, key);
        else if(k == "transpose_source")
            g.transpose_source = sval(f, key);
        else if(k == "kt_source")
            g.kt_source = sval(f, key);
        else if(k == "pt_route")
            g.pt_route = sval(f, key);
        else if(k == "stage_vec")
            g.stage_vec = ival(f, key);
        else if(k == "dq_mode")
            g.dq_mode = sval(f, key);
        else if(k == "sched")
            g.sched = sval(f, key);
        else if(k == "setprio")
            g.setprio = bval(f, key);
        else if(k == "kv_residency")
            g.kv_residency = sval(f, key);
        else if(k == "acc_in_lds")
            g.acc_in_lds = bval(f, key) ? 1 : 0;
        else if(k == "stats_load")
            g.stats_load = sval(f, key);
        else if(k == "dq_issue")
            g.dq_issue = sval(f, key);
        else if(k == "lds_swizzle")
        {
            g.has_lds_swizzle = true;
            g.lds_swizzle = sval(f, key);
        }
        else if(k == "g4_slicing")
            g.g4_slicing = sval(f, key);
        else if(k == "ring_issue")
            g.ring_issue = sval(f, key);
        else if(k == "head_pack")
            g.head_pack = bval(f, key);
        else if(k == "loop_sync")
            g.loop_sync = sval(f, key);
        else if(k == "counted_issue")
            g.counted_issue = sval(f, key);
        else
        {
            fprintf(stderr, "unknown geom field %s\n", key);
            exit(1);
        }
    }
    return g;
}

/* The table of a "geom" / "table" (or "dq" / "dqgeom") config; false and *e
 * set when the constructor raises. */
static bool make_table(
    const std::string& kind, const char* arch, const Fields& f, Table* t, rocke_attn_bwd_err_t* e)
{
    rocke_status_t st;
    if(kind == "geom")
    {
        rocke_attn_bwd_sched_geom_t g = make_geom(f);
        const char* dtype = has(f, "kw_dtype") ? sval(f, "kw_dtype") : "fp16";
        int sv = has(f, "kw_stage_vec") ? ival(f, "kw_stage_vec") : 8;
        const char* dq = has(f, "kw_dq_mode") ? sval(f, "kw_dq_mode") : "atomic";
        bool acc = has(f, "kw_acc_in_lds") ? bval(f, "kw_acc_in_lds") : false;
        st = rocke_attn_bwd_stage_table_from_geometry(&g, arch, dtype, sv, dq, acc, t, e);
    }
    else
    {
        *t = rocke_attn_bwd_stage_table_default();
        t->arch = arch;
        for(const auto& kv : f)
            set_table_field(t, kv.first, f);
        st = rocke_attn_bwd_stage_table_init(t, e);
    }
    return st == ROCKE_OK;
}

static bool make_dq(
    const std::string& kind, const char* arch, const Fields& f, DqTable* t, rocke_attn_bwd_err_t* e)
{
    rocke_status_t st;
    if(kind == "dqgeom")
    {
        rocke_attn_bwd_dq_sched_geom_t g;
        memset(&g, 0, sizeof(g));
        g.arch = arch;
        g.head_size = ival(f, "head_size");
        g.k_m0 = ival(f, "k_m0");
        g.k_n0 = ival(f, "k_n0");
        g.waves = ival(f, "waves");
        g.atom_g02 = ival(f, "atom_g02");
        g.atom_g4 = ival(f, "atom_g4");
        g.dtype = sval(f, "dtype");
        g.transpose_source = sval(f, "transpose_source");
        g.kv_pad = has(f, "kv_pad") ? ival(f, "kv_pad") : NONE;
        g.has_kv_prefetch = has(f, "kv_prefetch");
        g.kv_prefetch = g.has_kv_prefetch && bval(f, "kv_prefetch");
        g.stage_vec = has(f, "stage_vec") ? ival(f, "stage_vec") : NONE;
        g.sched = has(f, "sched") ? sval(f, "sched") : NULL;
        st = rocke_attn_bwd_dq_stage_table_from_geometry(&g, t, e);
    }
    else
    {
        *t = rocke_attn_bwd_dq_stage_table_default();
        t->arch = arch;
        for(const auto& kv : f)
            set_dq_field(t, kv.first, f);
        st = rocke_attn_bwd_dq_stage_table_init(t, e);
    }
    return st == ROCKE_OK;
}

/* The dump and main (attention_bwd_sched_ir_emit.c includes the parsing and
 * builders above only). */
#ifndef ROCKE_ATTN_BWD_SCHED_EMIT_TABLES_ONLY

/* ---------------------------------------------------------------- reprs */

static std::string S(long long v)
{
    return std::to_string(v);
}

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
    return v == NONE ? std::string("None") : S(v);
}

static std::string repr_grid(const int g[2])
{
    return "(" + S(g[0]) + ", " + S(g[1]) + ")";
}

static std::string repr_ints(const int* v, int n)
{
    std::string s = "(";
    for(int i = 0; i < n; ++i)
        s += (i ? ", " : "") + S(v[i]);
    return s + (n == 1 ? ",)" : ")");
}

static std::string pairs_text(const Pair* p, int n)
{
    std::string s = "[";
    for(int i = 0; i < n; ++i)
        s += (i ? " " : "") + S(p[i].mask) + ":" + S(p[i].count);
    return s + "]";
}

static std::string pairs_text(const Pairs& p)
{
    return pairs_text(p.data(), (int)p.size());
}

/* repr() of an {int: int} dict. */
static std::string int_dict(const Pair* p, int n)
{
    std::string s = "{";
    for(int i = 0; i < n; ++i)
        s += (i ? ", " : "") + S(p[i].mask) + ": " + S(p[i].count);
    return s + "}";
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

/* A Python exception the dump prints. */
struct Exc
{
    std::string text;
};

static void chk(rocke_status_t st, const rocke_attn_bwd_err_t& e)
{
    if(st != ROCKE_OK)
        throw Exc{exc_name(e.code) + ": " + e.msg};
}

template <class Fn>
static void try_line(Lines& lines, const std::string& label, Fn&& fn)
{
    try
    {
        lines.push_back(label + ": " + fn());
    }
    catch(const Exc& e)
    {
        lines.push_back(label + ": " + e.text);
    }
}

/* "m=c,m=c" */
static Pairs mlist(const std::string& text)
{
    Pairs out;
    size_t start = 0;
    while(start < text.size())
    {
        size_t comma = text.find(',', start);
        std::string item
            = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if(!item.empty())
        {
            size_t e = item.find('=');
            out.push_back(Pair{atoi(item.substr(0, e).c_str()), atoi(item.substr(e + 1).c_str())});
        }
        if(comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return out;
}

/* --------------------------------------------------------------- tables */

static const char* const GEMMS_PROBE[] = {"g0", "g1", "g2", "g3", "g4", "g9"};
static const char* const STAGES_PROBE[] = {"A", "B", "C", "D", "X"};
static const char* const DQ_STAGES_PROBE[] = {"A", "B", "C", "D"};

static std::string stage_text(const Counts& s)
{
    rocke_attn_bwd_err_t e;
    Pair mem[4], cen[5];
    rocke_attn_bwd_stage_counts_memory(&s, mem);
    chk(rocke_attn_bwd_stage_counts_census(&s, cen, &e), e);
    int n = 0;
    chk(rocke_attn_bwd_stage_counts_groups(&s, NULL, 0, &n, &e), e);
    std::vector<Pair> gs((size_t)n + 1);
    chk(rocke_attn_bwd_stage_counts_groups(&s, gs.data(), n, &n, &e), e);
    return std::string(s.stage) + " mfma=" + S(s.mfma) + " vr=" + S(s.vmem_read)
           + " vw=" + S(s.vmem_write) + " dr=" + S(s.ds_read) + " dw=" + S(s.ds_write)
           + " lead=" + pairs_text(s.lead, s.n_lead) + " spans=" + pairs_text(s.spans, s.n_spans)
           + " fixed=" + pairs_text(s.fixed, s.n_fixed) + " memory=" + pairs_text(mem, 4)
           + " census=" + int_dict(cen, 5) + " groups=" + pairs_text(gs.data(), n);
}

static void stage_lines(Lines& lines, const std::string& label, const Counts* st, int n)
{
    for(int i = 0; i < n; ++i)
    {
        try
        {
            lines.push_back(label + ": " + stage_text(st[i]));
        }
        catch(const Exc& x)
        {
            lines.push_back(label + ": " + st[i].stage + " " + x.text);
        }
    }
}

static std::string table_fields(const Table& t)
{
    return "arch=" + repr_s(t.arch) + " head_size=" + S(t.head_size) + " k_m0=" + S(t.k_m0)
           + " k_n0=" + S(t.k_n0) + " k_k4=" + S(t.k_k4) + " waves=" + S(t.waves)
           + " grid_g4=" + repr_grid(t.grid_g4) + " atom_g02=" + S(t.atom_g02)
           + " atom_g13=" + S(t.atom_g13) + " atom_g4=" + S(t.atom_g4) + " dtype=" + repr_s(t.dtype)
           + " transpose_source=" + repr_s(t.transpose_source) + " kt_source=" + repr_s(t.kt_source)
           + " global_path=" + repr_s(t.global_path) + " pt_route=" + repr_s(t.pt_route)
           + " stage_vec=" + S(t.stage_vec) + " dq_mode=" + repr_s(t.dq_mode)
           + " sched=" + repr_s(t.sched) + " setprio=" + repr_b(t.setprio)
           + " kv_residency=" + repr_s(t.kv_residency) + " acc_in_lds=" + repr_b(t.acc_in_lds)
           + " grid_g02=" + repr_grid(t.grid_g02) + " grid_g13=" + repr_grid(t.grid_g13)
           + " pipeline=" + repr_s(t.pipeline) + " stats_load=" + repr_s(t.stats_load)
           + " dq_issue=" + repr_s(t.dq_issue) + " lds_swizzle=" + repr_s(t.lds_swizzle)
           + " g4_slicing=" + repr_s(t.g4_slicing) + " ring_depth=" + S(t.ring_depth)
           + " barrier_merge=" + repr_s(t.barrier_merge) + " ring_issue=" + repr_s(t.ring_issue)
           + " head_pack=" + repr_b(t.head_pack) + " loop_sync=" + repr_s(t.loop_sync)
           + " counted_issue=" + repr_s(t.counted_issue);
}

static std::string gemm_dict(const rocke_attn_bwd_gemm_counts_t& m)
{
    return "{'g0': " + S(m.g0) + ", 'g1': " + S(m.g1) + ", 'g2': " + S(m.g2) + ", 'g3': " + S(m.g3)
           + ", 'g4': " + S(m.g4) + "}";
}

static std::string prefetch_repr(const rocke_attn_bwd_ring_prefetch_t& p)
{
    if(!p.present)
        return "{}";
    std::string s = "{'issue': " + repr_s(p.issue) + ", 'where': " + repr_s(p.where)
                    + ", 'vmem_read': " + S(p.vmem_read) + ", 'ds_write': " + S(p.ds_write);
    if(p.has_dma)
        s += ", 'dma_per_wave': " + repr_ints(p.dma_per_wave, 2) + ", 'dma_x4': " + S(p.dma_x4)
             + ", 'dma_x1': " + S(p.dma_x1);
    return s + "}";
}

static std::string hints_text(const rocke_attn_bwd_hint_t* h, int n)
{
    std::string s;
    for(int i = 0; i < n; ++i)
    {
        if(i)
            s += " ";
        switch(h[i].kind)
        {
        case ROCKE_ATTN_BWD_HINT_FENCE:
            s += "('fence', " + S(h[i].a) + ")";
            break;
        case ROCKE_ATTN_BWD_HINT_IGLP:
            s += "('iglp', " + S(h[i].a) + ")";
            break;
        case ROCKE_ATTN_BWD_HINT_PRIO:
            s += "('prio', " + S(h[i].a) + ")";
            break;
        case ROCKE_ATTN_BWD_HINT_GROUP:
            s += "('group', " + S(h[i].a) + ", " + S(h[i].b) + ")";
            break;
        }
    }
    return s;
}

static void region_lines(Lines& lines, const rocke_attn_bwd_sched_region_t* r, int n)
{
    for(int i = 0; i < n; ++i)
    {
        std::string lead = "None";
        if(r[i].has_lead)
            lead = int_dict(r[i].lead, r[i].n_lead);
        lines.push_back(std::string("region ") + r[i].name + " " + int_dict(r[i].ops, r[i].n_ops)
                        + " lead=" + lead);
    }
}

static std::string groups_text(const Table& t, const char* name, bool pf, bool pa)
{
    rocke_attn_bwd_err_t e;
    int n = 0;
    chk(rocke_attn_bwd_stage_table_groups(&t, name, pf, pa, NULL, 0, &n, &e), e);
    std::vector<Pair> gs((size_t)n + 1);
    chk(rocke_attn_bwd_stage_table_groups(&t, name, pf, pa, gs.data(), n, &n, &e), e);
    return pairs_text(gs.data(), n);
}

static Lines table_lines(const Table& t)
{
    Lines lines;
    rocke_attn_bwd_err_t e;
    lines.push_back("fields: " + table_fields(t));
    try_line(lines, "wave_size", [&] {
        int v;
        chk(rocke_attn_bwd_stage_table_wave_size(&t, &v, &e), e);
        return S(v);
    });
    try_line(lines, "instr_k", [&] {
        int k[5];
        chk(rocke_attn_bwd_stage_table_instr_k(&t, k, &e), e);
        return "{'g0': " + S(k[0]) + ", 'g1': " + S(k[1]) + ", 'g2': " + S(k[2])
               + ", 'g3': " + S(k[3]) + ", 'g4': " + S(k[4]) + "}";
    });
    for(const char* g : GEMMS_PROBE)
        try_line(lines, std::string("op ") + g, [&] {
            const char* id;
            int fl;
            chk(rocke_attn_bwd_stage_table_mma_op_id(&t, g, &id, &e), e);
            chk(rocke_attn_bwd_stage_table_frag_len(&t, g, &fl, &e), e);
            return std::string(id) + " frag_len=" + S(fl);
        });
    try_line(lines, "c_frag_len", [&] {
        int v;
        chk(rocke_attn_bwd_stage_table_c_frag_len(&t, &v, &e), e);
        return S(v);
    });
    try_line(lines, "mfma", [&] {
        rocke_attn_bwd_gemm_counts_t m;
        chk(rocke_attn_bwd_stage_table_mfma(&t, &m, &e), e);
        return gemm_dict(m) + " total=" + S(rocke_attn_bwd_gemm_counts_total(&m));
    });
    {
        const char* const* names;
        int n = rocke_attn_bwd_stage_table_stage_names(&t, &names);
        std::vector<std::string> v(names, names + n);
        lines.push_back("stage_names: " + rocke_attn_bwd::py_tuple(v));
    }
    try_line(lines, "dq_atomics_per_lane", [&] {
        int v;
        chk(rocke_attn_bwd_stage_table_dq_atomics_per_lane(&t, &v, &e), e);
        return S(v);
    });
    try_line(lines, "stats_loads", [&] {
        int v;
        chk(rocke_attn_bwd_stage_table_stats_loads(&t, &v, &e), e);
        return S(v);
    });
    lines.push_back("counted: " + repr_b(rocke_attn_bwd_stage_table_counted(&t)));
    try_line(lines, "counted_terms", [&] {
        rocke_attn_bwd_counted_terms_t c;
        chk(rocke_attn_bwd_stage_table_counted_terms(&t, &c, &e), e);
        if(!c.present)
            return std::string("{}");
        return "{'issue': " + repr_s(c.issue) + ", 'loads': " + S(c.loads)
               + ", 'stores': " + S(c.stores) + "}";
    });
    try_line(lines, "counted_waits", [&] {
        int w[3], n;
        chk(rocke_attn_bwd_stage_table_counted_waits(&t, w, &n, &e), e);
        return repr_ints(w, n);
    });
    try_line(lines, "ring_prefetch", [&] {
        rocke_attn_bwd_ring_prefetch_t p;
        chk(rocke_attn_bwd_stage_table_ring_prefetch(&t, &p, &e), e);
        return prefetch_repr(p);
    });
    try_line(lines, "prefetch_region", [&] {
        bool h;
        int v;
        chk(rocke_attn_bwd_stage_table_prefetch_region(&t, &h, &v, &e), e);
        return h ? "{" + S(ROCKE_SCHED_VMEM_READ) + ": " + S(v) + "}" : std::string("None");
    });
    try_line(lines, "prefetch_fence", [&] {
        bool v;
        chk(rocke_attn_bwd_stage_table_prefetch_fence(&t, &v, &e), e);
        return repr_b(v);
    });
    try_line(lines, "ring_waits", [&] {
        int w[1], n;
        chk(rocke_attn_bwd_stage_table_ring_waits(&t, w, &n, &e), e);
        return repr_ints(w, n);
    });
    try_line(lines, "ring_atomics_per_lane", [&] {
        int v;
        chk(rocke_attn_bwd_stage_table_ring_atomics_per_lane(&t, &v, &e), e);
        return S(v);
    });
    try_line(lines, "step_waits", [&] {
        int w[4], n;
        chk(rocke_attn_bwd_stage_table_step_waits(&t, w, &n, &e), e);
        return repr_ints(w, n);
    });
    for(bool pa : {true, false})
        try_line(lines, "vm_after_prefetch " + repr_b(pa), [&] {
            int v;
            chk(rocke_attn_bwd_stage_table_vm_after_prefetch(&t, pa, &v, &e), e);
            return S(v);
        });
    const bool combos[4][2] = {{true, true}, {true, false}, {false, true}, {false, false}};
    for(const auto& c : combos)
    {
        std::string label = "stages " + repr_b(c[0]) + " " + repr_b(c[1]);
        Counts st[4];
        rocke_attn_bwd_stage_pairs_store_t store;
        int n = 0;
        if(rocke_attn_bwd_stage_table_stages(&t, c[0], c[1], st, &n, &store, &e) != ROCKE_OK)
        {
            lines.push_back(label + ": " + exc_name(e.code) + ": " + e.msg);
            continue;
        }
        stage_lines(lines, label, st, n);
    }
    for(const char* name : STAGES_PROBE)
    {
        try_line(lines, std::string("stage ") + name, [&] {
            Counts s;
            rocke_attn_bwd_stage_pairs_store_t store;
            chk(rocke_attn_bwd_stage_table_stage(&t, name, true, true, &s, &store, &e), e);
            return stage_text(s);
        });
        try_line(
            lines, std::string("groups ") + name, [&] { return groups_text(t, name, true, true); });
        try_line(lines, std::string("groups ") + name + " False False", [&] {
            return groups_text(t, name, false, false);
        });
    }
    try_line(lines, "hint_sequence", [&] {
        int n = 0;
        chk(rocke_attn_bwd_stage_table_hint_sequence(&t, NULL, 0, &n, &e), e);
        std::vector<rocke_attn_bwd_hint_t> h((size_t)n + 1);
        chk(rocke_attn_bwd_stage_table_hint_sequence(&t, h.data(), n, &n, &e), e);
        return hints_text(h.data(), n);
    });
    {
        rocke_attn_bwd_sched_region_t rs[ROCKE_ATTN_BWD_SCHED_REGIONS_MAX];
        int n = 0;
        if(rocke_attn_bwd_stage_table_regions(&t, rs, &n, &e) != ROCKE_OK)
            lines.push_back("regions: " + exc_name(e.code) + ": " + e.msg);
        else
            region_lines(lines, rs, n);
    }
    const bool icm[2][2] = {{true, true}, {false, false}};
    for(const auto& c : icm)
        try_line(lines, "ir_count_model " + repr_b(c[0]) + " " + repr_b(c[1]), [&] {
            rocke_attn_bwd_ir_count_model_t m;
            chk(rocke_attn_bwd_stage_table_ir_count_model(&t, c[0], c[1], &m, &e), e);
            std::string sm = "{";
            for(int i = 0; i < m.n_stages; ++i)
                sm += (i ? ", " : "") + repr_s(m.stage_names[i]) + ": " + S(m.stage_mfma[i]);
            sm += "}";
            return "tile.sched_group_barrier=" + S(m.sched_group_barrier) + " tile.sched_barrier="
                   + S(m.sched_barrier) + " tile.s_setprio=" + S(m.s_setprio)
                   + " tile.iglp_opt=" + S(m.iglp_opt) + " tile.s_waitcnt=" + S(m.s_waitcnt)
                   + " group_counts=" + int_dict(m.group_counts, m.n_group_counts)
                   + " mfma=" + gemm_dict(m.mfma) + " mfma_total=" + S(m.mfma_total)
                   + " stage_mfma=" + sm + " pipeline=" + repr_s(m.pipeline)
                   + " ring_waits=" + repr_ints(m.ring_waits, m.n_ring_waits)
                   + " prefetch=" + prefetch_repr(m.prefetch);
        });
    return lines;
}

static std::string dq_fields(const DqTable& t)
{
    return "arch=" + repr_s(t.arch) + " head_size=" + S(t.head_size) + " k_m0=" + S(t.k_m0)
           + " k_n0=" + S(t.k_n0) + " waves=" + S(t.waves) + " atom_g02=" + S(t.atom_g02)
           + " atom_g4=" + S(t.atom_g4) + " dtype=" + repr_s(t.dtype) + " transpose_source="
           + repr_s(t.transpose_source) + " kv_pad=" + S(t.kv_pad) + " kv_prefetch="
           + repr_b(t.kv_prefetch) + " stage_vec=" + S(t.stage_vec) + " sched=" + repr_s(t.sched);
}

static std::string dq_mfma_repr(const rocke_attn_bwd_dq_mfma_t& m)
{
    return "{'g0': " + S(m.g0) + ", 'g2': " + S(m.g2) + ", 'g4': " + S(m.g4) + "}";
}

static std::string dq_prefetch_repr(const rocke_attn_bwd_dq_prefetch_t& p)
{
    if(!p.present)
        return "{}";
    return "{'vmem_read': " + S(p.vmem_read) + ", 'ds_write': " + S(p.ds_write) + "}";
}

static Lines dq_lines(const DqTable& t)
{
    Lines lines;
    rocke_attn_bwd_err_t e;
    lines.push_back("fields: " + dq_fields(t));
    try_line(lines, "wave_size", [&] {
        int v;
        chk(rocke_attn_bwd_dq_stage_table_wave_size(&t, &v, &e), e);
        return S(v);
    });
    lines.push_back("setprio=" + repr_b(rocke_attn_bwd_dq_stage_table_setprio(&t))
                    + " table=" + repr_b(rocke_attn_bwd_dq_stage_table_table(&t))
                    + " fenced=" + repr_b(rocke_attn_bwd_dq_stage_table_fenced(&t))
                    + " stage_names=('A', 'B', 'C')");
    try_line(lines, "mfma", [&] {
        rocke_attn_bwd_dq_mfma_t m;
        chk(rocke_attn_bwd_dq_stage_table_mfma(&t, &m, &e), e);
        return dq_mfma_repr(m);
    });
    try_line(lines, "prefetch", [&] {
        rocke_attn_bwd_dq_prefetch_t p;
        chk(rocke_attn_bwd_dq_stage_table_prefetch(&t, &p, &e), e);
        return dq_prefetch_repr(p);
    });
    {
        Counts st[3];
        rocke_attn_bwd_stage_pairs_store_t store;
        if(rocke_attn_bwd_dq_stage_table_stages(&t, st, &store, &e) != ROCKE_OK)
            lines.push_back("stages: " + exc_name(e.code) + ": " + e.msg);
        else
            stage_lines(lines, "stages", st, 3);
    }
    for(const char* name : DQ_STAGES_PROBE)
    {
        try_line(lines, std::string("stage ") + name, [&] {
            Counts s;
            rocke_attn_bwd_stage_pairs_store_t store;
            chk(rocke_attn_bwd_dq_stage_table_stage(&t, name, &s, &store, &e), e);
            return stage_text(s);
        });
        try_line(lines, std::string("groups ") + name, [&] {
            int n = 0;
            chk(rocke_attn_bwd_dq_stage_table_groups(&t, name, NULL, 0, &n, &e), e);
            std::vector<Pair> gs((size_t)n + 1);
            chk(rocke_attn_bwd_dq_stage_table_groups(&t, name, gs.data(), n, &n, &e), e);
            return pairs_text(gs.data(), n);
        });
    }
    try_line(lines, "hint_sequence", [&] {
        int n = 0;
        chk(rocke_attn_bwd_dq_stage_table_hint_sequence(&t, NULL, 0, &n, &e), e);
        std::vector<rocke_attn_bwd_hint_t> h((size_t)n + 1);
        chk(rocke_attn_bwd_dq_stage_table_hint_sequence(&t, h.data(), n, &n, &e), e);
        return hints_text(h.data(), n);
    });
    {
        rocke_attn_bwd_sched_region_t rs[3];
        if(rocke_attn_bwd_dq_stage_table_regions(&t, rs, &e) != ROCKE_OK)
            lines.push_back("regions: " + exc_name(e.code) + ": " + e.msg);
        else
            region_lines(lines, rs, 3);
    }
    try_line(lines, "ir_count_model", [&] {
        rocke_attn_bwd_dq_ir_count_model_t m;
        chk(rocke_attn_bwd_dq_stage_table_ir_count_model(&t, &m, &e), e);
        return "tile.sched_group_barrier=" + S(m.sched_group_barrier)
               + " tile.sched_barrier=" + S(m.sched_barrier) + " tile.s_setprio=" + S(m.s_setprio)
               + " tile.iglp_opt=" + S(m.iglp_opt) + " tile.s_waitcnt=" + S(m.s_waitcnt)
               + " group_counts=" + int_dict(m.group_counts, m.n_group_counts)
               + " mfma=" + dq_mfma_repr(m.mfma) + " stage_mfma={'A': " + S(m.stage_mfma[0])
               + ", 'B': " + S(m.stage_mfma[1]) + ", 'C': " + S(m.stage_mfma[2])
               + "} prefetch=" + dq_prefetch_repr(m.prefetch);
    });
    return lines;
}

/* The replace() variants of attention_bwd_sched_emit.py (_VARIANTS). */
struct Variant
{
    const char* label;
    const char* sched;
    int setprio; /* NONE: unchanged */
    const char* pipeline;
};

static const Variant VARIANTS[] = {
    {"sched='none'", "none", NONE, NULL},
    {"sched='iglp0'", "iglp0", NONE, NULL},
    {"sched='iglp1'", "iglp1", NONE, NULL},
    {"sched='fences_only'", "fences_only", NONE, NULL},
    {"sched='stage_table'", "stage_table", NONE, NULL},
    {"sched='stage_table_atomics'", "stage_table_atomics", NONE, NULL},
    {"sched='stage_table' setprio=True", "stage_table", 1, NULL},
    {"pipeline='ck' sched='stage_table_atomics'", "stage_table_atomics", NONE, "ck"},
    {"pipeline='ck'", NULL, NONE, "ck"},
};

static void table_variants(Lines& lines, const Table& base)
{
    for(const Variant& v : VARIANTS)
    {
        lines.push_back(std::string("variant ") + v.label);
        Table t = base;
        if(v.sched != NULL)
            t.sched = v.sched;
        if(v.setprio != NONE)
            t.setprio = v.setprio != 0;
        if(v.pipeline != NULL)
            t.pipeline = v.pipeline;
        rocke_attn_bwd_err_t e;
        if(rocke_attn_bwd_stage_table_init(&t, &e) != ROCKE_OK)
        {
            lines.push_back("init: " + exc_name(e.code) + ": " + e.msg);
            continue;
        }
        Lines more = table_lines(t);
        lines.insert(lines.end(), more.begin(), more.end());
    }
}

static void dq_variants(Lines& lines, const DqTable& base)
{
    for(int i = 0; i < ROCKE_ATTN_BWD_N_DQ_SCHEDS; ++i)
    {
        const char* s = ROCKE_ATTN_BWD_DQ_SCHEDS[i];
        lines.push_back("variant sched=" + repr_s(s));
        DqTable t = base;
        t.sched = s;
        rocke_attn_bwd_err_t e;
        if(rocke_attn_bwd_dq_stage_table_init(&t, &e) != ROCKE_OK)
        {
            lines.push_back("init: " + exc_name(e.code) + ": " + e.msg);
            continue;
        }
        Lines more = dq_lines(t);
        lines.insert(lines.end(), more.begin(), more.end());
    }
}

static std::string dma_instr(const rocke_attn_bwd_dma_batch_t& d, int k)
{
    const char* tensor;
    int j;
    rocke_attn_bwd_dma_batch_instruction(&d, k, &tensor, &j);
    return "(" + repr_s(tensor) + ", " + S(j) + ")";
}

static std::string dma_tuple(const rocke_attn_bwd_dma_batch_t& d, const std::vector<int>& ks)
{
    std::string s = "(";
    for(size_t i = 0; i < ks.size(); ++i)
        s += (i ? ", " : "") + dma_instr(d, ks[i]);
    return s + (ks.size() == 1 ? ",)" : ")");
}

static Lines dump(const std::string& kind, const std::string& arch_s, const Fields& f)
{
    Lines lines;
    lines.push_back("kind: " + kind + " arch: " + arch_s);
    const char* arch = keep(arch_s);
    rocke_attn_bwd_err_t e;
    if(kind == "geom" || kind == "table")
    {
        Table t;
        if(!make_table(kind, arch, f, &t, &e))
        {
            lines.push_back("init: " + exc_name(e.code) + ": " + e.msg);
            return lines;
        }
        lines.push_back("init: ok");
        Lines more = table_lines(t);
        lines.insert(lines.end(), more.begin(), more.end());
        if(kind == "geom")
            table_variants(lines, t);
    }
    else if(kind == "dq" || kind == "dqgeom")
    {
        DqTable t;
        if(!make_dq(kind, arch, f, &t, &e))
        {
            lines.push_back("init: " + exc_name(e.code) + ": " + e.msg);
            return lines;
        }
        lines.push_back("init: ok");
        Lines more = dq_lines(t);
        lines.insert(lines.end(), more.begin(), more.end());
        if(kind == "dqgeom")
            dq_variants(lines, t);
    }
    else if(kind == "groups")
    {
        try_line(lines, "groups", [&] {
            Pairs mem = mlist(f.at("mem"));
            Pairs lead = has(f, "lead") && f.at("lead") != "None" ? mlist(f.at("lead")) : Pairs();
            Pairs spans
                = has(f, "spans") && f.at("spans") != "None" ? mlist(f.at("spans")) : Pairs();
            int n = 0;
            chk(rocke_attn_bwd_interleave_groups(ival(f, "mfma"),
                                                 mem.data(),
                                                 (int)mem.size(),
                                                 lead.data(),
                                                 (int)lead.size(),
                                                 spans.data(),
                                                 (int)spans.size(),
                                                 NULL,
                                                 0,
                                                 &n,
                                                 &e),
                e);
            std::vector<Pair> gs((size_t)n + 1);
            chk(rocke_attn_bwd_interleave_groups(ival(f, "mfma"),
                                                 mem.data(),
                                                 (int)mem.size(),
                                                 lead.data(),
                                                 (int)lead.size(),
                                                 spans.data(),
                                                 (int)spans.size(),
                                                 gs.data(),
                                                 n,
                                                 &n,
                                                 &e),
                e);
            return pairs_text(gs.data(), n);
        });
    }
    else if(kind == "counts")
    {
        Counts s;
        memset(&s, 0, sizeof(s));
        s.stage = sval(f, "stage");
        s.mfma = ival(f, "mfma");
        s.vmem_read = has(f, "vmem_read") ? ival(f, "vmem_read") : 0;
        s.vmem_write = has(f, "vmem_write") ? ival(f, "vmem_write") : 0;
        s.ds_read = has(f, "ds_read") ? ival(f, "ds_read") : 0;
        s.ds_write = has(f, "ds_write") ? ival(f, "ds_write") : 0;
        Pairs lead = has(f, "lead") ? mlist(f.at("lead")) : Pairs();
        Pairs spans = has(f, "spans") ? mlist(f.at("spans")) : Pairs();
        Pairs fixed = has(f, "fixed") ? mlist(f.at("fixed")) : Pairs();
        s.lead = lead.data(), s.n_lead = (int)lead.size();
        s.spans = spans.data(), s.n_spans = (int)spans.size();
        s.fixed = fixed.data(), s.n_fixed = (int)fixed.size();
        Pair bm[5];
        rocke_attn_bwd_stage_counts_by_mask(&s, bm);
        lines.push_back("by_mask: " + int_dict(bm, 5));
        try_line(lines, "stage", [&] { return stage_text(s); });
    }
    else if(kind == "atoms")
    {
        const char* dtype = has(f, "dtype") ? sval(f, "dtype") : "fp16";
        try_line(lines, "native_atom_ks", [&] {
            int n = 0;
            chk(rocke_attn_bwd_native_atom_ks(arch, dtype, NULL, 0, &n, &e), e);
            std::vector<int> ks((size_t)n + 1);
            chk(rocke_attn_bwd_native_atom_ks(arch, dtype, ks.data(), n, &n, &e), e);
            return repr_ints(ks.data(), n);
        });
        for(int k : {16, 32, 64, 8, 48})
            try_line(lines, "instruction_k " + S(k), [&] {
                int v;
                chk(rocke_attn_bwd_instruction_k(arch, k, dtype, &v, &e), e);
                return S(v);
            });
    }
    else if(kind == "dma")
    {
        try_line(lines, "batch", [&] {
            rocke_attn_bwd_dma_batch_t d;
            chk(rocke_attn_bwd_dma_batch_init(&d,
                                              ival(f, "block_m"),
                                              ival(f, "head_size"),
                                              ival(f, "waves"),
                                              ival(f, "wave_size"),
                                              2,
                                              &e),
                e);
            std::vector<int> all;
            for(int k = 0; k < rocke_attn_bwd_dma_batch_n_instructions(&d); ++k)
                all.push_back(k);
            std::string waves;
            for(int w = 0; w < d.waves; ++w)
            {
                int n = rocke_attn_bwd_dma_batch_of_wave(&d, w, NULL, 0);
                std::vector<int> ks((size_t)n + 1);
                rocke_attn_bwd_dma_batch_of_wave(&d, w, ks.data(), n);
                ks.resize((size_t)n);
                waves += (w ? " " : "") + dma_tuple(d, ks);
            }
            return "pieces=" + S(rocke_attn_bwd_dma_batch_pieces(&d))
                   + " instructions=" + dma_tuple(d, all) + " of_wave=" + waves
                   + " min_per_wave=" + S(rocke_attn_bwd_dma_batch_min_per_wave(&d));
        });
        try_line(lines, "waits", [&] {
            rocke_attn_bwd_ring_waits_t w;
            chk(rocke_attn_bwd_dma_wait_counts(
                    ival(f, "ring_depth"), ival(f, "n_qd"), ival(f, "n_a"), &w, &e),
                e);
            return "prologue=" + repr_oi(w.prologue) + " step_top=" + repr_oi(w.step_top)
                   + " step_end=" + repr_oi(w.step_end);
        });
    }
    else
    {
        fprintf(stderr, "unknown kind %s\n", kind.c_str());
        exit(1);
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
    std::string kind, arch;
    Fields f;
    if(!parse_config(CONFIGS[idx], &kind, &arch, &f))
    {
        fprintf(stderr, "bad config %s\n", CONFIGS[idx]);
        return 1;
    }
    for(const std::string& line : dump(kind, arch, f))
    {
        fputs(line.c_str(), stdout);
        fputc('\n', stdout);
    }
    return 0;
}

#endif /* ROCKE_ATTN_BWD_SCHED_EMIT_TABLES_ONLY */
