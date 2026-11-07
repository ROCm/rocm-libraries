/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_dense_bwd_main_emit.c -- C-side emitter for the
 * dense attention backward main kernel and the aux kernels of the dense
 * plans. Builds, identically to the Python emitter
 * attention_dense_bwd_main_emit.py, the kernel of a config through
 * rocke_build_attn_dense_bwd_main_new (rocke/instance_attention_dense_bwd_spec.h)
 * or rocke_build_attn_bwd_aux_new, lowers it via rocke_lower_kernel_to_llvm
 * (flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared. The body-knob parser comes from attention_bwd_spec_emit.c
 * (included with ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS and
 * ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY). The table holds every Python config
 * resolved (the gate cases planned); regenerate it with
 * `attention_dense_bwd_main_emit.py --print-c-configs`.
 *
 * argv[1] is the config index. Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <string>
#include <utility>
#include <vector>

#include "rocke/instance_attention_bwd.h"
#include "rocke/instance_attention_bwd_aux.h"
#include "rocke/instance_attention_dense_bwd_spec.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static const char* const CONFIGS[] = {
    /* case dense/gfx950/tuned0 */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;dq_mode:split#acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_"
    "g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;global_"
    "path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_"
    "grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* case dense/gfx950/tuned1 */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;dq_mode:split#acc_in_lds:False;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;"
    "block_k4:32;block_m:32;block_n:128;bwd_exp2:fast;global_path:dma;kt_source:lds;kv_residency:"
    "reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;setprio:False;stats_fma:True;"
    "stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:2x2;"
    "waves:4;waves_per_eu:2",
    /* case dense/gfx950/tuned2 */
    "gfx950|seqlen_q:4096;seqlen_kv:4096;num_query_heads:64;num_kv_heads:8;head_size:128;ragged:"
    "True;dkv_mode:atomic#acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;atom_g02:16x16x32;"
    "atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:192;global_path:dma;kt_"
    "source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;setprio:"
    "False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;"
    "warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* case dense/gfx942/default0 */
    "gfx942|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True#",
    /* case dense/gfx950/default0 */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True#",
    /* case dense/gfx942/default1 */
    "gfx942|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False#",
    /* case dense/gfx950/default1 */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False#",
    /* case dense/gfx942/default2 */
    "gfx942|seqlen_q:4096;seqlen_kv:4096;num_query_heads:64;num_kv_heads:8;head_size:128#",
    /* case dense/gfx950/default2 */
    "gfx950|seqlen_q:4096;seqlen_kv:4096;num_query_heads:64;num_kv_heads:8;head_size:128#",
    /* prep of dense/gfx950/tuned0 */
    "aux:prep|gfx950|head_size:128;dtype:bf16",
    /* prep of dense/gfx950/tuned1 */
    "aux:prep|gfx950|head_size:64;dtype:bf16",
    /* convert of dense/gfx950/tuned2 */
    "aux:convert|gfx950|head_size:128;dtype:bf16",
    /* prep of dense/gfx942/default0 */
    "aux:prep|gfx942|head_size:128;dtype:bf16",
    /* convert of dense/gfx942/default0 */
    "aux:convert|gfx942|head_size:128;dtype:bf16",
    /* prep of dense/gfx942/default1 */
    "aux:prep|gfx942|head_size:64;dtype:bf16",
    /* convert of dense/gfx942/default1 */
    "aux:convert|gfx942|head_size:64;dtype:bf16",
    /* convert of dense/gfx950/default1 */
    "aux:convert|gfx950|head_size:64;dtype:bf16",
    /* causal tl gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:True#",
    /* causal br gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal_bottom_"
    "right:True#",
    /* no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False#",
    /* window 64 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;sliding_"
    "window:64#",
    /* window 64 br gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;sliding_"
    "window:64;causal_bottom_right:True#",
    /* band 32x16 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False;"
    "window_left:32;window_right:16#",
    /* band left only gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False;"
    "window_left:100#",
    /* band right only br gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False;"
    "window_right:0;causal_bottom_right:True#",
    /* ragged causal gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;seqlen_q:200;"
    "seqlen_kv:300;ragged:True#",
    /* ragged no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;seqlen_q:200;"
    "seqlen_kv:300;ragged:True;causal:False#",
    /* ragged clamp gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;seqlen_q:200;"
    "seqlen_kv:300;ragged:True;oob:clamp#",
    /* edge tiles off gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;edge_tiles:"
    "False#",
    /* edge tiles off no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;edge_tiles:"
    "False;causal:False#",
    /* bhsd gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;layout:bhsd#",
    /* fp16 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;dtype:fp16#",
    /* pinned batch gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;batch:3#",
    /* oob clamp gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;oob:clamp#",
    /* atomic dkv gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;dkv_mode:"
    "atomic#",
    /* atomic dkv token-major gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;dkv_mode:"
    "atomic;ws_layout:token_major#",
    /* split dq gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;dq_mode:"
    "split#",
    /* epilogue direct gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;epilogue:"
    "direct#",
    /* epilogue pair gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;epilogue:lds_"
    "staged_pair#",
    /* token-major gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;ws_layout:"
    "token_major#",
    /* persistent gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;persistent:"
    "True#",
    /* persistent hkv gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;persistent:"
    "True;persist_decode:hkv_major#",
    /* persistent atomic gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;persistent:"
    "True;dkv_mode:atomic#",
    /* xcd remap gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;xcd_remap:"
    "True#",
    /* xcd remap atomic gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;xcd_remap:"
    "True;dkv_mode:atomic#",
    /* name stem gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;name:my_"
    "dense#",
    /* ring vgpr 2 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#ring_depth:2",
    /* ring vgpr 3 gfx942 (gfx942 refuses) */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#ring_depth:3",
    /* ring vgpr 2 clamp gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;oob:clamp#"
    "ring_depth:2",
    /* ring vgpr 2 ragged gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;seqlen_q:200;"
    "seqlen_kv:300;ragged:True#ring_depth:2",
    /* static_scale runtime gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#static_scale:"
    "runtime",
    /* static_scale convert atomic gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;dkv_mode:"
    "atomic#static_scale:convert",
    /* lb_interleave gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#lb_interleave:"
    "True",
    /* lb_interleave persistent gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;persistent:"
    "True#lb_interleave:True",
    /* ptr_noalias workspace gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#ptr_noalias:"
    "workspace",
    /* ptr_noalias all atomic gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;dkv_mode:"
    "atomic#ptr_noalias:all",
    /* head_outer gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#step_order:"
    "head_outer",
    /* head_outer no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False#"
    "step_order:head_outer",
    /* q_rotate gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#q_rotate:True",
    /* counted gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#loop_sync:"
    "counted",
    /* stage_table_atomics gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#sched:stage_"
    "table_atomics",
    /* stats_vec 4 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#stats_vec:4;"
    "stats_fma:True",
    /* waves_per_eu 1 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#waves_per_eu:"
    "1",
    /* scheduler_strategy gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#scheduler_"
    "strategy:max-ilp",
    /* interior_clamp gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#interior_"
    "clamp:True",
    /* counted stage_table_atomics gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#loop_sync:"
    "counted;sched:stage_table_atomics",
    /* counted no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False#"
    "loop_sync:counted",
    /* stage_table_atomics no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False#"
    "sched:stage_table_atomics",
    /* head_outer q_rotate gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#step_order:"
    "head_outer;q_rotate:True",
    /* head_outer q_rotate no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False#"
    "step_order:head_outer;q_rotate:True",
    /* head_outer stage_table_atomics gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#step_order:"
    "head_outer;sched:stage_table_atomics",
    /* rejected: lb_mirror gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#lb_mirror:"
    "True",
    /* rejected: slab_carry gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#slab_carry:"
    "True",
    /* rejected: ws_bounds buffer gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#ws_bounds:"
    "buffer",
    /* rejected: owned knob gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#dtype:bf16",
    /* rejected: owned knob name gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#name:other",
    /* rejected: unaligned not ragged gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;seqlen_q:200#",
    /* causal tl gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:True#",
    /* causal br gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal_"
    "bottom_right:True#",
    /* no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:"
    "False#",
    /* window 64 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;sliding_"
    "window:64#",
    /* window 64 br gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;sliding_"
    "window:64;causal_bottom_right:True#",
    /* band 32x16 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False;"
    "window_left:32;window_right:16#",
    /* band left only gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False;"
    "window_left:100#",
    /* band right only br gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False;"
    "window_right:0;causal_bottom_right:True#",
    /* ragged causal gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;seqlen_q:200;"
    "seqlen_kv:300;ragged:True#",
    /* ragged no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;seqlen_q:200;"
    "seqlen_kv:300;ragged:True;causal:False#",
    /* ragged clamp gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;seqlen_q:200;"
    "seqlen_kv:300;ragged:True;oob:clamp#",
    /* edge tiles off gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;edge_tiles:"
    "False#",
    /* edge tiles off no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;edge_tiles:"
    "False;causal:False#",
    /* bhsd gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;layout:bhsd#",
    /* fp16 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;dtype:fp16#",
    /* pinned batch gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;batch:3#",
    /* oob clamp gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;oob:clamp#",
    /* atomic dkv gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;dkv_mode:"
    "atomic#",
    /* atomic dkv token-major gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;dkv_mode:"
    "atomic;ws_layout:token_major#",
    /* split dq gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;dq_mode:"
    "split#",
    /* epilogue direct gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;epilogue:"
    "direct#",
    /* epilogue pair gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;epilogue:lds_"
    "staged_pair#",
    /* token-major gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;ws_layout:"
    "token_major#",
    /* persistent gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;persistent:"
    "True#",
    /* persistent hkv gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;persistent:"
    "True;persist_decode:hkv_major#",
    /* persistent atomic gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;persistent:"
    "True;dkv_mode:atomic#",
    /* xcd remap gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;xcd_remap:"
    "True#",
    /* xcd remap atomic gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;xcd_remap:"
    "True;dkv_mode:atomic#",
    /* name stem gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;name:my_"
    "dense#",
    /* ring vgpr 2 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#ring_depth:2",
    /* ring vgpr 3 gfx950 (gfx942 refuses) */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#ring_depth:3",
    /* ring vgpr 2 clamp gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;oob:clamp#"
    "ring_depth:2",
    /* ring vgpr 2 ragged gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;seqlen_q:200;"
    "seqlen_kv:300;ragged:True#ring_depth:2",
    /* static_scale runtime gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#static_scale:"
    "runtime",
    /* static_scale convert atomic gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;dkv_mode:"
    "atomic#static_scale:convert",
    /* lb_interleave gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#lb_"
    "interleave:True",
    /* lb_interleave persistent gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;persistent:"
    "True#lb_interleave:True",
    /* ptr_noalias workspace gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#ptr_noalias:"
    "workspace",
    /* ptr_noalias all atomic gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;dkv_mode:"
    "atomic#ptr_noalias:all",
    /* head_outer gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#step_order:"
    "head_outer",
    /* head_outer no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False#"
    "step_order:head_outer",
    /* q_rotate gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#q_rotate:"
    "True",
    /* counted gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#loop_sync:"
    "counted",
    /* stage_table_atomics gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#sched:stage_"
    "table_atomics",
    /* stats_vec 4 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#stats_vec:4;"
    "stats_fma:True",
    /* waves_per_eu 1 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#waves_per_eu:"
    "1",
    /* scheduler_strategy gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#scheduler_"
    "strategy:max-ilp",
    /* interior_clamp gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#interior_"
    "clamp:True",
    /* counted stage_table_atomics gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#loop_sync:"
    "counted;sched:stage_table_atomics",
    /* counted no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False#"
    "loop_sync:counted",
    /* stage_table_atomics no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False#"
    "sched:stage_table_atomics",
    /* head_outer q_rotate gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#step_order:"
    "head_outer;q_rotate:True",
    /* head_outer q_rotate no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False#"
    "step_order:head_outer;q_rotate:True",
    /* head_outer stage_table_atomics gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#step_order:"
    "head_outer;sched:stage_table_atomics",
    /* rejected: lb_mirror gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#lb_mirror:"
    "True",
    /* rejected: slab_carry gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#slab_carry:"
    "True",
    /* rejected: ws_bounds buffer gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#ws_bounds:"
    "buffer",
    /* rejected: owned knob gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#dtype:bf16",
    /* rejected: owned knob name gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#name:other",
    /* rejected: unaligned not ragged gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;seqlen_q:"
    "200#",
    /* global_path dma ring 2 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#global_path:"
    "dma;ring_depth:2",
    /* global_path dma ring 3 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#global_path:"
    "dma;ring_depth:3",
    /* global_path dma ring 1 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#global_path:"
    "dma;ring_depth:1",
    /* global_path dma ring 2 split gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;dq_mode:"
    "split#global_path:dma;ring_depth:2",
    /* barrier merge gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#barrier_"
    "merge:on",
    /* asm_g13 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#agpr_alloc:"
    "128x128;agpr_route:asm_g13",
    /* vgpr_form gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;head_size:128#"
    "agpr_alloc:192x192;agpr_route:vgpr_form",
    /* agpr_alloc 0x0 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#agpr_alloc:"
    "0x0",
    /* body_variant xt not built gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;body_variant:"
    "xt#",
    /* body_variant trload not built gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;body_variant:"
    "trload#",
    /* rejected: owned knob at default */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#dtype:fp16",
    /* rejected: owned opt knob at default */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#q_rotate:"
    "True;edge_tiles:None",
    /* rejected: repeated knob */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#q_rotate:True;"
    "q_rotate:True",
    /* rejected: repeated owned knob */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#dtype:fp16;"
    "dtype:fp16",
    /* knob at default dropped */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#q_rotate:"
    "False",
    /* rejected: base class */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;cls:base#",
    /* rejected: gfx950 class on gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;cls:gfx950#",
    /* rejected: unknown arch */
    "gfx1201|seqlen_q:256;seqlen_kv:256;num_query_heads:8;num_kv_heads:8;head_size:64#",
    /* rejected: head size 32 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;head_size:32#",
    /* rejected: window without causal */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False;"
    "sliding_window:8#",
    /* rejected: xcd and persistent */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;persistent:"
    "True;xcd_remap:True#",
    /* rejected: heads not a multiple */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;num_query_"
    "heads:6;num_kv_heads:4#",
    /* rejected: seqlen_q 0 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;seqlen_q:0#",
    /* rejected: batch 0 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;batch:0#",
    /* rejected: negative window */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;sliding_"
    "window:-1#",
    /* rejected: causal and explicit band */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;window_left:"
    "4#",
    /* rejected: window_left -2 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False;"
    "window_left:-2#",
    /* rejected: persist_decode without persistent */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;persist_"
    "decode:hkv_major#",
    /* rejected: bad name */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;name:bad-"
    "name#",
    /* rejected: bad layout */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;layout:sbhd#",
    /* rejected: bad oob */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;oob:wrap#",
    /* rejected: slab past the 32-bit offset */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;seqlen_q:"
    "140000;ragged:True;num_query_heads:64;num_kv_heads:8#",
    /* rejected: pinned batch past 2^31 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;batch:200000#",
    /* tuned0 persistent */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;dq_mode:split;persistent:True#acc_in_lds:False;agpr_alloc:128x128;agpr_"
    "route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;"
    "block_n:128;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_"
    "agpr:False;sched:none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* tuned0 xcd remap */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;dq_mode:split;xcd_remap:True#acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* tuned0 atomic dQ */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;dq_mode:split;dq_mode:atomic#acc_in_lds:False;agpr_alloc:128x128;agpr_route:"
    "asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:"
    "128;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:"
    "False;sched:none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* tuned0 epilogue pair */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;dq_mode:split;epilogue:lds_staged_pair#acc_in_lds:False;agpr_alloc:128x128;"
    "agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:"
    "32;block_n:128;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_"
    "in_agpr:False;sched:none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;"
    "warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* tuned0 ring 3 */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;dq_mode:split#acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_"
    "g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;global_"
    "path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:"
    "none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_"
    "grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ring_depth:3",
    /* tuned1 causal */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;dq_mode:split;causal:True#acc_in_lds:False;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:"
    "16x16x32;block_k4:32;block_m:32;block_n:128;bwd_exp2:fast;global_path:dma;kt_source:lds;kv_"
    "residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;setprio:False;stats_"
    "fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_"
    "g4:2x2;waves:4;waves_per_eu:2",
    /* tuned1 ragged */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;dq_mode:split;ragged:True;seqlen_q:2000#acc_in_lds:False;atom_g02:16x16x32;atom_g13:"
    "16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_exp2:fast;global_path:dma;kt_"
    "source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;setprio:"
    "False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;"
    "warp_grid_g4:2x2;waves:4;waves_per_eu:2",
    /* tuned1 clamp */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;dq_mode:split;oob:clamp#acc_in_lds:False;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:"
    "16x16x32;block_k4:32;block_m:32;block_n:128;bwd_exp2:fast;global_path:dma;kt_source:lds;kv_"
    "residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;setprio:False;stats_"
    "fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_"
    "g4:2x2;waves:4;waves_per_eu:2",
    /* tuned2 direct dkv */
    "gfx950|seqlen_q:4096;seqlen_kv:4096;num_query_heads:64;num_kv_heads:8;head_size:128;ragged:"
    "True;dkv_mode:atomic;dkv_mode:direct#acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:192;"
    "global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;"
    "sched:none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:"
    "1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* tuned2 split */
    "gfx950|seqlen_q:4096;seqlen_kv:4096;num_query_heads:64;num_kv_heads:8;head_size:128;ragged:"
    "True;dkv_mode:atomic;dq_mode:split#acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:192;"
    "global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;"
    "sched:none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:"
    "1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* tuned2 aligned */
    "gfx950|seqlen_q:4096;seqlen_kv:4096;num_query_heads:64;num_kv_heads:8;head_size:128;ragged:"
    "True;dkv_mode:atomic;ragged:False;seqlen_q:4096;seqlen_kv:3840#acc_in_lds:False;agpr_alloc:"
    "192x192;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;"
    "block_m:16;block_n:192;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_"
    "depth:2;s_in_agpr:False;sched:none;setprio:False;stats_fma:True;stats_vec:4;transpose_source:"
    "tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1",
    /* default942 split */
    "gfx942|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;dq_mode:split#",
    /* default942 atomic */
    "gfx942|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;dkv_mode:atomic#",
    /* default942 window */
    "gfx942|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;sliding_window:512#",
    /* default942N bhsd */
    "gfx942|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;layout:bhsd#",
    /* default942C persistent hkv */
    "gfx942|seqlen_q:4096;seqlen_kv:4096;num_query_heads:64;num_kv_heads:8;head_size:128;"
    "persistent:True;persist_decode:hkv_major#",
    /* default950 edge off */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;edge_tiles:False#",
    /* default950N ring 2 */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False#ring_depth:2",
};

#define ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS 1
#define ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY 1
#include "attention_bwd_spec_emit.c"

/* "<field>:<value>;..." items of rest, in order. */
static std::vector<std::pair<std::string, std::string>> dense_items(const std::string& rest)
{
    std::vector<std::pair<std::string, std::string>> out;
    size_t start = 0;
    while(start < rest.size())
    {
        size_t semi = rest.find(';', start);
        std::string item
            = rest.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if(!item.empty())
        {
            size_t colon = item.find(':');
            out.emplace_back(item.substr(0, colon), item.substr(colon + 1));
        }
        if(semi == std::string::npos)
            break;
        start = semi + 1;
    }
    return out;
}

static bool
    dense_set_field(rocke_attn_bwd_dense_spec_t* s, const std::string& n, const std::string& v)
{
    auto opt_int = [&](int* f) {
        *f = v == "None" ? ROCKE_ATTN_BWD_NONE : atoi(v.c_str());
        return true;
    };
    auto str = [&](const char** f) {
        *f = v == "None" ? NULL : keep(v);
        return true;
    };
    auto flag = [&](bool* f) {
        if(v != "True" && v != "False")
            return false;
        *f = v == "True";
        return true;
    };
    if(n == "seqlen_q")
        return opt_int(&s->seqlen_q);
    if(n == "seqlen_kv")
        return opt_int(&s->seqlen_kv);
    if(n == "num_query_heads")
        return opt_int(&s->num_query_heads);
    if(n == "num_kv_heads")
        return opt_int(&s->num_kv_heads);
    if(n == "head_size")
        return opt_int(&s->head_size);
    if(n == "batch")
        return opt_int(&s->batch);
    if(n == "causal")
        return flag(&s->causal);
    if(n == "causal_bottom_right")
        return flag(&s->causal_bottom_right);
    if(n == "dtype")
        return str(&s->dtype);
    if(n == "layout")
        return str(&s->layout);
    if(n == "sliding_window")
        return opt_int(&s->sliding_window);
    if(n == "window_left")
        return opt_int(&s->window_left);
    if(n == "window_right")
        return opt_int(&s->window_right);
    if(n == "ragged")
        return flag(&s->ragged);
    if(n == "dkv_mode")
        return str(&s->dkv_mode);
    if(n == "dq_mode")
        return str(&s->dq_mode);
    if(n == "epilogue")
        return str(&s->epilogue);
    if(n == "oob")
        return str(&s->oob);
    if(n == "edge_tiles")
        return flag(&s->edge_tiles);
    if(n == "ws_layout")
        return str(&s->ws_layout);
    if(n == "persistent")
        return flag(&s->persistent);
    if(n == "persist_decode")
        return str(&s->persist_decode);
    if(n == "xcd_remap")
        return flag(&s->xcd_remap);
    if(n == "name")
        return str(&s->name);
    if(n == "body_variant")
        return str(&s->body_variant);
    return false;
}

static bool aux_set_field(rocke_attn_bwd_aux_spec_t* s, const std::string& n, const std::string& v)
{
    if(n == "head_size")
        s->head_size = atoi(v.c_str());
    else if(n == "dtype")
        s->dtype = keep(v);
    else if(n == "seq_mode")
        s->seq_mode = keep(v);
    else if(n == "stage_vec")
        s->stage_vec = atoi(v.c_str());
    else if(n == "ws_layout")
        s->ws_layout = keep(v);
    else if(n == "kv_target")
        s->kv_target = keep(v);
    else if(n == "rows_per_cta")
    {
        s->has_rows_per_cta = v != "None";
        s->rows_per_cta = v == "None" ? 0 : atoi(v.c_str());
    }
    else if(n == "convert_vec")
        s->convert_vec = atoi(v.c_str());
    else if(n == "worklist")
        s->worklist = v == "True";
    else if(n == "narrow_vec")
        s->narrow_vec = atoi(v.c_str());
    else if(n == "name")
        s->name = keep(v);
    else
        return false;
    return true;
}

/* Build config idx into b: 0 built, 1 build error (message in *msg), 2 bad
 * config. *arch receives the target. */
static int build_config(int idx,
                        rocke_ir_builder_t* b,
                        rocke_kernel_def_t** kernel,
                        std::string* arch,
                        std::string* msg)
{
    std::string c(CONFIGS[idx]);
    if(c.rfind("aux:", 0) == 0)
    {
        std::string body = c.substr(4);
        size_t bar1 = body.find('|');
        size_t bar2 = body.find('|', bar1 + 1);
        if(bar1 == std::string::npos || bar2 == std::string::npos)
            return 2;
        std::string stage = body.substr(0, bar1);
        *arch = body.substr(bar1 + 1, bar2 - bar1 - 1);
        rocke_attn_bwd_aux_spec_t s = rocke_attn_bwd_aux_spec_default();
        for(const auto& kv : dense_items(body.substr(bar2 + 1)))
            if(!aux_set_field(&s, kv.first, kv.second))
                return 2;
        char reason[512] = "";
        if(!rocke_attn_bwd_aux_spec_validate(&s, reason, sizeof(reason)))
        {
            *msg = reason;
            return 1;
        }
        *kernel = rocke_build_attn_bwd_aux_new(b, keep(stage), &s, keep(*arch));
        return 0;
    }
    size_t hash = c.find('#');
    std::string main_text = c.substr(0, hash);
    std::string knobs = hash == std::string::npos ? std::string() : c.substr(hash + 1);
    size_t bar = main_text.find('|');
    *arch = main_text.substr(0, bar);
    const char* cls = (*arch == "gfx942" || *arch == "gfx950") ? keep(*arch) : NULL;
    std::vector<std::pair<std::string, std::string>> items = dense_items(main_text.substr(bar + 1));
    for(const auto& kv : items)
        if(kv.first == "cls")
            cls = kv.second == "base" ? NULL : keep(kv.second);
    rocke_attn_bwd_dense_spec_t s = rocke_attn_bwd_dense_spec_default(cls);
    for(const auto& kv : items)
        if(kv.first != "cls" && !dense_set_field(&s, kv.first, kv.second))
            return 2;
    std::string karch;
    if(!parse_config(keep(*arch + "|" + knobs), &karch, &s.body_knobs))
        return 2;
    /* the knob names as passed (Python sees a repeated or default-valued one) */
    static std::vector<const char*> names;
    names.clear();
    for(size_t at = 0; at < knobs.size();)
    {
        size_t end = knobs.find(';', at);
        if(end == std::string::npos)
            end = knobs.size();
        std::string item = knobs.substr(at, end - at);
        if(!item.empty())
            names.push_back(keep(item.substr(0, item.find(':'))));
        at = end + 1;
    }
    s.body_knob_names = names.empty() ? NULL : names.data();
    s.n_body_knob_names = (int)names.size();
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_dense_spec_init(&s, &e) != ROCKE_OK)
    {
        *msg = e.msg;
        return 1;
    }
    *kernel = rocke_build_attn_dense_bwd_main_new(b, &s, keep(*arch));
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
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
    std::string arch, msg;
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof(b));
    rocke_kernel_def_t* kernel = NULL;
    int rc = build_config(idx, &b, &kernel, &arch, &msg);
    if(rc == 2)
    {
        fprintf(stderr, "bad config %d\n", idx);
        return 2;
    }
    if(rc != 0)
    {
        fprintf(stderr, "build failed for config %d: %s\n", idx, msg.c_str());
        return 1;
    }
    if(kernel == NULL || !rocke_ir_builder_ok(&b))
    {
        const char* err = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed for config %d: %s\n", idx, err ? err : "");
        rocke_ir_builder_free(&b);
        return 1;
    }

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
