/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_unified_main_emit.c -- C-side emitter for the
 * whole unified (paged) attention backward main kernel. Builds, identically
 * to the Python emitter attention_bwd_unified_main_emit.py, the kernel of
 * build_attn_bwd_unified_main(spec, arch=) through
 * rocke_build_attn_bwd_unified_main_new (rocke/instance_attention_bwd_unified.h),
 * lowers it via rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared. The general knob parser
 * comes from attention_bwd_spec_emit.c (included with
 * ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS and ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY);
 * the unified fields (head_size, dtype, page_mode, mask_class, sub_page,
 * stage_vec, name) are parsed here. The table holds every Python config
 * resolved to "<arch>|<field>:<value>;..." (the gate cases planned);
 * regenerate it with `attention_bwd_unified_main_emit.py --print-c-configs`.
 *
 * argv[1] is the config index. Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <string>

#include "rocke/instance_attention_bwd_unified.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static const char* const CONFIGS[] = {
    /* case unified/gfx942/d32_p64_sub64_direct */
    "gfx942|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx942/d32_p64_sub64_atomic */
    "gfx942|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx942/d64_p64_sub64_direct */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case unified/gfx942/d64_p64_sub64_atomic */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case unified/gfx950/d32_p64_sub64_direct */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx950/d32_p64_sub64_atomic */
    "gfx950|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx950/d64_p64_sub64_direct */
    "gfx950|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case unified/gfx950/d64_p64_sub64_atomic */
    "gfx950|head_size:64;sub_page:64;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case unified/gfx950/d128_p64_sub64_direct */
    "gfx950|head_size:128;page_mode:direct;sub_page:64;waves:8;block_m:16;block_n:128;block_k4:32;"
    "warp_grid_g4:1x8;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case unified/gfx950/d128_p64_sub64_atomic */
    "gfx950|head_size:128;sub_page:64;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case unified/gfx950/d64_p128_sub128_direct */
    "gfx950|head_size:64;page_mode:direct;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case unified/gfx950/d64_p128_sub128_atomic */
    "gfx950|head_size:64;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case unified/gfx942/d128_p64_k64_direct */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx942/d128_p64_k64_atomic */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx942/d128_p128_k64_direct */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx942/d128_p128_k64_atomic */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx942/d128_p256_k64_direct */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx942/d128_p256_k64_atomic */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx950/d128_p128_k64_direct */
    "gfx950|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    /* case unified/gfx950/d128_p128_k64_atomic */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1",
    /* atomic multi gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic",
    /* atomic no mask gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;mask_class:none",
    /* atomic fp16 gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;dtype:fp16",
    /* atomic stage_vec 1 gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;stage_vec:1",
    /* atomic sub 16 gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;sub_page:16",
    /* atomic sub 32 gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;sub_page:32",
    /* atomic runtime scale gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;static_scale:runtime",
    /* atomic dq_before_atomic gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;static_scale:dq_before_atomic",
    /* atomic convert gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;static_scale:convert",
    /* atomic head pack gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;head_pack:True",
    /* atomic edge tiles gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;edge_tiles:True",
    /* atomic head_outer grouped gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;step_order:head_outer;edge_tiles:True",
    /* atomic head_outer gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;step_order:head_outer",
    /* atomic q_rotate gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;q_rotate:True",
    /* atomic interior_clamp gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;edge_tiles:True;interior_clamp:True",
    /* atomic token major gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;ws_layout:token_major",
    /* atomic stats_vec 4 gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;stats_vec:4;stats_fma:True",
    /* atomic counted gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;loop_sync:counted",
    /* atomic lds_only gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;loop_sync:lds_only",
    /* atomic barrier merge gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;ring_depth:2;barrier_merge:on",
    /* atomic acc_in_lds gfx942 d64 d32 */
    "gfx942|head_size:32;page_mode:atomic;acc_in_lds:True",
    /* atomic kv in LDS gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;kv_residency:lds;kt_source:lds",
    /* atomic sched iglp0 gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;sched:iglp0",
    /* atomic sched fences_only gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;sched:fences_only",
    /* atomic sched stage_table gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;sched:stage_table",
    /* atomic sched stage_table_atomics gfx942 d64 */
    "gfx942|head_size:64;page_mode:atomic;sched:stage_table_atomics",
    /* atomic multi gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic",
    /* atomic no mask gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;mask_class:none",
    /* atomic fp16 gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;dtype:fp16",
    /* atomic stage_vec 1 gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;stage_vec:1",
    /* atomic sub 16 gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;sub_page:16",
    /* atomic sub 32 gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;sub_page:32",
    /* atomic runtime scale gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;static_scale:runtime",
    /* atomic dq_before_atomic gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;static_scale:dq_before_atomic",
    /* atomic convert gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;static_scale:convert",
    /* atomic head pack gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;head_pack:True",
    /* atomic edge tiles gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;edge_tiles:True",
    /* atomic head_outer grouped gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;step_order:head_outer;edge_tiles:True",
    /* atomic head_outer gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;step_order:head_outer",
    /* atomic q_rotate gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;q_rotate:True",
    /* atomic interior_clamp gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;edge_tiles:True;interior_clamp:True",
    /* atomic token major gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;ws_layout:token_major",
    /* atomic stats_vec 4 gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;stats_vec:4;stats_fma:True",
    /* atomic counted gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;loop_sync:counted",
    /* atomic lds_only gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;loop_sync:lds_only",
    /* atomic barrier merge gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;ring_depth:2;barrier_merge:on",
    /* atomic acc_in_lds gfx950 d128 d32 */
    "gfx950|head_size:32;page_mode:atomic;acc_in_lds:True",
    /* atomic kv in LDS gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;kv_residency:lds;kt_source:lds",
    /* atomic sched iglp0 gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;sched:iglp0",
    /* atomic sched fences_only gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;sched:fences_only",
    /* atomic sched stage_table gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;sched:stage_table",
    /* atomic sched stage_table_atomics gfx950 d128 */
    "gfx950|head_size:128;page_mode:atomic;sched:stage_table_atomics",
    /* direct multi gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct",
    /* direct no mask gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;mask_class:none",
    /* direct fp16 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dtype:fp16",
    /* direct stage_vec 1 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;stage_vec:1",
    /* direct sub 16 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;sub_page:16",
    /* direct sub 32 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;sub_page:32",
    /* direct runtime scale gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;static_scale:runtime",
    /* direct dq_before_atomic gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;static_scale:dq_before_atomic",
    /* direct convert gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;static_scale:convert",
    /* direct head pack gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;head_pack:True",
    /* direct edge tiles gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;edge_tiles:True",
    /* direct head_outer grouped gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;step_order:head_outer;edge_tiles:True",
    /* direct head_outer gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;step_order:head_outer",
    /* direct q_rotate gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;q_rotate:True",
    /* direct interior_clamp gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;edge_tiles:True;interior_clamp:True",
    /* direct token major gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;ws_layout:token_major",
    /* direct stats_vec 4 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;stats_vec:4;stats_fma:True",
    /* direct counted gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;loop_sync:counted",
    /* direct lds_only gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;loop_sync:lds_only",
    /* direct barrier merge gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;ring_depth:2;barrier_merge:on",
    /* direct acc_in_lds gfx942 d64 d32 */
    "gfx942|head_size:32;page_mode:direct;acc_in_lds:True",
    /* direct kv in LDS gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;kv_residency:lds;kt_source:lds",
    /* direct sched iglp0 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;sched:iglp0",
    /* direct sched fences_only gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;sched:fences_only",
    /* direct sched stage_table gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;sched:stage_table",
    /* direct sched stage_table_atomics gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;sched:stage_table_atomics",
    /* direct multi gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct",
    /* direct no mask gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;mask_class:none",
    /* direct fp16 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dtype:fp16",
    /* direct stage_vec 1 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;stage_vec:1",
    /* direct sub 16 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;sub_page:16",
    /* direct sub 32 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;sub_page:32",
    /* direct runtime scale gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;static_scale:runtime",
    /* direct dq_before_atomic gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;static_scale:dq_before_atomic",
    /* direct convert gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;static_scale:convert",
    /* direct head pack gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;head_pack:True",
    /* direct edge tiles gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;edge_tiles:True",
    /* direct head_outer grouped gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;step_order:head_outer;edge_tiles:True",
    /* direct head_outer gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;step_order:head_outer",
    /* direct q_rotate gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;q_rotate:True",
    /* direct interior_clamp gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;edge_tiles:True;interior_clamp:True",
    /* direct token major gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;ws_layout:token_major",
    /* direct stats_vec 4 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;stats_vec:4;stats_fma:True",
    /* direct counted gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;loop_sync:counted",
    /* direct lds_only gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;loop_sync:lds_only",
    /* direct barrier merge gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;ring_depth:2;barrier_merge:on",
    /* direct acc_in_lds gfx950 d128 d32 */
    "gfx950|head_size:32;page_mode:direct;acc_in_lds:True",
    /* direct kv in LDS gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;kv_residency:lds;kt_source:lds",
    /* direct sched iglp0 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;sched:iglp0",
    /* direct sched fences_only gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;sched:fences_only",
    /* direct sched stage_table gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;sched:stage_table",
    /* direct sched stage_table_atomics gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;sched:stage_table_atomics",
    /* atomic select gfx942 */
    "gfx942|head_size:64;dkv_atomic_guard:select",
    /* atomic select gfx950 sub */
    "gfx950|head_size:128;dkv_atomic_guard:select;sub_page:32",
    /* atomic select runtime gfx942 */
    "gfx942|head_size:64;dkv_atomic_guard:select;static_scale:runtime",
    /* direct direct gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:direct",
    /* direct direct sub 16 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:direct;sub_page:16",
    /* direct direct runtime gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:direct;static_scale:runtime",
    /* direct direct gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:direct",
    /* direct direct sub 16 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:direct;sub_page:16",
    /* direct direct runtime gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:direct;static_scale:runtime",
    /* direct lds_staged gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:lds_staged",
    /* direct lds_staged sub 16 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:lds_staged;sub_page:16",
    /* direct lds_staged runtime gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:lds_staged;static_scale:runtime",
    /* direct lds_staged gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged",
    /* direct lds_staged sub 16 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged;sub_page:16",
    /* direct lds_staged runtime gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged;static_scale:runtime",
    /* direct lds_staged_pair gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:lds_staged_pair",
    /* direct lds_staged_pair sub 16 gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:lds_staged_pair;sub_page:16",
    /* direct lds_staged_pair runtime gfx942 d64 */
    "gfx942|head_size:64;page_mode:direct;dkv_epilogue:lds_staged_pair;static_scale:runtime",
    /* direct lds_staged_pair gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged_pair",
    /* direct lds_staged_pair sub 16 gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged_pair;sub_page:16",
    /* direct lds_staged_pair runtime gfx950 d128 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged_pair;static_scale:runtime",
    /* direct lds_staged vec 1 gfx942 */
    "gfx942|head_size:64;page_mode:direct;dkv_store_vec:1",
    /* direct pair vec 1 gfx950 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged_pair;dkv_store_vec:1",
    /* direct lds_staged vec 2 gfx942 */
    "gfx942|head_size:64;page_mode:direct;dkv_store_vec:2",
    /* direct pair vec 2 gfx950 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged_pair;dkv_store_vec:2",
    /* direct lds_staged vec 4 gfx942 */
    "gfx942|head_size:64;page_mode:direct;dkv_store_vec:4",
    /* direct pair vec 4 gfx950 */
    "gfx950|head_size:128;page_mode:direct;dkv_epilogue:lds_staged_pair;dkv_store_vec:4",
    /* direct tail round gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:32;block_n:32;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "2x2;warp_grid_g13:2x2;pt_route:lds;page_mode:direct",
    /* direct pair tail round gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:32;block_n:32;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "2x2;warp_grid_g13:2x2;pt_route:lds;page_mode:direct;dkv_epilogue:lds_staged_pair",
    /* atomic tail round gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:32;block_n:32;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "2x2;warp_grid_g13:2x2;pt_route:lds",
    /* direct tail round sub 16 gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:32;block_n:32;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "2x2;warp_grid_g13:2x2;pt_route:lds;page_mode:direct;sub_page:16",
    /* rdna gfx1201 d64 atomic */
    "gfx1201|head_size:64",
    /* rdna gfx1201 d64 direct */
    "gfx1201|head_size:64;page_mode:direct",
    /* rdna gfx1151 d64 atomic no mask */
    "gfx1151|head_size:64;mask_class:none",
    /* rdna gfx1151 d64 direct fp16 */
    "gfx1151|head_size:64;page_mode:direct;dtype:fp16",
    /* unified/gfx942/d64_p64_sub64_atomic no mask */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;mask_class:none",
    /* unified/gfx942/d64_p64_sub64_atomic head pack */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;head_pack:True",
    /* unified/gfx942/d64_p64_sub64_atomic edge tiles */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;edge_tiles:True",
    /* unified/gfx942/d64_p64_sub64_atomic counted */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;loop_sync:counted",
    /* unified/gfx942/d64_p64_sub64_atomic stage_table_atomics */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;sched:stage_table_atomics",
    /* unified/gfx942/d64_p64_sub64_atomic runtime scale */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;static_scale:runtime",
    /* unified/gfx942/d64_p64_sub64_atomic fp16 */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;dtype:fp16",
    /* unified/gfx942/d64_p64_sub64_direct no mask */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;mask_class:none",
    /* unified/gfx942/d64_p64_sub64_direct head pack */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;head_pack:True",
    /* unified/gfx942/d64_p64_sub64_direct edge tiles */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;edge_tiles:True",
    /* unified/gfx942/d64_p64_sub64_direct counted */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;loop_sync:counted",
    /* unified/gfx942/d64_p64_sub64_direct stage_table_atomics */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;sched:stage_table_atomics",
    /* unified/gfx942/d64_p64_sub64_direct runtime scale */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;static_scale:runtime",
    /* unified/gfx942/d64_p64_sub64_direct fp16 */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;dtype:fp16",
    /* unified/gfx942/d128_p64_k64_atomic no mask */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;mask_class:none",
    /* unified/gfx942/d128_p64_k64_atomic head pack */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;head_pack:True",
    /* unified/gfx942/d128_p64_k64_atomic edge tiles */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;edge_tiles:True",
    /* unified/gfx942/d128_p64_k64_atomic counted */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;loop_sync:counted",
    /* unified/gfx942/d128_p64_k64_atomic stage_table_atomics */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;sched:stage_table_atomics",
    /* unified/gfx942/d128_p64_k64_atomic runtime scale */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;static_scale:runtime",
    /* unified/gfx942/d128_p64_k64_atomic fp16 */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;dtype:fp16",
    /* unified/gfx942/d128_p64_k64_direct no mask */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;mask_class:none",
    /* unified/gfx942/d128_p64_k64_direct head pack */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;head_pack:True",
    /* unified/gfx942/d128_p64_k64_direct edge tiles */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;edge_tiles:True",
    /* unified/gfx942/d128_p64_k64_direct counted */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;loop_sync:counted",
    /* unified/gfx942/d128_p64_k64_direct stage_table_atomics */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;sched:stage_table_atomics",
    /* unified/gfx942/d128_p64_k64_direct runtime scale */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;static_scale:runtime",
    /* unified/gfx942/d128_p64_k64_direct fp16 */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;dtype:fp16",
    /* unified/gfx950/d32_p64_sub64_atomic no mask */
    "gfx950|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;mask_class:none",
    /* unified/gfx950/d32_p64_sub64_atomic head pack */
    "gfx950|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;head_pack:True",
    /* unified/gfx950/d32_p64_sub64_atomic edge tiles */
    "gfx950|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;edge_tiles:True",
    /* unified/gfx950/d32_p64_sub64_atomic counted */
    "gfx950|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;loop_sync:counted",
    /* unified/gfx950/d32_p64_sub64_atomic stage_table_atomics */
    "gfx950|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;sched:stage_table_atomics",
    /* unified/gfx950/d32_p64_sub64_atomic runtime scale */
    "gfx950|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;static_scale:runtime",
    /* unified/gfx950/d32_p64_sub64_atomic fp16 */
    "gfx950|head_size:32;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;dtype:fp16",
    /* unified/gfx950/d32_p64_sub64_direct no mask */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;mask_class:none",
    /* unified/gfx950/d32_p64_sub64_direct head pack */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;head_pack:True",
    /* unified/gfx950/d32_p64_sub64_direct edge tiles */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;edge_tiles:True",
    /* unified/gfx950/d32_p64_sub64_direct counted */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;loop_sync:counted",
    /* unified/gfx950/d32_p64_sub64_direct stage_table_atomics */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;sched:stage_table_atomics",
    /* unified/gfx950/d32_p64_sub64_direct runtime scale */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;static_scale:runtime",
    /* unified/gfx950/d32_p64_sub64_direct fp16 */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;dtype:fp16",
    /* unified/gfx950/d128_p128_k64_atomic no mask */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;mask_class:none",
    /* unified/gfx950/d128_p128_k64_atomic head pack */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;head_pack:True",
    /* unified/gfx950/d128_p128_k64_atomic edge tiles */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;edge_tiles:True",
    /* unified/gfx950/d128_p128_k64_atomic counted */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;loop_sync:counted",
    /* unified/gfx950/d128_p128_k64_atomic stage_table_atomics */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;sched:stage_table_atomics",
    /* unified/gfx950/d128_p128_k64_atomic runtime scale */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;static_scale:runtime",
    /* unified/gfx950/d128_p128_k64_atomic fp16 */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x4;warp_grid_g13:4x1;dtype:fp16",
    /* unified/gfx950/d128_p128_k64_direct no mask */
    "gfx950|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;mask_class:none",
    /* unified/gfx950/d128_p128_k64_direct head pack */
    "gfx950|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;head_pack:True",
    /* unified/gfx950/d128_p128_k64_direct edge tiles */
    "gfx950|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;edge_tiles:True",
    /* unified/gfx950/d128_p128_k64_direct counted */
    "gfx950|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;loop_sync:counted",
    /* unified/gfx950/d128_p128_k64_direct stage_table_atomics */
    "gfx950|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;sched:stage_table_atomics",
    /* unified/gfx950/d128_p128_k64_direct runtime scale */
    "gfx950|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;static_scale:runtime",
    /* unified/gfx950/d128_p128_k64_direct fp16 */
    "gfx950|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;dtype:fp16",
    /* unified/gfx950/d64_p128_sub128_atomic no mask */
    "gfx950|head_size:64;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;mask_class:none",
    /* unified/gfx950/d64_p128_sub128_atomic head pack */
    "gfx950|head_size:64;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;head_pack:True",
    /* unified/gfx950/d64_p128_sub128_atomic edge tiles */
    "gfx950|head_size:64;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;edge_tiles:True",
    /* unified/gfx950/d64_p128_sub128_atomic counted */
    "gfx950|head_size:64;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;loop_sync:counted",
    /* unified/gfx950/d64_p128_sub128_atomic stage_table_atomics */
    "gfx950|head_size:64;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;sched:stage_table_atomics",
    /* unified/gfx950/d64_p128_sub128_atomic runtime scale */
    "gfx950|head_size:64;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;static_scale:runtime",
    /* unified/gfx950/d64_p128_sub128_atomic fp16 */
    "gfx950|head_size:64;sub_page:128;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;dtype:fp16",
    /* multi gfx942 atomic select */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;dkv_atomic_guard:select",
    /* multi gfx942 direct pair */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;dkv_epilogue:lds_staged_pair",
    /* multi gfx942 direct cells */
    "gfx942|head_size:128;page_mode:direct;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:"
    "1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x4;warp_grid_g13:4x1;dkv_epilogue:direct",
    /* sub gfx950 direct pair */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;dkv_epilogue:lds_staged_pair",
    /* sub gfx950 direct cells */
    "gfx950|head_size:32;page_mode:direct;sub_page:64;waves:4;block_m:32;block_n:128;block_k4:64;"
    "warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;dkv_epilogue:direct",
    /* sub gfx942 sub 32 */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;sub_page:32",
    /* sub gfx942 direct sub 16 */
    "gfx942|head_size:64;page_mode:direct;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;sub_page:16",
    /* name latin-1 letter */
    "gfx942|head_size:64;name:attn_é",
    /* name latin-1 digit */
    "gfx950|head_size:128;page_mode:direct;name:k²_½",
    /* agpr pair agpr_fence wpe 2 max-ilp gfx942 */
    "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:agpr_fence;waves_per_eu:2;scheduler_"
    "strategy:max-ilp",
    /* agpr pair vgpr_form max-memory-clause gfx950 direct */
    "gfx950|head_size:128;page_mode:direct;agpr_alloc:128x128;agpr_route:vgpr_form;scheduler_"
    "strategy:max-memory-clause",
    /* agpr asm_g13 iterative-minreg gfx942 direct */
    "gfx942|head_size:64;page_mode:direct;agpr_alloc:192x192;agpr_route:asm_g13;scheduler_strategy:"
    "iterative-minreg",
    /* agpr backend gfx942 */
    "gfx942|head_size:64;agpr_alloc:backend",
    /* agpr pair on shipped multi gfx942 */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x4;warp_grid_g13:4x1;agpr_alloc:128x128;waves_per_eu:2",
    /* rejected: last_cta sink */
    "gfx942|head_size:64;dkv_sink:last_cta",
    /* rejected: empty name */
    "gfx942|head_size:64;name:",
    /* rejected: name with dash */
    "gfx942|head_size:64;name:bad-name",
    /* rejected: name of underscores */
    "gfx942|head_size:64;name:__",
    /* rejected: name with multiply sign */
    "gfx942|head_size:64;name:a×b",
    /* rejected: wide name with dash */
    "gfx942|head_size:64;name:attn_Ω-x",
    /* rejected: scheduler_strategy */
    "gfx942|head_size:64;scheduler_strategy:bogus",
    /* rejected: xcd_remap */
    "gfx942|head_size:64;xcd_remap:True",
    /* rejected: lb_mirror */
    "gfx942|head_size:64;lb_mirror:True",
    /* rejected: ring without merge */
    "gfx942|head_size:64;ring_depth:2",
    /* rejected: dma path */
    "gfx950|head_size:128;global_path:dma;ring_depth:2",
    /* rejected: sub_page 0 */
    "gfx942|head_size:64;sub_page:0",
    /* rejected: sub_page 24 */
    "gfx942|head_size:64;sub_page:24",
    /* rejected: sub_page = block_n */
    "gfx942|head_size:64;sub_page:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;sub_page:128",
    /* rejected: slices sink */
    "gfx942|head_size:64;dkv_sink:slices",
    /* rejected: ws_bounds buffer */
    "gfx942|head_size:64;ws_bounds:buffer",
    /* rejected: narrow_vec 2 */
    "gfx942|head_size:64;stage_vec:1;narrow_vec:2",
    /* rejected: page_mode */
    "gfx942|head_size:64;page_mode:scatter",
    /* rejected: head_size */
    "gfx942|head_size:48",
    /* rejected: unknown arch */
    "gfx1100|head_size:64",
};

#define ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS 1
#define ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY 1
#include "attention_bwd_spec_emit.c"

/* "<arch>|<field>:<value>;..." into a unified spec: the unified fields here,
 * every other field a knob (set_field of attention_bwd_spec_emit.c). */
static bool parse_unified(const char* config, std::string* arch, rocke_attn_bwd_unified_spec_t* u)
{
    std::string c(config);
    size_t bar = c.find('|');
    *arch = c.substr(0, bar);
    *u = rocke_attn_bwd_unified_spec_default();
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
            std::string name = item.substr(0, colon);
            std::string text = item.substr(colon + 1);
            bool none = text == "None";
            if(name == "head_size")
                u->head_size = atoi(text.c_str());
            else if(name == "dtype")
                u->dtype = none ? NULL : keep(text);
            else if(name == "page_mode")
                u->page_mode = none ? NULL : keep(text);
            else if(name == "mask_class")
                u->mask_class = none ? NULL : keep(text);
            else if(name == "sub_page")
                u->sub_page = none ? ROCKE_ATTN_BWD_NONE : atoi(text.c_str());
            else if(name == "stage_vec")
                u->stage_vec = atoi(text.c_str());
            else if(name == "name")
                u->name = none ? NULL : keep(text);
            else if(!set_field(&u->knobs, name, text))
                return false;
        }
        if(semi == std::string::npos)
            break;
        start = semi + 1;
    }
    return true;
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
    std::string arch;
    rocke_attn_bwd_unified_spec_t spec;
    if(!parse_unified(CONFIGS[idx], &arch, &spec))
    {
        fprintf(stderr, "bad config %d\n", idx);
        return 2;
    }
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_unified_spec_init(&spec, &e) != ROCKE_OK)
    {
        fprintf(stderr, "build failed for config %d: %s\n", idx, e.msg);
        return 1;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_attn_bwd_unified_main_new(&b, &spec, arch.c_str());
    if(kernel == NULL || !rocke_ir_builder_ok(&b))
    {
        const char* msg = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed for config %d: %s\n", idx, msg ? msg : "");
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
