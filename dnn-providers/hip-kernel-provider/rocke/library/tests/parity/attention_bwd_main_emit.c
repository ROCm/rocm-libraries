/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_main_emit.c -- C-side emitter for the whole
 * layout-general attention backward main kernel. Builds, identically to the
 * Python emitter attention_bwd_main_emit.py, the kernel of
 * _emit_main(spec, arch, main_kernel_refusals=False) through
 * rocke_build_attn_bwd_main_new (rocke/instance_attention_bwd.h), lowers it
 * via rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout
 * so the two outputs can be byte-compared. The spec parser comes from
 * attention_bwd_spec_emit.c (included with ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS
 * and ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY). The table holds every Python
 * config resolved to "<arch>|<field>:<value>;..." (the gate cases planned);
 * regenerate it with `attention_bwd_main_emit.py --print-c-configs`.
 *
 * argv[1] is the config index. Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <string>

#include "rocke/instance_attention_bwd.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static const char* const CONFIGS[] = {
    /* case general/gfx942/tuned0 */
    "gfx942|head_size:128;dtype:bf16;waves:2;block_m:16;block_n:32;block_k4:32;warp_grid_g4:1x2;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x2;warp_grid_g13:2x1;stats_fma:True;stats_vec:4",
    /* case general/gfx942/default0 */
    "gfx942|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/tuned1 */
    "gfx942|head_size:128;dtype:bf16;waves:4;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:192x192;s_in_agpr:"
    "False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;lb_mirror:True;stats_fma:True;"
    "stats_vec:4",
    /* case general/gfx942/default1 */
    "gfx942|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/tuned2 */
    "gfx942|head_size:128;dtype:bf16;waves:4;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:192x192;s_in_agpr:"
    "False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;lb_mirror:True;stats_fma:True;"
    "stats_vec:4",
    /* case general/gfx942/default2 */
    "gfx942|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/tuned3 */
    "gfx942|head_size:128;dtype:bf16;mask_class:none;waves:4;block_m:16;block_n:128;block_k4:32;"
    "warp_grid_g4:1x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:192x192;s_in_agpr:False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;stats_"
    "fma:True;stats_vec:4",
    /* case general/gfx942/default3 */
    "gfx942|head_size:128;dtype:bf16;mask_class:none;waves:8;block_m:16;block_n:128;block_k4:32;"
    "warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/tuned4 */
    "gfx942|head_size:128;waves:4;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:192x192;s_in_agpr:False;"
    "agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;lb_mirror:True;stats_fma:True;stats_"
    "vec:4",
    /* case general/gfx942/default4 */
    "gfx942|head_size:128;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_"
    "eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_"
    "grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/tuned5 */
    "gfx942|head_size:64;dtype:bf16;waves:2;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x2;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x2;warp_grid_g13:2x1;lb_mirror:True;stats_fma:True;stats_vec:4",
    /* case general/gfx942/default5 */
    "gfx942|head_size:64;dtype:bf16;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/tuned6 */
    "gfx942|head_size:64;dtype:bf16;waves:2;block_m:16;block_n:64;block_k4:32;warp_grid_g4:1x2;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x2;warp_grid_g13:2x1;lb_mirror:True;stats_fma:True;stats_vec:4",
    /* case general/gfx942/default6 */
    "gfx942|head_size:64;dtype:bf16;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/tuned7 */
    "gfx942|head_size:64;dtype:bf16;seq_mode:thd;mask_class:none;waves:2;block_m:16;block_n:64;"
    "block_k4:32;warp_grid_g4:1x2;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:"
    "relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:"
    "vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:"
    "False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x2;warp_grid_g13:2x1;stats_fma:True;stats_"
    "vec:4",
    /* case general/gfx942/default7 */
    "gfx942|head_size:64;dtype:bf16;seq_mode:thd;mask_class:none;waves:8;block_m:32;block_n:128;"
    "block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:"
    "relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:"
    "vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:"
    "False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/tuned8 */
    "gfx942|head_size:64;seq_mode:thd;mask_class:none;waves:2;block_m:16;block_n:64;block_k4:32;"
    "warp_grid_g4:1x2;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x2;warp_grid_g13:2x1;stats_fma:True;stats_vec:4",
    /* case general/gfx942/default8 */
    "gfx942|head_size:64;seq_mode:thd;mask_class:none;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:"
    "none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned0 */
    "gfx950|head_size:128;dtype:bf16;waves:4;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:"
    "False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default0 */
    "gfx950|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned1 */
    "gfx950|head_size:128;dtype:bf16;waves:4;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:2;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:"
    "False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;step_order:head_outer;stats_fma:"
    "True;stats_vec:4",
    /* case general/gfx950/default1 */
    "gfx950|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned2 */
    "gfx950|head_size:128;dtype:bf16;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default2 */
    "gfx950|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned3 */
    "gfx950|head_size:128;dtype:bf16;mask_class:none;waves:4;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:"
    "128x128;s_in_agpr:False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;stats_fma:True;"
    "stats_vec:4",
    /* case general/gfx950/default3 */
    "gfx950|head_size:128;dtype:bf16;mask_class:none;waves:8;block_m:16;block_n:128;block_k4:32;"
    "warp_grid_g4:1x8;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned4 */
    "gfx950|head_size:128;dtype:bf16;mask_class:none;waves:4;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:"
    "128x128;s_in_agpr:False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;stats_fma:True;"
    "stats_vec:4",
    /* case general/gfx950/default4 */
    "gfx950|head_size:128;dtype:bf16;mask_class:none;waves:8;block_m:16;block_n:128;block_k4:32;"
    "warp_grid_g4:1x8;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned5 */
    "gfx950|head_size:128;dtype:bf16;mask_class:none;waves:4;block_m:16;block_n:192;block_k4:32;"
    "warp_grid_g4:1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:"
    "192x192;s_in_agpr:False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;stats_fma:True;"
    "stats_vec:4",
    /* case general/gfx950/default5 */
    "gfx950|head_size:128;dtype:bf16;mask_class:none;waves:8;block_m:16;block_n:128;block_k4:32;"
    "warp_grid_g4:1x8;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned6 */
    "gfx950|head_size:128;waves:4;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:tr_read;ring_depth:2;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:False;agpr_"
    "route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;lb_mirror:True;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default6 */
    "gfx950|head_size:128;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned7 */
    "gfx950|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;setprio:False;waves_per_eu:1;"
    "edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x8;warp_grid_g13:8x1;lb_interleave:True;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default7 */
    "gfx950|head_size:128;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned8 */
    "gfx950|head_size:128;waves:4;block_m:16;block_n:192;block_k4:32;warp_grid_g4:1x4;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;"
    "transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;setprio:False;waves_per_eu:1;"
    "edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:192x192;s_in_agpr:False;agpr_"
    "route:asm_g13;scheduler_strategy:iterative-ilp;warp_grid_g02:1x4;warp_grid_g13:4x1;stats_fma:"
    "True;stats_vec:4",
    /* case general/gfx950/default8 */
    "gfx950|head_size:128;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:"
    "16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;"
    "transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:"
    "1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_"
    "g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned9 */
    "gfx950|head_size:64;dtype:bf16;static_scale:convert;waves:8;block_m:32;block_n:256;block_k4:"
    "32;warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:vgpr;lds_"
    "swizzle:ds=pad32;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:True;acc_"
    "in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;lds_order:"
    "phase_low;lb_interleave:True;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default9 */
    "gfx950|head_size:64;dtype:bf16;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned10 */
    "gfx950|head_size:64;dtype:bf16;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;lb_mirror:True;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default10 */
    "gfx950|head_size:64;dtype:bf16;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;dq_mode:split;waves:4;block_m:32;block_n:128;"
    "block_k4:32;warp_grid_g4:1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:"
    "relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;"
    "sched:none;setprio:False;waves_per_eu:2;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;bwd_exp2:fast;stats_fma:True;"
    "stats_vec:4",
    /* case general/gfx950/default11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned12 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default12 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned13 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default13 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned14 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;dq_mode:split;waves:4;block_m:32;block_n:128;"
    "block_k4:32;warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:"
    "relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;"
    "sched:none;setprio:False;waves_per_eu:2;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_"
    "alloc:0x0;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;bwd_exp2:fast;stats_fma:True;"
    "stats_vec:4",
    /* case general/gfx950/default14 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned15 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default15 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned16 */
    "gfx950|head_size:64;dtype:bf16;seq_mode:thd;mask_class:none;waves:4;block_m:16;block_n:192;"
    "block_k4:32;warp_grid_g4:1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:"
    "relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;"
    "lds_swizzle:ds=pad32;sched:none;setprio:False;waves_per_eu:2;edge_tiles:False;head_pack:False;"
    "acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;bwd_exp2:"
    "fast;stats_fma:True;stats_vec:4;worklist:True",
    /* case general/gfx950/default16 */
    "gfx950|head_size:64;dtype:bf16;seq_mode:thd;mask_class:none;waves:8;block_m:32;block_n:256;"
    "block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:"
    "relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;"
    "sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;"
    "agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned17 */
    "gfx950|head_size:64;mask_class:none;dq_mode:split;waves:8;block_m:32;block_n:128;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;bwd_exp2:fast;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default17 */
    "gfx950|head_size:64;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:"
    "2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned18 */
    "gfx950|head_size:64;mask_class:none;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:"
    "2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default18 */
    "gfx950|head_size:64;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:"
    "2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx950/tuned19 */
    "gfx950|head_size:64;seq_mode:thd;mask_class:none;waves:4;block_m:16;block_n:192;block_k4:32;"
    "warp_grid_g4:1x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;"
    "setprio:False;waves_per_eu:2;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;bwd_exp2:fast;stats_fma:True;stats_vec:4",
    /* case general/gfx950/default19 */
    "gfx950|head_size:64;seq_mode:thd;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;"
    "warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_"
    "residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;"
    "setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;"
    "s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    /* case general/gfx942/ws_bounds=buffer */
    "gfx942|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;ws_bounds:buffer",
    /* case general/gfx950/ws_bounds=buffer */
    "gfx950|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;ws_bounds:buffer",
    /* xcd_remap direct gfx942 d64 */
    "gfx942|head_size:64;xcd_remap:True",
    /* xcd_remap atomic gfx950 d128 */
    "gfx950|head_size:128;xcd_remap:True;dkv_mode:atomic",
    /* xcd_remap mirror interleave gfx942 */
    "gfx942|head_size:64;xcd_remap:True;lb_mirror:True;lb_interleave:True",
    /* lb_interleave gfx950 */
    "gfx950|head_size:128;lb_interleave:True",
    /* lb_mirror gfx942 d64 */
    "gfx942|head_size:64;lb_mirror:True",
    /* lb_mirror atomic thd gfx950 */
    "gfx950|head_size:128;lb_mirror:True;dkv_mode:atomic;seq_mode:thd",
    /* worklist thd gfx942 */
    "gfx942|head_size:64;seq_mode:thd;worklist:True;dkv_mode:atomic",
    /* rejected: worklist with mirror */
    "gfx942|head_size:64;seq_mode:thd;worklist:True;lb_mirror:True",
    /* atomic branch gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic",
    /* atomic select gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_atomic_guard:select",
    /* atomic token-major gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;ws_layout:token_major",
    /* atomic mul-free gfx942 runtime scale */
    "gfx942|head_size:64;dkv_mode:atomic;static_scale:runtime",
    /* slices lds_staged gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:slices",
    /* slices lds_staged buffer gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;ws_bounds:buffer",
    /* slices pair gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:lds_staged_pair",
    /* slices pair buffer gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:lds_staged_pair;ws_bounds:"
    "buffer",
    /* slices direct gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct",
    /* slices direct buffer gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;ws_bounds:buffer",
    /* slices direct quad gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;dkv_cell_vec:4",
    /* slices direct quad buffer gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;dkv_cell_vec:4;ws_"
    "bounds:buffer",
    /* slices acc_in_lds gfx942 d32 */
    "gfx942|head_size:32;dkv_mode:atomic;dkv_sink:slices;acc_in_lds:True",
    /* slices acc_in_lds buffer gfx950 d32 */
    "gfx950|head_size:32;dkv_mode:atomic;dkv_sink:slices;acc_in_lds:True;ws_bounds:buffer",
    /* slices vec 4 runtime scale gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_store_vec:4;static_scale:runtime",
    /* slices token-major gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;ws_layout:token_major",
    /* rejected: slices buffer token-major */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;ws_layout:token_major;ws_bounds:buffer",
    /* rejected: last_cta sink (not built) */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:last_cta",
    /* direct cells gfx942 */
    "gfx942|head_size:64;dkv_epilogue:direct",
    /* direct quads gfx950 */
    "gfx950|head_size:128;dkv_epilogue:direct;dkv_cell_vec:4",
    /* direct pair gfx942 */
    "gfx942|head_size:64;dkv_epilogue:lds_staged_pair",
    /* direct vec 2 gfx950 */
    "gfx950|head_size:128;dkv_store_vec:2",
    /* direct acc_in_lds gfx942 d32 */
    "gfx942|head_size:32;acc_in_lds:True",
    /* dkv_slab_hoist thd gfx950 */
    "gfx950|head_size:128;seq_mode:thd;dkv_slab_hoist:True",
    /* dkv_slab_hoist thd quads gfx942 */
    "gfx942|head_size:64;seq_mode:thd;dkv_slab_hoist:True;dkv_epilogue:direct;dkv_cell_vec:4",
    /* edge_tiles gfx942 */
    "gfx942|head_size:64;edge_tiles:True",
    /* head_outer gfx942 */
    "gfx942|head_size:64;step_order:head_outer",
    /* head_outer edge tiles (grouped) gfx950 */
    "gfx950|head_size:128;step_order:head_outer;edge_tiles:True",
    /* head_outer grouped head pack gfx942 */
    "gfx942|head_size:64;step_order:head_outer;edge_tiles:True;head_pack:True",
    /* head_outer grouped deferred dQ gfx950 */
    "gfx950|head_size:128;step_order:head_outer;edge_tiles:True;sched:stage_table_atomics",
    /* rejected: head_outer grouped ring 2 */
    "gfx942|head_size:64;step_order:head_outer;edge_tiles:True;ring_depth:2",
    /* head_outer grouped rotate deferred head pack gfx950 */
    "gfx950|head_size:128;step_order:head_outer;edge_tiles:True;q_rotate:True;sched:stage_table_"
    "atomics;head_pack:True",
    /* q_rotate gfx942 */
    "gfx942|head_size:64;q_rotate:True",
    /* q_rotate head_outer gfx950 */
    "gfx950|head_size:128;q_rotate:True;step_order:head_outer",
    /* q_rotate grouped gfx942 */
    "gfx942|head_size:64;q_rotate:True;step_order:head_outer;edge_tiles:True",
    /* q_rotate ring 3 gfx950 */
    "gfx950|head_size:128;q_rotate:True;ring_depth:3",
    /* slab_carry gfx942 */
    "gfx942|head_size:64;slab_carry:True",
    /* slab_carry edge tiles gfx950 */
    "gfx950|head_size:128;slab_carry:True;edge_tiles:True",
    /* slab_carry head_outer gfx942 */
    "gfx942|head_size:64;slab_carry:True;step_order:head_outer",
    /* slab_carry deferred dQ gfx950 */
    "gfx950|head_size:128;slab_carry:True;sched:stage_table_atomics",
    /* slab_carry grouped (not carried) gfx942 */
    "gfx942|head_size:64;slab_carry:True;step_order:head_outer;edge_tiles:True",
    /* rejected: slab_carry with q_rotate */
    "gfx942|head_size:64;slab_carry:True;q_rotate:True",
    /* interior_clamp edge tiles gfx950 */
    "gfx950|head_size:128;interior_clamp:True;edge_tiles:True",
    /* head_pack gfx942 d64 */
    "gfx942|head_size:64;head_pack:True",
    /* head_pack edge tiles gfx950 */
    "gfx950|head_size:128;head_pack:True;edge_tiles:True",
    /* head_pack deferred dQ gfx942 */
    "gfx942|head_size:64;head_pack:True;sched:stage_table_atomics",
    /* stage_table_atomics gfx950 */
    "gfx950|head_size:128;sched:stage_table_atomics",
    /* stage_table_atomics edge tiles gfx942 */
    "gfx942|head_size:64;sched:stage_table_atomics;edge_tiles:True",
    /* stage_table setprio gfx950 */
    "gfx950|head_size:128;sched:stage_table;setprio:True",
    /* iglp1 gfx942 */
    "gfx942|head_size:64;sched:iglp1",
    /* counted gfx942 */
    "gfx942|head_size:64;loop_sync:counted",
    /* counted late edge tiles gfx950 */
    "gfx950|head_size:128;loop_sync:counted;counted_issue:late;edge_tiles:True",
    /* counted deferred dQ gfx942 */
    "gfx942|head_size:64;loop_sync:counted;sched:stage_table_atomics",
    /* counted head pack gfx950 */
    "gfx950|head_size:128;loop_sync:counted;head_pack:True",
    /* counted head_outer gfx942 */
    "gfx942|head_size:64;loop_sync:counted;step_order:head_outer",
    /* counted rotate gfx950 */
    "gfx950|head_size:128;loop_sync:counted;q_rotate:True",
    /* loop_sync lds_only gfx942 */
    "gfx942|head_size:64;loop_sync:lds_only",
    /* barrier merge gfx942 */
    "gfx942|head_size:64;barrier_merge:on;ring_depth:2",
    /* barrier merge late edge tiles gfx950 */
    "gfx950|head_size:128;barrier_merge:on;ring_depth:2;ring_issue:late;edge_tiles:True",
    /* barrier merge head_outer gfx942 */
    "gfx942|head_size:64;barrier_merge:on;ring_depth:2;step_order:head_outer",
    /* barrier merge head pack gfx950 */
    "gfx950|head_size:128;barrier_merge:on;ring_depth:2;head_pack:True",
    /* barrier merge deferred gfx942 */
    "gfx942|head_size:64;barrier_merge:on;ring_depth:2;sched:stage_table_atomics",
    /* ring vgpr 2 gfx942 */
    "gfx942|head_size:64;ring_depth:2",
    /* ring vgpr 3 gfx950 */
    "gfx950|head_size:128;ring_depth:3",
    /* ring vgpr 2 edge tiles gfx942 */
    "gfx942|head_size:64;ring_depth:2;edge_tiles:True",
    /* ring vgpr 3 head_outer gfx950 */
    "gfx950|head_size:128;ring_depth:3;step_order:head_outer",
    /* ring vgpr 2 head pack deferred gfx942 */
    "gfx942|head_size:64;ring_depth:2;head_pack:True;sched:stage_table_atomics",
    /* ring dma 1 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:1",
    /* ring dma 2 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2",
    /* ring dma 3 edge tiles gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:3;edge_tiles:True",
    /* ring dma 2 head_outer rotate gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;step_order:head_outer;q_rotate:True",
    /* ring dma 3 head_outer split dQ gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:3;step_order:head_outer;dq_mode:split",
    /* ring dma 2 mirror atomic gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;lb_mirror:True;dkv_mode:atomic",
    /* ptr_noalias workspace gfx942 */
    "gfx942|head_size:64;ptr_noalias:workspace",
    /* ptr_noalias all atomic gfx950 */
    "gfx950|head_size:128;ptr_noalias:all;dkv_mode:atomic",
    /* scheduler_strategy max-ilp waves_per_eu 2 gfx942 */
    "gfx942|head_size:64;scheduler_strategy:max-ilp;waves_per_eu:2",
    /* vgpr_form gfx942 d128 */
    "gfx942|head_size:128;agpr_alloc:192x192;agpr_route:vgpr_form",
    /* agpr_fence gfx950 d128 */
    "gfx950|head_size:128;agpr_alloc:128x128;agpr_route:agpr_fence",
    /* agpr_alloc backend gfx942 */
    "gfx942|head_size:64;agpr_alloc:backend",
    /* no mask gfx950 */
    "gfx950|head_size:128;mask_class:none;static_scale:runtime",
    /* split dQ gfx942 */
    "gfx942|head_size:64;dq_mode:split",
    /* thd gfx942 */
    "gfx942|head_size:64;seq_mode:thd",
    /* stats_vec 4 gfx950 */
    "gfx950|head_size:128;stats_vec:4;stats_fma:True",
    /* ws_bounds buffer gfx942 */
    "gfx942|head_size:64;ws_bounds:buffer",
    /* rdna gfx1201 d64 */
    "gfx1201|head_size:64",
    /* rdna gfx1151 d64 bf16 atomic */
    "gfx1151|head_size:64;dtype:bf16;dkv_mode:atomic",
    /* rejected: unknown arch */
    "gfx1100|head_size:64",
    /* slices direct runtime scale gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;static_scale:runtime",
    /* slices direct runtime scale buffer gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;static_scale:runtime;"
    "ws_bounds:buffer",
    /* slices acc_in_lds runtime scale gfx942 d32 */
    "gfx942|head_size:32;dkv_mode:atomic;dkv_sink:slices;acc_in_lds:True;static_scale:runtime",
    /* slices buffer vec 1 gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;ws_bounds:buffer;dkv_store_vec:1",
    /* tail rounds direct gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:64;block_n:64;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "1x4;warp_grid_g13:4x1;dkv_store_vec:8",
    /* tail rounds slices gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:64;block_n:64;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "1x4;warp_grid_g13:4x1;dkv_mode:atomic;dkv_sink:slices",
    /* tail rounds slices buffer gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:64;block_n:64;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "1x4;warp_grid_g13:4x1;dkv_mode:atomic;dkv_sink:slices;ws_bounds:buffer",
    /* ring dma 2 d32 gfx950 */
    "gfx950|head_size:32;global_path:dma;ring_depth:2",
    /* ring dma 2 one wave gfx950 */
    "gfx950|head_size:32;global_path:dma;ring_depth:2;waves:1;block_m:16;block_n:16;warp_grid_g4:"
    "1x1;warp_grid_g02:1x1;warp_grid_g13:1x1;block_k4:16;atom_g4:16x16x16;atom_g02:16x16x16",
    /* ring vgpr 2 d32 joint gfx942 */
    "gfx942|head_size:32;ring_depth:2",
    /* ring vgpr 3 d32 joint gfx950 */
    "gfx950|head_size:32;ring_depth:3",
    /* ring vgpr 2 d32 head pack gfx942 */
    "gfx942|head_size:32;ring_depth:2;head_pack:True",
    /* sched iglp0 dma 2 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;sched:iglp0",
    /* sched iglp0 vgpr 2 gfx942 */
    "gfx942|head_size:64;ring_depth:2;sched:iglp0",
    /* sched iglp0 counted gfx942 */
    "gfx942|head_size:64;loop_sync:counted;sched:iglp0",
    /* sched fences_only dma 2 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;sched:fences_only",
    /* sched fences_only vgpr 2 gfx942 */
    "gfx942|head_size:64;ring_depth:2;sched:fences_only",
    /* sched fences_only counted gfx942 */
    "gfx942|head_size:64;loop_sync:counted;sched:fences_only",
    /* sched stage_table dma 2 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;sched:stage_table",
    /* sched stage_table vgpr 2 gfx942 */
    "gfx942|head_size:64;ring_depth:2;sched:stage_table",
    /* sched stage_table counted gfx942 */
    "gfx942|head_size:64;loop_sync:counted;sched:stage_table",
    /* sched stage_table_atomics dma 2 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;sched:stage_table_atomics",
    /* sched stage_table_atomics vgpr 2 gfx942 */
    "gfx942|head_size:64;ring_depth:2;sched:stage_table_atomics",
    /* sched stage_table_atomics counted gfx942 */
    "gfx942|head_size:64;loop_sync:counted;sched:stage_table_atomics",
    /* stage_table setprio dma 3 gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:3;sched:stage_table;setprio:True",
    /* stage_table_atomics dma 3 edge tiles gfx950 */
    "gfx950|head_size:128;global_path:dma;ring_depth:3;edge_tiles:True;sched:stage_table_atomics",
    /* tuned gfx942 0 mirror */
    "gfx942|head_size:128;dtype:bf16;waves:2;block_m:16;block_n:32;block_k4:32;warp_grid_g4:1x2;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x2;warp_grid_g13:2x1;stats_fma:True;stats_vec:4;lb_mirror:True",
    /* tuned gfx942 0 atomic slices */
    "gfx942|head_size:128;dtype:bf16;waves:2;block_m:16;block_n:32;block_k4:32;warp_grid_g4:1x2;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x2;warp_grid_g13:2x1;stats_fma:True;stats_vec:4;dkv_mode:atomic;dkv_sink:"
    "slices",
    /* default gfx942 0 head_outer grouped */
    "gfx942|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;step_order:head_outer;edge_tiles:True",
    /* tuned gfx950 0 edge tiles */
    "gfx950|head_size:128;dtype:bf16;waves:4;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x4;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:"
    "False;agpr_route:asm_g13;warp_grid_g02:1x4;warp_grid_g13:4x1;stats_fma:True;stats_vec:4;edge_"
    "tiles:True",
    /* default gfx950 0 atomic xcd */
    "gfx950|head_size:128;dtype:bf16;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;"
    "atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_"
    "source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;"
    "waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:"
    "False;warp_grid_g02:1x8;warp_grid_g13:8x1;dkv_mode:atomic;xcd_remap:True",
    /* ring vgpr 3 head pack gfx950 */
    "gfx950|head_size:128;ring_depth:3;head_pack:True",
    /* ring vgpr 3 rotate head pack gfx950 */
    "gfx950|head_size:128;ring_depth:3;q_rotate:True;head_pack:True",
    /* ring vgpr 3 block_m 32 gfx950 */
    "gfx950|head_size:128;ring_depth:3;block_m:32",
    /* ring vgpr 2 d128 block_m 32 gfx942 */
    "gfx942|head_size:128;ring_depth:2;block_m:32",
    /* direct tail round gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:32;block_n:32;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "2x2;warp_grid_g13:2x2;pt_route:lds",
    /* direct pair tail round gfx942 d32 */
    "gfx942|head_size:32;waves:4;block_m:32;block_n:32;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "2x2;warp_grid_g13:2x2;pt_route:lds;dkv_epilogue:lds_staged_pair",
    /* direct tail round gfx950 d64 */
    "gfx950|head_size:64;waves:4;block_m:32;block_n:32;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:"
    "2x2;warp_grid_g13:2x2;pt_route:lds",
    /* slices vec 2 clamp gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_store_vec:2",
    /* slices vec 2 buffer gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;dkv_store_vec:2;ws_bounds:buffer",
    /* slices pair buffer vec 1 gfx950 */
    "gfx950|head_size:128;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:lds_staged_pair;ws_bounds:"
    "buffer;dkv_store_vec:1",
    /* slices pair buffer vec 2 gfx942 */
    "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:lds_staged_pair;ws_bounds:"
    "buffer;dkv_store_vec:2",
    /* stage_vec 1 gfx942 d64 */
    "gfx942|head_size:64;stage_vec:1",
    /* narrow 2 gfx942 d64 */
    "gfx942|head_size:64;stage_vec:1;narrow_vec:2",
    /* narrow 4 bf16 gfx942 d64 */
    "gfx942|head_size:64;stage_vec:1;narrow_vec:4;dtype:bf16",
    /* narrow 4 gfx950 d128 */
    "gfx950|head_size:128;stage_vec:1;narrow_vec:4",
    /* narrow 2 bf16 gfx950 d128 */
    "gfx950|head_size:128;stage_vec:1;narrow_vec:2;dtype:bf16",
    /* narrow 4 atomic gfx950 d32 */
    "gfx950|head_size:32;stage_vec:1;narrow_vec:4;dkv_mode:atomic",
    /* narrow 4 direct quads gfx942 */
    "gfx942|head_size:64;stage_vec:1;narrow_vec:4;dkv_epilogue:direct;dkv_cell_vec:4",
    /* narrow 4 direct cells gfx942 */
    "gfx942|head_size:64;stage_vec:1;narrow_vec:4;dkv_epilogue:direct",
    /* narrow 2 vec 1 gfx942 */
    "gfx942|head_size:64;stage_vec:1;narrow_vec:2;dkv_store_vec:1",
    /* narrow 4 thd ring 2 gfx950 d64 */
    "gfx950|head_size:64;stage_vec:1;narrow_vec:4;seq_mode:thd;ring_depth:2",
    /* narrow 4 head pack gfx950 d64 */
    "gfx950|head_size:64;stage_vec:1;narrow_vec:4;head_pack:True",
    /* lb_mirror thd gfx942 d64 */
    "gfx942|head_size:64;lb_mirror:True;seq_mode:thd",
    /* agpr_alloc 0x0 gfx942 */
    "gfx942|head_size:64;agpr_alloc:0x0",
    /* agpr_alloc 128x128 gfx950 */
    "gfx950|head_size:128;agpr_alloc:128x128",
    /* asm_g13 gfx950 d128 */
    "gfx950|head_size:128;agpr_alloc:128x128;agpr_route:asm_g13",
    /* rejected: split agpr_route on agpr_alloc backend */
    "gfx942|head_size:128;agpr_alloc:backend;agpr_route:vgpr_form",
    /* rejected: head_pack on a DMA ring */
    "gfx950|head_size:128;global_path:dma;ring_depth:2;head_pack:True",
};

#define ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS 1
#define ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY 1
#include "attention_bwd_spec_emit.c"

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
    rocke_attn_bwd_spec_t spec;
    if(!parse_config(CONFIGS[idx], &arch, &spec))
    {
        fprintf(stderr, "bad config %d\n", idx);
        return 2;
    }
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_spec_init(&spec, &e) != ROCKE_OK)
    {
        fprintf(stderr, "build failed for config %d: %s\n", idx, e.msg);
        return 1;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_attn_bwd_main_new(&b, &spec, arch.c_str());
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
