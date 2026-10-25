/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_sched_ir_emit.c -- C-side emitter for the
 * scheduling hints of the attention backward stage tables. Builds, identically
 * to the Python emitter attention_bwd_sched_ir_emit.py, a probe kernel that
 * calls the emit_loop_head / emit_stage_begin / emit_stage_end hooks of one
 * table of helper_rocke.instances.common._attention_bwd_sched.h, lowers it via
 * rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout so the
 * two outputs can be byte-compared. The config parsing and the table builders
 * come from attention_bwd_sched_emit.c (included with
 * ROCKE_ATTN_BWD_SCHED_EMIT_TABLES_ONLY); regenerate the table with
 * `attention_bwd_sched_ir_emit.py --print-c-configs`.
 *
 * argv[1] is the config index. Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define PROBE_NAME "attn_bwd_sched_probe"

static const char* const CONFIGS[] = {
    /* sync none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:none",
    /* sync iglp0 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:iglp0",
    /* sync iglp1 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:iglp1",
    /* sync fences_only */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:fences_only",
    /* sync stage_table */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:stage_table",
    /* sync stage_table_atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:stage_table_atomics",
    /* sync setprio */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:stage_table;setprio:True",
    /* sync wide none */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atom_g02:32;atom_g4:32;dtype:bf16;"
    "sched:none",
    /* sync wide iglp0 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atom_g02:32;atom_g4:32;dtype:bf16;"
    "sched:iglp0",
    /* sync wide iglp1 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atom_g02:32;atom_g4:32;dtype:bf16;"
    "sched:iglp1",
    /* sync wide fences_only */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atom_g02:32;atom_g4:32;dtype:bf16;"
    "sched:fences_only",
    /* sync wide stage_table */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atom_g02:32;atom_g4:32;dtype:bf16;"
    "sched:stage_table",
    /* sync wide stage_table_atomics */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atom_g02:32;atom_g4:32;dtype:bf16;"
    "sched:stage_table_atomics",
    /* sync wide setprio */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atom_g02:32;atom_g4:32;dtype:bf16;"
    "sched:stage_table;setprio:True",
    /* sync split none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:none",
    /* sync split iglp0 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:iglp0",
    /* sync split iglp1 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:iglp1",
    /* sync split fences_only */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:fences_only",
    /* sync split stage_table */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:stage_table",
    /* sync split stage_table_atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:stage_table_"
    "atomics",
    /* sync split setprio */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:stage_table;"
    "setprio:True",
    /* sync lds none */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "acc_in_lds:True;sched:none",
    /* sync lds iglp0 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "acc_in_lds:True;sched:iglp0",
    /* sync lds iglp1 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "acc_in_lds:True;sched:iglp1",
    /* sync lds fences_only */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "acc_in_lds:True;sched:fences_only",
    /* sync lds stage_table */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "acc_in_lds:True;sched:stage_table",
    /* sync lds stage_table_atomics */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "acc_in_lds:True;sched:stage_table_atomics",
    /* sync lds setprio */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;kv_residency:lds;kt_source:lds;"
    "acc_in_lds:True;sched:stage_table;setprio:True",
    /* dma ring none */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:2;sched:none",
    /* dma ring iglp0 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:2;sched:iglp0",
    /* dma ring iglp1 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:2;sched:iglp1",
    /* dma ring fences_only */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:2;sched:fences_only",
    /* dma ring stage_table */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:2;sched:stage_table",
    /* dma ring stage_table_atomics */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:2;sched:stage_table_atomics",
    /* dma ring setprio */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;global_path:dma;pipeline:ring;"
    "ring_depth:2;sched:stage_table;setprio:True",
    /* register ring none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:3;pipeline:ring;sched:"
    "none",
    /* register ring iglp0 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:3;pipeline:ring;sched:"
    "iglp0",
    /* register ring iglp1 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:3;pipeline:ring;sched:"
    "iglp1",
    /* register ring fences_only */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:3;pipeline:ring;sched:"
    "fences_only",
    /* register ring stage_table */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:3;pipeline:ring;sched:"
    "stage_table",
    /* register ring stage_table_atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:3;pipeline:ring;sched:"
    "stage_table_atomics",
    /* register ring setprio */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;ring_depth:3;pipeline:ring;sched:"
    "stage_table;setprio:True",
    /* merged early none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:none",
    /* merged early iglp0 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:iglp0",
    /* merged early iglp1 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:iglp1",
    /* merged early fences_only */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:fences_only",
    /* merged early stage_table */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:stage_table",
    /* merged early stage_table_atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:stage_table_atomics",
    /* merged early setprio */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;sched:stage_table;setprio:True",
    /* merged late none */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;ring_issue:late;sched:none",
    /* merged late iglp0 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;ring_issue:late;sched:iglp0",
    /* merged late iglp1 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;ring_issue:late;sched:iglp1",
    /* merged late fences_only */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;ring_issue:late;sched:fences_only",
    /* merged late stage_table */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;ring_issue:late;sched:stage_table",
    /* merged late stage_table_atomics */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;ring_issue:late;sched:stage_table_atomics",
    /* merged late setprio */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;barrier_merge:on;ring_depth:2;"
    "pipeline:merged;ring_issue:late;sched:stage_table;setprio:True",
    /* counted early none */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;sched:none",
    /* counted early iglp0 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;sched:iglp0",
    /* counted early iglp1 */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;sched:iglp1",
    /* counted early fences_only */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;sched:fences_"
    "only",
    /* counted early stage_table */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;sched:stage_"
    "table",
    /* counted early stage_table_atomics */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;sched:stage_"
    "table_atomics",
    /* counted early setprio */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;loop_sync:counted;sched:stage_"
    "table;setprio:True",
    /* counted late none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:none",
    /* counted late iglp0 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:iglp0",
    /* counted late iglp1 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:iglp1",
    /* counted late fences_only */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:fences_only",
    /* counted late stage_table */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:stage_table",
    /* counted late stage_table_atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:stage_table_atomics",
    /* counted late setprio */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;loop_sync:counted;counted_issue:"
    "late;sched:stage_table;setprio:True",
    /* ck none */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;transpose_source:xt_lds;"
    "sched:none",
    /* ck iglp0 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;transpose_source:xt_lds;"
    "sched:iglp0",
    /* ck iglp1 */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;transpose_source:xt_lds;"
    "sched:iglp1",
    /* ck fences_only */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;transpose_source:xt_lds;"
    "sched:fences_only",
    /* ck stage_table */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;transpose_source:xt_lds;"
    "sched:stage_table",
    /* ck stage_table_atomics */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;transpose_source:xt_lds;"
    "sched:stage_table_atomics",
    /* ck setprio */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;pipeline:ck;transpose_source:xt_lds;"
    "sched:stage_table;setprio:True",
    /* rdna none */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:lds_plain;sched:"
    "none",
    /* rdna iglp0 */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:lds_plain;sched:"
    "iglp0",
    /* rdna iglp1 */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:lds_plain;sched:"
    "iglp1",
    /* rdna fences_only */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:lds_plain;sched:"
    "fences_only",
    /* rdna stage_table */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:lds_plain;sched:"
    "stage_table",
    /* rdna stage_table_atomics */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:lds_plain;sched:"
    "stage_table_atomics",
    /* rdna setprio */
    "table|gfx1201|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:lds_plain;sched:"
    "stage_table;setprio:True",
    /* dq gfx950 none */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:none",
    /* dq gfx950 iglp0 */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:iglp0",
    /* dq gfx950 iglp1 */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:iglp1",
    /* dq gfx950 fences_only */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:fences_only",
    /* dq gfx950 stage_table */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:stage_table",
    /* dq gfx950 stage_table_setprio */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:stage_table_setprio",
    /* dq gfx942 prefetch none */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;transpose_source:lds_plain;kv_prefetch:True;kv_"
    "pad:8;sched:none",
    /* dq gfx942 prefetch iglp0 */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;transpose_source:lds_plain;kv_prefetch:True;kv_"
    "pad:8;sched:iglp0",
    /* dq gfx942 prefetch iglp1 */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;transpose_source:lds_plain;kv_prefetch:True;kv_"
    "pad:8;sched:iglp1",
    /* dq gfx942 prefetch fences_only */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;transpose_source:lds_plain;kv_prefetch:True;kv_"
    "pad:8;sched:fences_only",
    /* dq gfx942 prefetch stage_table */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;transpose_source:lds_plain;kv_prefetch:True;kv_"
    "pad:8;sched:stage_table",
    /* dq gfx942 prefetch stage_table_setprio */
    "dq|gfx942|head_size:64;k_m0:64;k_n0:64;waves:4;transpose_source:lds_plain;kv_prefetch:True;kv_"
    "pad:8;sched:stage_table_setprio",
    /* ck last step */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;pipeline:ck;sched:stage_table_"
    "atomics;probe:last",
    /* sync two steps deferred */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:stage_table_atomics;probe:two",
    /* setprio two steps */
    "table|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;sched:stage_table;setprio:True;"
    "probe:two",
    /* unknown stage */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;probe:bad",
    /* unknown stage, split */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;dq_mode:split;sched:fences_only;"
    "probe:bad",
    /* unknown dQ stage */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:stage_table_setprio;probe:bad",
    /* unknown dQ stage at the end */
    "dq|gfx950|head_size:64;k_m0:64;k_n0:64;waves:4;sched:stage_table_setprio;probe:badend",
    /* table refused */
    "table|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;sched:bogus",
    /* dma batch refused in the groups */
    "table|gfx942|head_size:16;k_m0:16;k_n0:16;k_k4:16;waves:1;global_path:dma;pipeline:ring",
    /* geometry */
    "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atoms:32/16/"
    "32;sched:iglp1;global_path:dma;ring_depth:2",
    /* dq geometry */
    "dqgeom|gfx950|head_size:128;k_m0:128;k_n0:32;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;"
    "transpose_source:tr_read;sched:stage_table_setprio;kv_pad:16;kv_prefetch:True",
};

#define ROCKE_ATTN_BWD_SCHED_EMIT_CONFIGS 1
#define ROCKE_ATTN_BWD_SCHED_EMIT_TABLES_ONLY 1
#include "attention_bwd_sched_emit.c"

/* One step: begin / end around every stage. */
static bool step(rocke_ir_builder_t* b,
                 bool dq,
                 const Table& t,
                 const DqTable& d,
                 const std::string& probe,
                 bool prev_atomics)
{
    if(dq)
    {
        for(int i = 0; i < ROCKE_ATTN_BWD_N_DQ_STAGES; ++i)
        {
            const char* name = ROCKE_ATTN_BWD_DQ_STAGES[i];
            if(!rocke_attn_bwd_dq_stage_table_emit_stage_begin(b, &d, name)
               || !rocke_attn_bwd_dq_stage_table_emit_stage_end(b, &d, name))
                return false;
        }
        return true;
    }
    const char* const* names;
    int n = rocke_attn_bwd_stage_table_stage_names(&t, &names);
    for(int i = 0; i < n; ++i)
        if(!rocke_attn_bwd_stage_table_emit_stage_begin(b, &t, names[i])
           || !rocke_attn_bwd_stage_table_emit_stage_end(
               b, &t, names[i], probe != "last", prev_atomics))
            return false;
    return true;
}

/* Build the probe kernel of config `idx` into `b`. 0 on success. */
static int build(int idx, rocke_ir_builder_t* b)
{
    std::string kind, arch_s;
    Fields f;
    if(!parse_config(CONFIGS[idx], &kind, &arch_s, &f))
        return -1;
    std::string probe = "steps";
    if(has(f, "probe"))
    {
        probe = f.at("probe");
        f.erase("probe");
    }
    const char* arch = keep(arch_s);
    const bool dq = kind == "dq" || kind == "dqgeom";
    Table t;
    DqTable d;
    rocke_attn_bwd_err_t e;
    bool ok = dq ? make_dq(kind, arch, f, &d, &e) : make_table(kind, arch, f, &t, &e);
    if(!ok)
    {
        rocke_i_set_err_msg(b, e.code, e.msg);
        return 1;
    }
    if(!(dq ? rocke_attn_bwd_dq_stage_table_emit_loop_head(b, &d)
            : rocke_attn_bwd_stage_table_emit_loop_head(b, &t)))
        return 1;
    if(probe == "bad")
    {
        bool begun = dq ? rocke_attn_bwd_dq_stage_table_emit_stage_begin(b, &d, "X")
                        : rocke_attn_bwd_stage_table_emit_stage_begin(b, &t, "X");
        if(!begun)
            return 1;
        if(!(dq ? rocke_attn_bwd_dq_stage_table_emit_stage_end(b, &d, "X")
                : rocke_attn_bwd_stage_table_emit_stage_end(b, &t, "X", true, true)))
            return 1;
    }
    else if(probe == "badend")
    {
        if(!(dq ? rocke_attn_bwd_dq_stage_table_emit_stage_end(b, &d, "X")
                : rocke_attn_bwd_stage_table_emit_stage_end(b, &t, "X", true, true)))
            return 1;
    }
    else
    {
        if(!step(b, dq, t, d, probe, probe != "last"))
            return 1;
        if(probe == "two")
        {
            if(!(dq ? rocke_attn_bwd_dq_stage_table_emit_loop_head(b, &d)
                    : rocke_attn_bwd_stage_table_emit_loop_head(b, &t)))
                return 1;
            if(!step(b, dq, t, d, probe, false))
                return 1;
        }
    }
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
    std::string kind, arch;
    Fields f;
    parse_config(CONFIGS[idx], &kind, &arch, &f);

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
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
    return ret;
}
