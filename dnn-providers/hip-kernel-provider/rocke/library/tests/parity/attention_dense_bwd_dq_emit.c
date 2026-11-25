/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_dense_bwd_dq_emit.c -- C-side emitter for the split
 * dQ kernel of the dense attention backward. Builds, identically to the
 * Python emitter attention_dense_bwd_dq_emit.py, the kernel of
 * build_attention_dense_bwd_dq(spec, arch=arch) through
 * rocke_build_attn_dense_bwd_dq_new (rocke/instance_attention_dense_bwd_dq.h),
 * lowers it via rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared. The dQ field parser is
 * attention_bwd_dq_emit.c's. The table holds every Python config resolved
 * (the gate cases planned); regenerate it with
 * `attention_dense_bwd_dq_emit.py --print-c-configs`.
 *
 * argv[1] is the config index. Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <deque>
#include <string>

#include "rocke/instance_attention_bwd_dq.h"
#include "rocke/instance_attention_dense_bwd_dq.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static const char* const CONFIGS[] = {
    /* case dense/gfx950/tuned0 */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True#waves:4;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:16x16x16;transpose_"
    "source:tr_read;head_loop:group;edge_tiles:True;kv_pad:16;agpr_alloc:0x0",
    /* case dense/gfx950/tuned1 */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False#waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:16x16x16;transpose_source:tr_"
    "read;edge_tiles:True;kv_pad:32;agpr_alloc:0x0",
    /* default gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64",
    /* no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False",
    /* causal br gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal_bottom_"
    "right:True",
    /* window 64 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;sliding_"
    "window:64",
    /* band 32x16 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False;"
    "window_left:32;window_right:16",
    /* band right only gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False;"
    "window_right:0",
    /* ragged gfx942 */
    "gfx942|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:64",
    /* ragged no mask gfx942 */
    "gfx942|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False",
    /* ragged clamp gfx942 */
    "gfx942|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:64;oob:clamp",
    /* bhsd gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;layout:bhsd",
    /* fp16 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;dtype:fp16",
    /* batch 2 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;batch:2",
    /* clamp gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;oob:clamp",
    /* token-major gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;ws_layout:"
    "token_major",
    /* name stem gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;name:my_dense",
    /* edge tiles gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#edge_tiles:"
    "True",
    /* edge tiles no mask gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False#"
    "edge_tiles:True",
    /* edge tiles ragged gfx942 */
    "gfx942|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:64#edge_tiles:"
    "True",
    /* edge tiles ragged no mask gfx942 */
    "gfx942|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False#"
    "edge_tiles:True",
    /* edge tiles window gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;sliding_"
    "window:64#edge_tiles:True",
    /* group gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#head_loop:"
    "group",
    /* group ragged gfx942 */
    "gfx942|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:64#head_loop:"
    "group",
    /* group clamp gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;oob:clamp#"
    "head_loop:group",
    /* group edge gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#head_loop:"
    "group;edge_tiles:True",
    /* group xcd gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#head_loop:"
    "group;xcd_remap:True",
    /* xcd gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#xcd_remap:"
    "True",
    /* prefetch gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#kv_prefetch:"
    "True",
    /* prefetch clamp gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;oob:clamp#kv_"
    "prefetch:True",
    /* prefetch edge ragged gfx942 */
    "gfx942|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:64#kv_prefetch:"
    "True;edge_tiles:True",
    /* stats_prescale off gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#stats_"
    "prescale:False",
    /* runtime scale gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#static_scale:"
    "runtime",
    /* convert scale gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#static_scale:"
    "convert",
    /* kv_pad 16 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#kv_pad:16",
    /* block_n 32 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#block_n:32",
    /* sched iglp1 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#sched:iglp1",
    /* sched stage_table gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#sched:stage_"
    "table",
    /* waves_per_eu 2 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#waves_per_eu:"
    "2",
    /* agpr_alloc 0x0 gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#agpr_alloc:"
    "0x0",
    /* scheduler_strategy gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#scheduler_"
    "strategy:max-ilp",
    /* rejected: group not dividing block_m gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;num_query_"
    "heads:6;num_kv_heads:2#head_loop:group;block_m:64",
    /* rejected: dq mask mismatch gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#mask_class:"
    "none",
    /* rejected: dq head size mismatch gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#head_size:32",
    /* rejected: thd dq gfx942 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64#seq_mode:thd",
    /* default gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128",
    /* no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False",
    /* causal br gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal_"
    "bottom_right:True",
    /* window 64 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;sliding_"
    "window:64",
    /* band 32x16 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False;"
    "window_left:32;window_right:16",
    /* band right only gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False;"
    "window_right:0",
    /* ragged gfx950 */
    "gfx950|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:128",
    /* ragged no mask gfx950 */
    "gfx950|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False",
    /* ragged clamp gfx950 */
    "gfx950|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:128;oob:clamp",
    /* bhsd gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;layout:bhsd",
    /* fp16 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;dtype:fp16",
    /* batch 2 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;batch:2",
    /* clamp gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;oob:clamp",
    /* token-major gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;ws_layout:"
    "token_major",
    /* name stem gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;name:my_"
    "dense",
    /* edge tiles gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#edge_tiles:"
    "True",
    /* edge tiles no mask gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False#"
    "edge_tiles:True",
    /* edge tiles ragged gfx950 */
    "gfx950|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:128#edge_tiles:"
    "True",
    /* edge tiles ragged no mask gfx950 */
    "gfx950|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:128;causal:False#"
    "edge_tiles:True",
    /* edge tiles window gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;sliding_"
    "window:64#edge_tiles:True",
    /* group gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#head_loop:"
    "group",
    /* group ragged gfx950 */
    "gfx950|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:128#head_loop:"
    "group",
    /* group clamp gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;oob:clamp#"
    "head_loop:group",
    /* group edge gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#head_loop:"
    "group;edge_tiles:True",
    /* group xcd gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#head_loop:"
    "group;xcd_remap:True",
    /* xcd gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#xcd_remap:"
    "True",
    /* prefetch gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#kv_prefetch:"
    "True",
    /* prefetch clamp gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;oob:clamp#kv_"
    "prefetch:True",
    /* prefetch edge ragged gfx950 */
    "gfx950|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:128#kv_prefetch:"
    "True;edge_tiles:True",
    /* stats_prescale off gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#stats_"
    "prescale:False",
    /* runtime scale gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#static_scale:"
    "runtime",
    /* convert scale gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#static_scale:"
    "convert",
    /* kv_pad 16 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#kv_pad:16",
    /* block_n 32 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#block_n:32",
    /* sched iglp1 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#sched:iglp1",
    /* sched stage_table gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#sched:stage_"
    "table",
    /* waves_per_eu 2 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#waves_per_eu:"
    "2",
    /* agpr_alloc 0x0 gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#agpr_alloc:"
    "0x0",
    /* scheduler_strategy gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#scheduler_"
    "strategy:max-ilp",
    /* rejected: group not dividing block_m gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;num_query_"
    "heads:6;num_kv_heads:2#head_loop:group;block_m:64",
    /* rejected: dq mask mismatch gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#mask_class:"
    "none",
    /* rejected: dq head size mismatch gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#head_size:32",
    /* rejected: thd dq gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#seq_mode:thd",
    /* lds_plain gfx950 */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128#transpose_"
    "source:lds_plain",
    /* rejected: unknown arch */
    "gfx1201|seqlen_q:256;seqlen_kv:256;num_query_heads:8;num_kv_heads:8;head_size:64",
    /* rejected: head size 32 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;head_size:32",
    /* rejected: window without causal */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;causal:False;"
    "sliding_window:8",
    /* rejected: slab past the 32-bit offset */
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128;seqlen_q:"
    "140000;num_query_heads:64;num_kv_heads:8",
    /* rejected: pinned batch past 2^31 */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;batch:200000",
    /* rejected: bad oob */
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64;oob:wrap",
    /* tuned0 single */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True#waves:4;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:16x16x16;transpose_"
    "source:tr_read;head_loop:group;edge_tiles:True;kv_pad:16;agpr_alloc:0x0;head_loop:single",
    /* tuned0 prefetch */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True#waves:4;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:16x16x16;transpose_"
    "source:tr_read;head_loop:group;edge_tiles:True;kv_pad:16;agpr_alloc:0x0;kv_prefetch:True",
    /* tuned0 xcd */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True#waves:4;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:16x16x16;transpose_"
    "source:tr_read;head_loop:group;edge_tiles:True;kv_pad:16;agpr_alloc:0x0;xcd_remap:True",
    /* tuned0 edge off */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True#waves:4;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:16x16x16;transpose_"
    "source:tr_read;head_loop:group;edge_tiles:True;kv_pad:16;agpr_alloc:0x0;edge_tiles:False",
    /* tuned0 clamp */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;oob:clamp#waves:4;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:16x16x16;"
    "transpose_source:tr_read;head_loop:group;edge_tiles:True;kv_pad:16;agpr_alloc:0x0",
    /* tuned0 ragged q */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;seqlen_q:1000#waves:4;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:"
    "16x16x16;transpose_source:tr_read;head_loop:group;edge_tiles:True;kv_pad:16;agpr_alloc:0x0",
    /* tuned0 window */
    "gfx950|seqlen_q:1024;seqlen_kv:4096;num_query_heads:32;num_kv_heads:8;head_size:128;causal_"
    "bottom_right:True;sliding_window:300#waves:4;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:"
    "16x16x16;transpose_source:tr_read;head_loop:group;edge_tiles:True;kv_pad:16;agpr_alloc:0x0",
    /* tuned1 group */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False#waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:16x16x16;transpose_source:tr_"
    "read;edge_tiles:True;kv_pad:32;agpr_alloc:0x0;head_loop:group",
    /* tuned1 causal */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;causal:True#waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:16x16x16;transpose_"
    "source:tr_read;edge_tiles:True;kv_pad:32;agpr_alloc:0x0;mask_class:band",
    /* tuned1 ragged */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;seqlen_q:2000;seqlen_kv:2001#waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:"
    "16x16x16;transpose_source:tr_read;edge_tiles:True;kv_pad:32;agpr_alloc:0x0",
    /* tuned1 batch */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;batch:4#waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:16x16x16;transpose_"
    "source:tr_read;edge_tiles:True;kv_pad:32;agpr_alloc:0x0",
    /* tuned1 bhsd */
    "gfx950|seqlen_q:2048;seqlen_kv:2048;num_query_heads:12;num_kv_heads:12;head_size:64;causal:"
    "False;layout:bhsd#waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:16x16x16;transpose_"
    "source:tr_read;edge_tiles:True;kv_pad:32;agpr_alloc:0x0",
};

static const int N_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

/* attention_bwd_dq_emit.c's dQ field parser */
static std::deque<std::string>& dq_strings()
{
    static std::deque<std::string> keep; /* push_back keeps references stable */
    return keep;
}

static const char* dq_keep(const std::string& v)
{
    dq_strings().push_back(v);
    return dq_strings().back().c_str();
}

static bool dq_set_field(rocke_attn_bwd_dq_spec_t* s, const std::string& n, const std::string& v)
{
    auto opt_int = [&](int* f) {
        *f = v == "None" ? ROCKE_ATTN_BWD_NONE : atoi(v.c_str());
        return true;
    };
    auto opt_str = [&](const char** f) {
        *f = v == "None" ? NULL : dq_keep(v);
        return true;
    };
    auto flag = [&](bool* f) {
        if(v != "True" && v != "False")
            return false;
        *f = v == "True";
        return true;
    };
    if(n == "head_size")
        return opt_int(&s->head_size);
    if(n == "dtype")
        return opt_str(&s->dtype);
    if(n == "seq_mode")
        return opt_str(&s->seq_mode);
    if(n == "stage_vec")
        return opt_int(&s->stage_vec);
    if(n == "mask_class")
        return opt_str(&s->mask_class);
    if(n == "ws_layout")
        return opt_str(&s->ws_layout);
    if(n == "static_scale")
        return opt_str(&s->static_scale);
    if(n == "waves")
        return opt_int(&s->waves);
    if(n == "block_m")
        return opt_int(&s->block_m);
    if(n == "block_n")
        return opt_int(&s->block_n);
    if(n == "atom_g02")
        return opt_str(&s->atom_g02);
    if(n == "atom_g4")
        return opt_str(&s->atom_g4);
    if(n == "transpose_source")
        return opt_str(&s->transpose_source);
    if(n == "head_loop")
        return opt_str(&s->head_loop);
    if(n == "stats_prescale")
        return flag(&s->stats_prescale);
    if(n == "edge_tiles")
        return flag(&s->edge_tiles);
    if(n == "kv_pad")
        return opt_int(&s->kv_pad);
    if(n == "kv_prefetch")
        return flag(&s->kv_prefetch);
    if(n == "waves_per_eu")
        return opt_int(&s->waves_per_eu);
    if(n == "agpr_alloc")
    {
        rocke_attn_bwd_agpr_alloc_t a;
        memset(&a, 0, sizeof(a));
        size_t x = v.find('x');
        if(v == "None")
            a.kind = ROCKE_ATTN_BWD_AGPR_NONE;
        else if(x != std::string::npos && x > 0 && isdigit((unsigned char)v[0]))
        {
            a.kind = ROCKE_ATTN_BWD_AGPR_PAIR;
            a.a = atoi(v.substr(0, x).c_str());
            a.b = atoi(v.substr(x + 1).c_str());
        }
        else
        {
            a.kind = ROCKE_ATTN_BWD_AGPR_STR;
            a.str = dq_keep(v);
        }
        s->agpr_alloc = a;
        return true;
    }
    if(n == "scheduler_strategy")
        return opt_str(&s->scheduler_strategy);
    if(n == "sched")
        return opt_str(&s->sched);
    if(n == "xcd_remap")
        return flag(&s->xcd_remap);
    if(n == "narrow_vec")
        return opt_int(&s->narrow_vec);
    if(n == "name")
        return opt_str(&s->name);
    return false;
}

/* "<field>:<value>;..." into *s (fields over its current values). */
static bool dq_parse_fields(const std::string& rest, rocke_attn_bwd_dq_spec_t* s)
{
    size_t start = 0;
    while(start < rest.size())
    {
        size_t semi = rest.find(';', start);
        std::string item
            = rest.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if(!item.empty())
        {
            size_t colon = item.find(':');
            if(!dq_set_field(s, item.substr(0, colon), item.substr(colon + 1)))
                return false;
        }
        if(semi == std::string::npos)
            break;
        start = semi + 1;
    }
    return true;
}

static bool dense_dq_set_field(rocke_attn_bwd_dense_dq_spec_t* s,
                               const std::string& n,
                               const std::string& v)
{
    auto opt_int = [&](int* f) {
        *f = v == "None" ? ROCKE_ATTN_BWD_NONE : atoi(v.c_str());
        return true;
    };
    auto str = [&](const char** f) {
        *f = v == "None" ? NULL : dq_keep(v);
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
    if(n == "oob")
        return str(&s->oob);
    if(n == "ws_layout")
        return str(&s->ws_layout);
    if(n == "name")
        return str(&s->name);
    return false;
}

/* The spec of a config: 0 built, 1 a spec error (*msg), 2 a bad config. */
static int dense_dq_config_spec(const char* config,
                                std::string* arch,
                                rocke_attn_bwd_dense_dq_spec_t* out,
                                std::string* msg)
{
    std::string c(config);
    size_t hash = c.find('#');
    std::string dense = c.substr(0, hash);
    size_t bar = dense.find('|');
    *arch = dense.substr(0, bar);
    *out = rocke_attn_bwd_dense_dq_spec_default();
    std::string rest = dense.substr(bar + 1);
    size_t start = 0;
    while(start < rest.size())
    {
        size_t semi = rest.find(';', start);
        std::string item
            = rest.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if(!item.empty())
        {
            size_t colon = item.find(':');
            if(!dense_dq_set_field(out, item.substr(0, colon), item.substr(colon + 1)))
                return 2;
        }
        if(semi == std::string::npos)
            break;
        start = semi + 1;
    }
    rocke_attn_bwd_err_t e;
    if(rocke_attn_bwd_dense_dq_spec_init(out, &e) != ROCKE_OK)
    {
        *msg = e.msg;
        return 1;
    }
    if(hash == std::string::npos)
        return 0;
    /* dataclasses.replace(spec.dq, **fields), then replace(spec, dq=dq) */
    rocke_attn_bwd_dq_spec_t dq = out->dq;
    if(!dq_parse_fields(c.substr(hash + 1), &dq))
        return 2;
    if(rocke_attn_bwd_dq_spec_init(&dq, &e) != ROCKE_OK)
    {
        *msg = e.msg;
        return 1;
    }
    out->dq = dq;
    out->has_dq = true;
    if(rocke_attn_bwd_dense_dq_spec_init(out, &e) != ROCKE_OK)
    {
        *msg = e.msg;
        return 1;
    }
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
    rocke_attn_bwd_dense_dq_spec_t spec;
    int rc = dense_dq_config_spec(CONFIGS[idx], &arch, &spec, &msg);
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

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_attn_dense_bwd_dq_new(&b, &spec, arch.c_str());
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
