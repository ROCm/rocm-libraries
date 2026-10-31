/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_dq_emit.c -- C-side emitter for the whole split
 * dQ kernel of the layout-general attention backward. Builds, identically to
 * the Python emitter attention_bwd_dq_emit.py, the kernel of
 * build_attn_bwd_dq(spec, arch=arch) through rocke_build_attn_bwd_dq_new
 * (rocke/instance_attention_bwd_dq.h), lowers it via
 * rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout so
 * the two outputs can be byte-compared. A "main=<arch>|<main fields>#<dq
 * knobs>" config derives the dQ spec through rocke_attn_bwd_spec_dq_spec
 * (AttnBwdSpec.dq_spec); its main-spec parser comes from
 * attention_bwd_spec_emit.c (included with ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS
 * and ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY). The table holds every Python
 * config resolved (the gate cases planned); regenerate it with
 * `attention_bwd_dq_emit.py --print-c-configs`.
 *
 * argv[1] is the config index. Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <ctype.h>
#include <string.h>

#include <deque>
#include <string>

#include "rocke/instance_attention_bwd.h"
#include "rocke/instance_attention_bwd_dq.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static const char* const CONFIGS[] = {
    /* case general/gfx950/tuned11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0",
    /* case general/gfx950/tuned14 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0",
    /* case general/gfx950/tuned17 */
    "gfx950|head_size:64;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:"
    "16x16x16;transpose_source:tr_read;agpr_alloc:0x0",
    /* sched none gfx942 */
    "gfx942|head_size:64;sched:none",
    /* sched none gfx950 */
    "gfx950|head_size:128;sched:none",
    /* sched none tuned11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;sched:none",
    /* sched none prefetch edge gfx950 d64 */
    "gfx950|head_size:64;sched:none;kv_prefetch:True;edge_tiles:True",
    /* sched iglp0 gfx942 */
    "gfx942|head_size:64;sched:iglp0",
    /* sched iglp0 gfx950 */
    "gfx950|head_size:128;sched:iglp0",
    /* sched iglp0 tuned11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;sched:iglp0",
    /* sched iglp0 prefetch edge gfx950 d64 */
    "gfx950|head_size:64;sched:iglp0;kv_prefetch:True;edge_tiles:True",
    /* sched iglp1 gfx942 */
    "gfx942|head_size:64;sched:iglp1",
    /* sched iglp1 gfx950 */
    "gfx950|head_size:128;sched:iglp1",
    /* sched iglp1 tuned11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;sched:iglp1",
    /* sched iglp1 prefetch edge gfx950 d64 */
    "gfx950|head_size:64;sched:iglp1;kv_prefetch:True;edge_tiles:True",
    /* sched fences_only gfx942 */
    "gfx942|head_size:64;sched:fences_only",
    /* sched fences_only gfx950 */
    "gfx950|head_size:128;sched:fences_only",
    /* sched fences_only tuned11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;sched:fences_only",
    /* sched fences_only prefetch edge gfx950 d64 */
    "gfx950|head_size:64;sched:fences_only;kv_prefetch:True;edge_tiles:True",
    /* sched stage_table gfx942 */
    "gfx942|head_size:64;sched:stage_table",
    /* sched stage_table gfx950 */
    "gfx950|head_size:128;sched:stage_table",
    /* sched stage_table tuned11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;sched:stage_table",
    /* sched stage_table prefetch edge gfx950 d64 */
    "gfx950|head_size:64;sched:stage_table;kv_prefetch:True;edge_tiles:True",
    /* sched stage_table_setprio gfx942 */
    "gfx942|head_size:64;sched:stage_table_setprio",
    /* sched stage_table_setprio gfx950 */
    "gfx950|head_size:128;sched:stage_table_setprio",
    /* sched stage_table_setprio tuned11 */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;sched:stage_table_setprio",
    /* sched stage_table_setprio prefetch edge gfx950 d64 */
    "gfx950|head_size:64;sched:stage_table_setprio;kv_prefetch:True;edge_tiles:True",
    /* block_n 16 gfx942 d32 */
    "gfx942|head_size:32;block_n:16",
    /* block_n 16 gfx950 d64 */
    "gfx950|head_size:64;block_n:16",
    /* block_n 32 gfx942 d32 */
    "gfx942|head_size:32;block_n:32",
    /* block_n 32 gfx950 d64 */
    "gfx950|head_size:64;block_n:32",
    /* block_n 64 gfx942 d32 */
    "gfx942|head_size:32;block_n:64",
    /* block_n 64 gfx950 d64 */
    "gfx950|head_size:64;block_n:64",
    /* block_n 128 gfx942 d32 */
    "gfx942|head_size:32;block_n:128",
    /* block_n 128 gfx950 d64 */
    "gfx950|head_size:64;block_n:128",
    /* block_n 256 gfx942 d32 */
    "gfx942|head_size:32;block_n:256",
    /* block_n 256 gfx950 d64 */
    "gfx950|head_size:64;block_n:256",
    /* block_m 16 waves 1 gfx942 */
    "gfx942|head_size:64;block_m:16;waves:1",
    /* block_m 32 waves 2 gfx950 */
    "gfx950|head_size:128;block_m:32;waves:2",
    /* block_m 128 waves 8 gfx942 d32 */
    "gfx942|head_size:32;block_m:128;waves:8",
    /* block_m 128 waves 4 gfx950 d64 */
    "gfx950|head_size:64;block_m:128",
    /* block_m 32 waves 1 gfx950 d32 */
    "gfx950|head_size:32;block_m:32;waves:1",
    /* bf16 gfx942 */
    "gfx942|head_size:64;dtype:bf16",
    /* bf16 gfx950 */
    "gfx950|head_size:128;dtype:bf16",
    /* d32 gfx942 */
    "gfx942|head_size:32",
    /* d128 gfx942 */
    "gfx942|head_size:128",
    /* d32 bf16 gfx950 */
    "gfx950|head_size:32;dtype:bf16",
    /* d64 gfx950 */
    "gfx950|head_size:64",
    /* no mask gfx942 */
    "gfx942|head_size:64;mask_class:none",
    /* no mask gfx950 edge */
    "gfx950|head_size:128;mask_class:none;edge_tiles:True",
    /* thd gfx942 */
    "gfx942|head_size:64;seq_mode:thd",
    /* thd gfx950 band edge */
    "gfx950|head_size:128;seq_mode:thd;edge_tiles:True",
    /* thd no mask gfx950 group */
    "gfx950|head_size:128;seq_mode:thd;mask_class:none;head_loop:group",
    /* stage_vec 1 gfx942 */
    "gfx942|head_size:64;stage_vec:1",
    /* narrow 2 gfx942 */
    "gfx942|head_size:64;stage_vec:1;narrow_vec:2",
    /* narrow 4 gfx950 */
    "gfx950|head_size:128;stage_vec:1;narrow_vec:4",
    /* narrow 4 thd bf16 gfx950 d64 */
    "gfx950|head_size:64;stage_vec:1;narrow_vec:4;seq_mode:thd;dtype:bf16",
    /* token-major gfx942 */
    "gfx942|head_size:64;ws_layout:token_major",
    /* token-major group gfx950 */
    "gfx950|head_size:128;ws_layout:token_major;head_loop:group",
    /* stats_prescale off gfx942 */
    "gfx942|head_size:64;stats_prescale:False",
    /* stats_prescale off edge gfx950 */
    "gfx950|head_size:128;stats_prescale:False;edge_tiles:True",
    /* stats_prescale off runtime gfx942 */
    "gfx942|head_size:64;stats_prescale:False;static_scale:runtime",
    /* stats_prescale off convert gfx950 */
    "gfx950|head_size:128;stats_prescale:False;static_scale:convert",
    /* convert gfx942 */
    "gfx942|head_size:64;static_scale:convert",
    /* runtime gfx950 */
    "gfx950|head_size:128;static_scale:runtime",
    /* runtime edge group gfx942 */
    "gfx942|head_size:64;static_scale:runtime;edge_tiles:True;head_loop:group",
    /* group gfx942 */
    "gfx942|head_size:64;head_loop:group",
    /* group gfx950 */
    "gfx950|head_size:128;head_loop:group",
    /* group edge gfx942 */
    "gfx942|head_size:64;head_loop:group;edge_tiles:True",
    /* group prefetch gfx950 d64 */
    "gfx950|head_size:64;head_loop:group;kv_prefetch:True",
    /* edge gfx942 */
    "gfx942|head_size:64;edge_tiles:True",
    /* edge gfx950 */
    "gfx950|head_size:128;edge_tiles:True",
    /* prefetch gfx942 */
    "gfx942|head_size:64;kv_prefetch:True",
    /* prefetch gfx950 */
    "gfx950|head_size:128;kv_prefetch:True",
    /* prefetch edge gfx942 */
    "gfx942|head_size:64;kv_prefetch:True;edge_tiles:True",
    /* prefetch tail block_n 16 waves 8 gfx942 */
    "gfx942|head_size:64;block_n:16;block_m:128;waves:8;kv_prefetch:True",
    /* prefetch tail block_n 32 waves 8 gfx950 d32 */
    "gfx950|head_size:32;block_n:32;block_m:128;waves:8;kv_prefetch:True",
    /* kv_pad 8 gfx942 */
    "gfx942|head_size:64;kv_pad:8",
    /* kv_pad 16 gfx950 */
    "gfx950|head_size:128;kv_pad:16",
    /* kv_pad 32 prefetch gfx942 */
    "gfx942|head_size:64;kv_pad:32;kv_prefetch:True",
    /* xcd_remap gfx942 */
    "gfx942|head_size:64;xcd_remap:True",
    /* xcd_remap group gfx950 */
    "gfx950|head_size:128;xcd_remap:True;head_loop:group",
    /* lds_plain gfx950 */
    "gfx950|head_size:128;transpose_source:lds_plain",
    /* lds_plain atom_g4 32 gfx950 */
    "gfx950|head_size:128;transpose_source:lds_plain;atom_g4:16x16x32",
    /* rejected: atom_g4 32 on gfx942 */
    "gfx942|head_size:64;atom_g4:16x16x32",
    /* atom_g02 16 gfx950 */
    "gfx950|head_size:128;atom_g02:16x16x16",
    /* rejected: atom_g02 32 on gfx942 */
    "gfx942|head_size:32;atom_g02:16x16x32",
    /* waves_per_eu 1 gfx942 */
    "gfx942|head_size:64;waves_per_eu:1",
    /* waves_per_eu 2 gfx950 d32 */
    "gfx950|head_size:32;waves_per_eu:2",
    /* agpr_alloc backend gfx942 */
    "gfx942|head_size:64;agpr_alloc:backend",
    /* agpr_alloc 128x128 gfx950 */
    "gfx950|head_size:128;agpr_alloc:128x128",
    /* agpr_alloc 0x0 gfx942 d128 */
    "gfx942|head_size:128;agpr_alloc:0x0",
    /* scheduler_strategy max-ilp gfx942 */
    "gfx942|head_size:64;scheduler_strategy:max-ilp",
    /* scheduler_strategy iterative-minreg gfx950 */
    "gfx950|head_size:128;scheduler_strategy:iterative-minreg",
    /* name stem gfx942 */
    "gfx942|head_size:64;name:my_bwd",
    /* tuned11 edge */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;edge_tiles:True",
    /* tuned11 group */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;head_loop:group",
    /* tuned11 prefetch */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;kv_prefetch:True",
    /* tuned14 xcd */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;xcd_remap:True",
    /* tuned14 stats_prescale off */
    "gfx950|head_size:64;dtype:bf16;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:"
    "16x16x32;atom_g4:16x16x16;transpose_source:tr_read;agpr_alloc:0x0;stats_prescale:False",
    /* tuned17 band */
    "gfx950|head_size:64;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:"
    "16x16x16;transpose_source:tr_read;agpr_alloc:0x0;mask_class:band",
    /* tuned17 thd */
    "gfx950|head_size:64;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:"
    "16x16x16;transpose_source:tr_read;agpr_alloc:0x0;seq_mode:thd",
    /* tuned17 lds_plain */
    "gfx950|head_size:64;mask_class:none;waves:4;block_m:128;block_n:64;atom_g02:16x16x32;atom_g4:"
    "16x16x16;transpose_source:tr_read;agpr_alloc:0x0;transpose_source:lds_plain",
    /* dq_spec gfx942 d64 */
    "main=gfx942|head_size:64;dq_mode:split#",
    /* dq_spec gfx950 d128 bf16 thd dq_before_atomic */
    "main=gfx950|head_size:128;dtype:bf16;seq_mode:thd;dq_mode:split;static_scale:dq_before_atomic#"
    "edge_tiles:True",
    /* dq_spec gfx942 d32 runtime token-major */
    "main=gfx942|head_size:32;static_scale:runtime;ws_layout:token_major#head_loop:group;sched:"
    "iglp1",
    /* dq_spec gfx950 d64 narrow none convert */
    "main=gfx950|head_size:64;stage_vec:1;narrow_vec:2;mask_class:none;static_scale:convert#kv_"
    "prefetch:True",
    /* rejected: rdna arch */
    "gfx1201|head_size:64",
    /* rejected: unknown arch */
    "gfx1100|head_size:64",
    /* rejected: block_m not a multiple of 16 waves */
    "gfx942|head_size:64;block_m:32;waves:4",
    /* rejected: tr_read on gfx942 */
    "gfx942|head_size:64;transpose_source:tr_read",
    /* rejected: narrow_vec with stage_vec 8 */
    "gfx942|head_size:64;narrow_vec:2",
    /* rejected: bad sched */
    "gfx942|head_size:64;sched:stage_table_atomics",
    /* rejected: tr_read K-permuted G4' */
    "gfx950|head_size:128;atom_g4:16x16x32",
    /* rejected: LDS plan */
    "gfx942|head_size:128;block_m:128;block_n:256;waves:2",
    /* rejected: AGPR estimate */
    "gfx942|head_size:128;block_m:128;block_n:128;waves:2",
    /* rejected: agpr_alloc below the AGPR estimate */
    "gfx942|head_size:128;block_m:128;block_n:64;waves:2;agpr_alloc:128x128",
    /* rejected: dq_spec static_scale */
    "main=gfx942|head_size:64;static_scale:bogus#",
    /* rejected: block_n vs the G4' atom K */
    "gfx950|head_size:64;block_n:16;atom_g4:16x16x32;transpose_source:lds_plain",
    /* rejected: arch-VGPR estimate past the slack */
    "gfx942|head_size:128;block_m:128;block_n:64;waves:2;agpr_alloc:0x0",
    /* rejected: agpr_alloc 0x0 over the arch VGPRs */
    "gfx942|head_size:64;block_m:64;block_n:32;waves:1;agpr_alloc:0x0",
    /* rejected: waves_per_eu 2 register share */
    "gfx942|head_size:128;block_m:128;block_n:64;waves:2;waves_per_eu:2",
    /* rejected: workgroup share of eight waves */
    "gfx950|head_size:128;block_m:128;block_n:256;waves:8",
    /* rejected: bad name */
    "gfx942|head_size:64;name:bad-name",
};

#define ROCKE_ATTN_BWD_SPEC_EMIT_CONFIGS 1
#define ROCKE_ATTN_BWD_SPEC_EMIT_PARSE_ONLY 1
#include "attention_bwd_spec_emit.c"

/* Strings of the table live as long as the process; a parsed value that is
 * not a literal of it is kept here. */
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

/* Build the dQ spec of one config; false and *msg on a spec error (a build
 * error, as Python's constructor raising inside build_fn). */
static int dq_config_spec(const char* config,
                          std::string* arch,
                          rocke_attn_bwd_dq_spec_t* out,
                          std::string* msg)
{
    std::string c(config);
    rocke_attn_bwd_err_t e;
    if(c.rfind("main=", 0) == 0)
    {
        std::string body = c.substr(5);
        size_t hash = body.find('#');
        std::string main_text = body.substr(0, hash);
        std::string knobs = hash == std::string::npos ? std::string() : body.substr(hash + 1);
        rocke_attn_bwd_spec_t ms;
        if(!parse_config(dq_keep(main_text), arch, &ms))
            return 2;
        if(rocke_attn_bwd_spec_init(&ms, &e) != ROCKE_OK)
        {
            *msg = e.msg;
            return 1;
        }
        rocke_attn_bwd_dq_spec_t k = rocke_attn_bwd_dq_spec_default();
        if(!dq_parse_fields(knobs, &k))
            return 2;
        if(rocke_attn_bwd_spec_dq_spec(&ms, &k, out, &e) != ROCKE_OK)
        {
            *msg = e.msg;
            return 1;
        }
        return 0;
    }
    size_t bar = c.find('|');
    *arch = c.substr(0, bar);
    *out = rocke_attn_bwd_dq_spec_default();
    if(!dq_parse_fields(c.substr(bar + 1), out))
        return 2;
    if(rocke_attn_bwd_dq_spec_init(out, &e) != ROCKE_OK)
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
    rocke_attn_bwd_dq_spec_t spec;
    int rc = dq_config_spec(CONFIGS[idx], &arch, &spec, &msg);
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
    rocke_kernel_def_t* kernel = rocke_build_attn_bwd_dq_new(&b, &spec, arch.c_str());
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
