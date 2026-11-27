/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx950_attention_dense_builder.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static int make_spec(long idx, rocke_gfx950_attention_dense_spec_t* s)
{
    *s = rocke_gfx950_attention_dense_spec_default();
    rocke_attention_dense_spec_t* b = &s->base;
    b->batch = 1;
    b->seqlen_q = b->seqlen_kv = 512;
    b->num_query_heads = 8;
    b->num_kv_heads = 2;
    b->head_size = 128;
    b->dtype = "bf16";
    b->causal = true;

    /* Sparse overrides of the native defaults, keyed as in the Python tuple.
     * OPT sets presence even for false, zero or -1; omission means None. */
#define ROW(i, ...)  \
    case i:          \
        __VA_ARGS__; \
        break
#define OPT(field, value)  \
    s->has_##field = true; \
    s->field = (value)
#define LENGTHS(q, k)  \
    b->seqlen_q = (q); \
    b->seqlen_kv = (k)
#define EMPTY_WINDOW  \
    LENGTHS(256, 64); \
    b->sliding_window = 64
#define PAGED           \
    b->paged = true;    \
    b->block_size = 16; \
    b->num_kv_blocks = 64
#define PERSIST(n)        \
    b->persistent = true; \
    b->num_persistent = (n)
#define WIDE      \
    PERSIST(256); \
    s->wide_lds_dma = true
#define MANUAL_OFF              \
    OPT(pv_sched_fence, false); \
    OPT(pv_sched_group_template, false)
    switch(idx)
    {
        ROW(0, (void)0);
        ROW(1, b->causal = false; b->dtype = "fp16");
        ROW(2, b->head_size = 64);
        ROW(3, b->head_size = 64; b->dtype = "fp16"; b->causal = false;
            b->num_query_heads = b->num_kv_heads = 4);
        ROW(4, b->num_kv_heads = 1);
        ROW(5, b->batch = 4; LENGTHS(1024, 1024); b->sliding_window = 512);
        ROW(6, b->block_m = 128);
        ROW(7, b->block_n = 128);
        ROW(8, b->head_size = 64; b->block_n = 32);
        ROW(9, b->block_m = 128; b->block_n = 32; b->head_size = 64);
        ROW(10, b->sliding_window = 256);
        ROW(11, b->causal = false; b->sliding_window = 128);
        ROW(12, s->right_bound = 64);
        ROW(13, b->sliding_window = 256; s->right_bound = 64);
        ROW(14, b->seqlen_q = 256; b->causal_bottom_right = true);
        ROW(15, b->causal_bottom_right = true);
        ROW(16, EMPTY_WINDOW);
        ROW(17, LENGTHS(500, 500); b->ragged = true);
        ROW(18, LENGTHS(300, 500); b->ragged = true; b->causal = false);
        ROW(19, LENGTHS(200, 500); b->ragged = true; b->causal_bottom_right = true);
        ROW(20, LENGTHS(500, 500); b->ragged = true; b->head_size = 64; b->dtype = "fp16");
        ROW(21, b->use_sinks = true);
        ROW(22, b->use_sinks = true; EMPTY_WINDOW);
        ROW(23, s->emit_lse = true);
        ROW(24, s->emit_lse = true; b->use_sinks = true; EMPTY_WINDOW);
        ROW(25, s->emit_lse = true; LENGTHS(500, 500); b->ragged = true);
        ROW(26, s->emit_lse = true; b->causal = false; b->head_size = 64; b->dtype = "fp16");
        ROW(27, b->varlen = true; b->batch = 2);
        ROW(28, b->varlen = true; b->batch = 2; s->emit_lse = true);
        ROW(29, PAGED; b->sliding_window = 256);
        ROW(30, PAGED; b->sliding_window = 256; s->emit_lse = true);
        ROW(31, s->o_store_width = 2);
        ROW(32, s->o_store_width = 1);
        ROW(33, b->lazy_rescale = false);
        ROW(34, s->lazy_rescale_threshold = 4.0);
        ROW(35, s->use_exp2_fast = false);
        ROW(36, OPT(exp_per_pv_step, 2));
        ROW(37, s->partial_vmcnt_prefetch = false);
        ROW(38, s->pv_priority = 0);
        ROW(39, s->pv_priority = 3);
        ROW(40, OPT(pv_sched_fence, false));
        ROW(41, s->pv_sched_fence_mask = 0x8);
        ROW(42, OPT(pv_sched_group_template, false));
        ROW(43, s->pv_sched_group_ds_read = 4);
        ROW(44, OPT(iglp_mode, 1); MANUAL_OFF);
        ROW(45, OPT(iglp_mode, 0); MANUAL_OFF);
        ROW(46, s->pv_loop_order = "k_major");
        ROW(47, s->causal_diag_split = false);
        ROW(48, s->lds_k_row_pad = 16; s->lds_v_row_pad = 16);
        ROW(49, b->head_size = 64; b->lds_k_group_pad = 0);
        ROW(50, b->waves_per_eu = 1);
        ROW(51, PERSIST(256));
        ROW(52, PERSIST(256); b->persist_decode = "hkv_major");
        ROW(53, PERSIST(4); b->persist_decode = "gqa_pair");
        ROW(54, PERSIST(8); b->persist_decode = "gqa_pair_2phase");
        ROW(55, PERSIST(256); b->interleave = true);
        ROW(56, PERSIST(256); b->causal = false; b->dtype = "fp16");
        ROW(57, PERSIST(256); b->sliding_window = 256);
        ROW(58, PERSIST(256); s->right_bound = 64);
        ROW(59, PERSIST(256); b->use_sinks = true; s->emit_lse = true);
        ROW(60, PERSIST(256); b->use_sinks = true; s->emit_lse = true; EMPTY_WINDOW);
        ROW(61, PERSIST(256); LENGTHS(500, 500); b->ragged = true);
        ROW(62, PERSIST(256); b->head_size = 64; b->dtype = "fp16");
        ROW(63, PERSIST(256); b->block_m = 128);
        ROW(64, PERSIST(256); b->lazy_rescale = false);
        ROW(65, PERSIST(256); OPT(exp_per_pv_step, 2));
        ROW(66, PERSIST(256); s->pv_loop_order = "k_major"; s->o_store_width = 2);
        ROW(67, WIDE);
        ROW(68, WIDE; s->emit_lse = true; b->use_sinks = true);
        ROW(69, WIDE; b->causal = false);
        ROW(70, WIDE; s->pv_loop_order = "d_major"; OPT(iglp_mode, 0));
        ROW(71, WIDE; OPT(iglp_mode, -1); OPT(pv_sched_fence, true));
        ROW(72, s->kv_segments = 2);
        ROW(73, s->kv_storage_dtype = "fp8e4m3");
        ROW(74, PAGED; b->head_size = 64; b->sliding_window = 0);
        ROW(75, b->nonpersist_decode = "hq_minor_swz");
        ROW(76, PERSIST(256); b->persist_decode = "hq_minor_swz");
        ROW(77, s->kv_segments = 2; s->kv_storage_dtype = "fp8e4m3");
        ROW(78, s->kv_storage_dtype = "fp8e4m3"; s->emit_lse = true);
        ROW(79, LENGTHS(500, 500); b->ragged = true; b->use_sinks = true);
        ROW(80, PERSIST(256); LENGTHS(500, 500); b->ragged = true; s->emit_lse = true);
    default:
        return -1;
    }
#undef ROW
#undef OPT
#undef LENGTHS
#undef EMPTY_WINDOW
#undef PAGED
#undef PERSIST
#undef WIDE
#undef MANUAL_OFF
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2 || argc > 3)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    const char* mode = argc == 3 ? argv[2] : "ll";
    if(strcmp(mode, "ll") && strcmp(mode, "ir") && strcmp(mode, "verify"))
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    char* end;
    long idx = strtol(argv[1], &end, 10);
    rocke_gfx950_attention_dense_spec_t spec;
    if(end == argv[1] || *end || make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %s\n", argv[1]);
        return 1;
    }
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_gfx950_build_attention_dense_new(&b, &spec, "gfx950");
    int rc = 0;
    if(!kernel)
    {
        fprintf(stderr, "build failed: err=%s\n", b.err);
        rc = 1;
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diags = NULL;
        size_t n = 0;
        rocke_verify(kernel, &diags, &n);
        for(size_t i = 0; i < n; ++i)
        {
            char* text = rocke_diag_to_string(&diags[i]);
            if(text)
                puts(text);
            else
                rc = 1;
            free(text);
        }
        rocke_diags_free(diags, n);
    }
    else
    {
        char* text = NULL;
        rocke_status_t status
            = strcmp(mode, "ir") == 0
                  ? rocke_ir_serialize(kernel, &text)
                  : rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &text);
        if(status == ROCKE_OK && text)
            fputs(text, stdout);
        else
        {
            fprintf(stderr, "%s emission failed: status=%d\n", mode, (int)status);
            rc = 1;
        }
        free(text);
    }
    rocke_ir_builder_free(&b);
    return rc;
}
