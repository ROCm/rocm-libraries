/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx950_attention_tiled_3d_emit.c -- C-side emitter for the
 * gfx950 (CDNA4) WIDE-K tiled split-KV 3D attention parity harness.
 *
 * Selects one of the sampled configs by argv[1], fills a
 * ckc_unified_attention_3d_tiled_spec_t identically to the Python emitter
 * gfx950_attention_tiled_3d_emit.py, then emits, in the same Python execution
 * order, BOTH kernels the module exposes:
 *   1. the per-segment SEGMENT kernel via
 *      ckc_build_unified_attention_3d_tiled_gfx950(&b, &s, "gfx950")
 *   2. the arch-neutral REDUCE kernel via
 *      ckc_build_unified_attention_reduce_tiled_gfx950(&rb, &rs, "gfx950")
 * whose ckc_unified_attention_reduce_tiled_spec_t is derived from the same
 * config (head_size / num_query_heads / num_kv_heads / dtype / num_segments).
 *
 * Each kernel is lowered with ckc_lower_kernel_to_llvm(kernel, AUTO, "gfx950",
 * ...) and the two .ll texts are concatenated (segment first, then reduce) to
 * stdout for byte comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_gfx950_attention_tiled_3d.h"
#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"

/* Fill the segment spec `s` for config index `idx`. Returns 0 on success,
 * -1 on unknown idx. Mirrors the Python emitter's _CONFIGS exactly. */
static int make_spec(int idx, ckc_unified_attention_3d_tiled_spec_t* s)
{
    *s = ckc_unified_attention_3d_tiled_spec_default();
    switch(idx)
    {
    case 0:
        s->head_size = 64;
        s->block_size = 16;
        s->num_query_heads = 8;
        s->num_kv_heads = 8;
        s->dtype = "fp16";
        s->num_segments = 8;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        break;
    case 1:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 16;
        s->num_kv_heads = 16;
        s->dtype = "bf16";
        s->num_segments = 16;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        break;
    case 2:
        s->head_size = 256;
        s->block_size = 64;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "fp16";
        s->num_segments = 8;
        s->use_sinks = false;
        s->sliding_window = 4096;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        break;
    case 3:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->num_segments = 16;
        s->use_sinks = true;
        s->sliding_window = 0;
        s->has_softcap = true;
        s->kv_storage_dtype = "fp8e4m3";
        break;
    case 4:
        s->head_size = 64;
        s->block_size = 16;
        s->num_query_heads = 16;
        s->num_kv_heads = 4;
        s->dtype = "fp16";
        s->num_segments = 8;
        s->use_sinks = false;
        s->sliding_window = 2048;
        s->has_softcap = false;
        s->use_alibi = true;
        s->use_qq_bias = false;
        s->kv_storage_dtype = NULL;
        break;
    case 5:
        /* 64-bit paged-KV addressing (caches > 2 GiB). Decode-shaped MHA. */
        s->head_size = 128;
        s->block_size = 16;
        s->num_query_heads = 16;
        s->num_kv_heads = 16;
        s->dtype = "fp16";
        s->num_segments = 16;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_i64_kv_addr = true;
        break;
    default:
        return -1;
    }
    return 0;
}

/* Derive the reduce spec from the segment spec, exactly as the Python emitter
 * (head_size / num_query_heads / num_kv_heads / dtype / num_segments). */
static void make_reduce_spec(const ckc_unified_attention_3d_tiled_spec_t* s,
                             ckc_unified_attention_reduce_tiled_spec_t* r)
{
    *r = ckc_unified_attention_reduce_tiled_spec_default();
    r->head_size = s->head_size;
    r->num_query_heads = s->num_query_heads;
    r->num_kv_heads = s->num_kv_heads;
    r->dtype = s->dtype;
    r->num_segments = s->num_segments;
}

/* Build kernel via `which` (0=segment, 1=reduce), lower to .ll, print to stdout.
 * Returns 0 on success, 1 on failure. */
static int emit_segment(const ckc_unified_attention_3d_tiled_spec_t* s)
{
    ckc_ir_builder_t b;
    if(ckc_ir_builder_init(&b, "attention_tiled_3d_segment") != CKC_OK)
    {
        fprintf(stderr, "segment builder init failed\n");
        return 1;
    }
    ckc_kernel_def_t* kernel = ckc_build_unified_attention_3d_tiled_gfx950(&b, s, "gfx950");
    if(!kernel)
    {
        fprintf(stderr, "segment build failed: err=%s\n", ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return 1;
    }
    char* llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
    if(st != CKC_OK || !llvm_text)
    {
        fprintf(stderr, "segment lower failed: status=%d\n", (int)st);
        ckc_ir_builder_free(&b);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    ckc_ir_builder_free(&b);
    return 0;
}

static int emit_reduce(const ckc_unified_attention_reduce_tiled_spec_t* r)
{
    ckc_ir_builder_t b;
    if(ckc_ir_builder_init(&b, "attention_tiled_3d_reduce") != CKC_OK)
    {
        fprintf(stderr, "reduce builder init failed\n");
        return 1;
    }
    ckc_kernel_def_t* kernel = ckc_build_unified_attention_reduce_tiled_gfx950(&b, r, "gfx950");
    if(!kernel)
    {
        fprintf(stderr, "reduce build failed: err=%s\n", ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return 1;
    }
    char* llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
    if(st != CKC_OK || !llvm_text)
    {
        fprintf(stderr, "reduce lower failed: status=%d\n", (int)st);
        ckc_ir_builder_free(&b);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    ckc_ir_builder_free(&b);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    ckc_unified_attention_3d_tiled_spec_t s;
    if(make_spec(idx, &s) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }

    ckc_unified_attention_reduce_tiled_spec_t r;
    make_reduce_spec(&s, &r);

    if(strcmp(mode, "ll") == 0)
    {
        if(emit_segment(&s) != 0)
        {
            return 1;
        }
        if(emit_reduce(&r) != 0)
        {
            return 1;
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        /* segment */
        ckc_ir_builder_t b;
        if(ckc_ir_builder_init(&b, "attention_tiled_3d_segment") != CKC_OK)
        {
            fprintf(stderr, "segment builder init failed\n");
            return 1;
        }
        ckc_kernel_def_t* seg_k = ckc_build_unified_attention_3d_tiled_gfx950(&b, &s, "gfx950");
        if(!seg_k)
        {
            fprintf(stderr, "segment build failed: err=%s\n", ckc_ir_builder_error(&b));
            ckc_ir_builder_free(&b);
            return 1;
        }
        char* t = NULL;
        ckc_status_t st = ckc_ir_serialize(seg_k, &t);
        if(st != CKC_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
        ckc_ir_builder_free(&b);

        /* reduce */
        ckc_ir_builder_t rb;
        if(ckc_ir_builder_init(&rb, "attention_tiled_3d_reduce") != CKC_OK)
        {
            fprintf(stderr, "reduce builder init failed\n");
            return 1;
        }
        ckc_kernel_def_t* red_k
            = ckc_build_unified_attention_reduce_tiled_gfx950(&rb, &r, "gfx950");
        if(!red_k)
        {
            fprintf(stderr, "reduce build failed: err=%s\n", ckc_ir_builder_error(&rb));
            ckc_ir_builder_free(&rb);
            return 1;
        }
        char* t2 = NULL;
        ckc_status_t st2 = ckc_ir_serialize(red_k, &t2);
        if(st2 != CKC_OK || !t2)
        {
            fprintf(stderr, "reduce serialize failed: status=%d\n", (int)st2);
            ckc_ir_builder_free(&rb);
            return 1;
        }
        fputs(t2, stdout);
        free(t2);
        ckc_ir_builder_free(&rb);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        /* segment */
        ckc_ir_builder_t b;
        if(ckc_ir_builder_init(&b, "attention_tiled_3d_segment") != CKC_OK)
        {
            fprintf(stderr, "segment builder init failed\n");
            return 1;
        }
        ckc_kernel_def_t* seg_k = ckc_build_unified_attention_3d_tiled_gfx950(&b, &s, "gfx950");
        if(!seg_k)
        {
            fprintf(stderr, "segment build failed: err=%s\n", ckc_ir_builder_error(&b));
            ckc_ir_builder_free(&b);
            return 1;
        }
        ckc_diag_t* d = NULL;
        size_t n = 0;
        ckc_verify(seg_k, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s2 = ckc_diag_to_string(&d[i]);
            if(s2)
            {
                puts(s2);
                free(s2);
            }
        }
        ckc_diags_free(d, n);
        ckc_ir_builder_free(&b);

        /* reduce */
        ckc_ir_builder_t rb;
        if(ckc_ir_builder_init(&rb, "attention_tiled_3d_reduce") != CKC_OK)
        {
            fprintf(stderr, "reduce builder init failed\n");
            return 1;
        }
        ckc_kernel_def_t* red_k
            = ckc_build_unified_attention_reduce_tiled_gfx950(&rb, &r, "gfx950");
        if(!red_k)
        {
            fprintf(stderr, "reduce build failed: err=%s\n", ckc_ir_builder_error(&rb));
            ckc_ir_builder_free(&rb);
            return 1;
        }
        ckc_diag_t* d2 = NULL;
        size_t n2 = 0;
        ckc_verify(red_k, &d2, &n2);
        for(size_t i = 0; i < n2; i++)
        {
            char* s2 = ckc_diag_to_string(&d2[i]);
            if(s2)
            {
                puts(s2);
                free(s2);
            }
        }
        ckc_diags_free(d2, n2);
        ckc_ir_builder_free(&rb);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    return 0;
}
