/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx942_attention_tiled_3d_emit.c -- C-side emitter for the
 * gfx942 narrow-atom tiled split-KV 3D attention parity harness.
 * Selects one of the sampled configs by argv[1], fills a
 * rocke_unified_attention_3d_tiled_spec_t identically to the Python emitter
 * gfx942_attention_tiled_3d_emit.py, and emits BOTH kernels, segment first:
 *   1. the segment kernel via
 *      rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm (paged K/V)
 *      or rocke_build_unified_attention_3d_tiled_strided_gfx942 (configs
 *      18-27: strided K/V, the Python kv_layout="strided"), then
 *   2. the reduce kernel via
 *      rocke_build_unified_attention_reduce_tiled_gfx942_lower_to_llvm, with a
 *      rocke_unified_attention_reduce_tiled_spec_t derived from the same config
 *      (head_size / num_query_heads / num_kv_heads / dtype / num_segments /
 *      waves_per_eu, as the Python reduce_spec_for_segment)
 * (arch gfx942, flavor AUTO), printing the .ll to stdout for byte comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* A strided (non-paged) decode spec as the strided decode candidate selects it:
 * one CTA per sequence, streamed K/V, the runtime's auto tile for the block
 * (tile <= 0: no override). Mirrors the Python _strided_decode. */
static void strided_decode(rocke_unified_attention_3d_tiled_spec_t* s,
                           int head_size,
                           int block_size,
                           int tile,
                           const char* dtype,
                           int num_segments,
                           int num_query_heads,
                           int num_kv_heads)
{
    s->head_size = head_size;
    s->block_size = block_size;
    s->num_query_heads = num_query_heads;
    s->num_kv_heads = num_kv_heads;
    s->dtype = dtype;
    s->num_segments = num_segments;
    s->use_sinks = false;
    s->sliding_window = 0;
    s->has_softcap = false;
    s->kv_storage_dtype = NULL;
    s->num_seqs = 4;
    s->has_tile_size_override = tile > 0;
    s->tile_size_override = tile > 0 ? tile : 0;
    s->use_decode_grid = true;
    s->kv_cache_policy = "stream";
}

/* Fill `s` for config index `idx` and set `*strided` when its K/V is strided
 * (non-paged). Returns 0 on success, -1 on unknown idx. */
static int make_spec(int idx, rocke_unified_attention_3d_tiled_spec_t* s, bool* strided)
{
    *s = rocke_unified_attention_3d_tiled_spec_default();
    *strided = false;
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
        /* All-decode grid (seq_idx = block_id_x, no search). Decode-shaped GQA. */
        s->head_size = 128;
        s->block_size = 16;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "fp16";
        s->num_segments = 16;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_decode_grid = true;
        break;
    case 6:
        /* All-decode grid at NQK=16 (BLOCK_Q=1) with fp8 KV and sinks. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 16;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->num_segments = 32;
        s->use_sinks = true;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = "fp8e4m3";
        s->use_decode_grid = true;
        break;
    case 7:
        /* D256 at block 64 (dispatch tile T=32): fits gfx942's 64 KiB LDS only
         * with one V slot. */
        s->head_size = 256;
        s->block_size = 64;
        s->num_query_heads = 16;
        s->num_kv_heads = 2;
        s->dtype = "bf16";
        s->num_segments = 8;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->has_tile_size_override = true;
        s->tile_size_override = 32;
        break;
    case 8:
        /* Decode K/V streamed (nt) through the wide loader. */
        s->head_size = 128;
        s->block_size = 16;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->num_segments = 16;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_wide_kv_load = true;
        s->use_decode_grid = true;
        s->kv_cache_policy = "stream";
        break;
    case 9:
        /* Decode K/V with SC0 + NT through the 1-DWORD async loader, with sinks. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "fp16";
        s->num_segments = 16;
        s->use_sinks = true;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_decode_grid = true;
        s->kv_cache_policy = "nt";
        break;
    case 10:
        /* Decode K/V streamed through the wide loader into swizzled LDS tiles. */
        s->head_size = 128;
        s->block_size = 16;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->num_segments = 16;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_wide_kv_load = true;
        s->use_decode_grid = true;
        s->kv_cache_policy = "stream";
        s->use_kv_lds_swizzle = true;
        break;
    case 11:
        /* Swizzled LDS tiles at head size 64, with sinks and softcap. */
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 16;
        s->num_kv_heads = 4;
        s->dtype = "fp16";
        s->num_segments = 8;
        s->use_sinks = true;
        s->sliding_window = 0;
        s->has_softcap = true;
        s->kv_storage_dtype = NULL;
        s->use_wide_kv_load = true;
        s->use_kv_lds_swizzle = true;
        break;
    case 12:
        /* Decode K/V streamed into swizzled LDS tiles, addressed from the wave's
         * page-id window. */
        s->head_size = 128;
        s->block_size = 16;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->num_segments = 16;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_wide_kv_load = true;
        s->use_decode_grid = true;
        s->kv_cache_policy = "stream";
        s->use_kv_lds_swizzle = true;
        s->use_wave_page_id_load = true;
        break;
    case 13:
        /* The wave's page-id window without the swizzle, at head size 64 with a
         * sliding window. */
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 16;
        s->num_kv_heads = 4;
        s->dtype = "fp16";
        s->num_segments = 8;
        s->use_sinks = false;
        s->sliding_window = 64;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_wide_kv_load = true;
        s->use_wave_page_id_load = true;
        break;
    case 14:
        /* Decode K/V streamed by the async copies into swizzled LDS tiles. */
        s->head_size = 256;
        s->block_size = 16;
        s->num_query_heads = 64;
        s->num_kv_heads = 64;
        s->dtype = "bf16";
        s->num_segments = 32;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_decode_grid = true;
        s->kv_cache_policy = "stream";
        s->use_kv_lds_swizzle = true;
        break;
    case 15:
        /* Decode K/V streamed by the async copies into swizzled LDS tiles, addressed
         * from the wave's page-id window. */
        s->head_size = 256;
        s->block_size = 16;
        s->num_query_heads = 64;
        s->num_kv_heads = 64;
        s->dtype = "bf16";
        s->num_segments = 32;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->use_decode_grid = true;
        s->kv_cache_policy = "stream";
        s->use_kv_lds_swizzle = true;
        s->use_wave_page_id_load = true;
        break;
    case 16:
        /* The wave's page-id window with half-page tiles on the wide loader. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "fp16";
        s->num_segments = 16;
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->has_tile_size_override = true;
        s->tile_size_override = 16;
        s->use_wide_kv_load = true;
        s->use_wave_page_id_load = true;
        break;
    case 17:
        /* The wave's page-id window with half-page tiles on the async copies. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->num_segments = 8;
        s->use_sinks = true;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        s->has_tile_size_override = true;
        s->tile_size_override = 16;
        s->use_wave_page_id_load = true;
        break;
    /* 18-26: representative strided decode specs spanning the catalog's D x block
     * axes (D64/128/256 x block 16/32/64); 27 is 22 with waves_per_eu=2. */
    case 18:
        strided_decode(s, 64, 16, 0, "fp16", 64, 32, 8);
        break;
    case 19:
        strided_decode(s, 64, 32, 0, "bf16", 16, 32, 8);
        break;
    case 20:
        strided_decode(s, 64, 64, 0, "fp16", 8, 32, 8);
        break;
    case 21:
        strided_decode(s, 128, 16, 0, "bf16", 60, 16, 16);
        break;
    case 22:
        strided_decode(s, 128, 32, 16, "fp16", 16, 32, 8);
        break;
    case 23:
        strided_decode(s, 128, 64, 32, "bf16", 2, 32, 8);
        break;
    case 24:
        strided_decode(s, 256, 16, 0, "fp16", 22, 32, 8);
        break;
    case 25:
        strided_decode(s, 256, 32, 16, "bf16", 64, 64, 4);
        break;
    case 26:
        strided_decode(s, 256, 64, 32, "fp16", 8, 32, 8);
        break;
    case 27:
        strided_decode(s, 128, 32, 16, "fp16", 16, 32, 8);
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        break;
    default:
        return -1;
    }
    if(idx >= 18)
        *strided = true;
    return 0;
}

/* Derive the reduce spec from the segment spec, exactly as the Python
 * reduce_spec_for_segment (head_size / num_query_heads / num_kv_heads / dtype /
 * num_segments / waves_per_eu). */
static void make_reduce_spec(const rocke_unified_attention_3d_tiled_spec_t* s,
                             rocke_unified_attention_reduce_tiled_spec_t* r)
{
    *r = rocke_unified_attention_reduce_tiled_spec_default();
    r->head_size = s->head_size;
    r->num_query_heads = s->num_query_heads;
    r->num_kv_heads = s->num_kv_heads;
    r->dtype = s->dtype;
    r->num_segments = s->num_segments;
    r->has_waves_per_eu = s->has_waves_per_eu;
    r->waves_per_eu = s->waves_per_eu;
}

/* Build the segment kernel into `b` (initialised here, named by the spec):
 * the strided entry for strided K/V, the paged one otherwise. Returns the
 * kernel, or NULL after printing the builder error (b is then freed). */
static rocke_kernel_def_t* build_segment(rocke_ir_builder_t* b,
                                         const rocke_unified_attention_3d_tiled_spec_t* s,
                                         bool strided)
{
    char kname[256];
    kname[0] = 0;
    rocke_unified_attention_3d_tiled_spec_kernel_name(s, kname, sizeof(kname));
    if(rocke_ir_builder_init(b, kname) != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return NULL;
    }
    rocke_kernel_def_t* k
        = strided ? rocke_build_unified_attention_3d_tiled_strided_gfx942(b, s, "gfx942")
                  : rocke_build_unified_attention_3d_tiled_gfx942(b, s, "gfx942");
    if(!k || !rocke_ir_builder_ok(b))
    {
        fprintf(stderr, "segment build failed: err=%s\n", rocke_ir_builder_error(b));
        rocke_ir_builder_free(b);
        return NULL;
    }
    return k;
}

/* Emit the reduce kernel in `mode` (ll / ir / verify) after the segment.
 * Returns 0 on success, 1 on failure. */
static int emit_reduce(const rocke_unified_attention_reduce_tiled_spec_t* r, const char* mode)
{
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_build_unified_attention_reduce_tiled_gfx942_lower_to_llvm(
            r, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "reduce build/lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
        return 0;
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "attention_tiled_3d_reduce") != ROCKE_OK)
    {
        fprintf(stderr, "reduce builder init failed\n");
        return 1;
    }
    rocke_kernel_def_t* red_k = rocke_build_unified_attention_reduce_tiled_gfx942(&b, r, "gfx942");
    if(!red_k)
    {
        fprintf(stderr, "reduce build failed: err=%s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(red_k, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "reduce serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(red_k, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s2 = rocke_diag_to_string(&d[i]);
            if(s2)
            {
                puts(s2);
                free(s2);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
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

    rocke_unified_attention_3d_tiled_spec_t s;
    bool strided = false;
    if(make_spec(idx, &s, &strided) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }
    rocke_unified_attention_reduce_tiled_spec_t r;
    make_reduce_spec(&s, &r);

    if(strcmp(mode, "ll") == 0 && !strided)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
            &s, "gfx942", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "build/lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ll") == 0 || strcmp(mode, "ir") == 0)
    {
        rocke_ir_builder_t b;
        rocke_kernel_def_t* seg_k = build_segment(&b, &s, strided);
        if(!seg_k)
            return 1;
        char* t = NULL;
        rocke_status_t st
            = strcmp(mode, "ir") == 0
                  ? rocke_ir_serialize(seg_k, &t)
                  : rocke_lower_kernel_to_llvm(seg_k, ROCKE_LLVM_FLAVOR_AUTO, "gfx942", &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "segment %s failed: status=%d\n", mode, (int)st);
            free(t);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
        rocke_ir_builder_free(&b);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_ir_builder_t b;
        rocke_kernel_def_t* seg_k = build_segment(&b, &s, strided);
        if(!seg_k)
            return 1;
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(seg_k, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s2 = rocke_diag_to_string(&d[i]);
            if(s2)
            {
                puts(s2);
                free(s2);
            }
        }
        rocke_diags_free(d, n);
        rocke_ir_builder_free(&b);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    return emit_reduce(&r, mode);
}
