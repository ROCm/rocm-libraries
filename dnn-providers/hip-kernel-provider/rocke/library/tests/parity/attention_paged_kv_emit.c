/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 * Mirrors attention_paged_kv_emit.py without changing public spec layouts. */
#include "rocke/instance_attention_paged_decode_bounded.h"
#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/instance_gfx950_attention_tiled_3d.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
    if(argc < 2)
        return 2;
    const int idx = atoi(argv[1]);
    if(idx < 0 || idx >= 98)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }
    const int use942 = idx < 24 || (idx >= 48 && idx < 57) || (idx >= 66 && idx < 82);
    const int ci = idx / 4 % 3;
    const char* arch = use942 ? "gfx942" : "gfx950";
    const char* mode = argc > 2 ? argv[2] : "ll";
    const int dims[] = {64, 128, 256};
    const int blocks[] = {1, 16, 32, 64};
    rocke_unified_attention_3d_tiled_spec_t s = rocke_unified_attention_3d_tiled_spec_default();
    s.head_size = dims[ci];
    s.block_size = blocks[idx % 4];
    s.num_query_heads = 8;
    s.num_kv_heads = 2;
    s.dtype = idx / 12 % 2 == 0 ? "fp16" : "bf16";
    s.use_sinks = false;
    s.sliding_window = idx % 2 ? 17 : 0;
    s.has_softcap = false;
    s.num_segments = 8;
    s.has_tile_size_override = true;
    s.tile_size_override = 32;
    s.has_waves_per_eu = true;
    s.waves_per_eu = 3;
    s.num_seqs = 3;
    /* Append feature cases without changing the original 48 configurations. */
    if(idx >= 48 && idx < 66)
    {
        const int feature = (idx - 48) % 9;
        const int combined = feature >= 4;
        s.head_size = 128;
        s.block_size = feature >= 4 && feature < 8 ? blocks[feature - 4] : 1;
        s.dtype = feature == 8 ? "bf16" : "fp16";
        s.sliding_window = 33;
        s.has_softcap = combined || feature == 0;
        s.use_sinks = combined || feature == 1;
        s.use_alibi = combined || feature == 2;
        s.use_qq_bias = combined || feature == 3;
    }
    const int bounded = idx >= 66;
    const int local = (idx - 66) % 16;
    const int reducer = bounded && local >= 12;
    rocke_unified_attention_reduce_tiled_spec_t red
        = rocke_unified_attention_reduce_tiled_spec_default();
    if(bounded)
    {
        s.head_size = dims[local / 4 % 3];
        s.block_size = blocks[local % 4];
        s.dtype = local % 2 ? "bf16" : "fp16";
        s.use_sinks = local >= 8;
        s.sliding_window = local >= 8 ? 33 : 0;
        s.has_softcap = local >= 8;
        s.use_alibi = local >= 8;
        s.use_qq_bias = local >= 8;
        s.use_decode_grid = true;
        red.head_size = local == 15 ? 64 : dims[local % 3];
        red.num_query_heads = 8;
        red.num_kv_heads = 2;
        red.dtype = "bf16";
        red.num_segments = 8;
        red.has_waves_per_eu = true;
        red.waves_per_eu = 3;
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "strided_kv") != ROCKE_OK)
        return 1;
    rocke_kernel_def_t* k;
    if(reducer)
        k = (use942 ? rocke_build_paged_decode_reduce_bounded_v1_gfx942
                    : rocke_build_paged_decode_reduce_bounded_v1_gfx950)(&b, &red, arch);
    else if(bounded)
        k = (use942 ? rocke_build_paged_decode_bounded_v1_gfx942
                    : rocke_build_paged_decode_bounded_v1_gfx950)(&b, &s, arch);
    else
        k = (use942 ? rocke_build_unified_attention_3d_tiled_gfx942
                    : rocke_build_unified_attention_3d_tiled_gfx950)(&b, &s, arch);

    if(!k || !rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "%s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    int rc = 0;
    if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* ds = NULL;
        size_t n = 0;
        rocke_verify(k, &ds, &n);
        for(size_t i = 0; i < n; ++i)
        {
            char* text = rocke_diag_to_string(&ds[i]);
            if(text)
            {
                puts(text);
                free(text);
            }
        }
        rocke_diags_free(ds, n);
    }
    else
    {
        char* text = NULL;
        rocke_status_t status
            = strcmp(mode, "ir") == 0
                  ? rocke_ir_serialize(k, &text)
                  : rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_AUTO, arch, &text);
        if(status != ROCKE_OK || !text)
            rc = 1;
        else
            fputs(text, stdout);
        free(text);
    }
    rocke_ir_builder_free(&b);
    return rc;
}
