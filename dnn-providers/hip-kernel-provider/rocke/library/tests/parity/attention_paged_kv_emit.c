/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 * Mirrors attention_paged_kv_emit.py without changing public spec layouts. */
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
    if(idx < 0 || idx >= 48)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }
    const int use942 = idx < 24;
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
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "strided_kv") != ROCKE_OK)
        return 1;
    rocke_kernel_def_t* k = use942 ? rocke_build_unified_attention_3d_tiled_gfx942(&b, &s, arch)
                                   : rocke_build_unified_attention_3d_tiled_gfx950(&b, &s, arch);
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
