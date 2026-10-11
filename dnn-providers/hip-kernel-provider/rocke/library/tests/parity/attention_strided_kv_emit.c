/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 * Mirrors attention_strided_kv_emit.py without changing public spec layouts. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/instance_gfx950_attention_tiled_3d.h"
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

int main(int argc, char** argv)
{
    if(argc < 2)
        return 2;
    const int idx = atoi(argv[1]);
    if(idx < 0 || idx >= 12)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }
    const int use942 = idx % 6 < 3;
    const int ci = idx % 3;
    const char* arch = use942 ? "gfx942" : "gfx950";
    const char* mode = argc > 2 ? argv[2] : "ll";
    const int dims[] = {64, 128, 256};
    const int blocks[] = {16, 32, 64};
    const int qheads[] = {4, 8, 8};
    const int kheads[] = {2, 2, 1};
    rocke_unified_attention_3d_tiled_spec_t s = rocke_unified_attention_3d_tiled_spec_default();
    s.head_size = dims[ci];
    s.block_size = blocks[ci];
    s.num_query_heads = qheads[ci];
    s.num_kv_heads = kheads[ci];
    s.dtype = ci == 1 ? "bf16" : "fp16";
    s.use_sinks = false;
    s.sliding_window = ci == 2 ? 17 : 0;
    s.has_softcap = false;
    s.num_segments = 4;
    s.num_seqs = 3;
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "strided_kv") != ROCKE_OK)
        return 1;
    rocke_kernel_def_t* k
        = use942 ? (idx < 6 ? rocke_build_unified_attention_3d_tiled_strided_gfx942(&b, &s, arch)
                            : rocke_build_unified_attention_3d_tiled_gfx942(&b, &s, arch))
                 : (idx < 6 ? rocke_build_unified_attention_3d_tiled_strided_gfx950(&b, &s, arch)
                            : rocke_build_unified_attention_3d_tiled_gfx950(&b, &s, arch));
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
