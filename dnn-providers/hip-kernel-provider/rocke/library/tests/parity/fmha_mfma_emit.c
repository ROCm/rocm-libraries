/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_mfma_emit.c -- C-side emitter for the tiled FMHA-forward
 * (fmha_mfma) instance parity harness. Selects one of the sampled
 * (head_size,heads,seqlen,mask) configs by argv[1] (the config index), builds
 * rocke_fmha_mfma_spec_t identically to the Python emitter fmha_mfma_emit.py,
 * builds the kernel via rocke_build_fmha_fwd_mfma (the C build entry, arch
 * gfx950), lowers via rocke_lower_kernel_to_llvm (flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_fmha_mfma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_fmha_mfma_spec_t* spec)
{
    *spec = rocke_fmha_mfma_spec_default();

    switch(idx)
    {
    case 0:
        spec->head_size = 64;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 8;
        spec->seqlen_q = 256;
        spec->seqlen_k = 256;
        spec->dtype = "f16";
        spec->mask_mode = ROCKE_FMHA_MASK_NONE;
        spec->sliding_window = 0;
        spec->scale_log2 = 0.0;
        break;
    case 1:
        spec->head_size = 128;
        spec->num_query_heads = 16;
        spec->num_kv_heads = 16;
        spec->seqlen_q = 512;
        spec->seqlen_k = 512;
        spec->dtype = "f16";
        spec->mask_mode = ROCKE_FMHA_MASK_NONE;
        spec->sliding_window = 0;
        spec->scale_log2 = 0.0;
        break;
    case 2:
        spec->head_size = 64;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 8;
        spec->seqlen_q = 256;
        spec->seqlen_k = 1024;
        spec->dtype = "f16";
        spec->mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        spec->sliding_window = 0;
        spec->scale_log2 = 0.0;
        break;
    case 3:
        spec->head_size = 256;
        spec->num_query_heads = 32;
        spec->num_kv_heads = 32;
        spec->seqlen_q = 512;
        spec->seqlen_k = 2048;
        spec->dtype = "f16";
        spec->mask_mode = ROCKE_FMHA_MASK_SLIDING_WINDOW;
        spec->sliding_window = 512;
        spec->scale_log2 = 0.0;
        break;
    case 4:
        spec->head_size = 192;
        spec->num_query_heads = 12;
        spec->num_kv_heads = 12;
        spec->seqlen_q = 128;
        spec->seqlen_k = 512;
        spec->dtype = "f16";
        spec->mask_mode = ROCKE_FMHA_MASK_NONE;
        spec->sliding_window = 0;
        spec->scale_log2 = 0.0;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..4>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_fmha_mfma_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char* arch = "gfx950";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[256];
    reason[0] = 0;
    if(!rocke_fmha_mfma_is_valid_spec(&spec, arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* build_fmha_fwd_mfma(builder, spec, "gfx950"). The returned kernel is owned
     * by the build entry's internal builder and stays valid for an immediate
     * same-scope lower. */
    rocke_kernel_def_t* kernel = rocke_build_fmha_fwd_mfma(NULL, &spec, arch);
    if(!kernel)
    {
        fprintf(stderr, "build failed\n");
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        /* lower_kernel_to_llvm(kernel, arch='gfx950', flavor=AUTO). */
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else if(strcmp(mode, "verify") == 0)
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
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    return 0;
}
