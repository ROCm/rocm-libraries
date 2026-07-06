/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/sparse_attention_emit.c -- C-side emitter for the sparse-attention
 * forward parity harness. Selects one of 6 sampled configs by argv[1] (the config
 * index 0..5), builds either a rocke_jenga_sparse_spec_t (via
 * rocke_build_jenga_sparse_attention) or a rocke_vsa_sparse_spec_t (via
 * rocke_build_vsa_sparse_attention) identically to the Python emitter
 * sparse_attention_emit.py, lowers the returned KernelDef via
 * rocke_lower_kernel_to_llvm_ex (arch gfx950, flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 *
 * Optional argv[2] = mode: "ll" (default), "ir", "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/instance_sparse_attention.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Build the KernelDef for config index `idx`. Returns NULL on unknown idx or
 * build failure. */
static rocke_kernel_def_t* make_kernel(int idx)
{
    rocke_fmha_shape_t shape;
    rocke_fmha_common_spec_t common;

    switch(idx)
    {
    case 0:
    {
        shape = rocke_fmha_shape_default(64, 8, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        rocke_jenga_sparse_spec_t spec
            = rocke_jenga_sparse_spec_default(common, /*seqlen_q*/ 32, /*seqlen_k*/ 128);
        spec.block_q = 1;
        spec.block_k = 64;
        return rocke_build_jenga_sparse_attention(NULL, &spec, "gfx950");
    }
    case 1:
    {
        shape = rocke_fmha_shape_default(128, 16, 16);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        rocke_jenga_sparse_spec_t spec
            = rocke_jenga_sparse_spec_default(common, /*seqlen_q*/ 64, /*seqlen_k*/ 256);
        spec.block_q = 2;
        spec.block_k = 64;
        return rocke_build_jenga_sparse_attention(NULL, &spec, "gfx950");
    }
    case 2:
    {
        shape = rocke_fmha_shape_default(64, 8, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        rocke_vsa_sparse_spec_t spec
            = rocke_vsa_sparse_spec_default(common, /*seqlen_q*/ 32, /*seqlen_k*/ 128);
        spec.block_q = 1;
        spec.block_k = 64;
        spec.max_blocks_per_q = 16;
        return rocke_build_vsa_sparse_attention(NULL, &spec, "gfx950");
    }
    case 3:
    {
        shape = rocke_fmha_shape_default(128, 16, 16);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        rocke_vsa_sparse_spec_t spec
            = rocke_vsa_sparse_spec_default(common, /*seqlen_q*/ 64, /*seqlen_k*/ 256);
        spec.block_q = 2;
        spec.block_k = 64;
        spec.max_blocks_per_q = 32;
        spec.use_wave_ballot_scatter = true;
        return rocke_build_vsa_sparse_attention(NULL, &spec, "gfx950");
    }
    case 4:
    {
        shape = rocke_fmha_shape_default(256, 32, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        rocke_jenga_sparse_spec_t spec
            = rocke_jenga_sparse_spec_default(common, /*seqlen_q*/ 96, /*seqlen_k*/ 512);
        spec.block_q = 4;
        spec.block_k = 128;
        return rocke_build_jenga_sparse_attention(NULL, &spec, "gfx950");
    }
    case 5:
    {
        shape = rocke_fmha_shape_default(256, 32, 32);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        rocke_vsa_sparse_spec_t spec
            = rocke_vsa_sparse_spec_default(common, /*seqlen_q*/ 128, /*seqlen_k*/ 1024);
        spec.block_q = 8;
        spec.block_k = 64;
        spec.max_blocks_per_q = 24;
        spec.use_wave_ballot_scatter = false;
        return rocke_build_vsa_sparse_attention(NULL, &spec, "gfx950");
    }
    default:
        return NULL;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..5> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_kernel_def_t* kernel = make_kernel(idx);
    if(!kernel)
    {
        fprintf(stderr, "build failed / unknown config index %d\n", idx);
        return 1;
    }

    if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
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
        /* mode == "ll" */
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    return 0;
}
