/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/mfma_gemm_emit.c -- C-side emitter for the MFMA-GEMM instance
 * parity harness. Selects one of the sampled (M,N,K,dtype,tile,kpack) configs
 * by argv[1] (the config index), builds rocke_mfma_gemm_spec_t identically to the
 * Python emitter mfma_gemm_emit.py, validates via
 * rocke_mfma_gemm_is_valid_spec, builds into a fresh IRBuilder via
 * rocke_build_mfma_gemm (the C build entry), lowers via rocke_lower_kernel_to_llvm
 * (arch gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs
 * can be byte-compared.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_mfma_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_mfma_gemm_spec_t* spec)
{
    *spec = rocke_mfma_gemm_spec_default();

    switch(idx)
    {
    case 0:
        spec->M = 256;
        spec->N = 256;
        spec->K = 256;
        spec->dtype = "f16";
        spec->tile_m = 16;
        spec->tile_n = 16;
        spec->kpack = true;
        break;
    case 1:
        spec->M = 512;
        spec->N = 512;
        spec->K = 256;
        spec->dtype = "f16";
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->kpack = true;
        break;
    case 2:
        spec->M = 256;
        spec->N = 256;
        spec->K = 512;
        spec->dtype = "bf16";
        spec->tile_m = 16;
        spec->tile_n = 16;
        spec->kpack = true;
        break;
    case 3:
        spec->M = 256;
        spec->N = 256;
        spec->K = 256;
        spec->dtype = "f16";
        spec->tile_m = 16;
        spec->tile_n = 16;
        spec->kpack = false;
        break;
    case 4:
        spec->M = 512;
        spec->N = 512;
        spec->K = 512;
        spec->dtype = "f16";
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->kpack = false;
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
        fprintf(stderr, "usage: %s <config_index 0..4> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_mfma_gemm_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char* arch = "gfx950";

    /* 2. Validate the spec (mirrors is_valid_spec). */
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = 0;
    if(!rocke_mfma_gemm_is_valid_spec(&spec, arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* 3+4. Init IRBuilder with spec.kernel_name() and build into it. */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_mfma_gemm_new(&b, &spec, arch);
    if(!kernel)
    {
        fprintf(stderr, "build failed: %s\n", b.err);
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        /* lower_kernel_to_llvm(kernel, arch='gfx950'). */
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    { /* verify */
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

    rocke_ir_builder_free(&b);
    return 0;
}
