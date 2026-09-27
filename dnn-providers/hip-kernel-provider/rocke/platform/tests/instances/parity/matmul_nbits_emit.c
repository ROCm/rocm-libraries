/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/matmul_nbits_emit.c -- C-side emitter for the MatMulNBits parity
 * harness. Selects one of 6 configs by argv[1] (config index 0..5), builds the
 * rocke_matmul_nbits_spec_t identically to the Python emitter matmul_nbits_emit.py,
 * dispatches+builds via rocke_build_matmul_nbits (after initialising the builder
 * with spec.kernel_name(), exactly as Python's IRBuilder(spec.kernel_name())),
 * lowers via rocke_lower_kernel_to_llvm (arch gfx1201, flavor AUTO) and prints the
 * .ll to stdout so the two outputs can be byte-compared. gfx1201 is one of the
 * matmul_nbits SUPPORTED_ARCHES (gfx1151/gfx1201); gfx950 is rejected by the
 * validator on both sides, so it must NOT be used here.
 *
 * On a validation reject (or any other build/lower failure) nothing is written
 * to stdout and the program exits non-zero; the harness treats a both-sides
 * reject (empty stdout + nonzero exit) as parity.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_matmul_nbits.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_matmul_nbits_spec_t* spec)
{
    *spec = rocke_matmul_nbits_spec_default();

    switch(idx)
    {
    case 0:
        spec->name = "matmul_nbits_gfx950";
        spec->N = 4096;
        spec->K = 4096;
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 128,
                                              .tile_k = 16,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "large_n";
        spec->optimized = false;
        break;
    case 1:
        spec->name = "matmul_nbits_gfx950_skinny";
        spec->N = 32;
        spec->K = 4096;
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 32,
                                              .tile_k = 16,
                                              .warp_m = 2,
                                              .warp_n = 1,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "skinny_n";
        spec->optimized = false;
        break;
    case 2:
        spec->name = "matmul_nbits_gfx950_gemv";
        spec->N = 248320;
        spec->K = 4096;
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 1,
                                              .tile_n = 256,
                                              .tile_k = 16,
                                              .warp_m = 1,
                                              .warp_n = 8,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "decode_gemv";
        spec->optimized = false;
        break;
    case 3:
        spec->name = "matmul_nbits_gfx950_large_8k";
        spec->N = 8192;
        spec->K = 4096;
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 128,
                                              .tile_k = 16,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp32";
        spec->family = "large_n";
        spec->optimized = false;
        break;
    case 4:
        spec->name = "matmul_nbits_gfx950_opt";
        spec->N = 4096;
        spec->K = 4096;
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "large_n";
        spec->optimized = true;
        break;
    case 5:
        spec->name = "matmul_nbits_gfx950_12k";
        spec->N = 12288;
        spec->K = 4096;
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 128,
                                              .tile_k = 16,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->group_size = 32;
        spec->scale_dtype = "fp16";
        spec->family = "large_n";
        spec->optimized = false;
        break;
    default:
        return -1;
    }
    rocke_matmul_nbits_spec_finalize(spec);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..5> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_matmul_nbits_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    /* Mirror Python build_matmul_nbits: IRBuilder(spec.kernel_name()) then
     * dispatch via build_matmul_nbits(spec, arch). */
    char kname[256];
    if(rocke_matmul_nbits_kernel_name(&spec, kname, sizeof kname) != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, kname) != ROCKE_OK)
    {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_build_matmul_nbits(&b, &spec, "gfx1201");
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx1201", &llvm_text);
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
