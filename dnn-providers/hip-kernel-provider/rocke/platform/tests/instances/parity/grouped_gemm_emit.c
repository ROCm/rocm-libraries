/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/grouped_gemm_emit.c -- C-side emitter for the grouped-GEMM
 * parity harness. Selects one of the sampled GroupedGemmSpec configs by
 * argv[1] (the config index), builds rocke_grouped_gemm_spec_t identically to
 * the Python emitter grouped_gemm_emit.py, builds via the C build entry
 * (rocke_build_grouped_gemm) and lowers via rocke_lower_kernel_to_llvm (arch
 * gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared. Uses the convenience one-call lower path
 * rocke_grouped_gemm_lower_to_llvm which owns its own IRBuilder.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_grouped_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_grouped_gemm_spec_t* spec)
{
    *spec = rocke_grouped_gemm_spec_default();

    switch(idx)
    {
    case 0:
        spec->name = "ggemm_fp16_m128n128k32";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    case 1:
        spec->name = "ggemm_bf16_m32n32k32";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 32,
                                              .tile_n = 32,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "cshuffle";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->dtype = "bf16";
        spec->wave_size = 64;
        break;
    case 2:
        spec->name = "ggemm_fp16_m64n64k64";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 64,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    case 3:
        spec->name = "ggemm_fp16_m256n256k128";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 128,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.chiplet_swizzle = true;
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    default:
        return -1;
    }
    rocke_grouped_gemm_spec_finalize(spec);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..3>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_grouped_gemm_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_grouped_gemm_lower_to_llvm(
            &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        rocke_ir_builder_t b;
        rocke_kernel_def_t* kernel = rocke_build_grouped_gemm_new(&b, &spec, "gfx950");
        if(!kernel)
        {
            fprintf(stderr, "build failed: err=%s\n", rocke_ir_builder_error(&b));
            rocke_ir_builder_free(&b);
            return 1;
        }
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
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
        rocke_kernel_def_t* kernel = rocke_build_grouped_gemm_new(&b, &spec, "gfx950");
        if(!kernel)
        {
            fprintf(stderr, "build failed: err=%s\n", rocke_ir_builder_error(&b));
            rocke_ir_builder_free(&b);
            return 1;
        }
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
        rocke_ir_builder_free(&b);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    return 0;
}
