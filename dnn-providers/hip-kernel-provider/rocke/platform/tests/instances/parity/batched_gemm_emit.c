/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/batched_gemm_emit.c -- C-side emitter for the batched-GEMM
 * parity harness. Selects one of 5 sampled BatchedGemmSpec configs by argv[1]
 * (0..4), builds rocke_batched_gemm_spec_t identically to the Python emitter
 * batched_gemm_emit.py, lowers via rocke_batched_gemm_lower_to_llvm (arch
 * gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_batched_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_batched_gemm_spec_t* spec)
{
    *spec = rocke_batched_gemm_spec_default();

    switch(idx)
    {
    case 0: /* bgm_64x64x32_2x2 */
        spec->name = "bgm_64x64x32_2x2";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 64,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    case 1: /* bgm_128x128x32_2x2_cshuffle */
        spec->name = "bgm_128x128x32_2x2_cshuffle";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "cshuffle";
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    case 2: /* bgm_256x256x64_4x4 */
        spec->name = "bgm_256x256x64_4x4";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    case 3: /* bgm_128x128x64_2x2_wide */
        spec->name = "bgm_128x128x64_2x2_wide";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    case 4: /* bgm_64x128x32_1x2_skip */
        spec->name = "bgm_64x128x32_1x2_skip";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 1,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->trait.active_tile_skip = true;
        spec->dtype = "fp16";
        spec->wave_size = 64;
        break;
    default:
        return -1;
    }
    rocke_batched_gemm_spec_finalize(spec);
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

    rocke_batched_gemm_spec_t spec;
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
        rocke_status_t st = rocke_batched_gemm_lower_to_llvm(
            &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
        return 0;
    }

    /* ir / verify modes: need the kernel object. */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_batched_gemm_new(&b, &spec, "gfx950");
    if(!kernel)
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ir") == 0)
    {
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
