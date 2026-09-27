/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * grouped_gemm_stress_emit.c -- C-side emitter for the WIDE adversarial
 * grouped-GEMM parity stress test. Mirrors grouped_gemm_stress_emit.py config
 * table 1:1. Selects config by argv[1], builds the spec identically, lowers via
 * rocke_grouped_gemm_lower_to_llvm(arch gfx950, flavor AUTO) and prints the .ll.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_grouped_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Helper: set tile fields compactly. */
#define TILE(tm, tn, tk, wm, wn, wk, wtm, wtn, wtk)                                      \
    (rocke_gemm_tile_spec_t)                                                             \
    {                                                                                    \
        .tile_m = (tm), .tile_n = (tn), .tile_k = (tk), .warp_m = (wm), .warp_n = (wn),  \
        .warp_k = (wk), .warp_tile_m = (wtm), .warp_tile_n = (wtn), .warp_tile_k = (wtk) \
    }

static int make_spec(int idx, rocke_grouped_gemm_spec_t* spec)
{
    *spec = rocke_grouped_gemm_spec_default();
    spec->wave_size = 64;

    switch(idx)
    {
    case 0:
        spec->name = "g_fp16_m128n128k32";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 1:
        spec->name = "g_bf16_m32n32k32";
        spec->dtype = "bf16";
        spec->tile = TILE(32, 32, 32, 2, 2, 1, 16, 16, 32);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "cshuffle";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        break;
    case 2:
        spec->name = "g_fp16_m64n64k64";
        spec->dtype = "fp16";
        spec->tile = TILE(64, 64, 64, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        break;
    case 3:
        spec->name = "g_fp16_m256n256k128";
        spec->dtype = "fp16";
        spec->tile = TILE(256, 256, 128, 4, 4, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.chiplet_swizzle = true;
        break;
    case 4:
        spec->name = "g_fp16_smallest_16atom";
        spec->dtype = "fp16";
        spec->tile = TILE(16, 16, 16, 1, 1, 1, 16, 16, 16);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        break;
    case 5:
        spec->name = "g_fp16_smallest_32atom";
        spec->dtype = "fp16";
        spec->tile = TILE(32, 32, 8, 1, 1, 1, 32, 32, 8);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        break;
    case 6:
        spec->name = "g_bf16_smallest_16x16x16";
        spec->dtype = "bf16";
        spec->tile = TILE(16, 16, 16, 1, 1, 1, 16, 16, 16);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        break;
    case 7:
        spec->name = "g_bf16_smallest_16x16x32";
        spec->dtype = "bf16";
        spec->tile = TILE(16, 16, 32, 1, 1, 1, 16, 16, 32);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        break;
    case 8:
        spec->name = "g_fp16_atom16x16x16";
        spec->dtype = "fp16";
        spec->tile = TILE(64, 64, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 9:
        spec->name = "g_fp16_atom16x16x32";
        spec->dtype = "fp16";
        spec->tile = TILE(64, 64, 64, 2, 2, 1, 16, 16, 32);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 10:
        spec->name = "g_fp16_atom32x32x8";
        spec->dtype = "fp16";
        spec->tile = TILE(64, 64, 16, 1, 1, 1, 32, 32, 8);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 11:
        spec->name = "g_fp16_atom32x32x16";
        spec->dtype = "fp16";
        spec->tile = TILE(64, 64, 32, 1, 1, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 12:
        spec->name = "g_fp16_pipe_mem";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        break;
    case 13:
        spec->name = "g_fp16_pipe_compv3";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        break;
    case 14:
        spec->name = "g_fp16_pipe_compv4_dir";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        break;
    case 15:
        spec->name = "g_fp16_pipe_wsp3";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "wsp3";
        spec->trait.epilogue = "default";
        break;
    case 16:
        spec->name = "g_fp16_interwave";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.scheduler = "interwave";
        break;
    case 17:
        spec->name = "g_fp16_intrawave_explicit";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.scheduler = "intrawave";
        break;
    case 18:
        spec->name = "g_fp16_compv3_cshuffle";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "cshuffle";
        break;
    case 19:
        spec->name = "g_fp16_pad_k";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.pad_k = true;
        break;
    case 20:
        spec->name = "g_fp16_pad_all";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.pad_k = true;
        break;
    case 21:
        spec->name = "g_bf16_pad_mem_default";
        spec->dtype = "bf16";
        spec->tile = TILE(64, 64, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.pad_k = true;
        break;
    case 22:
        spec->name = "g_fp16_persistent";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.persistent = true;
        break;
    case 23:
        spec->name = "g_fp16_chiplet_wgm4";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 4;
        break;
    case 24:
        spec->name = "g_fp16_chiplet_xcd4_chunk32";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_num_xcds = 4;
        spec->trait.chiplet_chunk_size = 32;
        break;
    case 25:
        spec->name = "g_fp16_waves_per_eu2";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.waves_per_eu_set = true;
        spec->trait.waves_per_eu = 2;
        break;
    case 26:
        spec->name = "g_fp16_lds_swizzle";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.lds_swizzle = true;
        break;
    case 27:
        spec->name = "g_fp16_lds_k_pad8";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.lds_k_pad = 8;
        break;
    case 28:
        spec->name = "g_fp16_active_tile_skip";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.active_tile_skip = true;
        break;
    case 29:
        spec->name = "g_fp16_deepk_256";
        spec->dtype = "fp16";
        spec->tile = TILE(64, 64, 256, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 30:
        spec->name = "g_fp16_deepk_512";
        spec->dtype = "fp16";
        spec->tile = TILE(64, 64, 512, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        break;
    case 31:
        spec->name = "g_bf16_deepk_256_16x16x32";
        spec->dtype = "bf16";
        spec->tile = TILE(32, 32, 256, 2, 2, 1, 16, 16, 32);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 32:
        spec->name = "g_fp16_wide_m256n128";
        spec->dtype = "fp16";
        spec->tile = TILE(256, 128, 32, 4, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 33:
        spec->name = "g_fp16_tall_m128n256";
        spec->dtype = "fp16";
        spec->tile = TILE(128, 256, 32, 2, 4, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 34:
        spec->name = "g_fp16_big_warpcount_8x8";
        spec->dtype = "fp16";
        spec->tile = TILE(256, 256, 32, 8, 8, 1, 16, 16, 16);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        break;
    case 35:
        spec->name = "g_fp16_asym_warp4x1";
        spec->dtype = "fp16";
        spec->tile = TILE(256, 32, 32, 4, 1, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 36:
        spec->name = "g_fp16_asym_warp1x4";
        spec->dtype = "fp16";
        spec->tile = TILE(32, 256, 32, 1, 4, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 37:
        spec->name = "g_fp16_multi_mfma_4x4";
        spec->dtype = "fp16";
        spec->tile = TILE(256, 256, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 38:
        spec->name = "g_f16alias_m128";
        spec->dtype = "f16";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        break;
    case 39:
        spec->name = "g_bf16_compv3_default";
        spec->dtype = "bf16";
        spec->tile = TILE(64, 64, 64, 2, 2, 1, 16, 16, 32);
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
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
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_grouped_gemm_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_grouped_gemm_new(&b, &spec, "gfx950");
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
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
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
