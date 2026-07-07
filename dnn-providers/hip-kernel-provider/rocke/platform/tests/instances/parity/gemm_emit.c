/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gemm_emit.c -- C-side emitter for the universal-GEMM parity
 * harness. Selects one of 7 sampled (spec,knobs) configs by argv[1] (the
 * config index 0..6), builds rocke_gemm_universal_spec_t identically to the
 * Python emitter gemm_emit.py, lowers via rocke_gemm_universal_lower_to_llvm
 * (arch gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs
 * can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_gemm_universal_spec_t* spec)
{
    *spec = rocke_gemm_universal_spec_default();

    switch(idx)
    {
    case 0: /* test1 */
        spec->name = "test1";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "fp16";
        spec->data.dtype_b = "fp16";
        spec->data.dtype_c = "fp16";
        spec->data.dtype_acc = "fp32";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 1: /* test2 */
        spec->name = "test2";
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
        spec->trait.epilogue = "cshuffle";
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 2: /* test3 */
        spec->name = "test3";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 8};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 512;
        spec->batched = false;
        break;
    case 3: /* test4 */
        spec->name = "test4";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 1,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 4: /* test5 */
        spec->name = "test5";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 64,
                                              .tile_k = 64,
                                              .warp_m = 1,
                                              .warp_n = 1,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "cshuffle";
        spec->trait.chiplet_swizzle = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 64;
        spec->batched = false;
        break;
    case 5: /* test6 */
        spec->name = "test6";
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
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = true;
        break;
    case 6: /* test7 */
        spec->name = "test7";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 192,
                                              .tile_n = 192,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.lds_swizzle = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 7: /* test8 -- split-K regression: split_k=1, byte-identical to idx 0 */
        spec->name = "test8";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->trait.split_k = 1;
        spec->data.dtype_a = "fp16";
        spec->data.dtype_b = "fp16";
        spec->data.dtype_c = "fp16";
        spec->data.dtype_acc = "fp32";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 8: /* test9 -- split-K active: split_k=8, decode shape, atomic-add */
        spec->name = "test9";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 16,
                                              .tile_n = 64,
                                              .tile_k = 64,
                                              .warp_m = 1,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.split_k = 8;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 9: /* test1 shape, lowered for gfx942 (CDNA coverage of common GEMM) */
        spec->name = "test1_gfx942";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "fp16";
        spec->data.dtype_b = "fp16";
        spec->data.dtype_c = "fp16";
        spec->data.dtype_acc = "fp32";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    default:
        return -1;
    }
    rocke_gemm_universal_spec_finalize(spec);
    return 0;
}

/* Config 9 exercises gfx942; all others use the gfx950 baseline. */
static const char* arch_for(int idx)
{
    return idx == 9 ? "gfx942" : "gfx950";
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..9>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    const char* arch = arch_for(idx);

    rocke_gemm_universal_spec_t spec;
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
        rocke_status_t st = rocke_gemm_universal_lower_to_llvm(
            &spec, arch, ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
        return 0;
    }

    /* ir / verify modes need the kernel object */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_universal_gemm_new(&b, &spec, arch);
    if(!kernel || !rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(t, stdout);
            free(t);
        }
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
        ret = 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
