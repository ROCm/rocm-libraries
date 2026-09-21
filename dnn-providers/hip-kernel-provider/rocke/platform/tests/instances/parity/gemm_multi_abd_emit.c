/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gemm_multi_abd_emit.c -- C-side emitter for the multi-A/B/D
 * GEMM parity harness. Selects one of 6 sampled configs by argv[1] (the config
 * index 0..5), builds rocke_gemm_multi_abd_spec_t identically to the Python
 * emitter gemm_multi_abd_emit.py, builds the kernel via
 * rocke_build_gemm_multi_abd_new (the C build entry) and lowers it via
 * rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO), printing the .ll to
 * stdout so the two outputs can be byte-compared.
 *
 * Mirrors the Python defaults: each config sets only tile_m/n/k + warp_m/n and
 * the trait/data fields; warp_tile_m/n/k default to 32/32/16, warp_k to 1, and
 * block_size is derived at finalize() (warp_m*warp_n*warp_k*wave_size). All
 * configs use num_a == num_b == 1 (a single (A,fp16)/(B,fp16)) -- the v1 path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/instance_gemm_multi_abd.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Single-element A/B operand pools shared by every config (Python default
 * (("A","fp16"),) / (("B","fp16"),)). */
static const rocke_gemm_abd_a_operand_t kA[] = {{"A", "fp16"}};
static const rocke_gemm_abd_b_operand_t kB[] = {{"B", "fp16"}};

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_gemm_multi_abd_spec_t* spec)
{
    *spec = rocke_gemm_multi_abd_spec_default();
    spec->base = rocke_gemm_universal_spec_default();
    spec->base.name = "test";
    spec->a_operands = kA;
    spec->num_a_operands = 1;
    spec->b_operands = kB;
    spec->num_b_operands = 1;
    spec->d_dtype = "fp16";
    /* name field defaults to "rocke_gemm_multi_abd" (Python dataclass default);
     * the spec_default() above already set it. */

    switch(idx)
    {
    case 0:
        spec->base.tile.tile_m = 128;
        spec->base.tile.tile_n = 128;
        spec->base.tile.tile_k = 32;
        spec->base.tile.warp_m = 2;
        spec->base.tile.warp_n = 2;
        spec->base.trait.pipeline = "compv4";
        spec->base.trait.scheduler = "intrawave";
        spec->base.trait.epilogue = "cshuffle";
        spec->base.data.dtype_a = "fp16";
        spec->base.data.dtype_b = "fp16";
        spec->base.data.dtype_c = "fp16";
        spec->base.data.dtype_acc = "fp32";
        spec->num_d_operands = 0;
        spec->d_load_kind = ROCKE_D_LOAD_VECTOR;
        break;
    case 1:
        spec->base.tile.tile_m = 64;
        spec->base.tile.tile_n = 64;
        spec->base.tile.tile_k = 32;
        spec->base.tile.warp_m = 1;
        spec->base.tile.warp_n = 1;
        spec->base.trait.pipeline = "compv4";
        spec->base.trait.scheduler = "intrawave";
        spec->base.trait.epilogue = "cshuffle";
        spec->base.data.dtype_a = "fp16";
        spec->base.data.dtype_b = "fp16";
        spec->base.data.dtype_c = "fp16";
        spec->base.data.dtype_acc = "fp32";
        spec->d_operands[0] = (rocke_gemm_multi_d_op_t){"D0", false}; /* add */
        spec->num_d_operands = 1;
        spec->d_load_kind = ROCKE_D_LOAD_VECTOR;
        break;
    case 2:
        spec->base.tile.tile_m = 256;
        spec->base.tile.tile_n = 128;
        spec->base.tile.tile_k = 64;
        spec->base.tile.warp_m = 4;
        spec->base.tile.warp_n = 2;
        spec->base.trait.pipeline = "compv3";
        spec->base.trait.scheduler = "intrawave";
        spec->base.trait.epilogue = "cshuffle";
        spec->base.data.dtype_a = "fp16";
        spec->base.data.dtype_b = "fp16";
        spec->base.data.dtype_c = "fp16";
        spec->base.data.dtype_acc = "fp32";
        spec->d_operands[0] = (rocke_gemm_multi_d_op_t){"D0", false}; /* add */
        spec->d_operands[1] = (rocke_gemm_multi_d_op_t){"D1", true}; /* mul */
        spec->num_d_operands = 2;
        spec->d_load_kind = ROCKE_D_LOAD_TILED;
        break;
    case 3:
        spec->base.tile.tile_m = 64;
        spec->base.tile.tile_n = 128;
        spec->base.tile.tile_k = 32;
        spec->base.tile.warp_m = 1;
        spec->base.tile.warp_n = 2;
        spec->base.trait.pipeline = "mem";
        spec->base.trait.scheduler = "intrawave";
        spec->base.trait.epilogue = "cshuffle";
        spec->base.data.dtype_a = "fp16";
        spec->base.data.dtype_b = "fp16";
        spec->base.data.dtype_c = "fp16";
        spec->base.data.dtype_acc = "fp32";
        spec->d_operands[0] = (rocke_gemm_multi_d_op_t){"D0", false}; /* add */
        spec->num_d_operands = 1;
        spec->d_load_kind = ROCKE_D_LOAD_STOCK;
        break;
    case 4:
        spec->base.tile.tile_m = 192;
        spec->base.tile.tile_n = 192;
        spec->base.tile.tile_k = 64;
        spec->base.tile.warp_m = 3;
        spec->base.tile.warp_n = 3;
        spec->base.trait.pipeline = "compv4";
        spec->base.trait.scheduler = "interwave";
        spec->base.trait.epilogue = "cshuffle";
        spec->base.data.dtype_a = "fp16";
        spec->base.data.dtype_b = "fp16";
        spec->base.data.dtype_c = "fp16";
        spec->base.data.dtype_acc = "fp32";
        spec->d_operands[0] = (rocke_gemm_multi_d_op_t){"D0", true}; /* mul */
        spec->num_d_operands = 1;
        spec->d_load_kind = ROCKE_D_LOAD_VECTOR;
        break;
    case 5:
        spec->base.tile.tile_m = 128;
        spec->base.tile.tile_n = 64;
        spec->base.tile.tile_k = 16;
        spec->base.tile.warp_m = 2;
        spec->base.tile.warp_n = 1;
        spec->base.trait.pipeline = "compv4";
        spec->base.trait.scheduler = "intrawave";
        spec->base.trait.epilogue = "cshuffle";
        spec->base.data.dtype_a = "fp16";
        spec->base.data.dtype_b = "fp16";
        spec->base.data.dtype_c = "fp16";
        spec->base.data.dtype_acc = "fp32";
        spec->d_operands[0] = (rocke_gemm_multi_d_op_t){"D0", false}; /* add */
        spec->d_operands[1] = (rocke_gemm_multi_d_op_t){"D1", false}; /* add */
        spec->d_operands[2] = (rocke_gemm_multi_d_op_t){"D2", true}; /* mul */
        spec->num_d_operands = 3;
        spec->d_load_kind = ROCKE_D_LOAD_VECTOR;
        break;
    default:
        return -1;
    }
    /* __post_init__ / _init_block_size(): derive block_size from the warp grid
     * (block_size left at 0 by spec_default). */
    rocke_gemm_universal_spec_finalize(&spec->base);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_gemm_multi_abd_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    /* Build the kernel via the C build entry (init builder with kernel_name()),
     * then dispatch on mode. */
    rocke_ir_builder_t b;
    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        fprintf(stderr, "arena init failed\n");
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_build_gemm_multi_abd_new(&b, &arena, &spec, "gfx950");
    if(kernel == NULL)
    {
        rocke_status_t st = rocke_ir_builder_status(&b);
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: status=%d err=%s\n", (int)st, m ? m : "(none)");
        rocke_ir_builder_free(&b);
        rocke_arena_destroy(&arena);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
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
    rocke_arena_destroy(&arena);
    return ret;
}
