/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gemm_multi_d_emit.c -- C-side emitter for the gemm_multi_d
 * parity harness. Selects one of the sampled GemmMultiDSpec configs by argv[1]
 * (the config index), builds the spec via the task-mandated facade
 * ckc_gemm_multi_d_spec_new(), lowers via ckc_build_gemm_multi_d(spec, arch)
 * then ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO), and prints the .ll
 * to stdout so the output can be byte-compared with the Python emitter
 * gemm_multi_d_emit.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/instance_gemm_multi_d.h"
#include "ckc/instance_gemm_universal.h"
#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"

/* Build the base UniversalGemmSpec for config `idx` (tile geometry + cshuffle
 * epilogue + dtype). Warp-tile knobs and DataSpec are left at the Python
 * dataclass defaults (warp_tile 32x32x16, RCR fp16). */
static int make_base(int idx, ckc_gemm_universal_spec_t* base)
{
    *base = ckc_gemm_universal_spec_default();
    base->trait.epilogue = "cshuffle";
    switch(idx)
    {
    case 0: /* gemmmd1: TileSpec(128,128,64, 2,2), DataSpec(dtype_a='fp16') */
        base->name = "gemmmd1";
        base->tile.tile_m = 128;
        base->tile.tile_n = 128;
        base->tile.tile_k = 64;
        base->tile.warp_m = 2;
        base->tile.warp_n = 2;
        base->data.dtype_a = "fp16";
        break;
    case 1: /* gemmmd2: TileSpec(64,64,64, 1,2) */
        base->name = "gemmmd2";
        base->tile.tile_m = 64;
        base->tile.tile_n = 64;
        base->tile.tile_k = 64;
        base->tile.warp_m = 1;
        base->tile.warp_n = 2;
        break;
    case 2: /* gemmmd3: TileSpec(256,128,128, 4,2) */
        base->name = "gemmmd3";
        base->tile.tile_m = 256;
        base->tile.tile_n = 128;
        base->tile.tile_k = 128;
        base->tile.warp_m = 4;
        base->tile.warp_n = 2;
        break;
    case 3: /* gemmmd4: TileSpec(192,192,64, 2,2) */
        base->name = "gemmmd4";
        base->tile.tile_m = 192;
        base->tile.tile_n = 192;
        base->tile.tile_k = 64;
        base->tile.warp_m = 2;
        base->tile.warp_n = 2;
        break;
    case 4: /* gemmmd5: TileSpec(128,128,128, 2,2) */
        base->name = "gemmmd5";
        base->tile.tile_m = 128;
        base->tile.tile_n = 128;
        base->tile.tile_k = 128;
        base->tile.warp_m = 2;
        base->tile.warp_n = 2;
        break;
    case 5: /* gemmmd6: TileSpec(256,256,32, 4,4) */
        base->name = "gemmmd6";
        base->tile.tile_m = 256;
        base->tile.tile_n = 256;
        base->tile.tile_k = 32;
        base->tile.warp_m = 4;
        base->tile.warp_n = 4;
        break;
    default:
        return -1;
    }
    ckc_gemm_universal_spec_finalize(base);
    return 0;
}

/* D-operand chain + load-kind for config `idx`. Returns num_d, or -1. */
static int make_d_ops(int idx, ckc_gemm_multi_d_operand_t* ops, const char** load_kind)
{
    switch(idx)
    {
    case 0:
        ops[0] = (ckc_gemm_multi_d_operand_t){"D0", "add"};
        *load_kind = "vector";
        return 1;
    case 1:
        ops[0] = (ckc_gemm_multi_d_operand_t){"D0", "add"};
        ops[1] = (ckc_gemm_multi_d_operand_t){"D1", "mul"};
        *load_kind = "tiled";
        return 2;
    case 2:
        ops[0] = (ckc_gemm_multi_d_operand_t){"D0", "add"};
        ops[1] = (ckc_gemm_multi_d_operand_t){"D1", "add"};
        ops[2] = (ckc_gemm_multi_d_operand_t){"D2", "mul"};
        *load_kind = "vector";
        return 3;
    case 3:
        ops[0] = (ckc_gemm_multi_d_operand_t){"D0", "mul"};
        ops[1] = (ckc_gemm_multi_d_operand_t){"D1", "add"};
        *load_kind = "vector";
        return 2;
    case 4:
        ops[0] = (ckc_gemm_multi_d_operand_t){"D0", "add"};
        ops[1] = (ckc_gemm_multi_d_operand_t){"D1", "mul"};
        ops[2] = (ckc_gemm_multi_d_operand_t){"D2", "add"};
        ops[3] = (ckc_gemm_multi_d_operand_t){"D3", "mul"};
        *load_kind = "tiled";
        return 4;
    case 5:
        ops[0] = (ckc_gemm_multi_d_operand_t){"D0", "add"};
        *load_kind = "stock";
        return 1;
    default:
        return -1;
    }
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

    ckc_gemm_universal_spec_t base;
    if(make_base(idx, &base) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_gemm_multi_d_operand_t ops[CKC_GEMM_MULTI_D_MAX_D];
    const char* load_kind = "vector";
    int num_d = make_d_ops(idx, ops, &load_kind);
    if(num_d < 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_arena_t arena;
    if(ckc_arena_init(&arena, 0) != 0)
    {
        fprintf(stderr, "arena init failed\n");
        return 1;
    }

    /* name=NULL -> the GemmMultiDSpec default "ck_dsl_gemm_multi_d", matching
     * the Python emitter which constructs GemmMultiDSpec without a name= kwarg.
     * (The per-config "gemmmd<N>" string is the *base* UniversalGemmSpec name,
     * already set in make_base, not the multi-D spec name.) */
    ckc_gemm_multi_d_spec_t* spec
        = ckc_gemm_multi_d_spec_new(&arena, &base, ops, num_d, "fp16", NULL, load_kind);
    if(spec == NULL)
    {
        fprintf(stderr, "spec_new failed for config %d\n", idx);
        ckc_arena_destroy(&arena);
        return 1;
    }

    ckc_kernel_def_t* kernel = ckc_build_gemm_multi_d(spec, "gfx950");
    if(kernel == NULL)
    {
        fprintf(stderr, "build failed for config %d\n", idx);
        ckc_arena_destroy(&arena);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[CKC_ERR_MSG_CAP];
        err[0] = 0;
        ckc_status_t st = ckc_lower_kernel_to_llvm_ex(
            kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
        if(st != CKC_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
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
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if(st != CKC_OK || !t)
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
        ckc_diag_t* d = NULL;
        size_t n = 0;
        ckc_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = ckc_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        ckc_diags_free(d, n);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        ret = 2;
    }
    ckc_gemm_multi_d_kernel_free(kernel);
    ckc_arena_destroy(&arena);
    return ret;
}
