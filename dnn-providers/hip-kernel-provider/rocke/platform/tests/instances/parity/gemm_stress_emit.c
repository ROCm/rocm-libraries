/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gemm_stress_emit.c -- WIDE adversarial C-side emitter for the
 * universal-GEMM parity harness. Mirrors gemm_stress_emit.py config table 1:1.
 * Selects a config by argv[1], builds, lowers (gfx950, AUTO) and prints .ll.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* tile field order: tm,tn,tk, wm,wn,wk, wtm,wtn,wtk */
#define TILE(tm, tn, tk, wm, wn, wk, wtm, wtn, wtk)                                         \
    (rocke_gemm_tile_spec_t)                                                                \
    {                                                                                       \
        .tile_m = tm, .tile_n = tn, .tile_k = tk, .warp_m = wm, .warp_n = wn, .warp_k = wk, \
        .warp_tile_m = wtm, .warp_tile_n = wtn, .warp_tile_k = wtk                          \
    }

static int make_spec(int idx, rocke_gemm_universal_spec_t* spec)
{
    *spec = rocke_gemm_universal_spec_default();

    switch(idx)
    {
    case 0:
        spec->name = "s00";
        spec->tile = TILE(64, 64, 16, 1, 1, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 64;
        break;
    case 1:
        spec->name = "s01";
        spec->tile = TILE(64, 64, 32, 1, 1, 1, 16, 16, 32);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 64;
        break;
    case 2:
        spec->name = "s02";
        spec->tile = TILE(64, 64, 16, 1, 1, 1, 32, 32, 8);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 64;
        break;
    case 3:
        spec->name = "s03";
        spec->tile = TILE(64, 64, 16, 1, 1, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 64;
        break;
    case 4:
        spec->name = "s04";
        spec->tile = TILE(64, 64, 16, 1, 1, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->block_size = 64;
        break;
    case 5:
        spec->name = "s05";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 32);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->block_size = 256;
        break;
    case 6:
        spec->name = "s06";
        spec->tile = TILE(32, 64, 16, 1, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 128;
        break;
    case 7:
        spec->name = "s07";
        spec->tile = TILE(64, 32, 16, 2, 1, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 128;
        break;
    case 8:
        spec->name = "s08";
        spec->tile = TILE(64, 128, 16, 1, 4, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 9:
        spec->name = "s09";
        spec->tile = TILE(128, 64, 16, 4, 1, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 10:
        spec->name = "s10";
        spec->tile = TILE(64, 128, 16, 2, 4, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 512;
        break;
    case 11:
        spec->name = "s11";
        spec->tile = TILE(128, 64, 16, 4, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 512;
        break;
    case 12:
        spec->name = "s12";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 13:
        spec->name = "s13";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 14:
        spec->name = "s14";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "cshuffle";
        spec->block_size = 256;
        break;
    case 15:
        spec->name = "s15";
        spec->tile = TILE(256, 128, 32, 2, 4, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->block_size = 512;
        break;
    case 16:
        spec->name = "s16";
        spec->tile = TILE(192, 64, 16, 2, 1, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 128;
        break;
    case 17:
        spec->name = "s17";
        spec->tile = TILE(96, 96, 16, 1, 1, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 64;
        break;
    case 18:
        spec->name = "s18";
        spec->tile = TILE(224, 32, 16, 1, 1, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 64;
        break;
    case 19:
        spec->name = "s19";
        spec->tile = TILE(320, 64, 16, 1, 2, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 128;
        break;
    case 20:
        spec->name = "s20";
        spec->tile = TILE(128, 128, 128, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 21:
        spec->name = "s21";
        spec->tile = TILE(128, 128, 256, 2, 2, 1, 16, 16, 32);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 22:
        spec->name = "s22";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 4;
        spec->trait.chiplet_num_xcds = 4;
        spec->trait.chiplet_chunk_size = 32;
        spec->block_size = 256;
        break;
    case 23:
        spec->name = "s23";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.lds_swizzle = true;
        spec->block_size = 256;
        break;
    case 24:
        spec->name = "s24";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.lds_k_pad = 16;
        spec->block_size = 256;
        break;
    case 25:
        spec->name = "s25";
        spec->tile = TILE(128, 128, 64, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->block_size = 256;
        break;
    case 26:
        spec->name = "s26";
        spec->tile = TILE(128, 128, 64, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->block_size = 256;
        break;
    case 27:
        spec->name = "s27";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.persistent = true;
        spec->block_size = 256;
        break;
    case 28:
        spec->name = "s28";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.pad_k = true;
        spec->block_size = 256;
        break;
    case 29:
        spec->name = "s29";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.waves_per_eu_set = true;
        spec->trait.waves_per_eu = 2;
        spec->block_size = 256;
        break;
    case 30:
        spec->name = "s30";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.scheduler = "interwave";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 31:
        spec->name = "s31";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        spec->batched = true;
        break;
    case 32:
        spec->name = "s32";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.active_tile_skip = true;
        spec->block_size = 256;
        spec->batched = true;
        break;
    case 33:
        spec->name = "s33";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.preshuffle_b = true;
        spec->block_size = 256;
        break;
    case 34:
        spec->name = "s34";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 16, 16, 16);
        spec->trait.pipeline = "wsp3";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 35:
        spec->name = "s35";
        spec->tile = TILE(256, 256, 64, 4, 4, 1, 32, 32, 16);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 1024;
        break;
    case 36:
        spec->name = "s36";
        spec->tile = TILE(128, 128, 32, 2, 2, 1, 32, 32, 8);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->block_size = 256;
        break;
    case 37:
        spec->name = "s37";
        spec->tile = TILE(256, 256, 32, 4, 4, 1, 16, 16, 32);
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->block_size = 1024;
        break;
    default:
        return -1;
    }
    rocke_gemm_universal_spec_finalize(spec);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

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

    /* ir / verify modes need the kernel object */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_universal_gemm_new(&b, &spec, "gfx950");
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
