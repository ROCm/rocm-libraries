/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx950_deep_fused_conv_pool_emit.c -- C reference emitter for the
 * gfx950 deep fused conv0 -> conv1 -> maxpool parity harness. Selects one of the
 * sampled spec configs by argv[1], builds the gfx950-pinned spec (the common
 * spec with the gfx950 name + wave64 MFMA 16x16x16 geometry, re-wrapped as the
 * gfx950 spec), builds+lowers via
 * rocke_gfx950_deep_fused_conv_pool_lower_to_llvm(spec, "gfx950", AUTO, ...) and
 * prints the .ll to stdout so it can be byte/sha256-compared with the Python
 * reference gfx950_deep_fused_conv_pool_emit.py.
 *
 * WHY NOT rocke_gfx950_deep_fused_conv_pool_make_spec:
 *   That factory hard-pins the CDNA MFMA 32x32x16 atom, which has NO verified MMA
 *   fragment layout map -- the build raises before any IR is emitted, so every
 *   config would be a no-emit reject (false pass). The only wave64 geometry that
 *   actually LOWERS is the MFMA 16x16x16 atom (the same atom the common emit.c
 *   idx-6 uses to exercise the gfx950 emit path). The gfx950 shim differs from
 *   the common spec in `name` only, so we build the common spec via
 *   rocke_make_deep_fused_conv_pool_spec with the wave64 16x16x16 geometry + the
 *   pinned gfx950 kernel name, then wrap it as the gfx950 spec (.base = base) --
 *   exactly what the gfx950 factory does internally, minus the non-emittable
 *   32x32 pin. The v1 "one CTA owns all conv channels" gate (K <= tile_n) is
 *   satisfied with tile_n=16 and K0=K1=16.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx950_deep_fused_conv_pool.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Build a gfx950 spec on the emittable wave64 MFMA 16x16x16 atom: common
 * make_spec with the gfx950 name + wave64 16x16x16 geometry, re-wrapped as the
 * gfx950 spec (mirrors gfx950_deep_fused_conv_pool_emit.py _mk). */
static rocke_gfx950_deep_fused_conv_pool_spec_t mk(int n,
                                                   int h,
                                                   int w,
                                                   int c,
                                                   int k0,
                                                   int k1,
                                                   int r,
                                                   int s,
                                                   int pool_tile_h,
                                                   int pool_tile_w,
                                                   bool cache_input_footprint,
                                                   bool direct_conv0_from_input_cache)
{
    rocke_gfx950_deep_fused_conv_pool_spec_t out;
    rocke_deep_fused_conv_pool_spec_t base
        = rocke_make_deep_fused_conv_pool_spec(n,
                                               h,
                                               w,
                                               c,
                                               k0,
                                               k1,
                                               r,
                                               s,
                                               pool_tile_h,
                                               pool_tile_w,
                                               /*tile_n*/ 16,
                                               /*tile_k*/ 16,
                                               /*conv1_tile_k*/ 0,
                                               /*warp_m*/ 2,
                                               /*warp_n*/ 1,
                                               /*warp_tile_m*/ 16,
                                               /*warp_tile_n*/ 16,
                                               /*warp_tile_k*/ 16,
                                               /*wave_size*/ 64,
                                               /*name*/ ROCKE_GFX950_DEEP_FUSED_CONV_POOL_NAME,
                                               /*pipeline*/ NULL,
                                               /*unroll_k*/ false,
                                               /*async_dma*/ false,
                                               cache_input_footprint,
                                               direct_conv0_from_input_cache);
    memset(&out, 0, sizeof(out));
    out.base = base;
    return out;
}

static int make_cfg(int idx, rocke_gfx950_deep_fused_conv_pool_spec_t* spec)
{
    switch(idx)
    {
    case 0:
        *spec = mk(1, 64, 128, 8, 16, 16, 3, 3, 4, 8, false, false);
        return 0;
    case 1:
        *spec = mk(1, 80, 80, 8, 16, 16, 3, 3, 2, 4, false, false);
        return 0;
    case 2:
        *spec = mk(1, 96, 96, 8, 16, 16, 3, 3, 4, 8, false, false);
        return 0;
    case 3:
        /* cache_input_footprint=true */
        *spec = mk(1, 56, 56, 8, 16, 16, 3, 3, 4, 4, true, false);
        return 0;
    case 4:
        /* direct_conv0_from_input_cache=true */
        *spec = mk(1, 32, 64, 8, 16, 16, 3, 3, 4, 8, false, true);
        return 0;
    default:
        return -1;
    }
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

    rocke_gfx950_deep_fused_conv_pool_spec_t spec;
    if(make_cfg(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = '\0';
        rocke_status_t st = rocke_gfx950_deep_fused_conv_pool_lower_to_llvm(
            &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &llvm, err, sizeof(err));
        if(st != ROCKE_OK || !llvm)
        {
            fprintf(stderr,
                    "lower failed cfg%d status=%d (%s)\n",
                    idx,
                    (int)st,
                    err[0] ? err : "(no message)");
            return 1;
        }
        fputs(llvm, stdout);
        free(llvm);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        rocke_ir_builder_t b;
        rocke_kernel_def_t* kernel
            = rocke_build_gfx950_deep_fused_conv_pool_new(&b, &spec, "gfx950");
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
        rocke_kernel_def_t* kernel
            = rocke_build_gfx950_deep_fused_conv_pool_new(&b, &spec, "gfx950");
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
