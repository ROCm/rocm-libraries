/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/deep_fused_conv_pool_emit.c -- C-side emitter for the deep fused
 * conv0 -> conv1 -> maxpool parity harness. Selects one of N sampled spec
 * configs by argv[1] (the config index), builds the
 * rocke_deep_fused_conv_pool_spec_t identically to the Python emitter
 * deep_fused_conv_pool_emit.py via rocke_make_deep_fused_conv_pool_spec, builds
 * the kernel via rocke_build_deep_fused_conv_pool_new (initialized builder + spec
 * + arch) and lowers via rocke_lower_kernel_to_llvm (per-config arch, flavor
 * AUTO), printing the .ll to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_deep_fused_conv_pool.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *spec and *arch. Mirrors deep_fused_conv_pool_emit.py _spec. */
static int make_cfg(int idx, rocke_deep_fused_conv_pool_spec_t* spec, const char** arch)
{
    switch(idx)
    {
    case 0:
        *spec = rocke_make_deep_fused_conv_pool_spec(
            /*n*/ 1,
            /*h*/ 112,
            /*w*/ 112,
            /*c*/ 64,
            /*k0*/ 64,
            /*k1*/ 64,
            /*r*/ 3,
            /*s*/ 3,
            /*pool_tile_h*/ 4,
            /*pool_tile_w*/ 8,
            /*tile_n*/ 32,
            /*tile_k*/ 16,
            /*conv1_tile_k*/ 0,
            /*warp_m*/ 2,
            /*warp_n*/ 1,
            /*warp_tile_m*/ 32,
            /*warp_tile_n*/ 32,
            /*warp_tile_k*/ 16,
            /*wave_size*/ 64,
            /*name*/ NULL,
            /*pipeline*/ NULL,
            /*unroll_k*/ false,
            /*async_dma*/ false,
            /*cache_input_footprint*/ false,
            /*direct_conv0_from_input_cache*/ false);
        *arch = "gfx950";
        return 0;
    case 1:
        *spec = rocke_make_deep_fused_conv_pool_spec(1,
                                                     56,
                                                     56,
                                                     128,
                                                     128,
                                                     128,
                                                     3,
                                                     3,
                                                     4,
                                                     8,
                                                     32,
                                                     16,
                                                     0,
                                                     2,
                                                     1,
                                                     32,
                                                     32,
                                                     16,
                                                     64,
                                                     NULL,
                                                     NULL,
                                                     false,
                                                     false,
                                                     false,
                                                     false);
        *arch = "gfx950";
        return 0;
    case 2:
        *spec = rocke_make_deep_fused_conv_pool_spec(1,
                                                     28,
                                                     28,
                                                     256,
                                                     256,
                                                     256,
                                                     3,
                                                     3,
                                                     4,
                                                     8,
                                                     32,
                                                     16,
                                                     0,
                                                     2,
                                                     1,
                                                     32,
                                                     32,
                                                     16,
                                                     64,
                                                     NULL,
                                                     NULL,
                                                     false,
                                                     false,
                                                     false,
                                                     false);
        *arch = "gfx950";
        return 0;
    case 3:
        *spec = rocke_make_deep_fused_conv_pool_spec(1,
                                                     112,
                                                     112,
                                                     64,
                                                     64,
                                                     64,
                                                     3,
                                                     3,
                                                     4,
                                                     8,
                                                     32,
                                                     16,
                                                     0,
                                                     2,
                                                     1,
                                                     16,
                                                     16,
                                                     16,
                                                     32,
                                                     NULL,
                                                     NULL,
                                                     false,
                                                     false,
                                                     false,
                                                     false);
        *arch = "gfx1201";
        return 0;
    case 4:
        *spec = rocke_make_deep_fused_conv_pool_spec(1,
                                                     56,
                                                     56,
                                                     32,
                                                     32,
                                                     32,
                                                     3,
                                                     3,
                                                     4,
                                                     8,
                                                     32,
                                                     16,
                                                     0,
                                                     2,
                                                     1,
                                                     32,
                                                     32,
                                                     16,
                                                     64,
                                                     NULL,
                                                     NULL,
                                                     false,
                                                     false,
                                                     /*cache_input_footprint*/ true,
                                                     false);
        *arch = "gfx950";
        return 0;
    case 5:
        *spec = rocke_make_deep_fused_conv_pool_spec(1,
                                                     28,
                                                     28,
                                                     64,
                                                     64,
                                                     64,
                                                     3,
                                                     3,
                                                     4,
                                                     8,
                                                     32,
                                                     16,
                                                     0,
                                                     2,
                                                     1,
                                                     32,
                                                     32,
                                                     16,
                                                     64,
                                                     NULL,
                                                     NULL,
                                                     false,
                                                     false,
                                                     false,
                                                     /*direct_conv0_from_input_cache*/ true);
        *arch = "gfx950";
        return 0;
    case 6:
        /* Passing config that BOTH the gate accepts AND the emitter supports
         * (16x16x16 MFMA atom has a verified layout map; the 32x32x16 atom the
         * original configs use does not -- Python itself raises NotImplemented
         * for those). Proves the emit path is byte-faithful, not just reject. */
        *spec = rocke_make_deep_fused_conv_pool_spec(1,
                                                     64,
                                                     128,
                                                     8,
                                                     16,
                                                     16,
                                                     3,
                                                     3,
                                                     4,
                                                     8,
                                                     16,
                                                     16,
                                                     0,
                                                     2,
                                                     1,
                                                     16,
                                                     16,
                                                     16,
                                                     64,
                                                     NULL,
                                                     NULL,
                                                     false,
                                                     false,
                                                     false,
                                                     false);
        *arch = "gfx950";
        return 0;
    case 7:
        /* gfx1201 WMMA passing/emittable config. */
        *spec = rocke_make_deep_fused_conv_pool_spec(1,
                                                     64,
                                                     128,
                                                     8,
                                                     16,
                                                     16,
                                                     3,
                                                     3,
                                                     4,
                                                     8,
                                                     16,
                                                     16,
                                                     0,
                                                     2,
                                                     1,
                                                     16,
                                                     16,
                                                     16,
                                                     32,
                                                     NULL,
                                                     NULL,
                                                     false,
                                                     false,
                                                     false,
                                                     false);
        *arch = "gfx1201";
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

    rocke_deep_fused_conv_pool_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_deep_fused_conv_pool_new(&b, &spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
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
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
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
        rocke_ir_builder_free(&b);
        return 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
