/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1201_deep_fused_conv_pool_emit.c -- C-side emitter for the
 * gfx1201 (RDNA4, wave32, WMMA 16x16x16) arch shim over the deep fused
 * conv0 -> conv1 -> maxpool parity harness. Selects one of N sampled spec
 * configs by argv[1] (the config index), builds the gfx1201-pinned
 * ckc_gfx1201_deep_fused_conv_pool_spec_t identically to the Python emitter
 * gfx1201_deep_fused_conv_pool_emit.py via
 * ckc_gfx1201_deep_fused_conv_pool_make_spec (which stamps the wave32 WMMA
 * geometry + gfx1201 kernel name), builds the kernel via
 * ckc_build_gfx1201_deep_fused_conv_pool_new (initialized builder + spec +
 * arch="gfx1201") and lowers via ckc_lower_kernel_to_llvm (arch="gfx1201",
 * flavor AUTO), printing the .ll to stdout so the two outputs can be
 * byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_gfx1201_deep_fused_conv_pool.h"
#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *spec and *arch. Mirrors gfx1201_deep_fused_conv_pool_emit.py
 * _spec. The gfx1201 factory pins wave_size=32 and warp_tile 16x16x16; all the
 * non-overridden fields take the common defaults (tile_n=32, tile_k=16,
 * conv1_tile_k=0, warp_m=2, warp_n=1, pipeline="mem"). */
static int make_cfg(int idx, ckc_gfx1201_deep_fused_conv_pool_spec_t* spec, const char** arch)
{
    *arch = "gfx1201";
    switch(idx)
    {
    case 0:
        *spec = ckc_gfx1201_deep_fused_conv_pool_make_spec(
            /*n*/ 1,
            /*h*/ 64,
            /*w*/ 128,
            /*c*/ 8,
            /*k0*/ 16,
            /*k1*/ 16,
            /*r*/ 3,
            /*s*/ 3,
            /*pool_tile_h*/ 4,
            /*pool_tile_w*/ 8,
            /*tile_n*/ 32,
            /*tile_k*/ 16,
            /*conv1_tile_k*/ 0,
            /*warp_m*/ 2,
            /*warp_n*/ 1,
            /*pipeline*/ NULL,
            /*unroll_k*/ false,
            /*async_dma*/ false,
            /*cache_input_footprint*/ false,
            /*direct_conv0_from_input_cache*/ false);
        return 0;
    case 1:
        *spec = ckc_gfx1201_deep_fused_conv_pool_make_spec(
            1, 80, 80, 16, 16, 16, 3, 3, 4, 8, 32, 16, 0, 2, 1, NULL, false, false, false, false);
        return 0;
    case 2:
        *spec = ckc_gfx1201_deep_fused_conv_pool_make_spec(
            1, 80, 80, 8, 16, 16, 3, 3, 2, 4, 32, 16, 0, 2, 1, NULL, false, false, false, false);
        return 0;
    case 3:
        *spec = ckc_gfx1201_deep_fused_conv_pool_make_spec(
            1, 96, 96, 8, 16, 16, 3, 3, 4, 8, 32, 16, 0, 2, 1, NULL, false, false, false, false);
        return 0;
    case 4:
        *spec = ckc_gfx1201_deep_fused_conv_pool_make_spec(
            1, 32, 64, 8, 16, 16, 3, 3, 4, 8, 32, 16, 0, 2, 1, NULL, false, false, false, false);
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

    ckc_gfx1201_deep_fused_conv_pool_spec_t spec;
    const char* arch = "gfx1201";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = ckc_build_gfx1201_deep_fused_conv_pool_new(&b, &spec, arch);
    if(kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        ckc_status_t st = ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if(st != CKC_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if(st != CKC_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
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
        ckc_ir_builder_free(&b);
        return 2;
    }

    ckc_ir_builder_free(&b);
    return 0;
}
