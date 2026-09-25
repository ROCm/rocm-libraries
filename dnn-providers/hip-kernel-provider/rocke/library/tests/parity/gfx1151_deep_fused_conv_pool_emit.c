/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1151_deep_fused_conv_pool_emit.c -- C-side emitter for the
 * gfx1151 (RDNA3.5 / Strix Halo, wave32, WMMA 16x16x16) deep fused
 * conv0 -> conv1 -> maxpool parity harness. Selects one of N sampled spec
 * configs by argv[1] (the config index), builds the gfx1151-pinned
 * rocke_gfx1151_deep_fused_conv_pool_spec_t identically to the Python emitter
 * gfx1151_deep_fused_conv_pool_emit.py via
 * rocke_gfx1151_deep_fused_conv_pool_make_spec (Python keyword defaults passed
 * explicitly), builds the kernel via
 * rocke_build_gfx1151_deep_fused_conv_pool_new (initialized builder + spec +
 * arch="gfx1151") and lowers via rocke_lower_kernel_to_llvm (arch="gfx1151",
 * flavor AUTO), printing the .ll to stdout so the two outputs can be
 * byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1151_deep_fused_conv_pool.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Build the spec for config index `idx`, mirroring
 * gfx1151_deep_fused_conv_pool_emit.py _spec. All non-shape arguments take the
 * Python keyword defaults: tile_n=32, warp_m=4, warp_n=2,
 * vectorize_conv0_a=True, vectorize_maxpool=True, early_w1=True,
 * direct_conv0=True, w_fast=False, waves_per_eu=0, sched_policy="mem", every
 * other bool False except batch_loads=True, persistent_ctas=16,
 * m0=0.0625, m0b=0.5, m1=0.25, mf=1.0. */
static rocke_gfx1151_deep_fused_conv_pool_spec_t
    mk(int n, int h, int w, int c, int k0, int k1, int r, int s, int pool_tile_h, int pool_tile_w)
{
    return rocke_gfx1151_deep_fused_conv_pool_make_spec(n,
                                                        h,
                                                        w,
                                                        c,
                                                        k0,
                                                        k1,
                                                        r,
                                                        s,
                                                        pool_tile_h,
                                                        pool_tile_w,
                                                        /*tile_n*/ 32,
                                                        /*warp_m*/ 4,
                                                        /*warp_n*/ 2,
                                                        /*vectorize_conv0_a*/ true,
                                                        /*vectorize_maxpool*/ true,
                                                        /*early_w1*/ true,
                                                        /*direct_conv0*/ true,
                                                        /*w_fast*/ false,
                                                        /*waves_per_eu*/ 0,
                                                        /*sched_policy*/ "mem",
                                                        /*mask_maxpool*/ false,
                                                        /*specialized_rne*/ false,
                                                        /*interior_fastpath*/ false,
                                                        /*static_direct_kmap*/ false,
                                                        /*packed_c0_handoff*/ false,
                                                        /*repack_c0*/ false,
                                                        /*fused_c0a1*/ false,
                                                        /*butterfly_conv01*/ false,
                                                        /*native_int*/ false,
                                                        /*batch_loads*/ true,
                                                        /*pk_maxpool*/ false,
                                                        /*conv1_prefetch_k*/ false,
                                                        /*conv1_sched_fuse*/ false,
                                                        /*conv1_int8*/ false,
                                                        /*persistent*/ false,
                                                        /*persistent_ctas*/ 16,
                                                        /*m0*/ 0.0625f,
                                                        /*m0b*/ 0.5f,
                                                        /*m1*/ 0.25f,
                                                        /*mf*/ 1.0f);
}

static int make_cfg(int idx, rocke_gfx1151_deep_fused_conv_pool_spec_t* spec, const char** arch)
{
    *arch = "gfx1151";
    switch(idx)
    {
    case 0:
        *spec = mk(1, 64, 128, 8, 16, 16, 3, 3, 4, 8);
        return 0;
    case 1:
        *spec = mk(1, 80, 80, 8, 16, 24, 3, 3, 4, 8);
        return 0;
    case 2:
        *spec = mk(1, 56, 112, 8, 16, 16, 3, 3, 2, 4);
        return 0;
    case 3:
        *spec = mk(1, 112, 112, 8, 16, 16, 3, 3, 4, 8);
        return 0;
    case 4:
        *spec = mk(1, 56, 56, 8, 24, 16, 3, 3, 4, 8);
        return 0;
    case 5:
        *spec = mk(1, 112, 224, 8, 16, 32, 3, 3, 4, 8);
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

    rocke_gfx1151_deep_fused_conv_pool_spec_t spec;
    const char* arch = "gfx1151";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_gfx1151_deep_fused_conv_pool_new(&b, &spec, arch);
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
    return ret;
}
