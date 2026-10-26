/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_winograd_gemm_wmma_emit.c -- C-side emitter for the
 * Winograd batched-GEMM (K3) parity harness. Selects one of N sampled spec
 * configs by argv[1] (the config index), builds the kernel identically to the
 * Python emitter conv_winograd_gemm_wmma_emit.py (rocke_build_winograd_gemm_wmma_new
 * or rocke_build_winograd_gemm_reduce_new) and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_winograd_gemm_wmma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* 0: GEMM, 1: reduce, -1: unknown. */
static int make_cfg(int idx,
                    rocke_winograd_gemm_wmma_spec_t* s,
                    rocke_winograd_gemm_reduce_spec_t* r,
                    const char** arch)
{
    *s = rocke_winograd_gemm_wmma_spec_default();
    *r = rocke_winograd_gemm_reduce_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 0:
        return 0;
    case 1:
        s->dtype = "bf16";
        return 0;
    case 2:
        s->tile_k = 64;
        s->stages = 3;
        return 0;
    case 3:
        s->waves_m = 2;
        s->waves_n = 2;
        s->frags_m = 4;
        s->frags_n = 4;
        return 0;
    case 4:
        s->waves_m = 1;
        s->waves_n = 2;
        s->frags_m = 2;
        s->frags_n = 2;
        s->tile_k = 64;
        return 0;
    case 5:
        s->waves_m = 2;
        s->waves_n = 2;
        s->frags_m = 2;
        s->frags_n = 2;
        s->tile_k = 32;
        s->stages = 3;
        return 0;
    case 6:
        s->epilogue = "direct";
        return 0;
    case 7:
        s->split_k = true;
        return 0;
    case 8:
        s->dtype = "bf16";
        s->split_k = true;
        s->epilogue = "direct";
        return 0;
    case 9:
        s->lds_pad = "clip";
        s->prefetch = false;
        return 0;
    case 10:
        s->xcd_remap = false;
        s->sched = NULL;
        return 0;
    case 11:
        r->dtype = "fp16";
        return 1;
    case 12:
        r->dtype = "bf16";
        return 1;
    case 13:
        *arch = "gfx950";
        return 0;
    case 14:
        s->tile_k = 48;
        s->lds_pad = "clip";
        return 0;
    case 15:
        s->waves_m = 1;
        s->waves_n = 1;
        s->frags_m = 4;
        s->frags_n = 2;
        s->tile_k = 64;
        s->stages = 3;
        return 0;
    default:
        return -1;
    }
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

    rocke_winograd_gemm_wmma_spec_t spec;
    rocke_winograd_gemm_reduce_spec_t rspec;
    const char* arch = "gfx1250";
    int kind = make_cfg(idx, &spec, &rspec, &arch);
    if(kind < 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = kind == 0 ? rocke_build_winograd_gemm_wmma_new(&b, &spec, arch)
                                           : rocke_build_winograd_gemm_reduce_new(&b, &rspec, arch);
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
        char* ir_text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &ir_text);
        if(st != ROCKE_OK || !ir_text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(ir_text, stdout);
        free(ir_text);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diags = NULL;
        size_t ndiag = 0;
        rocke_verify(kernel, &diags, &ndiag);
        for(size_t i = 0; i < ndiag; i++)
        {
            char* s = rocke_diag_to_string(&diags[i]);
            if(s)
            {
                fputs(s, stdout);
                fputc('\n', stdout);
                free(s);
            }
        }
        rocke_diags_free(diags, ndiag);
    }
    else
    {
        fprintf(stderr, "unknown mode '%s' (expected ll|ir|verify)\n", mode);
        ret = 2;
    }

    rocke_ir_builder_free(&b);
    return ret;
}
