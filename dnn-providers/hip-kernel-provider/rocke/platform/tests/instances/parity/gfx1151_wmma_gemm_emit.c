/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1151_wmma_gemm_emit.c -- C-side emitter for the gfx1151
 * (RDNA3.5 / Strix Halo) WMMA GEMM parity harness. Selects one of 6 sampled
 * WmmaGemmSpec configs by argv[1] (0..5), builds it exactly as the Python
 * emitter gfx1151_wmma_gemm_emit.py does, and lowers to LLVM .ll text at
 * arch=gfx1151 (flavor AUTO) so the two outputs can be byte-compared.
 *
 * Build flow (mirrors the task spec):
 *   (1) rocke_ir_builder_init(b, spec.kernel_name())
 *   (2) rocke_build_wmma_gemm_gfx1151(b, &spec, "gfx1151")  -> KernelDef
 *   (3) rocke_lower_kernel_to_llvm(kernel, AUTO, "gfx1151", &ll)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1151_wmma_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_wmma_gemm_gfx1151_spec_t* spec)
{
    *spec = rocke_wmma_gemm_gfx1151_spec_default();

    switch(idx)
    {
    case 0: /* WmmaGemmSpec() */
        break;
    case 1: /* WmmaGemmSpec(name="wmma_probe_gfx1151", block_x_is_m=True) */
        spec->name = "wmma_probe_gfx1151";
        spec->block_x_is_m = true;
        break;
    case 2: /* WmmaGemmSpec(dtype="fp16", block_x_is_m=True) */
        spec->dtype = "fp16";
        spec->block_x_is_m = true;
        break;
    case 3: /* WmmaGemmSpec(name="rocke_wmma_gemm_v2", dtype="fp16", block_x_is_m=True) */
        spec->name = "rocke_wmma_gemm_v2";
        spec->dtype = "fp16";
        spec->block_x_is_m = true;
        break;
    case 4: /* WmmaGemmSpec(name="wmma_gemm_tile16x16x16", block_x_is_m=False) */
        spec->name = "wmma_gemm_tile16x16x16";
        spec->block_x_is_m = false;
        break;
    case 5: /* WmmaGemmSpec(dtype="fp16", name="wmma_f16_16x16x16", block_x_is_m=False) */
        spec->dtype = "fp16";
        spec->name = "wmma_f16_16x16x16";
        spec->block_x_is_m = false;
        break;
    default:
        return -1;
    }
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

    rocke_wmma_gemm_gfx1151_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    /* (1) init builder with spec.kernel_name() */
    char name[256];
    if(rocke_wmma_gemm_gfx1151_kernel_name(&spec, name, sizeof name) != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    /* (2) build */
    rocke_kernel_def_t* kernel = rocke_build_wmma_gemm_gfx1151(&b, &spec, "gfx1151");
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") == 0)
    {
        /* (3) lower to .ll (arch gfx1151, flavor AUTO) */
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx1151", &llvm_text);
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
    return 0;
}
