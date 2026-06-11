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
 *   (1) ckc_ir_builder_init(b, spec.kernel_name())
 *   (2) ckc_build_wmma_gemm_gfx1151(b, &spec, "gfx1151")  -> KernelDef
 *   (3) ckc_lower_kernel_to_llvm(kernel, AUTO, "gfx1151", &ll)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_gfx1151_wmma_gemm.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_wmma_gemm_gfx1151_spec_t *spec) {
    *spec = ckc_wmma_gemm_gfx1151_spec_default();

    switch (idx) {
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
    case 3: /* WmmaGemmSpec(name="ck_dsl_wmma_gemm_v2", dtype="fp16", block_x_is_m=True) */
        spec->name = "ck_dsl_wmma_gemm_v2";
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_wmma_gemm_gfx1151_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    /* (1) init builder with spec.kernel_name() */
    char name[256];
    if (ckc_wmma_gemm_gfx1151_kernel_name(&spec, name, sizeof name) != CKC_OK) {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, name) != CKC_OK) {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    /* (2) build */
    ckc_kernel_def_t *kernel = ckc_build_wmma_gemm_gfx1151(&b, &spec, "gfx1151");
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    /* (3) lower to .ll (arch gfx1151, flavor AUTO) */
    char *llvm_text = NULL;
    ckc_status_t st =
        ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO, "gfx1151", &llvm_text);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d\n", (int)st);
        ckc_ir_builder_free(&b);
        return 1;
    }

    fputs(llvm_text, stdout);
    free(llvm_text);
    ckc_ir_builder_free(&b);
    return 0;
}
