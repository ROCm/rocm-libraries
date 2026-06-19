/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1151_wmma_gemm_iu8_dequant_emit.c -- C-side emitter for the
 * gfx1151 (RDNA3.5 / Strix Halo) true-INT8 WMMA GEMM with f16 dequant output
 * parity harness. Builds the kernel exactly as the Python emitter does and
 * lowers to LLVM .ll text at arch=gfx1151 (flavor AUTO) for byte-comparison.
 *
 * WmmaGemmIu8DequantSpec has no config fields: M/N/K are runtime kernel args
 * (not build-time, so they do not affect the emitted IR) and arch is fixed
 * gfx1151. The argv[1] config index is accepted for harness symmetry; every
 * index emits the identical kernel.
 *
 * Build flow (mirrors the task spec):
 *   (1) ckc_ir_builder_init(b, spec.kernel_name())
 *   (2) ckc_build_wmma_gemm_iu8_dequant(b, &spec, "gfx1151")  -> KernelDef
 *   (3) ckc_lower_kernel_to_llvm(kernel, AUTO, "gfx1151", &ll)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_gfx1151_wmma_gemm_iu8_dequant.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }

    /* The index does not affect the emitted IR (M/N/K are runtime args), but the
     * valid config range is 0..5 to match the Python emitter; reject beyond it so
     * the harness sees a shared end-of-range. */
    int idx = atoi(argv[1]);
    if (idx < 0 || idx > 5) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_wmma_gemm_iu8_dequant_spec_t spec =
        ckc_wmma_gemm_iu8_dequant_spec_default();

    /* (1) init builder with spec.kernel_name() */
    char name[256];
    if (ckc_wmma_gemm_iu8_dequant_kernel_name(&spec, name, sizeof name) != CKC_OK) {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, name) != CKC_OK) {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    /* (2) build */
    ckc_kernel_def_t *kernel = ckc_build_wmma_gemm_iu8_dequant(&b, &spec, "gfx1151");
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    const char *mode = (argc > 2) ? argv[2] : "ll";

    if (strcmp(mode, "ll") == 0) {
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
    } else if (strcmp(mode, "ir") == 0) {
        char *t = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if (st != CKC_OK || !t) {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    } else if (strcmp(mode, "verify") == 0) {
        ckc_diag_t *d = NULL;
        size_t n = 0;
        ckc_verify(kernel, &d, &n);
        for (size_t i = 0; i < n; i++) {
            char *s = ckc_diag_to_string(&d[i]);
            if (s) { puts(s); free(s); }
        }
        ckc_diags_free(d, n);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        ckc_ir_builder_free(&b);
        return 2;
    }

    ckc_ir_builder_free(&b);
    return 0;
}
