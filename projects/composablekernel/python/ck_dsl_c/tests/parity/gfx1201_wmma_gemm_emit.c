/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1201_wmma_gemm_emit.c -- C-side emitter for the gfx1201
 * (RDNA4 / Navi48) WMMA GEMM parity harness. Selects one of 6 sampled
 * WmmaGemmSpec configs by argv[1] (0..5), builds it exactly as the Python
 * emitter gfx1201_wmma_gemm_emit.py does, and lowers to LLVM .ll text at
 * arch=gfx1201 (flavor AUTO) so the two outputs can be byte-compared.
 *
 * Build flow (mirrors the task spec):
 *   (1) ckc_ir_builder_init(b, spec.kernel_name())
 *   (2) ckc_build_wmma_gemm_gfx1201(b, &spec, "gfx1201")  -> KernelDef
 *   (3) ckc_lower_kernel_to_llvm(kernel, AUTO, "gfx1201", &ll)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_gfx1201_wmma_gemm.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_wmma_gemm_gfx1201_spec_t *spec) {
    *spec = ckc_wmma_gemm_gfx1201_spec_default();

    switch (idx) {
    case 0: /* WmmaGemmSpec() */
        break;
    case 1: /* WmmaGemmSpec(name="wmma_probe_gfx1201") */
        spec->name = "wmma_probe_gfx1201";
        break;
    case 2: /* WmmaGemmSpec(dtype="fp16") */
        spec->dtype = "fp16";
        break;
    case 3: /* WmmaGemmSpec(name="ck_dsl_wmma_gemm_gfx12_v2", dtype="fp16") */
        spec->name = "ck_dsl_wmma_gemm_gfx12_v2";
        spec->dtype = "fp16";
        break;
    case 4: /* WmmaGemmSpec(name="wmma_gemm_tile16x16x16") */
        spec->name = "wmma_gemm_tile16x16x16";
        break;
    case 5: /* WmmaGemmSpec(dtype="fp16", name="wmma_f16_16x16x16") */
        spec->dtype = "fp16";
        spec->name = "wmma_f16_16x16x16";
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

    ckc_wmma_gemm_gfx1201_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    /* (1) init builder with spec.kernel_name() */
    char name[256];
    if (ckc_wmma_gemm_gfx1201_kernel_name(&spec, name, sizeof name) != CKC_OK) {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, name) != CKC_OK) {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    /* (2) build */
    ckc_kernel_def_t *kernel = ckc_build_wmma_gemm_gfx1201(&b, &spec, "gfx1201");
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    const char *mode = (argc > 2) ? argv[2] : "ll";

    if (strcmp(mode, "ll") == 0) {
        /* (3) lower to .ll (arch gfx1201, flavor AUTO) */
        char *llvm_text = NULL;
        ckc_status_t st =
            ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO, "gfx1201", &llvm_text);
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
