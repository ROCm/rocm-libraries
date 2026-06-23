/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/streamk_gemm_emit.c -- C-side emitter for the streamk-GEMM parity
 * harness. Selects one of the sampled configs by argv[1] (the config index),
 * builds ckc_streamk_gemm_spec_t identically to the Python emitter
 * streamk_gemm_emit.py, builds the kernel via ckc_build_streamk_gemm_new and
 * lowers via ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO), printing the
 * .ll to stdout so the two outputs can be byte-compared.
 *
 * Optional argv[2] = mode: "ll" (default), "ir", "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_streamk_gemm.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_streamk_gemm_spec_t *spec) {
    *spec = ckc_streamk_gemm_spec_default();
    spec->dtype = "f16";

    switch (idx) {
    case 0:
        spec->M = 32;  spec->N = 32;  spec->K = 32;
        spec->tile_m = 16; spec->tile_n = 16; spec->tile_k = 16;
        spec->num_cus = 8; spec->blocks_per_cu = 1; spec->persistent = false;
        break;
    case 1:
        spec->M = 64;  spec->N = 64;  spec->K = 64;
        spec->tile_m = 16; spec->tile_n = 16; spec->tile_k = 32;
        spec->num_cus = 8; spec->blocks_per_cu = 1; spec->persistent = false;
        break;
    case 2:
        spec->M = 128; spec->N = 128; spec->K = 128;
        spec->tile_m = 16; spec->tile_n = 16; spec->tile_k = 32;
        spec->num_cus = 256; spec->blocks_per_cu = 1; spec->persistent = true;
        break;
    case 3:
        spec->M = 256; spec->N = 256; spec->K = 256;
        spec->tile_m = 32; spec->tile_n = 32; spec->tile_k = 16;
        spec->num_cus = 304; spec->blocks_per_cu = 1; spec->persistent = false;
        break;
    case 4:
        spec->M = 128; spec->N = 128; spec->K = 64;
        spec->tile_m = 32; spec->tile_n = 32; spec->tile_k = 16;
        spec->num_cus = 256; spec->blocks_per_cu = 1; spec->persistent = false;
        break;
    case 5:
        spec->M = 512; spec->N = 512; spec->K = 512;
        spec->tile_m = 16; spec->tile_n = 16; spec->tile_k = 32;
        spec->num_cus = 304; spec->blocks_per_cu = 2; spec->persistent = true;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..5> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char *mode = (argc > 2) ? argv[2] : "ll";

    if (strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 &&
        strcmp(mode, "verify") != 0) {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    ckc_streamk_gemm_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_ir_builder_t b;
    ckc_kernel_def_t *kernel = ckc_build_streamk_gemm_new(&b, &spec, "gfx950");
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(unknown)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    if (strcmp(mode, "ir") == 0) {
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
        /* mode == "ll" */
        char *llvm_text = NULL;
        ckc_status_t st = ckc_lower_kernel_to_llvm(
            kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
        if (st != CKC_OK || !llvm_text) {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    ckc_ir_builder_free(&b);
    return 0;
}
