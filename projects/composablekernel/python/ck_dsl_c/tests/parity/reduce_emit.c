/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/reduce_emit.c -- C-side emitter for the reduce2d instance parity
 * harness. Selects one of the sampled configs by argv[1] (the config index),
 * builds ckc_reduce2d_spec_t identically to the Python emitter reduce_emit.py,
 * inits a fresh IRBuilder with the spec kernel name, builds into it via
 * ckc_build_reduce2d (the C build entry), lowers via ckc_lower_kernel_to_llvm
 * (arch gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs
 * can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_reduce.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_reduce2d_spec_t *spec) {
    *spec = ckc_reduce2d_spec_default();

    switch (idx) {
    case 0:
        spec->n_per_block = 4096; spec->op = "sum"; spec->block_size = 256;
        spec->vec = 4; spec->dtype = "f16"; spec->wave_size = 64;
        break;
    case 1:
        spec->n_per_block = 4096; spec->op = "max"; spec->block_size = 256;
        spec->vec = 4; spec->dtype = "f16"; spec->wave_size = 64;
        break;
    case 2:
        spec->n_per_block = 4096; spec->op = "mean"; spec->block_size = 256;
        spec->vec = 4; spec->dtype = "f16"; spec->wave_size = 64;
        break;
    case 3:
        spec->n_per_block = 2048; spec->op = "sum"; spec->block_size = 128;
        spec->vec = 4; spec->dtype = "bf16"; spec->wave_size = 64;
        break;
    case 4:
        spec->n_per_block = 4096; spec->op = "sum"; spec->block_size = 512;
        spec->vec = 2; spec->dtype = "f16"; spec->wave_size = 64;
        break;
    case 5:
        spec->n_per_block = 3072; spec->op = "max"; spec->block_size = 256;
        spec->vec = 8; spec->dtype = "bf16"; spec->wave_size = 64;
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

    ckc_reduce2d_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char *arch = "gfx950";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[CKC_ERR_MSG_CAP];
    reason[0] = 0;
    if (!ckc_reduce2d_is_valid_spec(&spec, reason, sizeof reason)) {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* Init IRBuilder with spec.kernel_name() then build into it via the
     * task-specified entry ckc_build_reduce2d(b, &spec, arch). */
    char name[CKC_ERR_MSG_CAP];
    if (ckc_reduce2d_kernel_name(&spec, name, sizeof name) != CKC_OK) {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }
    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, name) != CKC_OK) {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    ckc_kernel_def_t *kernel = ckc_build_reduce2d(&b, &spec, arch);
    if (!kernel) {
        fprintf(stderr, "build failed: %s\n", b.err);
        ckc_ir_builder_free(&b);
        return 1;
    }

    if (strcmp(mode, "ll") == 0) {
        char *llvm_text = NULL;
        ckc_status_t st = ckc_lower_kernel_to_llvm(
            kernel, CKC_LLVM_FLAVOR_AUTO, arch, &llvm_text);
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
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
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
