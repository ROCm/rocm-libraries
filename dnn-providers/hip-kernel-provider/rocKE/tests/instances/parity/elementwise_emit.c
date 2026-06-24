/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/elementwise_emit.c -- C-side emitter for the elementwise parity
 * harness. Selects one of N sampled ElementwiseSpec configs by argv[1] (the
 * config index), builds ckc_elementwise_spec_t identically to the Python
 * emitter elementwise_emit.py, lowers via ckc_elementwise_lower_to_llvm
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
#include "ckc/instance_elementwise.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_elementwise_spec_t *spec) {
    *spec = ckc_elementwise_spec_default();

    switch (idx) {
    case 0:
        spec->op = "relu";  spec->dtype = "f16";  spec->block_size = 256; spec->vec = 8;
        break;
    case 1:
        spec->op = "relu";  spec->dtype = "bf16"; spec->block_size = 256; spec->vec = 8;
        break;
    case 2:
        spec->op = "add";   spec->dtype = "f16";  spec->block_size = 128; spec->vec = 4;
        break;
    case 3:
        spec->op = "add";   spec->dtype = "f16";  spec->block_size = 512; spec->vec = 2;
        break;
    case 4:
        spec->op = "silu";  spec->dtype = "bf16"; spec->block_size = 64;  spec->vec = 8;
        break;
    case 5:
        spec->op = "gelu_tanh"; spec->dtype = "f16"; spec->block_size = 1024; spec->vec = 4;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char *mode = (argc > 2) ? argv[2] : "ll";

    ckc_elementwise_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    if (strcmp(mode, "ll") == 0) {
        char *llvm_text = NULL;
        char err[CKC_ERR_MSG_CAP];
        err[0] = 0;
        ckc_status_t st = ckc_elementwise_lower_to_llvm(
            &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if (st != CKC_OK || !llvm_text) {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    } else if (strcmp(mode, "ir") == 0) {
        ckc_ir_builder_t b;
        ckc_kernel_def_t *kernel = ckc_build_elementwise_new(&b, &spec);
        if (!kernel) {
            fprintf(stderr, "build failed: %s\n", ckc_ir_builder_error(&b));
            ckc_ir_builder_free(&b);
            return 1;
        }
        char *t = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if (st != CKC_OK || !t) {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
        ckc_ir_builder_free(&b);
    } else if (strcmp(mode, "verify") == 0) {
        ckc_ir_builder_t b;
        ckc_kernel_def_t *kernel = ckc_build_elementwise_new(&b, &spec);
        if (!kernel) {
            fprintf(stderr, "build failed: %s\n", ckc_ir_builder_error(&b));
            ckc_ir_builder_free(&b);
            return 1;
        }
        ckc_diag_t *d = NULL;
        size_t n = 0;
        ckc_verify(kernel, &d, &n);
        for (size_t i = 0; i < n; i++) {
            char *s = ckc_diag_to_string(&d[i]);
            if (s) { puts(s); free(s); }
        }
        ckc_diags_free(d, n);
        ckc_ir_builder_free(&b);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    return 0;
}
