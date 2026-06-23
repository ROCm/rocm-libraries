/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/permute_nd_emit.c -- C-side emitter for the permute_nd instance
 * parity harness. Selects one of the sampled configs by argv[1] (the config
 * index), builds ckc_permute_spec_t identically to the Python emitter
 * permute_nd_emit.py, inits a fresh IRBuilder with the spec kernel name, builds
 * into it via ckc_build_permute (the C build entry), lowers via
 * ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_permute_nd.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_permute_spec_t *spec) {
    *spec = ckc_permute_spec_default();

    switch (idx) {
    case 0: /* (4,8,16) perm (2,0,1) f16 bs256 -> vec1 scalar */
        spec->rank = 3;
        spec->x_shape[0] = 4; spec->x_shape[1] = 8; spec->x_shape[2] = 16;
        spec->perm[0] = 2; spec->perm[1] = 0; spec->perm[2] = 1;
        spec->dtype = "f16"; spec->block_size = 256;
        break;
    case 1: /* (16,8,4) perm (2,1,0) f16 bs256 -> vec8 vectorized */
        spec->rank = 3;
        spec->x_shape[0] = 16; spec->x_shape[1] = 8; spec->x_shape[2] = 4;
        spec->perm[0] = 2; spec->perm[1] = 1; spec->perm[2] = 0;
        spec->dtype = "f16"; spec->block_size = 256;
        break;
    case 2: /* (32,32) perm (1,0) bf16 bs256 -> vec8 */
        spec->rank = 2;
        spec->x_shape[0] = 32; spec->x_shape[1] = 32;
        spec->perm[0] = 1; spec->perm[1] = 0;
        spec->dtype = "bf16"; spec->block_size = 256;
        break;
    case 3: /* (8,8,8,8) perm (3,2,1,0) f16 bs128 -> vec1 */
        spec->rank = 4;
        spec->x_shape[0] = 8; spec->x_shape[1] = 8;
        spec->x_shape[2] = 8; spec->x_shape[3] = 8;
        spec->perm[0] = 3; spec->perm[1] = 2; spec->perm[2] = 1; spec->perm[3] = 0;
        spec->dtype = "f16"; spec->block_size = 128;
        break;
    case 4: /* (64,64,64) perm (1,2,0) bf16 bs512 -> vec2 */
        spec->rank = 3;
        spec->x_shape[0] = 64; spec->x_shape[1] = 64; spec->x_shape[2] = 64;
        spec->perm[0] = 1; spec->perm[1] = 2; spec->perm[2] = 0;
        spec->dtype = "bf16"; spec->block_size = 512;
        break;
    case 5: /* (256,) perm (0,) f16 bs256 -> vec8 */
        spec->rank = 1;
        spec->x_shape[0] = 256;
        spec->perm[0] = 0;
        spec->dtype = "f16"; spec->block_size = 256;
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

    ckc_permute_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char *arch = "gfx950";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[CKC_ERR_MSG_CAP];
    reason[0] = 0;
    if (!ckc_permute_is_valid_spec(&spec, arch, reason, sizeof reason)) {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* Init IRBuilder with spec.kernel_name() then build into it via the
     * task-specified entry ckc_build_permute(b, &spec, arch). */
    char name[CKC_ERR_MSG_CAP];
    if (ckc_permute_kernel_name(&spec, name, sizeof name) != CKC_OK) {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }
    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, name) != CKC_OK) {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    ckc_kernel_def_t *kernel = ckc_build_permute(&b, &spec, arch);
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
