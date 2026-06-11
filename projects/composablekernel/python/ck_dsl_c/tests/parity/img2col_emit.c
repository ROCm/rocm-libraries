/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/img2col_emit.c -- C-side emitter for the img2col parity harness.
 * Selects one of N sampled spec configs by argv[1] (the config index), builds
 * ckc_img2col_spec_t identically to the Python emitter img2col_emit.py, builds
 * the kernel via ckc_build_img2col_new + lowers via ckc_lower_kernel_to_llvm
 * (arch gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs
 * can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_img2col.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_img2col_spec_t *spec) {
    *spec = ckc_img2col_spec_default();
    ckc_conv_problem_t p = ckc_img2col_conv_problem_default();

    switch (idx) {
    case 0:
        p.N = 1; p.Hi = 8; p.Wi = 8; p.C = 16; p.K = 16; p.R = 3; p.S = 3;
        spec->block_tile_m = 4; spec->block_tile_k = 64; spec->vec_k = 1;
        break;
    case 1:
        p.N = 2; p.Hi = 16; p.Wi = 16; p.C = 32; p.K = 32; p.R = 3; p.S = 3;
        p.pH = 1; p.pW = 1;
        spec->block_tile_m = 8; spec->block_tile_k = 128; spec->vec_k = 4;
        break;
    case 2:
        p.N = 4; p.Hi = 32; p.Wi = 32; p.C = 64; p.K = 64; p.R = 3; p.S = 3;
        p.pH = 1; p.pW = 1;
        spec->block_tile_m = 16; spec->block_tile_k = 256; spec->vec_k = 8;
        break;
    case 3:
        p.N = 8; p.Hi = 56; p.Wi = 56; p.C = 64; p.K = 64; p.R = 3; p.S = 3;
        p.pH = 1; p.pW = 1;
        spec->block_tile_m = 8; spec->block_tile_k = 128; spec->vec_k = 8;
        break;
    case 4:
        p.N = 2; p.Hi = 16; p.Wi = 16; p.C = 15; p.K = 32; p.R = 3; p.S = 3;
        spec->block_tile_m = 8; spec->block_tile_k = 120; spec->vec_k = 8;
        break;
    case 5:
        p.N = 2; p.Hi = 32; p.Wi = 32; p.C = 32; p.K = 32; p.R = 3; p.S = 3;
        p.dH = 2; p.dW = 2; p.pH = 2; p.pW = 2;
        spec->block_tile_m = 8; spec->block_tile_k = 128; spec->vec_k = 4;
        break;
    default:
        return -1;
    }
    spec->problem = p;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_img2col_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_ir_builder_t b;
    ckc_kernel_def_t *kernel = ckc_build_img2col_new(&b, &spec, "gfx950");
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    char *llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO,
                                               "gfx950", &llvm_text);
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
