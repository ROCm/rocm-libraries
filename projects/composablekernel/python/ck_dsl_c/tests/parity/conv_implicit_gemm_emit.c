/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_implicit_gemm_emit.c -- C-side emitter for the implicit-GEMM
 * convolution parity harness. Selects one of N sampled spec configs by argv[1]
 * (the config index), builds the ckc_implicit_gemm_conv_spec_t identically to
 * the Python emitter conv_implicit_gemm_emit.py, builds the kernel via
 * ckc_build_implicit_gemm_conv_new (initialized builder + spec + arch, NULL
 * overrides = stock conv body) and lowers via ckc_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two outputs
 * can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_conv_implicit_gemm.h"

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *spec and *arch. */
static int make_cfg(int idx, ckc_implicit_gemm_conv_spec_t *spec,
                    const char **arch) {
    *spec = ckc_implicit_gemm_conv_spec_default();
    spec->tile_m = 64;  spec->tile_n = 64;  spec->tile_k = 64;
    spec->warp_m = 2;   spec->warp_n = 2;
    spec->warp_tile_m = 32; spec->warp_tile_n = 32; spec->warp_tile_k = 16;
    spec->pipeline = "mem";
    spec->epilogue = "default";

    switch (idx) {
    case 0:
        spec->problem = ckc_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        *arch = "gfx950";
        return 0;
    case 1:
        spec->problem = ckc_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->tile_m = 128; spec->tile_n = 128; spec->tile_k = 64;
        spec->pipeline = "compv4";
        *arch = "gfx950";
        return 0;
    case 2:
        spec->problem = ckc_conv_problem_default(16, 112, 112, 128, 128, 3, 3);
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 3:
        spec->problem = ckc_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->async_dma = true;
        *arch = "gfx950";
        return 0;
    case 4:
        spec->problem = ckc_conv_problem_default(1, 224, 224, 3, 64, 7, 7);
        spec->tile_m = 128; spec->tile_n = 128; spec->tile_k = 64;
        *arch = "gfx950";
        return 0;
    case 5:
        spec->problem = ckc_conv_problem_default(8, 56, 56, 64, 64, 1, 1);
        *arch = "gfx950";
        return 0;
    default:
        return -1;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_implicit_gemm_conv_spec_t spec;
    const char *arch = "gfx950";
    if (make_cfg(idx, &spec, &arch) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_ir_builder_t b;
    ckc_kernel_def_t *kernel =
        ckc_build_implicit_gemm_conv_new(&b, &spec, arch, NULL);
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    char *llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO,
                                               arch, &llvm_text);
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
