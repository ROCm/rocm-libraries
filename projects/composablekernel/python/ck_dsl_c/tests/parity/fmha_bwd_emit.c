/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_bwd_emit.c -- C-side emitter for the FMHA-backward parity
 * harness. Selects one of 6 sampled FmhaBwdSpec configs by argv[1] (0..5),
 * builds ckc_fmha_bwd_spec_t identically to the Python emitter
 * fmha_bwd_emit.py, builds via ckc_build_fmha_bwd and lowers via
 * ckc_lower_kernel_to_llvm_ex (arch gfx950, flavor AUTO), printing the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_fmha_bwd.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_fmha_bwd_spec_t *spec) {
    ckc_fmha_shape_t shape;
    ckc_fmha_common_spec_t common;

    switch (idx) {
    case 0:
        shape = ckc_fmha_shape_default(64, 4, 4);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_bwd_spec_default(common, 16, 16);
        break;
    case 1:
        shape = ckc_fmha_shape_default(128, 8, 4);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_bwd_spec_default(common, 32, 64);
        break;
    case 2:
        shape = ckc_fmha_shape_default(256, 16, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_CAUSAL;
        *spec = ckc_fmha_bwd_spec_default(common, 64, 64);
        break;
    case 3:
        shape = ckc_fmha_shape_default(64, 2, 2);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_SLIDING_WINDOW;
        common.sliding_window = 16;
        *spec = ckc_fmha_bwd_spec_default(common, 128, 128);
        break;
    case 4:
        shape = ckc_fmha_shape_default(192, 8, 2);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_bwd_spec_default(common, 256, 256);
        break;
    case 5:
        shape = ckc_fmha_shape_default(128, 1, 1);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_CAUSAL;
        *spec = ckc_fmha_bwd_spec_default(common, 128, 256);
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

    ckc_fmha_bwd_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_fmha_kernel_builder_t kb;
    ckc_kernel_def_t *kernel = ckc_build_fmha_bwd(&kb, &spec, "gfx950");
    if (!kernel) {
        fprintf(stderr, "build failed: err=%s\n",
                ckc_ir_builder_error(ckc_fmha_kernel_builder_builder(&kb)));
        ckc_fmha_kernel_builder_free(&kb);
        return 1;
    }

    char *llvm_text = NULL;
    char lerr[CKC_ERR_MSG_CAP];
    lerr[0] = 0;
    ckc_status_t st = ckc_lower_kernel_to_llvm_ex(
        kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, lerr, sizeof lerr);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, lerr);
        ckc_fmha_kernel_builder_free(&kb);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    ckc_fmha_kernel_builder_free(&kb);
    return 0;
}
