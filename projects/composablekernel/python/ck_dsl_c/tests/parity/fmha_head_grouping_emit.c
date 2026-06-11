/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_head_grouping_emit.c -- C-side emitter for the GQA/MQA
 * head-grouped FMHA forward parity harness. Selects one of 6 sampled configs by
 * argv[1] (the config index 0..5), builds ckc_fmha_head_grouping_spec_t
 * identically to the Python emitter fmha_head_grouping_emit.py, builds the
 * kernel via ckc_build_fmha_fwd_head_grouping (arch gfx950) and lowers via
 * ckc_lower_kernel_to_llvm (flavor AUTO) and prints the .ll to stdout so the
 * two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_fmha_head_grouping.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_fmha_head_grouping_spec_t *spec) {
    ckc_fmha_shape_t shape;
    ckc_fmha_common_spec_t common;

    switch (idx) {
    case 0:
        shape = ckc_fmha_shape_default(64, 32, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_head_grouping_spec_make(common, 16, 16);
        break;
    case 1:
        shape = ckc_fmha_shape_default(128, 16, 1);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_CAUSAL;
        *spec = ckc_fmha_head_grouping_spec_make(common, 32, 32);
        break;
    case 2:
        shape = ckc_fmha_shape_default(64, 8, 4);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_head_grouping_spec_make(common, 32, 32);
        break;
    case 3:
        shape = ckc_fmha_shape_default(256, 32, 4);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_SLIDING_WINDOW;
        common.sliding_window = 512;
        *spec = ckc_fmha_head_grouping_spec_make(common, 16, 16);
        break;
    case 4:
        shape = ckc_fmha_shape_default(128, 8, 2);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_head_grouping_spec_make(common, 48, 64);
        break;
    case 5:
        shape = ckc_fmha_shape_default(64, 24, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_CAUSAL;
        *spec = ckc_fmha_head_grouping_spec_make(common, 64, 64);
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

    ckc_fmha_head_grouping_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    ckc_fmha_kernel_builder_t kb;
    memset(&kb, 0, sizeof kb);

    ckc_kernel_def_t *kernel =
        ckc_build_fmha_fwd_head_grouping(&kb, &spec, "gfx950");
    if (kernel == NULL) {
        fprintf(stderr, "build failed for config %d\n", idx);
        ckc_fmha_kernel_builder_free(&kb);
        return 1;
    }

    char *llvm_text = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = 0;
    ckc_status_t st = ckc_lower_kernel_to_llvm_ex(
        kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
        ckc_fmha_kernel_builder_free(&kb);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    ckc_fmha_kernel_builder_free(&kb);
    return 0;
}
