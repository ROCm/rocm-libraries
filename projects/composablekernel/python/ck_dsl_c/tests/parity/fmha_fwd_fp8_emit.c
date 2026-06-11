/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_fwd_fp8_emit.c -- C-side emitter for the FP8 FMHA forward
 * parity harness. Selects one of 6 sampled configs by argv[1] (the config index
 * 0..5), builds ckc_fmha_fwd_fp8_spec_t identically to the Python emitter
 * fmha_fwd_fp8_emit.py, lowers via ckc_fmha_fwd_fp8_lower_to_llvm (arch gfx950,
 * flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_fmha_fwd_fp8.h"
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_fmha_fwd_fp8_spec_t *spec) {
    ckc_fmha_shape_t shape;
    ckc_fmha_common_spec_t common;

    switch (idx) {
    case 0:
        shape = ckc_fmha_shape_make(64, 4, 4, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = CKC_KV_FP8_E4M3;
        spec->seqlen_q = 16;
        spec->seqlen_k = 64;
        break;
    case 1:
        shape = ckc_fmha_shape_make(64, 2, 2, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_CAUSAL;
        *spec = ckc_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = CKC_KV_FP8_E4M3;
        spec->seqlen_q = 32;
        spec->seqlen_k = 128;
        break;
    case 2:
        shape = ckc_fmha_shape_make(128, 8, 4, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = CKC_KV_BF8_E5M2;
        spec->seqlen_q = 16;
        spec->seqlen_k = 64;
        break;
    case 3:
        shape = ckc_fmha_shape_make(256, 4, 1, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_SLIDING_WINDOW;
        common.sliding_window = 32;
        *spec = ckc_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = CKC_KV_FP8_E4M3;
        spec->seqlen_q = 48;
        spec->seqlen_k = 256;
        spec->has_waves_per_eu = true;
        spec->waves_per_eu = 4;
        break;
    case 4:
        shape = ckc_fmha_shape_make(32, 16, 16, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = CKC_KV_FP8_E4M3;
        spec->seqlen_q = 64;
        spec->seqlen_k = 256;
        break;
    case 5:
        shape = ckc_fmha_shape_make(64, 8, 2, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = CKC_KV_BF8_E5M2;
        spec->seqlen_q = 32;
        spec->seqlen_k = 512;
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

    ckc_fmha_fwd_fp8_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char *llvm_text = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = 0;
    ckc_status_t st = ckc_fmha_fwd_fp8_lower_to_llvm(
        &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    return 0;
}
