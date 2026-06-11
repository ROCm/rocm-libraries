/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_splitkv_decode_emit.c -- C-side emitter for the split-KV
 * decode FMHA parity harness. Selects one of 6 sampled FmhaFwdSplitKvDecodeSpec
 * configs by argv[1] (0..5) and a phase by argv[2] ("seg" or "reduce"), builds
 * ckc_fmha_splitkv_decode_spec_t identically to the Python emitter
 * fmha_splitkv_decode_emit.py, and lowers the chosen kernel to LLVM .ll via the
 * convenience lower entries (arch gfx950, flavor AUTO), printing the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_fmha_splitkv_decode.h"
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_fmha_splitkv_decode_spec_t *spec) {
    ckc_fmha_shape_t shape;
    ckc_fmha_common_spec_t common;

    switch (idx) {
    case 0: /* H64 q8 kv8 f16 none, batch1, segs4 */
        shape = ckc_fmha_shape_default(64, 8, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_splitkv_decode_spec_default(common, 1, 4);
        break;
    case 1: /* H128 q8 kv8 f16 causal, batch2, segs8 */
        shape = ckc_fmha_shape_default(128, 8, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_CAUSAL;
        *spec = ckc_fmha_splitkv_decode_spec_default(common, 2, 8);
        break;
    case 2: /* H192 q16 kv2 bf16 none, batch4, segs16, use_mfma_body=False */
        shape = ckc_fmha_shape_default(192, 16, 2);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_splitkv_decode_spec_default(common, 4, 16);
        spec->use_mfma_body = false;
        break;
    case 3: /* H256 q32 kv4 f16 sliding_window 2048, batch1, segs32 */
        shape = ckc_fmha_shape_default(256, 32, 4);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_SLIDING_WINDOW;
        common.sliding_window = 2048;
        *spec = ckc_fmha_splitkv_decode_spec_default(common, 1, 32);
        break;
    case 4: /* H64 q12 kv3 bf16 none, batch8, segs64 */
        shape = ckc_fmha_shape_default(64, 12, 3);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        *spec = ckc_fmha_splitkv_decode_spec_default(common, 8, 64);
        break;
    case 5: /* H128 q16 kv8 f16 causal, batch2, segs128 */
        shape = ckc_fmha_shape_default(128, 16, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_CAUSAL;
        *spec = ckc_fmha_splitkv_decode_spec_default(common, 2, 128);
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <config_index 0..5> <seg|reduce>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char *phase = argv[2];

    ckc_fmha_splitkv_decode_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown/invalid config index %d\n", idx);
        return 2;
    }

    char *llvm_text = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = 0;
    ckc_status_t st;
    if (strcmp(phase, "seg") == 0) {
        st = ckc_fmha_splitkv_decode_segment_lower_to_llvm(
            &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    } else if (strcmp(phase, "reduce") == 0) {
        st = ckc_fmha_splitkv_decode_reduce_lower_to_llvm(
            &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    } else {
        fprintf(stderr, "unknown phase %s (want seg|reduce)\n", phase);
        return 2;
    }
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    return 0;
}
