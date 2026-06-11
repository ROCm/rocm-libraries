/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/mx_gemm_emit.c -- C-side emitter for the MX-GEMM parity harness.
 * Selects one of 6 sampled mx_gemm configs by argv[1] (index 0..5), builds a
 * ckc_mx_gemm_spec_t identically to the Python emitter mx_gemm_emit.py, lowers
 * via ckc_mx_gemm_lower_to_llvm (arch gfx950, flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_mx_gemm.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_mx_gemm_spec_t *spec) {
    *spec = ckc_mx_gemm_spec_default();
    spec->name = "ck_dsl_mx_gemm";
    spec->group_k = 32;
    spec->block_tile_m = 16;
    spec->block_tile_n = 16;
    spec->per_input_row = true;

    switch (idx) {
    case 0:
        spec->M = 16; spec->N = 16; spec->K = 64;
        spec->mantissa_dtype = "fp8e4m3";
        break;
    case 1:
        spec->M = 32; spec->N = 32; spec->K = 64;
        spec->mantissa_dtype = "fp8e4m3";
        break;
    case 2:
        spec->M = 16; spec->N = 16; spec->K = 32;
        spec->mantissa_dtype = "bf8e5m2";
        break;
    case 3:
        spec->M = 32; spec->N = 32; spec->K = 128;
        spec->mantissa_dtype = "fp8e4m3";
        break;
    case 4:
        spec->M = 48; spec->N = 64; spec->K = 96;
        spec->mantissa_dtype = "fp8e4m3";
        break;
    case 5:
        spec->M = 64; spec->N = 48; spec->K = 160;
        spec->mantissa_dtype = "bf8e5m2";
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

    ckc_mx_gemm_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char *llvm_text = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = 0;
    ckc_status_t st = ckc_mx_gemm_lower_to_llvm(
        &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    return 0;
}
