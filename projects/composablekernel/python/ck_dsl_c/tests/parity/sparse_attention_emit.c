/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/sparse_attention_emit.c -- C-side emitter for the sparse-attention
 * forward parity harness. Selects one of 6 sampled configs by argv[1] (the config
 * index 0..5), builds either a ckc_jenga_sparse_spec_t (via
 * ckc_build_jenga_sparse_attention) or a ckc_vsa_sparse_spec_t (via
 * ckc_build_vsa_sparse_attention) identically to the Python emitter
 * sparse_attention_emit.py, lowers the returned KernelDef via
 * ckc_lower_kernel_to_llvm_ex (arch gfx950, flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 *
 * Optional argv[2] = mode: "ll" (default), "ir", "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_sparse_attention.h"
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"

/* Build the KernelDef for config index `idx`. Returns NULL on unknown idx or
 * build failure. */
static ckc_kernel_def_t *make_kernel(int idx) {
    ckc_fmha_shape_t shape;
    ckc_fmha_common_spec_t common;

    switch (idx) {
    case 0: {
        shape = ckc_fmha_shape_default(64, 8, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        ckc_jenga_sparse_spec_t spec =
            ckc_jenga_sparse_spec_default(common, /*seqlen_q*/ 32, /*seqlen_k*/ 128);
        spec.block_q = 1;
        spec.block_k = 64;
        return ckc_build_jenga_sparse_attention(NULL, &spec, "gfx950");
    }
    case 1: {
        shape = ckc_fmha_shape_default(128, 16, 16);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        ckc_jenga_sparse_spec_t spec =
            ckc_jenga_sparse_spec_default(common, /*seqlen_q*/ 64, /*seqlen_k*/ 256);
        spec.block_q = 2;
        spec.block_k = 64;
        return ckc_build_jenga_sparse_attention(NULL, &spec, "gfx950");
    }
    case 2: {
        shape = ckc_fmha_shape_default(64, 8, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        ckc_vsa_sparse_spec_t spec =
            ckc_vsa_sparse_spec_default(common, /*seqlen_q*/ 32, /*seqlen_k*/ 128);
        spec.block_q = 1;
        spec.block_k = 64;
        spec.max_blocks_per_q = 16;
        return ckc_build_vsa_sparse_attention(NULL, &spec, "gfx950");
    }
    case 3: {
        shape = ckc_fmha_shape_default(128, 16, 16);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        ckc_vsa_sparse_spec_t spec =
            ckc_vsa_sparse_spec_default(common, /*seqlen_q*/ 64, /*seqlen_k*/ 256);
        spec.block_q = 2;
        spec.block_k = 64;
        spec.max_blocks_per_q = 32;
        spec.use_wave_ballot_scatter = true;
        return ckc_build_vsa_sparse_attention(NULL, &spec, "gfx950");
    }
    case 4: {
        shape = ckc_fmha_shape_default(256, 32, 8);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        ckc_jenga_sparse_spec_t spec =
            ckc_jenga_sparse_spec_default(common, /*seqlen_q*/ 96, /*seqlen_k*/ 512);
        spec.block_q = 4;
        spec.block_k = 128;
        return ckc_build_jenga_sparse_attention(NULL, &spec, "gfx950");
    }
    case 5: {
        shape = ckc_fmha_shape_default(256, 32, 32);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        ckc_vsa_sparse_spec_t spec =
            ckc_vsa_sparse_spec_default(common, /*seqlen_q*/ 128, /*seqlen_k*/ 1024);
        spec.block_q = 8;
        spec.block_k = 64;
        spec.max_blocks_per_q = 24;
        spec.use_wave_ballot_scatter = false;
        return ckc_build_vsa_sparse_attention(NULL, &spec, "gfx950");
    }
    default:
        return NULL;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..5> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char *mode = (argc > 2) ? argv[2] : "ll";

    if (strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 &&
        strcmp(mode, "verify") != 0) {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    ckc_kernel_def_t *kernel = make_kernel(idx);
    if (!kernel) {
        fprintf(stderr, "build failed / unknown config index %d\n", idx);
        return 1;
    }

    if (strcmp(mode, "ir") == 0) {
        char *t = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if (st != CKC_OK || !t) {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
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
        /* mode == "ll" */
        char *llvm_text = NULL;
        char err[CKC_ERR_MSG_CAP];
        err[0] = 0;
        ckc_status_t st = ckc_lower_kernel_to_llvm_ex(
            kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
        if (st != CKC_OK || !llvm_text) {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    return 0;
}
