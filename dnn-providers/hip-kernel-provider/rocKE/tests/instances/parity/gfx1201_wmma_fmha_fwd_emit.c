/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1201_wmma_fmha_fwd_emit.c -- C-side emitter for the gfx1201
 * (RDNA4 / Navi 48) WMMA FMHA forward parity harness. Same 6 sampled
 * WmmaFmhaFwdSpec configs as the gfx1151 harness, but built and lowered at
 * arch=gfx1201 (flavor AUTO) so the RDNA4 split-K WMMA attention path
 * (wmma_gfx12_f32_16x16x16_f16 atom, <8 x half> fragments) is byte-compared
 * against the Python emitter gfx1201_wmma_fmha_fwd_emit.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_gfx1151_wmma_fmha_fwd.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_wmma_fmha_fwd_spec_t *spec) {
    *spec = ckc_wmma_fmha_fwd_spec_default();

    switch (idx) {
    case 0: /* H64, HQ4, HK0 (MHA), NONE, v_lds=False */
        spec->head_size = 64; spec->num_query_heads = 4; spec->num_kv_heads = 0;
        spec->mask_mode = CKC_FMHA_MASK_NONE; spec->v_lds_stage = false;
        break;
    case 1: /* H128, HQ8, HK0 (MHA), NONE, v_lds=False */
        spec->head_size = 128; spec->num_query_heads = 8; spec->num_kv_heads = 0;
        spec->mask_mode = CKC_FMHA_MASK_NONE; spec->v_lds_stage = false;
        break;
    case 2: /* H64, HQ4, HK0 (MHA), CAUSAL, v_lds=False */
        spec->head_size = 64; spec->num_query_heads = 4; spec->num_kv_heads = 0;
        spec->mask_mode = CKC_FMHA_MASK_CAUSAL; spec->v_lds_stage = false;
        break;
    case 3: /* H256, HQ8, HK2 (GQA), NONE, v_lds=False */
        spec->head_size = 256; spec->num_query_heads = 8; spec->num_kv_heads = 2;
        spec->mask_mode = CKC_FMHA_MASK_NONE; spec->v_lds_stage = false;
        break;
    case 4: /* H128, HQ4, HK4, CAUSAL, v_lds=False */
        spec->head_size = 128; spec->num_query_heads = 4; spec->num_kv_heads = 4;
        spec->mask_mode = CKC_FMHA_MASK_CAUSAL; spec->v_lds_stage = false;
        break;
    case 5: /* H64, HQ6, HK0 (MHA), NONE, v_lds=True */
        spec->head_size = 64; spec->num_query_heads = 6; spec->num_kv_heads = 0;
        spec->mask_mode = CKC_FMHA_MASK_NONE; spec->v_lds_stage = true;
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
    const char *mode = (argc > 2) ? argv[2] : "ll";

    ckc_wmma_fmha_fwd_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char *arch = "gfx1201";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[256];
    reason[0] = 0;
    if (!ckc_wmma_fmha_fwd_is_valid_spec(&spec, arch, reason, sizeof reason)) {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* (1) init builder with spec.kernel_name() */
    char name[256];
    if (ckc_wmma_fmha_fwd_kernel_name(&spec, name, sizeof name) != CKC_OK) {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, name) != CKC_OK) {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    /* (2) build */
    ckc_kernel_def_t *kernel = ckc_build_wmma_fmha_fwd(&b, &spec, arch);
    if (kernel == NULL) {
        const char *m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if (strcmp(mode, "ll") == 0) {
        /* (3) lower to .ll (arch gfx1201, flavor AUTO) */
        char *llvm_text = NULL;
        ckc_status_t st =
            ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if (st != CKC_OK || !llvm_text) {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ret = 1;
        } else {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    } else if (strcmp(mode, "ir") == 0) {
        char *t = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if (st != CKC_OK || !t) {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ret = 1;
        } else {
            fputs(t, stdout);
            free(t);
        }
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
        ret = 2;
    }
    ckc_ir_builder_free(&b);
    return ret;
}
