/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1151_wmma_fmha_fwd_emit.c -- C-side emitter for the gfx1151
 * (RDNA3.5 / Strix Halo) WMMA FMHA forward parity harness. Selects one of 6
 * sampled WmmaFmhaFwdSpec configs by argv[1] (0..5), builds it exactly as the
 * Python emitter gfx1151_wmma_fmha_fwd_emit.py does, and lowers to LLVM .ll
 * text at arch=gfx1151 (flavor AUTO) so the two outputs can be byte-compared.
 *
 * Build flow (mirrors the Python build_wmma_fmha_fwd path):
 *   (1) ckc_ir_builder_init(b, spec.kernel_name())
 *   (2) ckc_build_wmma_fmha_fwd(b, &spec, "gfx1151")  -> KernelDef
 *   (3) ckc_lower_kernel_to_llvm(kernel, AUTO, "gfx1151", &ll)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
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

    ckc_wmma_fmha_fwd_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char *arch = "gfx1151";

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

    /* (3) lower to .ll (arch gfx1151, flavor AUTO) */
    char *llvm_text = NULL;
    ckc_status_t st =
        ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO, arch, &llvm_text);
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
