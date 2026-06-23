/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx950_attention_tiled_2d_fastkv_regp_emit.c -- C-side emitter
 * for the experimental gfx950 "fast paged-KV descriptor + register-P" tiled-2D
 * unified-attention parity harness.
 *
 * Selects one of the sampled configs by argv[1], fills a
 * ckc_attention_tiled_2d_spec_t identically to the Python emitter
 * gfx950_attention_tiled_2d_fastkv_regp_emit.py, builds the kernel via
 * ckc_build_unified_attention_2d_fastkv_register_p(&b, &spec, "gfx950"), lowers
 * it with ckc_lower_kernel_to_llvm(kernel, AUTO, "gfx950", ...) and prints the
 * .ll to stdout for byte comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_gfx950_attention_tiled_2d_fastkv_regp.h"

/* Fill `s` for config index `idx`. Returns 0 on success, -1 on unknown idx.
 *
 * Shared base for the bf16 d64_b32_h64kv8 / T=64 / num_warps=4 experiment
 * family, then per-config additive flags. Mirrors the Python emitter's _BASE /
 * _CONFIGS exactly. */
static void make_base(ckc_attention_tiled_2d_spec_t *s) {
    *s = ckc_attention_tiled_2d_spec_default();
    s->head_size = 64;
    s->block_size = 32;
    s->num_query_heads = 64;
    s->num_kv_heads = 8;
    s->dtype = "bf16";
    s->use_sinks = false;
    s->sliding_window = 0;
    s->has_softcap = false;
    s->num_warps = 4;
    s->has_waves_per_eu = true;
    s->waves_per_eu = 2;
    s->has_tile_size = true;
    s->tile_size = 64;
    s->block_m_per_warp = 32;
    s->use_mfma_32x32 = true;
    s->use_transposed_qk_32x32 = true;
    s->use_transposed_scalar_state = true;
    s->use_transposed_mask_once = true;
    s->use_fast_paged_kv_desc = true;
}

static int make_spec(int idx, ckc_attention_tiled_2d_spec_t *s) {
    make_base(s);
    switch (idx) {
    case 0:
        break;
    case 1:
        s->use_transposed_half_local_pv = true;
        break;
    case 2:
        s->use_mfma32_skip_legacy_qreg = true;
        break;
    case 3:
        s->use_transposed_half_local_pv = true;
        s->use_mfma32_skip_legacy_qreg = true;
        break;
    case 4:
        s->use_agpr_alloc_zero = true;
        s->use_transposed_half_local_pv = true;
        break;
    case 5:
        s->use_grouped_kv2_softmax = true;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char *mode = (argc > 2) ? argv[2] : "ll";

    ckc_attention_tiled_2d_spec_t s;
    if (make_spec(idx, &s) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }

    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, "attention_tiled_2d_fastkv_regp") != CKC_OK) {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }

    ckc_kernel_def_t *kernel =
        ckc_build_unified_attention_2d_fastkv_register_p(&b, &s, "gfx950");
    if (!kernel) {
        fprintf(stderr, "build failed: err=%s\n", ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return 1;
    }

    if (strcmp(mode, "ll") == 0) {
        char *llvm_text = NULL;
        ckc_status_t st = ckc_lower_kernel_to_llvm(
            kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
        if (st != CKC_OK || !llvm_text) {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    } else if (strcmp(mode, "ir") == 0) {
        char *t = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if (st != CKC_OK || !t) {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    } else if (strcmp(mode, "verify") == 0) {
        ckc_diag_t *d = NULL;
        size_t n = 0;
        ckc_verify(kernel, &d, &n);
        for (size_t i = 0; i < n; i++) {
            char *s2 = ckc_diag_to_string(&d[i]);
            if (s2) { puts(s2); free(s2); }
        }
        ckc_diags_free(d, n);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        ckc_ir_builder_free(&b);
        return 2;
    }

    ckc_ir_builder_free(&b);
    return 0;
}
