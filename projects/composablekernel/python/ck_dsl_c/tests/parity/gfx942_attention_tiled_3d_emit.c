/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx942_attention_tiled_3d_emit.c -- C-side emitter for the
 * gfx942 narrow-atom tiled split-KV 3D *segment* attention parity harness.
 * Selects one of the sampled configs by argv[1], fills a
 * ckc_unified_attention_3d_tiled_spec_t identically to the Python emitter
 * gfx942_attention_tiled_3d_emit.py, builds + lowers the segment kernel via
 * ckc_build_unified_attention_3d_tiled_gfx942_lower_to_llvm (arch gfx942,
 * flavor AUTO) and prints the .ll to stdout for byte comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"
#include "ckc/instance_gfx942_attention_tiled_3d.h"

/* Fill `s` for config index `idx`. Returns 0 on success, -1 on unknown idx. */
static int make_spec(int idx, ckc_unified_attention_3d_tiled_spec_t *s) {
    *s = ckc_unified_attention_3d_tiled_spec_default();
    switch (idx) {
    case 0:
        s->head_size = 64;  s->block_size = 16;
        s->num_query_heads = 8;  s->num_kv_heads = 8;
        s->dtype = "fp16"; s->num_segments = 8;
        s->use_sinks = false; s->sliding_window = 0; s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        break;
    case 1:
        s->head_size = 128; s->block_size = 32;
        s->num_query_heads = 16; s->num_kv_heads = 16;
        s->dtype = "bf16"; s->num_segments = 16;
        s->use_sinks = false; s->sliding_window = 0; s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        break;
    case 2:
        s->head_size = 256; s->block_size = 64;
        s->num_query_heads = 32; s->num_kv_heads = 8;
        s->dtype = "fp16"; s->num_segments = 8;
        s->use_sinks = false; s->sliding_window = 4096; s->has_softcap = false;
        s->kv_storage_dtype = NULL;
        break;
    case 3:
        s->head_size = 128; s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->num_segments = 16;
        s->use_sinks = true; s->sliding_window = 0; s->has_softcap = true;
        s->kv_storage_dtype = "fp8e4m3";
        break;
    case 4:
        s->head_size = 64;  s->block_size = 16;
        s->num_query_heads = 16; s->num_kv_heads = 4;
        s->dtype = "fp16"; s->num_segments = 8;
        s->use_sinks = false; s->sliding_window = 2048; s->has_softcap = false;
        s->use_alibi = true; s->use_qq_bias = false;
        s->kv_storage_dtype = NULL;
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

    ckc_unified_attention_3d_tiled_spec_t s;
    if (make_spec(idx, &s) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }

    if (strcmp(mode, "ll") == 0) {
        char *llvm_text = NULL;
        char err[CKC_ERR_MSG_CAP];
        err[0] = 0;
        ckc_status_t st = ckc_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
            &s, "gfx942", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if (st != CKC_OK || !llvm_text) {
            fprintf(stderr, "build/lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    } else if (strcmp(mode, "ir") == 0) {
        char kname[256]; kname[0] = 0;
        ckc_unified_attention_3d_tiled_spec_kernel_name(&s, kname, sizeof(kname));
        ckc_ir_builder_t b;
        if (ckc_ir_builder_init(&b, kname) != CKC_OK) {
            fprintf(stderr, "builder init failed\n");
            return 1;
        }
        ckc_kernel_def_t *seg_k = ckc_build_unified_attention_3d_tiled_gfx942(&b, &s, "gfx942");
        if (!seg_k) {
            fprintf(stderr, "segment build failed: err=%s\n", ckc_ir_builder_error(&b));
            ckc_ir_builder_free(&b);
            return 1;
        }
        char *t = NULL;
        ckc_status_t st = ckc_ir_serialize(seg_k, &t);
        if (st != CKC_OK || !t) {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ckc_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
        ckc_ir_builder_free(&b);
    } else if (strcmp(mode, "verify") == 0) {
        char kname[256]; kname[0] = 0;
        ckc_unified_attention_3d_tiled_spec_kernel_name(&s, kname, sizeof(kname));
        ckc_ir_builder_t b;
        if (ckc_ir_builder_init(&b, kname) != CKC_OK) {
            fprintf(stderr, "builder init failed\n");
            return 1;
        }
        ckc_kernel_def_t *seg_k = ckc_build_unified_attention_3d_tiled_gfx942(&b, &s, "gfx942");
        if (!seg_k) {
            fprintf(stderr, "segment build failed: err=%s\n", ckc_ir_builder_error(&b));
            ckc_ir_builder_free(&b);
            return 1;
        }
        ckc_diag_t *d = NULL; size_t n = 0;
        ckc_verify(seg_k, &d, &n);
        for (size_t i = 0; i < n; i++) { char *s2 = ckc_diag_to_string(&d[i]); if (s2) { puts(s2); free(s2); } }
        ckc_diags_free(d, n);
        ckc_ir_builder_free(&b);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    return 0;
}
