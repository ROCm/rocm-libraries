/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fused_moe_emit.c -- C-side emitter for the fused_moe parity
 * harness. Selects one of N sampled FusedMoeSpec configs by argv[1] (the config
 * index) and one of the five MoE-specific builders by argv[2] (the "phase"),
 * builds the matching FusedMoeSpec identically to the Python emitter
 * fused_moe_emit.py, builds the kernel via the matching ckc_build_moe_*_new
 * entry, lowers via ckc_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and
 * prints the .ll to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_fused_moe.h"

/* Mirrors the CONFIGS table in fused_moe_emit.py exactly. */
typedef struct {
    int tokens, experts, topk, hidden, intermediate;
    const char* dtype;
    int block_size, vec;
} cfg_t;

static const cfg_t CONFIGS[] = {
    {4,   4,  2, 128,  512,   "f16",  256, 4},
    {1,   8,  2, 1024, 2048,  "f16",  256, 4},
    {256, 16, 4, 4096, 16384, "bf16", 256, 8},
    {128, 32, 2, 2048, 8192,  "f16",  512, 4},
    {512, 64, 8, 8192, 32768, "bf16", 1024, 8},
    {16,  4,  1, 256,  1024,  "f16",  64,  2},
};
static const int N_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <config_index> <phase>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* phase = argv[2];
    if (idx < 0 || idx >= N_CONFIGS) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const cfg_t* c = &CONFIGS[idx];
    ckc_fused_moe_spec_t spec = ckc_fused_moe_spec_default();
    spec.tokens = c->tokens;
    spec.experts = c->experts;
    spec.topk = c->topk;
    spec.hidden = c->hidden;
    spec.intermediate = c->intermediate;
    spec.dtype = c->dtype;
    spec.block_size = c->block_size;
    spec.vec = c->vec;

    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel = NULL;

    if (strcmp(phase, "gather") == 0) {
        kernel = ckc_build_moe_gather_new(&b, &spec, "gfx950");
    } else if (strcmp(phase, "silu_mul") == 0) {
        kernel = ckc_build_moe_silu_mul_new(&b, &spec, "gfx950");
    } else if (strcmp(phase, "silu_mul_packed") == 0) {
        kernel = ckc_build_moe_silu_mul_packed_new(&b, &spec, "gfx950");
    } else if (strcmp(phase, "static_scatter_gather") == 0) {
        kernel = ckc_build_moe_static_scatter_gather_new(&b, &spec, "gfx950");
    } else if (strcmp(phase, "topk_weighted_reduce") == 0) {
        kernel = ckc_build_moe_topk_weighted_reduce_new(&b, &spec, "gfx950");
    } else {
        fprintf(stderr, "unknown phase %s\n", phase);
        return 2;
    }

    if (kernel == NULL) {
        const char* m = ckc_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        ckc_ir_builder_free(&b);
        return 1;
    }

    char* llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(kernel, CKC_LLVM_FLAVOR_AUTO,
                                               "gfx950", &llvm_text);
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
