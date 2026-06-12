/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/layernorm2d_emit.c -- C-side emitter for the LayerNorm2D parity
 * STRESS harness. Selects one config by argv[1] (the config index), builds
 * ckc_layernorm2d_spec_t identically to the Python emitter layernorm2d_emit.py,
 * lowers via ckc_layernorm2d_lower_to_llvm (arch gfx950, flavor AUTO) and
 * prints the .ll to stdout so the two outputs can be byte-compared.
 *
 * The config table MUST stay in lockstep with CONFIGS in layernorm2d_emit.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_layernorm2d.h"

typedef struct {
    int n_per_block;
    int block_size;
    int vec;
    const char *dtype;
    bool save_mean_invstd;
} ln_cfg_t;

/* MUST match CONFIGS in layernorm2d_emit.py (index for index). */
static const ln_cfg_t CONFIGS[] = {
    /* 0..5  original sampled */
    {4096, 256, 4, "f16", false},
    {4096, 256, 8, "f16", false},
    {4096, 256, 4, "bf16", false},
    {2048, 128, 4, "f16", true},
    {8192, 256, 8, "f16", false},
    {1024, 256, 2, "bf16", true},
    /* 6..9 tiny */
    {128, 64, 2, "f16", false},
    {256, 64, 4, "f16", true},
    {512, 64, 8, "bf16", false},
    {128, 64, 2, "bf16", true},
    /* 10..13 block sizes */
    {4096, 512, 4, "f16", false},
    {8192, 1024, 8, "f16", false},
    {2048, 1024, 2, "bf16", true},
    {1024, 128, 8, "f16", false},
    /* 14..15 fp16 alias */
    {4096, 256, 4, "fp16", false},
    {2048, 128, 2, "fp16", true},
    /* 16..21 odd multipliers */
    {1536, 256, 2, "f16", false},
    {3072, 256, 4, "bf16", false},
    {5120, 256, 4, "f16", true},
    {1792, 128, 2, "bf16", false},
    {2816, 128, 2, "f16", false},
    {6656, 256, 2, "bf16", true},
    /* 22..27 two-pass */
    {16384, 256, 8, "f16", false},
    {33280, 256, 2, "f16", false},
    {32768, 256, 8, "bf16", false},
    {65536, 256, 8, "f16", true},
    {131072, 512, 8, "bf16", false},
    {34816, 256, 2, "bf16", true},
    /* 28..29 very large single/two pass at 1024 block */
    {65536, 1024, 8, "f16", false},
    {133120, 1024, 2, "bf16", false},
    /* 30..32 vec sweep */
    {2048, 256, 2, "f16", false},
    {4096, 256, 4, "f16", true},
    {8192, 256, 8, "bf16", true},
    /* 33..34 block 512 two-pass */
    {66560, 512, 2, "f16", false},
    {133120, 512, 4, "bf16", true},
};

#define NCFG ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

static int make_spec(int idx, ckc_layernorm2d_spec_t *spec) {
    if (idx < 0 || idx >= NCFG)
        return -1;
    *spec = ckc_layernorm2d_spec_default();
    spec->n_per_block = CONFIGS[idx].n_per_block;
    spec->block_size = CONFIGS[idx].block_size;
    spec->vec = CONFIGS[idx].vec;
    spec->dtype = CONFIGS[idx].dtype;
    spec->save_mean_invstd = CONFIGS[idx].save_mean_invstd;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index 0..%d>\n", argv[0], NCFG - 1);
        return 2;
    }
    if (strcmp(argv[1], "--count") == 0) {
        printf("%d\n", NCFG);
        return 0;
    }
    int idx = atoi(argv[1]);

    ckc_layernorm2d_spec_t spec;
    if (make_spec(idx, &spec) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char *llvm_text = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = 0;
    ckc_status_t st = ckc_layernorm2d_lower_to_llvm(
        &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    return 0;
}
