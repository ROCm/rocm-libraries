/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/batchnorm2d_split_emit.c -- C-side emitter for the cross-CTA
 * spatial-split batchnorm2d parity STRESS harness. Selects one config by
 * argv[1] (the config index), builds rocke_batchnorm2d_split_spec_t
 * identically to the Python emitter batchnorm2d_split_emit.py, builds
 * whichever of the two split kernels ("reduce"/"combine") that config
 * selects, lowers via rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO)
 * and prints the .ll to stdout so the two outputs can be byte-compared.
 *
 * The config table MUST stay in lockstep with CONFIGS in
 * batchnorm2d_split_emit.py.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_batchnorm2d.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct
{
    const char* kind; /* "reduce" | "combine" */
    int n;
    int c;
    int h;
    int w;
    int split_hw;
    int block_size;
    int vec;
    const char* dtype;
    int channels_per_block;
    bool fast_reduce;
} bn_split_cfg_t;

/* MUST match CONFIGS in batchnorm2d_split_emit.py (index for index). */
static const bn_split_cfg_t CONFIGS[] = {
    {"reduce", 1, 16, 64, 64, 4, 64, 2, "f16", 1, false},
    {"combine", 1, 16, 64, 64, 4, 64, 2, "f16", 1, false},
    {"reduce", 1, 16, 64, 64, 4, 64, 2, "bf16", 1, false},
    {"combine", 1, 16, 64, 64, 4, 64, 2, "bf16", 1, false},
    {"reduce", 1, 8, 32, 32, 2, 64, 2, "f16", 1, false},
    {"combine", 1, 8, 32, 32, 2, 64, 2, "f16", 1, false},
    {"reduce", 1, 8, 32, 32, 8, 128, 4, "f16", 1, false},
    {"combine", 1, 8, 32, 32, 8, 128, 4, "f16", 1, false},
    {"reduce", 1, 16, 14, 14, 3, 64, 2, "f16", 1, false},
    {"combine", 1, 16, 14, 14, 3, 64, 2, "f16", 1, false},
    {"reduce", 2, 16, 32, 32, 4, 64, 2, "f16", 1, false},
    {"combine", 2, 16, 32, 32, 4, 64, 2, "f16", 1, false},
    {"reduce", 1, 16, 32, 32, 4, 64, 2, "f16", 2, false},
    {"combine", 1, 16, 32, 32, 4, 64, 2, "f16", 2, false},
    {"reduce", 1, 16, 32, 32, 4, 64, 2, "f16", 1, true},
    {"combine", 1, 16, 32, 32, 4, 64, 2, "f16", 1, true},
    {"reduce", 1, 16, 128, 128, 16, 256, 4, "f16", 1, false},
    {"combine", 1, 16, 128, 128, 16, 256, 4, "f16", 1, false},
};

#define NCFG ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

static int make_spec(int idx, rocke_batchnorm2d_split_spec_t* spec, const char** out_kind)
{
    if(idx < 0 || idx >= NCFG)
        return -1;
    *spec = rocke_batchnorm2d_split_spec_default();
    spec->n = CONFIGS[idx].n;
    spec->c = CONFIGS[idx].c;
    spec->h = CONFIGS[idx].h;
    spec->w = CONFIGS[idx].w;
    spec->split_hw = CONFIGS[idx].split_hw;
    spec->block_size = CONFIGS[idx].block_size;
    spec->vec = CONFIGS[idx].vec;
    spec->dtype = CONFIGS[idx].dtype;
    spec->channels_per_block = CONFIGS[idx].channels_per_block;
    spec->fast_reduce = CONFIGS[idx].fast_reduce;
    *out_kind = CONFIGS[idx].kind;
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NCFG - 1);
        return 2;
    }
    if(strcmp(argv[1], "--count") == 0)
    {
        printf("%d\n", NCFG);
        return 0;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_batchnorm2d_split_spec_t spec;
    const char* kind = NULL;
    if(make_spec(idx, &spec, &kind) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = (strcmp(kind, "reduce") == 0)
        ? rocke_build_batchnorm2d_reduce_partial_new(&b, &spec)
        : rocke_build_batchnorm2d_combine_apply_new(&b, &spec);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    { /* verify */
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }

    rocke_ir_builder_free(&b);
    return 0;
}
