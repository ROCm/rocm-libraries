/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/batchnorm2d_emit.c -- C-side emitter for the BatchNorm2D parity
 * STRESS harness. Selects one config by argv[1] (the config index), builds
 * rocke_batchnorm2d_spec_t identically to the Python emitter
 * batchnorm2d_emit.py, lowers via rocke_batchnorm2d_lower_to_llvm (arch
 * gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared.
 *
 * The config table MUST stay in lockstep with CONFIGS in batchnorm2d_emit.py.
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
    int n;
    int c;
    int h;
    int w;
    int block_size;
    int vec;
    const char* dtype;
    bool training;
    bool save_mean_invstd;
    int channels_per_block;
    int n_groups;
    bool fast_reduce;
} bn_cfg_t;

/* MUST match CONFIGS in batchnorm2d_emit.py (index for index). */
static const bn_cfg_t CONFIGS[] = {
    /* 0..2 small single-pass, training */
    {2, 8, 16, 16, 64, 2, "f16", true, false, 1, 1, false},
    {2, 8, 16, 16, 64, 2, "bf16", true, false, 1, 1, false},
    {2, 8, 16, 16, 64, 2, "f16", true, true, 1, 1, false},
    /* 3..4 inference */
    {2, 8, 16, 16, 64, 2, "f16", false, false, 1, 1, false},
    {2, 8, 16, 16, 64, 2, "bf16", false, false, 1, 1, false},
    /* 5..7 vec / block sweep */
    {2, 8, 32, 32, 128, 4, "f16", true, false, 1, 1, false},
    {4, 4, 16, 16, 64, 4, "bf16", true, true, 1, 1, false},
    {1, 16, 8, 8, 32, 2, "f16", true, false, 1, 1, false},
    /* 8 fp16 alias */
    {2, 8, 16, 16, 64, 2, "fp16", true, false, 1, 1, false},
    /* 9..11 two-pass */
    {8, 4, 32, 32, 64, 2, "f16", true, false, 1, 1, false},
    {8, 4, 32, 32, 64, 2, "bf16", false, false, 1, 1, false},
    {8, 4, 32, 32, 64, 4, "f16", true, true, 1, 1, false},
    /* 12..15 channels_per_block > 1 lever */
    {2, 8, 16, 16, 64, 2, "f16", true, false, 2, 1, false},
    {2, 8, 16, 16, 64, 2, "bf16", false, false, 4, 1, false},
    {4, 4, 16, 16, 64, 4, "bf16", true, true, 2, 1, false},
    {8, 4, 32, 32, 64, 2, "f16", true, false, 4, 1, false},
    /* 16..18 n_groups > 1 lever (inference-only) */
    {4, 4, 16, 16, 256, 2, "f16", false, false, 1, 2, false},
    {4, 4, 16, 16, 256, 2, "bf16", false, false, 1, 4, false},
    {4, 8, 16, 16, 256, 2, "f16", false, false, 2, 2, false},
    /* 19..21 fast_reduce lever (training-only) */
    {2, 8, 16, 16, 64, 2, "f16", true, false, 1, 1, true},
    {4, 4, 16, 16, 128, 2, "bf16", true, true, 1, 1, true},
    {2, 8, 16, 16, 64, 2, "f16", true, false, 2, 1, true},
};

#define NCFG ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

static int make_spec(int idx, rocke_batchnorm2d_spec_t* spec)
{
    if(idx < 0 || idx >= NCFG)
        return -1;
    *spec = rocke_batchnorm2d_spec_default();
    spec->n = CONFIGS[idx].n;
    spec->c = CONFIGS[idx].c;
    spec->h = CONFIGS[idx].h;
    spec->w = CONFIGS[idx].w;
    spec->block_size = CONFIGS[idx].block_size;
    spec->vec = CONFIGS[idx].vec;
    spec->dtype = CONFIGS[idx].dtype;
    spec->training = CONFIGS[idx].training;
    spec->save_mean_invstd = CONFIGS[idx].save_mean_invstd;
    spec->channels_per_block = CONFIGS[idx].channels_per_block;
    spec->n_groups = CONFIGS[idx].n_groups;
    spec->fast_reduce = CONFIGS[idx].fast_reduce;
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

    rocke_batchnorm2d_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_batchnorm2d_new(&b, &spec);
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
