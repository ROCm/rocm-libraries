/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/batchnorm2d_bwd_emit.c -- C-side emitter for the dynamic NCHW batchnorm2d backward
 * kernels (single launch "bwd", split pair "reduce" / "apply").
 * Selects one config by argv[1] (the config index), builds the same kernel as
 * the Python emitter batchnorm2d_bwd_emit.py, lowers via rocke_lower_kernel_to_llvm (arch
 * gfx950, flavor AUTO) and prints the .ll to stdout for byte comparison.
 *
 * The config table MUST stay in lockstep with CONFIGS in batchnorm2d_bwd_emit.py.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_batchnorm2d_bwd.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct
{
    const char* kind; /* "bwd" | "reduce" | "apply" */
    int block_size;
    int vec;
    const char* dtype;
    int channels_per_block;
    int n_groups;
    bool fast_reduce;
} bn_bwd_cfg_t;

/* MUST match CONFIGS in batchnorm2d_bwd_emit.py (index for index). */
static const bn_bwd_cfg_t CONFIGS[] = {
    {"bwd", 256, 4, "f16", 1, 1, false},     {"bwd", 256, 4, "bf16", 1, 1, false},
    {"bwd", 1024, 8, "f16", 1, 1, false},    {"bwd", 128, 2, "bf16", 1, 1, true},
    {"bwd", 512, 8, "f16", 1, 2, false},     {"bwd", 256, 4, "bf16", 1, 8, true},
    {"bwd", 1024, 4, "f16", 2, 1, false},    {"bwd", 128, 8, "bf16", 8, 1, true},
    {"bwd", 512, 2, "f16", 4, 1, true},      {"bwd", 256, 8, "f16", 1, 4, false},
    {"reduce", 256, 8, "f16", 1, 1, false},  {"apply", 256, 8, "f16", 1, 1, false},
    {"reduce", 512, 8, "bf16", 1, 1, true},  {"apply", 512, 8, "bf16", 1, 1, true},
    {"reduce", 128, 4, "f16", 1, 1, true},   {"apply", 128, 4, "f16", 1, 1, true},
    {"reduce", 512, 4, "bf16", 1, 1, false}, {"apply", 512, 4, "bf16", 1, 1, false},
    {"bwd", 256, 4, "f32", 1, 1, false},     {"bwd", 128, 2, "f32", 1, 2, true},
    {"reduce", 256, 4, "f32", 1, 1, false},  {"apply", 256, 4, "f32", 1, 1, false},
    {"reduce", 512, 2, "f32", 1, 1, true},   {"apply", 512, 2, "f32", 1, 1, true},
};

#define NCFG ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

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

    if(idx < 0 || idx >= NCFG)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const bn_bwd_cfg_t* cfg = &CONFIGS[idx];

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = NULL;
    if(strcmp(cfg->kind, "bwd") == 0)
    {
        rocke_batchnorm2d_bwd_spec_t spec = rocke_batchnorm2d_bwd_spec_default();
        spec.block_size = cfg->block_size;
        spec.vec = cfg->vec;
        spec.dtype = cfg->dtype;
        spec.channels_per_block = cfg->channels_per_block;
        spec.n_groups = cfg->n_groups;
        spec.fast_reduce = cfg->fast_reduce;
        kernel = rocke_build_batchnorm2d_bwd_dynamic_new(&b, &spec);
    }
    else
    {
        rocke_batchnorm2d_bwd_split_spec_t spec = rocke_batchnorm2d_bwd_split_spec_default();
        spec.block_size = cfg->block_size;
        spec.vec = cfg->vec;
        spec.dtype = cfg->dtype;
        spec.fast_reduce = cfg->fast_reduce;
        kernel = (strcmp(cfg->kind, "reduce") == 0)
                     ? rocke_build_batchnorm2d_bwd_split_reduce_new(&b, &spec)
                     : rocke_build_batchnorm2d_bwd_split_apply_new(&b, &spec);
    }
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
