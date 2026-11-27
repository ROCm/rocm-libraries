/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/batchnorm2d_nhwc_emit.c -- C-side emitter for the dynamic NHWC batchnorm2d kernels
 * (infer, training forward and backward; kinds infer / fused / reduce / apply).
 * Selects one config by argv[1] (the config index), builds the same kernel as
 * the Python emitter batchnorm2d_nhwc_emit.py, lowers via rocke_lower_kernel_to_llvm (arch
 * gfx950, flavor AUTO) and prints the .ll to stdout for byte comparison.
 *
 * The config table MUST stay in lockstep with CONFIGS in batchnorm2d_nhwc_emit.py.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_batchnorm2d_nhwc.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct
{
    const char* mode; /* "infer" | "train" | "bwd" */
    const char* kind; /* "infer" | "fused" | "reduce" | "apply" */
    int block_size;
    int tx;
    int vec;
    int unroll;
    const char* dtype;
} bn_nhwc_cfg_t;

/* MUST match CONFIGS in batchnorm2d_nhwc_emit.py (index for index). */
static const bn_nhwc_cfg_t CONFIGS[] = {
    {"infer", "infer", 256, 4, 8, 2, "f16"},     {"infer", "infer", 256, 64, 4, 2, "bf16"},
    {"infer", "infer", 256, 32, 8, 16, "f16"},   {"train", "fused", 256, 8, 8, 4, "f16"},
    {"train", "fused", 1024, 16, 4, 4, "bf16"},  {"train", "reduce", 1024, 32, 2, 16, "bf16"},
    {"train", "apply", 1024, 32, 2, 16, "bf16"}, {"train", "reduce", 512, 8, 8, 16, "f16"},
    {"train", "apply", 512, 8, 8, 16, "f16"},    {"bwd", "fused", 512, 16, 2, 4, "f16"},
    {"bwd", "fused", 1024, 8, 4, 4, "bf16"},     {"bwd", "reduce", 1024, 64, 4, 4, "bf16"},
    {"bwd", "apply", 1024, 64, 4, 4, "bf16"},    {"bwd", "reduce", 512, 16, 8, 8, "f16"},
    {"bwd", "apply", 512, 16, 8, 8, "f16"},      {"bwd", "reduce", 1024, 32, 2, 2, "bf16"},
    {"bwd", "apply", 1024, 32, 2, 2, "bf16"},    {"train", "reduce", 256, 8, 4, 4, "bf16"},
    {"train", "apply", 256, 8, 4, 4, "bf16"},
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
    const bn_nhwc_cfg_t* cfg = &CONFIGS[idx];
    rocke_batchnorm2d_nhwc_spec_t spec = rocke_batchnorm2d_nhwc_spec_default();
    spec.mode = cfg->mode;
    spec.block_size = cfg->block_size;
    spec.tx = cfg->tx;
    spec.vec = cfg->vec;
    spec.unroll = cfg->unroll;
    spec.dtype = cfg->dtype;

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_batchnorm2d_nhwc_new(&b, &spec, cfg->kind);
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
