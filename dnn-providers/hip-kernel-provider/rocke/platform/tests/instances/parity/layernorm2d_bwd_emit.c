/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/layernorm2d_bwd_emit.c -- C-side emitter for the LayerNorm2D
 * backward parity harness. Selects one config by argv[1] (the config index),
 * builds rocke_layernorm2d_bwd_spec_t identically to the Python emitter
 * layernorm2d_bwd_emit.py, lowers (arch gfx950, flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * The config table MUST stay in lockstep with CONFIGS in layernorm2d_bwd_emit.py.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_layernorm2d_bwd.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct
{
    const char* part;
    int block_size;
    int vec;
    const char* dtype;
    bool fast_reduce;
    int unroll;
} lnb_cfg_t;

/* MUST match CONFIGS in layernorm2d_bwd_emit.py (index for index). */
static const lnb_cfg_t CONFIGS[] = {
    {"dx", 256, 4, "f16", false, 4}, /* 0 */
    {"dx", 64, 8, "bf16", false, 4}, /* 1 */
    {"dx", 128, 2, "bf16", true, 4}, /* 2 */
    {"dx", 1024, 8, "f16", true, 4}, /* 3 */
    {"dx", 512, 4, "fp16", false, 4}, /* 4 */
    {"dwdb", 64, 2, "bf16", false, 4}, /* 5 */
    {"dwdb", 64, 8, "bf16", false, 1}, /* 6 */
    {"dwdb", 256, 4, "f16", false, 8}, /* 7 */
    {"dwdb", 128, 2, "f16", false, 2}, /* 8 */
    {"dwdb_reduce", 64, 2, "bf16", false, 4}, /* 9 */
    {"dwdb_reduce", 256, 8, "f16", false, 1}, /* 10 */
    {"dwdb_reduce", 128, 4, "bf16", false, 8}, /* 11 */
};

#define NCFG ((int)(sizeof(CONFIGS) / sizeof(CONFIGS[0])))

static int make_spec(int idx, rocke_layernorm2d_bwd_spec_t* spec)
{
    if(idx < 0 || idx >= NCFG)
        return -1;
    *spec = rocke_layernorm2d_bwd_spec_default();
    spec->part = CONFIGS[idx].part;
    spec->block_size = CONFIGS[idx].block_size;
    spec->vec = CONFIGS[idx].vec;
    spec->dtype = CONFIGS[idx].dtype;
    spec->fast_reduce = CONFIGS[idx].fast_reduce;
    spec->unroll = CONFIGS[idx].unroll;
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

    rocke_layernorm2d_bwd_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_layernorm2d_bwd_new(&b, &spec);
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
