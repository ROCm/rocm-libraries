/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/moe_smoothquant_emit.c -- C-side emitter for the
 * moe_smoothquant parity harness. Selects one of the sampled
 * MoeSmoothQuantSpec configs by argv[1] (the config index), builds the spec,
 * lowers via rocke_build_moe_smoothquant_new(b, spec, arch) then
 * rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO), and prints the .ll to
 * stdout so the output can be byte-compared with the Python emitter
 * moe_smoothquant_emit.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_moe_smoothquant.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Build the MoeSmoothQuantSpec for config `idx`. Returns 0 or -1. */
static int make_spec(int idx, rocke_moe_smoothquant_spec_t* s)
{
    switch(idx)
    {
    case 0: /* N512 topk2 E64 f16->i8 b256 v4 */
        rocke_moe_smoothquant_spec_init(s, 512, 2, 64);
        s->dtype = "f16";
        s->out_dtype = "i8";
        s->block_size = 256;
        s->vec = 4;
        break;
    case 1: /* N1024 topk4 E128 bf16->fp8e4m3 b256 v4 */
        rocke_moe_smoothquant_spec_init(s, 1024, 4, 128);
        s->dtype = "bf16";
        s->out_dtype = "fp8e4m3";
        s->block_size = 256;
        s->vec = 4;
        break;
    case 2: /* N2048 topk8 E256 f16->i8 b256 v4 tokens=256 */
        rocke_moe_smoothquant_spec_init(s, 2048, 8, 256);
        s->dtype = "f16";
        s->out_dtype = "i8";
        s->block_size = 256;
        s->vec = 4;
        s->tokens_set = true;
        s->tokens = 256;
        break;
    case 3: /* N4096 topk1 E8 f16->i8 b512 v8 */
        rocke_moe_smoothquant_spec_init(s, 4096, 1, 8);
        s->dtype = "f16";
        s->out_dtype = "i8";
        s->block_size = 512;
        s->vec = 8;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..3> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_moe_smoothquant_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_moe_smoothquant_new(&b, &spec, "gfx950");
    if(kernel == NULL || !rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "build failed for config %d: %s\n", idx, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else if(strcmp(mode, "verify") == 0)
    {
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
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        rocke_ir_builder_free(&b);
        return 2;
    }
    rocke_ir_builder_free(&b);
    return 0;
}
