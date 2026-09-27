/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/smoothquant_emit.c -- C-side emitter for the smoothquant parity
 * harness. Selects one of the sampled SmoothQuantSpec configs by argv[1] (the
 * config index), builds the spec, lowers via rocke_build_smoothquant(b, spec,
 * arch) then rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO), and prints
 * the .ll to stdout so the output can be byte-compared with the Python emitter
 * smoothquant_emit.py.
 *
 * Optional argv[2] = mode: "ll" (default), "ir", "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_smoothquant.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Build the SmoothQuantSpec for config `idx`. Returns 0 or -1. */
static int make_spec(int idx, rocke_smoothquant_spec_t* s)
{
    rocke_smoothquant_spec_init(s, 0);
    switch(idx)
    {
    case 0: /* N1024 f16->i8 b256 v4 */
        s->n_per_block = 1024;
        s->dtype = "f16";
        s->out_dtype = "i8";
        s->block_size = 256;
        s->vec = 4;
        break;
    case 1: /* N2048 bf16->i8 b256 v8 */
        s->n_per_block = 2048;
        s->dtype = "bf16";
        s->out_dtype = "i8";
        s->block_size = 256;
        s->vec = 8;
        break;
    case 2: /* N512 f16->fp8e4m3 b128 v4 */
        s->n_per_block = 512;
        s->dtype = "f16";
        s->out_dtype = "fp8e4m3";
        s->block_size = 128;
        s->vec = 4;
        break;
    case 3: /* N1024 bf16->bf8e5m2 b64 v2 */
        s->n_per_block = 1024;
        s->dtype = "bf16";
        s->out_dtype = "bf8e5m2";
        s->block_size = 64;
        s->vec = 2;
        break;
    case 4: /* N2048 f16->i8 b512 v4 */
        s->n_per_block = 2048;
        s->dtype = "f16";
        s->out_dtype = "i8";
        s->block_size = 512;
        s->vec = 4;
        break;
    case 5: /* N256 bf16->fp8e4m3 b256 v2 */
        s->n_per_block = 256;
        s->dtype = "bf16";
        s->out_dtype = "fp8e4m3";
        s->block_size = 256;
        s->vec = 2;
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
        fprintf(stderr, "usage: %s <config_index 0..5> [mode]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_smoothquant_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char name[256];
    if(rocke_smoothquant_kernel_name(&spec, name, sizeof name) != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed for config %d\n", idx);
        return 1;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_build_smoothquant(&b, &spec, "gfx950");
    if(kernel == NULL || !rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "build failed for config %d: %s\n", idx, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
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
        /* mode == "ll" */
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
    rocke_ir_builder_free(&b);
    return 0;
}
