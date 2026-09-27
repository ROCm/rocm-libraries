/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/rmsnorm2d_emit.c -- C-side emitter for the rmsnorm2d parity
 * harness. Selects one of the sampled RMSNorm2DSpec configs by argv[1] (the
 * config index), builds the spec, lowers via rocke_build_rmsnorm2d(b, spec, arch)
 * then rocke_lower_kernel_to_llvm_ex (arch gfx950, flavor AUTO), and prints the
 * .ll to stdout so the output can be byte-compared with the Python emitter
 * rmsnorm2d_emit.py.
 *
 * Optional argv[2] = mode: "ll" (default), "ir", "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_rmsnorm2d.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Build the RMSNorm2DSpec for config `idx`. Returns 0 or -1. */
static int make_spec(int idx, rocke_rmsnorm2d_spec_t* s)
{
    *s = rocke_rmsnorm2d_spec_default();
    switch(idx)
    {
    case 0: /* N1024 b256 v4 f16 */
        s->n_per_block = 1024;
        s->block_size = 256;
        s->vec = 4;
        s->dtype = "f16";
        s->save_inv_rms = false;
        break;
    case 1: /* N2048 b256 v4 bf16 */
        s->n_per_block = 2048;
        s->block_size = 256;
        s->vec = 4;
        s->dtype = "bf16";
        s->save_inv_rms = false;
        break;
    case 2: /* N4096 b256 v4 f16 */
        s->n_per_block = 4096;
        s->block_size = 256;
        s->vec = 4;
        s->dtype = "f16";
        s->save_inv_rms = false;
        break;
    case 3: /* N8192 b256 v8 bf16 */
        s->n_per_block = 8192;
        s->block_size = 256;
        s->vec = 8;
        s->dtype = "bf16";
        s->save_inv_rms = false;
        break;
    case 4: /* N16384 b256 v8 f16 save_inv_rms */
        s->n_per_block = 16384;
        s->block_size = 256;
        s->vec = 8;
        s->dtype = "f16";
        s->save_inv_rms = true;
        break;
    case 5: /* N131072 b256 v4 f16 (two-pass) */
        s->n_per_block = 131072;
        s->block_size = 256;
        s->vec = 4;
        s->dtype = "f16";
        s->save_inv_rms = false;
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

    rocke_rmsnorm2d_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    char name[256];
    if(rocke_rmsnorm2d_kernel_name(&spec, name, sizeof name) != ROCKE_OK)
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

    rocke_kernel_def_t* kernel = rocke_build_rmsnorm2d(&b, &spec, "gfx950");
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
