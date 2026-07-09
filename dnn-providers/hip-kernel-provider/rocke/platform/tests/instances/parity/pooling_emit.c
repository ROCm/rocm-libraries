/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/pooling_emit.c -- C-side emitter for the pooling2d instance
 * parity harness. Selects one of the sampled configs by argv[1] (the config
 * index), builds rocke_pooling2d_spec_t identically to the Python emitter
 * pooling_emit.py, inits a fresh IRBuilder with the spec kernel name, builds
 * into it via rocke_build_pooling2d (the C build entry), lowers via
 * rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_pooling.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_pooling2d_spec_t* spec)
{
    *spec = rocke_pooling2d_spec_default();
    rocke_pooling_problem_t* p = &spec->problem;

    switch(idx)
    {
    case 0:
        p->N = 1;
        p->H = 28;
        p->W = 28;
        p->C = 64;
        p->Y = 2;
        p->X = 2;
        p->sH = 2;
        p->sW = 2;
        p->pH = 0;
        p->pW = 0;
        p->dH = 1;
        p->dW = 1;
        spec->dtype = "f16";
        spec->op = "max";
        spec->block_size = 256;
        spec->vec = 1;
        break;
    case 1:
        p->N = 2;
        p->H = 56;
        p->W = 56;
        p->C = 128;
        p->Y = 3;
        p->X = 3;
        p->sH = 1;
        p->sW = 1;
        p->pH = 1;
        p->pW = 1;
        p->dH = 1;
        p->dW = 1;
        spec->dtype = "f16";
        spec->op = "avg";
        spec->block_size = 256;
        spec->vec = 2;
        break;
    case 2:
        p->N = 4;
        p->H = 112;
        p->W = 112;
        p->C = 256;
        p->Y = 7;
        p->X = 7;
        p->sH = 7;
        p->sW = 7;
        p->pH = 0;
        p->pW = 0;
        p->dH = 1;
        p->dW = 1;
        spec->dtype = "f16";
        spec->op = "sum";
        spec->block_size = 256;
        spec->vec = 4;
        break;
    case 3:
        p->N = 1;
        p->H = 224;
        p->W = 224;
        p->C = 64;
        p->Y = 2;
        p->X = 2;
        p->sH = 2;
        p->sW = 2;
        p->pH = 0;
        p->pW = 0;
        p->dH = 1;
        p->dW = 1;
        spec->dtype = "bf16";
        spec->op = "max";
        spec->block_size = 256;
        spec->vec = 1;
        break;
    case 4:
        p->N = 2;
        p->H = 32;
        p->W = 32;
        p->C = 256;
        p->Y = 3;
        p->X = 3;
        p->sH = 1;
        p->sW = 1;
        p->pH = 1;
        p->pW = 1;
        p->dH = 1;
        p->dW = 1;
        spec->dtype = "f16";
        spec->op = "avg";
        spec->block_size = 128;
        spec->vec = 8;
        break;
    case 5:
        p->N = 1;
        p->H = 64;
        p->W = 64;
        p->C = 128;
        p->Y = 2;
        p->X = 2;
        p->sH = 2;
        p->sW = 2;
        p->pH = 0;
        p->pW = 0;
        p->dH = 1;
        p->dW = 1;
        spec->dtype = "bf16";
        spec->op = "sum";
        spec->block_size = 512;
        spec->vec = 4;
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

    rocke_pooling2d_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char* arch = "gfx950";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = 0;
    if(!rocke_pooling2d_is_valid_spec(&spec, arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* Init IRBuilder with spec.kernel_name() then build into it via the
     * task-specified entry rocke_build_pooling2d(b, &spec, arch). */
    char name[ROCKE_ERR_MSG_CAP];
    if(rocke_pooling2d_kernel_name(&spec, name, sizeof name) != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    rocke_kernel_def_t* kernel = rocke_build_pooling2d(&b, &spec, arch);
    if(!kernel)
    {
        fprintf(stderr, "build failed: %s\n", b.err);
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
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
