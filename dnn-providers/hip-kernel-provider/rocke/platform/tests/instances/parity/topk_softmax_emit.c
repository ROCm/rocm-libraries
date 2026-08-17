/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/topk_softmax_emit.c -- C-side emitter for the topk-softmax
 * instance parity harness. Selects one of the sampled configs by argv[1] (the
 * config index), builds rocke_topk_softmax_spec_t identically to the Python
 * emitter topk_softmax_emit.py, validates via rocke_topk_softmax_is_valid_spec,
 * builds into a fresh IRBuilder via rocke_build_topk_softmax_new (the C build
 * entry), lowers via rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and
 * prints the .ll to stdout so the two outputs can be byte-compared.
 *
 * Optional argv[2] = mode: "ll" (default), "ir", "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_topk_softmax.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_topk_softmax_spec_t* spec)
{
    *spec = rocke_topk_softmax_spec_default();

    switch(idx)
    {
    case 0:
        spec->n_per_row = 32;
        spec->k = 1;
        spec->dtype = "f32";
        spec->out_dtype = "f32";
        spec->block_size = 32;
        break;
    case 1:
        spec->n_per_row = 64;
        spec->k = 4;
        spec->dtype = "f16";
        spec->out_dtype = "f32";
        spec->block_size = 64;
        break;
    case 2:
        spec->n_per_row = 128;
        spec->k = 8;
        spec->dtype = "bf16";
        spec->out_dtype = "bf16";
        spec->block_size = 64;
        break;
    case 3:
        spec->n_per_row = 4096;
        spec->k = 16;
        spec->dtype = "f32";
        spec->out_dtype = "f32";
        spec->block_size = 128;
        break;
    case 4:
        spec->n_per_row = 16384;
        spec->k = 32;
        spec->dtype = "f32";
        spec->out_dtype = "f32";
        spec->block_size = 256;
        break;
    case 5:
        spec->n_per_row = 768;
        spec->k = 2;
        spec->dtype = "f32";
        spec->out_dtype = "f32";
        spec->block_size = 64;
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

    rocke_topk_softmax_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char* arch = "gfx950";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = 0;
    if(!rocke_topk_softmax_is_valid_spec(&spec, arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* Init IRBuilder with spec.kernel_name() and build into it. */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_topk_softmax_new(&b, &spec, arch);
    if(!kernel)
    {
        fprintf(stderr, "build failed: %s\n", b.err);
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
    rocke_ir_builder_free(&b);
    return 0;
}
