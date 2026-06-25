/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/add_rmsnorm2d_rdquant_emit.c -- C-side emitter for the
 * add_rmsnorm2d_rdquant instance parity harness. Selects one of the sampled
 * configs by argv[1] (the config index), builds
 * rocke_add_rmsnorm2d_rdquant_spec_t identically to the Python emitter
 * add_rmsnorm2d_rdquant_emit.py, validates via
 * rocke_add_rmsnorm2d_rdquant_is_valid_spec, builds into a fresh IRBuilder via
 * rocke_build_add_rmsnorm2d_rdquant_new (the C build entry), lowers via
 * rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_add_rmsnorm2d_rdquant.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_add_rmsnorm2d_rdquant_spec_t* spec)
{
    *spec = rocke_add_rmsnorm2d_rdquant_spec_default();

    switch(idx)
    {
    case 0:
        spec->n_per_block = 4096;
        spec->dtype = "f16";
        spec->out_dtype = "i8";
        spec->block_size = 256;
        spec->vec = 4;
        spec->save_residual = true;
        spec->save_yscale = true;
        spec->wave_size = 64;
        break;
    case 1:
        spec->n_per_block = 8192;
        spec->dtype = "f16";
        spec->out_dtype = "fp8e4m3";
        spec->block_size = 256;
        spec->vec = 8;
        spec->save_residual = true;
        spec->save_yscale = true;
        spec->wave_size = 64;
        break;
    case 2:
        spec->n_per_block = 2048;
        spec->dtype = "bf16";
        spec->out_dtype = "bf8e5m2";
        spec->block_size = 128;
        spec->vec = 4;
        spec->save_residual = false;
        spec->save_yscale = false;
        spec->wave_size = 64;
        break;
    case 3:
        spec->n_per_block = 16384;
        spec->dtype = "f16";
        spec->out_dtype = "i8";
        spec->block_size = 256;
        spec->vec = 4;
        spec->save_residual = true;
        spec->save_yscale = true;
        spec->wave_size = 64;
        break;
    case 4:
        spec->n_per_block = 1024;
        spec->dtype = "bf16";
        spec->out_dtype = "i8";
        spec->block_size = 64;
        spec->vec = 4;
        spec->save_residual = true;
        spec->save_yscale = true;
        spec->wave_size = 64;
        break;
    case 5:
        spec->n_per_block = 6144;
        spec->dtype = "f16";
        spec->out_dtype = "fp8e4m3";
        spec->block_size = 256;
        spec->vec = 2;
        spec->save_residual = true;
        spec->save_yscale = false;
        spec->wave_size = 64;
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
        fprintf(stderr, "usage: %s <config_index 0..5> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_add_rmsnorm2d_rdquant_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char* arch = "gfx950";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = 0;
    if(!rocke_add_rmsnorm2d_rdquant_is_valid_spec(&spec, arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* Init IRBuilder with spec.kernel_name() and build into it. */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_add_rmsnorm2d_rdquant_new(&b, &spec, arch);
    if(!kernel)
    {
        fprintf(stderr, "build failed: %s\n", b.err);
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        /* lower_kernel_to_llvm(kernel, arch='gfx950'). */
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
