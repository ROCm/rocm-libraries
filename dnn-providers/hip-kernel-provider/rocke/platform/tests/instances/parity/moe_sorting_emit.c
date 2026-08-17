/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/moe_sorting_emit.c -- C-side emitter for the MoE-sorting
 * instance parity harness. Selects one of the sampled configs by argv[2] (the
 * config index) and the phase by argv[1] ("hist"/"scan"/"scatter"), builds
 * rocke_moe_sorting_spec_t identically to the Python emitter moe_sorting_emit.py,
 * validates via rocke_moe_sorting_is_valid_spec, builds into a fresh IRBuilder via
 * the matching rocke_build_moe_sort_*_new C build entry, lowers via
 * rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO) and prints the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_moe_sorting.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_moe_sorting_spec_t* spec)
{
    *spec = rocke_moe_sorting_spec_default();

    switch(idx)
    {
    case 0:
        spec->tokens = 2;
        spec->topk = 8;
        spec->experts = 8;
        spec->block_size = 64;
        break;
    case 1:
        spec->tokens = 16;
        spec->topk = 4;
        spec->experts = 32;
        spec->block_size = 256;
        break;
    case 2:
        spec->tokens = 32;
        spec->topk = 8;
        spec->experts = 64;
        spec->block_size = 256;
        break;
    case 3:
        spec->tokens = 128;
        spec->topk = 2;
        spec->experts = 32;
        spec->block_size = 512;
        break;
    case 4:
        spec->tokens = 8;
        spec->topk = 16;
        spec->experts = 16;
        spec->block_size = 128;
        break;
    case 5:
        spec->tokens = 2;
        spec->topk = 8;
        spec->experts = 64;
        spec->block_size = 256;
        break;
    default:
        return -1;
    }
    return 0;
}

/* Flat config index encoding:
 *   0.. 5 = hist    / spec 0..5
 *   6..11 = scan    / spec 0..5
 *  12..17 = scatter / spec 0..5
 * This allows the differential harness to call us as:
 *   moe_sorting_emit_c <flat_idx> [mode]
 */
static const char* phase_for(int flat)
{
    if(flat < 6)
        return "hist";
    if(flat < 12)
        return "scan";
    if(flat < 18)
        return "scatter";
    return NULL;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <flat_config_index 0..17> [mode]\n", argv[0]);
        return 2;
    }
    int flat = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    const char* phase = phase_for(flat);
    if(!phase)
    {
        fprintf(stderr, "unknown config index %d\n", flat);
        return 2;
    }
    int idx = flat % 6;

    rocke_moe_sorting_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char* arch = "gfx950";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[ROCKE_ERR_MSG_CAP];
    reason[0] = 0;
    if(!rocke_moe_sorting_is_valid_spec(&spec, arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* Select the phase build entry. */
    rocke_kernel_def_t* (*build_new)(
        rocke_ir_builder_t*, const rocke_moe_sorting_spec_t*, const char*)
        = NULL;
    if(strcmp(phase, "hist") == 0)
    {
        build_new = rocke_build_moe_sort_histogram_new;
    }
    else if(strcmp(phase, "scan") == 0)
    {
        build_new = rocke_build_moe_sort_scan_new;
    }
    else
    {
        build_new = rocke_build_moe_sort_scatter_new;
    }

    /* Init IRBuilder with spec.kernel_name(<phase>) and build into it. */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = build_new(&b, &spec, arch);
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
