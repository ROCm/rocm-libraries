/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_winograd_emit.c -- C-side emitter for the Winograd
 * convolution parity harness.
 *
 * Selects one of N sampled spec configs by argv[1], builds the appropriate
 * Winograd transform kernel via the C++ engine, and lowers it to LLVM IR so
 * run_diff.py can byte-compare it with the Python reference
 * conv_winograd_emit.py.
 *
 * Config index layout — mirrors conv_winograd_emit.py exactly:
 *   idx 0,1,2  — F(4,3) N8 H56 W56 C64 K64  gfx950  data/filter/output
 *   idx 3,4,5  — F(2,3) N8 H56 W56 C64 K64  gfx950  data/filter/output
 *   idx 6,7,8  — F(4,3) N4 H28 W28 C128 K128 gfx950  data/filter/output
 *   idx 9,10,11— F(4,3) N1 H7 W7 C512 K512  gfx942  data/filter/output
 *
 * Sub-kernel: idx % 3 → 0=data, 1=filter, 2=output.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_winograd.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Sub-kernel selector */
#define SUB_DATA 0
#define SUB_FILTER 1
#define SUB_OUTPUT 2

/* Fill spec and arch for config index idx.
 * Returns 0 on success, -1 if idx is out of range. */
static int make_cfg(int idx, rocke_winograd_conv_spec_t* spec, const char** arch, int* sub)
{
    *spec = rocke_winograd_conv_spec_default();
    *sub = idx % 3;

    switch(idx / 3)
    {
    case 0: /* F(4,3) N8 H56 W56 C64 K64 gfx950 */
        spec->problem = rocke_winograd_problem_default(8, 56, 56, 64, 64);
        spec->out_tile = 4;
        spec->block_c = 32;
        spec->block_k = 32;
        spec->block_nhw = 4;
        *arch = "gfx950";
        return 0;

    case 1: /* F(2,3) N8 H56 W56 C64 K64 gfx950 */
        spec->problem = rocke_winograd_problem_default(8, 56, 56, 64, 64);
        spec->out_tile = 2;
        spec->block_c = 32;
        spec->block_k = 32;
        spec->block_nhw = 4;
        *arch = "gfx950";
        return 0;

    case 2: /* F(4,3) N4 H28 W28 C128 K128 gfx950 */
        spec->problem = rocke_winograd_problem_default(4, 28, 28, 128, 128);
        spec->out_tile = 4;
        spec->block_c = 32;
        spec->block_k = 32;
        spec->block_nhw = 4;
        *arch = "gfx950";
        return 0;

    case 3: /* F(4,3) N1 H7 W7 C512 K512 gfx942 */
        spec->problem = rocke_winograd_problem_default(1, 7, 7, 512, 512);
        spec->out_tile = 4;
        spec->block_c = 32;
        spec->block_k = 32;
        spec->block_nhw = 1;
        *arch = "gfx942";
        return 0;

    default:
        return -1;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: conv_winograd_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }

    int idx = atoi(argv[1]);
    const char* mode = (argc >= 3) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_winograd_conv_spec_t spec;
    const char* arch = NULL;
    int sub = 0;

    if(make_cfg(idx, &spec, &arch, &sub) != 0)
    {
        /* Signal "unknown config" — run_diff.py stops enumeration. */
        fprintf(stderr, "unknown config %d\n", idx);
        return 1;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = NULL;

    switch(sub)
    {
    case SUB_DATA:
        kernel = rocke_build_winograd_data_transform_new(&b, &spec, arch);
        break;
    case SUB_FILTER:
        kernel = rocke_build_winograd_filter_transform_new(&b, &spec, arch);
        break;
    case SUB_OUTPUT:
        kernel = rocke_build_winograd_output_transform_new(&b, &spec, arch);
        break;
    }

    if(kernel == NULL)
    {
        fprintf(stderr, "build failed for config %d: %s\n", idx, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    int rc = 0;

    if(strcmp(mode, "ll") == 0)
    {
        char* ll_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &ll_text);
        if(st != ROCKE_OK || ll_text == NULL)
        {
            fprintf(stderr, "lower_kernel_to_llvm failed (status %d)\n", (int)st);
            rc = 1;
        }
        else
        {
            fputs(ll_text, stdout);
            free(ll_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* ir_text = NULL;
        rocke_status_t st2 = rocke_ir_serialize(kernel, &ir_text);
        if(st2 != ROCKE_OK || ir_text == NULL)
        {
            fprintf(stderr, "ir_serialize failed (status %d)\n", (int)st2);
            rc = 1;
        }
        else
        {
            fputs(ir_text, stdout);
            free(ir_text);
        }
    }
    else /* verify */
    {
        rocke_diag_t* diags = NULL;
        size_t n_diags = 0;
        rocke_verify(kernel, &diags, &n_diags);
        for(size_t i = 0; i < n_diags; ++i)
            fprintf(stdout, "%s\n", diags[i].message);
        rocke_diags_free(diags, n_diags);
    }

    rocke_ir_builder_free(&b);
    return rc;
}
