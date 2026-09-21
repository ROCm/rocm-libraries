/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_varlen_emit.c -- C-side emitter for the varlen-FMHA-fwd
 * parity harness. Selects one of 6 sampled FmhaFwdVarlenSpec configs by argv[1]
 * (0..5), builds rocke_fmha_fwd_varlen_spec_t identically to the Python emitter
 * fmha_varlen_emit.py, builds via rocke_build_fmha_fwd_varlen and lowers via
 * rocke_lower_kernel_to_llvm (arch gfx950, flavor AUTO), printing the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcmp (C++ build) */

#include "rocke/instance_fmha_varlen.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_fmha_fwd_varlen_spec_t* spec)
{
    rocke_fmha_shape_t shape;
    rocke_fmha_common_spec_t common;

    switch(idx)
    {
    case 0:
        shape = rocke_fmha_shape_default(64, 8, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_fwd_varlen_spec_default(common, 128, 256, 4);
        break;
    case 1:
        shape = rocke_fmha_shape_default(128, 8, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        *spec = rocke_fmha_fwd_varlen_spec_default(common, 256, 512, 2);
        break;
    case 2:
        shape = rocke_fmha_shape_default(128, 16, 4);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_fwd_varlen_spec_default(common, 64, 128, 8);
        break;
    case 3:
        shape = rocke_fmha_shape_default(256, 4, 4);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = ROCKE_FMHA_MASK_SLIDING_WINDOW;
        common.sliding_window = 64;
        *spec = rocke_fmha_fwd_varlen_spec_default(common, 512, 512, 1);
        break;
    case 4:
        shape = rocke_fmha_shape_default(64, 12, 12);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "fp16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_fwd_varlen_spec_default(common, 256, 256, 16);
        break;
    case 5:
        shape = rocke_fmha_shape_default(192, 12, 12);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        *spec = rocke_fmha_fwd_varlen_spec_default(common, 192, 384, 8);
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
        fprintf(stderr, "usage: %s <config_index 0..5>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_fmha_fwd_varlen_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_fmha_kernel_builder_t kb;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = 0;
    rocke_kernel_def_t* kernel = rocke_build_fmha_fwd_varlen(&kb, &spec, "gfx950", err, sizeof err);
    if(!kernel)
    {
        fprintf(stderr, "build failed: err=%s\n", err);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char lerr[ROCKE_ERR_MSG_CAP];
        lerr[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, lerr, sizeof lerr);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, lerr);
            rocke_fmha_kernel_builder_free(&kb);
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
            rocke_fmha_kernel_builder_free(&kb);
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
        rocke_fmha_kernel_builder_free(&kb);
        return 2;
    }
    rocke_fmha_kernel_builder_free(&kb);
    return 0;
}
