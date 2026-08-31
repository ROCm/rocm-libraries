/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_fwd_fp8_emit.c -- C-side emitter for the FP8 FMHA forward
 * parity harness. Selects one of 6 sampled configs by argv[1] (the config index
 * 0..5), builds rocke_fmha_fwd_fp8_spec_t identically to the Python emitter
 * fmha_fwd_fp8_emit.py, lowers via rocke_fmha_fwd_fp8_lower_to_llvm (arch gfx950,
 * flavor AUTO) and prints the .ll to stdout so the two outputs can be
 * byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/instance_fmha_fwd_fp8.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_fmha_fwd_fp8_spec_t* spec)
{
    rocke_fmha_shape_t shape;
    rocke_fmha_common_spec_t common;

    switch(idx)
    {
    case 0:
        shape = rocke_fmha_shape_make(64, 4, 4, 16, 64);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = ROCKE_KV_FP8_E4M3;
        spec->seqlen_q = 16;
        spec->seqlen_k = 64;
        break;
    case 1:
        shape = rocke_fmha_shape_make(64, 2, 2, 16, 64);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        *spec = rocke_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = ROCKE_KV_FP8_E4M3;
        spec->seqlen_q = 32;
        spec->seqlen_k = 128;
        break;
    case 2:
        shape = rocke_fmha_shape_make(128, 8, 4, 16, 64);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = ROCKE_KV_BF8_E5M2;
        spec->seqlen_q = 16;
        spec->seqlen_k = 64;
        break;
    case 3:
        shape = rocke_fmha_shape_make(256, 4, 1, 16, 64);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_SLIDING_WINDOW;
        common.sliding_window = 32;
        *spec = rocke_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = ROCKE_KV_FP8_E4M3;
        spec->seqlen_q = 48;
        spec->seqlen_k = 256;
        spec->has_waves_per_eu = true;
        spec->waves_per_eu = 4;
        break;
    case 4:
        shape = rocke_fmha_shape_make(32, 16, 16, 16, 64);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = ROCKE_KV_FP8_E4M3;
        spec->seqlen_q = 64;
        spec->seqlen_k = 256;
        break;
    case 5:
        shape = rocke_fmha_shape_make(64, 8, 2, 16, 64);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_fwd_fp8_spec_default();
        spec->common = common;
        spec->kv_dtype = ROCKE_KV_BF8_E5M2;
        spec->seqlen_q = 32;
        spec->seqlen_k = 512;
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

    rocke_fmha_fwd_fp8_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_fmha_fwd_fp8_lower_to_llvm(
            &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
        return 0;
    }

    /* For ir/verify modes, build the kernel explicitly. */
    rocke_fmha_kernel_builder_t kb;
    memset(&kb, 0, sizeof kb);
    rocke_kernel_def_t* kernel = rocke_build_fmha_fwd_fp8_new(&kb, &spec, "gfx950");
    if(!kernel)
    {
        fprintf(stderr, "build failed for config %d\n", idx);
        rocke_fmha_kernel_builder_free(&kb);
        return 1;
    }

    if(strcmp(mode, "ir") == 0)
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
