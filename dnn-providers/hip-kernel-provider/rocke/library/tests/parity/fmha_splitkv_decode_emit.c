/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_splitkv_decode_emit.c -- C-side emitter for the split-KV
 * decode FMHA parity harness. Selects one of 6 sampled FmhaFwdSplitKvDecodeSpec
 * configs by argv[1] (0..5) and a phase by argv[2] ("seg" or "reduce"), builds
 * rocke_fmha_splitkv_decode_spec_t identically to the Python emitter
 * fmha_splitkv_decode_emit.py, and lowers the chosen kernel to LLVM .ll via the
 * convenience lower entries (arch gfx950, flavor AUTO), printing the .ll to
 * stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/instance_fmha_splitkv_decode.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_fmha_splitkv_decode_spec_t* spec)
{
    rocke_fmha_shape_t shape;
    rocke_fmha_common_spec_t common;

    switch(idx)
    {
    case 0: /* H64 q8 kv8 f16 none, batch1, segs4 */
        shape = rocke_fmha_shape_default(64, 8, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_splitkv_decode_spec_default(common, 1, 4);
        break;
    case 1: /* H128 q8 kv8 f16 causal, batch2, segs8 */
        shape = rocke_fmha_shape_default(128, 8, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        *spec = rocke_fmha_splitkv_decode_spec_default(common, 2, 8);
        break;
    case 2: /* H192 q16 kv2 bf16 none, batch4, segs16, use_mfma_body=False */
        shape = rocke_fmha_shape_default(192, 16, 2);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_splitkv_decode_spec_default(common, 4, 16);
        spec->use_mfma_body = false;
        break;
    case 3: /* H256 q32 kv4 f16 sliding_window 2048, batch1, segs32 */
        shape = rocke_fmha_shape_default(256, 32, 4);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_SLIDING_WINDOW;
        common.sliding_window = 2048;
        *spec = rocke_fmha_splitkv_decode_spec_default(common, 1, 32);
        break;
    case 4: /* H64 q12 kv3 bf16 none, batch8, segs64 */
        shape = rocke_fmha_shape_default(64, 12, 3);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_splitkv_decode_spec_default(common, 8, 64);
        break;
    case 5: /* H128 q16 kv8 f16 causal, batch2, segs128 */
        shape = rocke_fmha_shape_default(128, 16, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        *spec = rocke_fmha_splitkv_decode_spec_default(common, 2, 128);
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
        fprintf(stderr, "usage: %s <config_index 0..5> [<seg|reduce>|<ll|ir|verify>]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    /* argv[2] may be a phase ("seg"/"reduce") or a mode ("ll"/"ir"/"verify").
     * If it looks like a mode, treat it as mode with default phase "seg".
     * If it looks like a phase, treat it as phase with default mode "ll".
     * If absent, default phase="seg", mode="ll" (runner ll-mode passes no argv[2]). */
    const char* phase = "seg";
    const char* mode = "ll";
    if(argc > 2)
    {
        const char* arg2 = argv[2];
        if(strcmp(arg2, "ir") == 0 || strcmp(arg2, "verify") == 0 || strcmp(arg2, "ll") == 0)
        {
            mode = arg2;
        }
        else if(strcmp(arg2, "seg") == 0 || strcmp(arg2, "reduce") == 0)
        {
            phase = arg2;
        }
        else
        {
            fprintf(stderr, "unknown phase %s (want seg|reduce)\n", arg2);
            return 2;
        }
    }

    rocke_fmha_splitkv_decode_spec_t spec;
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
        rocke_status_t st;
        if(strcmp(phase, "seg") == 0)
        {
            st = rocke_fmha_splitkv_decode_segment_lower_to_llvm(
                &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        }
        else
        {
            st = rocke_fmha_splitkv_decode_reduce_lower_to_llvm(
                &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        }
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
    rocke_kernel_def_t* kernel;
    if(strcmp(phase, "seg") == 0)
    {
        kernel = rocke_build_fmha_fwd_splitkv_decode_segment(&kb, &spec, "gfx950");
    }
    else
    {
        kernel = rocke_build_fmha_fwd_splitkv_decode_reduce(&kb, &spec, "gfx950");
    }
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
