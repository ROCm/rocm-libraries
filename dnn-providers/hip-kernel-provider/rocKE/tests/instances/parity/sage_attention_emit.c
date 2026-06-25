/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/sage_attention_emit.c -- C-side emitter for the Sage attention
 * forward (instance_sage_attention) parity harness. Selects one of the sampled
 * configs by argv[1] (the config index 0..5), builds ckc_sage_attention_spec_t
 * identically to the Python emitter sage_attention_emit.py, builds + lowers via
 * ckc_sage_attention_lower_to_llvm (arch gfx950, flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * Optional argv[2] = mode: "ll" (default), "ir", "verify".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/helper_ck_dsl.helpers.qk_scale.h"
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"
#include "ckc/instance_sage_attention.h"
#include "ckc/ir.h"
#include "ckc/ir_serialize.h"
#include "ckc/lower_llvm.h"
#include "ckc/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, ckc_sage_attention_spec_t* spec)
{
    ckc_fmha_shape_t shape;
    ckc_fmha_common_spec_t common;
    ckc_qk_scale_spec_t qs, ks;

    *spec = ckc_sage_attention_spec_default();

    switch(idx)
    {
    case 0:
        shape = ckc_fmha_shape_make(64, 8, 8, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        qs.layout = CKC_QK_SCALE_PER_BLOCK;
        qs.scale_block = 16;
        qs.stride_batch = 128;
        qs.stride_head = 8;
        qs.stride_block = 1;
        ks.layout = CKC_QK_SCALE_PER_BLOCK;
        ks.scale_block = 64;
        ks.stride_batch = 128;
        ks.stride_head = 8;
        ks.stride_block = 1;
        spec->common = common;
        spec->quant_mode = CKC_SAGE_QUANT_FP16_BF16;
        spec->q_scale = qs;
        spec->k_scale = ks;
        spec->seqlen_q = 16;
        spec->seqlen_k = 64;
        break;
    case 1:
        shape = ckc_fmha_shape_make(64, 8, 8, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        qs.layout = CKC_QK_SCALE_PER_BLOCK;
        qs.scale_block = 16;
        qs.stride_batch = 128;
        qs.stride_head = 8;
        qs.stride_block = 1;
        ks.layout = CKC_QK_SCALE_PER_BLOCK;
        ks.scale_block = 64;
        ks.stride_batch = 128;
        ks.stride_head = 8;
        ks.stride_block = 1;
        spec->common = common;
        spec->quant_mode = CKC_SAGE_QUANT_FP8_BF16;
        spec->q_scale = qs;
        spec->k_scale = ks;
        spec->seqlen_q = 16;
        spec->seqlen_k = 64;
        break;
    case 2:
        shape = ckc_fmha_shape_make(64, 8, 8, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        qs.layout = CKC_QK_SCALE_PER_HEAD;
        qs.scale_block = 0;
        qs.stride_batch = 8;
        qs.stride_head = 1;
        qs.stride_block = 1;
        ks.layout = CKC_QK_SCALE_PER_HEAD;
        ks.scale_block = 0;
        ks.stride_batch = 8;
        ks.stride_head = 1;
        ks.stride_block = 1;
        spec->common = common;
        spec->quant_mode = CKC_SAGE_QUANT_I8_FP8_BF16;
        spec->q_scale = qs;
        spec->k_scale = ks;
        spec->seqlen_q = 16;
        spec->seqlen_k = 64;
        break;
    case 3:
        shape = ckc_fmha_shape_make(128, 8, 8, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        qs.layout = CKC_QK_SCALE_PER_BLOCK;
        qs.scale_block = 16;
        qs.stride_batch = 128;
        qs.stride_head = 8;
        qs.stride_block = 1;
        ks.layout = CKC_QK_SCALE_PER_BLOCK;
        ks.scale_block = 64;
        ks.stride_batch = 128;
        ks.stride_head = 8;
        ks.stride_block = 1;
        spec->common = common;
        spec->quant_mode = CKC_SAGE_QUANT_I4_FP8_BF16;
        spec->q_scale = qs;
        spec->k_scale = ks;
        spec->seqlen_q = 32;
        spec->seqlen_k = 128;
        break;
    case 4:
        shape = ckc_fmha_shape_make(256, 16, 8, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = CKC_FMHA_MASK_CAUSAL;
        qs.layout = CKC_QK_SCALE_PER_BLOCK;
        qs.scale_block = 32;
        qs.stride_batch = 256;
        qs.stride_head = 16;
        qs.stride_block = 1;
        ks.layout = CKC_QK_SCALE_PER_BLOCK;
        ks.scale_block = 64;
        ks.stride_batch = 256;
        ks.stride_head = 16;
        ks.stride_block = 1;
        spec->common = common;
        spec->quant_mode = CKC_SAGE_QUANT_FP16_BF16;
        spec->q_scale = qs;
        spec->k_scale = ks;
        spec->seqlen_q = 64;
        spec->seqlen_k = 64;
        break;
    case 5:
        shape = ckc_fmha_shape_make(128, 8, 8, 16, 64);
        common = ckc_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = CKC_FMHA_MASK_NONE;
        qs.layout = CKC_QK_SCALE_PER_HEAD;
        qs.scale_block = 0;
        qs.stride_batch = 8;
        qs.stride_head = 1;
        qs.stride_block = 1;
        ks.layout = CKC_QK_SCALE_PER_HEAD;
        ks.scale_block = 0;
        ks.stride_batch = 8;
        ks.stride_head = 1;
        ks.stride_block = 1;
        spec->common = common;
        spec->quant_mode = CKC_SAGE_QUANT_FP8_BF16;
        spec->q_scale = qs;
        spec->k_scale = ks;
        spec->seqlen_q = 32;
        spec->seqlen_k = 128;
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

    ckc_sage_attention_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    if(strcmp(mode, "ll") == 0)
    {
        /* Fast path: use the convenience lower (existing behavior). */
        char* llvm_text = NULL;
        char err[CKC_ERR_MSG_CAP];
        err[0] = 0;
        ckc_status_t st = ckc_sage_attention_lower_to_llvm(
            &spec, "gfx950", CKC_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if(st != CKC_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
        return 0;
    }

    /* For ir/verify: build first to get the kernel. */
    ckc_fmha_kernel_builder_t kb;
    ckc_kernel_def_t* kernel = ckc_build_sage_attention_new(&kb, &spec, "gfx950");
    if(!kernel)
    {
        fprintf(stderr, "build failed for config %d\n", idx);
        ckc_fmha_kernel_builder_free(&kb);
        return 1;
    }

    if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        ckc_status_t st = ckc_ir_serialize(kernel, &t);
        if(st != CKC_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            ckc_fmha_kernel_builder_free(&kb);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else
    {
        /* mode == "verify" */
        ckc_diag_t* d = NULL;
        size_t n = 0;
        ckc_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = ckc_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        ckc_diags_free(d, n);
    }
    ckc_fmha_kernel_builder_free(&kb);
    return 0;
}
