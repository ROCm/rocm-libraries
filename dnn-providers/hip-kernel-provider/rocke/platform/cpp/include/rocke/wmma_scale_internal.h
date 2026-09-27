// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Shared gfx1250 operand contracts. Mirrors core/arch/wmma_scale.py:
 * ScalePacking    -> rocke_scale_packing_t
 * ScaledWmmaOp     -> rocke_scaled_wmma_op_t
 * gfx1250_scaled_wmma -> rocke_gfx1250_scaled_wmma
 */
#ifndef ROCKE_WMMA_SCALE_INTERNAL_H
#define ROCKE_WMMA_SCALE_INTERNAL_H

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/ir.h"

typedef struct rocke_scale_packing
{
    /* Eight-bit encoded scales for consecutive K groups, first group in the
     * low byte, independently for A/B. */
    int count;
    int block_k;
} rocke_scale_packing_t;

typedef struct rocke_scaled_wmma_op
{
    const char* op_id;
    int matrix_formats[2];
    int scale_formats[2];
    int matrix_words[2];
    char declaration_key[160];
    char intrinsic[160];
    rocke_scale_packing_t scales;
} rocke_scaled_wmma_op_t;

static inline const rocke_mma_op_t* rocke_gfx1250_scaled_wmma(const char* op_id)
{
    if(!op_id)
        return NULL;
    if(strncmp(op_id, "tile.", 5) == 0)
        op_id += 5;
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx("gfx1250");
    const rocke_mma_op_t* atom = rocke_mma_catalog_by_op_id(&target->mma, op_id);
    return atom && strcmp(atom->family, "wmma_scaled") == 0 ? atom : NULL;
}

static inline rocke_scaled_wmma_op_t rocke_scaled_wmma_contract(const rocke_mma_op_t* atom)
{
    rocke_scaled_wmma_op_t spec = {};
    spec.op_id = atom->op_id;
    const char* dtypes[2] = {atom->srcs[0].dtype, atom->srcs[1].dtype};
    const int words[2] = {atom->srcs[0].frag_len, atom->srcs[1].frag_len};
    const char* scale_dtypes[2] = {atom->srcs[0].scale_dtype, atom->srcs[1].scale_dtype};
    for(int i = 0; i < 2; ++i)
    {
        if(strcmp(dtypes[i], "fp8e4m3") == 0)
            spec.matrix_formats[i] = 0;
        else if(strcmp(dtypes[i], "bf8e5m2") == 0)
            spec.matrix_formats[i] = 1;
        else
            ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled WMMA matrix format");
        if(!scale_dtypes[i] || strcmp(scale_dtypes[i], "e8m0") != 0)
            ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled WMMA scale format");
        spec.scale_formats[i] = 0; // E8M0.
        spec.matrix_words[i] = words[i];
    }
    if((atom->srcs[0].scale_block_size != 16 && atom->srcs[0].scale_block_size != 32)
       || atom->srcs[0].scale_block_size != atom->srcs[1].scale_block_size
       || strcmp(atom->srcs[2].dtype, "fp32") != 0 || strcmp(atom->dst.dtype, "fp32") != 0
       || atom->m != 16 || atom->n != 16 || atom->k != 128)
        ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled WMMA backend contract");
    spec.scales.block_k = atom->srcs[0].scale_block_size;
    const int count = atom->k / spec.scales.block_k;
    if(atom->srcs[0].scale_frag_len != count || atom->srcs[1].scale_frag_len != count)
        ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled WMMA scale fragment lengths");
    spec.scales.count = atom->srcs[0].scale_frag_len;
    char suffix[96];
    snprintf(suffix,
             sizeof(suffix),
             "f32.%dx%dx%d.f8f6f4.v%df32.v%di32.v%di32",
             atom->m,
             atom->n,
             atom->k,
             atom->dst.frag_len,
             spec.matrix_words[0],
             spec.matrix_words[1]);
    snprintf(spec.intrinsic,
             sizeof(spec.intrinsic),
             "llvm.amdgcn.wmma.%s.%s",
             spec.scales.block_k == 16 ? "scale16" : "scale",
             suffix);
    snprintf(spec.declaration_key,
             sizeof(spec.declaration_key),
             "wmma.scale.block%d.gfx1250.%s",
             spec.scales.block_k,
             suffix);
    return spec;
}

static inline const rocke_mma_op_t* rocke_gfx1250_scaled_wmma_from_op(const rocke_op_t* op)
{
    const char* op_id = rocke_attr_get_str(&op->attrs, "op_id");
    return rocke_gfx1250_scaled_wmma(op_id ? op_id : op->name);
}

static inline int rocke_scale_word_bits(const rocke_scale_packing_t* packing)
{
    return packing->count * 8;
}

#endif /* ROCKE_WMMA_SCALE_INTERNAL_H */
