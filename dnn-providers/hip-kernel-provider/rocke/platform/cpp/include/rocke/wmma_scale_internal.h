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
#include "rocke/storage.h"

typedef struct rocke_scale_packing
{
    /* Eight-bit encoded scales for consecutive K groups, first group in the
     * low byte, independently for A/B. */
    int count;
    int block_k;
} rocke_scale_packing_t;

typedef struct rocke_scale_association
{
    int block_k;
} rocke_scale_association_t;

static inline rocke_scale_association_t rocke_scale_association(const rocke_scale_packing_t* p)
{
    if(p->block_k <= 0)
        ckc::raise_status(ROCKE_ERR_VALUE, "scale block_k must be positive");
    return {p->block_k};
}

static inline rocke_fragment_packing_t rocke_scale_fragment(const rocke_scale_packing_t* p)
{
    rocke_bit_packing_t bits;
    rocke_fragment_packing_t result;
    rocke_scale_association(p);
    if(p->count != 1 && p->count != 2 && p->count != 4 && p->count != 8)
        ckc::raise_status(ROCKE_ERR_VALUE, "carrier_bits must be 8, 16, 32, or 64");
    if(!rocke_bit_packing_init(&bits, rocke_dtype_info("e8m0")->encoded_bits, 0)
       || !rocke_fragment_packing_init(&result, &bits, p->count, p->count * 8, 1))
        ckc::raise_status(ROCKE_ERR_VALUE, "invalid scale packing");
    return result;
}

static inline rocke_matrix_fragment_layout_t rocke_scaled_matrix_layout(const char* dtype,
                                                                        int abi_words)
{
    const rocke_dtype_info_t* info = rocke_dtype_info(dtype);
    int chunk = 0;
    if(info && (strcmp(info->name, "fp8e4m3") == 0 || strcmp(info->name, "bf8e5m2") == 0))
        chunk = 16;
    else if(info
            && (strcmp(info->name, "fp4e2m1") == 0 || strcmp(info->name, "fp6e2m3") == 0
                || strcmp(info->name, "fp6e3m2") == 0))
        chunk = 32;
    rocke_bit_packing_t bits;
    rocke_fragment_packing_t fragment;
    rocke_matrix_fragment_layout_t result;
    if(!chunk || abi_words < 0 || !rocke_bit_packing_init(&bits, info->encoded_bits, 0)
       || !rocke_fragment_packing_init(&fragment, &bits, 64, 32, abi_words)
       || !rocke_matrix_fragment_layout_init(&result, &fragment, chunk, 2, 16))
        ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled matrix layout");
    return result;
}

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

static inline rocke_matrix_fragment_layout_t
    rocke_scaled_wmma_matrix_layout(const rocke_mma_op_t* atom, bool for_b)
{
    return rocke_scaled_matrix_layout(for_b ? atom->b_dtype : atom->a_dtype,
                                      for_b ? atom->b_frag_len : atom->a_frag_len);
}

static inline rocke_scaled_wmma_op_t rocke_scaled_wmma_contract(const rocke_mma_op_t* atom)
{
    rocke_scaled_wmma_op_t spec = {};
    spec.op_id = atom->op_id;
    const char* dtypes[2] = {atom->a_dtype, atom->b_dtype};
    const int words[2] = {atom->a_frag_len, atom->b_frag_len};
    const char* scale_dtypes[2] = {atom->a_scale_dtype, atom->b_scale_dtype};
    for(int i = 0; i < 2; ++i)
    {
        if(strcmp(dtypes[i], "fp8e4m3") == 0)
            spec.matrix_formats[i] = 0;
        else if(strcmp(dtypes[i], "bf8e5m2") == 0)
            spec.matrix_formats[i] = 1;
        else if(strcmp(dtypes[i], "fp6e2m3") == 0)
            spec.matrix_formats[i] = 2;
        else if(strcmp(dtypes[i], "fp6e3m2") == 0)
            spec.matrix_formats[i] = 3;
        else
            ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled WMMA matrix format");
        if(!scale_dtypes[i] || strcmp(scale_dtypes[i], "e8m0") != 0)
            ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled WMMA scale format");
        spec.scale_formats[i] = 0; // E8M0.
        spec.matrix_words[i] = words[i];
    }
    if((atom->scale_block_k != 16 && atom->scale_block_k != 32)
       || strcmp(atom->c_dtype, "fp32") != 0 || atom->m != 16 || atom->n != 16 || atom->k != 128)
        ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled WMMA backend contract");
    spec.scales.block_k = atom->scale_block_k;
    const int count = atom->k / spec.scales.block_k;
    if(atom->a_scale_frag_len != count || atom->b_scale_frag_len != count)
        ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled WMMA scale fragment lengths");
    spec.scales.count = atom->a_scale_frag_len;
    char suffix[96];
    snprintf(suffix,
             sizeof(suffix),
             "f32.%dx%dx%d.f8f6f4.v%df32.v%di32.v%di32",
             atom->m,
             atom->n,
             atom->k,
             atom->c_frag_len,
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
    return rocke_scale_fragment(packing).carrier_bits;
}

#endif /* ROCKE_WMMA_SCALE_INTERNAL_H */
