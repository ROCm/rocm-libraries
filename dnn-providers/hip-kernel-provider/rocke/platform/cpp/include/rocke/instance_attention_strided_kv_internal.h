// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Tiled attention family internals. Mirrors library/kernels/common/
 * _attention_strided_kv.py: parameter declaration, per-head buffer resource,
 * and masked token/dim byte offsets. Not a public caller-owned spec ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_STRIDED_KV_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_STRIDED_KV_INTERNAL_H

#include <stdio.h>

#include "rocke/ir.h"

typedef struct rocke_strided_kv_params
{
    rocke_value_t* batch;
    rocke_value_t* head;
    rocke_value_t* token;
    rocke_value_t* span;
} rocke_strided_kv_params_t;

static inline rocke_strided_kv_params_t rocke_declare_strided_kv_params(rocke_ir_builder_t* b,
                                                                        const char* prefix)
{
    rocke_strided_kv_params_t p;
    char name[64];
    snprintf(name, sizeof(name), "%s_stride_batch_bytes", prefix);
    p.batch = rocke_b_param(b, name, rocke_i64(), NULL);
    snprintf(name, sizeof(name), "%s_stride_head_bytes", prefix);
    p.head = rocke_b_param(b, name, rocke_i64(), NULL);
    snprintf(name, sizeof(name), "%s_stride_token_bytes", prefix);
    p.token = rocke_b_param(b, name, rocke_i32(), NULL);
    snprintf(name, sizeof(name), "%s_span_bytes", prefix);
    p.span = rocke_b_param(b, name, rocke_i32(), NULL);
    return p;
}

static inline rocke_value_t* rocke_strided_kv_resource(rocke_ir_builder_t* b,
                                                       rocke_value_t* ptr,
                                                       const rocke_strided_kv_params_t* p,
                                                       rocke_value_t* seq,
                                                       rocke_value_t* head)
{
    rocke_value_t* batch_offset = rocke_b_mul(b, rocke_b_zext(b, seq, rocke_i64()), p->batch);
    rocke_value_t* head_offset = rocke_b_mul(b, rocke_b_zext(b, head, rocke_i64()), p->head);
    rocke_value_t* base = rocke_b_global_ptr_add(b, ptr, rocke_b_add(b, batch_offset, head_offset));
    return rocke_b_buffer_rsrc(b, base, p->span);
}

static inline rocke_value_t* rocke_strided_kv_offset(rocke_ir_builder_t* b,
                                                     const rocke_strided_kv_params_t* p,
                                                     rocke_value_t* tile,
                                                     rocke_value_t* linear,
                                                     rocke_value_t* valid_length,
                                                     int head_size,
                                                     int tile_size)
{
    rocke_value_t* tile_start = rocke_b_mul(b, tile, rocke_b_const_i32(b, tile_size));
    rocke_value_t* row = rocke_b_div(b, linear, rocke_b_const_i32(b, head_size));
    rocke_value_t* token = rocke_b_add(b, tile_start, row);
    rocke_value_t* dim = rocke_b_mod(b, linear, rocke_b_const_i32(b, head_size));
    rocke_value_t* valid = rocke_b_cmp_lt(b, token, valid_length);
    rocke_value_t* safe_token = rocke_b_select(b, valid, token, rocke_b_const_i32(b, 0));
    rocke_value_t* row_offset = rocke_b_mul(b, safe_token, p->token);
    rocke_value_t* dim_offset = rocke_b_mul(b, dim, rocke_b_const_i32(b, 2));
    rocke_value_t* offset = rocke_b_add(b, row_offset, dim_offset);
    return rocke_b_select(b, valid, offset, rocke_b_const_i32(b, 0x7fffffff));
}
#endif
