// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_ATTENTION_PAGED_KV_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_PAGED_KV_INTERNAL_H
#include "rocke/ir.h"

/* Mirror of library/kernels/common/_attention_paged_kv.py. Pool bounds and
 * metadata contracts are validated before launch; speculative table accesses
 * and out-of-range data loads have separate guards. */
static inline rocke_value_t* rocke_paged_kv_offset(rocke_ir_builder_t* b,
                                                   rocke_value_t* table,
                                                   rocke_value_t* seq_base,
                                                   rocke_value_t* head,
                                                   rocke_value_t* tile,
                                                   rocke_value_t* linear,
                                                   rocke_value_t* length,
                                                   int hd,
                                                   int page,
                                                   int tile_size,
                                                   int heads)
{
    rocke_value_t* start = rocke_b_mul(b, tile, rocke_b_const_i32(b, tile_size));
    rocke_value_t* token = rocke_b_add(b, start, rocke_b_div(b, linear, rocke_b_const_i32(b, hd)));
    rocke_value_t* dim = rocke_b_mod(b, linear, rocke_b_const_i32(b, hd));
    rocke_value_t* valid = rocke_b_cmp_lt(b, token, length);
    rocke_value_t* safe_token = rocke_b_select(b, valid, token, rocke_b_const_i32(b, 0));
    rocke_value_t* page_index = rocke_b_div(b, safe_token, rocke_b_const_i32(b, page));
    rocke_value_t* physical
        = rocke_b_global_load(b, table, rocke_b_add(b, seq_base, page_index), rocke_i32(), 4);
    rocke_value_t* page_offset
        = rocke_b_mul(b, physical, rocke_b_const_i32(b, page * heads * hd * 2));
    rocke_value_t* row = rocke_b_mod(b, safe_token, rocke_b_const_i32(b, page));
    rocke_value_t* row_offset = rocke_b_mul(b, row, rocke_b_const_i32(b, heads * hd * 2));
    rocke_value_t* head_offset = rocke_b_mul(b, head, rocke_b_const_i32(b, hd * 2));
    rocke_value_t* dim_offset = rocke_b_mul(b, dim, rocke_b_const_i32(b, 2));
    rocke_value_t* page_row = rocke_b_add(b, page_offset, row_offset);
    rocke_value_t* head_dim = rocke_b_add(b, head_offset, dim_offset);
    rocke_value_t* offset = rocke_b_add(b, page_row, head_dim);
    return rocke_b_select(b, valid, offset, rocke_b_const_i32(b, 0x7fffffff));
}
#endif
