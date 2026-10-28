// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_ATTENTION_PAGED_KV_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_PAGED_KV_INTERNAL_H
#include "rocke/helper_rocke.helpers.transforms.h"

/* Descriptor-only decomposition; the caller owns coordinate validity. */
static inline bool rocke_paged_kv_split(rocke_ir_builder_t* b,
                                        const char* name,
                                        const char* upper,
                                        rocke_value_t* value,
                                        const char* left,
                                        const char* right,
                                        int rows,
                                        int cols,
                                        rocke_value_t** out_left,
                                        rocke_value_t** out_right)
{
    const int lengths[] = {rows, cols};
    const char* coords[] = {left, right};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, name, lengths, 2, NULL, coords, 2);
    const rocke_transform_t* chain[] = {rocke_unmerge(b, upper, coords, 2, lengths)};
    if(desc == NULL || chain[0] == NULL)
        return false;
    desc = rocke_tensor_descriptor_transform(b, desc, chain, 1);
    if(desc == NULL)
        return false;
    const char* inputs[] = {upper};
    rocke_value_t* values[] = {value};
    rocke_value_t* lower[3];
    if(rocke_tensor_descriptor_unmerge_lower(b, desc, inputs, values, 1, NULL, lower, 3) != 3)
        return false;
    /* unmerge_lower retains the input, then appends the two lower coordinates. */
    *out_left = lower[1];
    *out_right = lower[2];
    return true;
}

static inline rocke_value_t* rocke_paged_kv_token(rocke_ir_builder_t* b,
                                                  rocke_value_t* tile,
                                                  rocke_value_t* linear,
                                                  int hd,
                                                  int tile_size,
                                                  rocke_value_t** dim)
{
    rocke_value_t* row;
    if(!rocke_paged_kv_split(
           b, "paged_tile", "linear", linear, "row", "dim", tile_size, hd, &row, dim))
        return NULL;
    const int lengths[] = {1, tile_size};
    const int strides[] = {tile_size, 1};
    const char* coords[] = {"tile", "row"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "paged_token", lengths, 2, strides, coords, 2);
    if(desc == NULL)
        return NULL;
    rocke_value_t* values[] = {tile, row};
    rocke_value_t *token, *valid;
    if(!rocke_transforms_descriptor_offset(b, desc, coords, values, 2, &token, &valid))
        return NULL;
    return token;
}

static inline rocke_value_t* rocke_paged_kv_byte_offset(rocke_ir_builder_t* b,
                                                        rocke_value_t* physical,
                                                        rocke_value_t* row,
                                                        rocke_value_t* head,
                                                        rocke_value_t* dim,
                                                        int hd,
                                                        int page,
                                                        int heads)
{
    const int lengths[] = {1, page, heads, hd};
    const int strides[] = {page * heads * hd * 2, heads * hd * 2, hd * 2, 2};
    const char* coords[] = {"physical_page", "page_token", "kv_head", "dim"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "paged_kv_bytes", lengths, 4, strides, coords, 4);
    if(desc == NULL)
        return NULL;
    rocke_value_t* values[] = {physical, row, head, dim};
    rocke_value_t *offset, *valid;
    if(!rocke_transforms_descriptor_offset(b, desc, coords, values, 4, &offset, &valid))
        return NULL;
    return offset;
}

/* Mirror of library/kernels/common/_attention_paged_kv.py. Host admission checks
 * structural pool representability. Valid lengths, query offsets and referenced
 * physical page IDs remain caller-owned, including after mutation/graph replay.
 * Optional synchronized binding content validation must be repeated after
 * mutation; table-index guards do not validate referenced physical page IDs. */
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
    rocke_value_t* dim;
    rocke_value_t* token = rocke_paged_kv_token(b, tile, linear, hd, tile_size, &dim);
    if(token == NULL)
        return NULL;
    rocke_value_t* valid = rocke_b_cmp_lt(b, token, length);
    rocke_value_t* safe_token = rocke_b_select(b, valid, token, rocke_b_const_i32(b, 0));
    rocke_value_t *page_index, *row;
    if(!rocke_paged_kv_split(b,
                             "paged_page",
                             "safe_token",
                             safe_token,
                             "logical_page",
                             "page_token",
                             1,
                             page,
                             &page_index,
                             &row))
        return NULL;
    rocke_value_t* physical
        = rocke_b_global_load(b, table, rocke_b_add(b, seq_base, page_index), rocke_i32(), 4);
    rocke_value_t* offset
        = rocke_paged_kv_byte_offset(b, physical, row, head, dim, hd, page, heads);
    if(offset == NULL)
        return NULL;
    return rocke_b_select(b, valid, offset, rocke_b_const_i32(b, 0x7fffffff));
}
#endif
