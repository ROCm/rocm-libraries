// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_ATTENTION_KV_LOADER_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_KV_LOADER_INTERNAL_H

/* Mirror of library/kernels/common/_attention_kv_loader.py. Inputs have passed
 * the builder's validation. Storage conversion and transfer/address widths are
 * independent of this private addressing decision. */
typedef enum rocke_attention_kv_loader_kind
{
    ROCKE_ATTN_KV_STRIDED,
    ROCKE_ATTN_KV_PAGED_DESCRIPTOR,
    ROCKE_ATTN_KV_PAGED_TOKEN_GATHER,
    ROCKE_ATTN_KV_PAGED_MULTI_PAGE
} rocke_attention_kv_loader_kind_t;

static inline rocke_attention_kv_loader_kind_t
    rocke_gfx942_resolve_attention_kv_loader(bool strided_kv, int page, int tile)
{
    if(strided_kv)
        return ROCKE_ATTN_KV_STRIDED;
    return tile > page ? ROCKE_ATTN_KV_PAGED_TOKEN_GATHER : ROCKE_ATTN_KV_PAGED_DESCRIPTOR;
}

static inline rocke_attention_kv_loader_kind_t
    rocke_gfx950_resolve_attention_kv_loader(bool strided_kv, int page, int tile, bool i64_addr)
{
    if(strided_kv)
        return ROCKE_ATTN_KV_STRIDED;
    if(page >= 16 && tile > page && (tile != 32 || i64_addr))
        return ROCKE_ATTN_KV_PAGED_MULTI_PAGE;
    return tile != page ? ROCKE_ATTN_KV_PAGED_TOKEN_GATHER : ROCKE_ATTN_KV_PAGED_DESCRIPTOR;
}

#endif
