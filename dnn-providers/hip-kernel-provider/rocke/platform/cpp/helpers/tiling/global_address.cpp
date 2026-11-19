// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Descriptor-derived byte bases; mirrors tiling/global_address.py. */
#include "rocke/tiling_internal.h"

#include <limits.h>
#include <string.h>

#include "rocke/dtypes.h"
#include "rocke/error.hpp"
#include "rocke/storage_internal.h"

static int gcd(int a, int b)
{
    while(b)
    {
        int next = a % b;
        a = b;
        b = next;
    }
    return a;
}

bool rocke_i_tiling_can_rebase_global(rocke_value_t* ptr,
                                      const rocke_tiling_window_t* window,
                                      const rocke_tiling_tile_desc_t* tile,
                                      int alignment_bytes,
                                      bool in_bounds)
{
    if(!alignment_bytes || !in_bounds || ptr->type->kind != ROCKE_TYPE_PTR
       || strcmp(ptr->type->space, "global") != 0)
        return false;
    const auto* tensor = window->tensor;
    uint64_t span = 1;
    for(int axis = 0; axis < tensor->rank; ++axis)
    {
        if(!tensor->has_length[axis] || tensor->strides[axis] < 0 || tensor->lengths[axis] < 1)
            return false;
        if(window->origin[axis].value)
        {
            if(!rocke_type_eq(window->origin[axis].value->type, rocke_i32()))
                return false;
        }
        else if(window->origin[axis].constant < 0
                || window->origin[axis].constant >= tensor->lengths[axis])
            return false;
        const uint64_t term = uint64_t(tensor->lengths[axis] - 1) * tensor->strides[axis];
        if(term > uint64_t(INT_MAX) - span)
            return false;
        span += term;
    }
    for(int axis = 0; axis < tile->rank; ++axis)
    {
        uint64_t extent = 1;
        for(int level = 0; level < tile->layout->Hs[axis].count; ++level)
        {
            int length = tile->layout->Hs[axis].levels[level];
            if(length < 1 || extent > uint64_t(INT_MAX) / length)
                return false;
            extent *= length;
        }
        if(extent != uint64_t(tile->shape[axis]))
            return false;
    }
    const int bits = rocke_dtype_info(tensor->dtype->name)->encoded_bits;
    return (span * bits + 7) / 8 <= INT_MAX;
}

void rocke_i_tiling_split_window(const rocke_tiling_window_t* window,
                                 rocke_tiling_window_t* runtime_window,
                                 int64_t* constant_origin)
{
    *runtime_window = *window;
    *constant_origin = 0;
    for(int axis = 0; axis < window->tensor->rank; ++axis)
        if(!window->origin[axis].value)
        {
            *constant_origin
                += int64_t(window->origin[axis].constant) * window->tensor->strides[axis];
            runtime_window->origin[axis].constant = 0;
        }
}

rocke_value_t* rocke_i_tiling_byte_address(rocke_ir_builder_t* b,
                                           const rocke_tiling_window_t* window,
                                           int tile_rank,
                                           rocke_value_t* const* positions)
{
    const auto* tensor = window->tensor;
    const int bits = rocke_dtype_info(tensor->dtype->name)->encoded_bits;
    const int common = gcd(bits, 8), group = 8 / common, group_bytes = bits / common;
    const int lead = tensor->rank - tile_rank;
    rocke_value_t* leading[ROCKE_TILING_MAX_RANK];
    for(int axis = 0; axis < lead; ++axis)
        leading[axis] = rocke_i_tiling_as_value(b, window->origin[axis]);
    rocke_value_t* address = NULL;
    for(int order = 0; order < tensor->rank; ++order)
    {
        const int axis = order < tile_rank ? lead + order : order - tile_rank;
        auto* position = order < tile_rank ? positions[order] : leading[order - tile_rank];
        const int stride = tensor->strides[axis];
        rocke_value_t* term;
        if(stride % group == 0)
        {
            const int64_t factor = int64_t(stride / group) * group_bytes;
            term = factor == 1 ? position : rocke_b_mul(b, position, rocke_b_const_i32(b, factor));
        }
        else
        {
            if(stride != 1)
                ckc::raise_status(ROCKE_ERR_VALUE,
                                  "packed byte addressing requires whole-byte strides");
            term = rocke_b_div(b, position, rocke_b_const_i32(b, group));
            if(group_bytes != 1)
                term = rocke_b_mul(b, term, rocke_b_const_i32(b, group_bytes));
        }
        address = address ? rocke_b_add(b, address, term) : term;
    }
    return address ? address : rocke_b_const_i32(b, 0);
}

int64_t rocke_i_tiling_storage_displacement(const rocke_tiling_window_t* window,
                                            const rocke_tiling_tile_desc_t* tile,
                                            const int64_t* coordinates,
                                            int64_t constant_origin)
{
    const int lead = window->tensor->rank - tile->rank;
    int64_t elements = constant_origin;
    for(int axis = 0; axis < tile->rank; ++axis)
        elements += coordinates[axis] * window->tensor->strides[lead + axis];
    const int bits = rocke_dtype_info(window->tensor->dtype->name)->encoded_bits;
    // Byte-or-wider displacements may be signed (reversed views). Packed
    // admission still requires nonnegative, whole-word offsets.
    return bits >= 8 ? elements : rocke_i_elements_to_storage(bits, elements);
}
