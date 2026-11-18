// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors the coordinate, address and clip machinery in tiling/emit.py. */
#include "rocke/tiling_internal.h"

#include <limits.h>
#include <string.h>

#include "rocke/error.hpp"

rocke_value_t* rocke_i_tiling_as_value(rocke_ir_builder_t* b, rocke_tiling_index_t index)
{
    return index.value ? index.value : rocke_b_const_i32(b, index.constant);
}

bool rocke_i_tiling_constant_index(rocke_tiling_index_t index, int64_t* value)
{
    if(!index.value)
    {
        *value = index.constant;
        return true;
    }
    return index.value->op && strcmp(index.value->op->name, "arith.constant") == 0
           && rocke_attr_get_int(&index.value->op->attrs, "value", value);
}

rocke_tiling_index_t rocke_i_tiling_bound(const rocke_tiling_window_t* window, int axis)
{
    return window->has_bound[axis] ? window->bounds[axis]
                                   : rocke_tiling_index_t{NULL, window->tensor->lengths[axis]};
}

bool rocke_i_tiling_may_overhang(const rocke_tiling_window_t* window, int axis, int extent)
{
    const auto bound = rocke_i_tiling_bound(window, axis);
    const auto origin = window->origin[axis];
    return bound.value || bound.constant % extent != 0
           || (!origin.value && origin.constant % extent != 0);
}

static int bucket_length(const rocke_tile_distribution_encoding_t* layout, int major, int minor)
{
    return major ? layout->Hs[major - 1].levels[minor] : layout->Rs[minor];
}

int rocke_i_tiling_register_count(const rocke_tile_distribution_encoding_t* layout,
                                  const int* shape)
{
    const auto limits = [](bool invalid) {
        if(invalid)
            ckc::raise_status(ROCKE_ERR_NOTIMPL,
                              "native fragment encoding exceeds bounded rank/level limits");
    };
    const auto multiply = [](int product, int length) {
        if(length < 1 || product > INT_MAX / length)
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "tile layout extent is outside the positive i32 range");
        return product * length;
    };
    limits(layout->num_X < 1 || layout->num_X > ROCKE_TILING_MAX_RANK || layout->num_Y < 0
           || layout->num_Y > 64 || layout->num_R < 0 || layout->num_R > 8);
    for(int axis = 0; axis < layout->num_X; ++axis)
    {
        const auto& row = layout->Hs[axis];
        limits(row.count < 0 || row.count > 8);
        int extent = 1;
        for(int i = 0; i < row.count; ++i)
            extent = multiply(extent, row.levels[i]);
        if(extent != shape[axis])
            ckc::raise_status(ROCKE_ERR_VALUE, "tile shape does not match its distribution extent");
    }
    int partitions = 0, lanes = 1;
    for(int p = 0; p < layout->num_P; ++p)
    {
        const auto& partition = layout->Ps[p];
        limits(partition.count < 0 || partition.count > 72 - partitions);
        partitions += partition.count;
        for(int i = 0; i < partition.count; ++i)
            lanes = multiply(lanes, bucket_length(layout, partition.major[i], partition.minor[i]));
    }
    int registers = 1;
    for(int i = 0; i < layout->num_Y; ++i)
        registers
            = multiply(registers, bucket_length(layout, layout->Ys_major[i], layout->Ys_minor[i]));
    return registers;
}

void rocke_i_tiling_coordinates(rocke_ir_builder_t* b,
                                const rocke_tiling_tile_desc_t* tile,
                                rocke_value_t* thread,
                                int reg,
                                rocke_value_t** coordinates)
{
    const auto* layout = tile->layout;
    rocke_value_t* h[ROCKE_TILING_MAX_RANK][8] = {};
    int major[72], minor[72], partition_count = 0;
    for(int p = 0; p < layout->num_P; ++p)
        for(int i = 0; i < layout->Ps[p].count; ++i)
        {
            major[partition_count] = layout->Ps[p].major[i];
            minor[partition_count++] = layout->Ps[p].minor[i];
        }
    int64_t suffix = 1;
    for(int p = partition_count - 1; p >= 0; --p)
    {
        const int length = bucket_length(layout, major[p], minor[p]);
        rocke_value_t* contributor;
        if(length == 1)
            contributor = rocke_b_const_i32(b, 0);
        else
        {
            auto* divided
                = suffix == 1 ? thread : rocke_b_div(b, thread, rocke_b_const_i32(b, suffix));
            contributor = rocke_b_mod(b, divided, rocke_b_const_i32(b, length));
        }
        if(major[p])
            h[major[p] - 1][minor[p]] = contributor;
        suffix *= length;
    }
    int values[64], remainder = reg;
    for(int y = layout->num_Y - 1; y >= 0; --y)
    {
        const int length = bucket_length(layout, layout->Ys_major[y], layout->Ys_minor[y]);
        values[y] = remainder % length;
        remainder /= length;
    }
    for(int y = 0; y < layout->num_Y; ++y)
    {
        auto* value = rocke_b_const_i32(b, values[y]);
        if(layout->Ys_major[y])
            h[layout->Ys_major[y] - 1][layout->Ys_minor[y]] = value;
    }
    for(int axis = 0; axis < tile->rank; ++axis)
    {
        rocke_value_t* coordinate = NULL;
        int64_t stride = 1;
        for(int level = layout->Hs[axis].count - 1; level >= 0; --level)
        {
            auto* source = h[axis][level];
            if(source)
            {
                auto* term
                    = stride == 1 ? source : rocke_b_mul(b, source, rocke_b_const_i32(b, stride));
                coordinate = coordinate ? rocke_b_add(b, coordinate, term) : term;
            }
            stride *= layout->Hs[axis].levels[level];
        }
        coordinates[axis] = coordinate ? coordinate : rocke_b_const_i32(b, 0);
    }
}

void rocke_i_tiling_positions(rocke_ir_builder_t* b,
                              const rocke_tiling_window_t* window,
                              int tile_rank,
                              rocke_value_t* const* coordinates,
                              rocke_value_t** positions)
{
    const int lead = window->tensor->rank - tile_rank;
    for(int axis = 0; axis < tile_rank; ++axis)
        positions[axis] = rocke_b_add(
            b, rocke_i_tiling_as_value(b, window->origin[lead + axis]), coordinates[axis]);
}

rocke_value_t* rocke_i_tiling_address(rocke_ir_builder_t* b,
                                      const rocke_tiling_window_t* window,
                                      int tile_rank,
                                      rocke_value_t* const* positions)
{
    const int lead = window->tensor->rank - tile_rank;
    rocke_value_t* leading[ROCKE_TILING_MAX_RANK];
    // Python materializes leading positions before the tile's stride arithmetic.
    for(int axis = 0; axis < lead; ++axis)
        leading[axis] = rocke_i_tiling_as_value(b, window->origin[axis]);
    rocke_value_t* address = NULL;
    for(int axis = 0; axis < tile_rank; ++axis)
    {
        int stride = window->tensor->strides[lead + axis];
        auto* term = stride == 1 ? positions[axis]
                                 : rocke_b_mul(b, positions[axis], rocke_b_const_i32(b, stride));
        address = address ? rocke_b_add(b, address, term) : term;
    }
    for(int axis = 0; axis < lead; ++axis)
    {
        int stride = window->tensor->strides[axis];
        auto* term = stride == 1 ? leading[axis]
                                 : rocke_b_mul(b, leading[axis], rocke_b_const_i32(b, stride));
        address = address ? rocke_b_add(b, address, term) : term;
    }
    return address ? address : rocke_b_const_i32(b, 0);
}

rocke_value_t* rocke_i_tiling_clip_mask(rocke_ir_builder_t* b,
                                        const rocke_tiling_window_t* window,
                                        const rocke_tiling_tile_desc_t* tile,
                                        rocke_value_t* const* positions)
{
    const int lead = window->tensor->rank - tile->rank;
    rocke_value_t* mask = NULL;
    for(int axis = 0; axis < tile->rank; ++axis)
    {
        if(!rocke_i_tiling_may_overhang(window, lead + axis, tile->shape[axis]))
            continue;
        auto* in_axis
            = rocke_b_cmp_lt(b,
                             positions[axis],
                             rocke_i_tiling_as_value(b, rocke_i_tiling_bound(window, lead + axis)));
        mask = mask ? rocke_b_land(b, mask, in_axis) : in_axis;
    }
    return mask;
}

void rocke_i_tiling_register_offsets(const rocke_tiling_tile_desc_t* tile,
                                     int reg,
                                     int64_t* offsets)
{
    const auto* layout = tile->layout;
    int digits[ROCKE_TILING_MAX_RANK][8] = {};
    int remainder = reg;
    for(int y = layout->num_Y - 1; y >= 0; --y)
    {
        const int major = layout->Ys_major[y], minor = layout->Ys_minor[y];
        const int length = bucket_length(layout, major, minor);
        if(major)
            digits[major - 1][minor] = remainder % length;
        remainder /= length;
    }
    for(int axis = 0; axis < tile->rank; ++axis)
    {
        offsets[axis] = 0;
        int64_t stride = 1;
        for(int level = layout->Hs[axis].count - 1; level >= 0; --level)
        {
            offsets[axis] += digits[axis][level] * stride;
            stride *= layout->Hs[axis].levels[level];
        }
    }
}
