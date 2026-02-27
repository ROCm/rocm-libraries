// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef GUARD_CPU_CAT_HPP
#define GUARD_CPU_CAT_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_cat_forward(const std::vector<tensor<T>>& inputs,
                     tensor<T>& ref_output,
                     int32_t dim,
                     bool multi_threaded)
{
    const auto& dims             = ref_output.desc.GetLengths();
    const size_t output_dim_size = dims[dim];
    size_t outer_size            = 1;
    size_t inner_size            = 1;
    size_t k                     = 0;

    for(; k < dim; ++k)
    {
        outer_size *= dims[k];
    }

    for(; k < dims.size(); ++k)
    {
        inner_size *= dims[k];
    }

    const size_t inner_size_on_output_dim_size = inner_size / output_dim_size;
    const size_t n                             = inputs.size();
    const size_t min_grain                     = multi_threaded ? 1 : n;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // 1. Precompute output start offsets to avoid race conditions in the multi-threaded version.
    // 2. Cache copy_size values, since they are needed by the start offsets computation anyway.
    std::vector<size_t> copy_sizes;
    std::vector<size_t> output_start_offsets;

    copy_sizes.reserve(n);
    output_start_offsets.reserve(n);

    copy_sizes.emplace_back(inner_size_on_output_dim_size * inputs[0].desc.GetLengths()[dim]);
    output_start_offsets.emplace_back(0);

    for(size_t i{1}; i < n; ++i)
    {
        const size_t dim_size = inputs[i].desc.GetLengths()[dim];

        output_start_offsets.emplace_back(output_start_offsets.back() + copy_sizes.back());
        copy_sizes.emplace_back(inner_size_on_output_dim_size * dim_size);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////

    miopen::par_for(n, min_grain, [&](size_t i) {
        const auto& input       = inputs[i];
        const size_t copy_size  = copy_sizes[i];
        const size_t input_size = outer_size * copy_size;

        for(size_t o = 0; o < input_size; ++o)
        {
            const size_t outer_idx = o / copy_size;
            const size_t copy_idx  = o % copy_size;
            ref_output[output_start_offsets[i] + (outer_idx * inner_size) + copy_idx] =
                input[copy_size * outer_idx + copy_idx];
        }
    });
}

#endif
