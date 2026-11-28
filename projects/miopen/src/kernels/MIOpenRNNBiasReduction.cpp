// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif

#include "MIOpenRNNBiasReduction.hpp"
#include "bfloat16_dev.hpp"

extern "C" __global__ void RNNReduceNullHxBiasBFloat16(const ushort* __restrict__ gate_gradients,
                                                       ushort* __restrict__ bias_gradients,
                                                       float* __restrict__ reverse_partial,
                                                       size_t gate_offset,
                                                       size_t bias_offset,
                                                       int row_stride,
                                                       int gate_count,
                                                       int forward_begin,
                                                       int forward_end,
                                                       int first,
                                                       int last,
                                                       miopen::RNNBiasRowRanges reverse_ranges)
{
    constexpr int columns = miopen::rnn_bias_columns_per_block;
    constexpr int rows    = miopen::rnn_bias_rows_per_block;
    __shared__ float partial[rows][columns];

    // The first launch also reduces the forward direction. Later launches only
    // contribute the remaining reverse ranges to the same FP32 accumulator.
    const bool reverse  = !first || blockIdx.y != 0;
    const size_t column = static_cast<size_t>(blockIdx.x) * columns + threadIdx.x;
    float sum           = 0.0f;
    if(column < static_cast<size_t>(gate_count))
    {
        const size_t offset   = gate_offset + (reverse ? gate_count : 0) + column;
        const int range_count = reverse ? reverse_ranges.count : 1;
        for(int i = 0; i < range_count; ++i)
        {
            const int begin = reverse ? reverse_ranges.ranges[i].begin : forward_begin;
            const int end   = reverse ? reverse_ranges.ranges[i].end : forward_end;
            for(size_t row = static_cast<size_t>(begin) + threadIdx.y;
                row < static_cast<size_t>(end);
                row += rows)
                sum += bfloat16_to_float(gate_gradients[offset + row * row_stride]);
        }
    }
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.y == 0 && column < static_cast<size_t>(gate_count))
    {
        for(int i = 1; i < rows; ++i)
            sum += partial[i][threadIdx.x];

        if(reverse && !first)
            sum = reverse_partial[column] + sum;

        if(reverse && !last)
            reverse_partial[column] = sum;
        else
            bias_gradients[bias_offset + (reverse ? gate_count : 0) + column] =
                float_to_bfloat16(sum);
    }
}
