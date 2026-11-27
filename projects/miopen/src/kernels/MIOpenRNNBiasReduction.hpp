// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_RNN_BIAS_REDUCTION_HPP_
#define MIOPEN_RNN_BIAS_REDUCTION_HPP_

namespace miopen {

constexpr int rnn_bias_columns_per_block = 16;
constexpr int rnn_bias_rows_per_block    = 16;

// Passed by value in the kernel arguments to avoid a separate metadata transfer.
struct RNNBiasRowRanges
{
    static constexpr int capacity = 16;
    struct Range
    {
        int begin;
        int end;
    };
    Range ranges[capacity];
    int count;
};

} // namespace miopen

#endif // MIOPEN_RNN_BIAS_REDUCTION_HPP_
