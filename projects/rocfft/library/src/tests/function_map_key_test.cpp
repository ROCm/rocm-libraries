// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "function_map_key.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(rocfft_internal, function_map_key_comparison_and_hash)
{
    const std::string arch = "gfx942";

    FMKey base(
        64, rocfft_precision_single, CS_KERNEL_STOCKHAM, NONE, KernelConfig::EmptyConfig(), arch);

    std::vector<FMKey> different;
    different.emplace_back(
        128, rocfft_precision_single, CS_KERNEL_STOCKHAM, NONE, KernelConfig::EmptyConfig(), arch);
    different.emplace_back(
        64, rocfft_precision_double, CS_KERNEL_STOCKHAM, NONE, KernelConfig::EmptyConfig(), arch);
    different.emplace_back(64,
                           rocfft_precision_single,
                           CS_KERNEL_STOCKHAM_BLOCK_CC,
                           NONE,
                           KernelConfig::EmptyConfig(),
                           arch);
    different.emplace_back(64,
                           rocfft_precision_single,
                           CS_KERNEL_STOCKHAM,
                           TILE_ALIGNED,
                           KernelConfig::EmptyConfig(),
                           arch);
    different.emplace_back(64,
                           rocfft_precision_single,
                           CS_KERNEL_STOCKHAM,
                           NONE,
                           KernelConfig::EmptyConfig(),
                           "gfx90a");

    // Hash collisions are allowed; what must hold is that changing any field
    // changes the hash, so a field left out of it would be noticed.
    const auto base_hash = SimpleHash{}(base);

    for(const auto& other : different)
    {
        EXPECT_FALSE(base == other);
        EXPECT_TRUE(base != other);
        // exactly one of the two orderings must hold
        EXPECT_NE(base < other, other < base);
        EXPECT_NE(SimpleHash{}(other), base_hash);
    }

    // a copy is equal, hashes the same, and orders neither before nor after
    FMKey copy = base;
    EXPECT_TRUE(base == copy);
    EXPECT_EQ(SimpleHash{}(base), SimpleHash{}(copy));
    EXPECT_FALSE(base < copy);
    EXPECT_FALSE(copy < base);
}

TEST(rocfft_internal, partial_pass_key_orders_problem_before_batch)
{
    const std::string arch = "gfx942";

    auto make = [&arch](size_t length0, size_t batch_low, size_t batch_high) {
        return PPFMKey(length0,
                       64,
                       64,
                       rocfft_precision_single,
                       rocfft_transform_type_complex_forward,
                       CS_3D_PP,
                       batch_low,
                       batch_high,
                       KernelConfig::EmptyConfig(),
                       KernelConfig::EmptyConfig(),
                       arch);
    };

    auto small_problem_high_batch = make(32, 1000, 2000);
    auto large_problem_low_batch  = make(64, 1, 2);

    // problem identity wins over batch range
    EXPECT_TRUE(small_problem_high_batch < large_problem_low_batch);

    // within one problem, batch range decides
    auto low_batch  = make(64, 1, 8);
    auto high_batch = make(64, 9, 16);
    EXPECT_TRUE(low_batch < high_batch);
    EXPECT_FALSE(high_batch < low_batch);

    // and the config fields only matter once the batch range matches
    PPFMKey same_batch_other_config                        = low_batch;
    same_batch_other_config.kernel_config_1.workgroup_size = 512;
    EXPECT_NE(low_batch < same_batch_other_config, same_batch_other_config < low_batch);
    EXPECT_TRUE(low_batch < high_batch);
    EXPECT_TRUE(same_batch_other_config < high_batch);
}
