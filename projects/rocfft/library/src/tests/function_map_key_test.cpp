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

// FMKey is the key of the hash map that holds the built-in kernels. A field
// left out of == would let a lookup return a kernel built for a different
// problem, and equal keys hashing differently would make lookups miss.
TEST(rocfft_internal, fmkey_fields_all_take_part_in_lookup)
{
    const std::string arch = "gfx942";
    const auto        make = [&arch]() {
        return FMKey(64,
                     64,
                     rocfft_precision_single,
                     CS_KERNEL_2D_SINGLE,
                     NONE,
                     KernelConfig::EmptyConfig(),
                     arch);
    };
    const FMKey base = make();

    // equal keys built separately must hash the same, or lookups would miss
    EXPECT_TRUE(base == make());
    EXPECT_EQ(SimpleHash{}(base), SimpleHash{}(make()));

    // change each field in turn; the key must no longer match
    std::vector<FMKey> different(8, base);
    different[0].lengths[0]                   = 128;
    different[1].lengths[1]                   = 128;
    different[2].lengths[2]                   = 128;
    different[3].precision                    = rocfft_precision_double;
    different[4].scheme                       = CS_KERNEL_STOCKHAM;
    different[5].sbrcTrans                    = TILE_ALIGNED;
    different[6].kernel_config.workgroup_size = 512;
    different[7].gcn_arch_name                = "gfx90a";

    for(size_t i = 0; i < different.size(); ++i)
    {
        const auto& other = different[i];
        EXPECT_FALSE(base == other) << "field " << i << " is ignored by ==";
        // < must see the same fields as ==, so exactly one side is smaller
        EXPECT_NE(base < other, other < base) << "field " << i << " is ignored by <";
    }
}

// PPFMKey lives in a sorted map, and function_pool::find_pp_key_in_map
// binary-searches it for the entry whose batch range holds the batch. That only
// works if < sorts by problem first, then by batch range, then by config.

// a and b differ in one problem field, with a's value smaller. Give a the larger
// batch range: the problem field must still decide the order.
static void expect_problem_before_batch(PPFMKey a, PPFMKey b)
{
    a.batch_low  = 100;
    a.batch_high = 200;
    b.batch_low  = 1;
    b.batch_high = 2;
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
}

TEST(rocfft_internal, ppfmkey_orders_problem_then_batch_then_config)
{
    const PPFMKey base(32,
                       64,
                       64,
                       rocfft_precision_single,
                       rocfft_transform_type_complex_forward,
                       CS_3D_PP,
                       1,
                       8,
                       KernelConfig::EmptyConfig(),
                       KernelConfig::EmptyConfig(),
                       "gfx90a");

    // each problem field outranks the batch range
    for(size_t dim = 0; dim < 3; ++dim)
    {
        auto bigger         = base;
        bigger.lengths[dim] = 128;
        expect_problem_before_batch(base, bigger);
    }
    auto other      = base;
    other.precision = rocfft_precision_double;
    expect_problem_before_batch(base, other);
    other                = base;
    other.transform_type = rocfft_transform_type_complex_inverse;
    expect_problem_before_batch(base, other);
    other        = base;
    other.scheme = CS_KERNEL_3D_SINGLE;
    expect_problem_before_batch(base, other);
    other               = base;
    other.gcn_arch_name = "gfx942";
    expect_problem_before_batch(base, other);

    // the batch range outranks the configs
    auto low_batch                           = base;
    low_batch.kernel_config_1.workgroup_size = 512;
    low_batch.kernel_config_2.workgroup_size = 512;
    auto high_batch                          = base;
    high_batch.batch_low                     = 9;
    high_batch.batch_high                    = 16;
    EXPECT_TRUE(low_batch < high_batch);
    EXPECT_FALSE(high_batch < low_batch);

    // the configs still count, so two kernels for one batch range are
    // different keys
    for(auto config : {&PPFMKey::kernel_config_1, &PPFMKey::kernel_config_2})
    {
        auto other_config                     = base;
        (other_config.*config).workgroup_size = 512;
        EXPECT_FALSE(base == other_config);
        EXPECT_NE(base < other_config, other_config < base);
    }
}
