// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <vector>
#include <algorithm>

// Google Test
#include <gtest/gtest.h>

// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

// Params for tests
template<
    class KeyType,
    class ValueType = KeyType
>
struct DeviceMergeParams
{
    using key_type = KeyType;
    using value_type = ValueType;
};

template<class Params>
class RocprimDeviceMergeTests : public ::testing::Test
{
public:
    using key_type = typename Params::key_type;
    using value_type = typename Params::value_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceMergeParams<int>,
    DeviceMergeParams<unsigned long>,
    DeviceMergeParams<float, int>,
    DeviceMergeParams<int, float>
> RocprimDeviceMergeTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 17) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceMergeTests, RocprimDeviceMergeTestsParams);

TYPED_TEST(RocprimDeviceMergeTests, MergeKey)
{
    using key_type = typename TestFixture::key_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hc::accelerator acc;
        hc::accelerator_view acc_view = acc.create_view();

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_type> keys_input1 = test_utils::get_random_data<key_type>(size, 0, size);
        std::vector<key_type> keys_input2 = test_utils::get_random_data<key_type>(size, 0, size);
        std::sort(keys_input1.begin(), keys_input1.end());
        std::sort(keys_input2.begin(), keys_input2.end());

        hc::array<key_type> d_keys_input1(hc::extent<1>(size), keys_input1.begin(), acc_view);
        hc::array<key_type> d_keys_input2(hc::extent<1>(size), keys_input2.begin(), acc_view);
        hc::array<key_type> d_keys_output(size + size, acc_view);

        // Calculate expected results on host
        std::vector<key_type> expected(size + size);
        std::merge(
            keys_input1.begin(),
            keys_input1.end(),
            keys_input2.begin(),
            keys_input2.end(),
            expected.begin()
        );

        // compare function
        ::rocprim::less<key_type> lesser_op;
        // temp storage
        size_t temp_storage_size_bytes;

        // Get size of d_temp_storage
        rocprim::merge(
            nullptr, temp_storage_size_bytes,
            d_keys_input1.accelerator_pointer(),
            d_keys_input2.accelerator_pointer(),
            d_keys_output.accelerator_pointer(),
            keys_input1.size(), keys_input2.size(),
            lesser_op, acc_view, debug_synchronous
        );

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);

        // Run
        rocprim::merge(
            d_temp_storage.accelerator_pointer(), temp_storage_size_bytes,
            d_keys_input1.accelerator_pointer(),
            d_keys_input2.accelerator_pointer(),
            d_keys_output.accelerator_pointer(),
            keys_input1.size(), keys_input2.size(),
            lesser_op, acc_view, debug_synchronous
        );
        acc_view.wait();

        // Check if keys_output values are as expected
        std::vector<key_type> keys_output = d_keys_output;
        for(size_t i = 0; i < keys_output.size(); i++)
        {
            auto diff = std::max<key_type>(std::abs(0.01f * expected[i]), key_type(0.01f));
            if(std::is_integral<key_type>::value) diff = 0;
            ASSERT_NEAR(keys_output[i], expected[i], diff);
        }
    }
}
