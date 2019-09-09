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

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"
#include "test_utils_types.hpp"

#define HIP_CHECK(error) ASSERT_EQ(static_cast<hipError_t>(error), hipSuccess)

namespace rp = rocprim;

template<typename Params>
class RocprimBlockSortTests : public ::testing::Test {
public:
    using key_type = typename Params::input_type;
    using value_type = typename Params::output_type;
    static constexpr unsigned int block_size = Params::block_size;
};

TYPED_TEST_CASE(RocprimBlockSortTests, BlockParams);

template<
    unsigned int BlockSize,
    class key_type
>
__global__
void sort_key_kernel(key_type * device_key_output) 
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    key_type key = device_key_output[index];
    rp::block_sort<key_type, BlockSize> bsort;
    bsort.sort(key);
    device_key_output[index] = key;
}

TYPED_TEST(RocprimBlockSortTests, SortKey)
{
    using key_type = typename TestFixture::key_type;
    using binary_op_type = typename std::conditional<std::is_same<key_type, rp::half>::value, test_utils::half_less, rp::less<key_type>>::type;
    const size_t block_size = TestFixture::block_size;
    const size_t size = block_size * 1134;
    const size_t grid_size = size / block_size;

    // Generate data
    std::vector<key_type> output = test_utils::get_random_data<key_type>(size, -100, 100);

    // Calculate expected results on host
    std::vector<key_type> expected(output);
    binary_op_type binary_op;
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        std::sort(
            expected.begin() + (i * block_size),
            expected.begin() + ((i + 1) * block_size),
            binary_op
        );
    }

    // Preparing device
    key_type * device_key_output;
    HIP_CHECK(hipMalloc(&device_key_output, output.size() * sizeof(key_type)));

    HIP_CHECK(
        hipMemcpy(
            device_key_output, output.data(),
            output.size() * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sort_key_kernel<block_size, key_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_key_output
    );

    // Reading results back
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_key_output,
            output.size() * sizeof(key_type),
            hipMemcpyDeviceToHost
        )
    );

    test_utils::assert_eq(output, expected);

    HIP_CHECK(hipFree(device_key_output));
}

template<class Key, class Value>
struct pair_comparator
{
    using less_key = typename std::conditional<std::is_same<Key, rp::half>::value, test_utils::half_less, rp::less<Key>>::type;
    using eq_key = typename std::conditional<std::is_same<Key, rp::half>::value, test_utils::half_equal_to, rp::equal_to<Key>>::type;
    using less_value = typename std::conditional<std::is_same<Value, rp::half>::value, test_utils::half_less, rp::less<Value>>::type;

    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return (less_key()(lhs.first, rhs.first) || (eq_key()(lhs.first, rhs.first) && less_value()(lhs.second, rhs.second)));
    }
};

template<
    unsigned int BlockSize,
    class key_type,
    class value_type
>
__global__
void sort_key_value_kernel(key_type * device_key_output, value_type * device_value_output) 
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    key_type key = device_key_output[index];
    value_type value = device_value_output[index];
    rp::block_sort<key_type, BlockSize, value_type> bsort;
    bsort.sort(key, value);
    device_key_output[index] = key;
    device_value_output[index] = value;
}

TYPED_TEST(RocprimBlockSortTests, SortKeyValue)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using value_op_type = typename std::conditional<std::is_same<value_type, rp::half>::value, test_utils::half_less, rp::less<value_type>>::type;
    using eq_op_type = typename std::conditional<std::is_same<key_type, rp::half>::value, test_utils::half_equal_to, rp::equal_to<key_type>>::type;
    const size_t block_size = TestFixture::block_size;
    const size_t size = block_size * 1134;
    const size_t grid_size = size / block_size;

    // Generate data
    std::vector<key_type> output_key = test_utils::get_random_data<key_type>(size, 0, 100);
    std::vector<value_type> output_value = test_utils::get_random_data<value_type>(size, -100, 100);

    // Combine vectors to form pairs with key and value
    std::vector<std::pair<key_type, value_type>> target(size);
    for (unsigned i = 0; i < target.size(); i++)
        target[i] = std::make_pair(output_key[i], output_value[i]);

    // Calculate expected results on host
    using key_value = std::pair<key_type, value_type>;
    std::vector<key_value> expected(target);
    for(size_t i = 0; i < expected.size() / block_size; i++)
    {
        std::sort(
            expected.begin() + (i * block_size),
            expected.begin() + ((i + 1) * block_size),
            pair_comparator<key_type, value_type>()
        );
    }

    // Preparing device
    key_type * device_key_output;
    HIP_CHECK(hipMalloc(&device_key_output, output_key.size() * sizeof(key_type)));
    value_type * device_value_output;
    HIP_CHECK(hipMalloc(&device_value_output, output_value.size() * sizeof(value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_key_output, output_key.data(),
            output_key.size() * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_value_output, output_value.data(),
            output_value.size() * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(sort_key_value_kernel<block_size, key_type, value_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_key_output, device_value_output
    );

    // Reading results back
    HIP_CHECK(
        hipMemcpy(
            output_key.data(), device_key_output,
            output_key.size() * sizeof(key_type),
            hipMemcpyDeviceToHost
        )
    );

    HIP_CHECK(
        hipMemcpy(
            output_value.data(), device_value_output,
            output_value.size() * sizeof(value_type),
            hipMemcpyDeviceToHost
        )
    );

    std::vector<key_type> expected_key(expected.size());
    std::vector<value_type> expected_value(expected.size());
    for(size_t i = 0; i < expected.size(); i++)
    {
        expected_key[i] = expected[i].first;
        expected_value[i] = expected[i].second;
    }

    // Keys are sorted, Values order not guaranteed
    // Sort subsets where key was the same to make sure all values are still present
    value_op_type value_op;
    eq_op_type eq_op;
    for (size_t i = 0; i < output_key.size();)
    {
        auto j = i;
        for (; j < output_key.size() && eq_op(output_key[j], output_key[i]); ++j) { }
        std::sort(output_value.begin() + i, output_value.begin() + j, value_op);
        std::sort(expected_value.begin() + i, expected_value.begin() + j, value_op);
        i = j;
    }
    
    test_utils::assert_eq(output_key, expected_key);
    test_utils::assert_eq(output_value, expected_value);
}

template<class Key, class Value>
struct key_value_comparator
{
    using greater_key = typename std::conditional<std::is_same<Key, rp::half>::value, test_utils::half_greater, rp::greater<Key>>::type;
    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return greater_key()(lhs.first, rhs.first);
    }
};

template<
    unsigned int BlockSize,
    class key_type,
    class value_type
>
__global__
void custom_sort_key_value_kernel(key_type * device_key_output, value_type * device_value_output)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    key_type key = device_key_output[index];
    value_type value = device_value_output[index];
    rp::block_sort<key_type, BlockSize, value_type> bsort;
    bsort.sort(key, value, rocprim::greater<key_type>());
    device_key_output[index] = key;
    device_value_output[index] = value;
}

TYPED_TEST(RocprimBlockSortTests, CustomSortKeyValue)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using value_op_type = typename std::conditional<std::is_same<value_type, rp::half>::value, test_utils::half_less, rp::less<value_type>>::type;
    using eq_op_type = typename std::conditional<std::is_same<key_type, rp::half>::value, test_utils::half_equal_to, rp::equal_to<key_type>>::type;
    const size_t block_size = TestFixture::block_size;
    const size_t size = block_size * 1134;
    const size_t grid_size = size / block_size;

    // Generate data
    std::vector<key_type> output_key = test_utils::get_random_data<key_type>(size, 0, 100);
    std::vector<value_type> output_value = test_utils::get_random_data<value_type>(size, -100, 100);

    // Combine vectors to form pairs with key and value
    std::vector<std::pair<key_type, value_type>> target(size);
    for (unsigned i = 0; i < target.size(); i++)
        target[i] = std::make_pair(output_key[i], output_value[i]);

    // Calculate expected results on host
    using key_value = std::pair<key_type, value_type>;
    std::vector<key_value> expected(target);
    for(size_t i = 0; i < expected.size() / block_size; i++)
    {
        std::sort(
            expected.begin() + (i * block_size),
            expected.begin() + ((i + 1) * block_size),
            key_value_comparator<key_type, value_type>()
        );
    }

    // Preparing device
    key_type * device_key_output;
    HIP_CHECK(hipMalloc(&device_key_output, output_key.size() * sizeof(key_type)));
    value_type * device_value_output;
    HIP_CHECK(hipMalloc(&device_value_output, output_value.size() * sizeof(value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_key_output, output_key.data(),
            output_key.size() * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_value_output, output_value.data(),
            output_value.size() * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(custom_sort_key_value_kernel<block_size, key_type, value_type>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_key_output, device_value_output
    );

    // Reading results back
    HIP_CHECK(
        hipMemcpy(
            output_key.data(), device_key_output,
            output_key.size() * sizeof(key_type),
            hipMemcpyDeviceToHost
        )
    );

    HIP_CHECK(
        hipMemcpy(
            output_value.data(), device_value_output,
            output_value.size() * sizeof(value_type),
            hipMemcpyDeviceToHost
        )
    );

    std::vector<key_type> expected_key(expected.size());
    std::vector<value_type> expected_value(expected.size());
    for(size_t i = 0; i < expected.size(); i++)
    {
        expected_key[i] = expected[i].first;
        expected_value[i] = expected[i].second;
    }

    // Keys are sorted, Values order not guaranteed
    // Sort subsets where key was the same to make sure all values are still present
    value_op_type value_op;
    eq_op_type eq_op;
    for (size_t i = 0; i < output_key.size();)
    {
        auto j = i;
        for (; j < output_key.size() && eq_op(output_key[j], output_key[i]); ++j) { }
        std::sort(output_value.begin() + i, output_value.begin() + j, value_op);
        std::sort(expected_value.begin() + i, expected_value.begin() + j, value_op);
        i = j;
    }

    test_utils::assert_eq(output_key, expected_key);
    test_utils::assert_eq(output_value, expected_value);
}
