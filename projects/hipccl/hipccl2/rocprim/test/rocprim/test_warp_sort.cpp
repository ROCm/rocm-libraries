// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

// required rocprim headers
#include <rocprim/warp/warp_sort.hpp>

// required test headers
#include "test_utils_types.hpp"

template<typename Params>
class RocprimWarpSortShuffleBasedTests : public ::testing::Test {
public:
    using params = Params;
};

#if (defined(__gfx1030__))
    #define warp_sort_param_type(type) \
        warp_params<type, 2U>, \
        warp_params<type, 4U>, \
        warp_params<type, 8U>, \
        warp_params<type, 16U>, \
        warp_params<type, 32U>
#else
    #define warp_sort_param_type(type) \
       warp_params<type, 2U>, \
       warp_params<type, 4U>, \
       warp_params<type, 8U>, \
       warp_params<type, 16U>, \
       warp_params<type, 32U>, \
       warp_params<type, 64U>
#endif
typedef ::testing::Types<
    warp_sort_param_type(int),
    warp_sort_param_type(test_utils::custom_test_type<int>),
    warp_sort_param_type(uint8_t),
    warp_sort_param_type(int8_t),
    warp_sort_param_type(rocprim::half)
> WarpSortParams;

TYPED_TEST_CASE(RocprimWarpSortShuffleBasedTests, WarpSortParams);

template<
    class T,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void test_hip_warp_sort(T* d_output)
{
    unsigned int i = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);
    T value = d_output[i];
    rocprim::warp_sort<T, LogicalWarpSize> wsort;
    wsort.sort(value);
    d_output[i] = value;
}

TYPED_TEST(RocprimWarpSortShuffleBasedTests, Sort)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_less, rocprim::less<T>>::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    const size_t block_size = std::max<size_t>(rocprim::warp_size(), 4 * logical_warp_size);
    constexpr size_t grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size() || !rocprim::detail::is_power_of_two(logical_warp_size))
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 0, 100, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            std::sort(expected.begin() + (i * logical_warp_size), expected.begin() + ((i + 1) * logical_warp_size), binary_op);
        }

        // Writing to device memory
        T* d_output;
        HIP_CHECK(
            hipMalloc(&d_output, output.size() * sizeof(typename decltype(output)::value_type))
        );

        HIP_CHECK(
            hipMemcpy(
                d_output, output.data(),
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(test_hip_warp_sort<T, logical_warp_size>),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_output
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(d_output));
    }

}

template<
    class KeyType,
    class ValueType,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void test_hip_sort_key_value_kernel(KeyType* d_output_key, ValueType* d_output_value)
{
    unsigned int i = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);
    KeyType key = d_output_key[i];
    ValueType value = d_output_value[i];
    rocprim::warp_sort<KeyType, LogicalWarpSize, ValueType> wsort;
    wsort.sort(key, value);
    d_output_key[i] = key;
    d_output_value[i] = value;
}

TYPED_TEST(RocprimWarpSortShuffleBasedTests, SortKeyInt)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using pair = test_utils::custom_test_type<T>;
    using value_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_less, rocprim::less<T>>::type;
    using eq_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_equal_to, rocprim::equal_to<T>>::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    const size_t block_size = std::max<size_t>(rocprim::warp_size(), 4 * logical_warp_size);
    constexpr size_t grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size() || !rocprim::detail::is_power_of_two(logical_warp_size))
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output_key = test_utils::get_random_data<T>(size, 0, 100, seed_value);
        std::vector<T> output_value = test_utils::get_random_data<T>(size, 0, 100, seed_value);

        // Combine vectors to form pairs with key and value
        std::vector<pair> target(size);
        for(unsigned i = 0; i < target.size(); i++)
        {
            target[i].x = output_key[i];
            target[i].y = output_value[i];
        }

        // Calculate expected results on host
        std::vector<pair> expected(target);
        for(size_t i = 0; i < expected.size() / logical_warp_size; i++)
        {
            std::sort(expected.begin() + (i * logical_warp_size),
                    expected.begin() + ((i + 1) * logical_warp_size)
            );
        }

        // Writing to device memory
        T* d_output_key;
        T* d_output_value;
        HIP_CHECK(
            hipMalloc(&d_output_key, output_key.size() * sizeof(typename decltype(output_key)::value_type))
        );
        HIP_CHECK(
            hipMalloc(&d_output_value, output_value.size() * sizeof(typename decltype(output_value)::value_type))
        );

        HIP_CHECK(
            hipMemcpy(
                d_output_key, output_key.data(),
                output_key.size() * sizeof(typename decltype(output_key)::value_type),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemcpy(
                d_output_value, output_value.data(),
                output_value.size() * sizeof(typename decltype(output_value)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(test_hip_sort_key_value_kernel<T, T, logical_warp_size>),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_output_key, d_output_value
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output_key.data(), d_output_key,
                output_key.size() * sizeof(typename decltype(output_key)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_value.data(), d_output_value,
                output_value.size() * sizeof(typename decltype(output_value)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        std::vector<T> expected_key(expected.size());
        std::vector<T> expected_value(expected.size());
        for(size_t i = 0; i < expected.size(); i++)
        {
            expected_key[i] = expected[i].x;
            expected_value[i] = expected[i].y;
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

        test_utils::assert_near(output_key, expected_key, test_utils::precision_threshold<T>::percentage);
        test_utils::assert_near(output_value, expected_value, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(d_output_key));
        HIP_CHECK(hipFree(d_output_value));
    }

}
