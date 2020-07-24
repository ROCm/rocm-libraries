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
#include <rocprim/warp/warp_reduce.hpp>

// required test headers
#include "test_utils_types.hpp"

template<class Params>
class RocprimWarpReduceTests : public ::testing::Test {
public:
    using params = Params;
};

TYPED_TEST_CASE(RocprimWarpReduceTests, WarpParams);

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void warp_reduce_sum_kernel(T* device_input, T* device_output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, storage[warp_id]);

    if(hipThreadIdx_x%LogicalWarpSize == 0)
    {
        device_output[index/LogicalWarpSize] = value;
    }
}

TYPED_TEST(RocprimWarpReduceTests, ReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_plus, rocprim::plus<T>>::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(rocprim::warp_size(), logical_warp_size * 4)
            : (rocprim::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value); // used for input
        std::vector<T> output(input.size() / logical_warp_size, 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 1);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size(); i++)
        {
            T value = 0;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                value = apply(binary_op, value, input[idx]);
            }
            expected[i] = value;
        }

        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size, logical_warp_size>),
            dim3(size/block_size), dim3(block_size), 0, 0,
            device_input, device_output
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }

}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void warp_allreduce_sum_kernel(T* device_input, T* device_output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize, true>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, storage[warp_id]);

    device_output[index] = value;
}

TYPED_TEST(RocprimWarpReduceTests, AllReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_plus, rocprim::plus<T>>::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(rocprim::warp_size(), logical_warp_size * 4)
            : (rocprim::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value); // used for input
        std::vector<T> output(input.size(), 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                value = apply(binary_op, value, input[idx]);
            }
            for (size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                expected[idx] = value;
            }
        }

        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_allreduce_sum_kernel<T, block_size, logical_warp_size>),
            dim3(size/block_size), dim3(block_size), 0, 0,
            device_input, device_output
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }

}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void warp_reduce_sum_kernel(T* device_input, T* device_output, size_t valid)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, valid, storage[warp_id]);

    if(hipThreadIdx_x%LogicalWarpSize == 0)
    {
        device_output[index/LogicalWarpSize] = value;
    }
}

TYPED_TEST(RocprimWarpReduceTests, ReduceSumValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_plus, rocprim::plus<T>>::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(rocprim::warp_size(), logical_warp_size * 4)
            : (rocprim::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;
    const size_t valid = logical_warp_size - 1;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value); // used for input
        std::vector<T> output(input.size() / logical_warp_size, 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 1);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size(); i++)
        {
            T value = 0;
            for(size_t j = 0; j < valid; j++)
            {
                auto idx = i * logical_warp_size + j;
                value = apply(binary_op, value, input[idx]);
            }
            expected[i] = value;
        }

        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size, logical_warp_size>),
            dim3(size/block_size), dim3(block_size), 0, 0,
            device_input, device_output, valid
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }

}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void warp_allreduce_sum_kernel(T* device_input, T* device_output, size_t valid)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize, true>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, valid, storage[warp_id]);

    device_output[index] = value;
}

TYPED_TEST(RocprimWarpReduceTests, AllReduceSumValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_plus, rocprim::plus<T>>::type;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(rocprim::warp_size(), logical_warp_size * 4)
            : (rocprim::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;
    const size_t valid = logical_warp_size - 1;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value); // used for input
        std::vector<T> output(input.size(), 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < valid; j++)
            {
                auto idx = i * logical_warp_size + j;
                value = apply(binary_op, value, input[idx]);
            }
            for (size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                expected[idx] = value;
            }
        }

        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_allreduce_sum_kernel<T, block_size, logical_warp_size>),
            dim3(size/block_size), dim3(block_size), 0, 0,
            device_input, device_output, valid
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }

}

TYPED_TEST(RocprimWarpReduceTests, ReduceSumCustomStruct)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using base_type = typename TestFixture::params::type;
    using T = test_utils::custom_test_type<base_type>;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(rocprim::warp_size(), logical_warp_size * 4)
            : (rocprim::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input(size);
        {
            auto random_values =
                test_utils::get_random_data<base_type>(2 * input.size(), 2, 50, seed_value);
            for(size_t i = 0; i < input.size(); i++)
            {
                input[i].x = random_values[i];
                input[i].y = random_values[i + input.size()];
            }
        }
        std::vector<T> output(input.size() / logical_warp_size);

        // Calculate expected results on host
        std::vector<T> expected(output.size());
        for(size_t i = 0; i < output.size(); i++)
        {
            T value(0, 0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                value = value + input[idx];
            }
            expected[i] = value;
        }

        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size, logical_warp_size>),
            dim3(size/block_size), dim3(block_size), 0, 0,
            device_input, device_output
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }

}

template<
    class T,
    class Flag,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void head_segmented_warp_reduce_kernel(T* input, Flag* flags, T* output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = input[index];
    auto flag = flags[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize, true>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().head_segmented_reduce(value, value, flag, storage[warp_id]);

    output[index] = value;
}

TYPED_TEST(RocprimWarpReduceTests, HeadSegmentedReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_plus, rocprim::plus<T>>::type;
    using flag_type = unsigned char;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(rocprim::warp_size(), logical_warp_size * 4)
            : (rocprim::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value); // used for input
        std::vector<flag_type> flags = test_utils::get_random_data01<flag_type>(size, 0.25f, seed_value);
        for(size_t i = 0; i < flags.size(); i+= logical_warp_size)
        {
            flags[i] = 1;
        }
        std::vector<T> output(input.size());

        T* device_input;
        flag_type* device_flags;
        T* device_output;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        HIP_CHECK(hipMalloc(&device_flags, flags.size() * sizeof(typename decltype(flags)::value_type)));
        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                device_flags, flags.data(),
                flags.size() * sizeof(flag_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host
        std::vector<T> expected(output.size());
        binary_op_type binary_op;
        size_t segment_head_index = 0;
        T reduction = input[0];
        for(size_t i = 0; i < output.size(); i++)
        {
            if(i%logical_warp_size == 0 || flags[i])
            {
                expected[segment_head_index] = reduction;
                segment_head_index = i;
                reduction = input[i];
            }
            else
            {
                reduction = apply(binary_op, reduction, input[i]);
            }
        }
        expected[segment_head_index] = reduction;

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(head_segmented_warp_reduce_kernel<
                T, flag_type, block_size, logical_warp_size
            >),
            dim3(size/block_size), dim3(block_size), 0, 0,
            device_input, device_flags, device_output
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> output_segment(output.size(), 0);
        std::vector<T> expected_segment(output.size(), 0);
        for(size_t i = 0; i < output.size(); i++)
        {
            if(flags[i])
            {
                output_segment[i] = output[i];
                expected_segment[i] = expected[i];
            }
        }
        test_utils::assert_near(output_segment, expected_segment, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_flags));
        HIP_CHECK(hipFree(device_output));
    }

}

template<
    class T,
    class Flag,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void tail_segmented_warp_reduce_kernel(T* input, Flag* flags, T* output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = input[index];
    auto flag = flags[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize, true>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().tail_segmented_reduce(value, value, flag, storage[warp_id]);

    output[index] = value;
}

TYPED_TEST(RocprimWarpReduceTests, TailSegmentedReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_plus, rocprim::plus<T>>::type;
    using flag_type = unsigned char;
    constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    constexpr size_t block_size =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(rocprim::warp_size(), logical_warp_size * 4)
            : (rocprim::warp_size()/logical_warp_size) * logical_warp_size;
    const size_t size = block_size * 4;

    // Given warp size not supported
    if(logical_warp_size > rocprim::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10, seed_value); // used for input
        std::vector<flag_type> flags = test_utils::get_random_data01<flag_type>(size, 0.25f, seed_value);
        for(size_t i = logical_warp_size - 1; i < flags.size(); i+= logical_warp_size)
        {
            flags[i] = 1;
        }
        std::vector<T> output(input.size());

        T* device_input;
        flag_type* device_flags;
        T* device_output;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        HIP_CHECK(hipMalloc(&device_flags, flags.size() * sizeof(typename decltype(flags)::value_type)));
        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                device_flags, flags.data(),
                flags.size() * sizeof(flag_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host
        std::vector<T> expected(output.size());
        binary_op_type binary_op;
        std::vector<size_t> segment_indexes;
        size_t segment_index = 0;
        T reduction;
        for(size_t i = 0; i < output.size(); i++)
        {
            // single value segments
            if(flags[i])
            {
                expected[i] = input[i];
                segment_indexes.push_back(i);
            }
            else
            {
                segment_index = i;
                reduction = input[i];
                auto next = i + 1;
                while(next < output.size() && !flags[next])
                {
                    reduction = apply(binary_op, reduction, input[next]);
                    i++;
                    next++;
                }
                i++;
                expected[segment_index] = apply(binary_op, reduction, input[i]);
                segment_indexes.push_back(segment_index);
            }
        }

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(tail_segmented_warp_reduce_kernel<
                T, flag_type, block_size, logical_warp_size
            >),
            dim3(size/block_size), dim3(block_size), 0, 0,
            device_input, device_flags, device_output
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> output_segment(segment_indexes.size());
        std::vector<T> expected_segment(segment_indexes.size());
        for(size_t i = 0; i < segment_indexes.size(); i++)
        {
            auto index = segment_indexes[i];
            output_segment[i] = output[index];
            expected_segment[i] = expected[index];
        }
        test_utils::assert_near(output_segment, expected_segment, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_flags));
        HIP_CHECK(hipFree(device_output));
    }

}
