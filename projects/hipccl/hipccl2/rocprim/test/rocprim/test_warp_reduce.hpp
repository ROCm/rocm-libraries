// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_device_ptr.hpp"

#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"

#include <rocprim/config.hpp>
#include <rocprim/detail/various.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/functional.hpp>

#include <cstddef>
#include <vector>

test_suite_type_def(suite_name, name_suffix)

    typed_test_suite_def(RocprimWarpReduceTests, name_suffix, warp_params);

typed_test_def(RocprimWarpReduceTests, name_suffix, ReduceSum)
{
    // logical warp size for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    // Block size of warp size 32
    static constexpr size_t block_size_ws32
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
              : rocprim::max<size_t>((ws32 / logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
              : rocprim::max<size_t>((ws64 / logical_warp_size), 1) * logical_warp_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    static constexpr unsigned int grid_size = 4;
    const size_t                  size      = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 50, seed_value);
        std::vector<T> output(input.size() / logical_warp_size, T(0));

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(1));
        for(size_t i = 0; i < output.size(); i++)
        {
            acc_type value(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                value    = binary_op_host(input[idx], value);
            }
            expected[i] = static_cast<T>(value);
        }

        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(size / block_size_ws32),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(size / block_size_ws64),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get());
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpReduceTests, name_suffix, AllReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp size for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    // Block size of warp size 32
    static constexpr size_t block_size_ws32
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
              : rocprim::max<size_t>((ws32 / logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
              : rocprim::max<size_t>((ws64 / logical_warp_size), 1) * logical_warp_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    static constexpr unsigned int grid_size = 4;
    const size_t                  size      = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 50, seed_value);
        std::vector<T> output(input.size(), T(0));

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            acc_type value(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                value    = binary_op_host(input[idx], value);
            }
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                expected[idx] = static_cast<T>(value);
            }
        }

        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_allreduce_sum_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(size / block_size_ws32),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_allreduce_sum_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(size / block_size_ws64),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get());
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpReduceTests, name_suffix, ReduceSumValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp size for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    // Block size of warp size 32
    static constexpr size_t block_size_ws32
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
              : rocprim::max<size_t>((ws32 / logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
              : rocprim::max<size_t>((ws64 / logical_warp_size), 1) * logical_warp_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    static constexpr unsigned int grid_size = 4;
    const size_t                  size      = block_size * grid_size;
    const size_t                  valid     = logical_warp_size - 1;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 50, seed_value);
        std::vector<T> output(input.size() / logical_warp_size, T(0));

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(1));
        for(size_t i = 0; i < output.size(); i++)
        {
            acc_type value(0);
            for(size_t j = 0; j < valid; j++)
            {
                auto idx = i * logical_warp_size + j;
                value    = binary_op_host(input[idx], value);
            }
            expected[i] = static_cast<T>(value);
        }

        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(size / block_size_ws32),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get(),
                valid);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(size / block_size_ws64),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get(),
                valid);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpReduceTests, name_suffix, AllReduceSumValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp size for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    // Block size of warp size 32
    static constexpr size_t block_size_ws32
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
              : rocprim::max<size_t>((ws32 / logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
              : rocprim::max<size_t>((ws64 / logical_warp_size), 1) * logical_warp_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    static constexpr unsigned int grid_size = 4;
    const size_t                  size      = block_size * grid_size;
    const size_t                  valid     = logical_warp_size - 1;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 50, seed_value);
        std::vector<T> output(input.size(), T(0));

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            acc_type value(0);
            for(size_t j = 0; j < valid; j++)
            {
                auto idx = i * logical_warp_size + j;
                value    = binary_op_host(input[idx], value);
            }
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                expected[idx] = static_cast<T>(value);
            }
        }

        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_allreduce_sum_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(size / block_size_ws32),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get(),
                valid);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_allreduce_sum_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(size / block_size_ws64),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get(),
                valid);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        test_utils::assert_near(output, expected, test_utils::precision<T> * valid);
    }
}

typed_test_def(RocprimWarpReduceTests, name_suffix, ReduceCustomStruct)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using base_type = typename TestFixture::params::type;
    using T         = common::custom_type<base_type, base_type, true>;
    using acc_type  = typename test_utils::select_plus_operator_host<base_type>::acc_type;

    // logical warp size for warp primitive, execution warp size is always rocprim::warp_size()
    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    // Block size of warp size 32
    static constexpr size_t block_size_ws32
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
              : rocprim::max<size_t>((ws32 / logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
              : rocprim::max<size_t>((ws64 / logical_warp_size), 1) * logical_warp_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    static constexpr unsigned int grid_size = 4;
    const size_t                  size      = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input(size);
        {
            auto random_values = test_utils::get_random_data_wrapped<base_type>(2 * input.size(),
                                                                                2,
                                                                                50,
                                                                                seed_value);
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
            common::custom_type<acc_type, acc_type, true> value{(acc_type)0, (acc_type)0};
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                value    = value
                        + static_cast<common::custom_type<acc_type, acc_type, true>>(input[idx]);
            }
            expected[i] = static_cast<T>(value);
        }

        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(size / block_size_ws32),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_reduce_sum_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(size / block_size_ws64),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get());
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<base_type> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpReduceTests, name_suffix, HeadSegmentedReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp size for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    using flag_type                           = unsigned char;
    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    // Block size of warp size 32
    static constexpr size_t block_size_ws32
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
              : rocprim::max<size_t>((ws32 / logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
              : rocprim::max<size_t>((ws64 / logical_warp_size), 1) * logical_warp_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    static constexpr unsigned int grid_size = 4;
    const size_t                  size      = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 10, seed_value);
        std::vector<flag_type> flags
            = test_utils::get_random_data01<flag_type>(size, 0.25f, seed_value);
        for(size_t i = 0; i < flags.size(); i += logical_warp_size)
        {
            flags[i] = 1;
        }
        std::vector<T> output(input.size());

        common::device_ptr<T>         device_input(input);
        common::device_ptr<flag_type> device_flags(flags);
        common::device_ptr<T>         device_output(output.size());

        // Calculate expected results on host
        std::vector<T> expected(output.size());
        size_t         segment_head_index = 0;
        acc_type       reduction(input[0]);
        for(size_t i = 0; i < output.size(); i++)
        {
            if(i % logical_warp_size == 0 || flags[i])
            {
                expected[segment_head_index] = static_cast<T>(reduction);
                segment_head_index           = i;
                reduction                    = input[i];
            }
            else
            {
                reduction = binary_op_host(input[i], reduction);
            }
        }
        expected[segment_head_index] = static_cast<T>(reduction);

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(head_segmented_warp_reduce_kernel<T,
                                                                  flag_type,
                                                                  block_size_ws32,
                                                                  logical_warp_size>),
                dim3(size / block_size_ws32),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_flags.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(head_segmented_warp_reduce_kernel<T,
                                                                  flag_type,
                                                                  block_size_ws64,
                                                                  logical_warp_size>),
                dim3(size / block_size_ws64),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_flags.get(),
                device_output.get());
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        std::vector<T> output_segment(output.size(), T(0));
        std::vector<T> expected_segment(output.size(), T(0));
        for(size_t i = 0; i < output.size(); i++)
        {
            if(flags[i])
            {
                output_segment[i]   = output[i];
                expected_segment[i] = expected[i];
            }
        }
        test_utils::assert_near(output_segment,
                                expected_segment,
                                test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpReduceTests, name_suffix, TailSegmentedReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp size for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    using flag_type                           = unsigned char;
    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    // Block size of warp size 32
    static constexpr size_t block_size_ws32
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
              : rocprim::max<size_t>((ws32 / logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64
        = rocprim::detail::is_power_of_two(logical_warp_size)
              ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
              : rocprim::max<size_t>((ws64 / logical_warp_size), 1) * logical_warp_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    static constexpr unsigned int grid_size = 4;
    const size_t                  size      = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 10, seed_value);
        std::vector<flag_type> flags
            = test_utils::get_random_data01<flag_type>(size, 0.25f, seed_value);
        for(size_t i = logical_warp_size - 1; i < flags.size(); i += logical_warp_size)
        {
            flags[i] = 1;
        }
        std::vector<T> output(input.size());

        common::device_ptr<T>         device_input(input);
        common::device_ptr<flag_type> device_flags(flags);
        common::device_ptr<T>         device_output(output.size());

        // Calculate expected results on host
        std::vector<T>      expected(output.size());
        std::vector<size_t> segment_indexes;
        size_t              segment_index = 0;
        acc_type            reduction;
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
                reduction     = input[i];
                auto next     = i + 1;
                while(next < output.size() && !flags[next])
                {
                    reduction = binary_op_host(input[next], reduction);
                    i++;
                    next++;
                }
                i++;
                expected[segment_index] = static_cast<T>(binary_op_host(reduction, input[i]));
                segment_indexes.push_back(segment_index);
            }
        }

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(tail_segmented_warp_reduce_kernel<T,
                                                                  flag_type,
                                                                  block_size_ws32,
                                                                  logical_warp_size>),
                dim3(size / block_size_ws32),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_flags.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(tail_segmented_warp_reduce_kernel<T,
                                                                  flag_type,
                                                                  block_size_ws64,
                                                                  logical_warp_size>),
                dim3(size / block_size_ws64),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_flags.get(),
                device_output.get());
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        std::vector<T> output_segment(segment_indexes.size());
        std::vector<T> expected_segment(segment_indexes.size());
        for(size_t i = 0; i < segment_indexes.size(); i++)
        {
            auto index          = segment_indexes[i];
            output_segment[i]   = output[index];
            expected_segment[i] = expected[index];
        }
        test_utils::assert_near(output_segment,
                                expected_segment,
                                test_utils::precision<T> * logical_warp_size);
    }
}
