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

#if _CLANGD
    #include "test_warp_scan.cpp"
#endif

test_suite_type_def(suite_name, name_suffix);

typed_test_suite_def(RocprimWarpScanTests, name_suffix, warp_params);

typed_test_def(RocprimWarpScanTests, name_suffix, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> output(size);
        std::vector<T> expected(output.size(), T(0));

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size),
                0,
                0,
                device_input.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size),
                0,
                0,
                device_input.get(),
                device_output.get());
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, InclusiveScanInitialValue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(output.size(), T(0));
        T              initial_value = test_utils::get_random_data<T>(1, 2, 50, seed_value)[0];
        SCOPED_TRACE(testing::Message() << "with initial_value = " << initial_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(initial_value);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_initial_value_kernel<T,
                                                                         block_size_ws32,
                                                                         logical_warp_size>),
                dim3(grid_size),
                dim3(block_size),
                0,
                0,
                device_input.get(),
                device_output.get(),
                initial_value);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_initial_value_kernel<T,
                                                                         block_size_ws64,
                                                                         logical_warp_size>),
                dim3(grid_size),
                dim3(block_size),
                0,
                0,
                device_input.get(),
                device_output.get(),
                initial_value);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, InclusiveScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(output.size(), T(0));
        std::vector<T> expected_reductions(output_reductions.size(), T(0));

        // Calculate expected results on host
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            acc_type accumulator(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
            expected_reductions[i] = expected[(i + 1) * logical_warp_size - 1];
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());
        common::device_ptr<T> device_output_reductions(output_reductions.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    warp_inclusive_scan_reduce_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get(),
                device_output_reductions.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    warp_inclusive_scan_reduce_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get(),
                device_output_reductions.get());
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output            = device_output.load();
        output_reductions = device_output_reductions.load();

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
        test_utils::assert_near(output_reductions,
                                expected_reductions,
                                test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, InclusiveScanReduceInitialValue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(output.size(), T(0));
        std::vector<T> expected_reductions(output_reductions.size(), T(0));
        T              initial_value = test_utils::get_random_data<T>(1, 2, 50, seed_value)[0];
        SCOPED_TRACE(testing::Message() << "with initial_value = " << initial_value);

        // Calculate expected results on host
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            acc_type accumulator = acc_type(initial_value);
            acc_type reduction   = input[i * logical_warp_size];
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
                if(j > 0)
                {
                    reduction = binary_op_host(input[idx], reduction);
                }
            }
            expected_reductions[i] = static_cast<T>(reduction);
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());
        common::device_ptr<T> device_output_reductions(output_reductions.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_reduce_initial_value_kernel<T,
                                                                                block_size_ws32,
                                                                                logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get(),
                device_output_reductions.get(),
                initial_value);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_reduce_initial_value_kernel<T,
                                                                                block_size_ws64,
                                                                                logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get(),
                device_output_reductions.get(),
                initial_value);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output            = device_output.load();
        output_reductions = device_output_reductions.load();

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
        test_utils::assert_near(output_reductions,
                                expected_reductions,
                                test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> output(size);
        std::vector<T> expected(input.size(), T(0));
        const T        init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(init);
            expected[i * logical_warp_size] = init;
            for(size_t j = 1; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx - 1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get(),
                init);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get(),
                init);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, Broadcast)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    static constexpr size_t logical_warp_size_input = TestFixture::params::warp_size;
    static constexpr size_t logical_warp_size
        = !rocprim::detail::is_power_of_two(logical_warp_size_input)
              ? rocprim::detail::next_power_of_two(logical_warp_size_input)
              : logical_warp_size_input;

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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        GTEST_SKIP() << "Unsupported test warp size/computed block size: " << logical_warp_size
                     << "/" << block_size
                     << " Current device warp size: " << current_device_warp_size;
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(output.size());

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                auto warp_id  = idx / logical_warp_size;
                auto src_lane = warp_id % logical_warp_size;
                expected[idx] = static_cast<T>(input[i * logical_warp_size + src_lane]);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_broadcast_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size),
                0,
                0,
                device_input.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_broadcast_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size),
                0,
                0,
                device_input.get(),
                device_output.get());
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output = device_output.load();

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T>);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, ExclusiveScanWoInit)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    // The different warp sizes
    static constexpr size_t ws32{ROCPRIM_WARP_SIZE_32};
    static constexpr size_t ws64{ROCPRIM_WARP_SIZE_64};

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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%d.    Skipping test\n",
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
        std::vector<T> output(size);
        std::vector<T> expected(input.size(), T(0));

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            // expected[i * logical_warp_size] is unspecified because init is not passed
            acc_type accumulator(input[i * logical_warp_size]);

            static_assert(logical_warp_size > 2, "logical_warp_size assumed to be at least 2.");
            expected[i * logical_warp_size + 1] = static_cast<T>(accumulator);

            for(size_t j = 2; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx - 1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    warp_exclusive_scan_wo_init_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    warp_exclusive_scan_wo_init_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
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

        // The first value of each logical warp has an unspecified result, expect whatever we got
        // for those values to not fail the test.
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            expected[i * logical_warp_size] = output[i * logical_warp_size];
        }

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
    }
}

//typed_test_def(RocprimWarpScanTests, name_suffix, ExclusiveReduceScan)
typed_test_def(RocprimWarpScanTests, name_suffix, ExclusiveReduceScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(input.size(), T(0));
        std::vector<T> expected_reductions(output_reductions.size(), T(0));
        const T        init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(init);
            expected[i * logical_warp_size] = init;
            for(size_t j = 1; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx - 1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }

            acc_type accumulator_reductions(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx               = i * logical_warp_size + j;
                accumulator_reductions = binary_op_host(input[idx], accumulator_reductions);
                expected_reductions[i] = static_cast<T>(accumulator_reductions);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());
        common::device_ptr<T> device_output_reductions(output_reductions.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    warp_exclusive_scan_reduce_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get(),
                device_output_reductions.get(),
                init);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    warp_exclusive_scan_reduce_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get(),
                device_output_reductions.get(),
                init);
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output            = device_output.load();
        output_reductions = device_output_reductions.load();

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
        test_utils::assert_near(output_reductions,
                                expected_reductions,
                                test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, ExclusiveReduceScanWoInit)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only VirtualWaveSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%d.    Skipping test\n",
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
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(input.size(), T(0));
        std::vector<T> expected_reductions(output_reductions.size(), T(0));

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            // expected[i * logical_warp_size] is unspecified because init is not passed
            acc_type accumulator(input[i * logical_warp_size]);

            static_assert(logical_warp_size > 2, "logical_warp_size assumed to be at least 2.");
            expected[i * logical_warp_size + 1] = static_cast<T>(accumulator);

            for(size_t j = 2; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx - 1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }

            acc_type accumulator_reductions(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx               = i * logical_warp_size + j;
                accumulator_reductions = binary_op_host(input[idx], accumulator_reductions);
                expected_reductions[i] = static_cast<T>(accumulator_reductions);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());
        common::device_ptr<T> device_output_reductions(output_reductions.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_reduce_wo_init_kernel<T,
                                                                          block_size_ws32,
                                                                          logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get(),
                device_output_reductions.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_reduce_wo_init_kernel<T,
                                                                          block_size_ws64,
                                                                          logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_output.get(),
                device_output_reductions.get());
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output            = device_output.load();
        output_reductions = device_output_reductions.load();

        // The first value of each logical warp has an unspecified result, expect whatever we got
        // for those values to not fail the test.
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            expected[i * logical_warp_size] = output[i * logical_warp_size];
        }

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T> * logical_warp_size);
        test_utils::assert_near(output_reductions,
                                expected_reductions,
                                test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, Scan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> output_inclusive(size);
        std::vector<T> output_exclusive(size);
        std::vector<T> expected_inclusive(output_inclusive.size(), T(0));
        std::vector<T> expected_exclusive(output_exclusive.size(), T(0));
        const T        init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator_inclusive(0);
            acc_type accumulator_exclusive            = init;
            expected_exclusive[i * logical_warp_size] = init;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx                = i * logical_warp_size + j;
                accumulator_inclusive   = binary_op_host(input[idx], accumulator_inclusive);
                expected_inclusive[idx] = static_cast<T>(accumulator_inclusive);
                if(j > 0)
                {
                    accumulator_exclusive   = binary_op_host(input[idx - 1], accumulator_exclusive);
                    expected_exclusive[idx] = static_cast<T>(accumulator_exclusive);
                }
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_inclusive_output(output_inclusive.size());
        common::device_ptr<T> device_exclusive_output(output_exclusive.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_inclusive_output.get(),
                device_exclusive_output.get(),
                init);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_inclusive_output.get(),
                device_exclusive_output.get(),
                init);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output_inclusive = device_inclusive_output.load();
        output_exclusive = device_exclusive_output.load();

        // Validating results
        test_utils::assert_near(output_inclusive,
                                expected_inclusive,
                                test_utils::precision<T> * logical_warp_size);
        test_utils::assert_near(output_exclusive,
                                expected_exclusive,
                                test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, ScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> output_inclusive(size);
        std::vector<T> output_exclusive(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected_inclusive(output_inclusive.size(), T(0));
        std::vector<T> expected_exclusive(output_exclusive.size(), T(0));
        std::vector<T> expected_reductions(output_reductions.size(), T(0));
        const T        init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator_inclusive(0);
            acc_type accumulator_exclusive(init);
            expected_exclusive[i * logical_warp_size] = init;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx                = i * logical_warp_size + j;
                accumulator_inclusive   = binary_op_host(input[idx], accumulator_inclusive);
                expected_inclusive[idx] = static_cast<T>(accumulator_inclusive);
                if(j > 0)
                {
                    accumulator_exclusive   = binary_op_host(input[idx - 1], accumulator_exclusive);
                    expected_exclusive[idx] = static_cast<T>(accumulator_exclusive);
                }
            }
            expected_reductions[i] = expected_inclusive[(i + 1) * logical_warp_size - 1];
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_inclusive_output(output_inclusive.size());
        common::device_ptr<T> device_exclusive_output(output_exclusive.size());
        common::device_ptr<T> device_output_reductions(output_reductions.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_reduce_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_inclusive_output.get(),
                device_exclusive_output.get(),
                device_output_reductions.get(),
                init);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_reduce_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input.get(),
                device_inclusive_output.get(),
                device_exclusive_output.get(),
                device_output_reductions.get(),
                init);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        output_inclusive  = device_inclusive_output.load();
        output_exclusive  = device_exclusive_output.load();
        output_reductions = device_output_reductions.load();

        // Validating results
        test_utils::assert_near(output_inclusive,
                                expected_inclusive,
                                test_utils::precision<T> * logical_warp_size);
        test_utils::assert_near(output_exclusive,
                                expected_exclusive,
                                test_utils::precision<T> * logical_warp_size);
        test_utils::assert_near(output_reductions,
                                expected_reductions,
                                test_utils::precision<T> * logical_warp_size);
    }
}

typed_test_def(RocprimWarpScanTests, name_suffix, InclusiveScanCustomType)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using base_type = typename TestFixture::params::type;
    using T         = common::custom_type<base_type, base_type, true>;
    using acc_type  = typename test_utils::select_plus_operator_host<base_type>::acc_type;

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
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
    const unsigned int grid_size = 4;

    const size_t size = block_size * grid_size;

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
        std::vector<T> output(size);
        std::vector<T> expected(output.size(), (base_type)0);
        // Initializing input data
        {
            auto random_values = test_utils::get_random_data_wrapped<base_type>(2 * input.size(),
                                                                                0,
                                                                                100,
                                                                                seed_value);
            for(size_t i = 0; i < input.size(); i++)
            {
                input[i].x = random_values[i];
                input[i].y = random_values[i + input.size()];
            }
        }

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            common::custom_type<acc_type, acc_type, true> accumulator(acc_type(0));
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx    = i * logical_warp_size + j;
                accumulator = static_cast<common::custom_type<acc_type, acc_type, true>>(input[idx])
                              + accumulator;
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(output.size());

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input.get(),
                device_output.get());
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size),
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

        // Validating results
        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<base_type> * logical_warp_size);
    }
}
