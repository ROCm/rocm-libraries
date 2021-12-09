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

test_suite_type_def(suite_name, name_suffix)

typed_test_suite_def(RocprimWarpScanTests, name_suffix, warp_params);

typed_test_def(RocprimWarpScanTests, name_suffix, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
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
    static constexpr size_t block_size_ws32 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
            : rocprim::max<size_t>((ws32/logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
            : rocprim::max<size_t>((ws64/logical_warp_size), 1) * logical_warp_size;

    const unsigned int current_device_warp_size = rocprim::host_warp_size();

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    const unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(output.size(), (T)0);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size), 0, 0,
                device_input, device_output
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size), 0, 0,
                device_input, device_output
            );
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }

}

typed_test_def(RocprimWarpScanTests, name_suffix, InclusiveScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
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
    static constexpr size_t block_size_ws32 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
            : rocprim::max<size_t>((ws32/logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
            : rocprim::max<size_t>((ws64/logical_warp_size), 1) * logical_warp_size;

    const unsigned int current_device_warp_size = rocprim::host_warp_size();

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    const unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(output.size(), (T)0);
        std::vector<T> expected_reductions(output_reductions.size(), (T)0);

        // Calculate expected results on host
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            acc_type accumulator(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator = binary_op_host(input[idx],accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
            expected_reductions[i] = expected[(i+1) * logical_warp_size - 1];
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_output_reductions,
                output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_reduce_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output, device_output_reductions
            );
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_reduce_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output, device_output_reductions
            );
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);
        test_utils::assert_near(output_reductions, expected_reductions, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

typed_test_def(RocprimWarpScanTests, name_suffix, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
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
    static constexpr size_t block_size_ws32 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
            : rocprim::max<size_t>((ws32/logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
            : rocprim::max<size_t>((ws64/logical_warp_size), 1) * logical_warp_size;

    const unsigned int current_device_warp_size = rocprim::host_warp_size();

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    const unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(input.size(), (T)0);
        const T init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(init);
            expected[i * logical_warp_size] = init;
            for(size_t j = 1; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator = binary_op_host(input[idx-1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output, init
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output, init
            );
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }

}

//typed_test_def(RocprimWarpScanTests, name_suffix, ExclusiveReduceScan)
typed_test_def(RocprimWarpScanTests, name_suffix, ExclusiveReduceScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
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
    static constexpr size_t block_size_ws32 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
            : rocprim::max<size_t>((ws32/logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
            : rocprim::max<size_t>((ws64/logical_warp_size), 1) * logical_warp_size;

    const unsigned int current_device_warp_size = rocprim::host_warp_size();

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    const unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(input.size(), (T)0);
        std::vector<T> expected_reductions(output_reductions.size(), (T)0);
        const T init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(init);
            expected[i * logical_warp_size] = init;
            for(size_t j = 1; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator = binary_op_host(input[idx-1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }

            acc_type accumulator_reductions(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator_reductions = binary_op_host(input[idx], accumulator_reductions);
                expected_reductions[i] = static_cast<T>(accumulator_reductions);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_output_reductions,
                output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_reduce_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output, device_output_reductions, init
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_reduce_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output, device_output_reductions, init
            );
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision_threshold<T>::percentage);
        test_utils::assert_near(output_reductions, expected_reductions, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

typed_test_def(RocprimWarpScanTests, name_suffix, Scan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
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
    static constexpr size_t block_size_ws32 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
            : rocprim::max<size_t>((ws32/logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
            : rocprim::max<size_t>((ws64/logical_warp_size), 1) * logical_warp_size;

    const unsigned int current_device_warp_size = rocprim::host_warp_size();

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    const unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output_inclusive(size);
        std::vector<T> output_exclusive(size);
        std::vector<T> expected_inclusive(output_inclusive.size(), (T)0);
        std::vector<T> expected_exclusive(output_exclusive.size(), (T)0);
        const T init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator_inclusive(0);
            acc_type accumulator_exclusive = init;
            expected_exclusive[i * logical_warp_size] = init;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator_inclusive = binary_op_host(input[idx], accumulator_inclusive);
                expected_inclusive[idx] = static_cast<T>(accumulator_inclusive);
                if(j > 0)
                {
                    accumulator_exclusive = binary_op_host(input[idx-1], accumulator_exclusive);
                    expected_exclusive[idx] = static_cast<T>(accumulator_exclusive);
                }
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_inclusive_output;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_inclusive_output,
                output_inclusive.size() * sizeof(typename decltype(output_inclusive)::value_type)
            )
        );
        T* device_exclusive_output;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_exclusive_output,
                output_exclusive.size() * sizeof(typename decltype(output_exclusive)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_inclusive_output, device_exclusive_output, init
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_inclusive_output, device_exclusive_output, init
            );
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output_inclusive.data(), device_inclusive_output,
                output_inclusive.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_exclusive.data(), device_exclusive_output,
                output_exclusive.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        test_utils::assert_near(output_inclusive, expected_inclusive, test_utils::precision_threshold<T>::percentage);
        test_utils::assert_near(output_exclusive, expected_exclusive, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_inclusive_output));
        HIP_CHECK(hipFree(device_exclusive_output));
    }

}

typed_test_def(RocprimWarpScanTests, name_suffix, ScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
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
    static constexpr size_t block_size_ws32 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
            : rocprim::max<size_t>((ws32/logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
            : rocprim::max<size_t>((ws64/logical_warp_size), 1) * logical_warp_size;

    const unsigned int current_device_warp_size = rocprim::host_warp_size();

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    const unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output_inclusive(size);
        std::vector<T> output_exclusive(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected_inclusive(output_inclusive.size(), (T)0);
        std::vector<T> expected_exclusive(output_exclusive.size(), (T)0);
        std::vector<T> expected_reductions(output_reductions.size(), (T)0);
        const T init = test_utils::get_random_value<T>(0, 100, seed_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator_inclusive(0);
            acc_type accumulator_exclusive(init);
            expected_exclusive[i * logical_warp_size] = init;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator_inclusive = binary_op_host(input[idx], accumulator_inclusive);
                expected_inclusive[idx] = static_cast<T>(accumulator_inclusive);
                if(j > 0)
                {
                    accumulator_exclusive = binary_op_host(input[idx-1], accumulator_exclusive);
                    expected_exclusive[idx] = static_cast<T>(accumulator_exclusive);
                }
            }
            expected_reductions[i] = expected_inclusive[(i+1) * logical_warp_size - 1];
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_inclusive_output;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_inclusive_output,
                output_inclusive.size() * sizeof(typename decltype(output_inclusive)::value_type)
            )
        );
        T* device_exclusive_output;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_exclusive_output,
                output_exclusive.size() * sizeof(typename decltype(output_exclusive)::value_type)
            )
        );
        T* device_output_reductions;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_output_reductions,
                output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_reduce_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input,
                device_inclusive_output, device_exclusive_output, device_output_reductions, init
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_reduce_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input,
                device_inclusive_output, device_exclusive_output, device_output_reductions, init
            );
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output_inclusive.data(), device_inclusive_output,
                output_inclusive.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_exclusive.data(), device_exclusive_output,
                output_exclusive.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        test_utils::assert_near(output_inclusive, expected_inclusive, test_utils::precision_threshold<T>::percentage);
        test_utils::assert_near(output_exclusive, expected_exclusive, test_utils::precision_threshold<T>::percentage);
        test_utils::assert_near(output_reductions, expected_reductions, test_utils::precision_threshold<T>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_inclusive_output));
        HIP_CHECK(hipFree(device_exclusive_output));
    }

}

typed_test_def(RocprimWarpScanTests, name_suffix, InclusiveScanCustomType)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using base_type = typename TestFixture::params::type;
    using T = test_utils::custom_test_type<base_type>;
    using acc_type = base_type;
    if(std::is_same<acc_type, rocprim::bfloat16>::value)
        GTEST_SKIP();
    /* Enable the following when assignment operator has been added to hip_bfloat16
     * https://github.com/ROCm-Developer-Tools/HIP/pull/2303/files
     * using acc_type = typename std::conditional<std::is_same<base_type,rocprim::bfloat16>::value ||
                                                   std::is_same<base_type,rocprim::half>::value, double, base_type>::type; */

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    // Block size of warp size 32
    static constexpr size_t block_size_ws32 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws32, logical_warp_size * 4)
            : rocprim::max<size_t>((ws32/logical_warp_size), 1) * logical_warp_size;

    // Block size of warp size 64
    static constexpr size_t block_size_ws64 =
        rocprim::detail::is_power_of_two(logical_warp_size)
            ? rocprim::max<size_t>(ws64, logical_warp_size * 4)
            : rocprim::max<size_t>((ws64/logical_warp_size), 1) * logical_warp_size;

    const unsigned int current_device_warp_size = rocprim::host_warp_size();

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    const unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input(size);
        std::vector<T> output(size);
        std::vector<T> expected(output.size(), (base_type)0);
        // Initializing input data
        {
            auto random_values =
                test_utils::get_random_data<base_type>(2 * input.size(), 0, 100, seed_value);
            for(size_t i = 0; i < input.size(); i++)
            {
                input[i].x = random_values[i];
                input[i].y = random_values[i + input.size()];
            }
        }

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            test_utils::custom_test_type<acc_type> accumulator(acc_type(0));
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator = static_cast<test_utils::custom_test_type<acc_type>>(input[idx]) + accumulator;
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output
            );
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision_threshold<base_type>::percentage);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}
