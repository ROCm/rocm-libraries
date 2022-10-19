// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

typed_test_suite_def(RocprimWarpSortShuffleBasedTests, name_suffix, warp_params);

typed_test_def(RocprimWarpSortShuffleBasedTests, name_suffix, Sort)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using binary_op_type = rocprim::less<T>;

    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    static constexpr size_t items_per_thread = TestFixture::params::items_per_thread;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    const unsigned int current_device_warp_size = rocprim::host_warp_size();
    static constexpr size_t block_size = std::max<size_t>(256U, logical_warp_size * 4);

    static constexpr unsigned int grid_size = 4;
    const size_t size = items_per_thread * block_size * grid_size;

    SCOPED_TRACE(testing::Message() << "with size = " << size);

    // Check if warp size is supported
    if( logical_warp_size > current_device_warp_size ||
        !rocprim::detail::is_power_of_two(logical_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 0, 100, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / logical_warp_size / items_per_thread; i++)
        {
            std::sort(expected.begin() + (i * logical_warp_size * items_per_thread),
                      expected.begin() + ((i + 1) * logical_warp_size * items_per_thread),
                      binary_op);
        }

        // Writing to device memory
        T* d_output;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(typename decltype(output)::value_type))
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
            HIP_KERNEL_NAME(test_hip_warp_sort<
                items_per_thread, block_size, logical_warp_size, T
            >),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_output
        );

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_eq(output, expected);

        HIP_CHECK(hipFree(d_output));
    }

}

typed_test_def(RocprimWarpSortShuffleBasedTests, name_suffix, SortKeyInt)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // logical warp side for warp primitive, execution warp size is always rocprim::warp_size()
    using T = typename TestFixture::params::type;
    using pair = test_utils::custom_test_type<T>;

    using value_op_type = rocprim::less<T>;
    using eq_op_type    = rocprim::equal_to<T>;

    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    static constexpr size_t items_per_thread = TestFixture::params::items_per_thread;

    // The different warp sizes
    static constexpr size_t ws32 = size_t(ROCPRIM_WARP_SIZE_32);
    static constexpr size_t ws64 = size_t(ROCPRIM_WARP_SIZE_64);

    const unsigned int current_device_warp_size = rocprim::host_warp_size();
    static constexpr size_t block_size = std::max<size_t>(256U, logical_warp_size * 4);

    static constexpr unsigned int grid_size = 4;
    const size_t size = items_per_thread * block_size * grid_size;

    SCOPED_TRACE(testing::Message() << "with size = " << size);

    // Check if warp size is supported
    if( logical_warp_size > current_device_warp_size ||
        !rocprim::detail::is_power_of_two(logical_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %d.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

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
        for(size_t i = 0; i < expected.size() / logical_warp_size / items_per_thread; i++)
        {
            std::sort(expected.begin() + (i * logical_warp_size * items_per_thread),
                      expected.begin() + ((i + 1) * logical_warp_size * items_per_thread));
        }

        // Writing to device memory
        T* d_output_key;
        T* d_output_value;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_output_key, output_key.size() * sizeof(typename decltype(output_key)::value_type))
        );
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_output_value, output_value.size() * sizeof(typename decltype(output_value)::value_type))
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
            HIP_KERNEL_NAME(test_hip_sort_key_value_kernel<
                items_per_thread, block_size, logical_warp_size, T, T
            >),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_output_key, d_output_value
        );

        HIP_CHECK(hipGetLastError());
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

        test_utils::assert_eq(output_key, expected_key);
        test_utils::assert_eq(output_value, expected_value);

        HIP_CHECK(hipFree(d_output_key));
        HIP_CHECK(hipFree(d_output_value));
    }

}
