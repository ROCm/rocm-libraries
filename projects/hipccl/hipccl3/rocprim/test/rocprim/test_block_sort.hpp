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

block_sort_test_suite_type_def(suite_name, name_suffix)

typed_test_suite_def(suite_name, name_suffix, block_params);

typed_test_def(suite_name, name_suffix, SortKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::key_type;
    using binary_op_type = typename test_utils::select_less_operator<key_type>::type;
    static constexpr size_t block_size = TestFixture::block_size;
    const size_t size = block_size * 1134;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> output = test_utils::get_random_data<key_type>(size, -100, 100, seed_value);

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
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_key_output, output.size() * sizeof(key_type)));

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

}

typed_test_def(suite_name, name_suffix, SortKeyValue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using value_op_type = typename test_utils::select_less_operator<value_type>::type;
    using eq_op_type = typename test_utils::select_equal_to_operator<key_type>::type;;
    static constexpr size_t block_size = TestFixture::block_size;
    static constexpr size_t size = block_size * 1134;
    static constexpr size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> output_key = test_utils::get_random_data<key_type>(size, 0, 100, seed_value);
        std::vector<value_type> output_value = test_utils::get_random_data<value_type>(size, -100, 100, seed_value);

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
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_key_output, output_key.size() * sizeof(key_type)));
        value_type * device_value_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_value_output, output_value.size() * sizeof(value_type)));

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

        HIP_CHECK(hipFree(device_value_output));
        HIP_CHECK(hipFree(device_key_output));
    }

}

typed_test_def(suite_name, name_suffix, CustomSortKeyValue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using value_op_type = typename test_utils::select_less_operator<value_type>::type;
    using eq_op_type = typename test_utils::select_equal_to_operator<key_type>::type;;
    static constexpr size_t block_size = TestFixture::block_size;
    static constexpr size_t size = block_size * 1134;
    static constexpr size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> output_key = test_utils::get_random_data<key_type>(size, 0, 100, seed_value);
        std::vector<value_type> output_value = test_utils::get_random_data<value_type>(size, -100, 100, seed_value);

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
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_key_output, output_key.size() * sizeof(key_type)));
        value_type * device_value_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_value_output, output_value.size() * sizeof(value_type)));

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

        HIP_CHECK(hipFree(device_value_output));
        HIP_CHECK(hipFree(device_key_output));
    }

}

typed_test_def(suite_name, name_suffix, SortKeysMultipleItemsPerThread)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::key_type;
    using binary_op_type = typename std::conditional<std::is_same<key_type, rocprim::half>::value, test_utils::half_less, rocprim::less<key_type>>::type;
    static constexpr size_t block_size = TestFixture::block_size;
    static constexpr size_t items_per_thread = 4;
    const size_t size = block_size * items_per_thread * 1134;
    const size_t grid_size = size / ( block_size * items_per_thread);

    // Only power of two items_per_threads are supported
    // items_per_thread is only supported if blocksize is a power of two
    if(!is_power_of_two(items_per_thread) || !is_power_of_two(block_size))
        GTEST_SKIP();

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> output = test_utils::get_random_data<key_type>(size, -100, 100, seed_value);

        // Calculate expected results on host
        std::vector<key_type> expected(output);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / block_size / items_per_thread; i++)
        {
            std::sort(
                expected.begin() + (i * block_size * items_per_thread),
                expected.begin() + ((i + 1) * block_size * items_per_thread),
                binary_op
            );
        }

        // Preparing device
        key_type * device_key_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_key_output, output.size() * sizeof(key_type)));

        HIP_CHECK(
            hipMemcpy(
                device_key_output, output.data(),
                output.size() * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(sort_keys_kernel<block_size, items_per_thread, key_type>),
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
}
