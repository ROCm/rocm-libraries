// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

block_reduce_test_suite_type_def(suite_name_single, name_suffix)
block_reduce_test_suite_type_def(suite_name_array, name_suffix)

typed_test_suite_def(suite_name_single, name_suffix, block_params);
typed_test_suite_def(suite_name_array, name_suffix, block_params);

typed_test_def(suite_name_single, name_suffix, InclusiveScan)
{
    using T = typename TestFixture::input_type;
 // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;
    constexpr size_t block_size = TestFixture::block_size;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output2 = output;

        // Calculate expected results on host
        std::vector<T> expected(output.size(), (T)0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            acc_type accumulator(0);
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                accumulator = binary_op_host(output[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::using_warp_scan, 0>::run(
            output, output, expected, expected,
            device_output, NULL, T(0), grid_size
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::reduce_then_scan, 0>::run(
            output2, output2, expected, expected,
            device_output, NULL, T(0), grid_size
        );

        HIP_CHECK(hipFree(device_output));
    }

}

typed_test_def(suite_name_single, name_suffix, InclusiveScanReduce)
{
    using T = typename TestFixture::input_type;
 // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;
    constexpr size_t block_size = TestFixture::block_size;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output2 = output;
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), (T)0);
        std::vector<T> expected_reductions(output_reductions.size(), (T)0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            acc_type accumulator(0);
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                accumulator = binary_op_host(output[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
            expected_reductions[i] = expected[(i+1) * block_size - 1];
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_output_reductions,
                output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::using_warp_scan, 1>::run(
            output, output_reductions, expected, expected_reductions,
            device_output, device_output_reductions, T(0), grid_size
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::reduce_then_scan, 1>::run(
            output2, output_reductions, expected, expected_reductions,
            device_output, device_output_reductions, T(0), grid_size
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

typed_test_def(suite_name_single, name_suffix, InclusiveScanPrefixCallback)
{
    using T = typename TestFixture::input_type;
     // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    constexpr size_t block_size = TestFixture::block_size;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output2 = output;
        std::vector<T> output_block_prefixes(size / block_size);
        T block_prefix = test_utils::get_random_value<T>(0, 5, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), (T)0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), (T)0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            acc_type accumulator = block_prefix;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                accumulator = binary_op_host(output[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
            expected_block_prefixes[i] = expected[(i+1) * block_size - 1];
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_output_bp,
                output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::using_warp_scan, 2>::run(
            output, output_block_prefixes, expected, expected_block_prefixes,
            device_output, device_output_bp, block_prefix, grid_size
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::reduce_then_scan, 2>::run(
            output2, output_block_prefixes, expected, expected_block_prefixes,
            device_output, device_output_bp, block_prefix, grid_size
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_bp));
    }

}

typed_test_def(suite_name_single, name_suffix, ExclusiveScan)
{
    using T = typename TestFixture::input_type;
     // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    constexpr size_t block_size = TestFixture::block_size;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output2 = output;
        const T init = test_utils::get_random_value<T>(0, 5, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), (T)0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            acc_type accumulator(init);
            expected[i * block_size] = init;
            for(size_t j = 1; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                accumulator = binary_op_host(output[idx-1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::using_warp_scan, 3>::run(
            output, output, expected, expected,
            device_output, NULL, init, grid_size
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::reduce_then_scan, 3>::run(
            output2, output2, expected, expected,
            device_output, NULL, init, grid_size
        );

        HIP_CHECK(hipFree(device_output));
    }

}

typed_test_def(suite_name_single, name_suffix, ExclusiveScanReduce)
{
    using T = typename TestFixture::input_type;
     // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    constexpr size_t block_size = TestFixture::block_size;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output2 = output;
        const T init = test_utils::get_random_value<T>(0, 5, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), (T)0);
        std::vector<T> expected_reductions(output_reductions.size(), (T)0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            acc_type accumulator(init);
            expected[i * block_size] = init;
            for(size_t j = 1; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                accumulator = binary_op_host(output[idx-1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
            acc_type accumulator_reductions(0);
            expected_reductions[i] = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                accumulator_reductions = binary_op_host(accumulator_reductions, output[idx]);
                expected_reductions[i] = static_cast<T>(accumulator_reductions);
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_output_reductions,
                output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::using_warp_scan, 4>::run(
            output, output_reductions, expected, expected_reductions,
            device_output, device_output_reductions, init, grid_size
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::reduce_then_scan, 4>::run(
            output2, output_reductions, expected, expected_reductions,
            device_output, device_output_reductions, init, grid_size
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

typed_test_def(suite_name_single, name_suffix, ExclusiveScanPrefixCallback)
{
    using T = typename TestFixture::input_type;
     // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    constexpr size_t block_size = TestFixture::block_size;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output2 = output;
        std::vector<T> output_block_prefixes(size / block_size);
        T block_prefix = test_utils::get_random_value<T>(0, 5, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), (T)0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), (T)0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            acc_type accumulator = block_prefix;
            expected[i * block_size] = block_prefix;
            for(size_t j = 1; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                accumulator = binary_op_host(output[idx-1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }

            acc_type accumulator_block_prefixes = block_prefix;
            expected_block_prefixes[i] = block_prefix;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                accumulator_block_prefixes = binary_op_host(output[idx], accumulator_block_prefixes);
                expected_block_prefixes[i] = static_cast<T>(accumulator_block_prefixes);
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_output_bp,
                output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::using_warp_scan, 5>::run(
            output, output_block_prefixes, expected, expected_block_prefixes,
            device_output, device_output_bp, block_prefix, grid_size
        );

        static_run_algo<T, block_size, rocprim::block_scan_algorithm::reduce_then_scan, 5>::run(
            output2, output_block_prefixes, expected, expected_block_prefixes,
            device_output, device_output_bp, block_prefix, grid_size
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_bp));
    }

}

typed_test_def(suite_name_array, name_suffix, InclusiveScan)
{
    using T = typename TestFixture::input_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 0, block_size>::run();
}

typed_test_def(suite_name_array, name_suffix, InclusiveScanReduce)
{
    using T = typename TestFixture::input_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 1, block_size>::run();
}

typed_test_def(suite_name_array, name_suffix, InclusiveScanPrefixCallback)
{
    using T = typename TestFixture::input_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 2, block_size>::run();
}

typed_test_def(suite_name_array, name_suffix, ExclusiveScan)
{
    using T = typename TestFixture::input_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 3, block_size>::run();
}

typed_test_def(suite_name_array, name_suffix, ExclusiveScanReduce)
{
    using T = typename TestFixture::input_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 4, block_size>::run();
}

typed_test_def(suite_name_array, name_suffix, ExclusiveScanPrefixCallback)
{
    using T = typename TestFixture::input_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, 5, block_size>::run();
}
