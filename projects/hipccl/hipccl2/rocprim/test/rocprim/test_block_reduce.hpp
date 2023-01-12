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

block_reduce_test_suite_type_def(suite_name_single, name_suffix)
block_reduce_test_suite_type_def(suite_name_array, name_suffix)

typed_test_suite_def(suite_name_single, name_suffix, block_params);
typed_test_suite_def(suite_name_array, name_suffix, block_params);

typed_test_def(suite_name_single, name_suffix, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using binary_op_type = rocprim::plus<T>;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;
    using cast_type = typename test_utils::select_plus_operator_host<T>::cast_type;

    constexpr size_t block_size = TestFixture::block_size;

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
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, T(2), T(50), seed_value);
        std::vector<T> output_reductions(grid_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), T(0));
        for(size_t i = 0; i < grid_size; i++)
        {
            acc_type value(0);
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value = binary_op_host(value, output[idx]);
            }
            expected_reductions[i] = static_cast<cast_type>(value);
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output_reductions, output_reductions.size() * sizeof(T)));

        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::using_warp_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, false
        );
        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::raking_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, false
        );
        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::raking_reduce_commutative_only, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, false
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

typed_test_def(suite_name_single, name_suffix, ReduceMultiplies)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                     = typename TestFixture::input_type;
    using binary_op_type        = rocprim::multiplies<T>;
    constexpr size_t block_size = TestFixture::block_size;
    using cast_type = typename test_utils::select_plus_operator_host<T>::cast_type;
    
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size      = block_size * 58;
    const size_t grid_size = size / block_size;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, T(0.95), T(1.05), seed_value);
        std::vector<T> output_reductions(grid_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), T(0));
        for(size_t i = 0; i < grid_size; i++)
        {
            double value = 1;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value *= static_cast<double>(output[idx]);
            }
            expected_reductions[i] = static_cast<cast_type>(value);
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output_reductions,
                                                     output_reductions.size() * sizeof(T)));

        static_run_algo<T,
                        block_size,
                        rocprim::block_reduce_algorithm::using_warp_reduce,
                        binary_op_type>::run(output,
                                             output_reductions,
                                             expected_reductions,
                                             device_output,
                                             device_output_reductions,
                                             grid_size,
                                             false);
        static_run_algo<T,
                        block_size,
                        rocprim::block_reduce_algorithm::raking_reduce,
                        binary_op_type>::run(output,
                                             output_reductions,
                                             expected_reductions,
                                             device_output,
                                             device_output_reductions,
                                             grid_size,
                                             false);

        static_run_algo<T,
                        block_size,
                        rocprim::block_reduce_algorithm::raking_reduce_commutative_only,
                        binary_op_type>::run(output,
                                             output_reductions,
                                             expected_reductions,
                                             device_output,
                                             device_output_reductions,
                                             grid_size,
                                             false);

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

typed_test_def(suite_name_single, name_suffix, ReduceMultipliesExact)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                     = typename TestFixture::input_type;
    using binary_op_type        = rocprim::multiplies<T>;
    constexpr size_t block_size = TestFixture::block_size;

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
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output(size, T(1));
        auto two_places = test_utils::get_random_data<unsigned int>(size/32, 0, size-1, seed_value);
        for(auto i : two_places)
        {
            output[i] = T(2);
        }
        std::vector<T> output_reductions(grid_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(grid_size, T(0));
        binary_op_type binary_op;
        for(size_t i = 0; i < grid_size; i++)
        {
            T value = T(1);
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value = binary_op(value, output[idx]);
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output_reductions, output_reductions.size() * sizeof(T)));

        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::using_warp_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, true
        );
        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::raking_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, true
        );

        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::raking_reduce_commutative_only, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, true
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

typed_test_def(suite_name_single, name_suffix, ReduceValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using binary_op_type = rocprim::plus<T>;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;
    using cast_type = typename test_utils::select_plus_operator_host<T>::cast_type;

    constexpr size_t block_size = TestFixture::block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const size_t valid_items = test_utils::get_random_value<size_t>(block_size - 10, block_size, seed_value);

        // Given block size not supported
        if(block_size > test_utils::get_max_block_size())
        {
            return;
        }

        const size_t size = block_size * 58;
        const size_t grid_size = size / block_size;

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output_reductions(grid_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(grid_size, T(0));
        for(size_t i = 0; i < grid_size; i++)
        {
            acc_type value(0);
            for(size_t j = 0; j < valid_items; j++)
            {
                auto idx = i * block_size + j;
                value = binary_op_host(value, output[idx]);
            }
            expected_reductions[i] = static_cast<cast_type>(value);
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output_reductions, output_reductions.size() * sizeof(T)));

        static_run_valid<T, block_size, rocprim::block_reduce_algorithm::using_warp_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, valid_items, grid_size
        );
        static_run_valid<T, block_size, rocprim::block_reduce_algorithm::raking_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, valid_items, grid_size
        );
        static_run_valid<T, block_size, rocprim::block_reduce_algorithm::raking_reduce_commutative_only, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, valid_items, grid_size
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

typed_test_def(suite_name_array, name_suffix, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, block_size, rocprim::block_reduce_algorithm::using_warp_reduce>::run();
    static_for_input_array<0, 2, T, block_size, rocprim::block_reduce_algorithm::raking_reduce>::run();
    static_for_input_array<0, 2, T, block_size, rocprim::block_reduce_algorithm::raking_reduce_commutative_only>::run();

}
