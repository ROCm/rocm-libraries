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

test_suite_type_def(suite_name, name_suffix)

typed_test_suite_def(HipcubBlockLoadStoreTests, name_suffix, load_store_params);

typed_test_def(HipcubBlockLoadStoreTests, name_suffix, LoadStoreClass)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Type                                             = typename TestFixture::params::type;
    constexpr size_t                      block_size       = TestFixture::params::block_size;
    constexpr hipcub::BlockLoadAlgorithm  load_method      = TestFixture::params::load_method;
    constexpr hipcub::BlockStoreAlgorithm store_method     = TestFixture::params::store_method;
    const size_t                          items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto                        items_per_block  = block_size * items_per_thread;
    const size_t                          size             = items_per_block * 113;
    const auto                            grid_size        = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value =
            seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100, seed_value);
        std::vector<Type> output(input.size(), test_utils::convert_to_device<Type>(0));

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), test_utils::convert_to_device<Type>(0));
        for(size_t i = 0; i < 113; i++)
        {
            size_t block_offset = i * items_per_block;
            for(size_t j = 0; j < items_per_block; j++)
            {
                expected[j + block_offset] = input[j + block_offset];
            }
        }

        // Preparing device
        Type * device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_input,
            input.size() * sizeof(typename decltype(input)::value_type)));
        Type * device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_output,
            output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(hipMemcpy(device_input,
                            input.data(),
                            input.size() * sizeof(typename decltype(input)::value_type),
                            hipMemcpyHostToDevice));

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                load_store_kernel<Type, load_method, store_method, block_size, items_per_thread>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_input,
            device_output);

        // Reading results from device
        HIP_CHECK(hipMemcpy(output.data(),
                            device_output,
                            output.size() * sizeof(typename decltype(output)::value_type),
                            hipMemcpyDeviceToHost));

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(output[i]),
                      test_utils::convert_to_native(expected[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

typed_test_def(HipcubBlockLoadStoreTests, name_suffix, LoadStoreClassValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Type                                             = typename TestFixture::params::type;
    constexpr size_t                      block_size       = TestFixture::params::block_size;
    constexpr hipcub::BlockLoadAlgorithm  load_method      = TestFixture::params::load_method;
    constexpr hipcub::BlockStoreAlgorithm store_method     = TestFixture::params::store_method;
    const size_t                          items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto                        items_per_block  = block_size * items_per_thread;
    const size_t                          size             = items_per_block * 113;
    const auto                            grid_size        = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value =
            seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const size_t valid = items_per_block - 32;
        // Generate data
        std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100, seed_value);
        std::vector<Type> output(input.size(), test_utils::convert_to_device<Type>(0));

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), test_utils::convert_to_device<Type>(0));
        for(size_t i = 0; i < 113; i++)
        {
            size_t block_offset = i * items_per_block;
            for(size_t j = 0; j < items_per_block; j++)
            {
                if(j < valid)
                {
                    expected[j + block_offset] = input[j + block_offset];
                }
            }
        }

        // Preparing device
        Type * device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_input,
            input.size() * sizeof(typename decltype(input)::value_type)));
        Type * device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_output,
            output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(hipMemcpy(device_input,
                            input.data(),
                            input.size() * sizeof(typename decltype(input)::value_type),
                            hipMemcpyHostToDevice));

        // Have to initialize output for unvalid data to make sure they are not changed
        HIP_CHECK(hipMemcpy(device_output,
                            output.data(),
                            output.size() * sizeof(typename decltype(output)::value_type),
                            hipMemcpyHostToDevice));

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(load_store_valid_kernel<Type,
                                                                   load_method,
                                                                   store_method,
                                                                   block_size,
                                                                   items_per_thread>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           device_input,
                           device_output,
                           valid);

        // Reading results from device
        HIP_CHECK(hipMemcpy(output.data(),
                            device_output,
                            output.size() * sizeof(typename decltype(output)::value_type),
                            hipMemcpyDeviceToHost));

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(output[i]),
                      test_utils::convert_to_native(expected[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

typed_test_def(HipcubBlockLoadStoreTests, name_suffix, LoadStoreClassDefault)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Type                                             = typename TestFixture::params::type;
    constexpr size_t                      block_size       = TestFixture::params::block_size;
    constexpr hipcub::BlockLoadAlgorithm  load_method      = TestFixture::params::load_method;
    constexpr hipcub::BlockStoreAlgorithm store_method     = TestFixture::params::store_method;
    const size_t                          items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto                        items_per_block  = block_size * items_per_thread;
    const size_t                          size             = items_per_block * 113;
    const auto                            grid_size        = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value =
            seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const size_t valid    = items_per_thread + 1;
        Type         _default = test_utils::convert_to_device<Type>(-1);
        // Generate data
        std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100, seed_value);
        std::vector<Type> output(input.size(), test_utils::convert_to_device<Type>(0));

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), _default);
        for(size_t i = 0; i < 113; i++)
        {
            size_t block_offset = i * items_per_block;
            for(size_t j = 0; j < items_per_block; j++)
            {
                if(j < valid)
                {
                    expected[j + block_offset] = input[j + block_offset];
                }
            }
        }

        // Preparing device
        Type * device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_input,
            input.size() * sizeof(typename decltype(input)::value_type)));
        Type * device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_output,
            output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(hipMemcpy(device_input,
                            input.data(),
                            input.size() * sizeof(typename decltype(input)::value_type),
                            hipMemcpyHostToDevice));

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(load_store_valid_default_kernel<Type,
                                                                           load_method,
                                                                           store_method,
                                                                           block_size,
                                                                           items_per_thread>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           device_input,
                           device_output,
                           valid,
                           _default);

        // Reading results from device
        HIP_CHECK(hipMemcpy(output.data(),
                            device_output,
                            output.size() * sizeof(typename decltype(output)::value_type),
                            hipMemcpyDeviceToHost));

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(output[i]),
                      test_utils::convert_to_native(expected[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

typed_test_def(HipcubBlockLoadStoreTests, name_suffix, LoadStoreDiscardIterator)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Type                                             = typename TestFixture::params::type;
    constexpr size_t                      block_size       = TestFixture::params::block_size;
    constexpr hipcub::BlockLoadAlgorithm  load_method      = TestFixture::params::load_method;
    constexpr hipcub::BlockStoreAlgorithm store_method     = TestFixture::params::store_method;
    const size_t                          items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto                        items_per_block  = block_size * items_per_thread;
    const auto                            grid_size        = 113;
    const size_t                          size             = items_per_block * grid_size;

    constexpr double fraction_valid = 0.8f;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value =
            seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const size_t unguarded_elements = size;
        const size_t guarded_elements   = size_t(fraction_valid * double(unguarded_elements));

        // Generate data
        std::vector<Type> input =
            test_utils::get_random_data<Type>(unguarded_elements, -100, 100, seed_value);
        std::vector<Type> unguarded(unguarded_elements, test_utils::convert_to_device<Type>(0));
        std::vector<Type> guarded(guarded_elements, test_utils::convert_to_device<Type>(0));

        // Calculate expected results on host
        std::vector<Type> unguarded_expected(unguarded_elements);
        std::vector<Type> guarded_expected(guarded_elements);
        for(size_t i = 0; i < unguarded_elements; i++)
        {
            unguarded_expected[i] = input[i];
        }

        for(size_t i = 0; i < guarded_elements; i++)
        {
            guarded_expected[i] = input[i];
        }

        // Preparing device
        Type * device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_input,
            input.size() * sizeof(typename decltype(input)::value_type)));
        Type * device_guarded_elements;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_guarded_elements,
            guarded_expected.size() * sizeof(typename decltype(unguarded)::value_type)));
        Type * device_unguarded_elements;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_unguarded_elements,
            unguarded_expected.size() * sizeof(typename decltype(guarded)::value_type)));

        HIP_CHECK(hipMemcpy(device_input,
                            input.data(),
                            input.size() * sizeof(typename decltype(input)::value_type),
                            hipMemcpyHostToDevice));

        // Test with discard output iterator
        // using OffsetT = typename std::iterator_traits<Type>::difference_type;
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
        // TODO: Here in block load an store, it's not possible to use rocprim::discard_iterator
        hipcub::DiscardOutputIterator<size_t> discard_itr;

        // Running kernel
        load_store_guarded_kernel<Type *,
                                  hipcub::DiscardOutputIterator<size_t>,
                                  load_method,
                                  store_method,
                                  block_size,
                                  items_per_thread>
            <<<dim3(grid_size), dim3(block_size)>>>(device_input,
                                                    discard_itr,
                                                    discard_itr,
                                                    guarded_elements);
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
        // Running kernel
        load_store_guarded_kernel<Type *,
                                  Type *,
                                  load_method,
                                  store_method,
                                  block_size,
                                  items_per_thread>
            <<<dim3(grid_size), dim3(block_size)>>>(device_input,
                                                    device_unguarded_elements,
                                                    device_guarded_elements,
                                                    guarded_elements);

        // Reading results from device
        HIP_CHECK(hipMemcpy(unguarded.data(),
                            device_unguarded_elements,
                            unguarded.size() * sizeof(typename decltype(unguarded)::value_type),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(hipMemcpy(guarded.data(),
                            device_guarded_elements,
                            guarded.size() * sizeof(typename decltype(guarded)::value_type),
                            hipMemcpyDeviceToHost));

        // Validating results
        for(size_t i = 0; i < guarded_expected.size(); i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(guarded[i]),
                      test_utils::convert_to_native(guarded_expected[i]))
                << "where index = " << i;
        }
        for(size_t i = 0; i < unguarded_expected.size(); i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(unguarded[i]),
                      test_utils::convert_to_native(unguarded_expected[i]))
                << "where index = " << i;
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_guarded_elements));
        HIP_CHECK(hipFree(device_unguarded_elements));
    }
}
