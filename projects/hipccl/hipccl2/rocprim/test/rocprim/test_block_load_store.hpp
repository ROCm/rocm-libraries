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

#include "rocprim/block/block_load.hpp"
#include "rocprim/block/block_store.hpp"
#include <rocprim/test_utils.hpp>
test_suite_type_def(suite_name, name_suffix)

typed_test_suite_def(suite_name, name_suffix, warp_params);

typed_test_def(suite_name, name_suffix, LoadStoreClass)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Type = typename TestFixture::params::type;
    static constexpr size_t block_size = TestFixture::params::block_size;
    static constexpr rocprim::block_load_method load_method = TestFixture::params::load_method;
    static constexpr rocprim::block_store_method store_method = TestFixture::params::store_method;
    static constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    static constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_block;

    if(load_method == rocprim::block_load_method::block_load_warp_transpose
       || store_method == rocprim::block_store_method::block_store_warp_transpose)
    {
        unsigned int host_warp_size;
        HIP_CHECK(::rocprim::host_warp_size(device_id, host_warp_size));
        if(block_size % host_warp_size != 0)
        {
            GTEST_SKIP() << "Cannot run test of block size " << block_size
                         << " on a device with warp size " << host_warp_size;
        }
    }

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<Type> input
            = test_utils::get_random_data_wrapped<Type>(size, -100, 100, seed_value);

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), (Type)0);
        for (size_t i = 0; i < 113; i++)
        {
            size_t block_offset = i * items_per_block;
            for (size_t j = 0; j < items_per_block; j++)
            {
                expected[j + block_offset] = input[j + block_offset];
            }
        }

        // Preparing device
        common::device_ptr<Type> device_input(input);
        common::device_ptr<Type> device_output(size);

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                load_store_kernel<Type, load_method, store_method, block_size, items_per_thread>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_input.get(),
            device_output.get());
        HIP_CHECK(hipGetLastError());

        // Reading results from device
        const auto output = device_output.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }

}

typed_test_def(suite_name, name_suffix, LoadStoreClassValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Type = typename TestFixture::params::type;
    static constexpr size_t block_size = TestFixture::params::block_size;
    static constexpr rocprim::block_load_method load_method = TestFixture::params::load_method;
    static constexpr rocprim::block_store_method store_method = TestFixture::params::store_method;
    static constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    static constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    if(load_method == rocprim::block_load_method::block_load_warp_transpose
       || store_method == rocprim::block_store_method::block_store_warp_transpose)
    {
        unsigned int host_warp_size;
        HIP_CHECK(::rocprim::host_warp_size(device_id, host_warp_size));
        if(block_size % host_warp_size != 0)
        {
            GTEST_SKIP() << "Cannot run test of block size " << block_size
                         << " on a device with warp size " << host_warp_size;
        }
    }

    const size_t valid = items_per_block - 32;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<Type> input
            = test_utils::get_random_data_wrapped<Type>(size, -100, 100, seed_value);
        std::vector<Type> output(input.size(), (Type)0);

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), (Type)0);
        for (size_t i = 0; i < 113; i++)
        {
            size_t block_offset = i * items_per_block;
            for (size_t j = 0; j < items_per_block; j++)
            {
                if (j < valid)
                {
                    expected[j + block_offset] = input[j + block_offset];
                }
            }
        }

        // Preparing device
        common::device_ptr<Type> device_input(input);
        // Have to initialize output for unvalid data to make sure they are not changed
        common::device_ptr<Type> device_output(output);

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
                           device_input.get(),
                           device_output.get(),
                           valid);
        HIP_CHECK(hipGetLastError());

        // Reading results from device
        output = device_output.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }

}

typed_test_def(suite_name, name_suffix, LoadStoreClassDefault)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Type = typename TestFixture::params::type;
    static constexpr size_t block_size = TestFixture::params::block_size;
    static constexpr rocprim::block_load_method load_method = TestFixture::params::load_method;
    static constexpr rocprim::block_store_method store_method = TestFixture::params::store_method;
    static constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    static constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    if(load_method == rocprim::block_load_method::block_load_warp_transpose
       || store_method == rocprim::block_store_method::block_store_warp_transpose)
    {
        unsigned int host_warp_size;
        HIP_CHECK(::rocprim::host_warp_size(device_id, host_warp_size));
        if(block_size % host_warp_size != 0)
        {
            GTEST_SKIP() << "Cannot run test of block size " << block_size
                         << " on a device with warp size " << host_warp_size;
        }
    }

    const size_t valid = items_per_thread + 1;
    Type _default = (Type)-1;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<Type> input
            = test_utils::get_random_data_wrapped<Type>(size, -100, 100, seed_value);

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), _default);
        for (size_t i = 0; i < 113; i++)
        {
            size_t block_offset = i * items_per_block;
            for (size_t j = 0; j < items_per_block; j++)
            {
                if (j < valid)
                {
                    expected[j + block_offset] = input[j + block_offset];
                }
            }
        }

        // Preparing device
        common::device_ptr<Type> device_input(input);
        common::device_ptr<Type> device_output(size);

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
                           device_input.get(),
                           device_output.get(),
                           valid,
                           _default);
        HIP_CHECK(hipGetLastError());

        // Reading results from device
        const auto output = device_output.load();

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }
}
