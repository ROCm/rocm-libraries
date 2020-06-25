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

// hipcub API
#include "hipcub/block/block_load.hpp"
#include "hipcub/block/block_store.hpp"

template<
    class Type,
    hipcub::BlockLoadAlgorithm Load,
    hipcub::BlockStoreAlgorithm Store,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct class_params
{
    using type = Type;
    static constexpr hipcub::BlockLoadAlgorithm load_method = Load;
    static constexpr hipcub::BlockStoreAlgorithm store_method = Store;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

template<class ClassParams>
class HipcubBlockLoadStoreClassTests : public ::testing::Test {
public:
    using params = ClassParams;
};

typedef ::testing::Types<
    // BLOCK_LOAD_DIRECT
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 64U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 64U, 4>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 256U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 256U, 4>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 512U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 512U, 4>,

    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 64U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 64U, 4>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 256U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 256U, 4>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 512U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 512U, 4>,

    class_params<test_utils::custom_test_type<int>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 64U, 4>,
    class_params<test_utils::custom_test_type<double>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, 256U, 4>,

    // BLOCK_LOAD_VECTORIZE
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 64U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 64U, 4>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 256U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 256U, 4>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 512U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 512U, 4>,

    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 64U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 64U, 4>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 256U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 256U, 4>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 512U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 512U, 4>,

    class_params<test_utils::custom_test_type<int>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 64U, 4>,
    class_params<test_utils::custom_test_type<double>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, 256U, 4>,

    // BLOCK_LOAD_TRANSPOSE
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 64U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 64U, 4>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 256U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 256U, 4>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 512U, 1>,
    class_params<int, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 512U, 4>,

    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 64U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 64U, 4>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 256U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 256U, 4>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 512U, 1>,
    class_params<double, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 512U, 4>,

    class_params<test_utils::custom_test_type<int>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 64U, 4>,
    class_params<test_utils::custom_test_type<double>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, hipcub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE,
                 hipcub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, 256U, 4>

> ClassParams;

TYPED_TEST_CASE(HipcubBlockLoadStoreClassTests, ClassParams);

template<
    class Type,
    hipcub::BlockLoadAlgorithm LoadMethod,
    hipcub::BlockStoreAlgorithm StoreMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void load_store_kernel(Type* device_input, Type* device_output)
{
    Type items[ItemsPerThread];
    unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread;
    hipcub::BlockLoad<Type, BlockSize, ItemsPerThread, LoadMethod> load;
    hipcub::BlockStore<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.Load(device_input + offset, items);
    store.Store(device_output + offset, items);
}

TYPED_TEST(HipcubBlockLoadStoreClassTests, LoadStoreClass)
{
    using Type = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr hipcub::BlockLoadAlgorithm load_method = TestFixture::params::load_method;
    constexpr hipcub::BlockStoreAlgorithm store_method = TestFixture::params::store_method;
    const size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100, seed_value);
        std::vector<Type> output(input.size(), 0);

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), 0);
        for (size_t i = 0; i < 113; i++)
        {
            size_t block_offset = i * items_per_block;
            for (size_t j = 0; j < items_per_block; j++)
            {
                expected[j + block_offset] = input[j + block_offset];
            }
        }

        // Preparing device
        Type* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        Type* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(typename decltype(input)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                load_store_kernel<
                    Type, load_method, store_method,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output
        );

        // Reading results from device
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<
    class Type,
    hipcub::BlockLoadAlgorithm LoadMethod,
    hipcub::BlockStoreAlgorithm StoreMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void load_store_valid_kernel(Type* device_input, Type* device_output, size_t valid)
{
    Type items[ItemsPerThread];
    unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread;
    hipcub::BlockLoad<Type, BlockSize, ItemsPerThread, LoadMethod> load;
    hipcub::BlockStore<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.Load(device_input + offset, items, valid);
    store.Store(device_output + offset, items, valid);
}

TYPED_TEST(HipcubBlockLoadStoreClassTests, LoadStoreClassValid)
{
    using Type = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr hipcub::BlockLoadAlgorithm load_method = TestFixture::params::load_method;
    constexpr hipcub::BlockStoreAlgorithm store_method = TestFixture::params::store_method;
    const size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const size_t valid = items_per_block - 32;
        // Generate data
        std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100, seed_value);
        std::vector<Type> output(input.size(), 0);

        // Calculate expected results on host
        std::vector<Type> expected(input.size(), 0);
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
        Type* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        Type* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(typename decltype(input)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Have to initialize output for unvalid data to make sure they are not changed
        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                load_store_valid_kernel<
                    Type, load_method, store_method,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output, valid
        );

        // Reading results from device
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<
    class Type,
    hipcub::BlockLoadAlgorithm LoadMethod,
    hipcub::BlockStoreAlgorithm StoreMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void load_store_valid_default_kernel(Type* device_input, Type* device_output, size_t valid, int _default)
{
    Type items[ItemsPerThread];
    unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread;
    hipcub::BlockLoad<Type, BlockSize, ItemsPerThread, LoadMethod> load;
    hipcub::BlockStore<Type, BlockSize, ItemsPerThread, StoreMethod> store;
    load.Load(device_input + offset, items, valid, _default);
    store.Store(device_output + offset, items);
}

TYPED_TEST(HipcubBlockLoadStoreClassTests, LoadStoreClassDefault)
{
    using Type = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr hipcub::BlockLoadAlgorithm load_method = TestFixture::params::load_method;
    constexpr hipcub::BlockStoreAlgorithm store_method = TestFixture::params::store_method;
    const size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_block;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const size_t valid = items_per_thread + 1;
        int _default = -1;
        // Generate data
        std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100, seed_value);
        std::vector<Type> output(input.size(), 0);

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
        Type* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        Type* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(typename decltype(input)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                load_store_valid_default_kernel<
                    Type, load_method, store_method,
                    block_size, items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output, valid, _default
        );

        // Reading results from device
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}
