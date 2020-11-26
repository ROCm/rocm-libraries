/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "common_test_header.hpp"

// hipcub API
#include "hipcub/block/block_load.hpp"
#include "hipcub/block/block_store.hpp"
#include "hipcub/iterator/discard_output_iterator.hpp"

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


template <bool IF, typename ThenType, typename ElseType>
struct If
{
    /// Conditional type result
    typedef ThenType Type;      // true
};


template <
    typename            InputIteratorT,
    typename            OutputIteratorT,
    hipcub::BlockLoadAlgorithm LoadMethod,
    hipcub::BlockStoreAlgorithm StoreMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread>
__launch_bounds__ (BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
__global__ void load_store_guarded_kernel(
    InputIteratorT    d_in,
    OutputIteratorT   d_out_unguarded,
    OutputIteratorT   d_out_guarded,
    int               num_items)
{
    enum
    {
        TileSize = BlockSize * ItemsPerThread
    };

    // The input value type
    typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

    // The output value type
    typedef typename If<(std::is_same<typename std::iterator_traits<OutputIteratorT>::value_type, void>::value),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    // Threadblock load/store abstraction types
    typedef hipcub::BlockLoad<InputT, BlockSize, ItemsPerThread, LoadMethod> BlockLoad;
    typedef hipcub::BlockStore<OutputT, BlockSize, ItemsPerThread, StoreMethod> BlockStore;

    // Shared memory type for this thread block
    union TempStorage
    {
        typename BlockLoad::TempStorage     load;
        typename BlockStore::TempStorage    store;
    };

    // Allocate temp storage in shared memory
    __shared__ TempStorage temp_storage;

    // Threadblock work bounds
    int block_offset = blockIdx.x * TileSize;
    int guarded_elements = std::max(num_items - block_offset, 0);

    // Tile of items
    OutputT data[ItemsPerThread];

    // Load data
    BlockLoad(temp_storage.load).Load(d_in + block_offset, data);

    __syncthreads();

    // Store data
    BlockStore(temp_storage.store).Store(d_out_unguarded + block_offset, data);

    __syncthreads();

    // reset data
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; ++item)
        data[item] = OutputT();

    __syncthreads();

    // Load data
    BlockLoad(temp_storage.load).Load(d_in + block_offset, data, guarded_elements);

    __syncthreads();

    // Store data
    BlockStore(temp_storage.store).Store(d_out_guarded + block_offset, data, guarded_elements);
}

TYPED_TEST(HipcubBlockLoadStoreClassTests, LoadStoreDiscardIterator)
{
    using Type = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr hipcub::BlockLoadAlgorithm load_method = TestFixture::params::load_method;
    constexpr hipcub::BlockStoreAlgorithm store_method = TestFixture::params::store_method;
    const size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const auto grid_size = 113;
    const size_t size = items_per_block * grid_size;

    constexpr double fraction_valid = 0.8f;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const size_t unguarded_elements = size;
        const size_t guarded_elements   = size_t(fraction_valid * double(unguarded_elements));

        // Generate data
        std::vector<Type> input = test_utils::get_random_data<Type>(unguarded_elements, -100, 100, seed_value);
        std::vector<Type> unguarded(unguarded_elements, 0);
        std::vector<Type> guarded(guarded_elements, 0);

        // Calculate expected results on host
        std::vector<Type> unguarded_expected(unguarded_elements);
        std::vector<Type> guarded_expected(guarded_elements);
        for (size_t i = 0; i < unguarded_elements; i++)
        {
            unguarded_expected[i] = input[i];
        }

        for (size_t i = 0; i < guarded_elements; i++)
        {
            guarded_expected[i] = input[i];
        }

        // Preparing device
        Type* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        Type* device_guarded_elements;
        HIP_CHECK(hipMalloc(&device_guarded_elements, guarded_expected.size() * sizeof(typename decltype(unguarded)::value_type)));
        Type* device_unguarded_elements;
        HIP_CHECK(hipMalloc(&device_unguarded_elements, unguarded_expected.size() * sizeof(typename decltype(guarded)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(typename decltype(input)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Test with discard output iterator
        //typedef typename std::iterator_traits<Type>::difference_type OffsetT;
        hipcub::DiscardOutputIterator<size_t> discard_itr;

        // Running kernel
        load_store_guarded_kernel<Type*, hipcub::DiscardOutputIterator<size_t>, load_method, store_method, block_size, items_per_thread>
            <<<dim3(grid_size), dim3(block_size)>>>(
            device_input, discard_itr, discard_itr, guarded_elements
        );

        // Running kernel
        load_store_guarded_kernel<Type*, Type*, load_method, store_method, block_size, items_per_thread>
            <<<dim3(grid_size), dim3(block_size)>>>(
            device_input, device_unguarded_elements, device_guarded_elements, guarded_elements
        );

        // Reading results from device
        HIP_CHECK(
            hipMemcpy(
                unguarded.data(), device_unguarded_elements,
                unguarded.size() * sizeof(typename decltype(unguarded)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                guarded.data(), device_guarded_elements,
                guarded.size() * sizeof(typename decltype(guarded)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < guarded_expected.size(); i++)
        {
            ASSERT_EQ(guarded[i], guarded_expected[i]) << "where index = " << i;
        }
        for(size_t i = 0; i < unguarded_expected.size(); i++)
        {
            ASSERT_EQ(unguarded[i], unguarded_expected[i]) << "where index = " << i;
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_guarded_elements));
        HIP_CHECK(hipFree(device_unguarded_elements));
    }
}
