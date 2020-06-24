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
#include "hipcub/block/block_reduce.hpp"
#include "hipcub/thread/thread_operators.hpp"

// Params for tests
template<
    class T,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    hipcub::BlockReduceAlgorithm Algorithm = hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS
>
struct params
{
    using type = T;
    static constexpr hipcub::BlockReduceAlgorithm algorithm = Algorithm;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubBlockReduceSingleValueTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr hipcub::BlockReduceAlgorithm algorithm = Params::algorithm;
    static constexpr unsigned int block_size = Params::block_size;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS
    // -----------------------------------------------------------------------
    params<int, 64U>,
    params<int, 128U>,
    params<int, 192U>,
    params<int, 256U>,
    params<int, 512U>,
    params<int, 1024U>,
    params<int, 65U>,
    params<int, 37U>,
    params<int, 129U>,
    params<int, 162U>,
    params<int, 255U>,
    // uint tests
    params<unsigned int, 64U>,
    params<unsigned int, 256U>,
    params<unsigned int, 377U>,
    // long tests
    params<long, 64U>,
    params<long, 256U>,
    params<long, 377U>,
    // -----------------------------------------------------------------------
    // hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING
    // -----------------------------------------------------------------------
    params<int, 64U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<int, 128U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<int, 192U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<int, 256U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<int, 512U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<int, 1024U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<unsigned long, 65U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<long, 37U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<short, 162U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<unsigned int, 255U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<int, 377U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<unsigned char, 377U, 1, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>
> SingleValueTestParams;

TYPED_TEST_CASE(HipcubBlockReduceSingleValueTests, SingleValueTestParams);

template<
    unsigned int BlockSize,
    hipcub::BlockReduceAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void reduce_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    using breduce_t = hipcub::BlockReduce<T, BlockSize, Algorithm>;
    __shared__ typename breduce_t::TempStorage temp_storage;
    value = breduce_t(temp_storage).Reduce(value, hipcub::Sum());
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = value;
    }
}

TYPED_TEST(HipcubBlockReduceSingleValueTests, Reduce)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value += output[idx];
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_kernel<block_size, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_reductions
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

TYPED_TEST_CASE(HipcubBlockReduceSingleValueTests, SingleValueTestParams);

template<
    unsigned int BlockSize,
    hipcub::BlockReduceAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void reduce_valid_kernel(T* device_output, T* device_output_reductions, const unsigned int valid_items)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    using breduce_t = hipcub::BlockReduce<T, BlockSize, Algorithm>;
    __shared__ typename breduce_t::TempStorage temp_storage;
    value = breduce_t(temp_storage).Reduce(value, hipcub::Sum(), valid_items);
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = value;
    }
}

TYPED_TEST(HipcubBlockReduceSingleValueTests, ReduceValid)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;

    constexpr size_t block_size = TestFixture::block_size;
    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const unsigned int valid_items = test_utils::get_random_value(
            block_size - 10,
            block_size,
            seed_value
        );

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(
            size,
            2,
            200,
            seed_value + seed_value_addition
        );
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < valid_items; j++)
            {
                auto idx = i * block_size + j;
                value += output[idx];
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_valid_kernel<block_size, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_reductions, valid_items
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}


template<class Params>
class HipcubBlockReduceInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr hipcub::BlockReduceAlgorithm algorithm = Params::algorithm;
    static constexpr unsigned int items_per_thread = Params::items_per_thread;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS
    // -----------------------------------------------------------------------
    params<float, 6U,   32>,
    params<float, 32,   2>,
    params<unsigned int, 256,  3>,
    params<int, 512,  4>,
    params<float, 1024, 1>,
    params<float, 37,   2>,
    params<float, 65,   5>,
    params<float, 162,  7>,
    params<float, 255,  15>,
    // -----------------------------------------------------------------------
    // hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING
    // -----------------------------------------------------------------------
    params<float, 6U,   32, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<float, 32,   2,  hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<int, 256,  3,  hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<unsigned int, 512,  4,  hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<float, 1024, 1,  hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<float, 37,   2,  hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<float, 65,   5,  hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<float, 162,  7,  hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>,
    params<float, 255,  15, hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>
> InputArrayTestParams;

TYPED_TEST_CASE(HipcubBlockReduceInputArrayTests, InputArrayTestParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    hipcub::BlockReduceAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void reduce_array_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    T reduction;
    using breduce_t = hipcub::BlockReduce<T, BlockSize, Algorithm>;
    __shared__ typename breduce_t::TempStorage temp_storage;
    reduction = breduce_t(temp_storage).Reduce(in_out, hipcub::Sum());

    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}


TYPED_TEST(HipcubBlockReduceInputArrayTests, Reduce)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, 0);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            T value = 0;
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                value += output[idx];
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output_reductions, output_reductions.data(),
                output_reductions.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_array_kernel<block_size, items_per_thread, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_reductions
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_NEAR(
                output_reductions[i], expected_reductions[i],
                static_cast<T>(0.05) * expected_reductions[i]
            );
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}
