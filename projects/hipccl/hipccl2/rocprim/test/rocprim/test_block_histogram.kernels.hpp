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

#ifndef TEST_BLOCK_HISTOGRAM_KERNELS_HPP_
#define TEST_BLOCK_HISTOGRAM_KERNELS_HPP_

#include "../common_test_header.hpp"

// required test headers
#include "../../common/utils_device_ptr.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

// required rocprim headers
#include <rocprim/block/block_histogram.hpp>
#include <rocprim/config.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <type_traits>
#include <vector>

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int BinSize,
    rocprim::block_histogram_algorithm Algorithm,
    class T,
    class BinType
>
__global__
__launch_bounds__(BlockSize)
void histogram_kernel(T* device_output, BinType* device_output_bin)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
    unsigned int global_offset = blockIdx.x * BinSize;
    __shared__ BinType hist[BinSize];
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rocprim::block_histogram<T, BlockSize, ItemsPerThread, BinSize, Algorithm> bhist;
    bhist.histogram(in_out, hist);
    rocprim::syncthreads();

    ROCPRIM_UNROLL
    for (unsigned int offset = 0; offset < BinSize; offset += BlockSize)
    {
        if(offset + threadIdx.x < BinSize)
        {
            device_output_bin[global_offset + threadIdx.x] = hist[offset + threadIdx.x];
            global_offset += BlockSize;
        }
    }
}

/// \brief Reduce the value of `maxval` such that it can be represented by `T`.
template<typename T>
auto get_safe_maxval(size_t maxval) -> std::enable_if_t<rocprim::is_floating_point<T>::value, bool>
{
    // Assert that the cast is defined behavior, based on the assumption that all floating-point
    //   types can be represented by a double
    EXPECT_LT(static_cast<double>(maxval), static_cast<double>(rocprim::numeric_limits<T>::max()));
    return static_cast<T>(maxval);
}

/// \brief Reduce the value of `maxval` such that it can be represented by `T`.
template<typename T>
auto get_safe_maxval(size_t maxval) -> std::enable_if_t<!rocprim::is_floating_point<T>::value, bool>
{
    return test_utils::saturate_cast<T>(maxval);
}

// Test for histogram
template<
    class T,
    class BinType,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_histogram_algorithm Algorithm = rocprim::block_histogram_algorithm::using_atomic
>
void test_block_histogram_input_arrays()
{
    static constexpr auto algorithm = Algorithm;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;
    static constexpr size_t bin = BlockSize;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t bin_sizes = bin * 37;
    const size_t grid_size = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with bin_sizes = " << bin_sizes);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    // TODO: Use assert near for bin_type.
    if (std::is_same<BinType, ::rocprim::bfloat16>::value) {
        GTEST_SKIP() << "Temporary skipped test";
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
        SCOPED_TRACE(testing::Message() << "with ItemsPerThread = " << items_per_thread);

        // Generate data
        const size_t   max_value = bin - 1;
        std::vector<T> output
            = test_utils::get_random_data_wrapped<T>(size,
                                                     0,
                                                     get_safe_maxval<T>(max_value),
                                                     seed_value);

        // Output histogram results
        std::vector<BinType> output_bin(bin_sizes, 0);

        // Calculate expected results on host
        std::vector<BinType> expected_bin(output_bin.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto bin_idx = i * bin;
                auto idx = i * items_per_block + j;
                expected_bin[bin_idx + static_cast<unsigned int>(output[idx])]++;
            }
        }

        // Preparing device
        common::device_ptr<T>       device_output(output);
        common::device_ptr<BinType> device_output_bin(output_bin);

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                histogram_kernel<block_size, items_per_thread, bin, algorithm, T, BinType>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_output.get(),
            device_output_bin.get());
        HIP_CHECK(hipGetLastError());

        // Reading results back
        output_bin = device_output_bin.load();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_bin, expected_bin));
    }
}

// Static for-loop
template <
    unsigned int First,
    unsigned int Last,
    class T,
    class BinType,
    unsigned int BlockSize = 256U,
    rocprim::block_histogram_algorithm Algorithm = rocprim::block_histogram_algorithm::using_atomic
>
struct static_for_input_array
{
    static void run()
    {
        {
            SCOPED_TRACE(testing::Message() << "TestID = " << First);
            int device_id = test_common_utils::obtain_device_from_ctest();
            SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
            HIP_CHECK(hipSetDevice(device_id));

            test_block_histogram_input_arrays<T, BinType, BlockSize, items[First], Algorithm>();
        }
        static_for_input_array<First + 1, Last, T, BinType, BlockSize, Algorithm>::run();
    }
};

template <
    unsigned int N,
    class T,
    class BinType,
    unsigned int BlockSize,
    rocprim::block_histogram_algorithm Algorithm
>
struct static_for_input_array<N, N, T, BinType, BlockSize, Algorithm>
{
    static void run()
    {
    }
};

#endif // TEST_BLOCK_HISTOGRAM_KERNELS_HPP_
