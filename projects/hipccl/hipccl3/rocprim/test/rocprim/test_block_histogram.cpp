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

// required rocprim headers
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>
#include <rocprim/block/block_histogram.hpp>

// required test headers
#include "test_utils_types.hpp"

// Params for tests
template<class Params>
class RocprimBlockHistogramAtomicInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::input_type;
    using bin_type = typename Params::output_type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr unsigned int bin_size = Params::block_size;
};

template<class Params>
class RocprimBlockHistogramSortInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::input_type;
    using bin_type = typename Params::output_type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr unsigned int bin_size = Params::block_size;
};

typedef ::testing::Types<
    block_param_type(unsigned int, unsigned int),
    block_param_type(float, float),
    block_param_type(float, unsigned int),
    block_param_type(float, unsigned long long),
    block_param_type(double, float),
    block_param_type(double, unsigned long long)
> BlockHistAtomicParams;

typedef ::testing::Types<
    block_param_type(uint8_t, int),
    block_param_type(uint8_t, uint8_t),
    block_param_type(uint8_t, short),
    block_param_type(uint8_t, int8_t),
    block_param_type(unsigned short, rocprim::half),
    block_param_type(unsigned int, rocprim::half)
> BlockHistSortParams;

TYPED_TEST_CASE(RocprimBlockHistogramAtomicInputArrayTests, BlockHistAtomicParams);
TYPED_TEST_CASE(RocprimBlockHistogramSortInputArrayTests, BlockHistSortParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int BinSize,
    rocprim::block_histogram_algorithm Algorithm,
    class T,
    class BinType
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void histogram_kernel(T* device_output, T* device_output_bin)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    unsigned int global_offset = hipBlockIdx_x * BinSize;
    __shared__ BinType hist[BinSize];
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rocprim::block_histogram<T, BlockSize, ItemsPerThread, BinSize, Algorithm> bhist;
    bhist.histogram(in_out, hist);

    #pragma unroll
    for (unsigned int offset = 0; offset < BinSize; offset += BlockSize)
    {
        if(offset + hipThreadIdx_x < BinSize)
        {
            device_output_bin[global_offset + hipThreadIdx_x] = hist[offset + hipThreadIdx_x];
            global_offset += BlockSize;
        }
    }
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
    constexpr auto algorithm = Algorithm;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t bin = BlockSize;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t bin_sizes = bin * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 0, bin - 1, seed_value);

        // Output histogram results
        std::vector<T> output_bin(bin_sizes, 0);

        // Calculate expected results on host
        std::vector<T> expected_bin(output_bin.size(), 0);
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
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_bin;
        HIP_CHECK(hipMalloc(&device_output_bin, output_bin.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output_bin, output_bin.data(),
                output_bin.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(histogram_kernel<block_size, items_per_thread, bin, algorithm, T, BinType>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_bin
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_bin.data(), device_output_bin,
                output_bin.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_eq(output_bin, expected_bin);

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_bin));
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
        int device_id = test_common_utils::obtain_device_from_ctest();
        SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
        HIP_CHECK(hipSetDevice(device_id));

        test_block_histogram_input_arrays<T, BinType, BlockSize, items[First], Algorithm>();
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

TYPED_TEST(RocprimBlockHistogramAtomicInputArrayTests, Histogram)
{
    using T = typename TestFixture::type;
    using BinType = typename TestFixture::bin_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 4, T, BinType, block_size, rocprim::block_histogram_algorithm::using_atomic>::run();
}

TYPED_TEST(RocprimBlockHistogramSortInputArrayTests, Histogram)
{
    using T = typename TestFixture::type;
    using BinType = typename TestFixture::bin_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 4, T, BinType, block_size, rocprim::block_histogram_algorithm::using_sort>::run();
}
