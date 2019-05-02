// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <vector>

// Google Test
#include <gtest/gtest.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(static_cast<hipError_t>(error), hipSuccess)

namespace rp = rocprim;

// Params for tests
template<
    class T,
    class BinType,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    unsigned int BinSize = BlockSize,
    rocprim::block_histogram_algorithm Algorithm = rocprim::block_histogram_algorithm::using_atomic
>
struct params
{
    using type = T;
    using bin_type = BinType;
    static constexpr rocprim::block_histogram_algorithm algorithm = Algorithm;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr unsigned int bin_size = BinSize;
};

template<class Params>
class RocprimBlockHistogramInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    using bin_type = typename Params::bin_type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr rocprim::block_histogram_algorithm algorithm = Params::algorithm;
    static constexpr unsigned int items_per_thread = Params::items_per_thread;
    static constexpr unsigned int bin_size = Params::bin_size;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // rocprim::block_histogram_algorithm::using_atomic
    // -----------------------------------------------------------------------
    params<unsigned int, unsigned int, 6U,   32, 18U>,
    params<unsigned int, unsigned int, 32,   2, 64>,
    params<unsigned int, unsigned int, 256,  3, 512>,
    params<unsigned int, unsigned int, 512,  4>,
    params<unsigned int, unsigned int, 1024, 1>,
    params<unsigned int, unsigned int, 37,   2>,
    params<unsigned int, unsigned int, 65,   5>,
    params<unsigned int, unsigned int, 162,  7>,
    params<unsigned int, unsigned int, 255,  15>,
    params<float, float, 6U,   32, 18U>,
    params<float, float, 32,   2, 64>,
    params<float, float, 256,  3, 512>,
    params<float, unsigned int, 512,  4>,
    params<float, unsigned int, 1024, 1>,
    // -----------------------------------------------------------------------
    // rocprim::block_histogram_algorithm::using_sort
    // -----------------------------------------------------------------------
    params<unsigned int, unsigned int, 6U,   32,  18U, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned int, unsigned int, 32,   2,   64, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned int, unsigned int, 256,  3,  512, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned int, unsigned int, 512,  4,  512, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned int, unsigned int, 1024, 1, 1024, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned int, unsigned int, 37,   2,   37, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned int, unsigned int, 65,   5,   65, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned int, unsigned int, 162,  7,  162, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned int, unsigned int, 255,  15, 255, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned char, unsigned int, 6U,   32,  18U, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned char, unsigned int, 32,   2,   64, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned char, unsigned int, 256,  3,  512, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned char, unsigned char, 512,  4,  512, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned char, unsigned char, 1024, 1, 1024, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned short, unsigned int, 6U,   32,  18U, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned short, unsigned int, 32,   2,   64, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned short, unsigned int, 256,  3,  512, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned short, unsigned short, 512,  4,  512, rocprim::block_histogram_algorithm::using_sort>,
    params<unsigned short, unsigned short, 1024, 1, 1024, rocprim::block_histogram_algorithm::using_sort>
> InputArrayTestParams;

TYPED_TEST_CASE(RocprimBlockHistogramInputArrayTests, InputArrayTestParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int BinSize,
    rocprim::block_histogram_algorithm Algorithm,
    class T,
    class BinType
>
__global__
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
    
    rp::block_histogram<T, BlockSize, ItemsPerThread, BinSize, Algorithm> bhist;
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

TYPED_TEST(RocprimBlockHistogramInputArrayTests, Histogram)
{
    using T = typename TestFixture::type;
    using BinType = typename TestFixture::bin_type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;
    constexpr size_t bin = TestFixture::bin_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t bin_sizes = bin * 37;
    const size_t grid_size = size / items_per_block;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 0, bin - 1);

    // Output reduce results
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
    
    for(size_t i = 0; i < output_bin.size(); i++)
    {
        ASSERT_EQ(
            output_bin[i], expected_bin[i]
        );
    }
    
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_output_bin));
}
