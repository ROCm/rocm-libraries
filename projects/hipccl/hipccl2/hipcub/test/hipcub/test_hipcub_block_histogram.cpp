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
#include "hipcub/block/block_histogram.hpp"

// Params for tests
template<
    class T,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    unsigned int BinSize = BlockSize,
    hipcub::BlockHistogramAlgorithm Algorithm = hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_ATOMIC
>
struct params
{
    using type = T;
    static constexpr hipcub::BlockHistogramAlgorithm algorithm = Algorithm;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr unsigned int bin_size = BinSize;
};

template<class Params>
class HipcubBlockHistogramInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr hipcub::BlockHistogramAlgorithm algorithm = Params::algorithm;
    static constexpr unsigned int items_per_thread = Params::items_per_thread;
    static constexpr unsigned int bin_size = Params::bin_size;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_ATOMIC
    // -----------------------------------------------------------------------
    params<unsigned int, 6U,   32, 18U>,
    params<unsigned int, 32,   2, 64>,
    params<unsigned int, 256,  3, 512>,
    params<unsigned int, 512,  4>,
    params<unsigned int, 1024, 1>,
    params<unsigned int, 37,   2>,
    params<unsigned int, 65,   5>,
    params<unsigned int, 162,  7>,
    params<unsigned int, 255,  15>,
    // -----------------------------------------------------------------------
    // hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT
    // -----------------------------------------------------------------------
    params<unsigned int, 6U,   32,  18U, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned int, 32,   2,   64, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned int, 256,  3,  512, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned int, 512,  4,  512, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned int, 1024, 1, 1024, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned int, 37,   2,   37, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned int, 65,   5,   65, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned int, 162,  7,  162, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned int, 255,  15, 255, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned char, 6U,   32,  18U, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned char, 32,   2,   64, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned char, 256,  3,  512, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned char, 512,  4,  512, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned char, 1024, 1, 1024, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned short, 6U,   32,  18U, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned short, 32,   2,   64, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned short, 256,  3,  512, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned short, 512,  4,  512, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>,
    params<unsigned short, 1024, 1, 1024, hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>
> InputArrayTestParams;

TYPED_TEST_CASE(HipcubBlockHistogramInputArrayTests, InputArrayTestParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int BinSize,
    hipcub::BlockHistogramAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void histogram_kernel(T* device_output, T* device_output_bin)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    unsigned int global_offset = hipBlockIdx_x * BinSize;
    __shared__ T hist[BinSize];
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bhistogram_t = hipcub::BlockHistogram<T, BlockSize, ItemsPerThread, BinSize, Algorithm>;
    __shared__ typename bhistogram_t::TempStorage temp_storage;
    bhistogram_t(temp_storage).Histogram(in_out, hist);
    __syncthreads();

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

TYPED_TEST(HipcubBlockHistogramInputArrayTests, Histogram)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;
    constexpr size_t bin = TestFixture::bin_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const size_t items_per_block = block_size * items_per_thread;
        const size_t size = items_per_block * 37;
        const size_t bin_sizes = bin * 37;
        const size_t grid_size = size / items_per_block;
        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 0, T(bin - 1), seed_value);

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
            HIP_KERNEL_NAME(histogram_kernel<block_size, items_per_thread, bin, algorithm, T>),
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
}
