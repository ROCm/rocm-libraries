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
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

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

TYPED_TEST(RocprimBlockHistogramInputArrayTests, Histogram)
{
    using T = typename TestFixture::type;
    using BinType = typename TestFixture::bin_type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;
    constexpr size_t bin = TestFixture::bin_size;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc))
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t bin_sizes = bin * 37;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 0, bin - 1);

    // Output reduce results
    std::vector<T> output_bin(bin_sizes, 0);

    // Calculate expected results on host
    std::vector<T> expected_bin(output_bin.size(), 0);
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        auto bin_idx = i * bin;
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected_bin[bin_idx + static_cast<unsigned int>(output[idx])]++;
        }
    }

    // global/grid size
    const size_t global_size = output.size()/items_per_thread;
    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_h(
        output_bin.size(), output_bin.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(global_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            size_t global_offset = i.tile[0] * bin;
            size_t idx = i.global[0] * items_per_thread;
            tile_static BinType hist[bin];

            // load
            T in_out[items_per_thread];
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                in_out[j] = d_output[idx + j];
            }

            rp::block_histogram<T, block_size, items_per_thread, bin, algorithm> bhist;
            bhist.histogram(in_out, hist);
        
            #pragma unroll
            for (unsigned int offset = 0; offset < bin; offset += block_size)
            {
                if(offset + i.local[0] < bin)
                {
                    d_output_h[global_offset + i.local[0]] = hist[offset + i.local[0]];
                    global_offset += block_size;
                }    
            }
        }
    );

    d_output.synchronize();
    d_output_h.synchronize();
    for(size_t i = 0; i < output_bin.size(); i++)
    {
        ASSERT_EQ(
            output_bin[i], expected_bin[i]
        );
    }
}
