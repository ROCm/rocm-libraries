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
// rocPRIM
#include <block/block_scan.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

// Params for tests
template<
    class T,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U
>
struct params
{
    using type = T;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimBlockScanSingleValueTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr unsigned int block_size = Params::block_size;
};

typedef ::testing::Types<
    params<int, 64U>,
    params<int, 128U>,
    params<int, 256U>,
    params<int, 512U>,
    params<int, 1024U>,
    params<int, 65U>,
    params<int, 37U>,
    params<int, 162U>,
    params<int, 255U>,
    // uint tests
    params<unsigned int, 64U>,
    params<unsigned int, 256U>,
    params<unsigned int, 377U>,
    // long tests
    params<long, 64U>,
    params<long, 256U>,
    params<long, 377U>
> SingleValueTestParams;

TYPED_TEST_CASE(RocprimBlockScanSingleValueTests, SingleValueTestParams);

TYPED_TEST(RocprimBlockScanSingleValueTests, InclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<T> output = get_random_data<T>(size, 2, 200);

    // Calulcate expected results on host
    std::vector<T> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            T value = d_output[i];
            rp::block_scan<T, block_size> bscan;
            bscan.inclusive_scan(value, value);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimBlockScanSingleValueTests, InclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<T> output = get_random_data<T>(size, 2, 200);
    std::vector<T> output_reductions(size / block_size);

    // Calulcate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
        expected_reductions[i] = expected[(i+1) * block_size - 1];
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            T value = d_output[i];
            T reduction;
            rp::block_scan<T, block_size> bscan;
            bscan.inclusive_scan(value, value, reduction);
            d_output[i] = value;
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = reduction;
            }
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    d_output_r.synchronize();
    for(size_t i = 0; i < output_reductions.size(); i++)
    {
        ASSERT_EQ(output_reductions[i], expected_reductions[i]);
    }
}

TYPED_TEST(RocprimBlockScanSingleValueTests, ExclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<T> output = get_random_data<T>(size, 2, 200);
    const T init = get_random_value(0, 100);

    // Calulcate expected results on host
    std::vector<T> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        expected[i * block_size] = init;
        for(size_t j = 1; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = output[idx-1] + expected[idx-1];
        }
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            T value = d_output[i];
            rp::block_scan<T, block_size> bscan;
            bscan.exclusive_scan(value, value, init);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimBlockScanSingleValueTests, ExclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;

    hc::accelerator acc;
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<T> output = get_random_data<T>(size, 2, 200);
    const T init = get_random_value(0, 100);

    // Output reduce results
    std::vector<T> output_reductions(size / block_size);

    // Calulcate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        expected[i * block_size] = init;
        for(size_t j = 1; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected[idx] = output[idx-1] + expected[idx-1];
        }

        expected_reductions[i] = 0;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            expected_reductions[i] += output[idx];
        }
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            T value = d_output[i];
            T reduction;
            rp::block_scan<T, block_size> bscan;
            bscan.exclusive_scan(value, value, init, reduction);
            d_output[i] = value;
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = reduction;
            }
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    d_output_r.synchronize();
    for(size_t i = 0; i < output_reductions.size(); i++)
    {
        ASSERT_EQ(output_reductions[i], expected_reductions[i]);
    }
}

// ---------------------------------------------------------
// Test for scan ops taking array of values as input
// ---------------------------------------------------------

template<class Params>
class RocprimBlockScanInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr unsigned int items_per_thread = Params::items_per_thread;
};

typedef ::testing::Types<
    params<float, 64U, 32>,
    params<float, 32U, 2U>,
    params<float, 256U, 3>,
    params<float, 512U, 4>,
    params<float, 1024U, 1>,
    params<float, 37U, 2>,
    params<float, 65U, 5>,
    params<float, 162U, 7>,
    params<float, 255U, 15>
> InputArrayTestParams;

TYPED_TEST_CASE(RocprimBlockScanInputArrayTests, InputArrayTestParams);

TYPED_TEST(RocprimBlockScanInputArrayTests, InclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 1;
    // Generate data
    std::vector<T> output = get_random_data<T>(size, 2, 200);

    // Calulcate expected results on host
    std::vector<T> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
    }

    // global/grid size
    const size_t global_size = output.size()/items_per_thread;
    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(global_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            size_t idx = i.global[0] * items_per_thread;

            // load
            T in_out[items_per_thread];
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                in_out[j] = d_output[idx + j];
            }

            rp::block_scan<T, block_size> bscan;
            bscan.inclusive_scan(in_out, in_out);

            // store
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                d_output[idx + j] = in_out[j];
            }
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_NEAR(
            output[i], expected[i],
            static_cast<T>(0.05) * expected[i]
        );
    }
}

TYPED_TEST(RocprimBlockScanInputArrayTests, InclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 1;
    // Generate data
    std::vector<T> output = get_random_data<T>(size, 2, 200);

    // Output reduce results
    std::vector<T> output_reductions(size / block_size);

    // Calulcate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
        }
        expected_reductions[i] = expected[(i+1) * items_per_block - 1];
    }

    // global/grid size
    const size_t global_size = output.size()/items_per_thread;
    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(global_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            size_t idx = i.global[0] * items_per_thread;

            // load
            T in_out[items_per_thread];
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                in_out[j] = d_output[idx + j];
            }

            rp::block_scan<T, block_size> bscan;
            T reduction;
            bscan.inclusive_scan(in_out, in_out, reduction);

            // store
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                d_output[idx + j] = in_out[j];
            }
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = reduction;
            }
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_NEAR(
            output[i], expected[i],
            static_cast<T>(0.05) * expected[i]
        );
    }

    d_output_r.synchronize();
    for(size_t i = 0; i < output_reductions.size(); i++)
    {
        ASSERT_NEAR(
            output_reductions[i], expected_reductions[i],
            static_cast<T>(0.05) * expected_reductions[i]
        );
    }
}

TYPED_TEST(RocprimBlockScanInputArrayTests, ExclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 1;
    // Generate data
    std::vector<T> output = get_random_data<T>(size, 2, 200);
    const T init = get_random_value(0, 100);

    // Calulcate expected results on host
    std::vector<T> expected(output.size(), 0);
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        expected[i * items_per_block] = init;
        for(size_t j = 1; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = output[idx-1] + expected[idx-1];
        }
    }

    // global/grid size
    const size_t global_size = output.size()/items_per_thread;
    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(global_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            size_t idx = i.global[0] * items_per_thread;

            // load
            T in_out[items_per_thread];
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                in_out[j] = d_output[idx + j];
            }

            rp::block_scan<T, block_size> bscan;
            bscan.exclusive_scan(in_out, in_out, init);

            // store
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                d_output[idx + j] = in_out[j];
            }
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_NEAR(
            output[i], expected[i],
            static_cast<T>(0.05) * expected[i]
        );
    }
}

TYPED_TEST(RocprimBlockScanInputArrayTests, ExclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 1;
    // Generate data
    std::vector<T> output = get_random_data<T>(size, 2, 200);

    // Output reduce results
    std::vector<T> output_reductions(size / block_size);
    const T init = get_random_value(0, 100);

    // Calulcate expected results on host
    std::vector<T> expected(output.size(), 0);
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        expected[i * items_per_block] = init;
        for(size_t j = 1; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            expected[idx] = output[idx-1] + expected[idx-1];
        }
        for(size_t j = 0; j < items_per_block; j++)
        {
            expected_reductions[i] += output[i * items_per_block + j];
        }
    }

    // global/grid size
    const size_t global_size = output.size()/items_per_thread;
    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(global_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            size_t idx = i.global[0] * items_per_thread;

            // load
            T in_out[items_per_thread];
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                in_out[j] = d_output[idx + j];
            }

            rp::block_scan<T, block_size> bscan;
            T reduction;
            bscan.exclusive_scan(in_out, in_out, init, reduction);

            // store
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                d_output[idx + j] = in_out[j];
            }
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = reduction;
            }
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_NEAR(
            output[i], expected[i],
            static_cast<T>(0.05) * expected[i]
        );
    }

    d_output_r.synchronize();
    for(size_t i = 0; i < output_reductions.size(); i++)
    {
        ASSERT_NEAR(
            output_reductions[i], expected_reductions[i],
            static_cast<T>(0.05) * expected_reductions[i]
        );
    }
}