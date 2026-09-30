// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test_utils_custom_test_types.hpp"

#include <hipcub/agent/single_pass_scan_operators.hpp>
#include <hipcub/config.hpp>
#include <hipcub/thread/thread_operators.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>

template<typename K, typename V, typename OpK = hipcub::Sum, typename OpV = hipcub::Sum>
struct custom_key_value_pair_op
{
    using type = hipcub::KeyValuePair<K, V>;

    HIPCUB_HOST_DEVICE
    HIPCUB_FORCEINLINE type
        operator()(type a, type b)
    {
        return type(OpK{}(a.key, b.key), OpV{}(a.value, b.value));
    }
};

template<typename T>
struct make_value
{
    template<typename U>
    HIPCUB_HOST_DEVICE
    HIPCUB_FORCEINLINE T
        operator()(U v)
    {
        return T(v);
    }
};

template<typename K, typename V>
struct make_value<hipcub::KeyValuePair<K, V>>
{
    // We need a special constructor to produce key value pairs from singular values.
    template<typename U>
    HIPCUB_HOST_DEVICE
    HIPCUB_FORCEINLINE hipcub::KeyValuePair<K, V>
                       operator()(U v)
    {
        return hipcub::KeyValuePair<K, V>(make_value<K>{}(v), make_value<V>{}(v));
    }
};

template<typename TileState>
__global__
static void InitKernel(int num_tiles, TileState tile_state)
{
    tile_state.InitializeStatus(num_tiles);
}

template<typename ScanOp, typename TileState, typename T>
__global__
static void PrefixKernel(TileState tile_state, T* d_input, T* d_output)
{
    using TilePrefix = hipcub::TilePrefixCallbackOp<T, ScanOp, TileState>;

    HIPCUB_SHARED_MEMORY typename TilePrefix::TempStorage prefix_share;

    const unsigned int tile_id   = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;

    // Load value.
    T value = d_input[tile_id];

    // Compute the device-wide prefix sum.
    if(tile_id == 0)
    {
        // Mark tile 0 as complete.
        if(thread_id == 0)
        {
            // Only one thread has to mark the tile as complete.
            tile_state.SetInclusive(tile_id, value);
            d_output[tile_id] = value;
        }
    }
    else
    {
        TilePrefix tile_prefix = TilePrefix(tile_state, prefix_share, ScanOp{}, tile_id);
        if(thread_id < HIPCUB_DEVICE_WARP_THREADS)
        {
            // Compute tile prefix sum over all tiles, called by the first warp.
            tile_prefix(value);
        }

        if(thread_id == 0)
        {
            T a = ScanOp{}(tile_prefix.GetExclusivePrefix(), value);
            T b = tile_prefix.GetInclusivePrefix();

            // Check if: exclusive_scan + value = inclusive_scan.
            // cub::KeyValuePair only defines operator!= and not operator==.
            if(a != b)
            {
                printf("Incorrect inclusive or exclusive prefix at block index %u\n", tile_id);
            }
            else
            {
                d_output[tile_id] = a;
            }
        }
    }
}

template<typename T,
         typename TileState = hipcub::ScanTileState<T>,
         int BlockSize      = 64,
         typename ScanOp    = hipcub::Sum>
struct SinglePassScanRunner
{
    void run(int num_items, T* d_input, T* d_output)
    {
        size_t temp_storage_bytes;
        HIP_CHECK(TileState::AllocationSize(num_items, temp_storage_bytes));

        void* d_temp_storage;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_bytes));

        TileState tile_state{};
        HIP_CHECK(tile_state.Init(num_items, d_temp_storage, temp_storage_bytes));

        InitKernel<<<num_items, BlockSize>>>(num_items, tile_state);
        HIP_CHECK(hipGetLastError());

        PrefixKernel<ScanOp><<<num_items, BlockSize>>>(tile_state, d_input, d_output);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipFree(d_temp_storage));
    }
};

// Requried to test the 'hipcub::detail::ScanTileStateAsInternal' code-path.
template<typename T>
struct custom_scan_tile_state : hipcub::ScanTileState<T>
{};

template<typename T,
         typename ScanTileState = hipcub::ScanTileState<T>,
         typename ScanOp        = hipcub::Sum>
struct SinglePassScanParams
{
    using type            = T;
    using scan_tile_state = ScanTileState;
    using scan_op         = ScanOp;
};

template<class Params>
class HipcubSinglePassScanTest : public ::testing::Test
{
public:
    using params = Params;
};

using HipcubSinglePassScanTestParams = ::testing::Types<
    // rocPRIM: Tests passing the internal backing tile state to rocPRIM
    SinglePassScanParams<int8_t>,
    SinglePassScanParams<int16_t>,
    SinglePassScanParams<int32_t>,
    SinglePassScanParams<int64_t>,
    SinglePassScanParams<__int128_t>,
    SinglePassScanParams<test_utils::custom_test_type<int32_t>>,
    SinglePassScanParams<test_utils::custom_test_type<int64_t>>,
    // rocPRIM: Tests 'ScanTileStateAsInternal'
    SinglePassScanParams<int32_t, custom_scan_tile_state<int32_t>>,
    SinglePassScanParams<__int128_t, custom_scan_tile_state<__int128_t>>,
    SinglePassScanParams<test_utils::custom_test_type<int64_t>,
                         custom_scan_tile_state<test_utils::custom_test_type<int64_t>>>
    // Note: SinglePassScanParams<hipcub::KeyValuePair<...>, ...> is not here
    // due to missing hipcub::KeyValuePair<Key, Value>::operator==(...) in CUB and rocPRIM
    >;

TYPED_TEST_SUITE(HipcubSinglePassScanTest, HipcubSinglePassScanTestParams);

TYPED_TEST(HipcubSinglePassScanTest, IntSum)
{
    // We want to test multiple number of tiles
    const std::vector<int> num_tiles_to_test = {1, 2, 63, 64, 65};

    // Get the type to test
    using value_type      = typename TestFixture::params::type;
    using scan_tile_state = typename TestFixture::params::scan_tile_state;
    using scan_op         = typename TestFixture::params::scan_op;

    // We use a runner struct to handle the allocation of temporary buffers
    // and template instantiation.
    SinglePassScanRunner<value_type, scan_tile_state, HIPCUB_WARP_SIZE_64, scan_op> runner{};

    for(const auto num_tiles : num_tiles_to_test)
    {
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            const size_t num_bytes = sizeof(value_type) * num_tiles;

            std::vector<value_type> input(num_tiles, make_value<value_type>{}(0));
            std::vector<value_type> output(num_tiles, make_value<value_type>{}(0));

            // Fill input with random data
            std::default_random_engine             gen(seed_value);
            std::uniform_int_distribution<int32_t> distribution(1, 16);
            std::generate(input.begin(),
                          input.end(),
                          [&]() { return make_value<value_type>{}(distribution(gen)); });

            value_type* d_input;
            value_type* d_output;

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, num_bytes));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, num_bytes));

            HIP_CHECK(hipMemcpy(d_input, input.data(), num_bytes, hipMemcpyHostToDevice));

            runner.run(num_tiles, d_input, d_output);

            // Copy results from device to host.
            HIP_CHECK(hipMemcpy(output.data(), d_output, num_bytes, hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));

            // Compute expected on host
            std::vector<value_type> expected(num_tiles);
            std::partial_sum(input.begin(), input.end(), expected.begin(), scan_op{});

            // We can do direct comparison because we're working on integral types.
            test_utils::assert_eq(output, expected);
        }
    }
}

template<typename ScanOp, int num_items, typename T>
__global__
static void RunningPrefixKernel(T* d_input, T* d_output)
{
    using prefix_type = hipcub::BlockScanRunningPrefixOp<T, ScanOp>;

    if(threadIdx.x > 0 || blockIdx.x > 0)
        return;

    prefix_type prefix(T(), ScanOp{});

#pragma unroll
    for(int i = 0; i < num_items; ++i)
    {
        T value            = d_input[i];
        T exclusive_prefix = prefix(value);
        d_output[i]        = exclusive_prefix + value;
    }
}

template<typename T, int NumItems, typename ScanOp = hipcub::Sum>
struct RunningPrefixRunner
{
    void run(T* d_input, T* d_output)
    {
        RunningPrefixKernel<ScanOp, NumItems><<<1, 32>>>(d_input, d_output);
        HIP_CHECK(hipGetLastError());
    }
};

template<typename T>
struct RunningPrefixParams
{
    using type = T;
};

template<class Params>
class HipcubRunningPrefixTest : public ::testing::Test
{
public:
    using params = Params;
};

using HipcubRunningPrefixParams
    = ::testing::Types<RunningPrefixParams<int8_t>,
                       RunningPrefixParams<int16_t>,
                       RunningPrefixParams<int32_t>,
                       RunningPrefixParams<int64_t>,
                       RunningPrefixParams<__int128_t>,
                       RunningPrefixParams<test_utils::custom_test_type<int32_t>>,
                       RunningPrefixParams<test_utils::custom_test_type<int64_t>>>;

TYPED_TEST_SUITE(HipcubRunningPrefixTest, HipcubRunningPrefixParams);

TYPED_TEST(HipcubRunningPrefixTest, IntSum)
{
    constexpr int num_items = 8;

    // Get the type to test
    using value_type = typename TestFixture::params::type;

    // We use a runner struct to handle the allocation of temporary buffers
    // and template instantiation.
    RunningPrefixRunner<value_type, num_items> runner{};

    const size_t num_bytes = sizeof(value_type) * num_items;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        std::vector<value_type> input(num_items, value_type(0));
        std::vector<value_type> output(num_items, value_type(0));

        // Fill input with random data
        std::default_random_engine             gen(seed_value);
        std::uniform_int_distribution<int32_t> distribution(1, 16);
        std::generate(input.begin(),
                      input.end(),
                      [&]() { return make_value<value_type>{}(distribution(gen)); });

        value_type* d_input;
        value_type* d_output;

        HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, num_bytes));
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, num_bytes));

        HIP_CHECK(hipMemcpy(d_input, input.data(), num_bytes, hipMemcpyHostToDevice));

        runner.run(d_input, d_output);

        // Copy results from device to host.
        HIP_CHECK(hipMemcpy(output.data(), d_output, num_bytes, hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));

        // Compute expected on host
        std::vector<value_type> expected(num_items);
        std::partial_sum(input.begin(), input.end(), expected.begin());

        // We can do direct comparison because we're working on integral types.
        test_utils::assert_eq(output, expected);
    }
}
