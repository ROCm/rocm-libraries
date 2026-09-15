// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

#include "../../common/utils_data_generation.hpp"

// required test headers
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_data_generation.hpp"

// required common headers
#include "../../common/utils_device_ptr.hpp"

// required rocprim headers
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_run_length_decode.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdint.h>
#include <vector>

template<class ItemT,
         class LengthT,
         unsigned BlockSize,
         unsigned RunsPerThread,
         unsigned DecodedItemsPerThread>
struct Params
{
    using item_type                                    = ItemT;
    using length_type                                  = LengthT;
    static constexpr unsigned block_size               = BlockSize;
    static constexpr unsigned runs_per_thread          = RunsPerThread;
    static constexpr unsigned decoded_items_per_thread = DecodedItemsPerThread;
};

template<class Params>
class HipcubBlockRunLengthDecodeTest : public ::testing::Test
{
public:
    using params = Params;
};

using HipcubBlockRunLengthDecodeTestParams
    = ::testing::Types<Params<int, int, 256, 4, 4>,
                       Params<int8_t, int, 256, 4, 4>,
                       Params<uint8_t, int, 256, 4, 4>,
                       Params<double, char, 256, 4, 4>,
                       Params<char, long long, 256, 4, 4>,
                       Params<float, int, 256, 4, 4>,
                       Params<rocprim::half, int, 256, 4, 4>,
                       Params<rocprim::bfloat16, int, 256, 4, 4>,

                       Params<int, int, 256, 8, 8>,
                       Params<double, char, 256, 8, 8>,
                       Params<char, long long, 256, 8, 8>,
                       Params<float, int, 256, 8, 8>,
                       Params<rocprim::half, int, 256, 8, 8>,
                       Params<rocprim::bfloat16, int, 256, 8, 8>,

                       Params<int, int, 256, 1, 14>,
                       Params<double, char, 256, 1, 14>,
                       Params<char, long long, 256, 1, 14>,
                       Params<float, int, 256, 1, 14>,
                       Params<rocprim::half, int, 256, 1, 14>,
                       Params<rocprim::bfloat16, int, 256, 1, 14>,

                       Params<int, int, 256, 9, 7>,
                       Params<double, char, 256, 9, 7>,
                       Params<char, long long, 256, 9, 7>,
                       Params<float, int, 256, 9, 7>,
                       Params<rocprim::half, int, 256, 9, 7>,
                       Params<rocprim::bfloat16, int, 256, 9, 7>>;

TYPED_TEST_SUITE(HipcubBlockRunLengthDecodeTest, HipcubBlockRunLengthDecodeTestParams);

template<class ItemT,
         class LengthT,
         unsigned BlockSize,
         unsigned RunsPerThread,
         unsigned DecodedItemsPerThread>
__global__
    __launch_bounds__(BlockSize) void block_run_length_decode_kernel(const ItemT*   d_run_items,
                                                                     const LengthT* d_run_lengths,
                                                                     ItemT*         d_decoded_items,
                                                                     LengthT* d_decoded_offsets)
{
    using BlockRunLengthDecodeT
        = rocprim::block_run_length_decode<ItemT, BlockSize, RunsPerThread, DecodedItemsPerThread>;

    static constexpr unsigned int decoded_items_per_block = BlockSize * DecodedItemsPerThread;

    ROCPRIM_SHARED_MEMORY typename BlockRunLengthDecodeT::storage_type temp_storage;

    ItemT   run_items[RunsPerThread];
    LengthT run_lengths[RunsPerThread];

    const unsigned global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;
    rocprim::block_load_direct_blocked(global_thread_idx, d_run_items, run_items);
    rocprim::block_load_direct_blocked(global_thread_idx, d_run_lengths, run_lengths);

    unsigned              total_decoded_size{};
    BlockRunLengthDecodeT block_run_length_decode(temp_storage,
                                                  run_items,
                                                  run_lengths,
                                                  total_decoded_size);

    unsigned decoded_window_offset = 0;
    while(decoded_window_offset < total_decoded_size)
    {
        ItemT   decoded_items[DecodedItemsPerThread];
        LengthT decoded_offsets[DecodedItemsPerThread];

        block_run_length_decode.run_length_decode(decoded_items,
                                                  decoded_offsets,
                                                  decoded_window_offset);

        rocprim::block_store_direct_blocked(
            global_thread_idx,
            d_decoded_items + decoded_window_offset,
            decoded_items,
            rocprim::minimum<unsigned int>{}(total_decoded_size - decoded_window_offset,
                                             decoded_items_per_block));

        rocprim::block_store_direct_blocked(
            global_thread_idx,
            d_decoded_offsets + decoded_window_offset,
            decoded_offsets,
            rocprim::minimum<unsigned int>{}(total_decoded_size - decoded_window_offset,
                                             decoded_items_per_block));

        decoded_window_offset += decoded_items_per_block;
    }
}

TYPED_TEST(HipcubBlockRunLengthDecodeTest, TestDecode)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using ItemT                                 = typename TestFixture::params::item_type;
    using LengthT                               = typename TestFixture::params::length_type;
    constexpr unsigned block_size               = TestFixture::params::block_size;
    constexpr unsigned runs_per_thread          = TestFixture::params::runs_per_thread;
    constexpr unsigned decoded_items_per_thread = TestFixture::params::decoded_items_per_thread;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        size_t            num_runs       = runs_per_thread * block_size;
        constexpr LengthT max_run_length = static_cast<LengthT>(
            std::min(1000ll, static_cast<long long>(rocprim::numeric_limits<LengthT>::max())));

        auto run_items = std::vector<ItemT>(num_runs);
        run_items[0]   = test_utils::get_random_value<ItemT>(rocprim::numeric_limits<ItemT>::min(),
                                                           rocprim::numeric_limits<ItemT>::max(),
                                                           ++seed_value);

        size_t run_item_index = 1;
        while(run_item_index < num_runs)
        {
            run_items[run_item_index]
                = test_utils::get_random_value<ItemT>(rocprim::numeric_limits<ItemT>::min(),
                                                      rocprim::numeric_limits<ItemT>::max(),
                                                      ++seed_value);
            if(test_utils::convert_to_native(run_items[run_item_index])
               != test_utils::convert_to_native(run_items[run_item_index - 1]))
            {
                ++run_item_index;
            }
        }

        auto run_lengths = test_utils::get_random_data_wrapped<LengthT>(num_runs,
                                                                        static_cast<LengthT>(1),
                                                                        max_run_length,
                                                                        seed_value);

        std::default_random_engine               prng(seed_value);
        common::uniform_int_distribution<size_t> num_empty_runs_dist(1, 4);
        const size_t num_trailing_empty_runs = num_empty_runs_dist(prng);
        num_runs += num_trailing_empty_runs;

        const auto empty_run_items
            = test_utils::get_random_data_wrapped<ItemT>(num_trailing_empty_runs,
                                                         rocprim::numeric_limits<ItemT>::min(),
                                                         rocprim::numeric_limits<ItemT>::max(),
                                                         seed_value);
        run_items.insert(run_items.end(), empty_run_items.begin(), empty_run_items.end());
        run_lengths.insert(run_lengths.end(), num_trailing_empty_runs, static_cast<LengthT>(0));

        std::vector<ItemT> expected;
        for(size_t i = 0; i < run_items.size(); ++i)
        {
            for(size_t j = 0; j < static_cast<size_t>(run_lengths[i]); ++j)
            {
                expected.push_back(run_items[i]);
            }
        }

        common::device_ptr<ItemT>   d_run_items(run_items);
        common::device_ptr<LengthT> d_run_lengths(run_lengths);
        common::device_ptr<ItemT>   d_decoded_runs(expected.size());
        common::device_ptr<LengthT> d_decoded_offsets(expected.size());

        block_run_length_decode_kernel<ItemT,
                                       LengthT,
                                       block_size,
                                       runs_per_thread,
                                       decoded_items_per_thread>
            <<<dim3(1), dim3(block_size), 0, 0>>>(d_run_items.get(),
                                                  d_run_lengths.get(),
                                                  d_decoded_runs.get(),
                                                  d_decoded_offsets.get());

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<ItemT>   output  = d_decoded_runs.load();
        std::vector<LengthT> offsets = d_decoded_offsets.load();

        unsigned int expected_offset = -1;
        ItemT        previous_value  = ItemT{};
        for(size_t i = 0; i < output.size(); ++i)
        {
            ASSERT_EQ(test_utils::convert_to_native(output[i]),
                      test_utils::convert_to_native(expected[i]));
            if(test_utils::convert_to_native(output[i])
               != test_utils::convert_to_native(previous_value))
            {
                previous_value  = output[i];
                expected_offset = 0;
            }
            else
            {
                expected_offset = ++expected_offset;
            }

            ASSERT_EQ(offsets[i], expected_offset);
        }
    }
}
