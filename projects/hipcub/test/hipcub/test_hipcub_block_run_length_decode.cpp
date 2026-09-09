// MIT License
//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_run_length_decode.hpp>
#include <hipcub/block/block_store.hpp>

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
                       Params<double, char, 256, 4, 4>,
                       Params<char, long long, 256, 4, 4>,
                       Params<float, int, 256, 4, 4>,
                       Params<test_utils::half, int, 256, 4, 4>,
                       Params<test_utils::bfloat16, int, 256, 4, 4>,

                       Params<int, int, 256, 8, 8>,
                       Params<double, char, 256, 8, 8>,
                       Params<char, long long, 256, 8, 8>,
                       Params<float, int, 256, 8, 8>,
                       Params<test_utils::half, int, 256, 8, 8>,
                       Params<test_utils::bfloat16, int, 256, 8, 8>,

                       Params<int, int, 256, 1, 14>,
                       Params<double, char, 256, 1, 14>,
                       Params<char, long long, 256, 1, 14>,
                       Params<float, int, 256, 1, 14>,
                       Params<test_utils::half, int, 256, 1, 14>,
                       Params<test_utils::bfloat16, int, 256, 1, 14>,

                       Params<int, int, 256, 9, 7>,
                       Params<double, char, 256, 9, 7>,
                       Params<char, long long, 256, 9, 7>,
                       Params<float, int, 256, 9, 7>,
                       Params<test_utils::half, int, 256, 9, 7>,
                       Params<test_utils::bfloat16, int, 256, 9, 7>>;

TYPED_TEST_SUITE(HipcubBlockRunLengthDecodeTest, HipcubBlockRunLengthDecodeTestParams);

template<class ItemT,
         class LengthT,
         unsigned BlockSize,
         unsigned RunsPerThread,
         unsigned DecodedItemsPerThread>
__global__
__launch_bounds__(BlockSize)
void block_run_length_decode_kernel(const ItemT*   d_run_items,
                                    const LengthT* d_run_lengths,
                                    ItemT*         d_decoded_items)
{
    using BlockRunLengthDecodeT
        = hipcub::BlockRunLengthDecode<ItemT, BlockSize, RunsPerThread, DecodedItemsPerThread>;
    static constexpr unsigned int decoded_items_per_block = BlockSize * DecodedItemsPerThread;
    __shared__ typename BlockRunLengthDecodeT::TempStorage temp_storage;

    ItemT   run_items[RunsPerThread];
    LengthT run_lengths[RunsPerThread];

    const unsigned global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;
    hipcub::LoadDirectBlocked(global_thread_idx, d_run_items, run_items);
    hipcub::LoadDirectBlocked(global_thread_idx, d_run_lengths, run_lengths);

    unsigned              total_decoded_size{};
    BlockRunLengthDecodeT block_run_length_decode(temp_storage,
                                                  run_items,
                                                  run_lengths,
                                                  total_decoded_size);

    unsigned decoded_window_offset = 0;
    while(decoded_window_offset < total_decoded_size)
    {
        ItemT decoded_items[DecodedItemsPerThread];

        block_run_length_decode.RunLengthDecode(decoded_items, decoded_window_offset);
        hipcub::StoreDirectBlocked(
            global_thread_idx,
            d_decoded_items + decoded_window_offset,
            decoded_items,
            hipcub::Min{}(total_decoded_size - decoded_window_offset, decoded_items_per_block));

        decoded_window_offset += decoded_items_per_block;
    }
}

TYPED_TEST(HipcubBlockRunLengthDecodeTest, TestDecode)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using ItemT                                 = typename TestFixture::params::item_type;
    using LengthT                               = typename TestFixture::params::length_type;
    constexpr unsigned block_size               = TestFixture::params::block_size;
    constexpr unsigned runs_per_thread          = TestFixture::params::runs_per_thread;
    constexpr unsigned decoded_items_per_thread = TestFixture::params::decoded_items_per_thread;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const LengthT max_run_length = static_cast<LengthT>(
            std::min(1000ll, static_cast<long long>(std::numeric_limits<LengthT>::max())));

        size_t num_runs    = runs_per_thread * block_size;
        auto   run_items   = test_utils::get_random_data<ItemT>(num_runs,
                                                            std::numeric_limits<ItemT>::min(),
                                                            std::numeric_limits<ItemT>::max(),
                                                            seed_value);
        auto   run_lengths = test_utils::get_random_data<LengthT>(num_runs,
                                                                static_cast<LengthT>(1),
                                                                max_run_length,
                                                                seed_value);

        std::default_random_engine            prng(seed_value);
        std::uniform_int_distribution<size_t> num_empty_runs_dist(1, 4);
        const size_t                          num_trailing_empty_runs = num_empty_runs_dist(prng);
        num_runs += num_trailing_empty_runs;

        const auto empty_run_items
            = test_utils::get_random_data<ItemT>(num_trailing_empty_runs,
                                                 std::numeric_limits<ItemT>::min(),
                                                 std::numeric_limits<ItemT>::max(),
                                                 seed_value);
        // Not strictly required, but fixes a spurious GCC warning and good practice anyways
        run_items.reserve(run_items.size() + empty_run_items.size());
        run_items.insert(run_items.end(), empty_run_items.begin(), empty_run_items.end());
        run_lengths.reserve(run_lengths.size() + num_trailing_empty_runs);
        run_lengths.insert(run_lengths.end(), num_trailing_empty_runs, static_cast<LengthT>(0));

        std::vector<ItemT> expected;
        for(size_t i = 0; i < run_items.size(); ++i)
        {
            for(size_t j = 0; j < static_cast<size_t>(run_lengths[i]); ++j)
            {
                expected.push_back(run_items[i]);
            }
        }

        ItemT* d_run_items{};
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_run_items, run_items.size() * sizeof(ItemT)));
        HIP_CHECK(hipMemcpy(d_run_items,
                            run_items.data(),
                            run_items.size() * sizeof(ItemT),
                            hipMemcpyHostToDevice));

        LengthT* d_run_lengths{};
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_run_lengths,
                                                     run_lengths.size() * sizeof(LengthT)));
        HIP_CHECK(hipMemcpy(d_run_lengths,
                            run_lengths.data(),
                            run_lengths.size() * sizeof(LengthT),
                            hipMemcpyHostToDevice));

        ItemT* d_decoded_runs{};
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_decoded_runs, expected.size() * sizeof(ItemT)));

        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(block_run_length_decode_kernel<ItemT,
                                                           LengthT,
                                                           block_size,
                                                           runs_per_thread,
                                                           decoded_items_per_thread>),
            dim3(1),
            dim3(block_size),
            0,
            0,
            d_run_items,
            d_run_lengths,
            d_decoded_runs);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<ItemT> output(expected.size());
        HIP_CHECK(hipMemcpy(output.data(),
                            d_decoded_runs,
                            output.size() * sizeof(ItemT),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_run_items));
        HIP_CHECK(hipFree(d_run_lengths));
        HIP_CHECK(hipFree(d_decoded_runs));

        for(size_t i = 0; i < output.size(); ++i)
        {
            ASSERT_EQ(test_utils::convert_to_native(output[i]),
                      test_utils::convert_to_native(expected[i]));
        }
    }
}
