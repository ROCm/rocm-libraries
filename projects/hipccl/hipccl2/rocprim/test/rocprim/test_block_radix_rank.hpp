// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_BLOCK_RADIX_RANK_HPP_
#define TEST_BLOCK_RADIX_RANK_HPP_

#include "../common_test_header.hpp"

#include "../../common/utils_data_generation.hpp"

#include "../../common/utils_device_ptr.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_comparator.hpp"

#include <rocprim/block/block_exchange.hpp>
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_radix_rank.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/intrinsics/thread.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

template<class Params>
class RocprimBlockRadixRank : public ::testing::Test
{
public:
    using params = Params;
};

TYPED_TEST_SUITE_P(RocprimBlockRadixRank);

static constexpr size_t       n_sizes                   = 12;
static constexpr unsigned int items_per_thread[n_sizes] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
static constexpr unsigned int rank_desc[n_sizes]
    = {false, false, false, false, false, false, true, true, true, true, true, true};
static constexpr unsigned int pass_start_bit[n_sizes]  = {0, 0, 0, 6, 2, 1, 0, 0, 0, 1, 4, 7};
static constexpr unsigned int max_radix_bits[n_sizes]  = {4, 3, 5, 3, 1, 5, 4, 2, 4, 3, 1, 2};
static constexpr unsigned int pass_radix_bits[n_sizes] = {0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 1};

template<typename T,
         unsigned int                        BlockSize,
         unsigned int                        ItemsPerThread,
         unsigned int                        MaxRadixBits,
         rocprim::block_radix_rank_algorithm Algorithm>
__global__ __launch_bounds__(BlockSize)
void rank_kernel(const T* const      items_input,
                 unsigned int* const ranks_output,
                 const bool          descending,
                 const unsigned int  start_bit,
                 const unsigned int  radix_bits)
{
    using block_rank_type     = rocprim::block_radix_rank<BlockSize, MaxRadixBits, Algorithm>;
    using keys_exchange_type  = rocprim::block_exchange<T, BlockSize, ItemsPerThread>;
    using ranks_exchange_type = rocprim::block_exchange<unsigned int, BlockSize, ItemsPerThread>;

    constexpr bool warp_striped = Algorithm == rocprim::block_radix_rank_algorithm::match;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     lid             = threadIdx.x;
    const unsigned int     block_offset    = blockIdx.x * items_per_block;

    ROCPRIM_SHARED_MEMORY union
    {
        typename keys_exchange_type::storage_type  keys_exchange;
        typename block_rank_type::storage_type     rank;
        typename ranks_exchange_type::storage_type ranks_exchange;
    } storage;

    T            keys[ItemsPerThread];
    unsigned int ranks[ItemsPerThread];

    rocprim::block_load_direct_blocked(lid, items_input + block_offset, keys);
    if constexpr(warp_striped)
    {
        // block_radix_rank_match requires warp striped input and output. Instead of using
        // rocprim::block_load_direct_warp_striped though, we load directly and exchange the
        // values manually, as we can also test with block sizes that do not divide the hardware
        // warp size that way.
        keys_exchange_type().blocked_to_warp_striped(keys, keys, storage.keys_exchange);
        rocprim::syncthreads();
    }

    if(descending)
    {
        block_rank_type().rank_keys_desc(keys, ranks, storage.rank, start_bit, radix_bits);
    }
    else
    {
        block_rank_type().rank_keys(keys, ranks, storage.rank, start_bit, radix_bits);
    }

    if constexpr(warp_striped)
    {
        // See the comment above.
        rocprim::syncthreads();
        ranks_exchange_type().warp_striped_to_blocked(ranks, ranks, storage.ranks_exchange);
    }
    rocprim::block_store_direct_blocked(lid, ranks_output + block_offset, ranks);
}

template<typename T,
         unsigned int                        BlockSize,
         unsigned int                        ItemsPerThread,
         unsigned int                        StartBit,
         unsigned int                        MaxRadixBits,
         unsigned int                        RadixBits,
         bool                                Descending,
         rocprim::block_radix_rank_algorithm Algorithm>
void test_block_radix_rank()
{
    constexpr size_t                              block_size       = BlockSize;
    constexpr size_t                              items_per_thread = ItemsPerThread;
    constexpr size_t                              items_per_block  = block_size * items_per_thread;
    constexpr size_t                              start_bit        = StartBit;
    constexpr size_t                              max_radix_bits   = MaxRadixBits;
    constexpr size_t                              radix_bits       = RadixBits;
    constexpr size_t                              end_bit          = start_bit + radix_bits;
    constexpr bool                                descending       = Descending;
    constexpr rocprim::block_radix_rank_algorithm algorithm        = Algorithm;

    const size_t grid_size = 23;
    const size_t size      = items_per_block * grid_size;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size);
    SCOPED_TRACE(testing::Message() << "with items_per_thread = " << items_per_thread);
    SCOPED_TRACE(testing::Message() << "with descending = " << (descending ? "true" : "false"));
    SCOPED_TRACE(testing::Message() << "with start_bit = " << start_bit);
    SCOPED_TRACE(testing::Message() << "with max_radix_bits = " << MaxRadixBits);
    SCOPED_TRACE(testing::Message() << "with radix_bits = " << radix_bits);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << size);
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    for(size_t seed_index = 0; seed_index < number_of_runs; ++seed_index)
    {
        seed_type seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> keys_input
            = test_utils::get_random_data_wrapped<T>(size,
                                                     common::generate_limits<T>::min(),
                                                     common::generate_limits<T>::max(),
                                                     seed_value);

        // Calculated expected results on host
        std::vector<unsigned int> expected(size);
        for(size_t i = 0; i < grid_size; ++i)
        {
            size_t     block_offset = i * items_per_block;
            const auto key_cmp = test_utils::key_comparator<T, descending, start_bit, end_bit>();

            // Perform an 'argsort', which gives a sorted sequence of indices into `keys_input`.
            std::vector<int> indices(items_per_block);
            std::iota(indices.begin(), indices.end(), 0);
            std::stable_sort(
                indices.begin(),
                indices.end(),
                [&](const int& i, const int& j)
                { return key_cmp(keys_input[block_offset + i], keys_input[block_offset + j]); });

            // Invert the sorted indices sequence to obtain the ranks.
            for(size_t j = 0; j < items_per_block; ++j)
            {
                expected[block_offset + indices[j]] = static_cast<int>(j);
            }
        }

        common::device_ptr<T>            d_keys_input(keys_input);
        common::device_ptr<unsigned int> d_ranks_output(size);

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                rank_kernel<T, block_size, items_per_thread, max_radix_bits, algorithm>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            d_keys_input.get(),
            d_ranks_output.get(),
            descending,
            start_bit,
            radix_bits);
        HIP_CHECK(hipGetLastError());

        // Getting results to host
        auto ranks_output = d_ranks_output.load();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(ranks_output, expected));
    }
}

template<unsigned int First,
         unsigned int Last,
         typename T,
         unsigned int                        BlockSize,
         rocprim::block_radix_rank_algorithm Algorithm>
struct static_for
{
    static constexpr unsigned int radix_bits
        = pass_radix_bits[First] == 0 ? max_radix_bits[First] : pass_radix_bits[First];

    static void run()
    {
        {
            SCOPED_TRACE(testing::Message() << "TestID = " << First);
            test_block_radix_rank<T,
                                  BlockSize,
                                  items_per_thread[First],
                                  pass_start_bit[First],
                                  max_radix_bits[First],
                                  radix_bits,
                                  rank_desc[First],
                                  Algorithm>();
        }
        static_for<First + 1, Last, T, BlockSize, Algorithm>::run();
    }
};

template<unsigned int Last,
         typename T,
         unsigned int                        BlockSize,
         rocprim::block_radix_rank_algorithm Algorithm>
struct static_for<Last, Last, T, BlockSize, Algorithm>
{
    static void run() {}
};

template<rocprim::block_radix_rank_algorithm Algorithm, typename TestFixture>
void test_block_radix_rank_algorithm()
{
    using type                  = typename TestFixture::params::input_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    if(block_size > test_utils::get_max_block_size())
    {
        GTEST_SKIP();
    }

    static_for<0, n_sizes, type, block_size, Algorithm>::run();
}

#endif // TEST_BLOCK_RADIX_RANK_KERNELS_HPP_
