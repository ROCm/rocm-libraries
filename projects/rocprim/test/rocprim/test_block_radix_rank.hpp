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
static constexpr unsigned int use_storage[n_sizes]
    = {false, true, false, true, false, true, false, true, false, true, false, true};
static constexpr unsigned int end_bits[n_sizes]
        = {0x1, 0x3, 0x7, 0xf, 0x1, 0x3, 0x7, 0xf, 0x1, 0x3, 0x7, 0xf};
static constexpr unsigned int pass_start_bit[n_sizes]  = {0, 0, 0, 6, 2, 1, 0, 0, 0, 1, 4, 7};
static constexpr unsigned int max_radix_bits[n_sizes]  = {4, 3, 5, 3, 1, 5, 4, 2, 4, 3, 1, 2};
static constexpr unsigned int max_radix_bits_extractor[n_sizes]  = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
static constexpr unsigned int pass_radix_bits[n_sizes] = {0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 1};

template<typename T,
         unsigned int                        BlockSize,
         unsigned int                        ItemsPerThread,
         unsigned int                        MaxRadixBits,
         rocprim::block_radix_rank_algorithm Algorithm>
__global__ __launch_bounds__(BlockSize) void rank_kernel(const T* const      items_input,
                                                         unsigned int* const ranks_output,
                                                         const bool          descending,
                                                         const bool          use_storage,
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
    if ROCPRIM_IF_CONSTEXPR(warp_striped)
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
        if (use_storage)
            block_rank_type().rank_keys_desc(keys, ranks, storage.rank, start_bit, radix_bits);
        else
            block_rank_type().rank_keys_desc(keys, ranks, start_bit, radix_bits);
    }
    else
    {
        if (use_storage)
            block_rank_type().rank_keys(keys, ranks, storage.rank, start_bit, radix_bits);
        else
            block_rank_type().rank_keys(keys, ranks, start_bit, radix_bits);
    }

    if ROCPRIM_IF_CONSTEXPR(warp_striped)
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
         unsigned int                        MaxRadixBits,
         rocprim::block_radix_rank_algorithm Algorithm>
__global__ __launch_bounds__(BlockSize) void rank_kernel(const T* const      items_input,
                                                         unsigned int* const ranks_output,
                                                         const bool          descending,
                                                         const bool          use_storage,
                                                         const unsigned int  last_bits)
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
    if ROCPRIM_IF_CONSTEXPR(warp_striped)
    {
        // block_radix_rank_match requires warp striped input and output. Instead of using
        // rocprim::block_load_direct_warp_striped though, we load directly and exchange the
        // values manually, as we can also test with block sizes that do not divide the hardware
        // warp size that way.
        keys_exchange_type().blocked_to_warp_striped(keys, keys, storage.keys_exchange);
        rocprim::syncthreads();
    }

    union converter{
        T in;
        uint64_t out;
    };

    if(descending)
    {
        if (use_storage)
            block_rank_type().rank_keys_desc(keys, ranks, storage.rank, [=](const T & key){
                converter c;
                c.in = key;
                uint64_t out = c.out & last_bits;
                return out;
            });
        else
            block_rank_type().rank_keys_desc(keys, ranks, [=](const T & key){
                converter c;
                c.in = key;
                uint64_t out = c.out & last_bits;
                return out;
            });
    }
    else
    {
        if (use_storage)
            block_rank_type().rank_keys(keys, ranks, storage.rank, [=](const T & key){
                converter c;
                c.in = key;
                uint64_t out = c.out & last_bits;
                return out;
            });
        else
            block_rank_type().rank_keys(keys, ranks, [=](const T & key){
                converter c;
                c.in = key;
                uint64_t out = c.out & last_bits;
                return out;
            });
    }

    if ROCPRIM_IF_CONSTEXPR(warp_striped)
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
         unsigned int                        RadixBits,
         unsigned int                        MaxRadixBits,
         rocprim::block_radix_rank_algorithm Algorithm,
         class Extractor>
__global__ __launch_bounds__(BlockSize) void rank_kernel(const T* const      items_input,
                                                         unsigned int* const ranks_output,
                                                         unsigned int*       prefix_output,
                                                         unsigned int*       counts_output,
                                                         Extractor digit_extractor
                                                        )
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

    const unsigned int digits_per_thread = block_rank_type().digits_per_thread;

    unsigned int prefix[digits_per_thread];
    unsigned int counts[digits_per_thread];
    rocprim::block_load_direct_blocked(lid, items_input + block_offset, keys);
    if ROCPRIM_IF_CONSTEXPR(warp_striped)
    {
        // block_radix_rank_match requires warp striped input and output. Instead of using
        // rocprim::block_load_direct_warp_striped though, we load directly and exchange the
        // values manually, as we can also test with block sizes that do not divide the hardware
        // warp size that way.
        keys_exchange_type().blocked_to_warp_striped(keys, keys, storage.keys_exchange);
        rocprim::syncthreads();
    }

    block_rank_type().rank_keys(keys, ranks, storage.rank, digit_extractor, prefix, counts);

    if ROCPRIM_IF_CONSTEXPR(warp_striped)
    {
        // See the comment above.
        rocprim::syncthreads();
        ranks_exchange_type().warp_striped_to_blocked(ranks, ranks, storage.ranks_exchange);
    }
    rocprim::block_store_direct_blocked(lid, ranks_output + block_offset, ranks);

    // storing count and prefix output
    const size_t pc_offset = (threadIdx.x * digits_per_thread) + (blockIdx.x * (1 << RadixBits));

    for(size_t i = 0; i < digits_per_thread; i++){
        if((threadIdx.x * digits_per_thread) + i < (1 << RadixBits)){
            prefix_output[pc_offset + i] = prefix[i];
            counts_output[pc_offset + i] = counts[i];
        }
    }
}

template<typename T,
         unsigned int                        BlockSize,
         unsigned int                        ItemsPerThread,
         unsigned int                        StartBit,
         unsigned int                        MaxRadixBits,
         unsigned int                        RadixBits,
         bool                                Descending,
         bool                                UseStorage,
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
    constexpr bool                                use_storage      = UseStorage;
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
            use_storage,
            start_bit,
            radix_bits);
        HIP_CHECK(hipGetLastError());

        // Getting results to host
        auto ranks_output = d_ranks_output.load();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(ranks_output, expected));
    }
}

template<typename T,
         unsigned int                        BlockSize,
         unsigned int                        ItemsPerThread,
         unsigned int                        MaxRadixBits,
         unsigned int                        EndBits,
         bool                                Descending,
         bool                                UseStorage,
         rocprim::block_radix_rank_algorithm Algorithm>
void test_block_radix_extractor_rank()
{
    constexpr size_t                              block_size       = BlockSize;
    constexpr size_t                              items_per_thread = ItemsPerThread;
    constexpr size_t                              items_per_block  = block_size * items_per_thread;
    constexpr size_t                              max_radix_bits   = MaxRadixBits;
    constexpr size_t                              end_bits         = EndBits;
    constexpr bool                                descending       = Descending;
    constexpr bool                                use_storage      = UseStorage;
    constexpr rocprim::block_radix_rank_algorithm algorithm        = Algorithm;

    const size_t grid_size = 2;
    const size_t size      = items_per_block * grid_size;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size);
    SCOPED_TRACE(testing::Message() << "with items_per_thread = " << items_per_thread);
    SCOPED_TRACE(testing::Message() << "with descending = " << (descending ? "true" : "false"));
    SCOPED_TRACE(testing::Message() << "with max_radix_bits = " << MaxRadixBits);
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


        union converter{
            T in;
            uint64_t out;
        };
        // Calculated expected results on host
        std::vector<unsigned int> expected(size);
        for(size_t i = 0; i < grid_size; ++i)
        {
            size_t     block_offset = i * items_per_block;

            // Perform an 'argsort', which gives a sorted sequence of indices into `keys_input`.
            std::vector<int> indices(items_per_block);
            std::iota(indices.begin(), indices.end(), 0);

            std::stable_sort(
                indices.begin(),
                indices.end(),
                [&](const int& i, const int& j)
                { 
                    converter c;
                    c.in = keys_input[block_offset + i];
                    uint64_t left = c.out & end_bits;

                    c.in = keys_input[block_offset + j];

                    uint64_t right = c.out & end_bits;

                    return descending ? right < left : left < right; 
                });

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
            use_storage,
            end_bits);
        HIP_CHECK(hipGetLastError());

        // Getting results to host
        auto ranks_output = d_ranks_output.load();

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(ranks_output, expected));
    }
}

template<typename T,
         unsigned int                        BlockSize,
         unsigned int                        ItemsPerThread,
         unsigned int                        MaxRadixBits,
         unsigned int                        RadixBits,
         bool                                Descending,
         rocprim::block_radix_rank_algorithm Algorithm>
void test_block_radix_rank_with_prefix_and_count()
{
    constexpr size_t                              block_size       = BlockSize;
    constexpr size_t                              items_per_thread = ItemsPerThread;
    constexpr size_t                              items_per_block  = block_size * items_per_thread;
    constexpr size_t                              max_radix_bits   = MaxRadixBits;
    constexpr size_t                              radix_bits       = RadixBits;
    constexpr rocprim::block_radix_rank_algorithm algorithm        = Algorithm;
    
    const size_t grid_size = 2;
    const size_t size      = items_per_block * grid_size;
    const size_t pc_items_per_block = (1 << radix_bits);
    const uint64_t end_bits = ((pc_items_per_block) - 1);
    const size_t pc_size = (pc_items_per_block) * grid_size;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size);
    SCOPED_TRACE(testing::Message() << "with items_per_thread = " << items_per_thread);
    SCOPED_TRACE(testing::Message() << "with max_radix_bits = " << MaxRadixBits);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);
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


        union converter{
            T in;
            uint64_t out;
        } c;
        // Calculated expected results on host
        std::vector<unsigned int> expected(size);
        std::vector<unsigned int> expected_histogram(pc_size, 0);
        std::vector<unsigned int> expected_prefix(pc_size, 0);
        for(size_t i = 0; i < grid_size; ++i)
        {
            size_t     block_offset = i * items_per_block;

            // Perform an 'argsort', which gives a sorted sequence of indices into `keys_input`.
            std::vector<int> indices(items_per_block);
            std::iota(indices.begin(), indices.end(), 0);

            std::stable_sort(
                indices.begin(),
                indices.end(),
                [&](const int& i, const int& j)
                { 
                    c.in = keys_input[block_offset + i];
                    uint64_t left = c.out & end_bits;

                    c.in = keys_input[block_offset + j];

                    uint64_t right = c.out & end_bits;

                    return Descending ? right < left : left < right; 
                });

            // Invert the sorted indices sequence to obtain the ranks.
            for(size_t j = 0; j < items_per_block; ++j)
            {
                expected[block_offset + indices[j]] = static_cast<int>(j);
            }

            size_t pc_block_offset = i * (pc_items_per_block);
            for(size_t j = 0; j < items_per_block; j++){
                c.in = keys_input[block_offset + j];
                uint64_t bit_rep = c.out;
                bit_rep &= end_bits;

                if(Descending)
                    bit_rep = pc_items_per_block - (1 + bit_rep);
                
                ++expected_histogram[bit_rep + pc_block_offset];
            }
            std::exclusive_scan(
                expected_histogram.begin() + pc_block_offset, 
                expected_histogram.begin() + pc_block_offset + pc_items_per_block, 
                expected_prefix.begin() + pc_block_offset, 
                0
            );
        }
        common::device_ptr<T>            d_keys_input(keys_input);
        common::device_ptr<unsigned int> d_ranks_output(size);
        common::device_ptr<unsigned int> d_prefix_output(pc_size);
        common::device_ptr<unsigned int> d_counts_output(pc_size);

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                rank_kernel<T, block_size, items_per_thread, radix_bits, max_radix_bits, algorithm>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            d_keys_input.get(),
            d_ranks_output.get(),
            d_prefix_output.get(),
            d_counts_output.get(),
            [] (const T & key){
                const uint64_t end_bits = ((pc_items_per_block) - 1);
                union converter{
                    T in;
                    uint64_t out;
                } c;
                c.in = key;

                uint64_t out = c.out & end_bits;

                if(Descending)
                    out = pc_items_per_block - (1 + out);

                return out;
            }
        );
        HIP_CHECK(hipGetLastError());

        // Getting results to host
        auto ranks_output = d_ranks_output.load();
        auto prefix_output = d_prefix_output.load();
        auto counts_output = d_counts_output.load();

        for(size_t i = 0; i < ranks_output.size(); i++){
            ASSERT_EQ(ranks_output[i], expected[i]) << "Index: " << i << std::endl;

            if(i < pc_size){
                ASSERT_EQ(prefix_output[i], expected_prefix[i]) << "Index: " << i << std::endl;
                ASSERT_EQ(counts_output[i], expected_histogram[i]) << "Index: " << i << std::endl;
            }
        }
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
                                  use_storage[First],
                                  Algorithm>();
        }
        static_for<First + 1, Last, T, BlockSize, Algorithm>::run();
    }

    static void run_extractor()
    {
        {
            SCOPED_TRACE(testing::Message() << "TestID = " << First);
            test_block_radix_extractor_rank<T,
                                  BlockSize,
                                  items_per_thread[First],
                                  max_radix_bits_extractor[First],
                                  end_bits[First],
                                  rank_desc[First],
                                  use_storage[First],
                                  Algorithm>();
        }
        static_for<First + 1, Last, T, BlockSize, Algorithm>::run_extractor();
    }

    static void run_extractor_with_prefix_count()
    {
        {
            SCOPED_TRACE(testing::Message() << "TestID = " << First);
            test_block_radix_rank_with_prefix_and_count<T,
                                  BlockSize,
                                  items_per_thread[First],
                                  max_radix_bits_extractor[First],
                                  max_radix_bits_extractor[First],
                                  rank_desc[First],
                                  Algorithm>();
        }
        static_for<First + 1, Last, T, BlockSize, Algorithm>::run_extractor_with_prefix_count();
    }
};

template<unsigned int Last,
         typename T,
         unsigned int                        BlockSize,
         rocprim::block_radix_rank_algorithm Algorithm>
struct static_for<Last, Last, T, BlockSize, Algorithm>
{
    static void run() {}
    static void run_extractor() {}
    static void run_extractor_with_prefix_count() {}
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

template<rocprim::block_radix_rank_algorithm Algorithm, typename TestFixture>
void test_block_radix_rank_extractor_algorithm()
{
    using type                  = typename TestFixture::params::input_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    if(block_size > test_utils::get_max_block_size())
    {
        GTEST_SKIP();
    }

    static_for<0, n_sizes, type, block_size, Algorithm>::run_extractor();
}

template<rocprim::block_radix_rank_algorithm Algorithm, typename TestFixture>
void test_block_radix_rank_extractor_with_prefix_count_algorithm()
{
    using type                  = typename TestFixture::params::input_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    if(block_size > test_utils::get_max_block_size())
    {
        GTEST_SKIP();
    }

    static_for<0, n_sizes, type, block_size, Algorithm>::run_extractor_with_prefix_count();
}

#endif // TEST_BLOCK_RADIX_RANK_KERNELS_HPP_
