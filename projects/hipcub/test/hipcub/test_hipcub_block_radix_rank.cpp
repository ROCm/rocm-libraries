
/******************************************************************************
* Copyright (c) 2011, Duane Merrill.  All rights reserved.
* Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
* Modifications Copyright (c) 2021-2025, Advanced Micro Devices, Inc.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#include "common_test_header.hpp"

// hipcub API
#include "hipcub/block/block_exchange.hpp"
#include "hipcub/block/block_load.hpp"
#include "hipcub/block/block_radix_rank.hpp"
#include "hipcub/block/block_store.hpp"
#include "hipcub/util_type.hpp"

template<class Key,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         Descending   = false,
         unsigned int StartBit     = 0,
         unsigned int MaxRadixBits = 4,
         unsigned int RadixBits    = MaxRadixBits>
struct params
{
    using key_type                                 = Key;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool         descending       = Descending;
    static constexpr unsigned int start_bit = StartBit;
    static constexpr unsigned int max_radix_bits   = MaxRadixBits;
    static constexpr unsigned int radix_bits       = RadixBits;
};

template<class Params>
class HipcubBlockRadixRank : public ::testing::Test
{
public:
    using params = Params;
};

using Params = ::testing::Types<
    // Power of 2 BlockSize
    params<unsigned int, 64U, 1>,
    params<test_utils::half, 128U, 1>,
    params<test_utils::bfloat16, 128U, 1>,
    params<float, 256U, 1>,
    params<unsigned short, 512U, 1, true>,

    // Non-power of 2 BlockSize
    params<double, 65U, 1>,
    params<float, 37U, 1>,
    params<long long, 510U, 1, true>,
    params<unsigned int, 162U, 1, false>,
    params<unsigned char, 255U, 1>,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<unsigned long long, 64U, 2, true>,
    params<int, 128U, 4>,
    params<unsigned short, 256U, 7>,
    params<float, 512U, 2, false>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1
    params<double, 33U, 5>,
    params<char, 464U, 2, true>,
    params<unsigned short, 100U, 3>,
    params<test_utils::half, 234U, 9>,
    params<test_utils::bfloat16, 234U, 9>,

    // StartBit and MaxRadixBits
    params<unsigned long long, 64U, 1, false, 8, 5>,
    params<unsigned short, 102U, 3, true, 4, 3>,
    params<float, 60U, 1, true, 8, 3>,

    // RadixBits < MaxRadixBits
    params<unsigned int, 162U, 2, true, 3, 6, 2>,
    params<test_utils::half, 193U, 2, true, 1, 4, 3>,
    params<test_utils::bfloat16, 193U, 2, true, 1, 4, 3>>;

TYPED_TEST_SUITE(HipcubBlockRadixRank, Params);

enum class RadixRankAlgorithm
{
    RADIX_RANK_BASIC,
    RADIX_RANK_MEMOIZE,
    RADIX_RANK_MATCH,
};

template<unsigned int       BlockSize,
         unsigned int       ItemsPerThread,
         unsigned int       MaxRadixBits,
         bool               Descending,
         RadixRankAlgorithm Algorithm,
         typename KeyType>
__global__ __launch_bounds__(BlockSize) void rank_kernel(const KeyType* keys_input,
                                                         int*           ranks_output,
                                                         unsigned int   start_bit,
                                                         unsigned int   radix_bits)
{
    constexpr bool warp_striped = Algorithm == RadixRankAlgorithm::RADIX_RANK_MATCH;

    using KeyTraits      = hipcub::Traits<KeyType>;
    using UnsignedBits   = typename KeyTraits::UnsignedBits;
    using DigitExtractor = hipcub::BFEDigitExtractor<KeyType>;
    using RankType       = std::conditional_t<
        Algorithm == RadixRankAlgorithm::RADIX_RANK_MATCH,
        hipcub::BlockRadixRankMatch<BlockSize, MaxRadixBits, Descending>,
        hipcub::BlockRadixRank<BlockSize,
                               MaxRadixBits,
                               Descending,
                               Algorithm == RadixRankAlgorithm::RADIX_RANK_MEMOIZE>>;

    using KeyExchangeType  = hipcub::BlockExchange<KeyType, BlockSize, ItemsPerThread>;
    using RankExchangeType = hipcub::BlockExchange<int, BlockSize, ItemsPerThread>;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     lid             = hipThreadIdx_x;
    const unsigned int     block_offset    = hipBlockIdx_x * items_per_block;

    __shared__ union
    {
        typename KeyExchangeType::TempStorage  key_exchange;
        typename RankType::TempStorage         rank;
        typename RankExchangeType::TempStorage rank_exchange;
    } storage;

    KeyType keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, keys_input + block_offset, keys);

    if(warp_striped)
    {
        KeyExchangeType exchange(storage.key_exchange);
        exchange.BlockedToWarpStriped(keys, keys);
        __syncthreads();
    }

    UnsignedBits(&unsigned_keys)[ItemsPerThread]
        = reinterpret_cast<UnsignedBits(&)[ItemsPerThread]>(keys);

#pragma unroll
    for(unsigned int key = 0; key < ItemsPerThread; key++)
    {
        unsigned_keys[key] = KeyTraits::TwiddleIn(unsigned_keys[key]);
    }

    RankType             rank(storage.rank);
    const DigitExtractor digit_extractor(start_bit, radix_bits);
    int                  ranks[ItemsPerThread];

    rank.RankKeys(unsigned_keys, ranks, digit_extractor);

    if(warp_striped)
    {
        __syncthreads();
        RankExchangeType exchange(storage.rank_exchange);
        exchange.WarpStripedToBlocked(ranks, ranks);
    }

    hipcub::StoreDirectBlocked(lid, ranks_output + block_offset, ranks);
}

template<typename TestFixture, RadixRankAlgorithm Algorithm>
void test_radix_rank()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::params::key_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int max_radix_bits   = TestFixture::params::max_radix_bits;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int radix_bits       = TestFixture::params::radix_bits;
    constexpr unsigned     end_bit          = start_bit + radix_bits;
    constexpr size_t items_per_block = block_size * items_per_thread;

    static_assert(radix_bits <= max_radix_bits,
                  "radix_bits must be less than or equal to max_radix_bits");

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t grid_size = 42;
    const size_t size = items_per_block * grid_size;

    SCOPED_TRACE(testing::Message() << "with items_per_block= " << items_per_block << " size=" << size);

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> keys_input;
        if(test_utils::is_floating_point<key_type>::value)
        {
            keys_input = test_utils::get_random_data<key_type>(
                size,
                test_utils::convert_to_device<key_type>(-1000),
                test_utils::convert_to_device<key_type>(+1000),
                seed_value);
        }
        else
        {
            keys_input
                = test_utils::get_random_data<key_type>(size,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value);
        }

        test_utils::add_special_values(keys_input, seed_value);

        // Calculate expected results on host
        std::vector<int> expected(keys_input.size());
        for(size_t i = 0; i < grid_size; i++)
        {
            size_t     block_offset = i * items_per_block;
            const auto key_cmp
                = test_utils::key_comparator<key_type, descending, start_bit, end_bit>();

            // Perform an 'argsort', which gives a sorted sequence of indices into `keys_input`.
            std::vector<int> indices(items_per_block);
            std::iota(indices.begin(), indices.end(), 0);
            std::stable_sort(
                indices.begin(),
                indices.end(),
                [&](const int& i, const int& j)
                { return key_cmp(keys_input[block_offset + i], keys_input[block_offset + j]); });

            // Invert the sorted indices sequence to obtain the ranks.
            for(size_t j = 0; j < indices.size(); ++j)
            {
                expected[block_offset + indices[j]] = static_cast<int>(j);
            }
        }

        // Preparing device
        key_type* d_keys_input;
        int*      d_ranks_output;
        HIP_CHECK(hipMalloc(&d_keys_input, keys_input.size() * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_ranks_output, expected.size() * sizeof(int)));

        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            keys_input.size() * sizeof(key_type),
                            hipMemcpyHostToDevice));

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(rank_kernel<block_size,
                                                       items_per_thread,
                                                       max_radix_bits,
                                                       descending,
                                                       Algorithm,
                                                       key_type>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           d_keys_input,
                           d_ranks_output,
                           start_bit,
                           radix_bits);

        // Getting results to host
        std::vector<int> ranks_output(expected.size());
        HIP_CHECK(hipMemcpy(ranks_output.data(),
                            d_ranks_output,
                            ranks_output.size() * sizeof(int),
                            hipMemcpyDeviceToHost));

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            SCOPED_TRACE(testing::Message() << "with index= " << i);
            ASSERT_EQ(ranks_output[i], expected[i]);
        }

        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_ranks_output));
    }
}

TYPED_TEST(HipcubBlockRadixRank, BlockRadixRankBasic)
{
    test_radix_rank<TestFixture, RadixRankAlgorithm::RADIX_RANK_BASIC>();
}

TYPED_TEST(HipcubBlockRadixRank, BlockRadixRankMemoize)
{
    test_radix_rank<TestFixture, RadixRankAlgorithm::RADIX_RANK_MEMOIZE>();
}

TYPED_TEST(HipcubBlockRadixRank, BlockRadixRankMatch)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    constexpr unsigned int block_size = TestFixture::params::block_size;
    if(block_size % HIPCUB_DEVICE_WARP_THREADS != 0)
    {
        // The CUB implementation of BlockRadixRankMatch is currently broken when
        // the warp size does not divide the block size exactly, see
        // https://github.com/NVIDIA/cub/issues/552.
        GTEST_SKIP();
    }
#endif

    test_radix_rank<TestFixture, RadixRankAlgorithm::RADIX_RANK_MATCH>();
}
