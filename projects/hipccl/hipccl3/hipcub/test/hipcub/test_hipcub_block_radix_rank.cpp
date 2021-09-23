
/******************************************************************************
* Copyright (c) 2011, Duane Merrill.  All rights reserved.
* Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
* Modifications Copyright (c) 2021, Advanced Micro Devices, Inc.  All rights reserved.
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
#include "hipcub/util_type.hpp"
#include "hipcub/block/block_radix_sort.hpp"
#include "hipcub/block/block_load.hpp"
#include "hipcub/block/block_store.hpp"

#include "hipcub/block/block_exchange.hpp"
#include "hipcub/block/block_radix_rank.hpp"

#include "test_sort_comparator.hpp"

namespace hipcub_test {

template <
    typename                   KeyT,
    int                        BLOCK_DIM_X,
    int                        ITEMS_PER_THREAD,
    typename                   ValueT                  = hipcub::NullType,
    int                        RADIX_BITS              = 4,
    bool                       MEMOIZE_OUTER_SCAN      = false,
    hipcub::BlockScanAlgorithm INNER_SCAN_ALGORITHM    = hipcub::BLOCK_SCAN_WARP_SCANS,
    hipSharedMemConfig         SMEM_CONFIG             = hipSharedMemBankSizeFourByte,
    int                        BLOCK_DIM_Y             = 1,
    int                        BLOCK_DIM_Z             = 1,
    int                        ARCH                    = HIPCUB_ARCH>
class BlockRadixSort
{
private:

    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    enum
    {
        // The thread block size in threads
        BLOCK_THREADS               = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        // Whether or not there are values to be trucked along with keys
        #ifdef __HIP_PLATFORM_HCC__
        KEYS_ONLY                   = rocprim::Equals<ValueT, hipcub::NullType>::VALUE,
        #else
        KEYS_ONLY                   = cub::Equals<ValueT, hipcub::NullType>::VALUE,
        #endif
    };

    // KeyT traits and unsigned bits type
    typedef hipcub::Traits<KeyT>                KeyTraits;
    typedef typename KeyTraits::UnsignedBits    UnsignedBits;

    /// Ascending BlockRadixRank utility type
    typedef hipcub::BlockRadixRank<
            BLOCK_DIM_X,
            RADIX_BITS,
            false,
            MEMOIZE_OUTER_SCAN,
            INNER_SCAN_ALGORITHM,
            SMEM_CONFIG,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z,
            ARCH>
        AscendingBlockRadixRank;

    /// Descending BlockRadixRank utility type
    typedef hipcub::BlockRadixRank<
            BLOCK_DIM_X,
            RADIX_BITS,
            true,
            MEMOIZE_OUTER_SCAN,
            INNER_SCAN_ALGORITHM,
            SMEM_CONFIG,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z,
            ARCH>
        DescendingBlockRadixRank;

    /// BlockExchange utility type for keys
    typedef hipcub::BlockExchange<KeyT, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z, ARCH> BlockExchangeKeys;

    /// BlockExchange utility type for values
    typedef hipcub::BlockExchange<ValueT, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z, ARCH> BlockExchangeValues;

    /// Shared memory storage layout type
    union _TempStorage
    {
        typename AscendingBlockRadixRank::TempStorage  asending_ranking_storage;
        typename DescendingBlockRadixRank::TempStorage descending_ranking_storage;
        typename BlockExchangeKeys::TempStorage        exchange_keys;
        typename BlockExchangeValues::TempStorage      exchange_values;
    };


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;

    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }

    /// Rank keys (specialized for ascending sort)
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&unsigned_keys)[ITEMS_PER_THREAD],
        int             (&ranks)[ITEMS_PER_THREAD],
        int             begin_bit,
        int             pass_bits,
        hipcub::Int2Type<false> /*is_descending*/)
    {
        AscendingBlockRadixRank(temp_storage.asending_ranking_storage).RankKeys(
            unsigned_keys,
            ranks,
            begin_bit,
            pass_bits);
    }

    /// Rank keys (specialized for descending sort)
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&unsigned_keys)[ITEMS_PER_THREAD],
        int             (&ranks)[ITEMS_PER_THREAD],
        int             begin_bit,
        int             pass_bits,
        hipcub::Int2Type<true>  /*is_descending*/)
    {
        DescendingBlockRadixRank(temp_storage.descending_ranking_storage).RankKeys(
            unsigned_keys,
            ranks,
            begin_bit,
            pass_bits);
    }

    /// ExchangeValues (specialized for key-value sort, to-blocked arrangement)
    __device__ __forceinline__ void ExchangeValues(
        ValueT          (&values)[ITEMS_PER_THREAD],
        int             (&ranks)[ITEMS_PER_THREAD],
        hipcub::Int2Type<false> /*is_keys_only*/,
        hipcub::Int2Type<true>  /*is_blocked*/)
    {
        __syncthreads();

        // Exchange values through shared memory in blocked arrangement
        BlockExchangeValues(temp_storage.exchange_values).ScatterToBlocked(values, ranks);
    }

    /// ExchangeValues (specialized for key-value sort, to-striped arrangement)
    __device__ __forceinline__ void ExchangeValues(
        ValueT          (&values)[ITEMS_PER_THREAD],
        int             (&ranks)[ITEMS_PER_THREAD],
        hipcub::Int2Type<false> /*is_keys_only*/,
        hipcub::Int2Type<false> /*is_blocked*/)
    {
        __syncthreads();

        // Exchange values through shared memory in blocked arrangement
        BlockExchangeValues(temp_storage.exchange_values).ScatterToStriped(values, ranks);
    }

    /// ExchangeValues (specialized for keys-only sort)
    template <int IS_BLOCKED>
    __device__ __forceinline__ void ExchangeValues(
        ValueT                  (&/*values*/)[ITEMS_PER_THREAD],
        int                     (&/*ranks*/)[ITEMS_PER_THREAD],
        hipcub::Int2Type<true>          /*is_keys_only*/,
        hipcub::Int2Type<IS_BLOCKED>    /*is_blocked*/)
    {}

    /// Sort blocked arrangement
    template <int DESCENDING, int KEYS_ONLY>
    __device__ __forceinline__ void SortBlocked(
        KeyT                    (&keys)[ITEMS_PER_THREAD],          ///< Keys to sort
        ValueT                  (&values)[ITEMS_PER_THREAD],        ///< Values to sort
        int                     begin_bit,                          ///< The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                            ///< The past-the-end (most-significant) bit index needed for key comparison
        hipcub::Int2Type<DESCENDING>    is_descending,                      ///< Tag whether is a descending-order sort
        hipcub::Int2Type<KEYS_ONLY>     is_keys_only)                       ///< Tag whether is keys-only sort
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            int pass_bits = min(RADIX_BITS, end_bit - begin_bit);

            // Rank the blocked keys
            int ranks[ITEMS_PER_THREAD];
            RankKeys(unsigned_keys, ranks, begin_bit, pass_bits, is_descending);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Exchange keys through shared memory in blocked arrangement
            BlockExchangeKeys(temp_storage.exchange_keys).ScatterToBlocked(keys, ranks);

            // Exchange values through shared memory in blocked arrangement
            ExchangeValues(values, ranks, is_keys_only, hipcub::Int2Type<true>());

            // Quit if done
            if (begin_bit >= end_bit) break;

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }

public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /// Sort blocked -> striped arrangement
    template <int DESCENDING, int KEYS_ONLY>
    __device__ __forceinline__ void SortBlockedToStriped(
        KeyT                    (&keys)[ITEMS_PER_THREAD],          ///< Keys to sort
        ValueT                  (&values)[ITEMS_PER_THREAD],        ///< Values to sort
        int                     begin_bit,                          ///< The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                            ///< The past-the-end (most-significant) bit index needed for key comparison
        hipcub::Int2Type<DESCENDING>    is_descending,                      ///< Tag whether is a descending-order sort
        hipcub::Int2Type<KEYS_ONLY>     is_keys_only)                       ///< Tag whether is keys-only sort
    {
        UnsignedBits (&unsigned_keys)[ITEMS_PER_THREAD] =
            reinterpret_cast<UnsignedBits (&)[ITEMS_PER_THREAD]>(keys);

        // Twiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleIn(unsigned_keys[KEY]);
        }

        // Radix sorting passes
        while (true)
        {
            int pass_bits = min(RADIX_BITS, end_bit - begin_bit);

            // Rank the blocked keys
            int ranks[ITEMS_PER_THREAD];
            RankKeys(unsigned_keys, ranks, begin_bit, pass_bits, is_descending);
            begin_bit += RADIX_BITS;

            __syncthreads();

            // Check if this is the last pass
            if (begin_bit >= end_bit)
            {
                // Last pass exchanges keys through shared memory in striped arrangement
                BlockExchangeKeys(temp_storage.exchange_keys).ScatterToStriped(keys, ranks);

                // Last pass exchanges through shared memory in striped arrangement
                ExchangeValues(values, ranks, is_keys_only, hipcub::Int2Type<false>());

                // Quit
                break;
            }

            // Exchange keys through shared memory in blocked arrangement
            BlockExchangeKeys(temp_storage.exchange_keys).ScatterToBlocked(keys, ranks);

            // Exchange values through shared memory in blocked arrangement
            ExchangeValues(values, ranks, is_keys_only, hipcub::Int2Type<true>());

            __syncthreads();
        }

        // Untwiddle bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            unsigned_keys[KEY] = KeyTraits::TwiddleOut(unsigned_keys[KEY]);
        }
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// \smemstorage{BlockRadixSort}
    struct TempStorage : hipcub::Uninitialized<_TempStorage> {};

    __device__ __forceinline__ BlockRadixSort()
    :
        temp_storage(PrivateStorage()),
        linear_tid(hipcub::RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}

    __device__ __forceinline__ BlockRadixSort(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(hipcub::RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}

    __device__ __forceinline__ void Sort(
        KeyT    (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(KeyT) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        hipcub::NullType values[ITEMS_PER_THREAD];

        SortBlocked(keys, values, begin_bit, end_bit, hipcub::Int2Type<false>(), hipcub::Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ void Sort(
        KeyT    (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        ValueT  (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(KeyT) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        SortBlocked(keys, values, begin_bit, end_bit, hipcub::Int2Type<false>(), hipcub::Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ void SortDescending(
        KeyT    (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(KeyT) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        hipcub::NullType values[ITEMS_PER_THREAD];

        SortBlocked(keys, values, begin_bit, end_bit, hipcub::Int2Type<true>(), hipcub::Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ void SortDescending(
        KeyT    (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        ValueT  (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(KeyT) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        SortBlocked(keys, values, begin_bit, end_bit, hipcub::Int2Type<true>(), hipcub::Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ void SortBlockedToStriped(
        KeyT    (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(KeyT) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        hipcub::NullType values[ITEMS_PER_THREAD];

        SortBlockedToStriped(keys, values, begin_bit, end_bit, hipcub::Int2Type<false>(), hipcub::Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ void SortBlockedToStriped(
        KeyT    (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        ValueT  (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(KeyT) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        SortBlockedToStriped(keys, values, begin_bit, end_bit, hipcub::Int2Type<false>(), hipcub::Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ void SortDescendingBlockedToStriped(
        KeyT    (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(KeyT) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        hipcub::NullType values[ITEMS_PER_THREAD];

        SortBlockedToStriped(keys, values, begin_bit, end_bit, hipcub::Int2Type<true>(), hipcub::Int2Type<KEYS_ONLY>());
    }

    __device__ __forceinline__ void SortDescendingBlockedToStriped(
        KeyT    (&keys)[ITEMS_PER_THREAD],          ///< [in-out] Keys to sort
        ValueT  (&values)[ITEMS_PER_THREAD],        ///< [in-out] Values to sort
        int     begin_bit   = 0,                    ///< [in] <b>[optional]</b> The beginning (least-significant) bit index needed for key comparison
        int     end_bit     = sizeof(KeyT) * 8)      ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
    {
        SortBlockedToStriped(keys, values, begin_bit, end_bit, hipcub::Int2Type<true>(), hipcub::Int2Type<KEYS_ONLY>());
    }
};

}

template<
    class Key,
    class Value,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Descending = false,
    bool ToStriped = false,
    unsigned int StartBit = 0,
    unsigned int EndBit = sizeof(Key) * 8
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool descending = Descending;
    static constexpr bool to_striped = ToStriped;
    static constexpr unsigned int start_bit = StartBit;
    static constexpr unsigned int end_bit = EndBit;
};

template<class Params>
class HipcubBlockRadixSort : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    // Power of 2 BlockSize
    params<unsigned int, int, 64U, 1>,
    params<int, int, 128U, 1>,
    params<unsigned int, int, 256U, 1>,
    params<unsigned short, char, 1024U, 1, true>,

    // Non-power of 2 BlockSize
    params<double, unsigned int, 65U, 1>,
    params<float, int, 37U, 1>,
    params<long long, char, 510U, 1, true>,
    params<unsigned int, long long, 162U, 1, false, true>,
    params<unsigned char, float, 255U, 1>,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<float, char, 64U, 2, true>,
    params<int, short, 128U, 4>,
    params<unsigned short, char, 256U, 7>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1
    params<double, int, 33U, 5>,
    params<char, double, 464U, 2, true, true>,
    params<unsigned short, int, 100U, 3>,
    params<short, int, 234U, 9>,

    // StartBit and EndBit
    params<unsigned long long, char, 64U, 1, false, false, 8, 20>,
    params<unsigned short, int, 102U, 3, true, false, 4, 10>,
    params<unsigned int, short, 162U, 2, true, true, 3, 12>,

    // Stability (a number of key values is lower than BlockSize * ItemsPerThread: some keys appear
    // multiple times with different values or key parts outside [StartBit, EndBit))
    params<unsigned char, int, 512U, 2, false, true>,
    params<unsigned short, double, 60U, 1, true, false, 8, 11>
> Params;

TYPED_TEST_SUITE(HipcubBlockRadixSort, Params);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type
>
__global__
__launch_bounds__(BlockSize)
void sort_key_kernel(
    key_type* device_keys_output,
    bool to_striped,
    bool descending,
    unsigned int start_bit,
    unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    key_type keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_keys_output + block_offset, keys);

    hipcub_test::BlockRadixSort<key_type, BlockSize, ItemsPerThread> bsort;
    if(to_striped)
    {
        if(descending)
            bsort.SortDescendingBlockedToStriped(keys, start_bit, end_bit);
        else
            bsort.SortBlockedToStriped(keys, start_bit, end_bit);

        hipcub::StoreDirectStriped<BlockSize>(lid, device_keys_output + block_offset, keys);
    }
    else
    {
        if(descending)
            bsort.SortDescending(keys, start_bit, end_bit);
        else
            bsort.Sort(keys, start_bit, end_bit);

        hipcub::StoreDirectBlocked(lid, device_keys_output + block_offset, keys);
    }
}

TYPED_TEST(HipcubBlockRadixSort, SortKeys)
{
    using key_type = typename TestFixture::params::key_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool descending = TestFixture::params::descending;
    constexpr bool to_striped = TestFixture::params::to_striped;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr size_t items_per_block = block_size * items_per_thread;
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
        std::vector<key_type> keys_output;
        if(std::is_floating_point<key_type>::value)
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                (key_type)-1000,
                (key_type)+1000,
                seed_value
            );
        }
        else
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                std::numeric_limits<key_type>::min(),
                std::numeric_limits<key_type>::max(),
                seed_value
            );
        }

        // Calculate expected results on host
        std::vector<key_type> expected(keys_output);
        for(size_t i = 0; i < grid_size; i++)
        {
            std::stable_sort(
                expected.begin() + (i * items_per_block),
                expected.begin() + ((i + 1) * items_per_block),
                key_comparator<key_type, descending, start_bit, end_bit>()
            );
        }

        // Preparing device
        key_type* device_keys_output;
        HIP_CHECK(hipMalloc(&device_keys_output, keys_output.size() * sizeof(key_type)));

        HIP_CHECK(
            hipMemcpy(
                device_keys_output, keys_output.data(),
                keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(sort_key_kernel<block_size, items_per_thread, key_type>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_keys_output, to_striped, descending, start_bit, end_bit
        );

        // Getting results to host
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), device_keys_output,
                keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            SCOPED_TRACE(testing::Message() << "with index= " << i);
            ASSERT_EQ(keys_output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_keys_output));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    class value_type
>
__global__
__launch_bounds__(BlockSize)
void sort_key_value_kernel(
    key_type* device_keys_output,
    value_type* device_values_output,
    bool to_striped,
    bool descending,
    unsigned int start_bit,
    unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * items_per_block;

    key_type keys[ItemsPerThread];
    value_type values[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_keys_output + block_offset, keys);
    hipcub::LoadDirectBlocked(lid, device_values_output + block_offset, values);

    hipcub::BlockRadixSort<key_type, BlockSize, ItemsPerThread, value_type> bsort;
    if(to_striped)
    {
        if(descending)
            bsort.SortDescendingBlockedToStriped(keys, values, start_bit, end_bit);
        else
            bsort.SortBlockedToStriped(keys, values, start_bit, end_bit);

        hipcub::StoreDirectStriped<BlockSize>(lid, device_keys_output + block_offset, keys);
        hipcub::StoreDirectStriped<BlockSize>(lid, device_values_output + block_offset, values);
    }
    else
    {
        if(descending)
            bsort.SortDescending(keys, values, start_bit, end_bit);
        else
            bsort.Sort(keys, values, start_bit, end_bit);

        hipcub::StoreDirectBlocked(lid, device_keys_output + block_offset, keys);
        hipcub::StoreDirectBlocked(lid, device_values_output + block_offset, values);
    }
}


TYPED_TEST(HipcubBlockRadixSort, SortKeysValues)
{
    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool descending = TestFixture::params::descending;
    constexpr bool to_striped = TestFixture::params::to_striped;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 42;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> keys_output;
        if(std::is_floating_point<key_type>::value)
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                (key_type)-1000,
                (key_type)+1000,
                seed_value
            );
        }
        else
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                std::numeric_limits<key_type>::min(),
                std::numeric_limits<key_type>::max(),
                seed_value
            );
        }

        std::vector<value_type> values_output;
        if(std::is_floating_point<value_type>::value)
        {
            values_output = test_utils::get_random_data<value_type>(
                size,
                (value_type)-1000,
                (value_type)+1000,
                seed_value + seed_value_addition
            );
        }
        else
        {
            values_output = test_utils::get_random_data<value_type>(
                size,
                std::numeric_limits<value_type>::min(),
                std::numeric_limits<value_type>::max(),
                seed_value + seed_value_addition
            );
        }

        using key_value = std::pair<key_type, value_type>;

        // Calculate expected results on host
        std::vector<key_value> expected(size);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = key_value(keys_output[i], values_output[i]);
        }

        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(
                expected.begin() + (i * items_per_block),
                expected.begin() + ((i + 1) * items_per_block),
                key_value_comparator<key_type, value_type, descending, start_bit, end_bit>()
            );
        }

        key_type* device_keys_output;
        HIP_CHECK(hipMalloc(&device_keys_output, keys_output.size() * sizeof(key_type)));
        value_type* device_values_output;
        HIP_CHECK(hipMalloc(&device_values_output, values_output.size() * sizeof(value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_keys_output, keys_output.data(),
                keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_values_output, values_output.data(),
                values_output.size() * sizeof(typename decltype(values_output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(sort_key_value_kernel<block_size, items_per_thread, key_type, value_type>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_keys_output, device_values_output, to_striped, descending, start_bit, end_bit
        );

        // Getting results to host
        HIP_CHECK(
            hipMemcpy(
                keys_output.data(), device_keys_output,
                keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                values_output.data(), device_values_output,
                values_output.size() * sizeof(typename decltype(values_output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < size; i++)
        {
            SCOPED_TRACE(testing::Message() << "with index= " << i);
            ASSERT_EQ(keys_output[i], expected[i].first);
            ASSERT_EQ(values_output[i], expected[i].second);
        }

        HIP_CHECK(hipFree(device_keys_output));
        HIP_CHECK(hipFree(device_values_output));
    }
}
