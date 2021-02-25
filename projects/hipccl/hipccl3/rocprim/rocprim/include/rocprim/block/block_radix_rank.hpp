// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
#define ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"


#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"


/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/** Empty callback implementation */
template <int BINS_PER_THREAD>
struct block_radix_rank_empty_callback
{
    ROCPRIM_DEVICE inline
    void operator()(int (&bins)[BINS_PER_THREAD]) {}
};

template <
    int                     BlockSizeX,
    int                     RadixBits,
    bool                    IsDescending,
    bool                    MemoizeOuterScan      = true,
    BlockScanAlgorithm      InnerScanAlgorithm    = BLOCK_SCAN_WARP_SCANS,
    int                     BlockSizeY             = 1,
    int                     BlockSizeZ             = 1>
class block_radix_rank
{
private:

    /******************************************************************************
     * Type definitions and constants
     ******************************************************************************/

    // Integer type for digit counters (to be packed into words of type PackedCounters)
    typedef unsigned short DigitCounter;

    // Integer type for packing DigitCounters into columns of shared memory banks

    typedef typename unsigned int PackedCounter;
    // typedef typename If<(SMEM_CONFIG == cudaSharedMemBankSizeEightByte),
    //     unsigned long long,
    //     unsigned int>::Type PackedCounter;

    static constexpr unsigned int BlockSize   = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int RadixDigits = 1 <<RadixBits;
    static constexpr unsigned int WarpSize   =  detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // Number of warps in block
    static constexpr unsigned int Warp = (BlockSize + WarpSize - 1) / WarpSize;
    static constexpr unsigned int BytesPerCounter = sizeof(DigitCounter);

    static constexpr unsigned int PackingRatio      = sizeof(PackedCounter) / sizeof(DigitCounter);
    static constexpr unsigned int LogPackingRatio  = Log2<PackingRatio>::VALUE;
    static constexpr unsigned int LogCounterLanes = std::max(RadixBits - LogPackingRatio,0);
    static constexpr unsigned int CounterLanes = 1 << LogCounterLanes;
    static constexpr unsigned int PaddedCounterLanes = 1 +CounterLanes;
    static constexpr unsigned int RakingSegment = PaddedCounterLanes;

public:
    static constexpr unsigned int BinsTrackedPerThread = std::max(1,(RadixDigits + BlockSize -1 )/BlockSize);

private:


    /// BlockScan type
    typedef block_scan<
            PackedCounter,
            BlockSizeX,
            InnerScanAlgorithm,
            BlockSizeY,
            BlockSizeZ>
        block_scan;


    /// Shared memory storage layout type for BlockRadixRank
    struct storage_type_
    {
        union Aliasable
        {
            DigitCounter            digit_counters[PaddedCounterLanes][BlockSize][PackingRatio];
            PackedCounter           raking_grid[BlockSize][RakingSegment];

        } aliasable;

        // Storage for scanning local ranks
        typename BlockScan::TempStorage block_scan_storage;
    };

public:

  #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
  using storage_type = detail::raw_storage<storage_type_>;
  #else
  using storage_type = storage_type_; // only for Doxygen
  #endif

private:
    ROCPRIM_SHARED_MEMORY storage_type storage;

    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;

    /// Copy of raking segment, promoted to registers
    PackedCounter cached_segment[RakingSegment];


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /**
     * Internal storage allocator
     */
    ROCPRIM_DEVICE inline
    _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /**
     * Performs upsweep raking reduction, returning the aggregate
     */
    ROCPRIM_DEVICE inline
    PackedCounter Upsweep()
    {
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        PackedCounter *smem_raking_ptr = temp_storage.aliasable.raking_grid[linear_tid];
        PackedCounter *raking_ptr;

        if (MemoizeOuterScan)
        {
            // Copy data into registers
            #pragma unroll
            for (int i = 0; i < RakingSegment; i++)
            {
                cached_segment[i] = smem_raking_ptr[i];
            }
            raking_ptr = cached_segment;
        }
        else
        {
            raking_ptr = smem_raking_ptr;
        }

        return internal::ThreadReduce<RakingSegment>(raking_ptr, Sum());
    }


    /// Performs exclusive downsweep raking scan
    ROCPRIM_DEVICE inline
    void ExclusiveDownsweep(
        PackedCounter raking_partial)
    {
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        PackedCounter *smem_raking_ptr = temp_storage.aliasable.raking_grid[linear_tid];

        PackedCounter *raking_ptr = (MemoizeOuterScan) ?
            cached_segment :
            smem_raking_ptr;

        // Exclusive raking downsweep scan
        internal::ThreadScanExclusive<RakingSegment>(raking_ptr, raking_ptr, Sum(), raking_partial);

        if (MemoizeOuterScan)
        {
            // Copy data back to smem
            #pragma unroll
            for (int i = 0; i < RakingSegment; i++)
            {
                smem_raking_ptr[i] = cached_segment[i];
            }
        }
    }


    /**
     * Reset shared memory digit counters
     */
    ROCPRIM_DEVICE inline
    void ResetCounters()
    {
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        // Reset shared memory digit counters
        #pragma unroll
        for (int LANE = 0; LANE < PaddedCounterLanes; LANE++)
        {
            *((PackedCounter*) temp_storage.aliasable.digit_counters[LANE][linear_tid]) = 0;
        }
    }


    /**
     * Block-scan prefix callback
     */
    struct PrefixCallBack
    {
        ROCPRIM_DEVICE inline
        PackedCounter operator()(PackedCounter block_aggregate)
        {
            PackedCounter block_prefix = 0;

            // Propagate totals in packed fields
            #pragma unroll
            for (int PACKED = 1; PACKED < PackingRatio; PACKED++)
            {
                block_prefix += block_aggregate << (sizeof(DigitCounter) * 8 * PACKED);
            }

            return block_prefix;
        }
    };


    /**
     * Scan shared memory digit counters.
     */
    ROCPRIM_DEVICE inline
    void ScanCounters()
    {
        // Upsweep scan
        PackedCounter raking_partial = Upsweep();

        // Compute exclusive sum
        PackedCounter exclusive_partial;
        PrefixCallBack prefix_call_back;
        BlockScan(temp_storage.block_scan_storage).ExclusiveSum(raking_partial, exclusive_partial, prefix_call_back);

        // Downsweep scan with exclusive partial
        ExclusiveDownsweep(exclusive_partial);
    }

public:

    /// \smemstorage{BlockScan}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    ROCPRIM_DEVICE inline
    BlockRadixRank()
    :
        temp_storage(PrivateStorage()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    ROCPRIM_DEVICE inline
    BlockRadixRank(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Raking
     *********************************************************************/
    //@{

    /**
     * \brief Rank keys.
     */
    template <
        typename        UnsignedBits,
        int             KeysPerThread,
        typename        DigitExtractorT>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],           ///< [in] Keys for this tile
        int             (&ranks)[KeysPerThread],          ///< [out] For each key, the local rank within the tile
        DigitExtractorT digit_extractor)                    ///< [in] The digit extractor
    {
        DigitCounter    thread_prefixes[KeysPerThread];   // For each key, the count of previous keys in this tile having the same digit
        DigitCounter*   digit_counters[KeysPerThread];    // For each key, the byte-offset of its corresponding digit counter in smem

        // Reset shared memory digit counters
        ResetCounters();

        #pragma unroll
        for (int ITEM = 0; ITEM < KeysPerThread; ++ITEM)
        {
            // Get digit
            unsigned int digit = digit_extractor.Digit(keys[ITEM]);

            // Get sub-counter
            unsigned int sub_counter = digit >> LOG_COUNTER_LANES;

            // Get counter lane
            unsigned int counter_lane = digit & (COUNTER_LANES - 1);

            if (IS_DESCENDING)
            {
                sub_counter = PackingRatio - 1 - sub_counter;
                counter_lane = COUNTER_LANES - 1 - counter_lane;
            }

            // Pointer to smem digit counter
            digit_counters[ITEM] = &temp_storage.aliasable.digit_counters[counter_lane][linear_tid][sub_counter];

            // Load thread-exclusive prefix
            thread_prefixes[ITEM] = *digit_counters[ITEM];

            // Store inclusive prefix
            *digit_counters[ITEM] = thread_prefixes[ITEM] + 1;
        }

        ::rocprim::syncthreads();

        // Scan shared memory counters
        ScanCounters();

        ::rocprim::syncthreads();

        // Extract the local ranks of each key
        #pragma unroll
        for (int ITEM = 0; ITEM < KeysPerThread; ++ITEM)
        {
            // Add in thread block exclusive prefix
            ranks[ITEM] = thread_prefixes[ITEM] + *digit_counters[ITEM];
        }
    }


    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template <
        typename        UnsignedBits,
        int             KeysPerThread,
        typename        DigitExtractorT>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],           ///< [in] Keys for this tile
        int             (&ranks)[KeysPerThread],          ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor,                    ///< [in] The digit extractor
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD])            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
    {
        // Rank keys
        rank_keys(keys, ranks, digit_extractor);

        // Get the inclusive and exclusive digit totals corresponding to the calling thread.
        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (linear_tid * BINS_TRACKED_PER_THREAD) + track;

            if ((BlockSize == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
            {
                if (IS_DESCENDING)
                    bin_idx = RADIX_DIGITS - bin_idx - 1;

                // Obtain ex/inclusive digit counts.  (Unfortunately these all reside in the
                // first counter column, resulting in unavoidable bank conflicts.)
                unsigned int counter_lane   = (bin_idx & (COUNTER_LANES - 1));
                unsigned int sub_counter    = bin_idx >> (LOG_COUNTER_LANES);

                exclusive_digit_prefix[track] = temp_storage.aliasable.digit_counters[counter_lane][0][sub_counter];
            }
        }
    }
};





/**
 * Radix-rank using match.any
 */
template <
    int                     BLOCK_DIM_X,
    int                     RADIX_BITS,
    bool                    IS_DESCENDING,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM    = BLOCK_SCAN_WARP_SCANS,
    int                     BLOCK_DIM_Y             = 1,
    int                     BLOCK_DIM_Z             = 1,
    int                     PTX_ARCH                = CUB_PTX_ARCH>
class BlockRadixRankMatch
{
private:

    /******************************************************************************
     * Type definitions and constants
     ******************************************************************************/

    typedef int32_t    RankT;
    typedef int32_t    DigitCounterT;

    enum
    {
        // The thread block size in threads
        BlockSize               = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        RADIX_DIGITS                = 1 << RADIX_BITS,

        LOG_WARP_THREADS            = CUB_LOG_WARP_THREADS(PTX_ARCH),
        WARP_THREADS                = 1 << LOG_WARP_THREADS,
        WARPS                       = (BlockSize + WARP_THREADS - 1) / WARP_THREADS,

        PADDED_WARPS            = ((WARPS & 0x1) == 0) ?
                                    WARPS + 1 :
                                    WARPS,

        COUNTERS                = PADDED_WARPS * RADIX_DIGITS,
        RakingSegment          = (COUNTERS + BlockSize - 1) / BlockSize,
        PaddedRakingSegment   = ((RakingSegment & 0x1) == 0) ?
                                    RakingSegment + 1 :
                                    RakingSegment,
    };

public:

    enum
    {
        /// Number of bin-starting offsets tracked per thread
        BINS_TRACKED_PER_THREAD = CUB_MAX(1, (RADIX_DIGITS + BlockSize - 1) / BlockSize),
    };

private:

    /// BlockScan type
    typedef BlockScan<
            DigitCounterT,
            BlockSize,
            INNER_SCAN_ALGORITHM,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z,
            PTX_ARCH>
        BlockScanT;


    /// Shared memory storage layout type for BlockRadixRank
    struct __align__(16) _TempStorage
    {
        typename BlockScanT::TempStorage            block_scan_storage;

        union __align__(16) Aliasable
        {
            volatile DigitCounterT                  warp_digit_counters[RADIX_DIGITS][PADDED_WARPS];
            DigitCounterT                           raking_grid[BlockSize][PaddedRakingSegment];

        } aliasable;
    };


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;



public:

    /// \smemstorage{BlockScan}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    ROCPRIM_DEVICE inline
    BlockRadixRankMatch(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Raking
     *********************************************************************/
    //@{

    /** \brief Computes the count of keys for each digit value, and calls the
     * callback with the array of key counts.

     * @tparam CountsCallback The callback type. It should implement an instance
     * overload of operator()(int (&bins)[BINS_TRACKED_PER_THREAD]), where bins
     * is an array of key counts for each digit value distributed in block
     * distribution among the threads of the thread block. Key counts can be
     * used, to update other data structures in global or shared
     * memory. Depending on the implementation of the ranking algoirhtm
     * (see BlockRadixRankMatchEarlyCounts), key counts may become available
     * early, therefore, they are returned through a callback rather than a
     * separate output parameter of rank_keys().
     */
    template <int KeysPerThread, typename CountsCallback>
    ROCPRIM_DEVICE inline
    void CallBack(CountsCallback callback)
    {
        int bins[BINS_TRACKED_PER_THREAD];
        // Get count for each digit
        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (linear_tid * BINS_TRACKED_PER_THREAD) + track;
            const int TILE_ITEMS = KeysPerThread * BlockSize;

            if ((BlockSize == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
            {
                if (IS_DESCENDING)
                {
                    bin_idx = RADIX_DIGITS - bin_idx - 1;
                    bins[track] = (bin_idx > 0 ?
                        temp_storage.aliasable.warp_digit_counters[bin_idx - 1][0] : TILE_ITEMS) -
                        temp_storage.aliasable.warp_digit_counters[bin_idx][0];
                }
                else
                {
                    bins[track] = (bin_idx < RADIX_DIGITS - 1 ?
                        temp_storage.aliasable.warp_digit_counters[bin_idx + 1][0] : TILE_ITEMS) -
                        temp_storage.aliasable.warp_digit_counters[bin_idx][0];
                }
            }
        }
        callback(bins);
    }

    /**
     * \brief Rank keys.
     */
    template <
        typename        UnsignedBits,
        int             KeysPerThread,
        typename        DigitExtractorT,
        typename        CountsCallback>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],           ///< [in] Keys for this tile
        int             (&ranks)[KeysPerThread],          ///< [out] For each key, the local rank within the tile
        DigitExtractorT digit_extractor,                    ///< [in] The digit extractor
        CountsCallback    callback)
    {
        // Initialize shared digit counters

        #pragma unroll
        for (int ITEM = 0; ITEM < PaddedRakingSegment; ++ITEM)
            temp_storage.aliasable.raking_grid[linear_tid][ITEM] = 0;

        ::rocprim::syncthreads();

        // Each warp will strip-mine its section of input, one strip at a time

        volatile DigitCounterT  *digit_counters[KeysPerThread];
        uint32_t                warp_id         = linear_tid >> LOG_WARP_THREADS;
        uint32_t                lane_mask_lt    = LaneMaskLt();

        #pragma unroll
        for (int ITEM = 0; ITEM < KeysPerThread; ++ITEM)
        {
            // My digit
            uint32_t digit = digit_extractor.Digit(keys[ITEM]);

            if (IS_DESCENDING)
                digit = RADIX_DIGITS - digit - 1;

            // Mask of peers who have same digit as me
            uint32_t peer_mask = MatchAny<RADIX_BITS>(digit);

            // Pointer to smem digit counter for this key
            digit_counters[ITEM] = &temp_storage.aliasable.warp_digit_counters[digit][warp_id];

            // Number of occurrences in previous strips
            DigitCounterT warp_digit_prefix = *digit_counters[ITEM];

            // Warp-sync
            WARP_SYNC(0xFFFFFFFF);

            // Number of peers having same digit as me
            int32_t digit_count = __popc(peer_mask);

            // Number of lower-ranked peers having same digit seen so far
            int32_t peer_digit_prefix = __popc(peer_mask & lane_mask_lt);

            if (peer_digit_prefix == 0)
            {
                // First thread for each digit updates the shared warp counter
                *digit_counters[ITEM] = DigitCounterT(warp_digit_prefix + digit_count);
            }

            // Warp-sync
            WARP_SYNC(0xFFFFFFFF);

            // Number of prior keys having same digit
            ranks[ITEM] = warp_digit_prefix + DigitCounterT(peer_digit_prefix);
        }

        ::rocprim::syncthreads();

        // Scan warp counters

        DigitCounterT scan_counters[PaddedRakingSegment];

        #pragma unroll
        for (int ITEM = 0; ITEM < PaddedRakingSegment; ++ITEM)
            scan_counters[ITEM] = temp_storage.aliasable.raking_grid[linear_tid][ITEM];

        BlockScanT(temp_storage.block_scan_storage).ExclusiveSum(scan_counters, scan_counters);

        #pragma unroll
        for (int ITEM = 0; ITEM < PaddedRakingSegment; ++ITEM)
            temp_storage.aliasable.raking_grid[linear_tid][ITEM] = scan_counters[ITEM];

        ::rocprim::syncthreads();
        if (!Equals<CountsCallback, BlockRadixRankEmptyCallback<BINS_TRACKED_PER_THREAD>>::VALUE)
        {
            CallBack<KeysPerThread>(callback);
        }

        // Seed ranks with counter values from previous warps
        #pragma unroll
        for (int ITEM = 0; ITEM < KeysPerThread; ++ITEM)
            ranks[ITEM] += *digit_counters[ITEM];
    }

    template <
        typename        UnsignedBits,
        int             KeysPerThread,
        typename        DigitExtractorT>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread], int (&ranks)[KeysPerThread],
        DigitExtractorT digit_extractor)
    {
        rank_keys(keys, ranks, digit_extractor,
                 BlockRadixRankEmptyCallback<BINS_TRACKED_PER_THREAD>());
    }

    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template <
        typename        UnsignedBits,
        int             KeysPerThread,
        typename        DigitExtractorT,
        typename        CountsCallback>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],           ///< [in] Keys for this tile
        int             (&ranks)[KeysPerThread],          ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor,                    ///< [in] The digit extractor
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD],            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
        CountsCallback callback)
    {
        rank_keys(keys, ranks, digit_extractor, callback);

        // Get exclusive count for each digit
        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (linear_tid * BINS_TRACKED_PER_THREAD) + track;

            if ((BlockSize == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
            {
                if (IS_DESCENDING)
                    bin_idx = RADIX_DIGITS - bin_idx - 1;

                exclusive_digit_prefix[track] = temp_storage.aliasable.warp_digit_counters[bin_idx][0];
            }
        }
    }

    template <
        typename        UnsignedBits,
        int             KeysPerThread,
        typename        DigitExtractorT>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],           ///< [in] Keys for this tile
        int             (&ranks)[KeysPerThread],          ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD])            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
    {
        rank_keys(keys, ranks, digit_extractor, exclusive_digit_prefix,
                 BlockRadixRankEmptyCallback<BINS_TRACKED_PER_THREAD>());
    }
};

enum WarpMatchAlgorithm
{
    WARP_MATCH_ANY,
    WARP_MATCH_ATOMIC_OR
};

/**
 * Radix-rank using matching which computes the counts of keys for each digit
 * value early, at the expense of doing more work. This may be useful e.g. for
 * decoupled look-back, where it reduces the time other thread blocks need to
 * wait for digit counts to become available.
 */
template <int BLOCK_DIM_X, int RADIX_BITS, bool IS_DESCENDING,
          BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS,
          WarpMatchAlgorithm MATCH_ALGORITHM = WARP_MATCH_ANY, int NUM_PARTS = 1>
struct BlockRadixRankMatchEarlyCounts
{
    // constants
    enum
    {
        BlockSize = BLOCK_DIM_X,
        RADIX_DIGITS = 1 << RADIX_BITS,
        BINS_PER_THREAD = (RADIX_DIGITS + BlockSize - 1) / BlockSize,
        BINS_TRACKED_PER_THREAD = BINS_PER_THREAD,
        FULL_BINS = BINS_PER_THREAD * BlockSize == RADIX_DIGITS,
        WARP_THREADS = CUB_PTX_WARP_THREADS,
        BLOCK_WARPS = BlockSize / WARP_THREADS,
        WARP_MASK = ~0,
        NUM_MATCH_MASKS = MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR ? BLOCK_WARPS : 0,
        // Guard against declaring zero-sized array:
        MATCH_MASKS_ALLOC_SIZE = NUM_MATCH_MASKS < 1 ? 1 : NUM_MATCH_MASKS,
    };

    // types
    typedef cub::BlockScan<int, BlockSize, INNER_SCAN_ALGORITHM> BlockScan;



    // temporary storage
    struct TempStorage
    {
        union
        {
            int warp_offsets[BLOCK_WARPS][RADIX_DIGITS];
            int warp_histograms[BLOCK_WARPS][RADIX_DIGITS][NUM_PARTS];
        };

        int match_masks[MATCH_MASKS_ALLOC_SIZE][RADIX_DIGITS];

        typename BlockScan::TempStorage prefix_tmp;
    };

    TempStorage& temp_storage;

    // internal ranking implementation
    template <typename UnsignedBits, int KeysPerThread, typename DigitExtractorT,
              typename CountsCallback>
    struct BlockRadixRankMatchInternal
    {
        TempStorage& s;
        DigitExtractorT digit_extractor;
        CountsCallback callback;
        int warp;
        int lane;

        ROCPRIM_DEVICE inline
        int Digit(UnsignedBits key)
        {
            int digit =  digit_extractor.Digit(key);
            return IS_DESCENDING ? RADIX_DIGITS - 1 - digit : digit;
        }

        ROCPRIM_DEVICE inline
        int ThreadBin(int u)
        {
            int bin = threadIdx.x * BINS_PER_THREAD + u;
            return IS_DESCENDING ? RADIX_DIGITS - 1 - bin : bin;
        }

        ROCPRIM_DEVICE inline

        void compute_histograms_warp(UnsignedBits (&keys)[KeysPerThread])
        {
            //int* warp_offsets = &s.warp_offsets[warp][0];
            int (&warp_histograms)[RADIX_DIGITS][NUM_PARTS] = s.warp_histograms[warp];
            // compute warp-private histograms
            #pragma unroll
            for (int bin = lane; bin < RADIX_DIGITS; bin += WARP_THREADS)
            {
                #pragma unroll
                for (int part = 0; part < NUM_PARTS; ++part)
                {
                    warp_histograms[bin][part] = 0;
                }
            }
            if (MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR)
            {
                int* match_masks = &s.match_masks[warp][0];
                #pragma unroll
                for (int bin = lane; bin < RADIX_DIGITS; bin += WARP_THREADS)
                {
                    match_masks[bin] = 0;
                }
            }
            WARP_SYNC(WARP_MASK);

            // compute private per-part histograms
            int part = lane % NUM_PARTS;
            #pragma unroll
            for (int u = 0; u < KeysPerThread; ++u)
            {
                atomicAdd(&warp_histograms[Digit(keys[u])][part], 1);
            }

            // sum different parts;
            // no extra work is necessary if NUM_PARTS == 1
            if (NUM_PARTS > 1)
            {
                WARP_SYNC(WARP_MASK);
                // TODO: handle RADIX_DIGITS % WARP_THREADS != 0 if it becomes necessary
                const int WARP_BINS_PER_THREAD = RADIX_DIGITS / WARP_THREADS;
                int bins[WARP_BINS_PER_THREAD];
                #pragma unroll
                for (int u = 0; u < WARP_BINS_PER_THREAD; ++u)
                {
                    int bin = lane + u * WARP_THREADS;
                    bins[u] = internal::ThreadReduce(warp_histograms[bin], Sum());
                }
                ::rocprim::syncthreads();

                // store the resulting histogram in shared memory
                int* warp_offsets = &s.warp_offsets[warp][0];
                #pragma unroll
                for (int u = 0; u < WARP_BINS_PER_THREAD; ++u)
                {
                    int bin = lane + u * WARP_THREADS;
                    warp_offsets[bin] = bins[u];
                }
            }
        }

        ROCPRIM_DEVICE inline

        void compute_offsets_warp_upsweep(int (&bins)[BINS_PER_THREAD])
        {
            // sum up warp-private histograms
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                bins[u] = 0;
                int bin = ThreadBin(u);
                if (FULL_BINS || (bin >= 0 && bin < RADIX_DIGITS))
                {
                    #pragma unroll
                    for (int j_warp = 0; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        int warp_offset = s.warp_offsets[j_warp][bin];
                        s.warp_offsets[j_warp][bin] = bins[u];
                        bins[u] += warp_offset;
                    }
                }
            }
        }

        ROCPRIM_DEVICE inline

        void compute_offsets_warp_downsweep(int (&offsets)[BINS_PER_THREAD])
        {
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                int bin = ThreadBin(u);
                if (FULL_BINS || (bin >= 0 && bin < RADIX_DIGITS))
                {
                    int digit_offset = offsets[u];
                    #pragma unroll
                    for (int j_warp = 0; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        s.warp_offsets[j_warp][bin] += digit_offset;
                    }
                }
            }
        }

        ROCPRIM_DEVICE inline

        void compute_ranks_item(
            UnsignedBits (&keys)[KeysPerThread], int (&ranks)[KeysPerThread],
            Int2Type<WARP_MATCH_ATOMIC_OR>)
        {
            // compute key ranks
            int lane_mask = 1 << lane;
            int* warp_offsets = &s.warp_offsets[warp][0];
            int* match_masks = &s.match_masks[warp][0];
            #pragma unroll
            for (int u = 0; u < KeysPerThread; ++u)
            {
                int bin = Digit(keys[u]);
                int* p_match_mask = &match_masks[bin];
                atomicOr(p_match_mask, lane_mask);
                WARP_SYNC(WARP_MASK);
                int bin_mask = *p_match_mask;
                int leader = (WARP_THREADS - 1) - __clz(bin_mask);
                int warp_offset = 0;
                int popc = __popc(bin_mask & LaneMaskLe());
                if (lane == leader)
                {
                    // atomic is a bit faster
                    warp_offset = atomicAdd(&warp_offsets[bin], popc);
                }
                warp_offset = SHFL_IDX_SYNC(warp_offset, leader, bin_mask);
                if (lane == leader) *p_match_mask = 0;
                WARP_SYNC(WARP_MASK);
                ranks[u] = warp_offset + popc - 1;
            }
        }

        ROCPRIM_DEVICE inline

        void compute_ranks_item(
            UnsignedBits (&keys)[KeysPerThread], int (&ranks)[KeysPerThread],
            Int2Type<WARP_MATCH_ANY>)
        {
            // compute key ranks
            int* warp_offsets = &s.warp_offsets[warp][0];
            #pragma unroll
            for (int u = 0; u < KeysPerThread; ++u)
            {
                int bin = Digit(keys[u]);
                int bin_mask = MatchAny<RADIX_BITS>(bin);
                int leader = (WARP_THREADS - 1) - __clz(bin_mask);
                int warp_offset = 0;
                int popc = __popc(bin_mask & LaneMaskLe());
                if (lane == leader)
                {
                    // atomic is a bit faster
                    warp_offset = atomicAdd(&warp_offsets[bin], popc);
                }
                warp_offset = SHFL_IDX_SYNC(warp_offset, leader, bin_mask);
                ranks[u] = warp_offset + popc - 1;
            }
        }

        ROCPRIM_DEVICE inline
        void rank_keys(
            UnsignedBits (&keys)[KeysPerThread],
            int (&ranks)[KeysPerThread],
            int (&exclusive_digit_prefix)[BINS_PER_THREAD])
        {
            compute_histograms_warp(keys);

            ::rocprim::syncthreads();
            int bins[BINS_PER_THREAD];
            compute_offsets_warp_upsweep(bins);
            callback(bins);

            BlockScan(s.prefix_tmp).ExclusiveSum(bins, exclusive_digit_prefix);

            compute_offsets_warp_downsweep(exclusive_digit_prefix);
            ::rocprim::syncthreads();
            compute_ranks_item(keys, ranks, Int2Type<MATCH_ALGORITHM>());
        }

        ROCPRIM_DEVICE inline
        BlockRadixRankMatchInternal
        (TempStorage& temp_storage, DigitExtractorT digit_extractor, CountsCallback callback)
            : s(temp_storage), digit_extractor(digit_extractor),
              callback(callback), warp(threadIdx.x / WARP_THREADS), lane(LaneId())
            {}
    };

    ROCPRIM_DEVICE inline
    BlockRadixRankMatchEarlyCounts
    (TempStorage& temp_storage) : temp_storage(temp_storage) {}

    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template <typename UnsignedBits, int KeysPerThread, typename DigitExtractorT,
        typename CountsCallback>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],
        int             (&ranks)[KeysPerThread],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_PER_THREAD],
        CountsCallback  callback)
    {
        BlockRadixRankMatchInternal<UnsignedBits, KeysPerThread, DigitExtractorT, CountsCallback>
            internal(temp_storage, digit_extractor, callback);
        internal.rank_keys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, int KeysPerThread, typename DigitExtractorT>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],
        int             (&ranks)[KeysPerThread],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_PER_THREAD])
    {
        typedef BlockRadixRankEmptyCallback<BINS_PER_THREAD> CountsCallback;
        BlockRadixRankMatchInternal<UnsignedBits, KeysPerThread, DigitExtractorT, CountsCallback>
            internal(temp_storage, digit_extractor, CountsCallback());
        internal.rank_keys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, int KeysPerThread, typename DigitExtractorT>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],
        int             (&ranks)[KeysPerThread],
        DigitExtractorT digit_extractor)
    {
        int exclusive_digit_prefix[BINS_PER_THREAD];
        rank_keys(keys, ranks, digit_extractor, exclusive_digit_prefix);
    }
};


END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
