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
#include "block_scan.hpp"

#include "../thread/thread_reduce.hpp"
#include "../thread/thread_scan.hpp"
#include "../thread/thread_operators.hpp"
#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"


// using_warp_scan,
/// \brief An algorithm which limits calculations to a single hardware warp.
// reduce_then_scan,

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/** Empty callback implementation */
template <int BinsPerThread>
struct block_radix_rank_empty_callback
{
    ROCPRIM_DEVICE inline
    void operator()(int (&bins)[BinsPerThread]) { (void)bins; }
};

template <
    int                     BlockSizeX,
    int                     RadixBits,
    bool                    IsDescending,
    bool                    MemoizeOuterScan      = true,
    block_scan_algorithm    InnerScanAlgorithm    = block_scan_algorithm::using_warp_scan,
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

    typedef unsigned int PackedCounter;
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
    static constexpr unsigned int LogCounterLanes = maximum<unsigned int>()(RadixBits - LogPackingRatio,0);
    static constexpr unsigned int CounterLanes = 1 << LogCounterLanes;
    static constexpr unsigned int PaddedCounterLanes = 1 +CounterLanes;
    static constexpr unsigned int RakingSegment = PaddedCounterLanes;

public:
    static constexpr unsigned int BinsTrackedPerThread = maximum<unsigned int>()(1,(RadixDigits + BlockSize -1 )/BlockSize);

private:


    /// block_scan type
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
        typename block_scan::storage_type block_scan_storage;
    };

public:

  #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
  using storage_type = detail::raw_storage<storage_type_>;
  #else
  using storage_type = storage_type_; // only for Doxygen
  #endif

private:


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    ROCPRIM_SHARED_MEMORY storage_type_ &temp_storage;

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
    storage_type_& PrivateStorage()
    {
        __shared__ storage_type_ private_storage;
        return private_storage;
    }


    /**
     * Performs upsweep raking reduction, returning the aggregate
     */
    ROCPRIM_DEVICE inline
    PackedCounter up_sweep()
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

        return internal::thread_reduce<RakingSegment>(raking_ptr, sum());
    }


    /// Performs exclusive downsweep raking scan
    ROCPRIM_DEVICE inline
    void exclusive_downsweep(
        PackedCounter raking_partial)
    {
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        PackedCounter *smem_raking_ptr = temp_storage.aliasable.raking_grid[linear_tid];

        PackedCounter *raking_ptr = (MemoizeOuterScan) ?
            cached_segment :
            smem_raking_ptr;

        // Exclusive raking downsweep scan
        internal::thread_scan_exclusive<RakingSegment>(raking_ptr, raking_ptr, sum(), raking_partial);


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
    void reset_counters()
    {
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        // Reset shared memory digit counters
        #pragma unroll
        for (int Lane = 0; Lane < PaddedCounterLanes; Lane++)
        {
            *((PackedCounter*) temp_storage.aliasable.digit_counters[Lane][linear_tid]) = 0;
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
            for (int Packed = 1; Packed < PackingRatio; Packed++)
            {
                block_prefix += block_aggregate << (sizeof(DigitCounter) * 8 * Packed);
            }

            return block_prefix;
        }
    };


    /**
     * Scan shared memory digit counters.
     */
    ROCPRIM_DEVICE inline
    void scan_counters()
    {
        // up_sweep scan
        PackedCounter raking_partial = up_sweep();

        // Compute exclusive sum
        PackedCounter exclusive_partial;
        PrefixCallBack prefix_call_back;
        block_scan(temp_storage.block_scan_storage).exclusive_sum(raking_partial, exclusive_partial, prefix_call_back);

        // Downsweep scan with exclusive partial
        exclusive_downsweep(exclusive_partial);
    }

public:

    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    // ROCPRIM_DEVICE inline
    // BlockRadixRank()
    // :
    //     temp_storage(PrivateStorage()),
    //     linear_tid(RowMajorTid(BlockSizeX, BlockSizeY, BlockSizeZ))
    // {}
    //
    //
    // /**
    //  * \brief Collective constructor using the specified memory allocation as temporary storage.
    //  */
    // ROCPRIM_DEVICE inline
    // BlockRadixRank(
    //     storage_type &temp_storage)             ///< [in] Reference to memory allocation having layout type storage_type
    // :
    //     temp_storage(temp_storage.Alias()),
    //     linear_tid(RowMajorTid(BlockSizeX, BlockSizeY, BlockSizeZ))
    // {}


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
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        // Reset shared memory digit counters
        reset_counters();

        #pragma unroll
        for (int Item = 0; Item < KeysPerThread; ++Item)
        {
            // Get digit
            unsigned int digit = digit_extractor.digit(keys[Item]);

            // Get sub-counter
            unsigned int sub_counter = digit >> LogCounterLanes;

            // Get counter lane
            unsigned int counter_lane = digit & (CounterLanes - 1);

            if (IsDescending)
            {
                sub_counter = PackingRatio - 1 - sub_counter;
                counter_lane = CounterLanes - 1 - counter_lane;
            }

            // Pointer to smem digit counter
            digit_counters[Item] = &temp_storage.aliasable.digit_counters[counter_lane][linear_tid][sub_counter];

            // Load thread-exclusive prefix
            thread_prefixes[Item] = *digit_counters[Item];

            // Store inclusive prefix
            *digit_counters[Item] = thread_prefixes[Item] + 1;
        }

        ::rocprim::syncthreads();

        // Scan shared memory counters
        scan_counters();

        ::rocprim::syncthreads();

        // Extract the local ranks of each key
        #pragma unroll
        for (int Item = 0; Item < KeysPerThread; ++Item)
        {
            // Add in thread block exclusive prefix
            ranks[Item] = thread_prefixes[Item] + *digit_counters[Item];
        }
    }


    /**
     * \brief Rank keys.  For the lower \p RadixDigits threads, digit counts for each digit are provided for the corresponding thread.
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
        int             (&exclusive_digit_prefix)[BinsTrackedPerThread])            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BinsTrackedPerThread) ... (threadIdx.x * BinsTrackedPerThread) + BinsTrackedPerThread - 1]
    {
      const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        // Rank keys
        rank_keys(keys, ranks, digit_extractor);

        // Get the inclusive and exclusive digit totals corresponding to the calling thread.
        #pragma unroll
        for (int track = 0; track < BinsTrackedPerThread; ++track)
        {
            int bin_idx = (linear_tid * BinsTrackedPerThread) + track;

            if ((BlockSize == RadixDigits) || (bin_idx < RadixDigits))
            {
                if (IsDescending)
                    bin_idx = RadixDigits - bin_idx - 1;

                // Obtain ex/inclusive digit counts.  (Unfortunately these all reside in the
                // first counter column, resulting in unavoidable bank conflicts.)
                unsigned int counter_lane   = (bin_idx & (CounterLanes - 1));
                unsigned int sub_counter    = bin_idx >> (LogCounterLanes);

                exclusive_digit_prefix[track] = temp_storage.aliasable.digit_counters[counter_lane][0][sub_counter];
            }
        }
    }
};





/**
 * Radix-rank using match.any
 */
template <
    int                     BlockSizeX,
    int                     RadixBits,
    bool                    IsDescending,
    block_scan_algorithm    InnerScanAlgorithm    = block_scan_algorithm::using_warp_scan,
    int                     BlockSizeY             = 1,
    int                     BlockSizeZ             = 1>
class block_radix_rank_match
{
private:

    /******************************************************************************
     * Type definitions and constants
     ******************************************************************************/

    typedef int32_t    RankT;
    typedef int32_t    DigitCounterT;


    // The thread block size in threads
    static constexpr unsigned int  BlockSize           = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int  RadixDigits         = 1 << RadixBits;
    static constexpr unsigned int  WarpThreads         =  detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    static constexpr unsigned int  Warps               = (BlockSize + WarpThreads - 1) / WarpThreads;
    static constexpr unsigned int  PaddedWarps         = ((Warps & 0x1) == 0) ?  Warps + 1 :     Warps;
    static constexpr unsigned int  Counters            = PaddedWarps * RadixDigits;
    static constexpr unsigned int  RakingSegment       = (Counters + BlockSize - 1) / BlockSize;
    static constexpr unsigned int  PaddedRakingSegment = ((RakingSegment & 0x1) == 0) ?  RakingSegment + 1 :  RakingSegment;

public:

    static constexpr unsigned int BinsTrackedPerThread = maximum<unsigned int>()(1, (RadixDigits + BlockSize - 1) / BlockSize);

private:

    /// block_scan type
    typedef block_scan<
            DigitCounterT,
            BlockSize,
            InnerScanAlgorithm,
            BlockSizeY,
            BlockSizeZ>
        block_scan_t;


    /// Shared memory storage layout type for BlockRadixRank
    struct __align__(16) storage_type_
    {
        typename block_scan_t::storage_type            block_scan_storage;

        union __align__(16) Aliasable
        {
            volatile DigitCounterT                  warp_digit_counters[RadixDigits][PaddedWarps];
            DigitCounterT                           raking_grid[BlockSize][PaddedRakingSegment];

        } aliasable;
    };


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    storage_type_ &temp_storage;




public:

      #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
      using storage_type = detail::raw_storage<storage_type_>;
      #else
      using storage_type = storage_type_; // only for Doxygen
      #endif


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    ROCPRIM_DEVICE inline
    block_radix_rank_match(
        storage_type &temp_storage)             ///< [in] Reference to memory allocation having layout type storage_type
    :        temp_storage(temp_storage.Alias())
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Raking
     *********************************************************************/
    //@{

    /** \brief Computes the count of keys for each digit value, and calls the
     * callback with the array of key counts.

     * @tparam CountsCallback The callback type. It should implement an instance
     * overload of operator()(int (&bins)[BinsTrackedPerThread]), where bins
     * is an array of key counts for each digit value distributed in block
     * distribution among the threads of the thread block. Key counts can be
     * used, to update other data structures in global or shared
     * memory. Depending on the implementation of the ranking algoirhtm
     * (see block_radix_rank_match_early_counts), key counts may become available
     * early, therefore, they are returned through a callback rather than a
     * separate output parameter of rank_keys().
     */
    template <int KeysPerThread, typename CountsCallback>
    ROCPRIM_DEVICE inline
    void CallBack(CountsCallback callback)
    {
      const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        int bins[BinsTrackedPerThread];
        // Get count for each digit
        #pragma unroll
        for (int track = 0; track < BinsTrackedPerThread; ++track)
        {
            int bin_idx = (linear_tid * BinsTrackedPerThread) + track;
            const int TileItems = KeysPerThread * BlockSize;

            if ((BlockSize == RadixDigits) || (bin_idx < RadixDigits))
            {
                if (IsDescending)
                {
                    bin_idx = RadixDigits - bin_idx - 1;
                    bins[track] = (bin_idx > 0 ?
                        temp_storage.aliasable.warp_digit_counters[bin_idx - 1][0] : TileItems) -
                        temp_storage.aliasable.warp_digit_counters[bin_idx][0];
                }
                else
                {
                    bins[track] = (bin_idx < RadixDigits - 1 ?
                        temp_storage.aliasable.warp_digit_counters[bin_idx + 1][0] : TileItems) -
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
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        #pragma unroll
        for (int Item = 0; Item < PaddedRakingSegment; ++Item)
            temp_storage.aliasable.raking_grid[linear_tid][Item] = 0;

        ::rocprim::syncthreads();

        // Each warp will strip-mine its section of input, one strip at a time

        volatile DigitCounterT  *digit_counters[KeysPerThread];
        uint32_t                warp_id         = ::rocprim::warp_id(linear_tid);
        uint32_t                lane_mask_lt    = lane_mask_less_than();

        #pragma unroll
        for (int Item = 0; Item < KeysPerThread; ++Item)
        {
            // My digit
            uint32_t digit = digit_extractor.digit(keys[Item]);

            if (IsDescending)
                digit = RadixDigits - digit - 1;

            // Mask of peers who have same digit as me
            uint32_t peer_mask = MatchAny<RadixBits>(digit);

            // Pointer to smem digit counter for this key
            digit_counters[Item] = &temp_storage.aliasable.warp_digit_counters[digit][warp_id];

            // Number of occurrences in previous strips
            DigitCounterT warp_digit_prefix = *digit_counters[Item];

            // Warp-sync
            // WARP_SYNC(0xFFFFFFFF);

            // Number of peers having same digit as me
            int32_t digit_count = __popc(peer_mask);

            // Number of lower-ranked peers having same digit seen so far
            int32_t peer_digit_prefix = __popc(peer_mask & lane_mask_lt);

            if (peer_digit_prefix == 0)
            {
                // First thread for each digit updates the shared warp counter
                *digit_counters[Item] = DigitCounterT(warp_digit_prefix + digit_count);
            }

            // Warp-sync
            // WARP_SYNC(0xFFFFFFFF);

            // Number of prior keys having same digit
            ranks[Item] = warp_digit_prefix + DigitCounterT(peer_digit_prefix);
        }

        ::rocprim::syncthreads();

        // Scan warp counters

        DigitCounterT scan_counters[PaddedRakingSegment];

        #pragma unroll
        for (int Items = 0; Items < PaddedRakingSegment; ++Items)
            scan_counters[Items] = temp_storage.aliasable.raking_grid[linear_tid][Items];

        block_scan_t(temp_storage.block_scan_storage).exclusive_sum(scan_counters, scan_counters);

        #pragma unroll
        for (int Items = 0; Items < PaddedRakingSegment; ++Items)
            temp_storage.aliasable.raking_grid[linear_tid][Items] = scan_counters[Items];

        ::rocprim::syncthreads();
        if (!Equals<CountsCallback, block_radix_rank_empty_callback<BinsTrackedPerThread>>::VALUE)
        {
            CallBack<KeysPerThread>(callback);
        }

        // Seed ranks with counter values from previous warps
        #pragma unroll
        for (int Items = 0; Items < KeysPerThread; ++Items)
            ranks[Items] += *digit_counters[Items];
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
                 block_radix_rank_empty_callback<BinsTrackedPerThread>());
    }

    /**
     * \brief Rank keys.  For the lower \p RadixDigits threads, digit counts for each digit are provided for the corresponding thread.
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
        int             (&exclusive_digit_prefix)[BinsTrackedPerThread],            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BinsTrackedPerThread) ... (threadIdx.x * BinsTrackedPerThread) + BinsTrackedPerThread - 1]
        CountsCallback callback)
    {
        const size_t linear_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        rank_keys(keys, ranks, digit_extractor, callback);

        // Get exclusive count for each digit
        #pragma unroll
        for (int track = 0; track < BinsTrackedPerThread; ++track)
        {
            int bin_idx = (linear_tid * BinsTrackedPerThread) + track;

            if ((BlockSize == RadixDigits) || (bin_idx < RadixDigits))
            {
                if (IsDescending)
                    bin_idx = RadixDigits - bin_idx - 1;

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
        int             (&exclusive_digit_prefix)[BinsTrackedPerThread])            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BinsTrackedPerThread) ... (threadIdx.x * BinsTrackedPerThread) + BinsTrackedPerThread - 1]
    {
        rank_keys(keys, ranks, digit_extractor, exclusive_digit_prefix,
                 block_radix_rank_empty_callback<BinsTrackedPerThread>());
    }
};

enum warp_match_algorithm : int
{
    warp_match_any,
    warp_match_atomic_or
};

/**
 * Radix-rank using matching which computes the counts of keys for each digit
 * value early, at the expense of doing more work. This may be useful e.g. for
 * decoupled look-back, where it reduces the time other thread blocks need to
 * wait for digit counts to become available.
 */
template <int BlockSizeX,
          int RadixBits,
          bool IsDescending,
          block_scan_algorithm InnerScanAlgorithm = block_scan_algorithm::using_warp_scan,
          warp_match_algorithm MatchAlgorithm = warp_match_algorithm::warp_match_any,
          int NumParts = 1>
struct block_radix_rank_match_early_counts
{
    // constants

    static constexpr unsigned int BlockSize = BlockSizeX;
    static constexpr unsigned int RadixDigits = 1 << RadixBits;
    static constexpr unsigned int BinsPerThread = (RadixDigits + BlockSize - 1) / BlockSize;
    static constexpr unsigned int BinsTrackedPerThread = BinsPerThread;
    static constexpr unsigned int FullBins = BinsPerThread * BlockSize == RadixDigits;
    static constexpr unsigned int WarpThreads = ::rocprim::warp_size();
    static constexpr unsigned int BlockWarps = BlockSize / WarpThreads;
    static constexpr unsigned int WarpMask = ~0;
    static constexpr unsigned int NumMatchMasks = MatchAlgorithm == warp_match_algorithm::warp_match_atomic_or ? BlockWarps : 0;

    // Guard against declaring zero-sized array;
    static constexpr unsigned int MatchMasksAllocSize = NumMatchMasks < 1 ? 1 : NumMatchMasks;

    // types
    typedef ::rocprim::block_scan<int, BlockSize, InnerScanAlgorithm> block_scan;



    // temporary storage
    struct storage_type
    {
        union
        {
            int warp_offsets[BlockWarps][RadixDigits];
            int warp_histograms[BlockWarps][RadixDigits][NumParts];
        };

        int match_masks[MatchMasksAllocSize][RadixDigits];

        typename block_scan::storage_type prefix_tmp;
    };

    storage_type& temp_storage;

    // internal ranking implementation
    template <typename UnsignedBits, int KeysPerThread, typename DigitExtractorT,
              typename CountsCallback>
    struct block_radix_rank_match_internal
    {
        storage_type& s;
        DigitExtractorT digit_extractor;
        CountsCallback callback;
        int warp;
        int lane;

        ROCPRIM_DEVICE inline
        int digit(UnsignedBits key)
        {
            int digit =  digit_extractor.digit(key);
            return IsDescending ? RadixDigits - 1 - digit : digit;
        }

        ROCPRIM_DEVICE inline
        int thread_bin(int u)
        {
            int bin = threadIdx.x * BinsPerThread + u;
            return IsDescending ? RadixDigits - 1 - bin : bin;
        }

        ROCPRIM_DEVICE inline

        void compute_histograms_warp(UnsignedBits (&keys)[KeysPerThread])
        {
            //int* warp_offsets = &s.warp_offsets[warp][0];
            int (&warp_histograms)[RadixDigits][NumParts] = s.warp_histograms[warp];
            // compute warp-private histograms
            #pragma unroll
            for (int bin = lane; bin < RadixDigits; bin += WarpThreads)
            {
                #pragma unroll
                for (int part = 0; part < NumParts; ++part)
                {
                    warp_histograms[bin][part] = 0;
                }
            }
            if (MatchAlgorithm == warp_match_algorithm::warp_match_atomic_or)
            {
                int* match_masks = &s.match_masks[warp][0];
                #pragma unroll
                for (int bin = lane; bin < RadixDigits; bin += WarpThreads)
                {
                    match_masks[bin] = 0;
                }
            }
            // WARP_SYNC(WarpMask);

            // compute private per-part histograms
            int part = lane % NumParts;
            #pragma unroll
            for (int u = 0; u < KeysPerThread; ++u)
            {
                atomicAdd(&warp_histograms[digit(keys[u])][part], 1);
            }

            // sum different parts;
            // no extra work is necessary if NumParts == 1
            if (NumParts > 1)
            {
                // WARP_SYNC(WarpMask);
                // TODO: handle RadixDigits % WarpThreads != 0 if it becomes necessary
                const int WARP_BinsPerThread = RadixDigits / WarpThreads;
                int bins[WARP_BinsPerThread];
                #pragma unroll
                for (int u = 0; u < WARP_BinsPerThread; ++u)
                {
                    int bin = lane + u * WarpThreads;
                    bins[u] = internal::thread_reduce(warp_histograms[bin], sum());
                }
                ::rocprim::syncthreads();

                // store the resulting histogram in shared memory
                int* warp_offsets = &s.warp_offsets[warp][0];
                #pragma unroll
                for (int u = 0; u < WARP_BinsPerThread; ++u)
                {
                    int bin = lane + u * WarpThreads;
                    warp_offsets[bin] = bins[u];
                }
            }
        }

        ROCPRIM_DEVICE inline

        void compute_offsets_warp_upsweep(int (&bins)[BinsPerThread])
        {
            // sum up warp-private histograms
            #pragma unroll
            for (int u = 0; u < BinsPerThread; ++u)
            {
                bins[u] = 0;
                int bin = thread_bin(u);
                if (FullBins || (bin >= 0 && bin < RadixDigits))
                {
                    #pragma unroll
                    for (int j_warp = 0; j_warp < BlockWarps; ++j_warp)
                    {
                        int warp_offset = s.warp_offsets[j_warp][bin];
                        s.warp_offsets[j_warp][bin] = bins[u];
                        bins[u] += warp_offset;
                    }
                }
            }
        }

        ROCPRIM_DEVICE inline

        void compute_offsets_warp_downsweep(int (&offsets)[BinsPerThread])
        {
            #pragma unroll
            for (int u = 0; u < BinsPerThread; ++u)
            {
                int bin = thread_bin(u);
                if (FullBins || (bin >= 0 && bin < RadixDigits))
                {
                    int digit_offset = offsets[u];
                    #pragma unroll
                    for (int j_warp = 0; j_warp < BlockWarps; ++j_warp)
                    {
                        s.warp_offsets[j_warp][bin] += digit_offset;
                    }
                }
            }
        }

        ROCPRIM_DEVICE inline

        void compute_ranks_item(
            UnsignedBits (&keys)[KeysPerThread], int (&ranks)[KeysPerThread],
            Int2Type<warp_match_algorithm::warp_match_atomic_or>)
        {
            // compute key ranks
            int lane_mask = 1 << lane;
            int* warp_offsets = &s.warp_offsets[warp][0];
            int* match_masks = &s.match_masks[warp][0];
            #pragma unroll
            for (int u = 0; u < KeysPerThread; ++u)
            {
                int bin = digit(keys[u]);
                int* p_match_mask = &match_masks[bin];
                atomicOr(p_match_mask, lane_mask);
                // WARP_SYNC(WarpMask);
                int bin_mask = *p_match_mask;
                int leader = (WarpThreads - 1) - __clz(bin_mask);
                int warp_offset = 0;
                int popc = __popc(bin_mask & lane_mask_less_than_equal());
                if (lane == leader)
                {
                    // atomic is a bit faster
                    warp_offset = atomicAdd(&warp_offsets[bin], popc);
                }
                warp_offset = warp_shuffle(warp_offset, leader, bin_mask);
                if (lane == leader) *p_match_mask = 0;
                // WARP_SYNC(WarpMask);
                ranks[u] = warp_offset + popc - 1;
            }
        }

        ROCPRIM_DEVICE inline

        void compute_ranks_item(
            UnsignedBits (&keys)[KeysPerThread], int (&ranks)[KeysPerThread],
            Int2Type<warp_match_algorithm::warp_match_any>)
        {
            // compute key ranks
            int* warp_offsets = &s.warp_offsets[warp][0];
            #pragma unroll
            for (int u = 0; u < KeysPerThread; ++u)
            {
                int bin = digit(keys[u]);
                int bin_mask = MatchAny<RadixBits>(bin);
                int leader = (WarpThreads - 1) - __clz(bin_mask);
                int warp_offset = 0;
                int popc = __popc(bin_mask & lane_mask_less_than_equal());
                if (lane == leader)
                {
                    // atomic is a bit faster
                    warp_offset = atomicAdd(&warp_offsets[bin], popc);
                }
                warp_offset = warp_shuffle(warp_offset, leader, bin_mask);
                ranks[u] = warp_offset + popc - 1;
            }
        }

        ROCPRIM_DEVICE inline
        void rank_keys(
            UnsignedBits (&keys)[KeysPerThread],
            int (&ranks)[KeysPerThread],
            int (&exclusive_digit_prefix)[BinsPerThread])
        {
            compute_histograms_warp(keys);

            ::rocprim::syncthreads();
            int bins[BinsPerThread];
            compute_offsets_warp_upsweep(bins);
            callback(bins);

            block_scan(s.prefix_tmp).exclusive_sum(bins, exclusive_digit_prefix);

            compute_offsets_warp_downsweep(exclusive_digit_prefix);
            ::rocprim::syncthreads();
            compute_ranks_item(keys, ranks, Int2Type<MatchAlgorithm>());
        }

        ROCPRIM_DEVICE inline
        block_radix_rank_match_internal
        (storage_type& temp_storage, DigitExtractorT digit_extractor, CountsCallback callback)
            : s(temp_storage), digit_extractor(digit_extractor),
              callback(callback), warp(threadIdx.x / WarpThreads), lane(lane_id())
            {}
    };

    ROCPRIM_DEVICE inline
    block_radix_rank_match_early_counts
    (storage_type& temp_storage) : temp_storage(temp_storage) {}

    /**
     * \brief Rank keys.  For the lower \p RadixDigits threads, digit counts for each digit are provided for the corresponding thread.
     */
    template <typename UnsignedBits, int KeysPerThread, typename DigitExtractorT,
        typename CountsCallback>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],
        int             (&ranks)[KeysPerThread],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BinsPerThread],
        CountsCallback  callback)
    {
        block_radix_rank_match_internal<UnsignedBits, KeysPerThread, DigitExtractorT, CountsCallback>
            internal(temp_storage, digit_extractor, callback);
        internal.rank_keys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, int KeysPerThread, typename DigitExtractorT>
    ROCPRIM_DEVICE inline
    void rank_keys(
        UnsignedBits    (&keys)[KeysPerThread],
        int             (&ranks)[KeysPerThread],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BinsPerThread])
    {
        typedef block_radix_rank_empty_callback<BinsPerThread> CountsCallback;
        block_radix_rank_match_internal<UnsignedBits, KeysPerThread, DigitExtractorT, CountsCallback>
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
        int exclusive_digit_prefix[BinsPerThread];
        rank_keys(keys, ranks, digit_extractor, exclusive_digit_prefix);
    }
};


END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
