/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2022, Advanced Micro Devices, Inc.  All rights reserved.
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

/**
 * \file
 * hipcub::BlockRadixRank provides operations for ranking unsigned integer types within a CUDA thread block
 */

 #ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
 #define HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_

#include <stdint.h>

#include "../../../config.hpp"
#include "../../../util_type.hpp"
#include "../../../util_ptx.hpp"

#include "../block/block_scan.hpp"
#include "../block/radix_rank_sort_operations.hpp"
#include "../thread/thread_reduce.hpp"
#include "../thread/thread_scan.hpp"

#include <rocprim/block/block_radix_rank.hpp>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
template<typename DigitExtractorT, typename UnsignedBits, int RADIX_BITS, bool IS_DESCENDING>
struct DigitExtractorAdopter
{
    DigitExtractorT& digit_extractor_;

    HIPCUB_DEVICE DigitExtractorAdopter(DigitExtractorT& digit_extractor)
        : digit_extractor_(digit_extractor)
    {}

    HIPCUB_DEVICE inline UnsignedBits operator()(const UnsignedBits key)
    {
        UnsignedBits digit = digit_extractor_.Digit(key);
        if(IS_DESCENDING)
        {
            // Flip digit bits
            digit ^= (1 << RADIX_BITS) - 1;
        }
        return digit;
    }
};
} // namespace detail

/**
 * \brief BlockRadixRank provides operations for ranking unsigned integer types within a CUDA thread block.
 * \ingroup BlockModule
 *
 * \tparam BLOCK_DIM_X          The thread block length in threads along the X dimension
 * \tparam RADIX_BITS           The number of radix bits per digit place
 * \tparam IS_DESCENDING           Whether or not the sorted-order is high-to-low
 * \tparam MEMOIZE_OUTER_SCAN   <b>[optional]</b> Whether or not to buffer outer raking scan partials to incur fewer shared memory reads at the expense of higher register pressure (default: true for architectures SM35 and newer, false otherwise).  See BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE for more details.
 * \tparam INNER_SCAN_ALGORITHM <b>[optional]</b> The hipcub::BlockScanAlgorithm algorithm to use (default: hipcub::BLOCK_SCAN_WARP_SCANS)
 * \tparam SMEM_CONFIG          <b>[optional]</b> Shared memory bank mode (default: \p hipSharedMemBankSizeFourByte)
 * \tparam BLOCK_DIM_Y          <b>[optional]</b> The thread block length in threads along the Y dimension (default: 1)
 * \tparam BLOCK_DIM_Z          <b>[optional]</b> The thread block length in threads along the Z dimension (default: 1)
 * \tparam ARCH                 <b>[optional]</b> ptx version
 *
 * \par Overview
 * Blah...
 * - Keys must be in a form suitable for radix ranking (i.e., unsigned bits).
 *
 * \par Performance Considerations
 * - granularity
 *
 * \par Examples
 * \par
 * - <b>Example 1:</b> Simple radix rank of 32-bit integer keys
 *      \code
 *      #include <hipcub/hipcub.hpp>
 *
 *      template <int BLOCK_THREADS>
 *      __global__ void ExampleKernel(...)
 *      {
 *
 *      \endcode
 */
template<int                BLOCK_DIM_X,
         int                RADIX_BITS,
         bool               IS_DESCENDING,
         bool               MEMOIZE_OUTER_SCAN   = false,
         BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS,
         hipSharedMemConfig SMEM_CONFIG          = hipSharedMemBankSizeFourByte,
         int                BLOCK_DIM_Y          = 1,
         int                BLOCK_DIM_Z          = 1,
         int                ARCH                 = HIPCUB_ARCH /* ignored */>
class BlockRadixRank
    : private ::rocprim::block_radix_rank<BLOCK_DIM_X,
                                          RADIX_BITS,
                                          MEMOIZE_OUTER_SCAN
                                              ? ::rocprim::block_radix_rank_algorithm::basic_memoize
                                              : ::rocprim::block_radix_rank_algorithm::basic,
                                          BLOCK_DIM_Y,
                                          BLOCK_DIM_Z>
{
    static_assert(BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
                  "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0");

    using base_type
        = ::rocprim::block_radix_rank<BLOCK_DIM_X,
                                      RADIX_BITS,
                                      MEMOIZE_OUTER_SCAN
                                          ? ::rocprim::block_radix_rank_algorithm::basic_memoize
                                          : ::rocprim::block_radix_rank_algorithm::basic,
                                      BLOCK_DIM_Y,
                                      BLOCK_DIM_Z>;

public:
    using TempStorage = typename base_type::storage_type;

private:
    // Reference to temporary storage (usually shared memory)
    TempStorage& temp_storage_;

    HIPCUB_DEVICE inline TempStorage& PrivateStorage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }

public:
    enum
    {
        /// Number of bin-starting offsets tracked per thread
        BINS_TRACKED_PER_THREAD = base_type::digits_per_thread,
    };

    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    HIPCUB_DEVICE inline BlockRadixRank() : temp_storage_(PrivateStorage()) {}

    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    HIPCUB_DEVICE inline BlockRadixRank(
        TempStorage&
            temp_storage) ///< [in] Reference to memory allocation having layout type TempStorage
        : temp_storage_(temp_storage)
    {}

    //@}  end member group
    /******************************************************************/ /**
     * \name Ranking
     *********************************************************************/
    //@{

    /**
     * \brief Rank keys.
     */
    template<typename UnsignedBits,
             int KEYS_PER_THREAD,
             typename DigitExtractorT>
    HIPCUB_DEVICE inline void RankKeys(
        UnsignedBits (&keys)[KEYS_PER_THREAD], ///< [in] Keys for this tile
        int (&ranks)[KEYS_PER_THREAD], ///< [out] For each key, the local rank within the tile
        DigitExtractorT digit_extractor) ///< [in] The digit extractor
    {
        detail::DigitExtractorAdopter<DigitExtractorT, UnsignedBits, RADIX_BITS, IS_DESCENDING>
            digit_extractor_adopter(digit_extractor);
        base_type::rank_keys(keys,
                             reinterpret_cast<unsigned int(&)[KEYS_PER_THREAD]>(ranks),
                             temp_storage_,
                             digit_extractor_adopter);
    }

    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template<typename UnsignedBits,
             int KEYS_PER_THREAD,
             typename DigitExtractorT>
    HIPCUB_DEVICE inline void RankKeys(
        UnsignedBits (&keys)[KEYS_PER_THREAD], ///< [in] Keys for this tile
        int (&ranks)
            [KEYS_PER_THREAD], ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor, ///< [in] The digit extractor
        int (&exclusive_digit_prefix)
            [BINS_TRACKED_PER_THREAD]) ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
    {
        unsigned int counts[BINS_TRACKED_PER_THREAD];
        detail::DigitExtractorAdopter<DigitExtractorT, UnsignedBits, RADIX_BITS, IS_DESCENDING>
            digit_extractor_adopter(digit_extractor);
        base_type::rank_keys(
            keys,
            reinterpret_cast<unsigned int(&)[KEYS_PER_THREAD]>(ranks),
            temp_storage_,
            digit_extractor_adopter,
            reinterpret_cast<unsigned int(&)[BINS_TRACKED_PER_THREAD]>(exclusive_digit_prefix),
            counts);
    }
};

/**
 * Radix-rank using match.any
 */
template<int                BLOCK_DIM_X,
         int                RADIX_BITS,
         bool               IS_DESCENDING,
         BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS,
         int                BLOCK_DIM_Y          = 1,
         int                BLOCK_DIM_Z          = 1,
         int                ARCH                 = HIPCUB_ARCH>
class BlockRadixRankMatch
    : private ::rocprim::block_radix_rank<BLOCK_DIM_X,
                                          RADIX_BITS,
                                          ::rocprim::block_radix_rank_algorithm::match,
                                          BLOCK_DIM_Y,
                                          BLOCK_DIM_Z>
{
    static_assert(BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
                  "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0");

    using base_type = ::rocprim::block_radix_rank<BLOCK_DIM_X,
                                                  RADIX_BITS,
                                                  ::rocprim::block_radix_rank_algorithm::match,
                                                  BLOCK_DIM_Y,
                                                  BLOCK_DIM_Z>;

public:
    using TempStorage = typename base_type::storage_type;

private:
    // Reference to temporary storage (usually shared memory)
    TempStorage& temp_storage_;

    HIPCUB_DEVICE inline TempStorage& PrivateStorage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }

public:
    enum
    {
        /// Number of bin-starting offsets tracked per thread
        BINS_TRACKED_PER_THREAD = base_type::digits_per_thread,
    };

    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    HIPCUB_DEVICE inline BlockRadixRankMatch() : temp_storage_(PrivateStorage()) {}

    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    HIPCUB_DEVICE inline BlockRadixRankMatch(
        TempStorage&
            temp_storage) ///< [in] Reference to memory allocation having layout type TempStorage
        : temp_storage_(temp_storage)
    {}

    //@}  end member group
    /******************************************************************//**
     * \name Raking
     *********************************************************************/
    //@{

    /**
     * \brief Rank keys.
     */
    template<typename UnsignedBits,
             int KEYS_PER_THREAD,
             typename DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits (&keys)[KEYS_PER_THREAD], ///< [in] Keys for this tile
        int (&ranks)[KEYS_PER_THREAD], ///< [out] For each key, the local rank within the tile
        DigitExtractorT digit_extractor) ///< [in] The digit extractor
    {
        detail::DigitExtractorAdopter<DigitExtractorT, UnsignedBits, RADIX_BITS, IS_DESCENDING>
            digit_extractor_adopter(digit_extractor);
        base_type::rank_keys(keys,
                             reinterpret_cast<unsigned int(&)[KEYS_PER_THREAD]>(ranks),
                             temp_storage_,
                             digit_extractor_adopter);
    }

    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template<typename UnsignedBits,
             int KEYS_PER_THREAD,
             typename DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits (&keys)[KEYS_PER_THREAD], ///< [in] Keys for this tile
        int (&ranks)
            [KEYS_PER_THREAD], ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor, ///< [in] The digit extractor
        int (&exclusive_digit_prefix)
            [BINS_TRACKED_PER_THREAD]) ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
    {
        unsigned int counts[BINS_TRACKED_PER_THREAD];
        detail::DigitExtractorAdopter<DigitExtractorT, UnsignedBits, RADIX_BITS, IS_DESCENDING>
            digit_extractor_adopter(digit_extractor);
        base_type::rank_keys(
            keys,
            reinterpret_cast<unsigned int(&)[KEYS_PER_THREAD]>(ranks),
            temp_storage_,
            digit_extractor_adopter,
            reinterpret_cast<unsigned int(&)[BINS_TRACKED_PER_THREAD]>(exclusive_digit_prefix),
            counts);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
