/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_

#include "../../../config.hpp"

#include "../util_type.hpp"

#include <rocprim/functional.hpp>
#include <rocprim/block/block_radix_rank.hpp>

#include "block_scan.hpp"

BEGIN_HIPCUB_NAMESPACE

template <
    int                     BLOCK_DIM_X,
    int                     RADIX_BITS,
    bool                    IS_DESCENDING,
    bool                    MEMOIZE_OUTER_SCAN      = (CUB_PTX_ARCH >= 350) ? true : false,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM    = BLOCK_SCAN_WARP_SCANS,
    cudaSharedMemConfig     SMEM_CONFIG             = cudaSharedMemBankSizeFourByte,
    int                     BLOCK_DIM_Y             = 1,
    int                     BLOCK_DIM_Z             = 1,
    int                     PTX_ARCH                = HIPCUB_ARCH /* ignored */>
class BlockRadixRank
    : private ::rocprim::block_radix_rank<
        BLOCK_DIM_X,
        RADIX_BITS,
        IS_DESCENDING,
        MEMOIZE_OUTER_SCAN,
        INNER_SCAN_ALGORITHM,
        BLOCK_DIM_Y,
        BLOCK_DIM_Z,
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_radix_rank<
            BLOCK_DIM_X,
            RADIX_BITS,
            IS_DESCENDING,
            MEMOIZE_OUTER_SCAN,
            INNER_SCAN_ALGORITHM,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z,
          >;

    // Reference to temporary storage (usually shared memory)
    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockRadixRank() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockRadixRank(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<
    typename        UnsignedBits,
    int             KEYS_PER_THREAD,
    typename        DigitExtractorT>
    HIPCUB_DEVICE inline
    void RankKeys(
    UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
    int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile
    DigitExtractorT digit_extractor)                    ///< [in] The digit extractor
    {
        base_type::rank_keys(keys, ranks, digit_extractor);
    }

    template<
    typename        UnsignedBits,
    int             KEYS_PER_THREAD,
    typename        DigitExtractorT>
    HIPCUB_DEVICE inline
    void RankKeys(
    UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
    int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile
    DigitExtractorT digit_extractor,
    int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD]))                    ///< [in] The digit extractor
    {
        base_type::rank_keys(keys, ranks, digit_extractor, exclusive_digit_prefix);
    }

private:
    HIPCUB_DEVICE inline
    TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};


template <
    int                     BLOCK_DIM_X,
    int                     RADIX_BITS,
    bool                    IS_DESCENDING,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM    = BLOCK_SCAN_WARP_SCANS,
    int                     BLOCK_DIM_Y             = 1,
    int                     BLOCK_DIM_Z             = 1,
    int                     PTX_ARCH                = CUB_PTX_ARCH>
class BlockRadixRankMatch :
      private ::rocprim::block_radix_rank_match
        <BLOCK_DIM_X,
         RADIX_BITS,
         IS_DESCENDING,
         INNER_SCAN_ALGORITHM,
         BLOCK_DIM_Y,
         BLOCK_DIM_Z>
{
  static_assert(
      BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
      "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
  );

  using base_type =
      typename ::rocprim::block_radix_rank_match
        <BLOCK_DIM_X,
         RADIX_BITS,
         IS_DESCENDING,
         INNER_SCAN_ALGORITHM,
         BLOCK_DIM_Y,
         BLOCK_DIM_Z>;

  typename base_type::storage_type& temp_storage_;

public:
   using TempStorage = typename base_type::storage_type;


 private:
     HIPCUB_DEVICE inline
     TempStorage& private_storage()
     {
         HIPCUB_SHARED_MEMORY TempStorage private_storage;
         return private_storage;
     }

     /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    HIPCUB_DEVICE inline BlockRadixRankMatch(TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :temp_storage_(temp_storage) {}

    template <int KEYS_PER_THREAD, typename CountsCallback>
    HIPCUB_DEVICE inline void CallBack(CountsCallback callback)
    {
      base_type::CallBack(callback);
    }
      /**
       * \brief Rank keys.
       */
      template <
      typename        UnsignedBits,
      int             KEYS_PER_THREAD,
      typename        DigitExtractorT,
      typename        CountsCallback>
      HIPCUB_DEVICE inline void RankKeys(
      UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
      int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile
      DigitExtractorT digit_extractor,                    ///< [in] The digit extractor
      CountsCallback    callback)
      {
        base_type::rank_keys(keys,ranks,digit_extractor,callback);
      }

      template <
      typename        UnsignedBits,
      int             KEYS_PER_THREAD,
      typename        DigitExtractorT>
      HIPCUB_DEVICE inline void RankKeys(
      UnsignedBits    (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
      DigitExtractorT digit_extractor)
      {
        base_type::rank_keys(keys,ranks,digit_extractor);
      }

      template <
      typename        UnsignedBits,
      int             KEYS_PER_THREAD,
      typename        DigitExtractorT,
      typename        CountsCallback>
      HIPCUB_DEVICE inline void RankKeys(
      UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
      int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile (out parameter)
      DigitExtractorT digit_extractor,                    ///< [in] The digit extractor
      int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD],            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
      CountsCallback callback)
      {
        base_type::rank_keys(keys,ranks,digit_extractor,exclusive_digit_prefix,callback);
      }

      template <
      typename        UnsignedBits,
      int             KEYS_PER_THREAD,
      typename        DigitExtractorT>
      HIPCUB_DEVICE inline void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
        int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD])            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
      {
        base_type::rank_keys(keys,ranks,digit_extractor,exclusive_digit_prefix);
      }
};

enum WarpMatchAlgorithm
{
    WARP_MATCH_ANY,
    WARP_MATCH_ATOMIC_OR
};

template <int                 BLOCK_DIM_X,
          int                 RADIX_BITS,
          bool                IS_DESCENDING,
          BlockScanAlgorithm  INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS,
          WarpMatchAlgorithm  MATCH_ALGORITHM = WARP_MATCH_ANY,
          int                 NUM_PARTS = 1>
struct BlockRadixRankMatchEarlyCounts : public
    ::rocprim::block_radix_rank_match_early_counts
    <BLOCK_DIM_X
    RADIX_BITS
    IS_DESCENDING
    INNER_SCAN_ALGORITHM
    MATCH_ALGORITHM
    NUM_PARTS>
{

  using base_type =
      typename ::rocprim::block_radix_rank_match_early_counts
      <BLOCK_DIM_X,
       RADIX_BITS,
       IS_DESCENDING,
       INNER_SCAN_ALGORITHM,
       MATCH_ALGORITHM,
       NUM_PARTS>;

   typedef base_type::block_scan BlockScan;

   using TempStorage = typename base_type::storage_type;
   typename base_type::storage_type& temp_storage;

   HIPCUB_DEVICE inline BlockRadixRankMatchEarlyCounts
   (TempStorage& temp_storage) : temp_storage(temp_storage) {}

   template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT,
             typename CountsCallback>
   struct BlockRadixRankMatchInternal : public
        ::rocprim::block_radix_rank_match_internal
   {
      using base_type = typename
      ::rocprim::block_radix_rank_match_internal<
               UnsignedBits,
               KEYS_PER_THREAD,
               DigitExtractorT,
               CountsCallback
              >;

       HIPCUB_DEVICE inline
       int Digit(UnsignedBits key)
       {
         return base_type::digit(key);
       }

       HIPCUB_DEVICE inline
       int ThreadBin(int u)
       {
          return base_type::thread_bin(u);
       }

       HIPCUB_DEVICE inline
       void ComputeHistogramsWarp(UnsignedBits (&keys)[KEYS_PER_THREAD])
       {
          base_type::compute_histograms_warp(keys);
       }

        HIPCUB_DEVICE inline
        void ComputeOffsetsWarpUpsweep(int (&bins)[BINS_PER_THREAD])
        {
          base_type::compute_offsets_warp_upsweep(bins);
        }

        HIPCUB_DEVICE inline
        void ComputeOffsetsWarpDownsweep(int (&offsets)[BINS_PER_THREAD])
        {
            base_type::compute_offsets_warp_downsweep(offsets);
        }


        HIPCUB_DEVICE inline
        void ComputeRanksItem(
            UnsignedBits (&keys)[KEYS_PER_THREAD],
            int (&ranks)[KEYS_PER_THREAD],
            Int2Type<WARP_MATCH_ATOMIC_OR>)
        {
            base_type::compute_ranks_item(keys, ranks,Int2Type<::rocprim::warp_match_algorithm::warp_match_atomic_or> );
        }

        HIPCUB_DEVICE inline
        void ComputeRanksItem(
            UnsignedBits (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
            Int2Type<WARP_MATCH_ANY>)
        {
          base_type::compute_ranks_item(keys, ranks,Int2Type<::rocprim::warp_match_algorithm::warp_match_any> );
        }

        HIPCUB_DEVICE inline void RankKeys(
          UnsignedBits (&keys)[KEYS_PER_THREAD],
          int (&ranks)[KEYS_PER_THREAD],
          int (&exclusive_digit_prefix)[BINS_PER_THREAD])
        {
          base_type::rank_keys(keys,ranks,exclusive_digit_prefix);
        }
};

   template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT,
       typename CountsCallback>
   __device__ __forceinline__ void RankKeys(
       UnsignedBits    (&keys)[KEYS_PER_THREAD],
       int             (&ranks)[KEYS_PER_THREAD],
       DigitExtractorT digit_extractor,
       int             (&exclusive_digit_prefix)[BINS_PER_THREAD],
       CountsCallback  callback)
   {
     base_type::rank_keys(keys,ranks,digit_extractor,exclusive_digit_prefix,callback);
   }

   template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_PER_THREAD])
    {
      base_type::rank_keys(keys,ranks,digit_extractor,exclusive_digit_prefix);
    }

    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor)
    {
      base_type::rank_keys(keys,ranks,digit_extractor);
    }

}

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
