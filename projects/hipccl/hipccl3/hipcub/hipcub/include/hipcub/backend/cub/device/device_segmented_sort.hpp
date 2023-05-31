/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2023, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_CUB_DEVICE_DEVICE_SEGMENTED_SORT_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_SEGMENTED_SORT_HPP_

#include "../../../config.hpp"

#include <cub/device/device_segmented_sort.cuh>

BEGIN_HIPCUB_NAMESPACE

struct DeviceSegmentedSort
{
    template <typename KeyT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t SortKeys(void * d_temp_storage,
                        size_t& temp_storage_bytes,
                        const KeyT * d_keys_in,
                        KeyT * d_keys_out,
                        int num_items,
                        int num_segments,
                        BeginOffsetIteratorT d_begin_offsets,
                        EndOffsetIteratorT d_end_offsets,
                        hipStream_t stream = 0,
                        bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceSegmentedSort::SortKeys(d_temp_storage,
                                                                           temp_storage_bytes,
                                                                           d_keys_in,
                                                                           d_keys_out,
                                                                           num_items,
                                                                           num_segments,
                                                                           d_begin_offsets,
                                                                           d_end_offsets,
                                                                           stream));
    }

    template <typename KeyT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t SortKeysDescending(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  const KeyT * d_keys_in,
                                  KeyT * d_keys_out,
                                  int num_items,
                                  int num_segments,
                                  BeginOffsetIteratorT d_begin_offsets,
                                  EndOffsetIteratorT d_end_offsets,
                                  hipStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::SortKeysDescending(d_temp_storage,
                                                           temp_storage_bytes,
                                                           d_keys_in,
                                                           d_keys_out,
                                                           num_items,
                                                           num_segments,
                                                           d_begin_offsets,
                                                           d_end_offsets,
                                                           stream));
    }

    template <typename KeyT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t SortKeys(void * d_temp_storage,
                        size_t& temp_storage_bytes,
                        DoubleBuffer<KeyT> &d_keys,
                        int num_items,
                        int num_segments,
                        BeginOffsetIteratorT d_begin_offsets,
                        EndOffsetIteratorT d_end_offsets,
                        hipStream_t stream = 0,
                        bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceSegmentedSort::SortKeys(d_temp_storage,
                                                                           temp_storage_bytes,
                                                                           d_keys,
                                                                           num_items,
                                                                           num_segments,
                                                                           d_begin_offsets,
                                                                           d_end_offsets,
                                                                           stream));
    }
    
    template <typename KeyT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t SortKeysDescending(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  DoubleBuffer<KeyT> &d_keys,
                                  int num_items,
                                  int num_segments,
                                  BeginOffsetIteratorT d_begin_offsets,
                                  EndOffsetIteratorT d_end_offsets,
                                  hipStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::SortKeysDescending(d_temp_storage,
                                                           temp_storage_bytes,
                                                           d_keys,
                                                           num_items,
                                                           num_segments,
                                                           d_begin_offsets,
                                                           d_end_offsets,
                                                           stream));
    }

    template <typename KeyT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t StableSortKeys(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              const KeyT * d_keys_in,
                              KeyT * d_keys_out,
                              int num_items,
                              int num_segments,
                              BeginOffsetIteratorT d_begin_offsets,
                              EndOffsetIteratorT d_end_offsets,
                              hipStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceSegmentedSort::StableSortKeys(d_temp_storage,
                                                                                 temp_storage_bytes,
                                                                                 d_keys_in,
                                                                                 d_keys_out,
                                                                                 num_items,
                                                                                 num_segments,
                                                                                 d_begin_offsets,
                                                                                 d_end_offsets,
                                                                                 stream));
    }

    template <typename KeyT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t StableSortKeysDescending(void * d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        const KeyT * d_keys_in,
                                        KeyT * d_keys_out,
                                        int num_items,
                                        int num_segments,
                                        BeginOffsetIteratorT d_begin_offsets,
                                        EndOffsetIteratorT d_end_offsets,
                                        hipStream_t stream = 0,
                                        bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::StableSortKeysDescending(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_keys_in,
                                                                 d_keys_out,
                                                                 num_items,
                                                                 num_segments,
                                                                 d_begin_offsets,
                                                                 d_end_offsets,
                                                                 stream));
    }

    template <typename KeyT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t StableSortKeys(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              DoubleBuffer<KeyT> &d_keys,
                              int num_items,
                              int num_segments,
                              BeginOffsetIteratorT d_begin_offsets,
                              EndOffsetIteratorT d_end_offsets,
                              hipStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceSegmentedSort::StableSortKeys(d_temp_storage,
                                                                                 temp_storage_bytes,
                                                                                 d_keys,
                                                                                 num_items,
                                                                                 num_segments,
                                                                                 d_begin_offsets,
                                                                                 d_end_offsets,
                                                                                 stream));
    }
    
    template <typename KeyT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t StableSortKeysDescending(void * d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        DoubleBuffer<KeyT> &d_keys,
                                        int num_items,
                                        int num_segments,
                                        BeginOffsetIteratorT d_begin_offsets,
                                        EndOffsetIteratorT d_end_offsets,
                                        hipStream_t stream = 0,
                                        bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::StableSortKeysDescending(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_keys,
                                                                 num_items,
                                                                 num_segments,
                                                                 d_begin_offsets,
                                                                 d_end_offsets,
                                                                 stream));
    }

    template <typename KeyT,
              typename ValueT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t SortPairs(void * d_temp_storage,
                         size_t& temp_storage_bytes,
                         const KeyT * d_keys_in,
                         KeyT * d_keys_out,
                         const ValueT * d_values_in,
                         ValueT * d_values_out,
                         int num_items,
                         int num_segments,
                         BeginOffsetIteratorT d_begin_offsets,
                         EndOffsetIteratorT d_end_offsets,
                         hipStream_t stream = 0,
                         bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceSegmentedSort::SortPairs(d_temp_storage,
                                                                            temp_storage_bytes,
                                                                            d_keys_in,
                                                                            d_keys_out,
                                                                            d_values_in,
                                                                            d_values_out,
                                                                            num_items,
                                                                            num_segments,
                                                                            d_begin_offsets,
                                                                            d_end_offsets,
                                                                            stream));
    }

    template <typename KeyT,
              typename ValueT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t SortPairsDescending(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   const KeyT * d_keys_in,
                                   KeyT * d_keys_out,
                                   const ValueT * d_values_in,
                                   ValueT * d_values_out,
                                   int num_items,
                                   int num_segments,
                                   BeginOffsetIteratorT d_begin_offsets,
                                   EndOffsetIteratorT d_end_offsets,
                                   hipStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::SortPairsDescending(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_keys_in,
                                                            d_keys_out,
                                                            d_values_in,
                                                            d_values_out,
                                                            num_items,
                                                            num_segments,
                                                            d_begin_offsets,
                                                            d_end_offsets,
                                                            stream));
    }

    template <typename KeyT,
              typename ValueT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t SortPairs(void * d_temp_storage,
                         size_t& temp_storage_bytes,
                         DoubleBuffer<KeyT> &d_keys,
                         DoubleBuffer<ValueT> &d_values,
                         int num_items,
                         int num_segments,
                         BeginOffsetIteratorT d_begin_offsets,
                         EndOffsetIteratorT d_end_offsets,
                         hipStream_t stream = 0,
                         bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DeviceSegmentedSort::SortPairs(d_temp_storage,
                                                                            temp_storage_bytes,
                                                                            d_keys,
                                                                            d_values,
                                                                            num_items,
                                                                            num_segments,
                                                                            d_begin_offsets,
                                                                            d_end_offsets,
                                                                            stream));
    }
    
    template <typename KeyT,
              typename ValueT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t SortPairsDescending(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   DoubleBuffer<KeyT> &d_keys,
                                   DoubleBuffer<ValueT> &d_values,
                                   int num_items,
                                   int num_segments,
                                   BeginOffsetIteratorT d_begin_offsets,
                                   EndOffsetIteratorT d_end_offsets,
                                   hipStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::SortPairsDescending(d_temp_storage,
                                                            temp_storage_bytes,
                                                            d_keys,
                                                            d_values,
                                                            num_items,
                                                            num_segments,
                                                            d_begin_offsets,
                                                            d_end_offsets,
                                                            stream));
    }

    template <typename KeyT,
              typename ValueT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t StableSortPairs(void * d_temp_storage,
                               size_t& temp_storage_bytes,
                               const KeyT * d_keys_in,
                               KeyT * d_keys_out,
                               const ValueT * d_values_in,
                               ValueT * d_values_out,
                               int num_items,
                               int num_segments,
                               BeginOffsetIteratorT d_begin_offsets,
                               EndOffsetIteratorT d_end_offsets,
                               hipStream_t stream = 0,
                               bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys_in,
                                                        d_keys_out,
                                                        d_values_in,
                                                        d_values_out,
                                                        num_items,
                                                        num_segments,
                                                        d_begin_offsets,
                                                        d_end_offsets,
                                                        stream));
    }

    template <typename KeyT,
              typename ValueT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t StableSortPairsDescending(void * d_temp_storage,
                                         size_t& temp_storage_bytes,
                                         const KeyT * d_keys_in,
                                         KeyT * d_keys_out,
                                         const ValueT * d_values_in,
                                         ValueT * d_values_out,
                                         int num_items,
                                         int num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT d_end_offsets,
                                         hipStream_t stream = 0,
                                         bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_keys_in,
                                                                  d_keys_out,
                                                                  d_values_in,
                                                                  d_values_out,
                                                                  num_items,
                                                                  num_segments,
                                                                  d_begin_offsets,
                                                                  d_end_offsets,
                                                                  stream));
    }

    template <typename KeyT,
              typename ValueT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t StableSortPairs(void * d_temp_storage,
                               size_t& temp_storage_bytes,
                               DoubleBuffer<KeyT> &d_keys,
                               DoubleBuffer<ValueT> &d_values,
                               int num_items,
                               int num_segments,
                               BeginOffsetIteratorT d_begin_offsets,
                               EndOffsetIteratorT d_end_offsets,
                               hipStream_t stream = 0,
                               bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys,
                                                        d_values,
                                                        num_items,
                                                        num_segments,
                                                        d_begin_offsets,
                                                        d_end_offsets,
                                                        stream));
    }
    
    template <typename KeyT,
              typename ValueT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t StableSortPairsDescending(void * d_temp_storage,
                                         size_t& temp_storage_bytes,
                                         DoubleBuffer<KeyT> &d_keys,
                                         DoubleBuffer<ValueT> &d_values,
                                         int num_items,
                                         int num_segments,
                                         BeginOffsetIteratorT d_begin_offsets,
                                         EndOffsetIteratorT d_end_offsets,
                                         hipStream_t stream = 0,
                                         bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  d_keys,
                                                                  d_values,
                                                                  num_items,
                                                                  num_segments,
                                                                  d_begin_offsets,
                                                                  d_end_offsets,
                                                                  stream));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_SEGMENTED_SORT_HPP_
