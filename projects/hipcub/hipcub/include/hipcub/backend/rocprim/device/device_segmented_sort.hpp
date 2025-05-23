/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_SORT_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_SORT_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include "../util_type.hpp"

#include <rocprim/device/device_segmented_radix_sort.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

struct DeviceSegmentedSort
{
    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortPairs(void*           d_temp_storage,
                                                        size_t&         temp_storage_bytes,
                                                        const KeyT*     d_keys_in,
                                                        KeyT*           d_keys_out,
                                                        const ValueT*   d_values_in,
                                                        ValueT*         d_values_out,
                                                        int             num_items,
                                                        int             num_segments,
                                                        OffsetIteratorT d_begin_offsets,
                                                        OffsetIteratorT d_end_offsets,
                                                        hipStream_t     stream = 0)
    {
        return ::rocprim::segmented_radix_sort_pairs(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_keys_in,
                                                     d_keys_out,
                                                     d_values_in,
                                                     d_values_out,
                                                     num_items,
                                                     num_segments,
                                                     d_begin_offsets,
                                                     d_end_offsets,
                                                     0,
                                                     sizeof(KeyT) * 8,
                                                     stream,
                                                     HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        SortPairs(void*           d_temp_storage,
                  size_t&         temp_storage_bytes,
                  const KeyT*     d_keys_in,
                  KeyT*           d_keys_out,
                  const ValueT*   d_values_in,
                  ValueT*         d_values_out,
                  int             num_items,
                  int             num_segments,
                  OffsetIteratorT d_begin_offsets,
                  OffsetIteratorT d_end_offsets,
                  hipStream_t     stream,
                  bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SortPairs(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_keys_out,
                         d_values_in,
                         d_values_out,
                         num_items,
                         num_segments,
                         d_begin_offsets,
                         d_end_offsets,
                         stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortPairs(void*                 d_temp_storage,
                                                        size_t&               temp_storage_bytes,
                                                        DoubleBuffer<KeyT>&   d_keys,
                                                        DoubleBuffer<ValueT>& d_values,
                                                        int                   num_items,
                                                        int                   num_segments,
                                                        OffsetIteratorT       d_begin_offsets,
                                                        OffsetIteratorT       d_end_offsets,
                                                        hipStream_t           stream = 0)
    {
        ::rocprim::double_buffer<KeyT>   d_keys_db   = detail::to_double_buffer(d_keys);
        ::rocprim::double_buffer<ValueT> d_values_db = detail::to_double_buffer(d_values);
        hipError_t error = ::rocprim::segmented_radix_sort_pairs(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_keys_db,
                                                                 d_values_db,
                                                                 num_items,
                                                                 num_segments,
                                                                 d_begin_offsets,
                                                                 d_end_offsets,
                                                                 0,
                                                                 sizeof(KeyT) * 8,
                                                                 stream,
                                                                 HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
        detail::update_double_buffer(d_keys, d_keys_db);
        detail::update_double_buffer(d_values, d_values_db);
        return error;
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        SortPairs(void*                 d_temp_storage,
                  size_t&               temp_storage_bytes,
                  DoubleBuffer<KeyT>&   d_keys,
                  DoubleBuffer<ValueT>& d_values,
                  int                   num_items,
                  int                   num_segments,
                  OffsetIteratorT       d_begin_offsets,
                  OffsetIteratorT       d_end_offsets,
                  hipStream_t           stream,
                  bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SortPairs(d_temp_storage,
                         temp_storage_bytes,
                         d_keys,
                         d_values,
                         num_items,
                         num_segments,
                         d_begin_offsets,
                         d_end_offsets,
                         stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortPairsDescending(void*         d_temp_storage,
                                                                  size_t&       temp_storage_bytes,
                                                                  const KeyT*   d_keys_in,
                                                                  KeyT*         d_keys_out,
                                                                  const ValueT* d_values_in,
                                                                  ValueT*       d_values_out,
                                                                  int           num_items,
                                                                  int           num_segments,
                                                                  OffsetIteratorT d_begin_offsets,
                                                                  OffsetIteratorT d_end_offsets,
                                                                  hipStream_t     stream = 0)
    {
        return ::rocprim::segmented_radix_sort_pairs_desc(d_temp_storage,
                                                          temp_storage_bytes,
                                                          d_keys_in,
                                                          d_keys_out,
                                                          d_values_in,
                                                          d_values_out,
                                                          num_items,
                                                          num_segments,
                                                          d_begin_offsets,
                                                          d_end_offsets,
                                                          0,
                                                          sizeof(KeyT) * 8,
                                                          stream,
                                                          HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        SortPairsDescending(void*           d_temp_storage,
                            size_t&         temp_storage_bytes,
                            const KeyT*     d_keys_in,
                            KeyT*           d_keys_out,
                            const ValueT*   d_values_in,
                            ValueT*         d_values_out,
                            int             num_items,
                            int             num_segments,
                            OffsetIteratorT d_begin_offsets,
                            OffsetIteratorT d_end_offsets,
                            hipStream_t     stream,
                            bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SortPairsDescending(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys_in,
                                   d_keys_out,
                                   d_values_in,
                                   d_values_out,
                                   num_items,
                                   num_segments,
                                   d_begin_offsets,
                                   d_end_offsets,
                                   stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortPairsDescending(void*   d_temp_storage,
                                                                  size_t& temp_storage_bytes,
                                                                  DoubleBuffer<KeyT>&   d_keys,
                                                                  DoubleBuffer<ValueT>& d_values,
                                                                  int                   num_items,
                                                                  int             num_segments,
                                                                  OffsetIteratorT d_begin_offsets,
                                                                  OffsetIteratorT d_end_offsets,
                                                                  hipStream_t     stream = 0)
    {
        ::rocprim::double_buffer<KeyT>   d_keys_db   = detail::to_double_buffer(d_keys);
        ::rocprim::double_buffer<ValueT> d_values_db = detail::to_double_buffer(d_values);
        hipError_t                       error
            = ::rocprim::segmented_radix_sort_pairs_desc(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys_db,
                                                         d_values_db,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         0,
                                                         sizeof(KeyT) * 8,
                                                         stream,
                                                         HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
        detail::update_double_buffer(d_keys, d_keys_db);
        detail::update_double_buffer(d_values, d_values_db);
        return error;
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        SortPairsDescending(void*                 d_temp_storage,
                            size_t&               temp_storage_bytes,
                            DoubleBuffer<KeyT>&   d_keys,
                            DoubleBuffer<ValueT>& d_values,
                            int                   num_items,
                            int                   num_segments,
                            OffsetIteratorT       d_begin_offsets,
                            OffsetIteratorT       d_end_offsets,
                            hipStream_t           stream,
                            bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SortPairsDescending(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys,
                                   d_values,
                                   num_items,
                                   num_segments,
                                   d_begin_offsets,
                                   d_end_offsets,
                                   stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortKeys(void*           d_temp_storage,
                                                       size_t&         temp_storage_bytes,
                                                       const KeyT*     d_keys_in,
                                                       KeyT*           d_keys_out,
                                                       int             num_items,
                                                       int             num_segments,
                                                       OffsetIteratorT d_begin_offsets,
                                                       OffsetIteratorT d_end_offsets,
                                                       hipStream_t     stream = 0)
    {
        return ::rocprim::segmented_radix_sort_keys(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_keys_in,
                                                    d_keys_out,
                                                    num_items,
                                                    num_segments,
                                                    d_begin_offsets,
                                                    d_end_offsets,
                                                    0,
                                                    sizeof(KeyT) * 8,
                                                    stream,
                                                    HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        SortKeys(void*           d_temp_storage,
                 size_t&         temp_storage_bytes,
                 const KeyT*     d_keys_in,
                 KeyT*           d_keys_out,
                 int             num_items,
                 int             num_segments,
                 OffsetIteratorT d_begin_offsets,
                 OffsetIteratorT d_end_offsets,
                 hipStream_t     stream,
                 bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SortKeys(d_temp_storage,
                        temp_storage_bytes,
                        d_keys_in,
                        d_keys_out,
                        num_items,
                        num_segments,
                        d_begin_offsets,
                        d_end_offsets,
                        stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortKeys(void*               d_temp_storage,
                                                       size_t&             temp_storage_bytes,
                                                       DoubleBuffer<KeyT>& d_keys,
                                                       int                 num_items,
                                                       int                 num_segments,
                                                       OffsetIteratorT     d_begin_offsets,
                                                       OffsetIteratorT     d_end_offsets,
                                                       hipStream_t         stream = 0)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        hipError_t                     error = ::rocprim::segmented_radix_sort_keys(d_temp_storage,
                                                                temp_storage_bytes,
                                                                d_keys_db,
                                                                num_items,
                                                                num_segments,
                                                                d_begin_offsets,
                                                                d_end_offsets,
                                                                0,
                                                                sizeof(KeyT) * 8,
                                                                stream,
                                                                HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
        detail::update_double_buffer(d_keys, d_keys_db);
        return error;
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        SortKeys(void*               d_temp_storage,
                 size_t&             temp_storage_bytes,
                 DoubleBuffer<KeyT>& d_keys,
                 int                 num_items,
                 int                 num_segments,
                 OffsetIteratorT     d_begin_offsets,
                 OffsetIteratorT     d_end_offsets,
                 hipStream_t         stream,
                 bool                debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SortKeys(d_temp_storage,
                        temp_storage_bytes,
                        d_keys,
                        num_items,
                        num_segments,
                        d_begin_offsets,
                        d_end_offsets,
                        stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortKeysDescending(void*           d_temp_storage,
                                                                 size_t&         temp_storage_bytes,
                                                                 const KeyT*     d_keys_in,
                                                                 KeyT*           d_keys_out,
                                                                 int             num_items,
                                                                 int             num_segments,
                                                                 OffsetIteratorT d_begin_offsets,
                                                                 OffsetIteratorT d_end_offsets,
                                                                 hipStream_t     stream = 0)
    {
        return ::rocprim::segmented_radix_sort_keys_desc(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys_in,
                                                         d_keys_out,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         0,
                                                         sizeof(KeyT) * 8,
                                                         stream,
                                                         HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        SortKeysDescending(void*           d_temp_storage,
                           size_t&         temp_storage_bytes,
                           const KeyT*     d_keys_in,
                           KeyT*           d_keys_out,
                           int             num_items,
                           int             num_segments,
                           OffsetIteratorT d_begin_offsets,
                           OffsetIteratorT d_end_offsets,
                           hipStream_t     stream,
                           bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SortKeysDescending(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys_in,
                                  d_keys_out,
                                  num_items,
                                  num_segments,
                                  d_begin_offsets,
                                  d_end_offsets,
                                  stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t SortKeysDescending(void*   d_temp_storage,
                                                                 size_t& temp_storage_bytes,
                                                                 DoubleBuffer<KeyT>& d_keys,
                                                                 int                 num_items,
                                                                 int                 num_segments,
                                                                 OffsetIteratorT d_begin_offsets,
                                                                 OffsetIteratorT d_end_offsets,
                                                                 hipStream_t     stream = 0)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        hipError_t                     error
            = ::rocprim::segmented_radix_sort_keys_desc(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys_db,
                                                        num_items,
                                                        num_segments,
                                                        d_begin_offsets,
                                                        d_end_offsets,
                                                        0,
                                                        sizeof(KeyT) * 8,
                                                        stream,
                                                        HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
        detail::update_double_buffer(d_keys, d_keys_db);
        return error;
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        SortKeysDescending(void*               d_temp_storage,
                           size_t&             temp_storage_bytes,
                           DoubleBuffer<KeyT>& d_keys,
                           int                 num_items,
                           int                 num_segments,
                           OffsetIteratorT     d_begin_offsets,
                           OffsetIteratorT     d_end_offsets,
                           hipStream_t         stream,
                           bool                debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return SortKeysDescending(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys,
                                  num_items,
                                  num_segments,
                                  d_begin_offsets,
                                  d_end_offsets,
                                  stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t StableSortPairs(void*           d_temp_storage,
                                                              size_t&         temp_storage_bytes,
                                                              const KeyT*     d_keys_in,
                                                              KeyT*           d_keys_out,
                                                              const ValueT*   d_values_in,
                                                              ValueT*         d_values_out,
                                                              int             num_items,
                                                              int             num_segments,
                                                              OffsetIteratorT d_begin_offsets,
                                                              OffsetIteratorT d_end_offsets,
                                                              hipStream_t     stream = 0)
    {
        return SortPairs(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_keys_out,
                         d_values_in,
                         d_values_out,
                         num_items,
                         num_segments,
                         d_begin_offsets,
                         d_end_offsets,
                         stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortPairs(void*           d_temp_storage,
                        size_t&         temp_storage_bytes,
                        const KeyT*     d_keys_in,
                        KeyT*           d_keys_out,
                        const ValueT*   d_values_in,
                        ValueT*         d_values_out,
                        int             num_items,
                        int             num_segments,
                        OffsetIteratorT d_begin_offsets,
                        OffsetIteratorT d_end_offsets,
                        hipStream_t     stream,
                        bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return StableSortPairs(d_temp_storage,
                               temp_storage_bytes,
                               d_keys_in,
                               d_keys_out,
                               d_values_in,
                               d_values_out,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t StableSortPairs(void*   d_temp_storage,
                                                              size_t& temp_storage_bytes,
                                                              DoubleBuffer<KeyT>&   d_keys,
                                                              DoubleBuffer<ValueT>& d_values,
                                                              int                   num_items,
                                                              int                   num_segments,
                                                              OffsetIteratorT       d_begin_offsets,
                                                              OffsetIteratorT       d_end_offsets,
                                                              hipStream_t           stream = 0)
    {
        return SortPairs(d_temp_storage,
                         temp_storage_bytes,
                         d_keys,
                         d_values,
                         num_items,
                         num_segments,
                         d_begin_offsets,
                         d_end_offsets,
                         stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortPairs(void*                 d_temp_storage,
                        size_t&               temp_storage_bytes,
                        DoubleBuffer<KeyT>&   d_keys,
                        DoubleBuffer<ValueT>& d_values,
                        int                   num_items,
                        int                   num_segments,
                        OffsetIteratorT       d_begin_offsets,
                        OffsetIteratorT       d_end_offsets,
                        hipStream_t           stream,
                        bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return StableSortPairs(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortPairsDescending(void*           d_temp_storage,
                                  size_t&         temp_storage_bytes,
                                  const KeyT*     d_keys_in,
                                  KeyT*           d_keys_out,
                                  const ValueT*   d_values_in,
                                  ValueT*         d_values_out,
                                  int             num_items,
                                  int             num_segments,
                                  OffsetIteratorT d_begin_offsets,
                                  OffsetIteratorT d_end_offsets,
                                  hipStream_t     stream = 0)
    {
        return SortPairsDescending(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys_in,
                                   d_keys_out,
                                   d_values_in,
                                   d_values_out,
                                   num_items,
                                   num_segments,
                                   d_begin_offsets,
                                   d_end_offsets,
                                   stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortPairsDescending(void*           d_temp_storage,
                                  size_t&         temp_storage_bytes,
                                  const KeyT*     d_keys_in,
                                  KeyT*           d_keys_out,
                                  const ValueT*   d_values_in,
                                  ValueT*         d_values_out,
                                  int             num_items,
                                  int             num_segments,
                                  OffsetIteratorT d_begin_offsets,
                                  OffsetIteratorT d_end_offsets,
                                  hipStream_t     stream,
                                  bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return StableSortPairsDescending(d_temp_storage,
                                         temp_storage_bytes,
                                         d_keys_in,
                                         d_keys_out,
                                         d_values_in,
                                         d_values_out,
                                         num_items,
                                         num_segments,
                                         d_begin_offsets,
                                         d_end_offsets,
                                         stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortPairsDescending(void*                 d_temp_storage,
                                  size_t&               temp_storage_bytes,
                                  DoubleBuffer<KeyT>&   d_keys,
                                  DoubleBuffer<ValueT>& d_values,
                                  int                   num_items,
                                  int                   num_segments,
                                  OffsetIteratorT       d_begin_offsets,
                                  OffsetIteratorT       d_end_offsets,
                                  hipStream_t           stream = 0)
    {
        return SortPairsDescending(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys,
                                   d_values,
                                   num_items,
                                   num_segments,
                                   d_begin_offsets,
                                   d_end_offsets,
                                   stream);
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortPairsDescending(void*                 d_temp_storage,
                                  size_t&               temp_storage_bytes,
                                  DoubleBuffer<KeyT>&   d_keys,
                                  DoubleBuffer<ValueT>& d_values,
                                  int                   num_items,
                                  int                   num_segments,
                                  OffsetIteratorT       d_begin_offsets,
                                  OffsetIteratorT       d_end_offsets,
                                  hipStream_t           stream,
                                  bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return StableSortPairsDescending(d_temp_storage,
                                         temp_storage_bytes,
                                         d_keys,
                                         d_values,
                                         num_items,
                                         num_segments,
                                         d_begin_offsets,
                                         d_end_offsets,
                                         stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t StableSortKeys(void*           d_temp_storage,
                                                             size_t&         temp_storage_bytes,
                                                             const KeyT*     d_keys_in,
                                                             KeyT*           d_keys_out,
                                                             int             num_items,
                                                             int             num_segments,
                                                             OffsetIteratorT d_begin_offsets,
                                                             OffsetIteratorT d_end_offsets,
                                                             hipStream_t     stream = 0)
    {
        return SortKeys(d_temp_storage,
                        temp_storage_bytes,
                        d_keys_in,
                        d_keys_out,
                        num_items,
                        num_segments,
                        d_begin_offsets,
                        d_end_offsets,
                        stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortKeys(void*           d_temp_storage,
                       size_t&         temp_storage_bytes,
                       const KeyT*     d_keys_in,
                       KeyT*           d_keys_out,
                       int             num_items,
                       int             num_segments,
                       OffsetIteratorT d_begin_offsets,
                       OffsetIteratorT d_end_offsets,
                       hipStream_t     stream,
                       bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return StableSortKeys(d_temp_storage,
                              temp_storage_bytes,
                              d_keys_in,
                              d_keys_out,
                              num_items,
                              num_segments,
                              d_begin_offsets,
                              d_end_offsets,
                              stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t StableSortKeys(void*               d_temp_storage,
                                                             size_t&             temp_storage_bytes,
                                                             DoubleBuffer<KeyT>& d_keys,
                                                             int                 num_items,
                                                             int                 num_segments,
                                                             OffsetIteratorT     d_begin_offsets,
                                                             OffsetIteratorT     d_end_offsets,
                                                             hipStream_t         stream = 0)
    {
        return SortKeys(d_temp_storage,
                        temp_storage_bytes,
                        d_keys,
                        num_items,
                        num_segments,
                        d_begin_offsets,
                        d_end_offsets,
                        stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortKeys(void*               d_temp_storage,
                       size_t&             temp_storage_bytes,
                       DoubleBuffer<KeyT>& d_keys,
                       int                 num_items,
                       int                 num_segments,
                       OffsetIteratorT     d_begin_offsets,
                       OffsetIteratorT     d_end_offsets,
                       hipStream_t         stream,
                       bool                debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return StableSortKeys(d_temp_storage,
                              temp_storage_bytes,
                              d_keys,
                              num_items,
                              num_segments,
                              d_begin_offsets,
                              d_end_offsets,
                              stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortKeysDescending(void*           d_temp_storage,
                                 size_t&         temp_storage_bytes,
                                 const KeyT*     d_keys_in,
                                 KeyT*           d_keys_out,
                                 int             num_items,
                                 int             num_segments,
                                 OffsetIteratorT d_begin_offsets,
                                 OffsetIteratorT d_end_offsets,
                                 hipStream_t     stream = 0)
    {
        return SortKeysDescending(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys_in,
                                  d_keys_out,
                                  num_items,
                                  num_segments,
                                  d_begin_offsets,
                                  d_end_offsets,
                                  stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortKeysDescending(void*           d_temp_storage,
                                 size_t&         temp_storage_bytes,
                                 const KeyT*     d_keys_in,
                                 KeyT*           d_keys_out,
                                 int             num_items,
                                 int             num_segments,
                                 OffsetIteratorT d_begin_offsets,
                                 OffsetIteratorT d_end_offsets,
                                 hipStream_t     stream,
                                 bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return StableSortKeysDescending(d_temp_storage,
                                        temp_storage_bytes,
                                        d_keys_in,
                                        d_keys_out,
                                        num_items,
                                        num_segments,
                                        d_begin_offsets,
                                        d_end_offsets,
                                        stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortKeysDescending(void*               d_temp_storage,
                                 size_t&             temp_storage_bytes,
                                 DoubleBuffer<KeyT>& d_keys,
                                 int                 num_items,
                                 int                 num_segments,
                                 OffsetIteratorT     d_begin_offsets,
                                 OffsetIteratorT     d_end_offsets,
                                 hipStream_t         stream = 0)
    {
        return SortKeysDescending(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys,
                                  num_items,
                                  num_segments,
                                  d_begin_offsets,
                                  d_end_offsets,
                                  stream);
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        StableSortKeysDescending(void*               d_temp_storage,
                                 size_t&             temp_storage_bytes,
                                 DoubleBuffer<KeyT>& d_keys,
                                 int                 num_items,
                                 int                 num_segments,
                                 OffsetIteratorT     d_begin_offsets,
                                 OffsetIteratorT     d_end_offsets,
                                 hipStream_t         stream,
                                 bool                debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return StableSortKeysDescending(d_temp_storage,
                                        temp_storage_bytes,
                                        d_keys,
                                        num_items,
                                        num_segments,
                                        d_begin_offsets,
                                        d_end_offsets,
                                        stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_SORT_HPP_
