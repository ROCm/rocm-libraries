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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SCAN_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"
#include "../thread/thread_operators.hpp"

#include <rocprim/device/config_types.hpp> // IWYU pragma: export
#include <rocprim/device/device_scan.hpp> // IWYU pragma: export
#include <rocprim/device/device_scan_by_key.hpp> // IWYU pragma: export
#include <rocprim/type_traits.hpp> // IWYU pragma: export
#include <rocprim/types/future_value.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

class DeviceScan
{
public:
    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSum(void*           d_temp_storage,
                                   size_t&         temp_storage_bytes,
                                   InputIteratorT  d_in,
                                   OutputIteratorT d_out,
                                   NumItemsT       num_items,
                                   hipStream_t     stream = 0)
    {
        return InclusiveScan(d_temp_storage,
                             temp_storage_bytes,
                             d_in,
                             d_out,
                             ::hipcub::Sum(),
                             num_items,
                             stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSum(void*           d_temp_storage,
                                   size_t&         temp_storage_bytes,
                                   InputIteratorT  d_in,
                                   OutputIteratorT d_out,
                                   NumItemsT       num_items,
                                   hipStream_t     stream,
                                   bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
    }

    template<typename IteratorT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSum(void*       d_temp_storage,
                                   size_t&     temp_storage_bytes,
                                   IteratorT   d_data,
                                   NumItemsT   num_items,
                                   hipStream_t stream = 0)
    {
        return InclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items, stream);
    }

    template<typename IteratorT, typename NumItemsT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSum(void*       d_temp_storage,
                                   size_t&     temp_storage_bytes,
                                   IteratorT   d_data,
                                   NumItemsT   num_items,
                                   hipStream_t stream,
                                   bool        debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return InclusiveSum<IteratorT>(d_temp_storage,
                                       temp_storage_bytes,
                                       d_data,
                                       num_items,
                                       stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveScan(void*           d_temp_storage,
                                    size_t&         temp_storage_bytes,
                                    InputIteratorT  d_in,
                                    OutputIteratorT d_out,
                                    ScanOpT         scan_op,
                                    NumItemsT       num_items,
                                    hipStream_t     stream = 0)
    {
        return ::rocprim::
            inclusive_scan<::rocprim::default_config, InputIteratorT, OutputIteratorT, ScanOpT>(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                num_items,
                scan_op,
                stream,
                HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename NumItemsT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveScan(void*           d_temp_storage,
                                    size_t&         temp_storage_bytes,
                                    InputIteratorT  d_in,
                                    OutputIteratorT d_out,
                                    ScanOpT         scan_op,
                                    NumItemsT       num_items,
                                    hipStream_t     stream,
                                    bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return InclusiveScan(d_temp_storage,
                             temp_storage_bytes,
                             d_in,
                             d_out,
                             scan_op,
                             num_items,
                             stream);
    }

    template<typename IteratorT, typename ScanOpT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveScan(void*       d_temp_storage,
                                    size_t&     temp_storage_bytes,
                                    IteratorT   d_data,
                                    ScanOpT     scan_op,
                                    NumItemsT   num_items,
                                    hipStream_t stream = 0)
    {
        return InclusiveScan(d_temp_storage,
                             temp_storage_bytes,
                             d_data,
                             d_data,
                             scan_op,
                             num_items,
                             stream);
    }

    template<typename IteratorT, typename ScanOpT, typename NumItemsT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveScan(void*       d_temp_storage,
                                    size_t&     temp_storage_bytes,
                                    IteratorT   d_data,
                                    ScanOpT     scan_op,
                                    NumItemsT   num_items,
                                    hipStream_t stream,
                                    bool        debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return InclusiveScan<IteratorT, ScanOpT>(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_data,
                                                 scan_op,
                                                 num_items,
                                                 stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveScanInit(void*           d_temp_storage,
                                        size_t&         temp_storage_bytes,
                                        InputIteratorT  d_in,
                                        OutputIteratorT d_out,
                                        ScanOpT         scan_op,
                                        InitValueT      init_value,
                                        NumItemsT       num_items,
                                        hipStream_t     stream = 0)
    {
        return ::rocprim::inclusive_scan<::rocprim::default_config,
                                         InputIteratorT,
                                         OutputIteratorT,
                                         InitValueT,
                                         ScanOpT>(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_in,
                                                  d_out,
                                                  init_value,
                                                  num_items,
                                                  scan_op,
                                                  stream,
                                                  HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSum(void*           d_temp_storage,
                                   size_t&         temp_storage_bytes,
                                   InputIteratorT  d_in,
                                   OutputIteratorT d_out,
                                   NumItemsT       num_items,
                                   hipStream_t     stream = 0)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return ExclusiveScan(d_temp_storage,
                             temp_storage_bytes,
                             d_in,
                             d_out,
                             ::hipcub::Sum(),
                             T(0),
                             num_items,
                             stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSum(void*           d_temp_storage,
                                   size_t&         temp_storage_bytes,
                                   InputIteratorT  d_in,
                                   OutputIteratorT d_out,
                                   NumItemsT       num_items,
                                   hipStream_t     stream,
                                   bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
    }

    template<typename IteratorT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSum(void*       d_temp_storage,
                                   size_t&     temp_storage_bytes,
                                   IteratorT   d_data,
                                   NumItemsT   num_items,
                                   hipStream_t stream = 0)
    {
        return ExclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items, stream);
    }

    template<typename IteratorT, typename NumItemsT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSum(void*       d_temp_storage,
                                   size_t&     temp_storage_bytes,
                                   IteratorT   d_data,
                                   NumItemsT   num_items,
                                   hipStream_t stream,
                                   bool        debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ExclusiveSum<IteratorT>(d_temp_storage,
                                       temp_storage_bytes,
                                       d_data,
                                       num_items,
                                       stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(void*           d_temp_storage,
                                    size_t&         temp_storage_bytes,
                                    InputIteratorT  d_in,
                                    OutputIteratorT d_out,
                                    ScanOpT         scan_op,
                                    InitValueT      init_value,
                                    NumItemsT       num_items,
                                    hipStream_t     stream = 0)
    {
        return ::rocprim::exclusive_scan<::rocprim::default_config,
                                         InputIteratorT,
                                         OutputIteratorT,
                                         InitValueT,
                                         ScanOpT>(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_in,
                                                  d_out,
                                                  init_value,
                                                  num_items,
                                                  scan_op,
                                                  stream,
                                                  HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename NumItemsT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(void*           d_temp_storage,
                                    size_t&         temp_storage_bytes,
                                    InputIteratorT  d_in,
                                    OutputIteratorT d_out,
                                    ScanOpT         scan_op,
                                    InitValueT      init_value,
                                    NumItemsT       num_items,
                                    hipStream_t     stream,
                                    bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ExclusiveScan(d_temp_storage,
                             temp_storage_bytes,
                             d_in,
                             d_out,
                             scan_op,
                             init_value,
                             num_items,
                             stream);
    }

    template<typename IteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(void*       d_temp_storage,
                                    size_t&     temp_storage_bytes,
                                    IteratorT   d_data,
                                    ScanOpT     scan_op,
                                    InitValueT  init_value,
                                    NumItemsT   num_items,
                                    hipStream_t stream = 0)
    {
        return ExclusiveScan(d_temp_storage,
                             temp_storage_bytes,
                             d_data,
                             d_data,
                             scan_op,
                             init_value,
                             num_items,
                             stream);
    }

    template<typename IteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(void*       d_temp_storage,
                                    size_t&     temp_storage_bytes,
                                    IteratorT   d_data,
                                    ScanOpT     scan_op,
                                    InitValueT  init_value,
                                    NumItemsT   num_items,
                                    hipStream_t stream,
                                    bool        debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ExclusiveScan<IteratorT, ScanOpT, InitValueT>(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_data,
                                                             scan_op,
                                                             init_value,
                                                             num_items,
                                                             stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename InitValueIterT = InitValueT*,
             typename NumItemsT      = int>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(void*                                   d_temp_storage,
                                    size_t&                                 temp_storage_bytes,
                                    InputIteratorT                          d_in,
                                    OutputIteratorT                         d_out,
                                    ScanOpT                                 scan_op,
                                    FutureValue<InitValueT, InitValueIterT> init_value,
                                    NumItemsT                               num_items,
                                    hipStream_t                             stream = 0)
    {
        return ::rocprim::exclusive_scan<::rocprim::default_config,
                                         InputIteratorT,
                                         OutputIteratorT,
                                         FutureValue<InitValueT, InitValueIterT>,
                                         ScanOpT>(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_in,
                                                  d_out,
                                                  init_value,
                                                  static_cast<size_t>(num_items),
                                                  scan_op,
                                                  stream,
                                                  HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename InitValueIterT = InitValueT*,
             typename NumItemsT      = int>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(void*                                   d_temp_storage,
                                    size_t&                                 temp_storage_bytes,
                                    InputIteratorT                          d_in,
                                    OutputIteratorT                         d_out,
                                    ScanOpT                                 scan_op,
                                    FutureValue<InitValueT, InitValueIterT> init_value,
                                    NumItemsT                               num_items,
                                    hipStream_t                             stream,
                                    bool                                    debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ExclusiveScan(d_temp_storage,
                             temp_storage_bytes,
                             d_in,
                             d_out,
                             scan_op,
                             init_value,
                             num_items,
                             stream);
    }

    template<typename IteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename InitValueIterT = InitValueT*,
             typename NumItemsT      = int>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(void*                                   d_temp_storage,
                                    size_t&                                 temp_storage_bytes,
                                    IteratorT                               d_data,
                                    ScanOpT                                 scan_op,
                                    FutureValue<InitValueT, InitValueIterT> init_value,
                                    NumItemsT                               num_items,
                                    hipStream_t                             stream = 0)
    {
        return ExclusiveScan(d_temp_storage,
                             temp_storage_bytes,
                             d_data,
                             d_data,
                             scan_op,
                             init_value,
                             num_items,
                             stream);
    }

    template<typename IteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename InitValueIterT = InitValueT*,
             typename NumItemsT      = int>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(void*                                   d_temp_storage,
                                    size_t&                                 temp_storage_bytes,
                                    IteratorT                               d_data,
                                    ScanOpT                                 scan_op,
                                    FutureValue<InitValueT, InitValueIterT> init_value,
                                    NumItemsT                               num_items,
                                    hipStream_t                             stream,
                                    bool                                    debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ExclusiveScan<IteratorT, ScanOpT, InitValueT, InitValueIterT>(d_temp_storage,
                                                                             temp_storage_bytes,
                                                                             d_data,
                                                                             scan_op,
                                                                             init_value,
                                                                             num_items,
                                                                             stream);
    }

    template<typename KeysInputIteratorT,
             typename ValuesInputIteratorT,
             typename ValuesOutputIteratorT,
             typename EqualityOpT = ::hipcub::Equality,
             typename NumItemsT   = std::uint32_t>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSumByKey(void*                 d_temp_storage,
                                        size_t&               temp_storage_bytes,
                                        KeysInputIteratorT    d_keys_in,
                                        ValuesInputIteratorT  d_values_in,
                                        ValuesOutputIteratorT d_values_out,
                                        NumItemsT             num_items,
                                        EqualityOpT           equality_op = EqualityOpT(),
                                        hipStream_t           stream      = 0)
    {
        using in_value_type = typename std::iterator_traits<ValuesInputIteratorT>::value_type;

        return ExclusiveScanByKey(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys_in,
                                  d_values_in,
                                  d_values_out,
                                  ::hipcub::Sum(),
                                  static_cast<in_value_type>(0),
                                  num_items,
                                  equality_op,
                                  stream);
    }

    template<typename KeysInputIteratorT,
             typename ValuesInputIteratorT,
             typename ValuesOutputIteratorT,
             typename EqualityOpT = ::hipcub::Equality,
             typename NumItemsT   = std::uint32_t>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSumByKey(void*                 d_temp_storage,
                                        size_t&               temp_storage_bytes,
                                        KeysInputIteratorT    d_keys_in,
                                        ValuesInputIteratorT  d_values_in,
                                        ValuesOutputIteratorT d_values_out,
                                        NumItemsT             num_items,
                                        EqualityOpT           equality_op,
                                        hipStream_t           stream,
                                        bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ExclusiveSumByKey(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_values_in,
                                 d_values_out,
                                 num_items,
                                 equality_op,
                                 stream);
    }

    template<typename KeysInputIteratorT,
             typename ValuesInputIteratorT,
             typename ValuesOutputIteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename EqualityOpT = ::hipcub::Equality,
             typename NumItemsT   = std::uint32_t>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScanByKey(void*                 d_temp_storage,
                                         size_t&               temp_storage_bytes,
                                         KeysInputIteratorT    d_keys_in,
                                         ValuesInputIteratorT  d_values_in,
                                         ValuesOutputIteratorT d_values_out,
                                         ScanOpT               scan_op,
                                         InitValueT            init_value,
                                         NumItemsT             num_items,
                                         EqualityOpT           equality_op = EqualityOpT(),
                                         hipStream_t           stream      = 0)
    {
        using acc_t = rocprim::accumulator_t<ScanOpT, rocprim::detail::input_type_t<InitValueT>>;

        return ::rocprim::exclusive_scan_by_key<::rocprim::default_config,
                                                KeysInputIteratorT,
                                                ValuesInputIteratorT,
                                                ValuesOutputIteratorT,
                                                InitValueT,
                                                ScanOpT,
                                                EqualityOpT,
                                                acc_t>(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys_in,
                                                       d_values_in,
                                                       d_values_out,
                                                       init_value,
                                                       static_cast<size_t>(num_items),
                                                       scan_op,
                                                       equality_op,
                                                       stream,
                                                       HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename KeysInputIteratorT,
             typename ValuesInputIteratorT,
             typename ValuesOutputIteratorT,
             typename ScanOpT,
             typename InitValueT,
             typename EqualityOpT = ::hipcub::Equality,
             typename NumItemsT   = std::uint32_t>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScanByKey(void*                 d_temp_storage,
                                         size_t&               temp_storage_bytes,
                                         KeysInputIteratorT    d_keys_in,
                                         ValuesInputIteratorT  d_values_in,
                                         ValuesOutputIteratorT d_values_out,
                                         ScanOpT               scan_op,
                                         InitValueT            init_value,
                                         NumItemsT             num_items,
                                         EqualityOpT           equality_op,
                                         hipStream_t           stream,
                                         bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ExclusiveScanByKey(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys_in,
                                  d_values_in,
                                  d_values_out,
                                  scan_op,
                                  init_value,
                                  num_items,
                                  equality_op,
                                  stream);
    }

    template<typename KeysInputIteratorT,
             typename ValuesInputIteratorT,
             typename ValuesOutputIteratorT,
             typename EqualityOpT = ::hipcub::Equality,
             typename NumItemsT   = std::uint32_t>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSumByKey(void*                 d_temp_storage,
                                        size_t&               temp_storage_bytes,
                                        KeysInputIteratorT    d_keys_in,
                                        ValuesInputIteratorT  d_values_in,
                                        ValuesOutputIteratorT d_values_out,
                                        NumItemsT             num_items,
                                        EqualityOpT           equality_op = EqualityOpT(),
                                        hipStream_t           stream      = 0)
    {
        return InclusiveScanByKey(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys_in,
                                  d_values_in,
                                  d_values_out,
                                  ::hipcub::Sum(),
                                  num_items,
                                  equality_op,
                                  stream);
    }

    template<typename KeysInputIteratorT,
             typename ValuesInputIteratorT,
             typename ValuesOutputIteratorT,
             typename EqualityOpT = ::hipcub::Equality,
             typename NumItemsT   = std::uint32_t>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSumByKey(void*                 d_temp_storage,
                                        size_t&               temp_storage_bytes,
                                        KeysInputIteratorT    d_keys_in,
                                        ValuesInputIteratorT  d_values_in,
                                        ValuesOutputIteratorT d_values_out,
                                        NumItemsT             num_items,
                                        EqualityOpT           equality_op,
                                        hipStream_t           stream,
                                        bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return InclusiveSumByKey(d_temp_storage,
                                 temp_storage_bytes,
                                 d_keys_in,
                                 d_values_in,
                                 d_values_out,
                                 num_items,
                                 equality_op,
                                 stream);
    }

    template<typename KeysInputIteratorT,
             typename ValuesInputIteratorT,
             typename ValuesOutputIteratorT,
             typename ScanOpT,
             typename EqualityOpT = ::hipcub::Equality,
             typename NumItemsT   = std::uint32_t>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveScanByKey(void*                 d_temp_storage,
                                         size_t&               temp_storage_bytes,
                                         KeysInputIteratorT    d_keys_in,
                                         ValuesInputIteratorT  d_values_in,
                                         ValuesOutputIteratorT d_values_out,
                                         ScanOpT               scan_op,
                                         NumItemsT             num_items,
                                         EqualityOpT           equality_op = EqualityOpT(),
                                         hipStream_t           stream      = 0)
    {
        using acc_t = ::rocprim::
            accumulator_t<ScanOpT, typename std::iterator_traits<ValuesInputIteratorT>::value_type>;

        return ::rocprim::inclusive_scan_by_key<::rocprim::default_config,
                                                KeysInputIteratorT,
                                                ValuesInputIteratorT,
                                                ValuesOutputIteratorT,
                                                ScanOpT,
                                                EqualityOpT,
                                                acc_t>(d_temp_storage,
                                                       temp_storage_bytes,
                                                       d_keys_in,
                                                       d_values_in,
                                                       d_values_out,
                                                       static_cast<size_t>(num_items),
                                                       scan_op,
                                                       equality_op,
                                                       stream,
                                                       HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename KeysInputIteratorT,
             typename ValuesInputIteratorT,
             typename ValuesOutputIteratorT,
             typename ScanOpT,
             typename EqualityOpT = ::hipcub::Equality,
             typename NumItemsT   = std::uint32_t>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveScanByKey(void*                 d_temp_storage,
                                         size_t&               temp_storage_bytes,
                                         KeysInputIteratorT    d_keys_in,
                                         ValuesInputIteratorT  d_values_in,
                                         ValuesOutputIteratorT d_values_out,
                                         ScanOpT               scan_op,
                                         NumItemsT             num_items,
                                         EqualityOpT           equality_op,
                                         hipStream_t           stream,
                                         bool                  debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return InclusiveScanByKey(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys_in,
                                  d_values_in,
                                  d_values_out,
                                  scan_op,
                                  num_items,
                                  equality_op,
                                  stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
