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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SCAN_HPP_

#include <iostream>
#include "../../../config.hpp"

#include "../thread/thread_operators.hpp"

#include <rocprim/device/device_scan.hpp>
#include <rocprim/device/device_scan_by_key.hpp>

BEGIN_HIPCUB_NAMESPACE

class DeviceScan
{
public:
    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t InclusiveSum(void *d_temp_storage,
                            size_t &temp_storage_bytes,
                            InputIteratorT d_in,
                            OutputIteratorT d_out,
                            size_t num_items,
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
    {
        return InclusiveScan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, ::hipcub::Sum(), num_items,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ScanOpT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t InclusiveScan(void *d_temp_storage,
                             size_t &temp_storage_bytes,
                             InputIteratorT d_in,
                             OutputIteratorT d_out,
                             ScanOpT scan_op,
                             size_t num_items,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return ::rocprim::inclusive_scan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items,
            scan_op,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ExclusiveSum(void *d_temp_storage,
                            size_t &temp_storage_bytes,
                            InputIteratorT d_in,
                            OutputIteratorT d_out,
                            size_t num_items,
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return ExclusiveScan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, ::hipcub::Sum(), T(0), num_items,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ScanOpT,
        typename InitValueT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ExclusiveScan(void *d_temp_storage,
                             size_t &temp_storage_bytes,
                             InputIteratorT d_in,
                             OutputIteratorT d_out,
                             ScanOpT scan_op,
                             InitValueT init_value,
                             size_t num_items,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return ::rocprim::exclusive_scan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, init_value, num_items,
            scan_op,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ScanOpT,
        typename InitValueT,
        typename InitValueIterT = InitValueT*
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ExclusiveScan(void *d_temp_storage,
                             size_t &temp_storage_bytes,
                             InputIteratorT d_in,
                             OutputIteratorT d_out,
                             ScanOpT scan_op,
                             FutureValue<InitValueT, InitValueIterT> init_value,
                             int num_items,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return ::rocprim::exclusive_scan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, init_value, num_items,
            scan_op,
            stream, debug_synchronous
        );
    }

    template <
        typename KeysInputIteratorT,
        typename ValuesInputIteratorT,
        typename ValuesOutputIteratorT,
        typename EqualityOpT = ::hipcub::Equality
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ExclusiveSumByKey(void *d_temp_storage,
                                 size_t &temp_storage_bytes,
                                 KeysInputIteratorT d_keys_in,
                                 ValuesInputIteratorT d_values_in,
                                 ValuesOutputIteratorT d_values_out,
                                 int num_items,
                                 EqualityOpT equality_op = EqualityOpT(),
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false)
    {
        using in_value_type = typename std::iterator_traits<ValuesInputIteratorT>::value_type;

        return ::rocprim::exclusive_scan_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, d_values_out,
            static_cast<in_value_type>(0), static_cast<size_t>(num_items),
            ::hipcub::Sum(), equality_op, stream, debug_synchronous
        );
    }

    template <
        typename KeysInputIteratorT,
        typename ValuesInputIteratorT,
        typename ValuesOutputIteratorT,
        typename ScanOpT,
        typename InitValueT,
        typename EqualityOpT = ::hipcub::Equality
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ExclusiveScanByKey(void *d_temp_storage,
                                  size_t &temp_storage_bytes,
                                  KeysInputIteratorT d_keys_in,
                                  ValuesInputIteratorT d_values_in,
                                  ValuesOutputIteratorT d_values_out,
                                  ScanOpT scan_op,
                                  InitValueT init_value,
                                  int num_items,
                                  EqualityOpT equality_op = EqualityOpT(),
                                  hipStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        return ::rocprim::exclusive_scan_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, d_values_out,
            init_value, static_cast<size_t>(num_items),
            scan_op, equality_op, stream, debug_synchronous
        );
    }

    template <
        typename KeysInputIteratorT,
        typename ValuesInputIteratorT,
        typename ValuesOutputIteratorT,
        typename EqualityOpT = ::hipcub::Equality
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t InclusiveSumByKey(void *d_temp_storage,
                                 size_t &temp_storage_bytes,
                                 KeysInputIteratorT d_keys_in,
                                 ValuesInputIteratorT d_values_in,
                                 ValuesOutputIteratorT d_values_out,
                                 int num_items,
                                 EqualityOpT equality_op = EqualityOpT(),
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false)
    {
        return ::rocprim::inclusive_scan_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, d_values_out,
            static_cast<size_t>(num_items), ::hipcub::Sum(),
            equality_op, stream, debug_synchronous
        );
    }

    template <
        typename KeysInputIteratorT,
        typename ValuesInputIteratorT,
        typename ValuesOutputIteratorT,
        typename ScanOpT,
        typename EqualityOpT = ::hipcub::Equality
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t InclusiveScanByKey(void *d_temp_storage,
                                  size_t &temp_storage_bytes,
                                  KeysInputIteratorT d_keys_in,
                                  ValuesInputIteratorT d_values_in,
                                  ValuesOutputIteratorT d_values_out,
                                  ScanOpT scan_op,
                                  int num_items,
                                  EqualityOpT equality_op = EqualityOpT(),
                                  hipStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        return ::rocprim::inclusive_scan_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, d_values_out,
            static_cast<size_t>(num_items), scan_op,
            equality_op, stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
