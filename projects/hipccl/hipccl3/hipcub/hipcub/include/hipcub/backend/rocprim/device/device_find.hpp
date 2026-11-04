// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_FIND_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_FIND_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include <rocprim/device/device_binary_search.hpp> // IWYU pragma: export
#include <rocprim/device/device_find_if.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

struct DeviceFind
{
    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename ScanOpT,
             typename NumItemsT>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t FindIf(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             ScanOpT         scan_op,
                             NumItemsT       num_items,
                             hipStream_t     stream = 0)
    {
        return ::rocprim::find_if(d_temp_storage,
                                  temp_storage_bytes,
                                  d_in,
                                  d_out,
                                  num_items,
                                  scan_op,
                                  stream,
                                  HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename RangeIteratorT,
             typename RangeNumItemsT,
             typename ValuesIteratorT,
             typename ValuesNumItemsT,
             typename OutputIteratorT,
             typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t LowerBound(void*           d_temp_storage,
                                 size_t&         temp_storage_bytes,
                                 RangeIteratorT  d_range,
                                 RangeNumItemsT  range_num_items,
                                 ValuesIteratorT d_values,
                                 ValuesNumItemsT values_num_items,
                                 OutputIteratorT d_output,
                                 CompareOpT      comp,
                                 hipStream_t     stream = 0)
    {
        return ::rocprim::lower_bound(d_temp_storage,
                                      temp_storage_bytes,
                                      d_range,
                                      d_values,
                                      d_output,
                                      range_num_items,
                                      values_num_items,
                                      comp,
                                      stream,
                                      HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename RangeIteratorT,
             typename RangeNumItemsT,
             typename ValuesIteratorT,
             typename ValuesNumItemsT,
             typename OutputIteratorT,
             typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t UpperBound(void*           d_temp_storage,
                                 size_t&         temp_storage_bytes,
                                 RangeIteratorT  d_range,
                                 RangeNumItemsT  range_num_items,
                                 ValuesIteratorT d_values,
                                 ValuesNumItemsT values_num_items,
                                 OutputIteratorT d_output,
                                 CompareOpT      comp,
                                 hipStream_t     stream = 0)
    {
        return ::rocprim::upper_bound(d_temp_storage,
                                      temp_storage_bytes,
                                      d_range,
                                      d_values,
                                      d_output,
                                      range_num_items,
                                      values_num_items,
                                      comp,
                                      stream,
                                      HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_FIND_HPP_
