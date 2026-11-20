// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include <rocprim/device/device_segmented_scan.hpp> // IWYU pragma: export

#include _HIPCUB_STD_INCLUDE(__functional/operations.h)

BEGIN_HIPCUB_NAMESPACE

struct DeviceSegmentedScan
{
    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSegmentedSum(void*                     d_temp_storage,
                                            size_t&                   temp_storage_bytes,
                                            InputIteratorT            d_in,
                                            OutputIteratorT           d_out,
                                            BeginOffsetIteratorInputT d_in_begin_offsets,
                                            EndOffsetIteratorInputT   d_in_end_offsets,
                                            _HIPCUB_STD::int64_t      num_segments,
                                            hipStream_t               stream = 0)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;
        return ::rocprim::segmented_exclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   input_type{},
                                                   _HIPCUB_STD::plus<>{},
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT,
             typename BeginOffsetIteratorOutputT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSegmentedSum(void*                      d_temp_storage,
                                            size_t&                    temp_storage_bytes,
                                            InputIteratorT             d_in,
                                            OutputIteratorT            d_out,
                                            BeginOffsetIteratorInputT  d_in_begin_offsets,
                                            EndOffsetIteratorInputT    d_in_end_offsets,
                                            BeginOffsetIteratorOutputT d_out_begin_offsets,
                                            _HIPCUB_STD::int64_t       num_segments,
                                            hipStream_t                stream = 0)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;
        return ::rocprim::segmented_exclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   d_out_begin_offsets,
                                                   input_type{},
                                                   _HIPCUB_STD::plus<>{},
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT,
             typename ScanOpT,
             typename InitValueT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSegmentedScan(void*                     d_temp_storage,
                                             size_t&                   temp_storage_bytes,
                                             InputIteratorT            d_in,
                                             OutputIteratorT           d_out,
                                             BeginOffsetIteratorInputT d_in_begin_offsets,
                                             EndOffsetIteratorInputT   d_in_end_offsets,
                                             _HIPCUB_STD::int64_t      num_segments,
                                             ScanOpT                   scan_op,
                                             InitValueT                init_value,
                                             hipStream_t               stream = 0)
    {
        return ::rocprim::segmented_exclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   init_value,
                                                   scan_op,
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT,
             typename BeginOffsetIteratorOutputT,
             typename ScanOpT,
             typename InitValueT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSegmentedScan(void*                      d_temp_storage,
                                             size_t&                    temp_storage_bytes,
                                             InputIteratorT             d_in,
                                             OutputIteratorT            d_out,
                                             BeginOffsetIteratorInputT  d_in_begin_offsets,
                                             EndOffsetIteratorInputT    d_in_end_offsets,
                                             BeginOffsetIteratorOutputT d_out_begin_offsets,
                                             _HIPCUB_STD::int64_t       num_segments,
                                             ScanOpT                    scan_op,
                                             InitValueT                 init_value,
                                             hipStream_t                stream = 0)
    {
        return ::rocprim::segmented_exclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   d_out_begin_offsets,
                                                   init_value,
                                                   scan_op,
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSegmentedSum(void*                     d_temp_storage,
                                            size_t&                   temp_storage_bytes,
                                            InputIteratorT            d_in,
                                            OutputIteratorT           d_out,
                                            BeginOffsetIteratorInputT d_in_begin_offsets,
                                            EndOffsetIteratorInputT   d_in_end_offsets,
                                            _HIPCUB_STD::int64_t      num_segments,
                                            hipStream_t               stream = 0)
    {
        return ::rocprim::segmented_inclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   _HIPCUB_STD::plus<>{},
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT,
             typename BeginOffsetIteratorOutputT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSegmentedSum(void*                      d_temp_storage,
                                            size_t&                    temp_storage_bytes,
                                            InputIteratorT             d_in,
                                            OutputIteratorT            d_out,
                                            BeginOffsetIteratorInputT  d_in_begin_offsets,
                                            EndOffsetIteratorInputT    d_in_end_offsets,
                                            BeginOffsetIteratorOutputT d_out_begin_offsets,
                                            _HIPCUB_STD::int64_t       num_segments,
                                            hipStream_t                stream = 0)
    {
        return ::rocprim::segmented_inclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   d_out_begin_offsets,
                                                   _HIPCUB_STD::plus<>{},
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT,
             typename ScanOpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSegmentedScan(void*                     d_temp_storage,
                                             size_t&                   temp_storage_bytes,
                                             InputIteratorT            d_in,
                                             OutputIteratorT           d_out,
                                             BeginOffsetIteratorInputT d_in_begin_offsets,
                                             EndOffsetIteratorInputT   d_in_end_offsets,
                                             _HIPCUB_STD::int64_t      num_segments,
                                             ScanOpT                   scan_op,
                                             hipStream_t               stream = 0)
    {
        return ::rocprim::segmented_inclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   scan_op,
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT,
             typename BeginOffsetIteratorOutputT,
             typename ScanOpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSegmentedScan(void*                      d_temp_storage,
                                             size_t&                    temp_storage_bytes,
                                             InputIteratorT             d_in,
                                             OutputIteratorT            d_out,
                                             BeginOffsetIteratorInputT  d_in_begin_offsets,
                                             EndOffsetIteratorInputT    d_in_end_offsets,
                                             BeginOffsetIteratorOutputT d_out_begin_offsets,
                                             _HIPCUB_STD::int64_t       num_segments,
                                             ScanOpT                    scan_op,
                                             hipStream_t                stream = 0)
    {
        return ::rocprim::segmented_inclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   d_out_begin_offsets,
                                                   scan_op,
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT,
             typename ScanOpT,
             typename InitValueT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSegmentedScanInit(void*                     d_temp_storage,
                                                 size_t&                   temp_storage_bytes,
                                                 InputIteratorT            d_in,
                                                 OutputIteratorT           d_out,
                                                 BeginOffsetIteratorInputT d_in_begin_offsets,
                                                 EndOffsetIteratorInputT   d_in_end_offsets,
                                                 _HIPCUB_STD::int64_t      num_segments,
                                                 ScanOpT                   scan_op,
                                                 InitValueT                init_value,
                                                 hipStream_t               stream = 0)
    {
        return ::rocprim::segmented_inclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   init_value,
                                                   scan_op,
                                                   stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename BeginOffsetIteratorInputT,
             typename EndOffsetIteratorInputT,
             typename BeginOffsetIteratorOutputT,
             typename ScanOpT,
             typename InitValueT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSegmentedScanInit(void*                      d_temp_storage,
                                                 size_t&                    temp_storage_bytes,
                                                 InputIteratorT             d_in,
                                                 OutputIteratorT            d_out,
                                                 BeginOffsetIteratorInputT  d_in_begin_offsets,
                                                 EndOffsetIteratorInputT    d_in_end_offsets,
                                                 BeginOffsetIteratorOutputT d_out_begin_offsets,
                                                 _HIPCUB_STD::int64_t       num_segments,
                                                 ScanOpT                    scan_op,
                                                 InitValueT                 init_value,
                                                 hipStream_t                stream = 0)
    {
        return ::rocprim::segmented_inclusive_scan(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_in,
                                                   d_out,
                                                   num_segments,
                                                   d_in_begin_offsets,
                                                   d_in_end_offsets,
                                                   d_out_begin_offsets,
                                                   init_value,
                                                   scan_op,
                                                   stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
