// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.

#ifndef HIPCUB_CUB_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include <cub/device/device_segmented_scan.cuh> // IWYU pragma: export

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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::ExclusiveSegmentedSum(d_temp_storage,
                                                              temp_storage_bytes,
                                                              d_in,
                                                              d_out,
                                                              d_in_begin_offsets,
                                                              d_in_end_offsets,
                                                              num_segments,
                                                              stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::ExclusiveSegmentedSum(d_temp_storage,
                                                              temp_storage_bytes,
                                                              d_in,
                                                              d_out,
                                                              d_in_begin_offsets,
                                                              d_in_end_offsets,
                                                              d_out_begin_offsets,
                                                              num_segments,
                                                              stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::ExclusiveSegmentedScan(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_in,
                                                               d_out,
                                                               d_in_begin_offsets,
                                                               d_in_end_offsets,
                                                               num_segments,
                                                               scan_op,
                                                               init_value,
                                                               stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::ExclusiveSegmentedScan(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_in,
                                                               d_out,
                                                               d_in_begin_offsets,
                                                               d_in_end_offsets,
                                                               d_out_begin_offsets,
                                                               num_segments,
                                                               scan_op,
                                                               init_value,
                                                               stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::InclusiveSegmentedSum(d_temp_storage,
                                                              temp_storage_bytes,
                                                              d_in,
                                                              d_out,
                                                              d_in_begin_offsets,
                                                              d_in_end_offsets,
                                                              num_segments,
                                                              stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::InclusiveSegmentedSum(d_temp_storage,
                                                              temp_storage_bytes,
                                                              d_in,
                                                              d_out,
                                                              d_in_begin_offsets,
                                                              d_in_end_offsets,
                                                              d_out_begin_offsets,
                                                              num_segments,
                                                              stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::InclusiveSegmentedScan(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_in,
                                                               d_out,
                                                               d_in_begin_offsets,
                                                               d_in_end_offsets,
                                                               num_segments,
                                                               scan_op,
                                                               stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::InclusiveSegmentedScan(d_temp_storage,
                                                               temp_storage_bytes,
                                                               d_in,
                                                               d_out,
                                                               d_in_begin_offsets,
                                                               d_in_end_offsets,
                                                               d_out_begin_offsets,
                                                               num_segments,
                                                               scan_op,
                                                               stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(d_temp_storage,
                                                                   temp_storage_bytes,
                                                                   d_in,
                                                                   d_out,
                                                                   d_in_begin_offsets,
                                                                   d_in_end_offsets,
                                                                   num_segments,
                                                                   scan_op,
                                                                   init_value,
                                                                   stream));
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
        return hipCUDAErrorTohipError(
            ::cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(d_temp_storage,
                                                                   temp_storage_bytes,
                                                                   d_in,
                                                                   d_out,
                                                                   d_in_begin_offsets,
                                                                   d_in_end_offsets,
                                                                   d_out_begin_offsets,
                                                                   num_segments,
                                                                   scan_op,
                                                                   init_value,
                                                                   stream));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
