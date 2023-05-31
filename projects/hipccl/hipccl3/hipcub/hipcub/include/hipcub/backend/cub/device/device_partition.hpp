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

#ifndef HIPCUB_CUB_DEVICE_DEVICE_PARTITION_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_PARTITION_HPP_

#include "../../../config.hpp"

#include <cub/device/device_partition.cuh>

BEGIN_HIPCUB_NAMESPACE

struct DevicePartition
{
    template <
        typename                    InputIteratorT,
        typename                    FlagIterator,
        typename                    OutputIteratorT,
        typename                    NumSelectedIteratorT>
    HIPCUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t Flagged(
        void*               d_temp_storage,                ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                           ///< [in] Pointer to the input sequence of data items
        FlagIterator                d_flags,                        ///< [in] Pointer to the input sequence of selection flags
        OutputIteratorT             d_out,                          ///< [out] Pointer to the output sequence of partitioned data items
        NumSelectedIteratorT        d_num_selected_out,             ///< [out] Pointer to the output total number of items selected (i.e., the offset of the unselected partition)
        int                         num_items,                      ///< [in] Total number of items to select from
        hipStream_t                 stream             = 0,         ///< [in] <b>[optional]</b> hip stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous  = false)     ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DevicePartition::Flagged(d_temp_storage,
                                                                      temp_storage_bytes,
                                                                      d_in,
                                                                      d_flags,
                                                                      d_out,
                                                                      d_num_selected_out,
                                                                      num_items,
                                                                      stream));
    }

    template <
        typename                    InputIteratorT,
        typename                    OutputIteratorT,
        typename                    NumSelectedIteratorT,
        typename                    SelectOp>
    HIPCUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t If(
        void*               d_temp_storage,                ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                           ///< [in] Pointer to the input sequence of data items
        OutputIteratorT             d_out,                          ///< [out] Pointer to the output sequence of partitioned data items
        NumSelectedIteratorT        d_num_selected_out,             ///< [out] Pointer to the output total number of items selected (i.e., the offset of the unselected partition)
        int                         num_items,                      ///< [in] Total number of items to select from
        SelectOp                    select_op,                      ///< [in] Unary selection operator
        hipStream_t                 stream             = 0,         ///< [in] <b>[optional]</b> hip stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous  = false)     ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DevicePartition::If(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_in,
                                                                 d_out,
                                                                 d_num_selected_out,
                                                                 num_items,
                                                                 select_op,
                                                                 stream));
    }

    template <typename InputIteratorT,
              typename FirstOutputIteratorT,
              typename SecondOutputIteratorT,
              typename UnselectedOutputIteratorT,
              typename NumSelectedIteratorT,
              typename SelectFirstPartOp,
              typename SelectSecondPartOp>
    HIPCUB_RUNTIME_FUNCTION __forceinline__ static hipError_t
    If(void *d_temp_storage,
       std::size_t &temp_storage_bytes,
       InputIteratorT d_in,
       FirstOutputIteratorT d_first_part_out,
       SecondOutputIteratorT d_second_part_out,
       UnselectedOutputIteratorT d_unselected_out,
       NumSelectedIteratorT d_num_selected_out,
       int num_items,
       SelectFirstPartOp select_first_part_op,
       SelectSecondPartOp select_second_part_op,
       hipStream_t stream     = 0,
       bool debug_synchronous = false)
    {
        (void)debug_synchronous;
        return hipCUDAErrorTohipError(::cub::DevicePartition::If(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 d_in,
                                                                 d_first_part_out,
                                                                 d_second_part_out,
                                                                 d_unselected_out,
                                                                 d_num_selected_out,
                                                                 num_items,
                                                                 select_first_part_op,
                                                                 select_second_part_op,
                                                                 stream));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_PARTITION_HPP_
