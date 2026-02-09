/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_CUB_DEVICE_DEVICE_FOR_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_FOR_HPP_

#include "../../../config.hpp"

#include <cub/device/device_for.cuh> // IWYU pragma: export

#if __cccl_lib_mdspan
    #include <cuda/std/__mdspan/extents.h>
BEGIN_HIPCUB_NAMESPACE
template<class IndexType, size_t... Extents>
using extents = ::cuda::std::extents<IndexType, Extents...>;
END_HIPCUB_NAMESPACE
#endif // __cccl_lib_mdspan

BEGIN_HIPCUB_NAMESPACE

struct DeviceFor
{
    template<class RandomAccessIteratorT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEach(RandomAccessIteratorT first,
                              RandomAccessIteratorT last,
                              OpT                   op,
                              hipStream_t           stream = 0)
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEach(first, last, op, stream));
    }

    template<class RandomAccessIteratorT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEach(void*                 d_temp_storage,
                              size_t&               temp_storage_bytes,
                              RandomAccessIteratorT first,
                              RandomAccessIteratorT last,
                              OpT                   op,
                              hipStream_t           stream = 0)
    {
        return hipCUDAErrorTohipError(
            cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, first, last, op, stream));
    }

    template<class RandomAccessIteratorT, class OffsetT, class OpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        ForEachN(RandomAccessIteratorT first, OffsetT num_items, OpT op, hipStream_t stream = 0)
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEachN(first, num_items, op, stream));
    }

    template<class RandomAccessIteratorT, class OffsetT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachN(void*                 d_temp_storage,
                               size_t&               temp_storage_bytes,
                               RandomAccessIteratorT first,
                               OffsetT               num_items,
                               OpT                   op,
                               hipStream_t           stream = 0)
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEachN(d_temp_storage,
                                                               temp_storage_bytes,
                                                               first,
                                                               num_items,
                                                               op,
                                                               stream));
    }

    template<class RandomAccessIteratorT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachCopy(RandomAccessIteratorT first,
                                  RandomAccessIteratorT last,
                                  OpT                   op,
                                  hipStream_t           stream = 0)
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEachCopy(first, last, op, stream));
    }

    template<class RandomAccessIteratorT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachCopy(void*                 d_temp_storage,
                                  size_t&               temp_storage_bytes,
                                  RandomAccessIteratorT first,
                                  RandomAccessIteratorT last,
                                  OpT                   op,
                                  hipStream_t           stream = 0)
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEachCopy(d_temp_storage,
                                                                  temp_storage_bytes,
                                                                  first,
                                                                  last,
                                                                  op,
                                                                  stream));
    }

    template<class RandomAccessIteratorT, class OffsetT, class OpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        ForEachCopyN(RandomAccessIteratorT first, OffsetT num_items, OpT op, hipStream_t stream = 0)
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEachCopyN(first, num_items, op, stream));
    }

    template<class RandomAccessIteratorT, class OffsetT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachCopyN(void*                 d_temp_storage,
                                   size_t&               temp_storage_bytes,
                                   RandomAccessIteratorT first,
                                   OffsetT               num_items,
                                   OpT                   op,
                                   hipStream_t           stream = 0)
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEachCopyN(d_temp_storage,
                                                                   temp_storage_bytes,
                                                                   first,
                                                                   num_items,
                                                                   op,
                                                                   stream));
    }

    template<class ShapeT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t Bulk(ShapeT shape, OpT op, hipStream_t stream = 0)
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::Bulk(shape, op, stream));
    }

    template<class ShapeT, class OpT>
HIPCUB_RUNTIME_FUNCTION
    static hipError_t Bulk(void*       d_temp_storage,
                           size_t&     temp_storage_bytes,
                           ShapeT      shape,
                           OpT         op,
                           hipStream_t stream = 0)
    {
        return hipCUDAErrorTohipError(
            cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, shape, op, stream));
    }

// ForEachInExtents only enables when the cccl mdspan extension is enabled
#ifdef __cccl_lib_mdspan
    template<class IndexType, size_t... Extents, typename OpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachInExtents(void*   d_temp_storage,
                                       size_t& temp_storage_bytes,
                                       const ::cuda::std::extents<IndexType, Extents...>& extents,
                                       OpT                                                op,
                                       hipStream_t stream = {})
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEachInExtents(d_temp_storage,
                                                                       temp_storage_bytes,
                                                                       extents,
                                                                       op,
                                                                       stream));
    }

    template<class IndexType, size_t... Extents, typename OpT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ForEachInExtents(const ::cuda::std::extents<IndexType, Extents...>& extents,
                                       OpT                                                op,
                                       hipStream_t stream = {})
    {
        return hipCUDAErrorTohipError(cub::DeviceFor::ForEachInExtents(extents, op, stream));
    }
#endif // __cccl_lib_mdspan
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_FOR_HPP_
