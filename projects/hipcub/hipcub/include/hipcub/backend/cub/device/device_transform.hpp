/******************************************************************************.
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIBCUB_ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_
#define HIBCUB_ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_

#include "../../../config.hpp"

#include <hip/hip_runtime.h>

#include <cub/device/device_transform.cuh> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

struct DeviceTransform
{
    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Transform(std::tuple<RandomAccessIteratorsIn...> inputs,
                                RandomAccessIteratorOut                output,
                                NumItemsT                              num_items,
                                TransformOp                            transform_op,
                                hipStream_t                            stream = nullptr)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceTransform::Transform(inputs, output, num_items, transform_op, stream));
    }

    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Transform(void*                                  d_temp_storage,
                                size_t&                                temp_storage_bytes,
                                std::tuple<RandomAccessIteratorsIn...> inputs,
                                RandomAccessIteratorOut                output,
                                NumItemsT                              num_items,
                                TransformOp                            transform_op,
                                hipStream_t                            stream = nullptr)
    {
        return hipCUDAErrorTohipError(::cub::DeviceTransform::Transform(d_temp_storage,
                                                                        temp_storage_bytes,
                                                                        inputs,
                                                                        output,
                                                                        num_items,
                                                                        transform_op,
                                                                        stream));
    }

    template<typename RandomAccessIteratorIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Transform(RandomAccessIteratorIn  input,
                                RandomAccessIteratorOut output,
                                NumItemsT               num_items,
                                TransformOp             transform_op,
                                hipStream_t             stream = nullptr)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceTransform::Transform(input, output, num_items, transform_op, stream));
    }

    template<typename RandomAccessIteratorIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Transform(void*                   d_temp_storage,
                                size_t&                 temp_storage_bytes,
                                RandomAccessIteratorIn  input,
                                RandomAccessIteratorOut output,
                                NumItemsT               num_items,
                                TransformOp             transform_op,
                                hipStream_t             stream = nullptr)
    {

        return hipCUDAErrorTohipError(::cub::DeviceTransform::Transform(d_temp_storage,
                                                                        temp_storage_bytes,
                                                                        input,
                                                                        output,
                                                                        num_items,
                                                                        transform_op,
                                                                        stream));
    }

    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        TransformStableArgumentAddresses(std::tuple<RandomAccessIteratorsIn...> inputs,
                                         RandomAccessIteratorOut                output,
                                         NumItemsT                              num_items,
                                         TransformOp                            transform_op,
                                         hipStream_t                            stream = nullptr)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceTransform::TransformStableArgumentAddresses(inputs,
                                                                     output,
                                                                     num_items,
                                                                     transform_op,
                                                                     stream));
    }

    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        TransformStableArgumentAddresses(void*                                  d_temp_storage,
                                         size_t&                                temp_storage_bytes,
                                         std::tuple<RandomAccessIteratorsIn...> inputs,
                                         RandomAccessIteratorOut                output,
                                         NumItemsT                              num_items,
                                         TransformOp                            transform_op,
                                         hipStream_t                            stream = nullptr)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceTransform::TransformStableArgumentAddresses(d_temp_storage,
                                                                     temp_storage_bytes,
                                                                     inputs,
                                                                     output,
                                                                     num_items,
                                                                     transform_op,
                                                                     stream));
    }

    template<typename RandomAccessIteratorIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformStableArgumentAddresses(RandomAccessIteratorIn  input,
                                                       RandomAccessIteratorOut output,
                                                       NumItemsT               num_items,
                                                       TransformOp             transform_op,
                                                       hipStream_t             stream = nullptr)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceTransform::TransformStableArgumentAddresses(input,
                                                                     output,
                                                                     num_items,
                                                                     transform_op,
                                                                     stream));
    }

    template<typename RandomAccessIteratorIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformStableArgumentAddresses(void*                   d_temp_storage,
                                                       size_t&                 temp_storage_bytes,
                                                       RandomAccessIteratorIn  input,
                                                       RandomAccessIteratorOut output,
                                                       NumItemsT               num_items,
                                                       TransformOp             transform_op,
                                                       hipStream_t             stream = nullptr)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceTransform::TransformStableArgumentAddresses(d_temp_storage,
                                                                     temp_storage_bytes,
                                                                     input,
                                                                     output,
                                                                     num_items,
                                                                     transform_op,
                                                                     stream));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIBCUB_ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_
