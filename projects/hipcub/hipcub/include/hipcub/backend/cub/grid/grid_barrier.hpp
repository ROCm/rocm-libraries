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

#ifndef HIPCUB_CUB_GRID_GRID_BARRIER_HPP_
#define HIPCUB_CUB_GRID_GRID_BARRIER_HPP_

#include "../../../config.hpp"

#include <cub/grid/grid_barrier.cuh> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE
_CCCL_SUPPRESS_DEPRECATED_PUSH
// This API will be deprecated, suggest to use hip cooperative groups apis.
class HIPCUB_DEPRECATED_BECAUSE("Use the APIs from cooperative groups instead") GridBarrierLifetime
    : public ::cub::GridBarrierLifetime
{
public:
    hipError_t HostReset()
    {
        return hipCUDAErrorTohipError(
            ::cub::GridBarrierLifetime::HostReset()
        );
    }

    hipError_t Setup(int sweep_grid_size)
    {
        return hipCUDAErrorTohipError(
            ::cub::GridBarrierLifetime::Setup(sweep_grid_size)
        );
    }
};
_CCCL_SUPPRESS_DEPRECATED_POP
END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_GRID_GRID_BARRIER_HPP_
