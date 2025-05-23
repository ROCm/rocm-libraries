/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2024, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_REDUCE_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_REDUCE_HPP_

#include "../../../config.hpp"

BEGIN_HIPCUB_NAMESPACE

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal {

template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp,
    bool        NoPrefix = false>
__device__ __forceinline__ T ThreadReduce(
    T*           input,
    ReductionOp reduction_op,
    T           prefix = T(0))
{
    T retval;
    if(NoPrefix)
        retval = input[0];
    else
        retval = prefix;

    #pragma unroll
    for (int i = 0 + NoPrefix; i < LENGTH; ++i)
        retval = reduction_op(retval, input[i]);

    return retval;
}

template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T           (&input)[LENGTH],
    ReductionOp reduction_op,
    T           prefix)
{
    return ThreadReduce<LENGTH, false>((T*)input, reduction_op, prefix);
}

template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T           (&input)[LENGTH],
    ReductionOp reduction_op)
{
    return ThreadReduce<LENGTH, true>((T*)input, reduction_op);
}

}

END_HIPCUB_NAMESPACE

#endif
