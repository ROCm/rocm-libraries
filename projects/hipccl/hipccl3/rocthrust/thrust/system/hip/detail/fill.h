/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
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
#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/transform.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

template <class Derived, class OutputIterator, class Size, class T>
OutputIterator THRUST_HIP_FUNCTION
fill_n(execution_policy<Derived>& policy,
       OutputIterator             first,
       Size                       count,
       const T&                   value)
{
    return hip_rocprim::transform(policy,
                                  thrust::make_counting_iterator<Size>(0),
                                  thrust::make_counting_iterator<Size>(count),
                                  first,
                                  [value] __host__ __device__ (Size) { return value; });
} // func fill_n

template <class Derived, class ForwardIterator, class T>
void THRUST_HIP_FUNCTION
fill(execution_policy<Derived>& policy,
     ForwardIterator            first,
     ForwardIterator            last,
     const T&                   value)
{
    hip_rocprim::fill_n(policy, first, thrust::distance(first, last), value);
} // func filll

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
