/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#  include <thrust/iterator/permutation_iterator.h>
#  include <thrust/system/hip/detail/transform.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

template <class Derived, class MapIt, class ItemsIt, class ResultIt>
ResultIt THRUST_HIP_FUNCTION
gather(execution_policy<Derived>& policy, MapIt map_first, MapIt map_last, ItemsIt items, ResultIt result)
{
  return hip_rocprim::transform(
    policy,
    thrust::make_permutation_iterator(items, map_first),
    thrust::make_permutation_iterator(items, map_last),
    result,
    identity());
}

template <class Derived, class MapIt, class StencilIt, class ItemsIt, class ResultIt, class Predicate>
ResultIt THRUST_HIP_FUNCTION gather_if(
  execution_policy<Derived>& policy,
  MapIt map_first,
  MapIt map_last,
  StencilIt stencil,
  ItemsIt items,
  ResultIt result,
  Predicate predicate)
{
  return hip_rocprim::transform_if(
    policy,
    thrust::make_permutation_iterator(items, map_first),
    thrust::make_permutation_iterator(items, map_last),
    stencil,
    result,
    identity(),
    predicate);
}

template <class Derived, class MapIt, class StencilIt, class ItemsIt, class ResultIt>
ResultIt THRUST_HIP_FUNCTION gather_if(
  execution_policy<Derived>& policy, MapIt map_first, MapIt map_last, StencilIt stencil, ItemsIt items, ResultIt result)
{
  return hip_rocprim::gather_if(policy, map_first, map_last, stencil, items, result, identity());
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#endif
