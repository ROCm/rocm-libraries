/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright© 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if THRUST_HAS_HIP_COMPILER()
#  include <thrust/system/hip/config.h>

#  include <thrust/system/hip/detail/transform.h>
#  include <thrust/system/hip/execution_policy.h>

#  include _THRUST_LIBCXX_INCLUDE(__iterator/counting_iterator.h)
#  include _THRUST_STD_INCLUDE(__iterator/distance.h)
#  include _THRUST_STD_INCLUDE(__iterator/incrementable_traits.h)

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
template <class Derived, class Iterator, class TabulateOp>
void THRUST_HOST_DEVICE
tabulate(execution_policy<Derived>& policy, Iterator first, Iterator last, TabulateOp tabulate_op)
{
  using size_type  = _THRUST_STD::iter_difference_t<Iterator>;
  const auto count = _THRUST_STD::distance(first, last);
  const auto begin = _THRUST_LIBCXX::counting_iterator<size_type>{};
  hip_rocprim::transform(policy, begin, begin + count, first, tabulate_op);
}
} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
