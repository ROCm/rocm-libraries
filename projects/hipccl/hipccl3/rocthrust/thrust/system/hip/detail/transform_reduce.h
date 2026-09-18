/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#  include <thrust/system/hip/config.h>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/minmax.h>
#  include <thrust/detail/raw_reference_cast.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#  include <thrust/distance.h>
#  include <thrust/functional.h>
#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/system/hip/detail/dispatch.h>
#  include <thrust/system/hip/detail/get_value.h>
#  include <thrust/system/hip/detail/par_to_seq.h>
#  include <thrust/system/hip/detail/reduce.h>
#  include <thrust/system/hip/detail/util.h>

#  include <cstdint>
#  include <iterator>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class TransformOp, class T, class ReduceOp>
T THRUST_HOST_DEVICE transform_reduce(
  execution_policy<Derived>& policy, InputIt first, InputIt last, TransformOp transform_op, T init, ReduceOp reduce_op)
{
  using size_type              = typename iterator_traits<InputIt>::difference_type;
  const size_type num_items    = static_cast<size_type>(thrust::distance(first, last));
  using transformed_iterator_t = transform_iterator<TransformOp, InputIt, T, T>;

  return reduce_n(policy, transformed_iterator_t(first, transform_op), num_items, init, reduce_op);
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
