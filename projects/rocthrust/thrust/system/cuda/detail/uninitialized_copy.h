/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#  include <thrust/distance.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <iterator>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{

namespace __uninitialized_copy
{

template <class InputIt, class OutputIt>
struct functor
{
  InputIt input;
  OutputIt output;

  using InputType  = typename iterator_traits<InputIt>::value_type;
  using OutputType = typename iterator_traits<OutputIt>::value_type;

  THRUST_FUNCTION
  functor(InputIt input_, OutputIt output_)
      : input(input_)
      , output(output_)
  {}

  template <class Size>
  void THRUST_DEVICE_FUNCTION operator()(Size idx)
  {
    InputType const& in = raw_reference_cast(input[idx]);
    OutputType& out     = raw_reference_cast(output[idx]);

#  if defined(__CUDA__) && defined(__clang__)
    // XXX unsafe, but clang is seemngly unable to call in-place new
    out = in;
#  else
    ::new (static_cast<void*>(&out)) OutputType(in);
#  endif
  }
}; // struct functor

} // namespace __uninitialized_copy

template <class Derived, class InputIt, class Size, class OutputIt>
OutputIt _CCCL_HOST_DEVICE
uninitialized_copy_n(execution_policy<Derived>& policy, InputIt first, Size count, OutputIt result)
{
  using functor_t = __uninitialized_copy::functor<InputIt, OutputIt>;

  cuda_cub::parallel_for(policy, functor_t(first, result), count);

  return result + count;
}

template <class Derived, class InputIt, class OutputIt>
OutputIt _CCCL_HOST_DEVICE
uninitialized_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  return cuda_cub::uninitialized_copy_n(policy, first, thrust::distance(first, last), result);
}

} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
