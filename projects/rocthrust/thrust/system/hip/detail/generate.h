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

#  include <thrust/system/hip/config.h>

#  include <thrust/distance.h>
#  include <thrust/system/hip/detail/for_each.h>

#  include <iterator>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

// for_each functor
template <class Generator>
struct generate_f
{
  Generator generator;

  THRUST_HIP_FUNCTION
  generate_f(Generator generator_)
      : generator(generator_)
  {}

  template <class T>
  THRUST_HIP_DEVICE_FUNCTION void operator()(T const& value)
  {
    T& lvalue = const_cast<T&>(value);
    lvalue    = generator();
  }
};

// for_each_n
template <class Derived, class OutputIt, class Size, class Generator>
OutputIt THRUST_HIP_FUNCTION
generate_n(execution_policy<Derived>& policy, OutputIt result, Size count, Generator generator)
{
  return hip_rocprim::for_each_n(policy, result, count, generate_f<Generator>(generator));
}

// for_each
template <class Derived, class OutputIt, class Generator>
void THRUST_HIP_FUNCTION generate(execution_policy<Derived>& policy, OutputIt first, OutputIt last, Generator generator)
{
  hip_rocprim::generate_n(policy, first, thrust::distance(first, last), generator);
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
