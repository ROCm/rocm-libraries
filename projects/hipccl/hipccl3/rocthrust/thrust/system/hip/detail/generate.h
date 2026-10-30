/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2026, Advanced Micro Devices, Inc.  All rights reserved.
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

#  include <thrust/system/hip/detail/parallel_for.h>
#  include <thrust/system/hip/detail/util.h>

#  include _THRUST_STD_INCLUDE(iterator)

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

namespace __generate
{
// generate functor
template <class OutputIt, class Generator>
struct functor
{
  OutputIt result;
  Generator generator;

  template <class Size>
  THRUST_HIP_DEVICE_FUNCTION void operator()(Size idx)
  {
    result[idx] = generator();
  }
}; // struct functor

} // namespace __generate

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class OutputIt, class Size, class Generator>
OutputIt THRUST_HOST_DEVICE
generate_n(execution_policy<Derived>& policy, OutputIt result, Size count, Generator generator)
{
  hip_rocprim::parallel_for(policy, __generate::functor<OutputIt, Generator>{result, generator}, count);

  return result + count;
}

template <class Derived, class OutputIt, class Generator>
void THRUST_HOST_DEVICE generate(execution_policy<Derived>& policy, OutputIt first, OutputIt last, Generator generator)
{
  hip_rocprim::generate_n(policy, first, _THRUST_STD::distance(first, last), generator);
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
