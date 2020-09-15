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

#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/cross_system.h>
#include <thrust/system/hip/detail/execution_policy.h>

THRUST_BEGIN_NS

template <typename DerivedPolicy, typename InputIt, typename OutputIt>
__host__ __device__
OutputIt copy(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
              InputIt                                                     first,
              InputIt                                                     last,
              OutputIt                                                    result);

template <class DerivedPolicy, class InputIt, class Size, class OutputIt>
__host__ __device__
OutputIt copy_n(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                InputIt                                                     first,
                Size                                                        n,
                OutputIt                                                    result);

namespace hip_rocprim
{

// D->D copy requires HCC compiler
template <class System, class InputIterator, class OutputIterator>
OutputIterator THRUST_HIP_FUNCTION
copy(execution_policy<System>& system,
     InputIterator             first,
     InputIterator             last,
     OutputIterator            result);

template <class System1, class System2, class InputIterator, class OutputIterator>
OutputIterator __host__ /* WORKAROUND */ __device__
copy(cross_system<System1, System2> systems,
     InputIterator                  first,
     InputIterator                  last,
     OutputIterator                 result);

template <class System, class InputIterator, class Size, class OutputIterator>
OutputIterator THRUST_HIP_FUNCTION
copy_n(execution_policy<System>& system,
       InputIterator             first,
       Size                      n,
       OutputIterator            result);

template <class System1, class System2, class InputIterator, class Size, class OutputIterator>
OutputIterator __host__ /* WORKAROUND */ __device__
copy_n(cross_system<System1, System2> systems,
       InputIterator                  first,
       Size                           n,
       OutputIterator                 result);

} // namespace hip_rocprim
THRUST_END_NS

#include <thrust/system/hip/detail/internal/copy_cross_system.h>
#include <thrust/system/hip/detail/internal/copy_device_to_device.h>
#include <thrust/system/hip/detail/par_to_seq.h>

THRUST_BEGIN_NS
namespace hip_rocprim
{

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
// D->D copy requires HCC compiler

__thrust_exec_check_disable__ template <class System, class InputIterator, class OutputIterator>
OutputIterator THRUST_HIP_FUNCTION
copy(execution_policy<System>& system,
     InputIterator             first,
     InputIterator             last,
     OutputIterator            result)
{
  // struct workaround is required for HIP-clang
  // THRUST_HIP_PRESERVE_KERNELS_WORKAROUND is required for HCC
  struct workaround
  {
      __host__
      static OutputIterator par(
          execution_policy<System>& system,
          InputIterator             first,
          InputIterator             last,
          OutputIterator            result)
      {
      #if __HCC__ && __HIP_DEVICE_COMPILE__
      THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
          (__copy::device_to_device<System, InputIterator, OutputIterator>)
      );
      #else
      return __copy::device_to_device(system, first, last, result);
      #endif
      }
      __device__
      static OutputIterator seq(
          execution_policy<System>& system,
          InputIterator             first,
          InputIterator             last,
          OutputIterator            result)
      {
          return thrust::copy(cvt_to_seq(derived_cast(system)), first, last, result);
      }
  };

#if __THRUST_HAS_HIPRT__
    return workaround::par(system, first, last, result);
#else
    return workaround::seq(system, first, last, result);
#endif
} // end copy()

__thrust_exec_check_disable__ template <class System,
                                        class InputIterator,
                                        class Size,
                                        class OutputIterator>
OutputIterator THRUST_HIP_FUNCTION
copy_n(execution_policy<System>& system,
       InputIterator             first,
       Size                      n,
       OutputIterator            result)
{
  // struct workaround is required for HIP-clang
  // THRUST_HIP_PRESERVE_KERNELS_WORKAROUND is required for HCC
  struct workaround
  {
      __host__
      static OutputIterator par(
          execution_policy<System>& system,
          InputIterator             first,
          Size                      n,
          OutputIterator            result)
      {
      #if __HCC__ && __HIP_DEVICE_COMPILE__
      THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
          (__copy::device_to_device<System, InputIterator, OutputIterator>)
      );
      #else
      return __copy::device_to_device(system, first, first + n, result);
      #endif
      }
      __device__
      static OutputIterator seq(
          execution_policy<System>& system,
          InputIterator             first,
          Size                      n,
          OutputIterator            result)
      {
          return thrust::copy_n(cvt_to_seq(derived_cast(system)), first, n, result);
      }
  };
  #if __THRUST_HAS_HIPRT__
      return workaround::par(system, first, n, result);
  #else
      return workaround::seq(system, first, n, result);
  #endif
} // end copy_n()
#endif

template <class System1, class System2, class InputIterator, class OutputIterator>
OutputIterator __host__ /* WORKAROUND */ __device__
copy(cross_system<System1, System2> systems,
     InputIterator                  first,
     InputIterator                  last,
     OutputIterator                 result)
{
    return __copy::cross_system_copy(systems, first, last, result);
} // end copy()

template <class System1, class System2, class InputIterator, class Size, class OutputIterator>
OutputIterator __host__ /* WORKAROUND */ __device__
copy_n(cross_system<System1, System2> systems,
       InputIterator                  first,
       Size                           n,
       OutputIterator                 result)
{
    return __copy::cross_system_copy_n(systems, first, n, result);
} // end copy_n()

} // namespace hip_rocprim
THRUST_END_NS

#include <thrust/detail/temporary_array.h>
#include <thrust/memory.h>
