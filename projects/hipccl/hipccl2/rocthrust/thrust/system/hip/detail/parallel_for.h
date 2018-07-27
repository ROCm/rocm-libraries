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


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/system/hip/config.h>

#include <thrust/system/hip/detail/util.h>
#include <thrust/detail/type_traits/result_of_adaptable_function.h>
#include <thrust/system/hip/detail/par_to_seq.h>
// #include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/hip/detail/par_to_seq.h>

BEGIN_NS_THRUST

namespace hip_rocprim {

namespace __parallel_for {

  template <class F,
            class Size>
  THRUST_RUNTIME_FUNCTION hipError_t
  parallel_for(Size         num_items,
               F            f,
               hipStream_t stream)
  {
    bool debug_sync = THRUST_DEBUG_SYNC_FLAG;
    // STREAMHPC TODO implement parallel_for
    (void) debug_sync;
    (void) num_items; (void) f; (void) stream;
    return hipSuccess;
  }
}    // __parallel_for

__thrust_exec_check_disable__
template <class Derived,
          class F,
          class Size>
void __host__ __device__
parallel_for(execution_policy<Derived> &policy,
             F                          f,
             Size                       count)
{
  if (count == 0)
    return;

  if(__THRUST_HAS_HIPRT__)
  {
    hipStream_t stream = hip_rocprim::stream(policy);
    hipError_t  status = __parallel_for::parallel_for(count, f, stream);
    hip_rocprim::throw_on_error(status, "parallel_for failed");
  }
  else
  {
#if !__THRUST_HAS_HIPRT__
    for (Size idx = 0; idx != count; ++idx)
      f(idx);
#endif
  }
}

}    // namespace hip_rocprim

END_NS_THRUST
#endif
