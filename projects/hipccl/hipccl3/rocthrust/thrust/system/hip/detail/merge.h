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
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>
#include <thrust/detail/range/head_flags.h>
#include <thrust/merge.h>

// rocPRIM includes
#include <rocprim/rocprim.hpp>


BEGIN_NS_THRUST

namespace hip_rocprim {

namespace __merge{

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ResultIt,
          class CompareOp>
ResultIt THRUST_HIP_RUNTIME_FUNCTION
merge(execution_policy<Derived>& policy,
      KeysIt1                    keys1_first,
      KeysIt1                    keys1_last,
      KeysIt2                    keys2_first,
      KeysIt2                    keys2_last,
      ResultIt                   result,
      CompareOp                  compare_op)

{
    typedef size_t size_type;

    size_type    input1_size         = static_cast<size_type>(thrust::distance(keys1_first, keys1_last));
    size_type    input2_size         = static_cast<size_type>(thrust::distance(keys2_first, keys2_last));

    void *       d_temp_storage     = NULL;
    size_t       temp_storage_bytes = 0;
    hipStream_t  stream             = hip_rocprim::stream(policy);
    bool         debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

    hipError_t status;
    status = rocprim::merge(  d_temp_storage, 
                              temp_storage_bytes,
                              keys1_first, 
                              keys2_first, 
                              result, 
                              input1_size, 
                              input2_size,
                              compare_op,
                              stream,
                              debug_sync
                              );
    hip_rocprim::throw_on_error(status, "merge failed on 1st step");

    temp_storage_bytes = rocprim::detail::align_size(temp_storage_bytes);
    d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
    hip_rocprim::throw_on_error(hipGetLastError(),
                                "merge failed to get memory buffer");

    status = rocprim::merge(  d_temp_storage, 
                              temp_storage_bytes,
                              keys1_first, 
                              keys2_first, 
                              result, 
                              input1_size, 
                              input2_size,
                              compare_op,
                              stream,
                              debug_sync
                              );
    hip_rocprim::throw_on_error(status, "merge failed on 2nd step");

    hip_rocprim::return_memory_buffer(policy, d_temp_storage);
    hip_rocprim::throw_on_error(hipGetLastError(),
                                "merge failed to return memory buffer");

    ResultIt result_end = result + input1_size + input2_size;
    return result_end;
}
} //namespace merge

//-------------------------
// Thrust API entry points
//-------------------------
__thrust_exec_check_disable__
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ResultIt,
          class CompareOp>
ResultIt THRUST_HIP_FUNCTION
merge(execution_policy<Derived>& policy,
      KeysIt1                    keys1_first,
      KeysIt1                    keys1_last,
      KeysIt2                    keys2_first,
      KeysIt2                    keys2_last,
      ResultIt                   result,
      CompareOp                  compare_op)

{
  ResultIt ret = result;
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
      __merge::merge<Derived, KeysIt1, KeysIt2, ResultIt, CompareOp>
  ));
#if __THRUST_HAS_HIPRT__
    typedef typename thrust::iterator_value<KeysIt1>::type keys_type;
    keys_type* null_ = NULL;

    ret = __merge::merge(policy,
                        keys1_first,
                        keys1_last,
                        keys2_first,
                        keys2_last,
                        result,
                        compare_op);
#else
    ret = thrust::merge(cvt_to_seq(derived_cast(policy)),
                        keys1_first,
                        keys1_last,
                        keys2_first,
                        keys2_last,
                        result,
                        compare_op);

#endif
  return ret;
}

__thrust_exec_check_disable__
template <class Derived, class KeysIt1, class KeysIt2, class ResultIt>
ResultIt
THRUST_HIP_FUNCTION
merge(execution_policy<Derived>& policy,
      KeysIt1                    keys1_first,
      KeysIt1                    keys1_last,
      KeysIt2                    keys2_first,
      KeysIt2                    keys2_last,
      ResultIt                   result)
{
  typedef typename thrust::iterator_value<KeysIt1>::type keys_type;
  return hip_rocprim::merge(policy,
                         keys1_first,
                         keys1_last,
                         keys2_first,
                         keys2_last,
                         result,
                         less<keys_type>());
}
}// namespace hip_rocprim

END_NS_THRUST
#endif
