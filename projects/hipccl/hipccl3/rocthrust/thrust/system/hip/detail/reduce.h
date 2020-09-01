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
#include <thrust/system/hip/config.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/hip/detail/util.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>
#include <thrust/detail/alignment.h>


// rocprim include
#include <rocprim/rocprim.hpp>

THRUST_BEGIN_NS

// forward declare generic reduce
// to circumvent circular dependency
template <typename DerivedPolicy,
          typename InputIterator,
          typename T,
          typename BinaryFunction>
__host__ __device__
T reduce(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
         InputIterator                                               first,
         InputIterator                                               last,
         T                                                           init,
         BinaryFunction                                              binary_op);

namespace hip_rocprim
{
namespace __reduce
{
    template <typename Derived,
              typename InputIt,
              typename Size,
              typename T,
              typename BinaryOp>
    THRUST_HIP_RUNTIME_FUNCTION
    T reduce(execution_policy<Derived>& policy, InputIt first, Size num_items, T init, BinaryOp binary_op)
    {
        if(num_items == 0)
            return init;


        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::reduce(NULL,
                                                    temp_storage_bytes,
                                                    first,
                                                    reinterpret_cast<T*>(NULL),
                                                    init,
                                                    num_items,
                                                    binary_op,
                                                    stream,
                                                    debug_sync),
                                    "reduce failed on 1st step");

        size_t storage_size = temp_storage_bytes + sizeof(T);

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        T* d_ret_ptr = reinterpret_cast<T*>(
            reinterpret_cast<char*>(ptr) + temp_storage_bytes);

        hip_rocprim::throw_on_error(rocprim::reduce(ptr,
                                                    temp_storage_bytes,
                                                    first,
                                                    d_ret_ptr,
                                                    init,
                                                    num_items,
                                                    binary_op,
                                                    stream,
                                                    debug_sync),
                                    "reduce failed on 2nd step");

        T return_value = hip_rocprim::get_value(policy, d_ret_ptr);

        return return_value;
    }
}

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class InputIt, class Size, class T, class BinaryOp>
THRUST_HIP_FUNCTION
T reduce_n(execution_policy<Derived>& policy,
           InputIt                    first,
           Size                       num_items,
           T                          init,
           BinaryOp                   binary_op)
{

  struct workaround
  {
      __host__
      static T par(execution_policy<Derived>& policy,
                      InputIt                    first,
                      Size                       num_items,
                      T                          init,
                      BinaryOp                   binary_op)
      {
      #if __HCC__ && __HIP_DEVICE_COMPILE__
      THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
          (__reduce::reduce<Derived, InputIt, Size, T, BinaryOp>)
      );
      #else
      return __reduce::reduce(
          policy,
          first,
          num_items,
          init,
          binary_op
      );
      #endif
      }
      __device__
      static T seq(execution_policy<Derived>& policy,
                      InputIt                    first,
                      Size                       num_items,
                      T                          init,
                      BinaryOp                   binary_op)
      {
        return thrust::reduce(
             cvt_to_seq(derived_cast(policy)),
             first,
             first + num_items,
             init,
             binary_op
          );
      }
  };

  #if __THRUST_HAS_HIPRT__
    return workaround::par(policy, first, num_items, init, binary_op);
  #else
    return workaround::seq(policy, first, num_items, init, binary_op);
  #endif


}

template <class Derived, class InputIt, class T, class BinaryOp>
THRUST_HIP_FUNCTION
T reduce(execution_policy<Derived>& policy,
         InputIt                    first,
         InputIt                    last,
         T                          init,
         BinaryOp                   binary_op)
{
    typedef typename iterator_traits<InputIt>::difference_type size_type;
    // FIXME: Check for RA iterator.
    size_type num_items = static_cast<size_type>(thrust::distance(first, last));
    return hip_rocprim::reduce_n(policy, first, num_items, init, binary_op);
}

template <class Derived, class InputIt, class T>
THRUST_HIP_FUNCTION
T reduce(execution_policy<Derived>& policy,
         InputIt                    first,
         InputIt                    last,
         T                          init)
{
    return hip_rocprim::reduce(policy, first, last, init, plus<T>());
}

template <class Derived, class InputIt>
THRUST_HIP_FUNCTION
typename iterator_traits<InputIt>::value_type
reduce(execution_policy<Derived>& policy,
       InputIt                    first,
       InputIt                    last)
{
    typedef typename iterator_traits<InputIt>::value_type value_type;
    return hip_rocprim::reduce(policy, first, last, value_type(0));
}

} // namespace  hip_rocprim

THRUST_END_NS

#include <thrust/memory.h>
#include <thrust/reduce.h>

#endif
