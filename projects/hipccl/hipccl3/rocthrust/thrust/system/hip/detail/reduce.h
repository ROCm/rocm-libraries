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
#include <thrust/detail/config.h>

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

// rocprim includes
#include <rocprim/functional.hpp>
#include <rocprim/device/device_reduce_hip.hpp>


BEGIN_NS_THRUST

// forward declare generic reduce
// to circumvent circular dependency
template <typename DerivedPolicy,
          typename InputIterator,
          typename T,
          typename BinaryFunction>
T __host__ __device__
reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
       InputIterator                                               first,
       InputIterator                                               last,
       T                                                           init,
       BinaryFunction                                              binary_op);

namespace hip_rocprim {

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class InputIt, class Size, class T, class BinaryOp>
T THRUST_HIP_FUNCTION
reduce_n(execution_policy<Derived> &policy,
         InputIt                    first,
         Size                       num_items,
         T                          init,
         BinaryOp                   binary_op)
{
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
    rocprim::reduce<rocprim::default_config, InputIt, T*, T, BinaryOp>
  ));
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
    hip_rocprim::get_value<Derived, T*>
  ));
#if __THRUST_HAS_HIPRT__
  hipStream_t stream = hip_rocprim::stream(policy);


  // Determine temporary device storage requirements.
  T* ret_ptr = NULL;
  size_t tmp_size = 0;
  hip_rocprim::throw_on_error(
    rocprim::reduce(nullptr, tmp_size,
                    first, ret_ptr, init, num_items, binary_op,
                    stream, THRUST_HIP_DEBUG_SYNC_FLAG),
    "after reduction step 1");

    // Allocate temporary storage.

    detail::temporary_array<detail::uint8_t, Derived>
      tmp(policy, sizeof(T) + tmp_size);

  // Run reduction.

  // `tmp.begin()` yields a `normal_iterator`, which dereferences to a
  // `reference`, which has an `operator&` that returns a `pointer`, which
  // has a `.get` method that returns a raw pointer, which we can (finally)
  // `static_cast` to `void*`.
  //
  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.
  ret_ptr = detail::aligned_reinterpret_cast<T*>((&*tmp.begin()).get());
  void* tmp_ptr = static_cast<void*>((&*(tmp.begin() + sizeof(T))).get());
  hip_rocprim::throw_on_error(
    rocprim::reduce(tmp_ptr, tmp_size,
                    first, ret_ptr, init, num_items, binary_op,
                    stream, THRUST_HIP_DEBUG_SYNC_FLAG),
    "after reduction step 2");

  // Synchronize the stream and get the value.
  hipDeviceSynchronize();
  hip_rocprim::throw_on_error(hipGetLastError(),
    "reduce failed to synchronize");

  // `tmp.begin()` yields a `normal_iterator`, which dereferences to a
  // `reference`, which has an `operator&` that returns a `pointer`, which
  // has a `.get` method that returns a raw pointer, which we can (finally)
  // `static_cast` to `void*`.
  //
  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.
  return hip_rocprim::get_value(policy,
    detail::aligned_reinterpret_cast<T*>((&*tmp.begin()).get()));
#else // __THRUST_HAS_HIPRT__
  return thrust::reduce(
    cvt_to_seq(derived_cast(policy)), first, first + num_items, init, binary_op);
#endif // __THRUST_HAS_HIPRT__
}

template <class Derived, class InputIt, class T, class BinaryOp>
T __host__ __device__
reduce(execution_policy<Derived> &policy,
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

template <class Derived,
          class InputIt,
          class T>
T __host__ __device__
reduce(execution_policy<Derived> &policy,
       InputIt                    first,
       InputIt                    last,
       T                          init)
{
  return hip_rocprim::reduce(policy, first, last, init, plus<T>());
}

template <class Derived,
          class InputIt>
typename iterator_traits<InputIt>::value_type __host__ __device__
reduce(execution_policy<Derived> &policy,
       InputIt                    first,
       InputIt                    last)
{
  typedef typename iterator_traits<InputIt>::value_type value_type;
  return hip_rocprim::reduce(policy, first, last, value_type(0));
}


} // namespace rocprim

END_NS_THRUST

#include <thrust/memory.h>
#include <thrust/reduce.h>

#endif
