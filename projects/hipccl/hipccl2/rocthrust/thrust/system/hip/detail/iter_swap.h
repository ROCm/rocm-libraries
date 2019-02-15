/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/detail/config.h>
#include <thrust/system/hip/config.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/swap.h>

BEGIN_NS_THRUST
namespace hip_rocprim {


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline __host__ __device__
void iter_swap(thrust::hip::execution_policy<DerivedPolicy> &, Pointer1 a, Pointer2 b)
{
#if defined(THRUST_HIP_DEVICE_CODE)

  /*THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
    thrust::swap<typename detail::pointer_element<Pointer1>::type, typename detail::pointer_element<Pointer2>::type>
  ));*/

  using thrust::swap;
  void (*fptr)(typename detail::pointer_element<Pointer1>::type&, typename detail::pointer_element<Pointer2>::type&) = swap;
  (void) fptr;

  swap(*thrust::raw_pointer_cast(a),
       *thrust::raw_pointer_cast(b));

#else

  thrust::swap_ranges(a, a + 1, b);

#endif
} // end iter_swap()


} // end hip_rocprim
END_NS_THRUST
#endif
