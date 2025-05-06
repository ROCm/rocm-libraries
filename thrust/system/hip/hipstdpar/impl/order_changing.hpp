/*
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

/*! \file thrust/system/hip/hipstdpar/impl/order_changing.hpp
 *  \brief <tt>Order-changing operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/reverse.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN REVERSE
    template<
        typename I,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    void reverse(execution::parallel_unsequenced_policy, I f, I l)
    {
        return ::thrust::reverse(::thrust::device, f, l);
    }

    template<
        typename I,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    void reverse(execution::parallel_unsequenced_policy, I f, I l)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::reverse(::std::execution::par, f, l);
    }
    // END REVERSE

    // BEGIN REVERSE_COPY
    template<
        typename I,
        typename O,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    void reverse_copy(
        execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        return ::thrust::reverse_copy(::thrust::device, fi, li, fo);
    }

    template<
        typename I,
        typename O,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    void reverse_copy(
        execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::reverse_copy(::std::execution::par, fi, li, fo);
    }
    // END REVERSE_COPY

    // BEGIN ROTATE
    // TODO: UNIMPLEMENTED IN THRUST
    // END ROTATE

    // BEGIN ROTATE_COPY
    // TODO: UNIMPLEMENTED IN THRUST
    // END ROTATE_COPY

    // BEGIN SHIFT_LEFT
    // TODO: UNIMPLEMENTED IN THRUST
    // END SHIFT_LEFT

    // BEGIN SHIFT_RIGHT
    // TODO: UNIMPLEMENTED IN THRUST
    // END SHIFT_RIGHT
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
