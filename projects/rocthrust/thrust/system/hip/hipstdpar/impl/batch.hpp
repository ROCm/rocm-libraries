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

/*! \file thrust/system/hip/hipstdpar/impl/batch.hpp
 *  \brief <tt>Batch operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN FOR_EACH
    template<
        typename I,
        typename F,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<F>()>* = nullptr>
    inline
    void for_each(execution::parallel_unsequenced_policy, I f, I l, F fn)
    {
        ::thrust::for_each(::thrust::device, f, l, ::std::move(fn));
    }

    template<
        typename I,
        typename F,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<F>()>* = nullptr>
    inline
    void for_each(execution::parallel_unsequenced_policy, I f, I l, F fn)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<F>()) {
            ::hipstd::unsupported_callable_type<F>();
        }

        return
            ::std::for_each(::std::execution::par, f, l, ::std::move(fn));
    }
    // END FOR_EACH

    // BEGIN FOR_EACH_N
    template<
        typename I,
        typename N,
        typename F,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<F>()>* = nullptr>
    inline
    I for_each_n(execution::parallel_unsequenced_policy, I f, N n, F fn)
    {
        return
            ::thrust::for_each_n(::thrust::device, f, n, ::std::move(fn));
    }

    template<
        typename I,
        typename N,
        typename F,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<F>()>* = nullptr>
    inline
    I for_each_n(execution::parallel_unsequenced_policy, I f, N n, F fn)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<F>()) {
            ::hipstd::unsupported_callable_type<F>();
        }

        return
            ::std::for_each_n(::std::execution::par, f, n, ::std::move(fn));
    }
    // END FOR_EACH_N
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
