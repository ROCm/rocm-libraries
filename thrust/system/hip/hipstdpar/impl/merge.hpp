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

/*! \file thrust/system/hip/hipstdpar/impl/merge.hpp
 *  \brief <tt>Merge operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/merge.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN MERGE
    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1, O>()>* = nullptr>
    inline
    O merge(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        O fo)
    {
        return ::thrust::merge(::thrust::device, f0, l0, f1, l1, fo);
    }

    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1, O>()>* = nullptr>
    inline
    O merge(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::merge(::std::execution::par, f0, l0, f1, l1, fo);
    }

    template<
        typename I0,
        typename I1,
        typename O,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1, O>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    O merge(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        O fo,
        R r)
    {
        return ::thrust::merge(
            ::thrust::device, f0, l0, f1, l1, fo, ::std::move(r));
    }

    template<
        typename I0,
        typename I1,
        typename O,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1, O>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    O merge(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        O fo,
        R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }

        return ::std::merge(
            ::std::execution::par, f0, l0, f1, l1, fo, ::std::move(r));
    }
    // END MERGE

    // BEGIN INPLACE_MERGE
    // TODO: UNIMPLEMENTED IN THRUST
    // END INPLACE_MERGE
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
