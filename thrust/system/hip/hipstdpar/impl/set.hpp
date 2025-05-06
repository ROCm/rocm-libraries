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

/*! \file thrust/system/hip/hipstdpar/impl/set.hpp
 *  \brief <tt>Set operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/set_operations.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN INCLUDES
    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool includes(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        ::thrust::discard_iterator<> cnt{0};

        return ::thrust::set_difference(
            ::thrust::device, f1, l1, f0, l0, cnt) == cnt;
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool includes(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::includes(::std::execution::par, f0, l0, f1, l1);
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool includes(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        R r)
    {
        ::thrust::discard_iterator<> cnt{0};

        return ::thrust::set_difference(
            ::thrust::device, f1, l1, f0, l0, cnt, ::std::move(r)) == cnt;
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool includes(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }

        return ::std::includes(
            ::std::execution::par, f1, l1, f0, l0, ::std::move(r));
    }
    // END INCLUDES

    // BEGIN SET_UNION
    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1, O>>* = nullptr>
    inline
    O set_union(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo)
    {
        return
            ::thrust::set_union(::thrust::device, fi0, li0, fi1, li1, fo);
    }

    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1, O>>* = nullptr>
    inline
    O set_union(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return
            ::std::set_union(::std::execution::par, fi0, li0, fi1, li1, fo);
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
    O set_union(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo,
        R r)
    {
        return ::thrust::set_union(
            ::thrust::device, fi0, li0, fi1, li1, fo, ::std::move(r));
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
    O set_union(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
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

        return ::std::set_union(
            ::std::execution::par, fi0, li0, fi1, li1, fo, ::std::move(r));
    }
    // END SET_UNION

    // BEGIN SET_INTERSECTION
    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1, O>()>* = nullptr>
    inline
    O set_intersection(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo)
    {
        return ::thrust::set_intersection(
            ::thrust::device, fi0, li0, fi1, li1, fo);
    }

    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1, O>()>* = nullptr>
    inline
    O set_intersection(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::set_intersection(
            ::std::execution::par, fi0, li0, fi1, li1, fo);
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
    O set_intersection(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo,
        R r)
    {
        return ::thrust::set_intersection(
            ::thrust::device, fi0, li0, fi1, li1, fo, ::std::move(r));
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
    O set_intersection(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
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

        return ::std::set_intersection(
            ::std::execution::par, fi0, li0, fi1, li1, fo, ::std::move(r));
    }
    // END SET_INTERSECTION

    // BEGIN SET_DIFFERENCE
    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1, O>()>* = nullptr>
    inline
    O set_difference(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo)
    {
        return ::thrust::set_difference(
            ::thrust::device, fi0, li0, fi1, li1, fo);
    }

    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1, O>()>* = nullptr>
    inline
    O set_difference(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::set_difference(
            ::std::execution::par, fi0, li0, fi1, li1, fo);
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
    O set_difference(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo,
        R r)
    {
        return ::thrust::set_difference(
            ::thrust::device, fi0, li0, fi1, li1, fo, ::std::move(r));
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
    O set_difference(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
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

        return ::std::set_difference(
            ::std::execution::par, fi0, li0, fi1, li1, fo, ::std::move(r));
    }
    // END SET_DIFFERENCE

    // BEGIN SET_SYMMETRIC_DIFFERENCE
    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1, O>()>* = nullptr>
    inline
    O set_symmetric_difference(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo)
    {
        return ::thrust::set_symmetric_difference(
            ::thrust::device, fi0, li0, fi1, li1, fo);
    }

    template<
        typename I0,
        typename I1,
        typename O,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1, O>()>* = nullptr>
    inline
    O set_symmetric_difference(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::set_symmetric_difference(
            ::std::execution::par, fi0, li0, fi1, li1, fo);
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
    O set_symmetric_difference(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
        O fo,
        R r)
    {
        return ::thrust::set_symmetric_difference(
            ::thrust::device, fi0, li0, fi1, li1, fo, ::std::move(r));
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
    O set_symmetric_difference(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        I1 li1,
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

        return ::std::set_symmetric_difference(
            ::std::execution::par, fi0, li0, fi1, li1, fo, ::std::move(r));
    }
    // END SET_SYMMETRIC_DIFFERENCE
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
