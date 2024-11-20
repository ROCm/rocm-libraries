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

/*! \file thrust/system/hip/hipstdpar/impl/search.hpp
 *  \brief <tt>Search operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/logical.h>
#include <thrust/mismatch.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN ALL_OF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool all_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::all_of(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool all_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::all_of(::std::execution::par, f, l, ::std::move(p));
    }
    // END ALL_OF

    // BEGIN ANY_OF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool any_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::any_of(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool any_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::any_of(::std::execution::par, f, l, ::std::move(p));
    }
    // END ANY_OF

    // BEGIN NONE_OF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool none_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::none_of(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    bool none_of(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::none_of(::std::execution::par, f, l, ::std::move(p));
    }
    // END NONE_OF

    // BEGIN FIND
    template<
        typename I,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I find(execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        return ::thrust::find(::thrust::device, f, l, x);
    }

    template<
        typename I,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I find(execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::find(::std::execution::par, f, l, x);
    }
    // END FIND

    // BEGIN FIND_IF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I find_if(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::find_if(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I find_if(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::find_if(::std::execution::par, f, l, ::std::move(p));
    }
    // END FIND_IF

    // BEGIN FIND_IF_NOT
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I find_if_not(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return
            ::thrust::find_if_not(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I find_if_not(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return
            ::std::find_if_not(::std::execution::par, f, l, ::std::move(p));
    }
    // END FIND_IF_NOT

    // BEGIN FIND_END
    // TODO: UNIMPLEMENTED IN THRUST
    // END FIND_END

    // BEGIN FIND_FIRST_OF
    // TODO: UNIMPLEMENTED IN THRUST
    // END FIND_FIRST_OF

    // BEGIN ADJACENT_FIND
    template<
        typename I,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I adjacent_find(execution::parallel_unsequenced_policy, I f, I l)
    {
        if (f == l) return l;

        const auto r = ::thrust::mismatch(
            ::thrust::device, f + 1, l, f, not_equal_to<>{});

        return (r.first == l) ? l : r.second;
    }

    template<
        typename I,
        typename P,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I adjacent_find(execution::parallel_unsequenced_policy, I f, I l)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::adjacent_find(::std::execution::par, f, l);
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I adjacent_find(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if (f == l) return l;

        const auto r = ::thrust::mismatch(
            ::thrust::device, f + 1, l, f, not_fn(::std::move(p)));

        return (r.first == l) ? l : r.second;
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I adjacent_find(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::adjacent_find(
            ::std::execution::par, f, l, ::std::move(p));
    }
    // END ADJACENT_FIND

    // BEGIN COUNT
    template<
        typename I,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    typename iterator_traits<I>::difference_type count(
        execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        return ::thrust::count(::thrust::device, f, l, x);
    }

    template<
        typename I,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    typename iterator_traits<I>::difference_type count(
        execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::count(::std::execution::par, f, l, x);
    }
    // END COUNT

    // BEGIN COUNT_IF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    typename iterator_traits<I>::difference_type count_if(
        execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::count_if(::thrust::device, f, l, ::std::move(p));
    }

        template<
        typename I,
        typename O,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    typename iterator_traits<I>::difference_type count_if(
        execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::count_if(::std::execution::par, f, l, ::std::move(p));
    }
    // END COUNT_IF

    // BEGIN MISMATCH
    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        auto [m0, m1] = ::thrust::mismatch(::thrust::device, f0, l0, f1);

        return {::std::move(m0), ::std::move(m1)};
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::mismatch(::std::execution::par, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, P p)
    {
        auto [m0, m1] = ::thrust::mismatch(
            ::thrust::device, f0, l0, f1, ::std::move(p));

        return {::std::move(m0), ::std::move(m1)};
    }

    template<
        typename I0,
        typename I1,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::mismatch(
            ::std::execution::par, f0, l0, f1, ::std::move(p));
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        const auto n = ::std::min(l0 - f0, l1 - f1);

        auto [m0, m1] =
            ::thrust::mismatch(::thrust::device, f0, f0 + n, f1);

        return {::std::move(m0), ::std::move(m1)};
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::mismatch(::std::execution::par, f0, l0, f1, l1);
    }

    template<
        typename I0,
        typename I1,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        P p)
    {
        const auto n = ::std::min(l0 - f0, l1 - f1);

        auto [m0, m1] = ::thrust::mismatch(
            ::thrust::device, f0, f0 + n, f1, ::std::move(p));

        return {::std::move(m0), ::std::move(m1)};
    }

    template<
        typename I0,
        typename I1,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    pair<I0, I1> mismatch(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::mismatch(
            ::std::execution::par, f0, l0, f1, l1, ::std::move(p));
    }
    // END MISMATCH

    // BEGIN EQUAL
    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool equal(execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        return ::thrust::equal(::thrust::device, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool equal(execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::equal(::std::execution::par, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, R r)
    {
        return
            ::thrust::equal(::thrust::device, f0, l0, f1, ::std::move(r));
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }
        return
            ::std::equal(::std::execution::par, f0, l0, f1, ::std::move(r));
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        if (l0 - f0 != l1 - f1) return false;

        return ::thrust::equal(::thrust::device, f0, l0, f1);
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::equal(::std::execution::par, f0, l0, f1, l1);
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool equal(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        R r)
    {
        if (l0 - f0 != l1 - f1) return false;

        return ::thrust::equal(
            ::thrust::device, f0, l0, f1, ::std::move(r));
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool equal(
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
        return ::std::equal(
            ::std::execution::par, f0, l0, f1, l1, ::std::move(r));
    }
    // END EQUAL

    // BEGIN SEARCH
    // TODO: UNIMPLEMENTED IN THRUST
    // END SEARCH

    // BEGIN SEARCH_N
    // TODO: UNIMPLEMENTED IN THRUST
    // END SEARCH_N
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
