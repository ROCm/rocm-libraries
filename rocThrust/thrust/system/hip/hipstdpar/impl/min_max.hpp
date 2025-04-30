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

/*! \file thrust/system/hip/hipstdpar/impl/min_max.hpp
 *  \brief <tt>Minimum/maximum operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN MAX_ELEMENT
    template<
        typename I,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I max_element(execution::parallel_unsequenced_policy, I f, I l)
    {
        return ::thrust::max_element(::thrust::device, f, l);
    }

    template<
        typename I,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I max_element(execution::parallel_unsequenced_policy, I f, I l)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::max_element(::std::execution::par, f, l);
    }

    template<
        typename I,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    I max_element(execution::parallel_unsequenced_policy, I f, I l, R r)
    {
        return
            ::thrust::max_element(::thrust::device, f, l, ::std::move(r));
    }

    template<
        typename I,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    I max_element(execution::parallel_unsequenced_policy, I f, I l, R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }

        return
            ::std::max_element(::std::execution::par, f, l, ::std::move(r));
    }
    // END MAX_ELEMENT

    // BEGIN MIN_ELEMENT
    template<
        typename I,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I min_element(execution::parallel_unsequenced_policy, I f, I l)
    {
        return ::thrust::min_element(::thrust::device, f, l);
    }

    template<
        typename I,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I min_element(execution::parallel_unsequenced_policy, I f, I l)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::min_element(::std::execution::par, f, l);
    }

    template<
        typename I,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    I min_element(execution::parallel_unsequenced_policy, I f, I l, R r)
    {
        return
            ::thrust::min_element(::thrust::device, f, l, ::std::move(r));
    }

    template<
        typename I,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    I min_element(execution::parallel_unsequenced_policy, I f, I l, R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }

        return
            ::std::min_element(::std::execution::par, f, l, ::std::move(r));
    }
    // END MIN_ELEMENT

    // BEGIN MINMAX_ELEMENT
    template<
        typename I,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    pair<I, I> minmax_element(
        execution::parallel_unsequenced_policy, I f, I l)
    {
        auto [m, M] = ::thrust::minmax_element(::thrust::device, f, l);

        return {::std::move(m), ::std::move(M)};
    }

    template<
        typename I,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    pair<I, I> minmax_element(
        execution::parallel_unsequenced_policy, I f, I l)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::minmax_element(::std::execution::par, f, l);
    }

    template<
        typename I,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    pair<I, I> minmax_element(
        execution::parallel_unsequenced_policy, I f, I l, R r)
    {
        auto [m, M] = ::thrust::minmax_element(
            ::thrust::device, f, l, ::std::move(r));

        return {::std::move(m), ::std::move(M)};
    }

    template<
        typename I,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    pair<I, I> minmax_element(
        execution::parallel_unsequenced_policy, I f, I l, R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }

        return ::std::minmax_element(
            ::std::execution::par, f, l, ::std::move(r));
    }
    // END MINMAX_ELEMENT
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
