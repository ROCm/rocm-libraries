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

/*! \file thrust/system/hip/hipstdpar/impl/removing.hpp
 *  \brief <tt>Removing operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/unique.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN REMOVE
    template<
        typename I,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I remove(execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        return ::thrust::remove(::thrust::device, f, l, x);
    }

    template<
        typename I,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I remove(execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::remove(::std::execution::par, f, l, x);
    }
    // END REMOVE

    // BEGIN REMOVE_IF
    template<
        typename I,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I remove_if(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        return ::thrust::remove_if(::thrust::device, f, l, ::std::move(p));
    }

    template<
        typename I,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    I remove_if(execution::parallel_unsequenced_policy, I f, I l, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return
            ::std::remove_if(::std::execution::par, f, l, ::std::move(p));
    }
    // END REMOVE_IF

    // BEGIN REMOVE_COPY
    template<
        typename I,
        typename O,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O remove_copy(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        const T& x)
    {
        return ::thrust::remove_copy(::thrust::device, fi, li, fo, x);
    }

    template<
        typename I,
        typename O,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O remove_copy(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        const T& x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::remove_copy(::std::execution::par, fi, li, fo, x);
    }
    // END REMOVE_COPY

    // BEGIN REMOVE_COPY_IF
    template<
        typename I,
        typename O,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    O remove_copy_if(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, P p)
    {
        return ::thrust::remove_copy_if(
            ::thrust::device, fi, li, fo, ::std::move(p));
    }

    template<
        typename I,
        typename O,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    O remove_copy_if(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::remove_copy_if(
            ::std::execution::par, fi, li, fo, ::std::move(p));
    }
    // END REMOVE_COPY_IF

    // BEGIN UNIQUE
    template<
        typename I,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I unique(execution::parallel_unsequenced_policy, I f, I l)
    {
        return ::thrust::unique(::thrust::device, f, l);
    }

    template<
        typename I,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    I unique(execution::parallel_unsequenced_policy, I f, I l)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::unique(::std::execution::par, f, l);
    }

    template<
        typename I,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    I unique(execution::parallel_unsequenced_policy, I f, I l, R r)
    {
        return ::thrust::unique(::thrust::device, f, l, ::std::move(r));
    }

    template<
        typename I,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    I unique(execution::parallel_unsequenced_policy, I f, I l, R r)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }

        return ::std::unique(::std::execution::par, f, l, ::std::move(r));
    }
    // END UNIQUE

    // BEGIN UNIQUE_COPY
    template<
        typename I,
        typename O,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O unique_copy(execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        return ::thrust::unique_copy(::thrust::device, fi, li, fo);
    }

    template<
        typename I,
        typename O,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O unique_copy(execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::unique_copy(::std::execution::par, fi, li, fo);
    }

    template<
        typename I,
        typename O,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    O unique_copy(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, R r)
    {
        return ::thrust::unique_copy(
            ::thrust::device, fi, li, fo, ::std::move(r));
    }

    template<
        typename I,
        typename O,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    O unique_copy(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, R r)
    {

        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<R>()) {
            ::hipstd::unsupported_callable_type<R>();
        }

        return ::std::unique_copy(
            ::std::execution::par, fi, li, fo, ::std::move(r));
    }
    // END UNIQUE_COPY
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
