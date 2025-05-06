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

/*! \file thrust/system/hip/hipstdpar/impl/transformation.hpp
 *  \brief <tt>Transformation operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/replace.h>
#include <thrust/transform.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN TRANSFORM
    template<
        typename I,
        typename O,
        typename F,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<F>()>* = nullptr>
    inline
    O transform(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, F fn)
    {
        return ::thrust::transform(
            ::thrust::device, fi, li, fo, ::std::move(fn));
    }

    template<
        typename I,
        typename O,
        typename F,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<F>()>* = nullptr>
    inline
    O transform(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, F fn)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<F>()) {
            ::hipstd::unsupported_callable_type<F>();
        }

        return ::std::transform(
            ::std::execution::par, fi, li, fo, ::std::move(fn));
    }

    template<
        typename I0,
        typename I1,
        typename O,
        typename F,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1, O>() &&
            ::hipstd::is_offloadable_callable<F>()>* = nullptr>
    inline
    O transform(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        O fo,
        F fn)
    {
        return ::thrust::transform(
            ::thrust::device, fi0, li0, fi1, fo, ::std::move(fn));
    }

    template<
        typename I0,
        typename I1,
        typename O,
        typename F,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1, O>() ||
            !::hipstd::is_offloadable_callable<F>()>* = nullptr>
    inline
    O transform(
        execution::parallel_unsequenced_policy,
        I0 fi0,
        I0 li0,
        I1 fi1,
        O fo,
        F fn)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<F>()) {
            ::hipstd::unsupported_callable_type<F>();
        }

        return ::std::transform(
            ::std::execution::par, fi0, li0, fi1, fo, ::std::move(fn));
    }
    // END TRANSFORM

    // BEGIN REPLACE
    template<
        typename I,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    void replace(
        execution::parallel_unsequenced_policy,
        I f,
        I l,
        const T& x,
        const T& y)
    {
        return ::thrust::replace(::thrust::device, f, l, x, y);
    }

    template<
        typename I,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    void replace(
        execution::parallel_unsequenced_policy,
        I f,
        I l,
        const T& x,
        const T& y)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::replace(::std::execution::par, f, l, x, y);
    }
    // END REPLACE

    // BEGIN REPLACE_IF
    template<
        typename I,
        typename P,
        typename T,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    void replace_if(
        execution::parallel_unsequenced_policy, I f, I l, P p, const T& x)
    {
        return
            ::thrust::replace_if(::thrust::device, f, l, ::std::move(p), x);
    }

    template<
        typename I,
        typename P,
        typename T,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    void replace_if(
        execution::parallel_unsequenced_policy, I f, I l, P p, const T& x)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::replace_if(
            ::std::execution::par, f, l, ::std::move(p), x);
    }
    // END REPLACE_IF

    // BEGIN REPLACE_COPY
    template<
        typename I,
        typename O,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    void replace_copy(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        const T& x,
        const T& y)
    {
        return ::thrust::replace_copy(::thrust::device, fi, li, fo, x, y);
    }

    template<
        typename I,
        typename O,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    void replace_copy(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        const T& x,
        const T& y)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::replace_copy(::std::execution::par, fi, li, fo, x, y);
    }
    // END REPLACE_COPY

    // BEGIN REPLACE_COPY_IF
    template<
        typename I,
        typename O,
        typename P,
        typename T,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    void replace_copy_if(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        P p,
        const T& x)
    {
        return ::thrust::replace_copy_if(
            ::thrust::device, fi, li, fo, ::std::move(p), x);
    }

    template<
        typename I,
        typename O,
        typename P,
        typename T,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    void replace_copy_if(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        P p,
        const T& x)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::replace_copy_if(
            ::std::execution::par, fi, li, fo, ::std::move(p), x);
    }
    // END REPLACE_COPY_IF
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
