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

/*! \file thrust/system/hip/hipstdpar/impl/generation.hpp
 *  \brief <tt>Generation operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/generate.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN FILL
    template<
        typename I,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    void fill(execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        return ::thrust::fill(::thrust::device, f, l, x);
    }

    template<
        typename I,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    void fill(execution::parallel_unsequenced_policy, I f, I l, const T& x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::fill(::std::execution::par, f, l, x);
    }
    // END FILL

    // BEGIN FILL_N
    template<
        typename I,
        typename N,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    void fill_n(
        execution::parallel_unsequenced_policy, I f, N n, const T& x)
    {
        return ::thrust::fill_n(::thrust::device, f, n, x);
    }

    template<
        typename I,
        typename N,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    void fill_n(
        execution::parallel_unsequenced_policy, I f, N n, const T& x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::fill_n(::std::execution::par, f, n, x);
    }
    // END FILL_N

    // BEGIN GENERATE
    template<
        typename I,
        typename G,
        enable_if_t<
                ::hipstd::is_offloadable_iterator<I>() &&
                ::hipstd::is_offloadable_callable<G>()>* = nullptr>
    inline
    void generate(execution::parallel_unsequenced_policy, I f, I l, G g)
    {
        return ::thrust::generate(::thrust::device, f, l, ::std::move(g));
    }

    template<
        typename I,
        typename G,
        enable_if_t<
                !::hipstd::is_offloadable_iterator<I>() ||
                !::hipstd::is_offloadable_callable<G>()>* = nullptr>
    inline
    void generate(execution::parallel_unsequenced_policy, I f, I l, G g)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
                ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<G>()) {
                ::hipstd::unsupported_callable_type<G>();
        }

        return
                ::std::generate(::std::execution::par, f, l, ::std::move(g));
    }
    // END GENERATE

    // BEGIN GENERATE_N
    template<
        typename I,
        typename N,
        typename G,
        enable_if_t<
                ::hipstd::is_offloadable_iterator<I>() &&
                ::hipstd::is_offloadable_callable<G>()>* = nullptr>
    inline
    void generate_n(execution::parallel_unsequenced_policy, I f, N n, G g)
    {
        return ::thrust::generate_n(::thrust::device, f, n, ::std::move(g));
    }

    template<
        typename I,
        typename N,
        typename G,
        enable_if_t<
                !::hipstd::is_offloadable_iterator<I>() ||
                !::hipstd::is_offloadable_callable<G>()>* = nullptr>
    inline
    void generate_n(execution::parallel_unsequenced_policy, I f, N n, G g)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
                ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<G>()) {
                ::hipstd::unsupported_callable_type<G>();
        }

        return
                ::std::generate_n(::std::execution::par, f, n, ::std::move(g));
    }
    // END GENERATE_N
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
