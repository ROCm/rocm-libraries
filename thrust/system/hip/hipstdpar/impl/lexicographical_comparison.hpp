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

/*! \file thrust/system/hip/hipstdpar/impl/lexicographical_comparison.hpp
 *  \brief <tt>Lexicographical comparison operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/execution_policy.h>
#include <thrust/mismatch.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN LEXICOGRAPHICAL_COMPARE
    template<
        typename I0,
        typename I1,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool lexicographical_compare(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        if (f0 == l0) return f1 != l1;
        if (f1 == l1) return false;

        const auto n0 = l0 - f0;
        const auto n1 = l1 - f1;
        const auto n = ::std::min(n0, n1);

        const auto m = ::thrust::mismatch(::thrust::device, f0, f0 + n, f1);

        if (m.first == f0 + n) return n0 < n1;

        return *m.first < *m.second;
    }

    template<
        typename I0,
        typename I1,
        enable_if_t<!::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    bool lexicographical_compare(
        execution::parallel_unsequenced_policy, I0 f0, I0 l0, I1 f1, I1 l1)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::lexicographical_compare(
            ::std::execution::par, f0, l0, f1, l1);
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool lexicographical_compare(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        I1 l1,
        R r)
    {
        if (f0 == l0) return f1 != l1;
        if (f1 == l1) return false;

        const auto n0 = l0 - f0;
        const auto n1 = l1 - f1;
        const auto n = ::std::min(n0, n1);

        const auto m = ::thrust::mismatch(
            ::thrust::device,
            f0,
            f0 + n,
            f1,
            [=](auto&& x, auto&& y) { return !r(x, y) && !r(y, x); });

        if (m.first == f0 + n) return n0 < n1;

        return r(*m.first, *m.second);
    }

    template<
        typename I0,
        typename I1,
        typename R,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<R>()>* = nullptr>
    inline
    bool lexicographical_compare(
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

        return ::std::lexicographical_compare(
            ::std::execution::par, f0, l0, f1, l1, ::std::move(r));
    }
    // END LEXICOGRAPHICAL_COMPARE
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
