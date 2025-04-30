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

/*! \file thrust/system/hip/hipstdpar/impl/numeric.hpp
 *  \brief <tt>Numeric operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN ADJACENT_DIFFERENCE
    template<
        typename I,
        typename O,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O adjacent_difference(
        execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        return ::thrust::adjacent_difference(::thrust::device, fi, li, fo);
    }

    template<
        typename I,
        typename O,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O adjacent_difference(
        execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return
            ::std::adjacent_difference(::std::execution::par, fi, li, fo);
    }


    template<
        typename I,
        typename O,
        typename Op,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    O adjacent_difference(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, Op op)
    {
        return ::thrust::adjacent_difference(
            ::thrust::device, fi, li, fo, ::std::move(op));
    }

    template<
        typename I,
        typename O,
        typename Op,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    O adjacent_difference(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, Op op)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op>()) {
            ::hipstd::unsupported_callable_type<Op>();
        }

        return ::std::adjacent_difference(
            ::std::execution::par, fi, li, fo, ::std::move(op));
    }
    // END ADJACENT_DIFFERENCE

    // BEGIN REDUCE
    template<
        typename I,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    typename iterator_traits<I>::value_type reduce(
        execution::parallel_unsequenced_policy, I f, I l)
    {
        return ::thrust::reduce(::thrust::device, f, l);
    }

    template<
        typename I,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    typename iterator_traits<I>::value_type reduce(
        execution::parallel_unsequenced_policy, I f, I l)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::reduce(::std::execution::par, f, l);
    }

    template<
        typename I,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    T reduce(execution::parallel_unsequenced_policy, I f, I l, T x)
    {
        return ::thrust::reduce(::thrust::device, f, l, ::std::move(x));
    }

    template<
        typename I,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
    inline
    T reduce(execution::parallel_unsequenced_policy, I f, I l, T x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category>();

        return ::std::reduce(::std::execution::par, f, l, ::std::move(x));
    }

    template<
        typename I,
        typename T,
        typename Op,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    T reduce(execution::parallel_unsequenced_policy, I f, I l, T x, Op op)
    {
        return ::thrust::reduce(
            ::thrust::device, f, l, ::std::move(x), ::std::move(op));
    }

    template<
        typename I,
        typename T,
        typename Op,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    T reduce(execution::parallel_unsequenced_policy, I f, I l, T x, Op op)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op>()) {
            ::hipstd::unsupported_callable_type<Op>();
        }

        return ::std::reduce(
            ::std::execution::par, f, l, ::std::move(x), ::std::move(op));
    }
    // END REDUCE

    // BEGIN EXCLUSIVE_SCAN
    template<
        typename I,
        typename O,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O exclusive_scan(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, T x)
    {
        return ::thrust::exclusive_scan(
            ::thrust::device, fi, li, fo, ::std::move(x));
    }

    template<
        typename I,
        typename O,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O exclusive_scan(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, T x)
    {
        ::hipstd::unsupported_iterator_category<
            typename std::iterator_traits<I>::iterator_category,
            typename std::iterator_traits<O>::iterator_category>();

        return ::std::exclusive_scan(
            ::std::execution::par, fi, li, fo, ::std::move(x));
    }

    template<
        typename I,
        typename O,
        typename T,
        typename Op,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    O exclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        T x,
        Op op)
    {
        return ::thrust::exclusive_scan(
            ::thrust::device, fi, li, fo, ::std::move(x), ::std::move(op));
    }

    template<
        typename I,
        typename O,
        typename T,
        typename Op,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    O exclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        T x,
        Op op)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op>()) {
            ::hipstd::unsupported_callable_type<Op>();
        }

        return ::std::exclusive_scan(
            ::std::execution::par,
            fi,
            li,
            fo,
            ::std::move(x),
            ::std::move(op));
    }
    // END EXCLUSIVE_SCAN

    // BEGIN INCLUSIVE_SCAN
    template<
        typename I,
        typename O,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O inclusive_scan(
        execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        return ::thrust::inclusive_scan(::thrust::device, fi, li, fo);
    }

    template<
        typename I,
        typename O,
        typename T,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O inclusive_scan(
        execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::inclusive_scan(::std::execution::par, fi, li, fo);
    }

    template<
        typename I,
        typename O,
        typename Op,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    O inclusive_scan(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, Op op)
    {
        return ::thrust::inclusive_scan(
            ::thrust::device, fi, li, fo, ::std::move(op));
    }

    template<
        typename I,
        typename O,
        typename Op,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    O inclusive_scan(
        execution::parallel_unsequenced_policy, I fi, I li, O fo, Op op)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op>()) {
            ::hipstd::unsupported_callable_type<Op>();
        }

        return ::std::inclusive_scan(
            ::std::execution::par, fi, li, fo, ::std::move(op));
    }

    template<
        typename I,
        typename O,
        typename Op,
        typename T,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    O inclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        Op op,
        T x)
    {   // TODO: this is highly inefficient due to rocThrust not exposing
        //       this particular interface where the user provides x.
        if (fi == li) return fo;

        auto lo =
            ::thrust::inclusive_scan(::thrust::device, fi, li, fo, op);

        return ::thrust::transform(
            ::thrust::device,
            fo,
            lo,
            fo,
            [op = ::std::move(op), x = ::std::move(x)](auto&& y) {
            return op(x, y);
        });
    }

    template<
        typename I,
        typename O,
        typename Op,
        typename T,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<Op>()>* = nullptr>
    inline
    O inclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        Op op,
        T x)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op>()) {
            ::hipstd::unsupported_callable_type<Op>();
        }

        return ::std::inclusive_scan(
            ::std::execution::par,
            fi,
            li,
            fo,
            ::std::move(op),
            ::std::move(x));
    }
    // END INCLUSIVE_SCAN

    // BEGIN TRANSFORM_REDUCE
    template<
        typename I0,
        typename I1,
        typename T,
        enable_if_t<::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    T transform_reduce(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        T x)
    {
        return ::thrust::inner_product(
            ::thrust::device, f0, l0, f1, ::std::move(x));
    }

    template<
        typename I0,
        typename I1,
        typename T,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>()>* = nullptr>
    inline
    T transform_reduce(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        T x)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I0>::iterator_category,
            typename iterator_traits<I1>::iterator_category>();

        return ::std::transform_reduce(
            ::std::execution::par, f0, l0, f1, ::std::move(x));
    }

    template<
        typename I0,
        typename I1,
        typename T,
        typename Op0,
        typename Op1,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I0, I1>() &&
            ::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    T transform_reduce(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        T x,
        Op0 op0,
        Op1 op1)
    {
        return ::thrust::inner_product(
            ::thrust::device,
            f0,
            l0,
            f1,
            ::std::move(x),
            ::std::move(op0),
            ::std::move(op1));
    }

    template<
        typename I0,
        typename I1,
        typename T,
        typename Op0,
        typename Op1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I0, I1>() ||
            !::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    T transform_reduce(
        execution::parallel_unsequenced_policy,
        I0 f0,
        I0 l0,
        I1 f1,
        T x,
        Op0 op0,
        Op1 op1)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I0, I1>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I0>::iterator_category,
                typename iterator_traits<I1>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op0, Op1>()) {
            ::hipstd::unsupported_callable_type<Op0, Op1>();
        }

        return ::std::transform_reduce(
            ::std::execution::par,
            f0,
            l0,
            f1,
            ::std::move(x),
            ::std::move(op0),
            ::std::move(op1));
    }

    template<
        typename I,
        typename T,
        typename Op0,
        typename Op1,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I>() &&
            ::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    T transform_reduce(
        execution::parallel_unsequenced_policy,
        I f,
        I l,
        T x,
        Op0 op0,
        Op1 op1)
    {
        return ::thrust::transform_reduce(
            ::thrust::device,
            f,
            l,
            ::std::move(op1),
            ::std::move(x),
            ::std::move(op0));
    }

    template<
        typename I,
        typename T,
        typename Op0,
        typename Op1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I>() ||
            !::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    T transform_reduce(
        execution::parallel_unsequenced_policy,
        I f,
        I l,
        T x,
        Op0 op0,
        Op1 op1)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op0, Op1>()) {
            ::hipstd::unsupported_callable_type<Op0, Op1>();
        }

        return ::std::transform_reduce(
            ::std::execution::par,
            f,
            l,
            ::std::move(x),
            ::std::move(op0),
            ::std::move(op1));
    }
    // END TRANSFORM_REDUCE

    // BEGIN TRANSFORM_EXCLUSIVE_SCAN
    template<
        typename I,
        typename O,
        typename T,
        typename Op0,
        typename Op1,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    O transform_exclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        T x,
        Op0 op0,
        Op1 op1)
    {
        return ::thrust::transform_exclusive_scan(
            ::thrust::device,
            fi,
            li,
            fo,
            ::std::move(op1),
            ::std::move(x),
            ::std::move(op0));
    }

    template<
        typename I,
        typename O,
        typename T,
        typename Op0,
        typename Op1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    O transform_exclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        T x,
        Op0 op0,
        Op1 op1)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op0, Op1>()) {
            ::hipstd::unsupported_callable_type<Op0, Op1>();
        }

        return ::std::transform_exclusive_scan(
            ::std::execution::par,
            fi,
            li,
            fo,
            ::std::move(x),
            ::std::move(op0),
            ::std::move(op1));
    }
    // END TRANSFORM_EXCLUSIVE_SCAN

    // BEGIN TRANSFORM_INCLUSIVE_SCAN
    template<
        typename I,
        typename O,
        typename Op0,
        typename Op1,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    O transform_inclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        Op0 op0,
        Op1 op1)
    {
        return ::thrust::transform_inclusive_scan(
            ::thrust::device,
            fi,
            li,
            fo,
            ::std::move(op1),
            ::std::move(op0));
    }

    template<
        typename I,
        typename O,
        typename Op0,
        typename Op1,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    O transform_inclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        Op0 op0,
        Op1 op1)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op0, Op1>()) {
            ::hipstd::unsupported_callable_type<Op0, Op1>();
        }

        return ::std::transform_inclusive_scan(
            ::std::execution::par,
            fi,
            li,
            fo,
            ::std::move(op0),
            ::std::move(op1));
    }

    template<
        typename I,
        typename O,
        typename Op0,
        typename Op1,
        typename T,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    O transform_inclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        Op0 op0,
        Op1 op1,
        T x)
    {   // TODO: this is inefficient.
        if (fi == li) return fo;

        auto lo = ::thrust::transform_inclusive_scan(
            ::thrust::device,
            fi,
            li,
            fo,
            ::std::move(op1),
            op0);

        return ::thrust::transform(
            ::thrust::device,
            fo,
            lo,
            fo,
            [op0 = ::std::move(op0), x = ::std::move(x)](auto&& y) {
            return op0(x, y);
        });
    }

    template<
        typename I,
        typename O,
        typename Op0,
        typename Op1,
        typename T,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<Op0, Op1>()>* = nullptr>
    inline
    O transform_inclusive_scan(
        execution::parallel_unsequenced_policy,
        I fi,
        I li,
        O fo,
        Op0 op0,
        Op1 op1,
        T x)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<Op0, Op1>()) {
            ::hipstd::unsupported_callable_type<Op0, Op1>();
        }

        return ::std::transform_inclusive_scan(
            ::std::execution::par,
            fi,
            li,
            fo,
            ::std::move(op0),
            ::std::move(op1),
            ::std::move(x));
    }
    // END TRANSFORM_INCLUSIVE_SCAN
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
