/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2026, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
#define ROCPRIM_THREAD_THREAD_OPERATORS_HPP_

#include "../config.hpp"
#include "../functional.hpp"
#include "../types/key_value_pair.hpp"

#include <iterator>
#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

/// \brief Functor that tests for equality.
struct [[deprecated("Duplicate, use 'rocprim::equal_to' in functional.hpp instead.")]] equality
{
    /// \brief Invocation operator
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

/// \brief Functor that tests for inequality.
struct [[deprecated(
    "Duplicate, use 'rocprim::not_equal_to' in functional.hpp instead.")]] inequality
{
    /// \brief Invocation operator
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

/// \brief Functor that tests for inequality using a user-supplied equality comparator.
template <class EqualityOp>
struct inequality_wrapper
{
    EqualityOp op; ///< A binary equality comparator (see constructor for details).

    /// \brief Constructs the wrapper using the provided equality comparator.
    /// \param op a binary equality comparator.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ROCPRIM_HOST_DEVICE inline
    inequality_wrapper(EqualityOp op) : op(op) {}

    /// \brief Invocation operator
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const T &a, const T &b)
    {
        return !op(a, b);
    }
};

/// \brief Functor that returns the sum of its arguments.
struct [[deprecated("Duplicate, use 'rocprim::plus' in functional.hpp instead.")]] sum
{
    /// \brief Invocation operator
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

/// \brief Functor that returns the maximum of its arguments.
struct [[deprecated("Duplicate, use 'rocprim::max' in functional.hpp instead.")]] max
{
    /// \brief Invocation operator
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? b : a;
    }
};

/// \brief Functor that returns the minimum of its arguments.
struct [[deprecated("Duplicate, use 'rocprim::min' in functional.hpp instead.")]] min
{
    /// \brief Invocation operator
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? a : b;
    }
};

/// \brief Functor that returns the "arg max" of the given key-value pairs.
///
/// The "arg max" of a key-value pair is defined as:
/// - b: if b's value is larger, or if the values are equal but b's key is smaller.
/// - a: otherwise
struct arg_max
{
    /// \brief Invocation operator
    template<
        class Key,
        class Value
    >
    ROCPRIM_HOST_DEVICE inline
    constexpr key_value_pair<Key, Value>
    operator()(const key_value_pair<Key, Value>& a,
               const key_value_pair<Key, Value>& b) const
    {
        greater<Value>  greater_v;
        equal_to<Value> eq_v;
        less<Key>       less_k;
        return (greater_v(b.value, a.value) || (eq_v(a.value, b.value) && less_k(b.key, a.key)))
                   ? b
                   : a;
    }
};

/// \brief Functor that returns the "arg min" of the given key-value pairs.
///
/// The "arg min" of a key-value pair is defined as:
/// - b: if b's value is smaller, or if the values are equal but b's key is smaller.
/// - a: otherwise
struct arg_min
{
    /// \brief Invocation operator (see struct description for details)
    template<
        class Key,
        class Value
    >
    ROCPRIM_HOST_DEVICE inline
    constexpr key_value_pair<Key, Value>
    operator()(const key_value_pair<Key, Value>& a,
               const key_value_pair<Key, Value>& b) const
    {
        less<Value>     less_v;
        equal_to<Value> eq_v;
        less<Key>       less_k;
        return (less_v(b.value, a.value) || (eq_v(a.value, b.value) && less_k(b.key, a.key))) ? b
                                                                                              : a;
    }
};

/// @}
// end group thread_operators

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
