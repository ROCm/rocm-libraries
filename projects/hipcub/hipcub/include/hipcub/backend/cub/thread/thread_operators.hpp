// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPCUB_CUB_THREAD_THREAD_OPERATORS_HPP_
#define HIPCUB_CUB_THREAD_THREAD_OPERATORS_HPP_

#include "../../../config.hpp"

#include <cub/thread/thread_operators.cuh> // IWYU pragma: export

#include <cuda/std/__functional/invoke.h>

#include <cuda/std/utility>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

template<typename Invokable, typename InputT, typename InitT = InputT>
using accumulator_t = ::cuda::std::__accumulator_t<Invokable, InputT, InitT>;

} // namespace detail

//! deprecated [Since 5.0]
struct HIPCUB_DEPRECATED_BECAUSE("Use hip::std::equal_to<T> instead.") Equality
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE
    inline constexpr bool operator()(T&& t, U&& u) const
    {
        return ::cuda::std::forward<T>(t) == ::cuda::std::forward<U>(u);
    }
};

//! deprecated [Since 5.0]
struct HIPCUB_DEPRECATED_BECAUSE("Use hip::std::not_equal_to<T> instead.") Inequality
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE
    inline constexpr bool operator()(T&& t, U&& u) const
    {
        return ::cuda::std::forward<T>(t) != ::cuda::std::forward<U>(u);
    }
};

//! deprecated [Since 5.0]
struct HIPCUB_DEPRECATED_BECAUSE("Use hip::std::plus<T> instead.") Sum
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE
    inline constexpr auto operator()(T&& t, U&& u) const -> decltype(auto)
    {
        return ::cuda::std::forward<T>(t) + ::cuda::std::forward<U>(u);
    }
};

//! deprecated [Since 5.0]
struct HIPCUB_DEPRECATED_BECAUSE("Use hip::std::minus<T> instead.") Difference
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE
    inline constexpr auto operator()(T&& t, U&& u) const -> decltype(auto)
    {
        return ::cuda::std::forward<T>(t) - ::cuda::std::forward<U>(u);
    }
};

//! deprecated [Since 5.0]
struct HIPCUB_DEPRECATED_BECAUSE("Use hip::std::divides<T> instead") Division
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE
    inline constexpr auto operator()(T&& t, U&& u) const -> decltype(auto)
    {
        return std::forward<T>(t) / std::forward<U>(u);
    }
};

//! deprecated [Since 5.0]
struct HIPCUB_DEPRECATED_BECAUSE("Use hip::maximum<T> instead.") Max
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE
    inline constexpr auto operator()(const T& t, const U& u) const ->
        typename ::cuda::std::common_type<T, U>::type
    {
        using R = typename ::cuda::std::common_type<T, U>::type;
        return (t < u) ? static_cast<R>(u) : static_cast<R>(t);
    }
};

//! deprecated [Since 5.0]
struct HIPCUB_DEPRECATED_BECAUSE("Use hip::minimum<T> instead") Min
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE
    inline constexpr auto operator()(const T& t, const U& u) const ->
        typename ::cuda::std::common_type<T, U>::type
    {
        using R = typename ::cuda::std::common_type<T, U>::type;
        return (u < t) ? static_cast<R>(u) : static_cast<R>(t);
    }
};

struct ArgMax
{
    template<class Key, class Value>
    HIPCUB_HOST_DEVICE
    inline constexpr ::cub::KeyValuePair<Key, Value>
        operator()(const ::cub::KeyValuePair<Key, Value>& a,
                   const ::cub::KeyValuePair<Key, Value>& b) const
    {
        return ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

struct ArgMin
{
    template<class Key, class Value>
    HIPCUB_HOST_DEVICE
    inline constexpr ::cub::KeyValuePair<Key, Value>
        operator()(const ::cub::KeyValuePair<Key, Value>& a,
                   const ::cub::KeyValuePair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_THREAD_THREAD_OPERATORS_HPP_
