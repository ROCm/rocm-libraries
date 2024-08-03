// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/type_traits.h>

#include <hip/hip_runtime.h>

THRUST_NAMESPACE_BEGIN
namespace system { namespace hip { namespace detail {
template <typename T,
          typename U,
          std::enable_if_t<thrust::detail::is_integral<T>::value && std::is_unsigned<U>::value, int>
          = 0>
__host__ __device__ inline constexpr auto ceiling_div(const T a, const U b)
{
    return a / b + (a % b > 0 ? 1 : 0);
}

__host__ __device__ inline size_t align_size(size_t size, size_t alignment = 256)
{
    return ceiling_div(size, alignment) * alignment;
}

template <class Tuple, class Function, size_t... Indices>
__host__ __device__ inline void
apply_to_each_in_tuple_impl(Tuple&& t, Function&& f, thrust::index_sequence<Indices...>)
{
    int swallow[]
        = {(std::forward<Function>(f)(thrust::get<Indices>(std::forward<Tuple>(t))), 0)...};
    (void)swallow;
}

template <class Tuple, class Function>
__host__ __device__ inline auto apply_to_each_in_tuple(Tuple&& t, Function&& f)
    -> void_t<tuple_size<std::remove_reference_t<Tuple>>>
{
    static constexpr size_t size = tuple_size<std::remove_reference_t<Tuple>>::value;
    apply_to_each_in_tuple_impl(
        std::forward<Tuple>(t), std::forward<Function>(f), thrust::make_index_sequence<size>());
}

} // end namespace detail
} // end namespace hip
} // end namespace system
THRUST_NAMESPACE_END
