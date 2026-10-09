// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef ROCPRIM_THREAD_THREAD_COPY_HPP_
#define ROCPRIM_THREAD_THREAD_COPY_HPP_

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "../functional.hpp"
#include "../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<typename T, typename U>
struct tuple_prepend;

template<typename T, typename... Us>
struct tuple_prepend<T, std::tuple<Us...>>
{
    using type = std::tuple<T, Us...>;
};

template<typename T, size_t Alignment, size_t Size>
constexpr bool can_load = Alignment % sizeof(T) == 0 && Size >= sizeof(T);

/// This struct defines the chain of types that will be tried
/// (in order) to load types with. Be sure to define these types
/// in order of descending "efficiency" (higher size & alignment first).
template<typename T>
struct next_type;

template<>
struct next_type<uint128_t>
{
    using type = uint64_t;
};

template<>
struct next_type<uint64_t>
{
    using type = uint32_t;
};

template<>
struct next_type<uint32_t>
{
    using type = uint16_t;
};

template<>
struct next_type<uint16_t>
{
    using type = uint8_t;
};

template<>
struct next_type<uint8_t>
{
    using type = void;
};

/// \brief This struct is used to build a tuple of primitive types which
/// a copy can be broken down in.
///
/// When performing a non-trivial copy, such as a non-temporal, volatile, or atomic
/// load, the compiler may not accept non-primitive types. In the case of volatile
/// copies, this is because a `const volatile T&` constructor must be defined.
/// For non-temporal loads, only trivial types are allowed.
/// This struct builds a tuple of primitive types in which such a copy can be broken
/// down in. The types in the resulting tuple are understood to be packed in memory,
/// with total alignment and size as passed to this structure. `T` is the first
/// type that we try to load with (leave this as the default); if it doesn't apply,
/// the chain defined by `next_type` is followed until one is found that applies.
///
/// \tparam Alignment Total alignment of the memory to be copied.
/// \tparam Size Total size of the memory to be copied.
template<size_t Alignment, size_t Size, typename T = uint128_t, typename Enable = void>
struct fused_copy;

template<size_t Alignment, size_t Size, typename T>
struct fused_copy<Alignment, Size, T, std::enable_if_t<(Size > 0 && can_load<T, Alignment, Size>)>>
{
    using type =
        typename tuple_prepend<T, typename fused_copy<Alignment, Size - sizeof(T), T>::type>::type;
};

template<size_t Alignment, size_t Size, typename T>
struct fused_copy<Alignment, Size, T, std::enable_if_t<(Size > 0 && !can_load<T, Alignment, Size>)>>
{
    using type = typename fused_copy<Alignment, Size, typename next_type<T>::type>::type;
};

template<size_t Alignment, typename T, typename Enable>
struct fused_copy<Alignment, 0, T, Enable>
{
    using type = std::tuple<>;
};

template<size_t Alignment, size_t Size>
using fused_copy_t = typename fused_copy<Alignment, Size>::type;

template<typename... Ts, size_t... Is, typename F>
ROCPRIM_DEVICE ROCPRIM_INLINE
void unrolled_tuple_copy_impl(std::tuple<Ts...>* __restrict__ dst,
                              const std::tuple<Ts...>* __restrict__ src,
                              F copy_op,
                              std::index_sequence<Is...>)
{
    // TODO: Replace with fold expression when we switch to C++17.
    int dummy[] = {(copy_op(std::get<Is>(*dst), std::get<Is>(*src)), 0)...};
    (void)dummy;
}

template<typename... Ts, typename F>
ROCPRIM_DEVICE ROCPRIM_INLINE
void unrolled_tuple_copy(std::tuple<Ts...>* __restrict__ dst,
                         const std::tuple<Ts...>* __restrict__ src,
                         F copy_op)
{
    unrolled_tuple_copy_impl(dst, src, copy_op, std::make_index_sequence<sizeof...(Ts)>{});
}

template<typename T,
         typename U,
         size_t Alignment = ::rocprim::min(alignof(T), alignof(U)),
         typename F>
ROCPRIM_DEVICE ROCPRIM_INLINE
void thread_fused_copy(T* __restrict__ dst, const U* __restrict__ src, F copy_op)
{
    static_assert(sizeof(T) == sizeof(U), "thread_fused_copy types must have the same size");

    using fused_copy_tuple = detail::fused_copy_t<Alignment, sizeof(T)>;
    static_assert(
        sizeof(T) == sizeof(fused_copy_tuple),
        "internal error: fused copy tuple size should be the same as input/output data type");

    unrolled_tuple_copy(reinterpret_cast<fused_copy_tuple*>(dst),
                        reinterpret_cast<const fused_copy_tuple*>(src),
                        copy_op);
}

} // namespace detail

/// \brief Thread copy with fused loads.
///
/// This function performs a "fused" copy: The copy operation
/// is explicitly unrolled into primitive copy operations
/// of the highest alignment and size possible, depending
/// on the alignment and size of the input types.
///
/// \tparam T The destination data type.
/// \tparam U The source data type.
/// \tparam Alignment If given, explicit alignment of the input types. Both `dst` and `src`
/// must be aligned to this size.
/// \param dst Target memory, where the result is written.
/// \param src Source memory, where the data is read from.
template<typename T, typename U, size_t Alignment = ::rocprim::min(alignof(T), alignof(U))>
ROCPRIM_DEVICE ROCPRIM_INLINE
void thread_fused_copy(T* __restrict__ dst, const U* __restrict__ src)
{
    detail::thread_fused_copy<T, U, Alignment>(dst,
                                               src,
                                               [](auto& dst, const auto& src) { dst = src; });
}

namespace detail
{

template<typename T, typename InputIteratorT, int... Is>
ROCPRIM_DEVICE ROCPRIM_INLINE
void unrolled_copy_impl(InputIteratorT src, T* dst, std::integer_sequence<int, Is...>)
{
    // Unroll multiple thread loads by unpacking an integer sequence
    // into a dummy array. We assign the destination values inside the
    // constructor of this dummy array.
    int dummy[] = {(dst[Is] = src[Is], 0)...};
    (void)dummy;
}

} // namespace detail

/// \brief Copy Count number of items from src to dst.
/// \tparam Count number of items to copy
/// \tparam InputIteratorT the input iterator type
/// \tparam T Type of Data to be copied to
/// \param src [in] Input iterator for data that will be copied
/// \param dst [out] The pointer the data will be copied to.
template<int Count, typename InputIteratorT, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
void unrolled_copy(InputIteratorT src, T* dst)
{
    detail::unrolled_copy_impl(src, dst, std::make_integer_sequence<int, Count>{});
}

END_ROCPRIM_NAMESPACE

#endif
