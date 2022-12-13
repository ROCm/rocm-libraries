// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_VARIOUS_HPP_
#define ROCPRIM_DETAIL_VARIOUS_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../types.hpp"
#include "../type_traits.hpp"

#include <hip/hip_runtime.h>

// Check for c++ standard library features, in a backwards compatible manner
#ifndef __has_include
    #define __has_include(x) 0
#endif

#if __has_include(<version>) // version is only mandated in c++20
    #include <version>
    #if __cpp_lib_as_const >= 201510L
        #include <utility>
    #endif
#else
    #include <utility>
#endif

// TODO: Refactor when it gets crowded

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

struct empty_storage_type
{

};

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr bool is_power_of_two(const T x)
{
    static_assert(::rocprim::is_integral<T>::value, "T must be integer type");
    return (x > 0) && ((x & (x - 1)) == 0);
}

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T next_power_of_two(const T x, const T acc = 1)
{
    static_assert(::rocprim::is_unsigned<T>::value, "T must be unsigned type");
    return acc >= x ? acc : next_power_of_two(x, 2 * acc);
}

template <
    typename T,
    typename U,
    std::enable_if_t<::rocprim::is_integral<T>::value && ::rocprim::is_unsigned<U>::value, int> = 0>
ROCPRIM_HOST_DEVICE inline constexpr auto ceiling_div(const T a, const U b)
{
    return a / b + (a % b > 0 ? 1 : 0);
}

ROCPRIM_HOST_DEVICE inline
size_t align_size(size_t size, size_t alignment = 256)
{
    return ceiling_div(size, alignment) * alignment;
}

// TOOD: Put the block algorithms with warp size variables at device side with macro.
// Temporary workaround
template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T warp_size_in_class(const T warp_size)
{
    return warp_size;
}

// Select the minimal warp size for block of size block_size, it's
// useful for blocks smaller than maximal warp size.
template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T get_min_warp_size(const T block_size, const T max_warp_size)
{
    static_assert(::rocprim::is_unsigned<T>::value, "T must be unsigned type");
    return block_size >= max_warp_size ? max_warp_size : next_power_of_two(block_size);
}

template<unsigned int WarpSize>
struct is_warpsize_shuffleable {
    static const bool value = detail::is_power_of_two(WarpSize);
};

// Selects an appropriate vector_type based on the input T and size N.
// The byte size is calculated and used to select an appropriate vector_type.
template<class T, unsigned int N>
struct match_vector_type
{
    static constexpr unsigned int size = sizeof(T) * N;
    using vector_base_type =
        typename std::conditional<
            sizeof(T) >= 4,
            int,
            typename std::conditional<
                sizeof(T) >= 2,
                short,
                char
            >::type
        >::type;

    using vector_4 = typename make_vector_type<vector_base_type, 4>::type;
    using vector_2 = typename make_vector_type<vector_base_type, 2>::type;
    using vector_1 = typename make_vector_type<vector_base_type, 1>::type;

    using type =
        typename std::conditional<
            size % sizeof(vector_4) == 0,
            vector_4,
            typename std::conditional<
                size % sizeof(vector_2) == 0,
                vector_2,
                vector_1
            >::type
        >::type;
};

// Checks if Items is odd and ensures that size of T is smaller than vector_type.
template<class T, unsigned int Items>
struct is_vectorizable : std::integral_constant<bool, (Items % 2 == 0) &&(sizeof(T) < sizeof(typename match_vector_type<T, Items>::type))> {};

// Returns the number of LDS (local data share) banks.
ROCPRIM_HOST_DEVICE
constexpr unsigned int get_lds_banks_no()
{
    // Currently all devices supported by ROCm have 32 banks (4 bytes each)
    return 32;
}

// Finds biggest fundamental type for type T that sizeof(T) is
// a multiple of that type's size.
template<class T>
struct match_fundamental_type
{
    using type =
        typename std::conditional<
            sizeof(T)%8 == 0,
            unsigned long long,
            typename std::conditional<
                sizeof(T)%4 == 0,
                unsigned int,
                typename std::conditional<
                    sizeof(T)%2 == 0,
                    unsigned short,
                    unsigned char
                >::type
            >::type
        >::type;
};

// A storage-backing wrapper that allows types with non-trivial constructors to be aliased in unions
template <typename T>
struct raw_storage
{
    // Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename detail::match_fundamental_type<T>::type device_word;

    // Backing storage
    device_word storage[sizeof(T) / sizeof(device_word)];

    // Alias
    ROCPRIM_HOST_DEVICE T& get()
    {
        return reinterpret_cast<T&>(*this);
    }

    ROCPRIM_HOST_DEVICE const T& get() const
    {
        return reinterpret_cast<const T&>(*this);
    }
};

// Checks if two iterators can possibly alias
template<class Iterator1, class Iterator2>
inline bool can_iterators_alias(Iterator1, Iterator2, const size_t size)
{
    (void)size;
    return true;
}

template<typename Value1, typename Value2>
inline bool can_iterators_alias(Value1* iter1, Value2* iter2, const size_t size)
{
    const uintptr_t start1 = reinterpret_cast<uintptr_t>(iter1);
    const uintptr_t start2 = reinterpret_cast<uintptr_t>(iter2);
    const uintptr_t end1   = reinterpret_cast<uintptr_t>(iter1 + size);
    const uintptr_t end2   = reinterpret_cast<uintptr_t>(iter2 + size);
    return start1 < end2 && start2 < end1;
}

template<class...>
using void_t = void;

template<typename T>
struct type_identity {
    using type = T;
};

template<class T, class = void>
struct extract_type_impl : type_identity<T> { };

template<class T>
struct extract_type_impl<T, void_t<typename T::type> > : extract_type_impl<typename T::type> { };

template <typename T>
using extract_type = typename extract_type_impl<T>::type;

template<bool Value, class T>
struct select_type_case
{
    static constexpr bool value = Value;
    using type = T;
};

template<class Case, class... OtherCases>
struct select_type_impl
    : std::conditional<
        Case::value,
        type_identity<extract_type<typename Case::type>>,
        select_type_impl<OtherCases...>
    >::type { };

template<class T>
struct select_type_impl<select_type_case<true, T>> : type_identity<extract_type<T>> { };

template<class T>
struct select_type_impl<select_type_case<false, T>>
{
    static_assert(
        sizeof(T) == 0,
        "Cannot select any case. "
        "The last case must have true condition or be a fallback type."
    );
};

template<class Fallback>
struct select_type_impl<Fallback> : type_identity<extract_type<Fallback>> { };

template <typename... Cases>
using select_type = typename select_type_impl<Cases...>::type;

template <bool Value>
using bool_constant = std::integral_constant<bool, Value>;

/**
 * \brief Copy data from src to dest with stream ordering and synchronization
 *
 * Equivalent to `hipStreamMemcpyAsync(...,stream)` followed by `hipStreamSynchronize(stream)`,
 * but is potentially more performant.
 *
 * \param[out] dst Destination to copy
 * \param[in] src Source of copy
 * \param[in] size_bytes Number of bytes to copy
 * \param[in] kind Memory copy type
 * \param[in] stream Stream to perform the copy. The copy is performed after all prior operations
 * on stream have been completed.
 * \return hipError_t error code
 */
inline hipError_t memcpy_and_sync(
    void* dst, const void* src, size_t size_bytes, hipMemcpyKind kind, hipStream_t stream)
{
    // hipMemcpyWithStream is only supported on rocm 3.1 and above
#if(HIP_VERSION_MAJOR == 3 && HIP_VERSION_MINOR >= 1) || HIP_VERSION_MAJOR > 3
    return hipMemcpyWithStream(dst, src, size_bytes, kind, stream);
#else
    const hipError_t result = hipMemcpyAsync(dst src, size_bytes, kind, stream);
    if(hipSuccess != result)
    {
        return result;
    }
    return hipStreamSynchronize(stream);
#endif
}

#if __cpp_lib_as_const >= 201510L
using ::std::as_const;
#else
template<typename T>
constexpr std::add_const_t<T>& as_const(T& t) noexcept
{
    return t;
}
template<typename T>
void as_const(const T&& t) = delete;
#endif

/// \brief Add `const` to the top level pointed to object type.
///
/// \tparam T type of the pointed object
/// \param ptr the pointer to make constant
/// \return ptr
///
template<typename T>
constexpr std::add_const_t<T>* as_const_ptr(T* ptr)
{
    return ptr;
}

template<class... Types, class Function, size_t... Indices>
ROCPRIM_HOST_DEVICE inline void for_each_in_tuple_impl(::rocprim::tuple<Types...>& t,
                                                       Function                    f,
                                                       ::rocprim::index_sequence<Indices...>)
{
    auto swallow = {(f(::rocprim::get<Indices>(t)), 0)...};
    (void)swallow;
}

template<class... Types, class Function>
ROCPRIM_HOST_DEVICE inline void for_each_in_tuple(::rocprim::tuple<Types...>& t, Function f)
{
    for_each_in_tuple_impl(t, f, ::rocprim::index_sequence_for<Types...>());
}

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_VARIOUS_HPP_
