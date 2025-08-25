// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <iostream>
#include <type_traits>

#include "../common.hpp"
#include "../config.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../types.hpp"

#include <hip/hip_runtime.h>

// Check for c++ standard library features, in a backwards compatible manner
#ifndef __has_include
    #define __has_include(x) 0
#endif

// Check for builtins (clang-extension) and fallback
#ifndef __has_builtin
    #define __has_builtin(X) 0
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

    using type = typename std::conditional<
        size % sizeof(vector_4) == 0,
        vector_4,
        typename std::conditional<size % sizeof(vector_2) == 0, vector_2, vector_base_type>::type>::
        type;
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

/// \brief Returns the minimum LDS size in bytes available on this device architecture.
ROCPRIM_HOST_DEVICE
constexpr unsigned int get_min_lds_size()
{
#if defined(__GFX11__) || defined(__GFX10__)
    return (1 << 17) /* 128 KiB*/;
#else
    // On host the lowest should be returned!
    return (1 << 16) /* 64 KiB */;
#endif
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

// A storage-backing wrapper that allows types with non-trivial constructors to be aliased in unions.
// Due to the reinterpret cast, it may in some cases be technically UB, but the generated code is usually
// more performant than proper code.
template<typename T>
struct raw_storage
{
    // Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    using device_word = typename detail::match_fundamental_type<T>::type;

    // Backing storage
    alignas(device_word) unsigned char storage[sizeof(T)];

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
    return hipMemcpyWithStream(dst, src, size_bytes, kind, stream);
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

/// \brief Reinterprets the pointer as another type and increments it to match the alignment of
/// the new type.
///
/// \tparam DstPtr Destination Type to align to
/// \tparam Src Type of source pointer
/// \param pointer The pointer to align
/// \return Aligned pointer
template<typename DstPtr, typename Src>
ROCPRIM_HOST_DEVICE ROCPRIM_INLINE DstPtr cast_align_up(Src* pointer)
{
    static_assert(std::is_pointer<DstPtr>::value, "DstPtr must be a pointer type");
    using Dst = std::remove_pointer_t<DstPtr>;
#if __has_builtin(__builtin_align_up)
    return reinterpret_cast<DstPtr>(__builtin_align_up(pointer, alignof(Dst)));
#else
    // https://github.com/KabukiStarship/KabukiToolkit/wiki/Fastest-Method-to-Align-Pointers
    constexpr size_t mask  = alignof(Dst) - 1;
    auto             value = reinterpret_cast<uintptr_t>(pointer);
    value += (-value) & mask;
    return reinterpret_cast<DstPtr>(value);
#endif
}

/// \brief Reinterprets the pointer as another type and decrements it to match the alignment of
/// the new type.
///
/// \tparam Ptr Destination Type to align to
/// \tparam Src Type of source pointer
/// \param pointer The pointer to align
/// \return Aligned pointer
template<typename DstPtr, typename Src>
ROCPRIM_HOST_DEVICE ROCPRIM_INLINE DstPtr cast_align_down(Src* pointer)
{
    static_assert(std::is_pointer<DstPtr>::value, "DstPtr must be a pointer type");
    using Dst = std::remove_pointer_t<DstPtr>;
#if __has_builtin(__builtin_align_down)
    return reinterpret_cast<DstPtr>(__builtin_align_down(pointer, alignof(Dst)));
#else
    // https://github.com/KabukiStarship/KabukiToolkit/wiki/Fastest-Method-to-Align-Pointers
    constexpr size_t mask  = ~(alignof(Dst) - 1);
    auto             value = reinterpret_cast<uintptr_t>(pointer);
    value &= mask;
    return reinterpret_cast<DstPtr>(value);
#endif
}

template<typename Destination, typename Source>
ROCPRIM_INLINE ROCPRIM_HOST_DEVICE
auto bit_cast(const Source& source)
{
    return ::rocprim::traits::radix_key_codec::bit_cast<Destination, Source>(source);
}

template<typename... Ts>
struct select_max_by_value;

template<typename T>
struct select_max_by_value<T>
{
    using type = T;
};

template<typename T, typename U, typename... Vs>
struct select_max_by_value<T, U, Vs...>
{
    using tail = typename select_max_by_value<U, Vs...>::type;
    using type = std::conditional_t<(T::value >= tail::value), T, tail>;
};

template<typename... Ts>
using select_max_by_value_t = typename select_max_by_value<Ts...>::type;

/// \brief Gets the maximum grid size to have all blocks active.
template<typename Kernel>
ROCPRIM_HOST
hipError_t
    grid_dim_for_max_active_blocks(int& grid_dim, int block_size, Kernel kernel, hipStream_t stream)
{
    hipDevice_t default_device;
    ROCPRIM_RETURN_ON_ERROR(hipStreamGetDevice(0, &default_device));

    hipDevice_t stream_device;
    ROCPRIM_RETURN_ON_ERROR(hipStreamGetDevice(stream, &stream_device));

    // After setting device, we can't just exit on non-success.
    hipError_t result = hipSetDevice(stream_device);

    int occupancy;
    if(result == hipSuccess)
    {
        result = hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, block_size, 0);

        // workaround for when 'hipOccupancyMaxActiveBlocksPerMultiprocessor'
        // outputs 0.
        if(occupancy == 0)
        {
            std::cerr << "Could not get max active blocks per multiprocessor! "
                         "Assuming '1'..."
                      << std::endl;
            occupancy = 1;
        }
    }

    int num_multi_processors;
    if(result == hipSuccess)
    {
        result = hipDeviceGetAttribute(&num_multi_processors,
                                       hipDeviceAttribute_t::hipDeviceAttributeMultiprocessorCount,
                                       stream_device);
        // Sanity check.
        if(num_multi_processors == 0)
        {
            result = hipErrorUnknown;
        }
    }

    if(result == hipSuccess)
    {
        grid_dim = occupancy * num_multi_processors;
    }

    // Always attempt to restore to default device.
    hipError_t set_result = hipSetDevice(default_device);
    ROCPRIM_RETURN_ON_ERROR(result);

    return set_result;
}

/// \brief Checks if the device on stream supports cooperative groups
inline hipError_t supports_cooperative_groups(bool& has_support, hipStream_t stream)
{
    hipDevice_t stream_device;
    hipError_t  result = hipStreamGetDevice(stream, &stream_device);
    if(result != hipSuccess)
    {
        return result;
    }

    int value;
    result = hipDeviceGetAttribute(&value,
                                   hipDeviceAttribute_t::hipDeviceAttributeCooperativeLaunch,
                                   stream_device);
    if(result != hipSuccess)
    {
        return result;
    }

    has_support = value != 0;

    return hipSuccess;
}

/// Computes the time difference between now and the passed time reference.
/// Returns the time difference in milliseconds and updates the time reference
/// with now.
inline float update_time_point(std::chrono::high_resolution_clock::time_point& t_time)
{
    std::chrono::high_resolution_clock::time_point t_stop
        = std::chrono::high_resolution_clock::now();

    float delta_time
        = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(t_stop - t_time)
              .count();
    t_time = t_stop;

    return delta_time;
}

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_VARIOUS_HPP_
