/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_UTIL_TYPE_HPP_
#define HIPCUB_ROCPRIM_UTIL_TYPE_HPP_

#include "../../config.hpp"
#include "../../util_deprecated.hpp"

#include <rocprim/detail/various.hpp> // IWYU pragma: export
#include <rocprim/type_traits.hpp> // IWYU pragma: export
#include <rocprim/types/future_value.hpp> // IWYU pragma: export

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#include <limits>
#include <type_traits>

BEGIN_HIPCUB_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

using NullType = ::rocprim::empty_type;

#endif

#ifndef HIPCUB_IS_INT128_ENABLED
    #define HIPCUB_IS_INT128_ENABLED 1
#endif // !defined(HIPCUB_IS_INT128_ENABLED)

template<bool B, typename T, typename F> struct
[[deprecated("[Since 1.16] If is deprecated use std::conditional instead.")]] If
{
    using Type = typename std::conditional<B, T, F>::type;
};

template<typename T> struct
[[deprecated("[Since 1.16] IsPointer is deprecated use std::is_pointer instead.")]] IsPointer
{
    static constexpr bool VALUE = std::is_pointer<T>::value;
};

template<typename T> struct
[[deprecated("[Since 1.16] IsVolatile is deprecated use std::is_volatile instead.")]] IsVolatile
{
    static constexpr bool VALUE = std::is_volatile<T>::value;
};

template<typename T> struct 
[[deprecated("[Since 1.16] RemoveQualifiers is deprecated use std::remove_cv instead.")]] RemoveQualifiers
{
    using Type = typename std::remove_cv<T>::type;
};

template<int N>
struct PowerOfTwo
{
    static constexpr bool VALUE = ::rocprim::detail::is_power_of_two(N);
};

namespace detail
{

template<int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2Impl
{
    static constexpr int VALUE = Log2Impl<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE;
};

template<int N, int COUNT>
struct Log2Impl<N, 0, COUNT>
{
    static constexpr int VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1;
};

} // end of detail namespace

template<int N>
struct Log2
{
    static_assert(N != 0, "The logarithm of zero is undefined");
    static constexpr int VALUE = detail::Log2Impl<N>::VALUE;
};

template<typename T>
struct DoubleBuffer
{
    T * d_buffers[2];

    int selector;

    HIPCUB_HOST_DEVICE inline
    DoubleBuffer()
    {
        selector = 0;
        d_buffers[0] = nullptr;
        d_buffers[1] = nullptr;
    }

    HIPCUB_HOST_DEVICE inline
    DoubleBuffer(T * d_current, T * d_alternate)
    {
        selector = 0;
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    HIPCUB_HOST_DEVICE inline
    T * Current()
    {
        return d_buffers[selector];
    }

    HIPCUB_HOST_DEVICE inline
    T * Alternate()
    {
        return d_buffers[selector ^ 1];
    }
};

template<int A>
struct HIPCUB_DEPRECATED_BECAUSE("Use ::std::integral_constant instead") Int2Type
{
    enum {VALUE = A};
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

template<
    class Key,
    class Value
>
using KeyValuePair = ::rocprim::key_value_pair<Key, Value>;

#endif

template <typename T, typename Iter = T*>
using FutureValue = ::rocprim::future_value<T, Iter>;

namespace detail
{

template<typename T>
inline
::rocprim::double_buffer<T> to_double_buffer(DoubleBuffer<T>& source)
{
    return ::rocprim::double_buffer<T>(source.Current(), source.Alternate());
}

template<typename T>
inline
void update_double_buffer(DoubleBuffer<T>& target, ::rocprim::double_buffer<T>& source)
{
    if(target.Current() != source.current())
    {
        target.selector ^= 1;
    }
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

template <typename T>
using is_integral_or_enum =
  std::integral_constant<bool, std::is_integral<T>::value || std::is_enum<T>::value>;

#endif

}

// CUB deprecated this API, and suggests to use `::cuda::ceil_div` instead,
// which is implemented in file `libcudacxx/include/cuda/__cmath/ceil_div.h`.
template<typename NumeratorT, typename DenominatorT>
HIPCUB_DEPRECATED_BECAUSE("Use hip::ceil_div instead from 'libhipcxx'")
HIPCUB_HOST_DEVICE __forceinline__ constexpr NumeratorT
    DivideAndRoundUp(NumeratorT n, DenominatorT d)
{
  static_assert(hipcub::detail::is_integral_or_enum<NumeratorT>::value &&
                hipcub::detail::is_integral_or_enum<DenominatorT>::value,
                "DivideAndRoundUp is only intended for integral types.");

  // Static cast to undo integral promotion.
  return static_cast<NumeratorT>(n / d + (n % d != 0 ? 1 : 0));
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/******************************************************************************
 * Size and alignment
 ******************************************************************************/

/// Structure alignment
template <typename T>
struct AlignBytes
{
    struct Pad
    {
        T       val;
        char    byte;
    };

    enum
    {
        /// The "true CUDA" alignment of T in bytes
        ALIGN_BYTES = sizeof(Pad) - sizeof(T)
    };

    /// The "truly aligned" type
    using Type = T;
};

// Specializations where host C++ compilers (e.g., 32-bit Windows) may disagree
// with device C++ compilers (EDG) on types passed as template parameters through
// kernel functions

    #define __HIPCUB_ALIGN_BYTES(t, b)                  \
        template<>                                      \
        struct AlignBytes<t>                            \
        {                                               \
            enum                                        \
            {                                           \
                ALIGN_BYTES = b                         \
            };                                          \
            using Type = __attribute__((aligned(b))) t; \
        };

__HIPCUB_ALIGN_BYTES(short4, 8)
__HIPCUB_ALIGN_BYTES(ushort4, 8)
__HIPCUB_ALIGN_BYTES(int2, 8)
__HIPCUB_ALIGN_BYTES(uint2, 8)
__HIPCUB_ALIGN_BYTES(long long, 8)
__HIPCUB_ALIGN_BYTES(unsigned long long, 8)
__HIPCUB_ALIGN_BYTES(float2, 8)
__HIPCUB_ALIGN_BYTES(double, 8)
#ifdef _WIN32
    __HIPCUB_ALIGN_BYTES(long2, 8)
    __HIPCUB_ALIGN_BYTES(ulong2, 8)
#else
    __HIPCUB_ALIGN_BYTES(long2, 16)
    __HIPCUB_ALIGN_BYTES(ulong2, 16)
#endif
__HIPCUB_ALIGN_BYTES(int4, 16)
__HIPCUB_ALIGN_BYTES(uint4, 16)
__HIPCUB_ALIGN_BYTES(float4, 16)
__HIPCUB_ALIGN_BYTES(long4, 16)
__HIPCUB_ALIGN_BYTES(ulong4, 16)
__HIPCUB_ALIGN_BYTES(longlong2, 16)
__HIPCUB_ALIGN_BYTES(ulonglong2, 16)
__HIPCUB_ALIGN_BYTES(double2, 16)
__HIPCUB_ALIGN_BYTES(longlong4, 16)
__HIPCUB_ALIGN_BYTES(ulonglong4, 16)
__HIPCUB_ALIGN_BYTES(double4, 16)

template <typename T> struct AlignBytes<volatile T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const volatile T> : AlignBytes<T> {};


/// Unit-words of data movement
template <typename T>
struct UnitWord
{
    enum {
        ALIGN_BYTES = AlignBytes<T>::ALIGN_BYTES
    };

    template <typename Unit>
    struct IsMultiple
    {
        enum {
            UNIT_ALIGN_BYTES    = AlignBytes<Unit>::ALIGN_BYTES,
            IS_MULTIPLE         = (sizeof(T) % sizeof(Unit) == 0) && (int(ALIGN_BYTES) % int(UNIT_ALIGN_BYTES) == 0)
        };
    };

    /// Biggest shuffle word that T is a whole multiple of and is not larger than the alignment of T
    using ShuffleWord =
        typename std::conditional<IsMultiple<int>::IS_MULTIPLE,
                                  unsigned int,
                                  typename std::conditional<IsMultiple<short>::IS_MULTIPLE,
                                                            unsigned short,
                                                            unsigned char>::type>::type;

    /// Biggest volatile word that T is a whole multiple of and is not larger than the alignment of T
    using VolatileWord = typename std::
        conditional<IsMultiple<long long>::IS_MULTIPLE, unsigned long long, ShuffleWord>::type;

    /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    using DeviceWord = typename std::
        conditional<IsMultiple<longlong2>::IS_MULTIPLE, ulonglong2, VolatileWord>::type;

    /// Biggest texture reference word that T is a whole multiple of and is not larger than the alignment of T
    using TextureWord = typename std::conditional<
        IsMultiple<int4>::IS_MULTIPLE,
        uint4,
        typename std::conditional<IsMultiple<int2>::IS_MULTIPLE, uint2, ShuffleWord>::type>::type;
};


// float2 specialization workaround (for SM10-SM13)
template <>
struct UnitWord <float2>
{
    using ShuffleWord  = int;
    using VolatileWord = unsigned long long;
    using DeviceWord   = unsigned long long;
    using TextureWord  = float2;
};

// float4 specialization workaround (for SM10-SM13)
template <>
struct UnitWord <float4>
{
    using ShuffleWord  = int;
    using VolatileWord = unsigned long long;
    using DeviceWord   = ulonglong2;
    using TextureWord  = float4;
};


// char2 specialization workaround (for SM10-SM13)
template <>
struct UnitWord <char2>
{
    using ShuffleWord  = unsigned short;
    using VolatileWord = unsigned short;
    using DeviceWord   = unsigned short;
    using TextureWord  = unsigned short;
};


template <typename T> struct UnitWord<volatile T> : UnitWord<T> {};
template <typename T> struct UnitWord<const T> : UnitWord<T> {};
template <typename T> struct UnitWord<const volatile T> : UnitWord<T> {};

#endif // DOXYGEN_SHOULD_SKIP_THIS

/******************************************************************************
 * Vector type inference utilities.
 ******************************************************************************/

template<typename T, int vec_elements>
struct GenericCubVector
{
    static_assert(!sizeof(T), "CubVector can only have 1-4 elements");
};

enum
{
    /// The maximum number of elements in HIP vector types
    MAX_VEC_ELEMENTS = 4,
};

template<typename T>
struct GenericCubVector<T, 1>
{
    T x;

    using BaseType = T;
    using Type     = GenericCubVector<T, 1>;
};

template<typename T>
struct GenericCubVector<T, 2>
{
    T x;
    T y;

    using BaseType = T;
    using Type     = GenericCubVector<T, 2>;
};

template<typename T>
struct GenericCubVector<T, 3>
{
    T x;
    T y;
    T z;

    using BaseType = T;
    using Type     = GenericCubVector<T, 3>;
};

template<typename T>
struct GenericCubVector<T, 4>
{
    T x;
    T y;
    T z;
    T w;

    using BaseType = T;
    using Type     = GenericCubVector<T, 4>;
};

template<typename T, int vec_elements>
struct CubVectorType
{
    // Fallback on GenericCubVector
    using Type = GenericCubVector<T, vec_elements>;
};

template<typename T, int vec_elements>
using CubVector = typename CubVectorType<T, vec_elements>::Type;

#define HIPCUB_DEFINE_VECTOR_TYPE(base_type, vec_type)        \
    template<int vec_elements>                                \
    struct CubVectorType<base_type, vec_elements>             \
    {                                                         \
        using Type = HIP_vector_type<vec_type, vec_elements>; \
    };

HIPCUB_DEFINE_VECTOR_TYPE(char, char)
HIPCUB_DEFINE_VECTOR_TYPE(unsigned char, unsigned char)
HIPCUB_DEFINE_VECTOR_TYPE(short, short)
HIPCUB_DEFINE_VECTOR_TYPE(ushort, ushort)
HIPCUB_DEFINE_VECTOR_TYPE(int, int)
HIPCUB_DEFINE_VECTOR_TYPE(uint, uint)
HIPCUB_DEFINE_VECTOR_TYPE(long, long)
HIPCUB_DEFINE_VECTOR_TYPE(unsigned long, unsigned long)
HIPCUB_DEFINE_VECTOR_TYPE(long long, long long)
HIPCUB_DEFINE_VECTOR_TYPE(unsigned long long, unsigned long long)
HIPCUB_DEFINE_VECTOR_TYPE(float, float)
HIPCUB_DEFINE_VECTOR_TYPE(double, double)
HIPCUB_DEFINE_VECTOR_TYPE(bool, unsigned char)

#undef HIPCUB_DEFINE_VECTOR_TYPE

/******************************************************************************
 * Wrapper types
 ******************************************************************************/

/**
 * \brief A storage-backing wrapper that allows types with non-trivial constructors to be aliased in unions
 */
template <typename T>
struct Uninitialized
{
    /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    using DeviceWord = typename UnitWord<T>::DeviceWord;

    static constexpr std::size_t DATA_SIZE = sizeof(T);
    static constexpr std::size_t WORD_SIZE = sizeof(DeviceWord);
    static constexpr std::size_t WORDS = DATA_SIZE / WORD_SIZE;

    /// Backing storage
    DeviceWord storage[WORDS];

    /// Alias
    HIPCUB_HOST_DEVICE __forceinline__ T& Alias()
    {
        return reinterpret_cast<T&>(*this);
    }
};

/******************************************************************************
 * Simple type traits utilities.
 ******************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

/**
 * \brief Basic type traits categories.
 * This enum is deprecated, please use <rocprim/type_traits> instead. Or if you have
 * libhipcxx, please use the type_traits system in libhipcxx.
 */
enum HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.") Category
{
    NOT_A_NUMBER,
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOATING_POINT
};

/**
 * \brief Basic type traits
 */
HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
template<Category _CATEGORY,
         bool     _PRIMITIVE,
         bool     _nullptr_TYPE,
         typename _UnsignedBits,
         typename T>
struct BaseTraits
{
    /// Category
    HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.")
    static constexpr Category CATEGORY = _CATEGORY;
    enum
    {
        PRIMITIVE HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.") = _PRIMITIVE,
        nullptr_TYPE                                                              = _nullptr_TYPE,
    };
};
HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

/**
 * Basic type traits (unsigned primitive specialization)
 */
HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
template <typename _UnsignedBits, typename T>
struct BaseTraits<UNSIGNED_INTEGER, true, false, _UnsignedBits, T>
{
    using UnsignedBits = _UnsignedBits;

    HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.")
    static constexpr Category     CATEGORY   = UNSIGNED_INTEGER;
    static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(0);
    static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1);

    enum
    {
        PRIMITIVE HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.") = true,
        nullptr_TYPE                                                              = false,
    };

    using key_codec = decltype(::rocprim::traits::get<T>().template radix_key_codec<false>());

    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key;
    }

    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key;
    }

    static HIPCUB_HOST_DEVICE __forceinline__ T Max()
    {
        UnsignedBits retval_bits = MAX_KEY;
        T retval;
        memcpy(&retval, &retval_bits, sizeof(T));
        return retval;
    }

    static HIPCUB_HOST_DEVICE __forceinline__ T Lowest()
    {
        UnsignedBits retval_bits = LOWEST_KEY;
        T retval;
        memcpy(&retval, &retval_bits, sizeof(T));
        return retval;
    }
};
HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

/**
 * Basic type traits (signed primitive specialization)
 */
HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
template <typename _UnsignedBits, typename T>
struct BaseTraits<SIGNED_INTEGER, true, false, _UnsignedBits, T>
{
    using UnsignedBits = _UnsignedBits;

    HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.")
    static constexpr Category     CATEGORY   = SIGNED_INTEGER;
    static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static constexpr UnsignedBits LOWEST_KEY = HIGH_BIT;
    static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

    enum
    {
        PRIMITIVE HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.") = true,
        nullptr_TYPE                                                              = false,
    };

    using key_codec = decltype(::rocprim::traits::get<T>().template radix_key_codec<false>());

    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static HIPCUB_HOST_DEVICE __forceinline__ T Max()
    {
        UnsignedBits retval = MAX_KEY;
        return reinterpret_cast<T&>(retval);
    }

    static HIPCUB_HOST_DEVICE __forceinline__ T Lowest()
    {
        UnsignedBits retval = LOWEST_KEY;
        return reinterpret_cast<T&>(retval);
    }
};
HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

// This API needs to be deprecated once libhipcxx is available.
template <typename _T>
struct FpLimits;

// This API needs to be deprecated once libhipcxx is available.
template <>
struct FpLimits<float>
{
    static HIPCUB_HOST_DEVICE __forceinline__ float Max() {
        return std::numeric_limits<float>::max();
    }

    static HIPCUB_HOST_DEVICE __forceinline__ float Lowest() {
        return std::numeric_limits<float>::max() * float(-1);
    }
};

// This API needs to be deprecated once libhipcxx is available.
template <>
struct FpLimits<double>
{
    static HIPCUB_HOST_DEVICE __forceinline__ double Max() {
        return std::numeric_limits<double>::max();
    }

    static HIPCUB_HOST_DEVICE __forceinline__ double Lowest() {
        return std::numeric_limits<double>::max()  * double(-1);
    }
};

// This API needs to be deprecated once libhipcxx is available.
template <>
struct FpLimits<__half>
{
    static HIPCUB_HOST_DEVICE __forceinline__ __half Max() {
        unsigned short max_word = 0x7BFF;
        return reinterpret_cast<__half&>(max_word);
    }

    static HIPCUB_HOST_DEVICE __forceinline__ __half Lowest() {
        unsigned short lowest_word = 0xFBFF;
        return reinterpret_cast<__half&>(lowest_word);
    }
};

// This API needs to be deprecated once libhipcxx is available.
template <>
struct FpLimits<hip_bfloat16>
{
    static HIPCUB_HOST_DEVICE __forceinline__ hip_bfloat16  Max() {
        unsigned short max_word = 0x7F7F;
        return reinterpret_cast<hip_bfloat16 &>(max_word);
    }

    static HIPCUB_HOST_DEVICE __forceinline__ hip_bfloat16  Lowest() {
        unsigned short lowest_word = 0xFF7F;
        return reinterpret_cast<hip_bfloat16 &>(lowest_word);
    }
};

/**
 * Basic type traits (fp primitive specialization)
 */
HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
template <typename _UnsignedBits, typename T>
struct BaseTraits<FLOATING_POINT, true, false, _UnsignedBits, T>
{
    using UnsignedBits = _UnsignedBits;

    HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.")
    static constexpr Category     CATEGORY   = FLOATING_POINT;
    static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(-1);
    static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

    using key_codec = decltype(::rocprim::traits::get<T>().template radix_key_codec<false>());

    enum
    {
        PRIMITIVE HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.") = true,
        nullptr_TYPE                                                              = false,
    };

    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
        return key ^ mask;
    };

    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
        return key ^ mask;
    };

    static HIPCUB_HOST_DEVICE __forceinline__ T Max() {
        return FpLimits<T>::Max();
    }

    static HIPCUB_HOST_DEVICE __forceinline__ T Lowest() {
        return FpLimits<T>::Lowest();
    }
};
HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

/**
 * \brief Numeric type traits
 */
HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
template <typename T> struct NumericTraits :            BaseTraits<NOT_A_NUMBER, false, false, T, T> {};

template <> struct NumericTraits<NullType> :            BaseTraits<NOT_A_NUMBER, false, true, NullType, NullType> {};

template <> struct NumericTraits<char> :                BaseTraits<(std::numeric_limits<char>::is_signed) ? SIGNED_INTEGER : UNSIGNED_INTEGER, true, false, unsigned char, char> {};
template <> struct NumericTraits<signed char> :         BaseTraits<SIGNED_INTEGER, true, false, unsigned char, signed char> {};
template <> struct NumericTraits<short> :               BaseTraits<SIGNED_INTEGER, true, false, unsigned short, short> {};
template <> struct NumericTraits<int> :                 BaseTraits<SIGNED_INTEGER, true, false, unsigned int, int> {};
template <> struct NumericTraits<long> :                BaseTraits<SIGNED_INTEGER, true, false, unsigned long, long> {};
template <> struct NumericTraits<long long> :           BaseTraits<SIGNED_INTEGER, true, false, unsigned long long, long long> {};

template <> struct NumericTraits<unsigned char> :       BaseTraits<UNSIGNED_INTEGER, true, false, unsigned char, unsigned char> {};
template <> struct NumericTraits<unsigned short> :      BaseTraits<UNSIGNED_INTEGER, true, false, unsigned short, unsigned short> {};
template <> struct NumericTraits<unsigned int> :        BaseTraits<UNSIGNED_INTEGER, true, false, unsigned int, unsigned int> {};
template <> struct NumericTraits<unsigned long> :       BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long, unsigned long> {};
template <> struct NumericTraits<unsigned long long> :  BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long long, unsigned long long> {};

    #if HIPCUB_IS_INT128_ENABLED
template<>
struct NumericTraits<__uint128_t>
{
    using T            = __uint128_t;
    using UnsignedBits = __uint128_t;

    static constexpr Category     CATEGORY   = UNSIGNED_INTEGER;
    static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(0);
    static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1);

    HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.")
    static constexpr bool PRIMITIVE    = false;
    static constexpr bool nullptr_TYPE = false;

    using key_codec = decltype(::rocprim::traits::get<T>().template radix_key_codec<false>());

    static __host__ __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key;
    }

    static __host__ __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key;
    }

    static __host__ __device__ __forceinline__ T Max()
    {
        return MAX_KEY;
    }

    static __host__ __device__ __forceinline__ T Lowest()
    {
        return LOWEST_KEY;
    }
};

template<>
struct NumericTraits<__int128_t>
{
    using T            = __int128_t;
    using UnsignedBits = __uint128_t;

    static constexpr Category     CATEGORY   = SIGNED_INTEGER;
    static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static constexpr UnsignedBits LOWEST_KEY = HIGH_BIT;
    static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

    HIPCUB_DEPRECATED_BECAUSE("Use <rocprim/type_traits> instead.")
    static constexpr bool PRIMITIVE = false;
    static constexpr bool nullptr_TYPE = false;

    using key_codec = decltype(::rocprim::traits::get<T>().template radix_key_codec<false>());

    static __host__ __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static __host__ __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static __host__ __device__ __forceinline__ T Max()
    {
        UnsignedBits retval = MAX_KEY;
        return reinterpret_cast<T&>(retval);
    }

    static __host__ __device__ __forceinline__ T Lowest()
    {
        UnsignedBits retval = LOWEST_KEY;
        return reinterpret_cast<T&>(retval);
    }
};
    #endif

template <> struct NumericTraits<float> :               BaseTraits<FLOATING_POINT, true, false, unsigned int, float> {};
template <> struct NumericTraits<double> :              BaseTraits<FLOATING_POINT, true, false, unsigned long long, double> {};
template <> struct NumericTraits<__half> :              BaseTraits<FLOATING_POINT, true, false, unsigned short, __half> {};
template <> struct NumericTraits<hip_bfloat16 > :       BaseTraits<FLOATING_POINT, true, false, unsigned short, hip_bfloat16 > {};

template <> struct NumericTraits<bool> :                BaseTraits<UNSIGNED_INTEGER, true, false, typename UnitWord<bool>::VolatileWord, bool> {};
HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

/**
 * \brief Type traits
 */
template<typename T>
struct Traits : NumericTraits<typename std::remove_cv<T>::type>
{};

namespace detail
{
// __uint128_t and __int128_t are not primitive

HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
template<typename T>
struct is_primitive : ::std::bool_constant<Traits<T>::PRIMITIVE>
{};

template<typename T>
inline constexpr bool is_primitive_v = is_primitive<T>::value;

HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP

} // namespace detail

/**
 * \brief Common type of zero types.
 */
template<class...>
struct common_type
{};

/**
 * \brief Common type of a single type.
 */
template<class T>
struct common_type<T> : common_type<T, T>
{};

// Common type of a pair of types.
namespace detail
{

// Determines if type is half or bfloat16 (extended fp).
template<class T>
struct is_extended_fp
    : std::integral_constant<
          bool,
          std::is_same<__half, typename std::remove_cv<T>::type>::value
              || std::is_same<hip_bfloat16, typename std::remove_cv<T>::type>::value>
{};

// Gets "raw" type: drops reference and const qualifier.
template<typename T>
struct remove_cvref
{
    using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};
template<typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

template<template<typename...> class MFn, bool condition, typename T>
using apply_if_t = std::conditional_t<condition, MFn<T>, T>;

template<typename From, typename To>
using copy_cv_t
    = apply_if_t<std::add_volatile_t,
                 std::is_volatile_v<From>,
                 apply_if_t<std::add_const_t, std::is_const_v<From>, std::remove_cv_t<To>>>;

template<typename From, typename To>
using copy_ref_t = apply_if_t<std::add_rvalue_reference_t,
                              std::is_rvalue_reference_v<From>,
                              apply_if_t<std::add_lvalue_reference_t,
                                         std::is_lvalue_reference_v<From>,
                                         std::remove_reference_t<To>>>;

template<typename From, typename To>
using copy_cvref_t = copy_ref_t<From, copy_cv_t<std::remove_reference_t<From>, remove_cvref_t<To>>>;

// Captures non extended fp types.
template<class T1, class T2, class = void>
struct common_type_extended_fp
{
    using type = typename std::common_type<T1, T2>::type;
};

// Captures first type arithmetic, second one extended FP.
template<class T1, class T2>
struct common_type_extended_fp<
    T1,
    T2,
    typename std::enable_if_t<std::is_arithmetic<remove_cvref_t<T1>>::value
                              && is_extended_fp<remove_cvref_t<T2>>::value>>
{
    using type = typename std::common_type<T1, copy_cvref_t<T2, float>>::type;
};

// Captures first type extended FP, second one arithmetic.
template<class T1, class T2>
struct common_type_extended_fp<
    T1,
    T2,
    typename std::enable_if_t<is_extended_fp<remove_cvref_t<T1>>::value
                              && std::is_arithmetic<remove_cvref_t<T2>>::value>>
{
    using type = typename std::common_type<copy_cvref_t<T1, float>, T2>::type;
};

template<class...>
using void_t = void;

template<int Value>
using int_constant_t = ::std::integral_constant<int, Value>;

// Common type of more than two types.

template<class AlwaysVoid, class T1, class T2, class... Rest>
struct common_type_multi_impl
{};

template<class T1, class T2, class... Rest>
struct common_type_multi_impl<void_t<typename common_type<T1, T2>::type>, T1, T2, Rest...>
    : common_type<typename common_type<T1, T2>::type, Rest...>
{};
} // namespace detail

/**
 * \brief Common type of a pair of types
 */
template<class T1, class T2>
struct common_type<T1, T2> : detail::common_type_extended_fp<T1, T2>
{};

/**
 * \brief Common type of more than two types
 */
template<class T1, class T2, class... Rest>
struct common_type<T1, T2, Rest...> : detail::common_type_multi_impl<void, T1, T2, Rest...>
{};

/**
 * \brief Common type helper
 */
template<class... Ts>
using common_type_t = typename common_type<Ts...>::type;

#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace detail
{

template<typename T, typename = void>
struct is_fixed_size_random_access_range : ::std::false_type
{};

template<typename T, size_t N>
struct is_fixed_size_random_access_range<T[N], void> : ::std::true_type
{};

template<typename T>
struct is_fixed_size_random_access_range<T, void_t<decltype(std::declval<T&>()[0])>>
    : ::std::true_type
{};

// TODO: for is_fixed_size_random_access_range we also need to support array span and extents after we can use libhipcxx.
template<typename T>
using is_fixed_size_random_access_range_t = typename is_fixed_size_random_access_range<T>::type;

template<typename T, typename = void>
struct static_size
{
    static_assert(false, "static_size not supported for this type");
};

template<typename T, size_t N>
struct static_size<T[N], void> : ::std::integral_constant<int, N>
{};

template<typename T>
[[nodiscard]]
__host__ __device__ __forceinline__
constexpr size_t static_size_v()
{
    return static_size<T>::value;
}

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_TYPE_HPP_
