/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021, Advanced Micro Devices, Inc.  All rights reserved.
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

#include <limits>
#include <type_traits>

#include "../../config.hpp"

#include <rocprim/detail/various.hpp>
#include <rocprim/types/future_value.hpp>

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

BEGIN_HIPCUB_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

using NullType = ::rocprim::empty_type;

#endif

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

template <int A>
struct Int2Type
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

template <typename NumeratorT, typename DenominatorT>
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
    typedef T Type;
};

// Specializations where host C++ compilers (e.g., 32-bit Windows) may disagree
// with device C++ compilers (EDG) on types passed as template parameters through
// kernel functions

#define __HIPCUB_ALIGN_BYTES(t, b)         \
    template <> struct AlignBytes<t>    \
    { enum { ALIGN_BYTES = b }; typedef __align__(b) t Type; };

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
    typedef typename std::conditional<IsMultiple<int>::IS_MULTIPLE,
        unsigned int,
        typename std::conditional<IsMultiple<short>::IS_MULTIPLE,
            unsigned short,
            unsigned char>::type>::type         ShuffleWord;

    /// Biggest volatile word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename std::conditional<IsMultiple<long long>::IS_MULTIPLE,
        unsigned long long,
        ShuffleWord>::type                      VolatileWord;

    /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename std::conditional<IsMultiple<longlong2>::IS_MULTIPLE,
        ulonglong2,
        VolatileWord>::type                     DeviceWord;

    /// Biggest texture reference word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename std::conditional<IsMultiple<int4>::IS_MULTIPLE,
        uint4,
        typename std::conditional<IsMultiple<int2>::IS_MULTIPLE,
            uint2,
            ShuffleWord>::type>::type           TextureWord;
};


// float2 specialization workaround (for SM10-SM13)
template <>
struct UnitWord <float2>
{
    typedef int         ShuffleWord;
    typedef unsigned long long   VolatileWord;
    typedef unsigned long long   DeviceWord;
    typedef float2      TextureWord;
};

// float4 specialization workaround (for SM10-SM13)
template <>
struct UnitWord <float4>
{
    typedef int         ShuffleWord;
    typedef unsigned long long  VolatileWord;
    typedef ulonglong2          DeviceWord;
    typedef float4              TextureWord;
};


// char2 specialization workaround (for SM10-SM13)
template <>
struct UnitWord <char2>
{
    typedef unsigned short      ShuffleWord;
    typedef unsigned short      VolatileWord;
    typedef unsigned short      DeviceWord;
    typedef unsigned short      TextureWord;
};


template <typename T> struct UnitWord<volatile T> : UnitWord<T> {};
template <typename T> struct UnitWord<const T> : UnitWord<T> {};
template <typename T> struct UnitWord<const volatile T> : UnitWord<T> {};


#endif // DOXYGEN_SHOULD_SKIP_THIS




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
    typedef typename UnitWord<T>::DeviceWord DeviceWord;

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
 *
 * For example:
 *     Traits<int>::CATEGORY             // SIGNED_INTEGER
 *     Traits<NullType>::NULL_TYPE       // true
 *     Traits<uint4>::CATEGORY           // NOT_A_NUMBER
 *     Traits<uint4>::PRIMITIVE;         // false
 *
 ******************************************************************************/

 #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * \brief Basic type traits categories
 */
enum Category
{
    NOT_A_NUMBER,
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOATING_POINT
};


/**
 * \brief Basic type traits
 */
template <Category _CATEGORY, bool _PRIMITIVE, bool _NULL_TYPE, typename _UnsignedBits, typename T>
struct BaseTraits
{
    /// Category
    static const Category CATEGORY      = _CATEGORY;
    enum
    {
        PRIMITIVE       = _PRIMITIVE,
        NULL_TYPE       = _NULL_TYPE,
    };
};


/**
 * Basic type traits (unsigned primitive specialization)
 */
template <typename _UnsignedBits, typename T>
struct BaseTraits<UNSIGNED_INTEGER, true, false, _UnsignedBits, T>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = UNSIGNED_INTEGER;
    static const UnsignedBits   LOWEST_KEY  = UnsignedBits(0);
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1);

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };


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


/**
 * Basic type traits (signed primitive specialization)
 */
template <typename _UnsignedBits, typename T>
struct BaseTraits<SIGNED_INTEGER, true, false, _UnsignedBits, T>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = SIGNED_INTEGER;
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits   LOWEST_KEY  = HIGH_BIT;
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1) ^ HIGH_BIT;

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };

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

template <typename _T>
struct FpLimits;

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
template <typename _UnsignedBits, typename T>
struct BaseTraits<FLOATING_POINT, true, false, _UnsignedBits, T>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = FLOATING_POINT;
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits   LOWEST_KEY  = UnsignedBits(-1);
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1) ^ HIGH_BIT;

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
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


/**
 * \brief Numeric type traits
 */
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

template <> struct NumericTraits<float> :               BaseTraits<FLOATING_POINT, true, false, unsigned int, float> {};
template <> struct NumericTraits<double> :              BaseTraits<FLOATING_POINT, true, false, unsigned long long, double> {};
template <> struct NumericTraits<__half> :              BaseTraits<FLOATING_POINT, true, false, unsigned short, __half> {};
template <> struct NumericTraits<hip_bfloat16 > :       BaseTraits<FLOATING_POINT, true, false, unsigned short, hip_bfloat16 > {};

template <> struct NumericTraits<bool> :                BaseTraits<UNSIGNED_INTEGER, true, false, typename UnitWord<bool>::VolatileWord, bool> {};

/**
 * \brief Type traits
 */
template <typename T>
struct Traits : NumericTraits<typename std::remove_cv<T>::type> {};

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_TYPE_HPP_
