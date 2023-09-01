// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TYPE_TRAITS_HPP_
#define ROCPRIM_TYPE_TRAITS_HPP_

#include <type_traits>

// Meta configuration for rocPRIM
#include "config.hpp"
#include "types.hpp"

/// \addtogroup utilsmodule_typetraits
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Behaves like std::is_floating_point, but also includes half-precision and bfloat16-precision
/// floating point type (rocprim::half).
template<class T>
struct is_floating_point
    : std::integral_constant<
        bool,
        std::is_floating_point<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};

/// \brief Alias for std::is_integral.
template<class T>
using is_integral = std::is_integral<T>;

/// \brief Behaves like std::is_arithmetic, but also includes half-precision and bfloat16-precision
/// floating point type (\ref rocprim::half).
template<class T>
struct is_arithmetic
    : std::integral_constant<
        bool,
        std::is_arithmetic<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};

/// \brief Behaves like std::is_fundamental, but also includes half-precision and bfloat16-precision
/// floating point type (\ref rocprim::half).
template<class T>
struct is_fundamental
  : std::integral_constant<
        bool,
        std::is_fundamental<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
> {};

/// \brief Alias for std::is_unsigned.
template<class T>
using is_unsigned = std::is_unsigned<T>;

/// \brief Behaves like std::is_signed, but also includes half-precision and bfloat16-precision
/// floating point type (\ref rocprim::half).
template<class T>
struct is_signed
    : std::integral_constant<
        bool,
        std::is_signed<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};

/// \brief Behaves like std::is_scalar, but also includes half-precision and bfloat16-precision
/// floating point type (\ref rocprim::half).
template<class T>
struct is_scalar
    : std::integral_constant<
        bool,
        std::is_scalar<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};

/// \brief Behaves like std::is_compound, but also supports half-precision
/// floating point type (\ref rocprim::half). `value` for rocprim::half is `false`.
template<class T>
struct is_compound
    : std::integral_constant<
        bool,
        !is_fundamental<T>::value
    > {};

/// \brief Used to retrieve a type that can be treated as unsigned version of the template parameter.
/// \tparam T - The signed type to find an unsigned equivalent for.
/// \tparam size - the desired size (in bytes) of the unsigned type
template<typename T, int size = 0>
struct get_unsigned_bits_type
{
  typedef typename get_unsigned_bits_type<T,sizeof(T)>::unsigned_type unsigned_type; ///< Typedefed to the unsigned type.
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS // skip specialized versions
template<typename T>
struct get_unsigned_bits_type<T,1>
{
  typedef uint8_t unsigned_type;
};


template<typename T>
struct get_unsigned_bits_type<T,2>
{
  typedef uint16_t unsigned_type;
};


template<typename T>
struct get_unsigned_bits_type<T,4>
{
  typedef uint32_t unsigned_type;
};

template<typename T>
struct get_unsigned_bits_type<T,8>
{
  typedef uint64_t unsigned_type;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename T, typename UnsignedBits>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleIn(UnsignedBits key)
    -> typename std::enable_if<is_floating_point<T>::value, UnsignedBits>::type
{
  static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
  return key ^ mask;
}

template<typename T, typename UnsignedBits>
static ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleIn(UnsignedBits key)
    -> typename std::enable_if<is_unsigned<T>::value, UnsignedBits>::type
{
    return key ;
};

template<typename T, typename UnsignedBits>
static ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleIn(UnsignedBits key)
    -> typename std::enable_if<is_integral<T>::value && is_signed<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    return key ^ HIGH_BIT;
};

template<typename T, typename UnsignedBits>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleOut(UnsignedBits key)
    -> typename std::enable_if<is_floating_point<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
    return key ^ mask;
}

template<typename T, typename UnsignedBits>
static ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleOut(UnsignedBits key)
    -> typename std::enable_if<is_unsigned<T>::value, UnsignedBits>::type
{
    return key;
};

template<typename T, typename UnsignedBits>
static ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleOut(UnsignedBits key)
    -> typename std::enable_if<is_integral<T>::value && is_signed<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    return key ^ HIGH_BIT;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS


END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule_typetraits

#endif // ROCPRIM_TYPE_TRAITS_HPP_
