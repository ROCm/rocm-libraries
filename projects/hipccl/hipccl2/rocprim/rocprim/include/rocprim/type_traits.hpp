// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "config.hpp"
#include "functional.hpp"
#include "types.hpp"

#include "types/tuple.hpp"

#include <functional>
#include <type_traits>
#include <utility>

/// \addtogroup utilsmodule_typetraits
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Extension of `std::is_floating_point`, which includes support for \ref rocprim::half and \ref rocprim::bfloat16.
template<class T>
struct is_floating_point
    : std::integral_constant<
        bool,
        std::is_floating_point<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};

/// \brief Extension of `std::is_integral`, which includes support for 128-bit integers.
template<class T>
struct is_integral
    : std::integral_constant<
          bool,
          std::is_integral<T>::value
              || std::is_same<__int128_t, typename std::remove_cv<T>::type>::value
              || std::is_same<__uint128_t, typename std::remove_cv<T>::type>::value>
{};

/// \brief Extension of `std::is_arithmetic`, which includes support for \ref rocprim::half , \ref rocprim::bfloat16 and 128-bit integers.
template<class T>
struct is_arithmetic
    : std::integral_constant<
          bool,
          std::is_arithmetic<T>::value
              || std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value
              || std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
              || std::is_same<__int128_t, typename std::remove_cv<T>::type>::value
              || std::is_same<__uint128_t, typename std::remove_cv<T>::type>::value>
{};

/// \brief Extension of `std::is_fundamental`, which includes support for \ref rocprim::half , \ref rocprim::bfloat16 and 128-bit integers.
template<class T>
struct is_fundamental
    : std::integral_constant<
          bool,
          std::is_fundamental<T>::value
              || std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value
              || std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
              || std::is_same<__int128_t, typename std::remove_cv<T>::type>::value
              || std::is_same<__uint128_t, typename std::remove_cv<T>::type>::value>
{};

/// \brief Extension of `std::is_unsigned`, which includes support for 128-bit integers.
template<class T>
struct is_unsigned
    : std::integral_constant<
          bool,
          std::is_unsigned<T>::value
              || std::is_same<__uint128_t, typename std::remove_cv<T>::type>::value>
{};

/// \brief Extension of `std::is_signed`, which includes support for \ref rocprim::half , \ref rocprim::bfloat16 and 128-bit integers.
template<class T>
struct is_signed
    : std::integral_constant<
          bool,
          std::is_signed<T>::value
              || std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value
              || std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
              || std::is_same<__int128_t, typename std::remove_cv<T>::type>::value>
{};

/// \brief Extension of `std::is_scalar`, which includes support for \ref rocprim::half , \ref rocprim::bfloat16 and 128-bit integers.
template<class T>
struct is_scalar
    : std::integral_constant<
          bool,
          std::is_scalar<T>::value
              || std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value
              || std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
              || std::is_same<__int128_t, typename std::remove_cv<T>::type>::value
              || std::is_same<__uint128_t, typename std::remove_cv<T>::type>::value>
{};

/// \brief Extension of `std::make_unsigned`, which includes support for 128-bit integers.
template<class T>
struct make_unsigned : std::make_unsigned<T>
{};

#ifndef DOXYGEN_SHOULD_SKIP_THIS // skip specialized versions
template<>
struct make_unsigned<__int128_t>
{
    using type = __uint128_t;
};

template<>
struct make_unsigned<__uint128_t>
{
    using type = __uint128_t;
};
#endif

static_assert(std::is_same<make_unsigned<__int128_t>::type, __uint128_t>::value,
              "'__int128_t' needs to implement 'make_unsigned' trait.");

/// \brief Extension of `std::is_compound`, which includes support for \ref rocprim::half , \ref rocprim::bfloat16 and 128-bit integers.
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

template<typename T>
struct get_unsigned_bits_type<T, 16>
{
    typedef __uint128_t unsigned_type;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename T, typename UnsignedBits>
[[deprecated("TwiddleIn is deprecated."
             "Use radix_key_codec instead.")]] ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleIn(UnsignedBits key) ->
    typename std::enable_if<is_floating_point<T>::value, UnsignedBits>::type
{
  static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
  return key ^ mask;
}

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleIn is deprecated."
             "Use radix_key_codec instead.")]] static ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleIn(UnsignedBits key) ->
    typename std::enable_if<is_unsigned<T>::value, UnsignedBits>::type
{
    return key ;
};

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleIn is deprecated."
             "Use radix_key_codec instead.")]] static ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleIn(UnsignedBits key) ->
    typename std::enable_if<is_integral<T>::value && is_signed<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    return key ^ HIGH_BIT;
};

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleOut is deprecated."
             "Use radix_key_codec instead.")]] ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleOut(UnsignedBits key) ->
    typename std::enable_if<is_floating_point<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
    return key ^ mask;
}

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleOut is deprecated."
             "Use radix_key_codec instead.")]] static ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleOut(UnsignedBits key) ->
    typename std::enable_if<is_unsigned<T>::value, UnsignedBits>::type
{
    return key;
};

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleOut is deprecated."
             "Use radix_key_codec instead.")]] static ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleOut(UnsignedBits key) ->
    typename std::enable_if<is_integral<T>::value && is_signed<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    return key ^ HIGH_BIT;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace detail
{

// invoke_result is based on std::invoke_result.
// The main difference is using ROCPRIM_HOST_DEVICE, this allows to
// use invoke_result with device-only lambdas/functors in host-only functions
// on HIP-clang.

template<class T>
struct is_reference_wrapper : std::false_type
{};
template<class U>
struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type
{};

template<class T>
struct invoke_impl
{
    template<class F, class... Args>
    ROCPRIM_HOST_DEVICE static auto call(F&& f, Args&&... args)
        -> decltype(std::forward<F>(f)(std::forward<Args>(args)...));
};

template<class B, class MT>
struct invoke_impl<MT B::*>
{
    template<class T,
             class Td = typename std::decay<T>::type,
             class    = typename std::enable_if<std::is_base_of<B, Td>::value>::type>
    ROCPRIM_HOST_DEVICE static auto get(T&& t) -> T&&;

    template<class T,
             class Td = typename std::decay<T>::type,
             class    = typename std::enable_if<is_reference_wrapper<Td>::value>::type>
    ROCPRIM_HOST_DEVICE static auto get(T&& t) -> decltype(t.get());

    template<class T,
             class Td = typename std::decay<T>::type,
             class    = typename std::enable_if<!std::is_base_of<B, Td>::value>::type,
             class    = typename std::enable_if<!is_reference_wrapper<Td>::value>::type>
    ROCPRIM_HOST_DEVICE static auto get(T&& t) -> decltype(*std::forward<T>(t));

    template<class T,
             class... Args,
             class MT1,
             class = typename std::enable_if<std::is_function<MT1>::value>::type>
    ROCPRIM_HOST_DEVICE static auto call(MT1 B::*pmf, T&& t, Args&&... args)
        -> decltype((invoke_impl::get(std::forward<T>(t)).*pmf)(std::forward<Args>(args)...));

    template<class T>
    ROCPRIM_HOST_DEVICE static auto call(MT B::*pmd, T&& t)
        -> decltype(invoke_impl::get(std::forward<T>(t)).*pmd);
};

template<class F, class... Args, class Fd = typename std::decay<F>::type>
ROCPRIM_HOST_DEVICE auto INVOKE(F&& f, Args&&... args)
    -> decltype(invoke_impl<Fd>::call(std::forward<F>(f), std::forward<Args>(args)...));

// Conforming C++14 implementation (is also a valid C++11 implementation):
template<typename AlwaysVoid, typename, typename...>
struct invoke_result_impl
{};
template<typename F, typename... Args>
struct invoke_result_impl<decltype(void(INVOKE(std::declval<F>(), std::declval<Args>()...))),
                          F,
                          Args...>
{
    using type = decltype(INVOKE(std::declval<F>(), std::declval<Args>()...));
};

template<class T>
struct is_tuple_of_references
{
    static_assert(sizeof(T) == 0, "is_tuple_of_references is only implemented for rocprim::tuple");
};

template<class... Args>
struct is_tuple_of_references<::rocprim::tuple<Args...>>
{
private:
    template<size_t Index>
    ROCPRIM_HOST_DEVICE static constexpr bool is_tuple_of_references_impl()
    {
        using tuple_t   = ::rocprim::tuple<Args...>;
        using element_t = ::rocprim::tuple_element_t<Index, tuple_t>;
        return std::is_reference<element_t>::value && is_tuple_of_references_impl<Index + 1>();
    }

    template<>
    ROCPRIM_HOST_DEVICE static constexpr bool is_tuple_of_references_impl<sizeof...(Args)>()
    {
        return true;
    }

public:
    static constexpr bool value = is_tuple_of_references_impl<0>();
};

template<class Key>
struct float_bit_mask;

template<>
struct float_bit_mask<float>
{
    static constexpr uint32_t sign_bit = 0x80000000;
    static constexpr uint32_t exponent = 0x7F800000;
    static constexpr uint32_t mantissa = 0x007FFFFF;
    using bit_type                     = uint32_t;
};

template<>
struct float_bit_mask<double>
{
    static constexpr uint64_t sign_bit = 0x8000000000000000;
    static constexpr uint64_t exponent = 0x7FF0000000000000;
    static constexpr uint64_t mantissa = 0x000FFFFFFFFFFFFF;
    using bit_type                     = uint64_t;
};

template<>
struct float_bit_mask<rocprim::bfloat16>
{
    static constexpr uint16_t sign_bit = 0x8000;
    static constexpr uint16_t exponent = 0x7F80;
    static constexpr uint16_t mantissa = 0x007F;
    using bit_type                     = uint16_t;
};

template<>
struct float_bit_mask<rocprim::half>
{
    static constexpr uint16_t sign_bit = 0x8000;
    static constexpr uint16_t exponent = 0x7C00;
    static constexpr uint16_t mantissa = 0x03FF;
    using bit_type                     = uint16_t;
};

template<class...>
using void_t = void;

} // end namespace detail

/// \brief Behaves like ``std::invoke_result``, but allows the use of invoke_result
/// with device-only lambdas/functors in host-only functions on HIP-clang.
///
/// \tparam F Type of the function.
/// \tparam Args Input type(s) to the function ``F``.
template<class F, class... Args>
struct invoke_result : detail::invoke_result_impl<void, F, Args...>
{
#ifdef DOXYGEN_DOCUMENTATION_BUILD
    /// \brief The return type of the Callable type F if invoked with the arguments Args.
    /// \hideinitializer
    using type = detail::invoke_result_impl<void, F, Args...>::type;
#endif // DOXYGEN_DOCUMENTATION_BUILD
};

/// \brief Helper type. It is an alias for ``invoke_result::type``.
///
/// \tparam F Type of the function.
/// \tparam Args Input type(s) to the function ``F``.
template<class F, class... Args>
using invoke_result_t = typename invoke_result<F, Args...>::type;

/// \brief Utility wrapper around ``invoke_result`` for binary operators.
///
/// \tparam T Input type to the binary operator.
/// \tparam F Type of the binary operator.
template<class T, class F>
struct invoke_result_binary_op
{
    /// \brief The result type of the binary operator.
    using type = typename invoke_result<F, T, T>::type;
};

/// \brief Helper type. It is an alias for ``invoke_result_binary_op::type``.
///
/// \tparam T Input type to the binary operator.
/// \tparam F Type of the binary operator.
template<class T, class F>
using invoke_result_binary_op_t = typename invoke_result_binary_op<T, F>::type;

namespace detail
{

/// \brief If `T` is a rocPRIM binary functional type, provides the member constant `value` equal `true`.
///   For any other type, `value` is `false`.
template<typename T>
struct is_binary_functional
{
    static constexpr bool value = false;
};

template<typename T>
struct is_binary_functional<less<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<less_equal<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<greater<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<greater_equal<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<equal_to<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<not_equal_to<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<plus<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<minus<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<multiplies<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<maximum<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<minimum<T>>
{
    static constexpr bool value = true;
};

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule_typetraits

#endif // ROCPRIM_TYPE_TRAITS_HPP_
