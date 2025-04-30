// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TYPE_TRAITS_INTERFACE_HPP_
#define ROCPRIM_TYPE_TRAITS_INTERFACE_HPP_

#include "types.hpp"

#include <type_traits>

// common macros

/// \brief A reverse version of static_assert aims to increase code readability
#ifndef ROCPRIM_DO_NOT_COMPILE_IF
    #define ROCPRIM_DO_NOT_COMPILE_IF(condition, msg) static_assert(!(condition), msg)
#endif
/// \brief Wrapper macro for std::enable_if aims to increase code readability
#ifndef ROCPRIM_REQUIRES
    #define ROCPRIM_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr
#endif
#ifndef DOXYGEN_DOCUMENTATION_BUILD
    /// \brief Since every definable traits need to use `is_defined`, this macro reduce the amount of code
    #define ROCPRIM_TRAITS_GENERATE_IS_DEFINE(traits_name)                                 \
        template<class InputType, class = void>                                            \
        static constexpr bool is_defined = false;                                          \
        template<class InputType>                                                          \
        static constexpr bool                                                              \
            is_defined<InputType, detail::void_t<typename define<InputType>::traits_name>> \
            = true
#endif

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class...>
using void_t = void;

template<class Key>
struct [[deprecated]] float_bit_mask;

} // namespace detail

namespace traits
{
/// \defgroup type_traits_interfaces Interfaces for defining and obtaining the traits
/// \addtogroup type_traits_interfaces
/// @{

/// \par Overview
/// This template struct provides an interface for downstream libraries to implement type traits for
/// their custom types. Users can utilize this template struct to define traits for these types. Users
/// should only implement traits as required by specific algorithms, and some traits cannot be defined
/// if they can be inferred from others. This API is not static because of ODR.
/// \tparam T The type for which you want to define traits.
///
/// \par Example
/// \parblock
/// The example below demonstrates how to implement traits for a custom floating-point type.
/// \code{.cpp}
/// // Your type definition
/// struct custom_float_type
/// {};
/// // Implement the traits
/// template<>
/// struct rocprim::traits::define<custom_float_type>
/// {
///     using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
///     using number_format = rocprim::traits::number_format::values<traits::number_format::kind::floating_point_type>;
///     using float_bit_mask = rocprim::traits::float_bit_mask::values<uint32_t, 10, 10, 10>;
/// };
/// \endcode
/// The example below demonstrates how to implement traits for a custom integral type.
/// \code{.cpp}
/// // Your type definition
/// struct custom_int_type
/// {};
/// // Implement the traits
/// template<>
/// struct rocprim::traits::define<custom_int_type>
/// {
///     using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
///     using number_format = rocprim::traits::number_format::values<traits::number_format::kind::integral_type>;
///     using integral_sign = rocprim::traits::integral_sign::values<traits::integral_sign::kind::signed_type>;
/// };
/// \endcode
/// \endparblock
template<class T>
struct define
{};

/// @}

/// predef
template<class T>
struct get;

/// \defgroup available_traits Traits that can be used
/// \addtogroup available_traits
/// @{

/// \par Definability
/// * **Undefinable**: For types with `predefined traits`.
/// * **Optional**:  For other types.
/// \par How to define
/// \parblock
/// \code{.cpp}
/// using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
/// \endcode
/// \endparblock
/// \par How to use
/// \parblock
/// \code{.cpp}
/// rocprim::traits::get<InputType>().is_arithmetic();
/// \endcode
/// \endparblock
struct is_arithmetic
{
    /// \brief Value of this trait
    template<bool Val>
    struct values
    {
        /// \brief This indicates if the `InputType` is arithmetic.
        static constexpr auto value = Val;
    };

#ifndef DOXYGEN_DOCUMENTATION_BUILD

    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(is_arithmetic);

    // For c++ arithmetic types, return true, but will throw compile error when user try to define this trait for them
    template<class InputType, ROCPRIM_REQUIRES(std::is_arithmetic<InputType>::value)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(is_defined<InputType>,
                                  "Do not define trait `is_arithmetic` for c++ arithmetic types");
        return values<true>{};
    }

    // For third party types, if trait `is_arithmetic` not defined, will return default value `false`
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value && !is_defined<InputType>)>
    static constexpr auto get()
    {
        return values<false>{};
    }

    // For third party types, if trait `is_arithmetic` is defined, then should return its value
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value && is_defined<InputType>)>
    static constexpr auto get()
    {
        return typename define<InputType>::is_arithmetic{};
    }
#endif
};

/// \brief Arithmetic types, pointers, member pointers, and null pointers are considered scalar types.
/// \par Definability
/// * **Undefinable**: For types with `predefined traits`.
/// * **Optional**: For other types. If both `is_arithmetic` and `is_scalar` are defined, their values
/// must be consistent; otherwise, a compile-time error will occur.
/// \par How to define
/// \parblock
/// \code{.cpp}
/// using is_scalar = rocprim::traits::is_scalar::values<true>;
/// \endcode
/// \endparblock
/// \par How to use
/// \parblock
/// \code{.cpp}
/// rocprim::traits::get<InputType>().is_scalar();
/// \endcode
/// \endparblock
struct is_scalar
{
    /// \brief Value of this trait
    template<bool Val>
    struct values
    {
        /// \brief This indicates if the `InputType` is scalar.
        static constexpr auto value = Val;
    };

#ifndef DOXYGEN_DOCUMENTATION_BUILD

    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(is_scalar);

    // For c++ scalar types, return true, but will throw compile error when user try to define this trait for them
    template<class InputType, ROCPRIM_REQUIRES(std::is_scalar<InputType>::value)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(is_defined<InputType>,
                                  "Do not define trait `is_scalar` for c++ scalar types");
        return values<true>{};
    }

    // For third party types, if trait `is_scalar` is not defined, will return default value `false`
    // For rocprim or third party types that defined trait `is_arithmetic` as true the result should be `true`
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_scalar<InputType>::value && !is_defined<InputType>)>
    static constexpr auto get()
    {
        return values<is_arithmetic::get<InputType>().value>{};
    }
    // For third party types and rocprim types, if trait `is_scalar` is defined, will return the value
    // check if the `is_scalar` equals to `is_arithmetic`, or throw a compile error
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_scalar<InputType>::value && is_defined<InputType>)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_arithmetic::get<InputType>().value != typename define<InputType>::is_scalar{}.value,
            "Trait `is_arithmetic` and trait `is_scalar` should have the same value");
        return typename define<InputType>::is_scalar{};
    }
#endif
};

/// \par Definability
/// * **Undefinable**: For types with `predefined traits` and non-arithmetic types.
/// * **Required**: If you define `is_arithmetic` as `true`, you must also define this trait; otherwise, a
/// compile-time error will occur.
/// \par How to define
/// \parblock
/// \code{.cpp}
/// using number_format = rocprim::traits::number_format::values<number_format::kind::integral_type>;
/// \endcode
/// \endparblock
/// \par How to use
/// \parblock
/// \code{.cpp}
/// rocprim::traits::get<InputType>().is_integral();
/// rocprim::traits::get<InputType>().is_floating_point();
/// \endcode
/// \endparblock
struct number_format
{
    /// \brief The kind enum that indecates the values avaliable for this trait
    enum class kind
    {
        unknown_type        = 0,
        floating_point_type = 1,
        integral_type       = 2
    };

    /// \brief Value of this trait
    template<kind Val>
    struct values
    {
        /// \brief This indicates if the `InputType` is floating_point_type or integral_type or unknown_type.
        static constexpr auto value = Val;
    };

#ifndef DOXYGEN_DOCUMENTATION_BUILD

    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(number_format);

    // For c++ arithmetic types
    template<class InputType, ROCPRIM_REQUIRES(std::is_arithmetic<InputType>::value)>
    static constexpr auto get()
    { // C++ build-in arithmetic types are either floating point or integral
        return values < std::is_floating_point<InputType>::value ? kind::floating_point_type
                                                                 : kind::integral_type > {};
    }

    // For rocprim arithmetic types
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value
                              && is_arithmetic::get<InputType>().value)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(!is_defined<InputType>,
                                  "You must define trait `number_format` for arithmetic types");
        return typename define<InputType>::number_format{};
    }

    // For other types
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value
                              && !is_arithmetic::get<InputType>().value)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_defined<InputType>,
            "You cannot define trait `number_format` for non-arithmetic types");
        return values<number_format::kind::unknown_type>{};
    }
#endif
};

/// \par Definability
/// * **Undefinable**: For types with `predefined traits`, non-arithmetic types and floating-point types.
/// * **Required**: If you define `number_format` as `number_format::kind::floating_point_type`, you must also define this trait; otherwise, a
/// compile-time error will occur.
/// \par How to define
/// \parblock
/// \code{.cpp}
/// using integral_sign = rocprim::traits::integral_sign::values<traits::integral_sign::kind::signed_type>;
/// \endcode
/// \endparblock
/// \par How to use
/// \parblock
/// \code{.cpp}
/// rocprim::traits::get<InputType>().is_signed();
/// rocprim::traits::get<InputType>().is_unsigned();
/// \endcode
/// \endparblock
struct integral_sign
{
    /// \brief The kind enum that indecates the values avaliable for this trait
    enum class kind
    {
        unknown_type  = 0,
        signed_type   = 1,
        unsigned_type = 2
    };

    /// \brief Value of this trait
    template<kind Val>
    struct values
    {
        /// \brief This indicates if the `InputType` is signed_type or unsigned_type or unknown_type.
        static constexpr auto value = Val;
    };

#ifndef DOXYGEN_DOCUMENTATION_BUILD

    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(integral_sign);

    // For c++ arithmetic types
    template<class InputType, ROCPRIM_REQUIRES(std::is_arithmetic<InputType>::value)>
    static constexpr auto get()
    { // cpp arithmetic types are either signed point or unsignned
        return values < std::is_signed<InputType>::value ? kind::signed_type
                                                         : kind::unsigned_type > {};
    }

    // For rocprim arithmetic integral
    template<class InputType,
             ROCPRIM_REQUIRES(
                 !std::is_arithmetic<InputType>::value && is_arithmetic::get<InputType>().value
                 && number_format::get<InputType>().value == number_format::kind::integral_type)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(!is_defined<InputType>,
                                  "Trait `integral_sign` is required for arithmetic "
                                  "integral types, please define");
        return typename define<InputType>::integral_sign{};
    }

    // For rocprim arithmetic non-integral
    template<class InputType,
             ROCPRIM_REQUIRES(
                 !std::is_arithmetic<InputType>::value && is_arithmetic::get<InputType>().value
                 && number_format::get<InputType>().value != number_format::kind::integral_type)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_defined<InputType>,
            "You cannot define trait `integral_sign` for arithmetic non-integral types");
        return values<kind::unknown_type>{};
    }

    // For other types
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value
                              && !is_arithmetic::get<InputType>().value)>
    static constexpr auto get()
    { // For other types, trait is_floating_point is a must
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_defined<InputType>,
            "You cannot define trait `integral_sign` for non-arithmetic types");
        return values<kind::unknown_type>{};
    }
#endif
};

/// \warning For some types, if this trait is not implemented in their traits definition, it will
/// link to `rocprim::detail::float_bit_mask` to maintain compatibility with downstream libraries.
/// However, this linkage will be removed in the next major release. Please ensure that these types
/// are updated to the latest interface.
/// \par Definability
/// * **Undefinable**: For types with `predefined traits`, non-arithmetic types and integral types.
/// * **Required**: If you define `number_format` as `number_format::kind::unknown_type`, you must also define this trait; otherwise, a
/// compile-time error will occur.
/// \par How to define
/// \parblock
/// \code{.cpp}
/// using float_bit_mask = rocprim::traits::float_bit_mask::values<int,1,1,1>;
/// \endcode
/// \endparblock
/// \par How to use
/// \parblock
/// \code{.cpp}
/// rocprim::traits::get<InputType>().float_bit_mask();
/// \endcode
/// \endparblock
struct float_bit_mask
{
    /// \brief Value of this trait
    template<class BitType, BitType SignBit, BitType Exponent, BitType Mantissa>
    struct values
    {
        ROCPRIM_DO_NOT_COMPILE_IF(number_format::get<BitType>().value
                                      != number_format::kind::integral_type,
                                  "BitType should be integral");
        /// \brief Trait sign_bit for the `InputType`.
        static constexpr BitType sign_bit = SignBit;
        /// \brief Trait exponent for the `InputType`.
        static constexpr BitType exponent = Exponent;
        /// \brief Trait mantissa for the `InputType`.
        static constexpr BitType mantissa = Mantissa;
    };

#ifndef DOXYGEN_DOCUMENTATION_BUILD

    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(float_bit_mask);

    template<class InputType, class = void>
    static constexpr bool has_old_float_bit_mask = false;
    template<class InputType>
    static constexpr bool has_old_float_bit_mask<
        InputType,
        detail::void_t<decltype(rocprim::detail::float_bit_mask<InputType>{})>>
        = true;

    // If this trait is defined, then use the new interface
    template<class InputType, ROCPRIM_REQUIRES(is_defined<InputType>)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            number_format::get<InputType>().value != number_format::kind::floating_point_type,
            "You cannot use trait `float_bit_mask` for `non-floating_point` types");
        return typename define<InputType>::float_bit_mask{};
    }

    // This function acts as a bridge for old interface. Will be removed in certain version
    // "`rocprim::detail::float_bit_mask` will be deprecated on next main release,"
    // "`please use rocprim::trait::define` to define tratis for types."
    template<class InputType,
             ROCPRIM_REQUIRES(!is_defined<InputType> && has_old_float_bit_mask<InputType>)>
    static constexpr auto get()
    {
        using mask = typename ::rocprim::detail::float_bit_mask<InputType>;
        return values<typename mask::bit_type, mask::sign_bit, mask::exponent, mask::mantissa>{};
    }

    // For types that don't have a trait `float_bit_mask` defined neither a rocprim::detail::float_bit_mask specialization
    template<class InputType,
             ROCPRIM_REQUIRES(!is_defined<InputType> && !has_old_float_bit_mask<InputType>)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            number_format::get<InputType>().value != number_format::kind::floating_point_type,
            "You cannot use trait `float_bit_mask` for `non-floating_point` types");
        ROCPRIM_DO_NOT_COMPILE_IF(number_format::get<InputType>().value
                                      == number_format::kind::floating_point_type,
                                  "Trait `float_bit_mask` is required for `floating_point` types");
        return values<int, 0, 0, 0>{};
    }
#endif
};

/// \brief The trait `is_fundamental` is undefinable, as it is the union of `std::is_fundamental`
/// and `rocprim::traits::is_arithmetic`.
/// \par Definability
/// * **Undefinable**: If you attempt to define this trait in any form, a compile-time error will occur.
/// \par How to use
/// \parblock
/// \code{.cpp}
/// rocprim::traits::get<InputType>().is_fundamental();
/// rocprim::traits::get<InputType>().is_compound();
/// \endcode
/// \endparblock
struct is_fundamental
{

    /// \brief Value of this trait
    template<bool Val>
    struct values
    {
        /// \brief This indicates if the `InputType` is fundamental.
        static constexpr auto value = Val;
    };

#ifndef DOXYGEN_DOCUMENTATION_BUILD

    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(is_fundamental);

    // For all types
    template<class InputType>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(is_defined<InputType>, "Trait `is_fundamental` is undefinable");
        return values < std::is_fundamental<InputType>::value
               || is_arithmetic::get<InputType>().value > {};
    }
#endif
};

/// @}

/// \addtogroup type_traits_interfaces
/// @{

/// \par Overview
/// This template struct is designed to allow rocPRIM algorithms to retrieve trait information from C++
/// build-in arithmetic types, rocPRIM types, and custom types. This API is not static because of ODR.
/// * All member functions are `compiled only when invoked`.
/// * Different algorithms require different traits.
/// \tparam T The type from which you want to retrieve the traits.
/// \par Example
/// \parblock
/// The following code demonstrates how to retrieve the traits of type `T`.
/// \code{.cpp}
/// // Get the trait in a template parameter
/// template<class T, std::enable_if<rocprim::traits::get<T>().is_integral()>::type* = nullptr>
/// void get_traits_in_template_parameter(){}
/// // Get the trait in a function body
/// template<class T>
/// void get_traits_in_function_body(){
///     constexpr auto input_traits = rocprim::traits::get<InputType>();
///     // Then you can use the member functinos
///     constexpr bool is_arithmetic = input_traits.is_arithmetic();
/// }
/// \endcode
/// \endparblock
template<class T>
struct get
{
    /// \brief Get the value of trait `is_arithmetic`.
    /// \returns `true` if `std::is_arithmetic_v<T>` is `true`, or if type `T` is a rocPRIM arithmetic
    /// type, or if the `is_arithmetic` trait has been defined as `true`; otherwise, returns `false`.
    constexpr bool is_arithmetic() const
    {
        return rocprim::traits::is_arithmetic{}.get<T>().value;
    };

    /// \brief Get trait `is_fundamental`.
    /// \returns `true` if `T` is a fundamental type (that is, rocPRIM arithmetic type, void, or nullptr_t);
    /// otherwise, returns `false`.
    constexpr bool is_fundamental() const
    {
        return rocprim::traits::is_fundamental{}.get<T>().value;
    };

    /// \brief If `T` is fundamental type, then returns `false`.
    /// \returns `false` if `T` is a fundamental type (that is, rocPRIM arithmetic type, void, or nullptr_t);
    /// otherwise, returns `true`.
    constexpr bool is_compound() const
    {
        return !rocprim::traits::is_fundamental{}.get<T>().value;
    }

    /// \brief To check if `T` is floating-point type.
    /// \warning You cannot call this function when `is_arithmetic()` returns `false`;
    /// doing so will result in a compile-time error.
    constexpr bool is_floating_point() const
    {
        return rocprim::traits::number_format{}.get<T>().value
               == number_format::kind::floating_point_type;
    };

    /// \brief To check if `T` is integral type.
    /// \warning You cannot call this function when `is_arithmetic()` returns `false`;
    /// doing so will result in a compile-time error.
    constexpr bool is_integral() const
    {
        return rocprim::traits::number_format{}.get<T>().value
               == number_format::kind::integral_type;
    }

    /// \brief To check if `T` is signed integral type.
    /// \warning You cannot call this function when `is_integral()` returns `false`;
    /// doing so will result in a compile-time error.
    constexpr bool is_signed() const
    {
        return rocprim::traits::integral_sign{}.get<T>().value == integral_sign::kind::signed_type;
    }

    /// \brief To check if `T` is unsigned integral type.
    /// \warning You cannot call this function when `is_integral()` returns `false`;
    /// doing so will result in a compile-time error.
    constexpr bool is_unsigned() const
    {
        return rocprim::traits::integral_sign{}.get<T>().value
               == integral_sign::kind::unsigned_type;
    }

    /// \brief Get trait `is_scalar`.
    /// \returns `true` if `std::is_scalar_v<T>` is `true`, or if type `T` is a rocPRIM arithmetic
    /// type, or if the `is_scalar` trait has been defined as `true`; otherwise, returns `false`.
    constexpr bool is_scalar() const
    {
        return rocprim::traits::is_scalar{}.get<T>().value;
    }

    /// \brief Get trait `float_bit_mask`.
    /// \warning You cannot call this function when `is_floating_point()` returns `false`;
    /// doing so will result in a compile-time error.
    /// \returns A constexpr instance of the specialization of `rocprim::traits::float_bit_mask::values`
    /// as provided in the traits definition of type T. If the `float_bit_mask trait` is not defined, it
    /// returns the rocprim::detail::float_bit_mask values, provided a specialization of
    /// `rocprim::detail::float_bit_mask<T>` exists.
    constexpr auto float_bit_mask() const
    {
        return rocprim::traits::float_bit_mask{}.get<T>();
    };
};

/// @}

} // namespace traits

/// \defgroup rocprim_pre_defined_traits Trait definitions for rocPRIM arithmetic types and additional traits for
/// C++ build-in arithmetic types.
/// \addtogroup rocprim_pre_defined_traits
/// @{

/// \brief This is the definition of traits of `float`
/// C++ build-in type
template<>
struct traits::define<float>
{
    /// \brief Trait `float_bit_mask` for this type
    using float_bit_mask
        = traits::float_bit_mask::values<uint32_t, 0x80000000, 0x7F800000, 0x007FFFFF>;
};

/// \brief This is the definition of traits of `double`
/// C++ build-in type
template<>
struct traits::define<double>
{
    /// \brief Trait `float_bit_mask` for this type
    using float_bit_mask = traits::float_bit_mask::
        values<uint64_t, 0x8000000000000000, 0x7FF0000000000000, 0x000FFFFFFFFFFFFF>;
};

/// \brief This is the definition of traits of `rocprim::bfloat16`
/// rocPRIM arithmetic type
template<>
struct traits::define<rocprim::bfloat16>
{
    /// \brief Trait `is_arithmetic` for this type
    using is_arithmetic = traits::is_arithmetic::values<true>;
    /// \brief Trait `number_format` for this type
    using number_format
        = traits::number_format::values<traits::number_format::kind::floating_point_type>;
    /// \brief Trait `float_bit_mask` for this type
    using float_bit_mask = traits::float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
};

/// \brief This is the definition of traits of `rocprim::half`
/// rocPRIM arithmetic type
template<>
struct traits::define<rocprim::half>
{
    /// \brief Trait `is_arithmetic` for this type
    using is_arithmetic = traits::is_arithmetic::values<true>;
    /// \brief Trait `number_format` for this type
    using number_format
        = traits::number_format::values<traits::number_format::kind::floating_point_type>;
    /// \brief Trait `float_bit_mask` for this type
    using float_bit_mask = traits::float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
};

// Type traits like std::is_integral and std::is_arithmetic may be defined for 128-bit integral
// types (__int128_t and __uint128_t) in several cases:
//  * with libstdc++ when GNU extensions are enabled (-std=gnu++17, which is the default C++
//    standard in clang);
//  * always with libc++ (it is used on HIP SDK for Windows).

namespace detail
{

struct define_int128_t
{
    /// \brief Trait `is_arithmetic` for this type
    using is_arithmetic = traits::is_arithmetic::values<true>;
    /// \brief Trait `number_format` for this type
    using number_format = traits::number_format::values<traits::number_format::kind::integral_type>;
    /// \brief Trait `integral_sign` for this type
    using integral_sign = traits::integral_sign::values<traits::integral_sign::kind::signed_type>;
};

struct define_uint128_t
{
    /// \brief Trait `is_arithmetic` for this type
    using is_arithmetic = traits::is_arithmetic::values<true>;
    /// \brief Trait `number_format` for this type
    using number_format = traits::number_format::values<traits::number_format::kind::integral_type>;
    /// \brief Trait `integral_sign` for this type
    using integral_sign = traits::integral_sign::values<traits::integral_sign::kind::unsigned_type>;
};

} // namespace detail

/// \brief This is the definition of traits of `rocprim::int128_t`
/// rocPRIM arithmetic type
template<>
struct traits::define<rocprim::int128_t>
    : std::conditional_t<std::is_arithmetic<rocprim::int128_t>::value,
                         traits::define<void>,
                         detail::define_int128_t>
{};

/// \brief This is the definition of traits of `rocprim::uint128_t`
/// rocPRIM arithmetic type
template<>
struct traits::define<rocprim::uint128_t>
    : std::conditional_t<std::is_arithmetic<rocprim::uint128_t>::value,
                         traits::define<void>,
                         detail::define_uint128_t>
{};

/// @}

/// \defgroup rocprim_type_traits_wrapper Handy wrappers for obtaining type traits
/// \addtogroup rocprim_type_traits_wrapper
/// @{

/// \brief An extension of `std::is_floating_point` that supports additional arithmetic types,
/// including `rocprim::half`, `rocprim::bfloat16`, and any types with trait
/// `rocprim::traits::number_format::values<number_format::kind::floating_point_type>` implemented.
template<class T>
struct is_floating_point
    : std::integral_constant<bool, ::rocprim::traits::get<T>().is_floating_point()>
{};

/// \brief An extension of `std::is_integral` that supports additional arithmetic types,
/// including `rocprim::int128_t`, `rocprim::uint128_t`, and any types with trait
/// `rocprim::traits::number_format::values<number_format::kind::integral_type>` implemented.
template<class T>
struct is_integral : std::integral_constant<bool, ::rocprim::traits::get<T>().is_integral()>
{};

/// \brief An extension of `std::is_arithmetic` that supports additional arithmetic types,
/// including any types with trait `rocprim::traits::is_arithmetic::values<true>` implemented.
template<class T>
struct is_arithmetic : std::integral_constant<bool, ::rocprim::traits::get<T>().is_arithmetic()>
{};

/// \brief An extension of `std::is_fundamental` that supports additional arithmetic types,
/// including any types with trait `rocprim::traits::is_arithmetic::values<true>` implemented.
template<class T>
struct is_fundamental : std::integral_constant<bool, ::rocprim::traits::get<T>().is_fundamental()>
{};

/// \brief An extension of `std::is_unsigned` that supports additional arithmetic types,
/// including `rocprim::uint128_t`, and any types with trait
/// `rocprim::traits::integral_sign::values<integral_sign::kind::unsigned_type>` implemented.
template<class T>
struct is_unsigned : std::integral_constant<bool, ::rocprim::traits::get<T>().is_unsigned()>
{};

/// \brief An extension of `std::is_signed` that supports additional arithmetic types,
/// including `rocprim::int128_t`, and any types with trait
/// `rocprim::traits::integral_sign::values<integral_sign::kind::signed_type>` implemented.
template<class T>
struct is_signed : std::integral_constant<bool, ::rocprim::traits::get<T>().is_signed()>
{};

/// \brief An extension of `std::is_scalar` that supports additional arithmetic types,
/// including any types with trait `rocprim::traits::is_scalar::values<true>` implemented.
template<class T>
struct is_scalar : std::integral_constant<bool, ::rocprim::traits::get<T>().is_scalar()>
{};

/// \brief An extension of `std::is_scalar` that supports additional non-arithmetic types.
template<class T>
struct is_compound : std::integral_constant<bool, ::rocprim::traits::get<T>().is_compound()>
{};

/// @}
END_ROCPRIM_NAMESPACE

#endif
