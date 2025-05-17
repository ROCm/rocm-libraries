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

#ifndef ROCPRIM_TYPE_TRAITS_HPP_
#define ROCPRIM_TYPE_TRAITS_HPP_

#include "type_traits_functions.hpp"
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

namespace traits
{
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

    // If this trait is defined, then use the new interface
    template<class InputType, ROCPRIM_REQUIRES(is_defined<InputType>)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            number_format::get<InputType>().value != number_format::kind::floating_point_type,
            "You cannot use trait `float_bit_mask` for `non-floating_point` types");
        return typename define<InputType>::float_bit_mask{};
    }

    // For types that don't have a trait `float_bit_mask` defined
    template<class InputType, ROCPRIM_REQUIRES(!is_defined<InputType>)>
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

/// \par Definability
/// * **Undefinable**: For all types.
/// \par Overview This triat is auto matically generated.
/// \par How to use
/// \parblock
/// \code{.cpp}
/// constexpr auto codec = rocprim::traits::get<InputType>().radix_key_codec();
/// using codec_t = decltype(codec);
/// \endcode
/// \endparblock
struct radix_key_codec
{
#ifndef DOXYGEN_DOCUMENTATION_BUILD
    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(radix_key_codec);
    template<typename Destination, typename Source>
    static ROCPRIM_HOST_DEVICE
    auto bit_cast(const Source& source)
        -> std::enable_if_t<rocprim::detail::is_valid_bit_cast<Destination, Source>, Destination>
    {
    #if defined(__has_builtin) && __has_builtin(__builtin_bit_cast)
        return __builtin_bit_cast(Destination, source);
    #else
        static_assert(std::is_trivially_constructable<Destination>::value,
                      "Fallback implementation of bit_cast requires Destination to be trivially "
                      "constructible");
        Destination dest;
        memcpy(&dest, &source, sizeof(Destination));
        return dest;
    #endif
    }
    template<class Key>
    using get_bit_key_type = typename std::conditional<
        sizeof(Key) == sizeof(char),
        unsigned char,
        typename std::conditional<
            sizeof(Key) == sizeof(short),
            unsigned short,
            typename std::conditional<
                sizeof(Key) == sizeof(int),
                unsigned int,
                typename std::conditional<
                    sizeof(Key) == sizeof(long long),
                    unsigned long long,
                    typename std::conditional<sizeof(Key) == sizeof(rocprim::int128_t),
                                              rocprim::uint128_t,
                                              void>::type>::type>::type>::type>::type;

    /// \brief Encode and decode integral and floating point values for radix sort in such a way that preserves
    /// correct order of negative and positive keys (i.e. negative keys go before positive ones,
    /// which is not true for a simple reinterpetation of the key's bits).
    ///
    /// Digit extractor takes into account that (+0.0 == -0.0) is true for floats,
    /// so both +0.0 and -0.0 are reflected into the same bit pattern for digit extraction.
    /// Maximum digit length is 32.
    template<class Key, class Enable = void>
    struct codec_base
    {};

    /// \brief For unsigned integral types
    template<class Key>
    struct codec_base<
        Key,
        typename std::enable_if<
            number_format::get<Key>().value == number_format::kind::integral_type
            && integral_sign::get<Key>().value == integral_sign::kind::unsigned_type>::type>
    {
        using bit_key_type = get_bit_key_type<Key>;

        ROCPRIM_HOST_DEVICE
        static bit_key_type encode(Key key)
        {
            return bit_cast<bit_key_type>(key);
        }
        ROCPRIM_HOST_DEVICE
        static Key decode(bit_key_type bit_key)
        {
            return bit_cast<Key>(bit_key);
        }

        template<bool Descending>
        ROCPRIM_HOST_DEVICE
        static unsigned int
            extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
        {
            unsigned int mask = (1u << length) - 1;
            return static_cast<unsigned int>(bit_key >> start) & mask;
        }
    };

    /// \brief For signed integral types
    template<class Key>
    struct codec_base<
        Key,
        typename std::enable_if<
            number_format::get<Key>().value == number_format::kind::integral_type
            && integral_sign::get<Key>().value == integral_sign::kind::signed_type>::type>
    {
        using bit_key_type = get_bit_key_type<Key>;

        static constexpr bit_key_type sign_bit = bit_key_type(1) << (sizeof(bit_key_type) * 8 - 1);

        ROCPRIM_HOST_DEVICE
        static bit_key_type           encode(Key key)
        {
            const auto bit_key = bit_cast<bit_key_type>(key);
            return sign_bit ^ bit_key;
        }

        ROCPRIM_HOST_DEVICE
        static Key decode(bit_key_type bit_key)
        {
            bit_key ^= sign_bit;
            return bit_cast<Key>(bit_key);
        }

        template<bool Descending>
        ROCPRIM_HOST_DEVICE
        static unsigned int
            extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
        {
            unsigned int mask = (1u << length) - 1;
            return static_cast<unsigned int>(bit_key >> start) & mask;
        }
    };

    /// \brief For floating point types
    template<class Key>
    struct codec_base<Key,
                      typename std::enable_if<number_format::get<Key>().value
                                              == number_format::kind::floating_point_type>::type>
    {
        using bit_key_type = get_bit_key_type<Key>;

        static constexpr bit_key_type sign_bit = float_bit_mask::get<Key>().sign_bit;

        ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
        static bit_key_type           encode(Key key)
        {
            bit_key_type bit_key = bit_cast<bit_key_type>(key);
            bit_key ^= (sign_bit & bit_key) == 0 ? sign_bit : bit_key_type(-1);
            return bit_key;
        }

        ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
        static Key decode(bit_key_type bit_key)
        {
            bit_key ^= (sign_bit & bit_key) == 0 ? bit_key_type(-1) : sign_bit;
            return bit_cast<Key>(bit_key);
        }

        template<bool Descending>
        ROCPRIM_HOST_DEVICE
        static unsigned int
            extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
        {
            unsigned int mask = (1u << length) - 1;

            // radix_key_codec_floating::encode() maps 0.0 to 0x8000'0000,
            // and -0.0 to 0x7FFF'FFFF.
            // radix_key_codec::encode() then flips the bits if descending, yielding:
            // value | descending  | ascending   |
            // ----- | ----------- | ----------- |
            //   0.0 | 0x7FFF'FFFF | 0x8000'0000 |
            //  -0.0 | 0x8000'0000 | 0x7FFF'FFFF |
            //
            // For ascending sort, both should be mapped to 0x8000'0000,
            // and for descending sort, both should be mapped to 0x7FFF'FFFF.
            if constexpr(Descending)
            {
                bit_key = bit_key == sign_bit ? static_cast<bit_key_type>(~sign_bit) : bit_key;
            }
            else
            {
                bit_key = bit_key == static_cast<bit_key_type>(~sign_bit) ? sign_bit : bit_key;
            }
            return static_cast<unsigned int>(bit_key >> start) & mask;
        }
    };

    /// \brief For bool
    template<>
    struct codec_base<bool>
    {
        using bit_key_type = unsigned char;

        ROCPRIM_HOST_DEVICE
        static bit_key_type encode(bool key)
        {
            return static_cast<bit_key_type>(key);
        }

        ROCPRIM_HOST_DEVICE
        static bool decode(bit_key_type bit_key)
        {
            return static_cast<bool>(bit_key);
        }

        template<bool Descending>
        ROCPRIM_HOST_DEVICE
        static unsigned int
            extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
        {
            unsigned int mask = (1u << length) - 1;
            return static_cast<unsigned int>(bit_key >> start) & mask;
        }
    };

    /// \brief Determines whether a type has bit_key_type
    template<class T>
    struct has_bit_key_type
    {
        template<class U>
        static std::true_type check(typename U::bit_key_type*);

        template<class U>
        static std::false_type check(...);

        using result = decltype(check<T>(nullptr));
    };

#endif
    /// \brief Determines whether the type is fundamental for `radix_key`.
    template<class T>
    using radix_key_fundamental = typename has_bit_key_type<codec_base<T>>::result;

    /// \brief codec_base wrapper for fundamental radix key types
    template<class Key,
             bool Descending     = false,
             bool is_fundamental = radix_key_fundamental<Key>::value>
    class codec : protected codec_base<Key>
    {
        using base_type = codec_base<Key>;

    public:
        /// \brief Type of the encoded key.
        using bit_key_type = typename base_type::bit_key_type;
        /// \brief Encodes a key of type \p Key into \p bit_key_type.
        ///
        /// \tparam Decomposer Being \p Key a fundamental type, \p Decomposer should be
        /// \p identity_decomposer. This is also the type by default.
        /// \param [in] key Key to encode.
        /// \param [in] decomposer [optional] Decomposer functor.
        /// \return A \p bit_key_type encoded key.
        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static bit_key_type encode(Key key, Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            bit_key_type bit_key = base_type::encode(key);
            return Descending ? ~bit_key : bit_key;
        }

        /// \brief Encodes in-place a key of type \p Key.
        ///
        /// \tparam Decomposer Being \p Key a fundamental type, \p Decomposer should be
        /// \p identity_decomposer. This is also the type by default.
        /// \param [in, out] key Key to encode.
        /// \param [in] decomposer [optional] Decomposer functor.
        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static void encode_inplace(Key& key, Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            key = bit_cast<Key>(encode(key));
        }

        /// \brief Decodes an encoded key of type \p bit_key_type back into \p Key.
        ///
        /// \tparam Decomposer Being \p Key a fundamental type, \p Decomposer should be
        /// \p identity_decomposer. This is also the type by default.
        /// \param [in] bit_key Key to decode.
        /// \param [in] decomposer [optional] Decomposer functor.
        /// \return A \p Key decoded key.
        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static Key decode(bit_key_type bit_key, Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            bit_key = Descending ? ~bit_key : bit_key;
            return base_type::decode(bit_key);
        }

        /// \brief Decodes in-place an encoded key of type \p Key.
        ///
        /// \tparam Decomposer Being \p Key a fundamental type, \p Decomposer should be
        /// \p identity_decomposer. This is also the type by default.
        /// \param [in, out] key Key to decode.
        /// \param [in] decomposer [optional] Decomposer functor.
        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static void decode_inplace(Key& key, Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            key = decode(bit_cast<bit_key_type>(key));
        }

        /// \brief Extracts the specified bits from a given encoded key.
        ///
        /// \param [in] bit_key Encoded key.
        /// \param [in] start Start bit of the sequence of bits to extract.
        /// \param [in] radix_bits How many bits to extract.
        /// \return Requested bits from the key.
        ROCPRIM_HOST_DEVICE
        static unsigned int
            extract_digit(bit_key_type bit_key, unsigned int start, unsigned int radix_bits)
        {
            return base_type::template extract_digit<Descending>(bit_key, start, radix_bits);
        }

        /// \brief Extracts the specified bits from a given in-place encoded key.
        ///
        /// \tparam Decomposer Being \p Key a fundamental type, \p Decomposer should be
        /// \p identity_decomposer. This is also the type by default.
        /// \param [in] key Key.
        /// \param [in] start Start bit of the sequence of bits to extract.
        /// \param [in] radix_bits How many bits to extract.
        /// \param [in] decomposer [optional] Decomposer functor.
        /// \return Requested bits from the key.
        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static unsigned int extract_digit(Key          key,
                                          unsigned int start,
                                          unsigned int radix_bits,
                                          Decomposer   decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            return extract_digit(bit_cast<bit_key_type>(key), start, radix_bits);
        }

        /// \brief Gives the default value for out-of-bound keys of type \p Key.
        ///
        /// \tparam Decomposer Being \p Key a fundamental type, \p Decomposer should be
        /// \p identity_decomposer. This is also the type by default.
        /// \param [in] decomposer [optional] Decomposer functor.
        /// \return Out-of-bound keys' value.
        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static Key get_out_of_bounds_key(Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            return decode(static_cast<bit_key_type>(-1));
        }
    };

#ifndef DOXYGEN_DOCUMENTATION_BUILD
    template<bool Descending>
    class codec<bool, Descending> : protected codec_base<bool>
    {
        using base_type = codec_base<bool>;

    public:
        using bit_key_type = typename base_type::bit_key_type;

        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static bit_key_type encode(bool key, Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            return Descending != key;
        }

        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static void encode_inplace(bool& key, Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            key = bit_cast<bool>(encode(key));
        }

        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static bool decode(bit_key_type bit_key, Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            const bool key_value = bit_key;
            return Descending != key_value;
        }

        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static void decode_inplace(bool& key, Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            key = decode(bit_cast<bit_key_type>(key));
        }

        ROCPRIM_HOST_DEVICE
        static unsigned int
            extract_digit(bit_key_type bit_key, unsigned int start, unsigned int radix_bits)
        {
            return base_type::template extract_digit<Descending>(bit_key, start, radix_bits);
        }

        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static unsigned int extract_digit(bool         key,
                                          unsigned int start,
                                          unsigned int radix_bits,
                                          Decomposer   decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            return extract_digit(bit_cast<bit_key_type>(key), start, radix_bits);
        }

        template<class Decomposer = ::rocprim::identity_decomposer>
        ROCPRIM_HOST_DEVICE
        static bool get_out_of_bounds_key(Decomposer decomposer = {})
        {
            static_assert(std::is_same<decltype(decomposer), ::rocprim::identity_decomposer>::value,
                          "Fundamental types don't use custom decomposer.");
            return decode(static_cast<bit_key_type>(-1));
        }
    };
#endif

    /// \brief Specialization of `class codec` for non-fundamental radix key types
    template<class Key, bool Descending>
    class codec<Key, Descending, false /*radix_key_fundamental*/>
    {
    public:
        /// \brief The key in this case is a custom type, so \p bit_key_type cannot be the type of the
        /// encoded key because it depends on the decomposer used. It is thus set as the type \p Key.
        using bit_key_type = Key;

        /// \brief Encodes a key of type \p Key into \p bit_key_type.
        ///
        /// \tparam Decomposer Decomposer functor type. Being \p Key a custom key type, the decomposer
        /// type must be other than the \p identity_decomposer.
        /// \param [in] key Key to encode.
        /// \param [in] decomposer [optional] \p Key is a custom key type, so a custom decomposer
        /// functor that returns a \p ::rocprim::tuple of references to fundamental types from a
        /// \p Key key is needed.
        /// \return A \p bit_key_type encoded key.
        template<class Decomposer>
    ROCPRIM_HOST_DEVICE
        static bit_key_type encode(Key key, Decomposer decomposer = {})
        {
            encode_inplace(key, decomposer);
            return static_cast<bit_key_type>(key);
        }

        /// \brief Encodes in-place a key of type \p Key.
        ///
        /// \tparam Decomposer Decomposer functor type. Being \p Key a custom key type, the decomposer
        /// type must be other than the \p identity_decomposer.
        /// \param [in, out] key Key to encode.
        /// \param [in] decomposer [optional] \p Key is a custom key type, so a custom decomposer
        /// functor that returns a \p ::rocprim::tuple of references to fundamental types from a
        /// \p Key key is needed.
        template<class Decomposer>
    ROCPRIM_HOST_DEVICE
        static void encode_inplace(Key& key, Decomposer decomposer = {})
        {
            static_assert(!std::is_same<Decomposer, ::rocprim::identity_decomposer>::value,
                          "The decomposer of a custom-type key cannot be the identity decomposer.");
            static_assert(
                ::rocprim::detail::is_tuple_of_references<decltype(decomposer(key))>::value,
                "The decomposer must return a tuple of references.");
            const auto per_element_encode = [](auto& tuple_element)
            {
                using element_type_t = std::decay_t<decltype(tuple_element)>;
                using codec_t        = codec<element_type_t, Descending>;
                codec_t::encode_inplace(tuple_element);
            };
            ::rocprim::detail::for_each_in_tuple(decomposer(key), per_element_encode);
        }

        /// \brief Decodes an encoded key of type \p bit_key_type back into \p Key.
        ///
        /// \tparam Decomposer Decomposer functor type. Being \p Key a custom key type, the decomposer
        /// type must be other than the \p identity_decomposer.
        /// \param [in] bit_key Key to decode.
        /// \param [in] decomposer [optional] \p Key is a custom key type, so a custom decomposer
        /// functor that returns a \p ::rocprim::tuple of references to fundamental types from a
        /// \p Key key is needed.
        /// \return A \p Key decoded key.
        template<class Decomposer>
    ROCPRIM_HOST_DEVICE
        static Key decode(bit_key_type bit_key, Decomposer decomposer = {})
        {
            decode_inplace(bit_key, decomposer);
            return static_cast<Key>(bit_key);
        }

        /// \brief Decodes in-place an encoded key of type \p Key.
        ///
        /// \tparam Decomposer Decomposer functor type. Being \p Key a custom key type, the decomposer
        /// type must be other than the \p identity_decomposer.
        /// \param [in, out] key Key to decode.
        /// \param [in] decomposer [optional] Decomposer functor.
        template<class Decomposer>
    ROCPRIM_HOST_DEVICE
        static void decode_inplace(Key& key, Decomposer decomposer = {})
        {
            static_assert(!std::is_same<Decomposer, ::rocprim::identity_decomposer>::value,
                          "The decomposer of a custom-type key cannot be the identity decomposer.");
            static_assert(
                ::rocprim::detail::is_tuple_of_references<decltype(decomposer(key))>::value,
                "The decomposer must return a tuple of references.");
            const auto per_element_decode = [](auto& tuple_element)
            {
                using element_type_t = std::decay_t<decltype(tuple_element)>;
                using codec_t        = codec<element_type_t, Descending>;
                codec_t::decode_inplace(tuple_element);
            };
            ::rocprim::detail::for_each_in_tuple(decomposer(key), per_element_decode);
        }

        /// \brief Extracts the specified bits from a given encoded key.
        ///
        /// \return Requested bits from the key.
    ROCPRIM_HOST_DEVICE
        static unsigned int extract_digit(bit_key_type, unsigned int, unsigned int)
        {
            static_assert(
                sizeof(bit_key_type) == 0,
                "Only fundamental types (integral and floating point) are supported as radix sort"
                "keys without a decomposer. "
                "For custom key types, use the extract_digit overloads with the decomposer "
                "argument");
        }

        /// \brief Extracts the specified bits from a given in-place encoded key.
        ///
        /// \tparam Decomposer Decomposer functor type. Being \p Key a custom key type, the decomposer
        /// type must be other than the \p identity_decomposer.
        /// \param [in] key Key.
        /// \param [in] start Start bit of the sequence of bits to extract.
        /// \param [in] radix_bits How many bits to extract.
        /// \param [in] decomposer \p Key is a custom key type, so a custom decomposer
        /// functor that returns a \p ::rocprim::tuple of references to fundamental types from a
        /// \p Key key is needed.
        /// \return Requested bits from the key.
        template<class Decomposer>
    ROCPRIM_HOST_DEVICE
        static unsigned int extract_digit(Key          key,
                                          unsigned int start,
                                          unsigned int radix_bits,
                                          Decomposer   decomposer)
        {
            static_assert(!std::is_same<Decomposer, ::rocprim::identity_decomposer>::value,
                          "The decomposer of a custom-type key cannot be the identity decomposer.");
            static_assert(
                ::rocprim::detail::is_tuple_of_references<decltype(decomposer(key))>::value,
                "The decomposer must return a tuple of references.");
            constexpr size_t tuple_size
                = ::rocprim::tuple_size<std::decay_t<decltype(decomposer(key))>>::value;
            return extract_digit_from_key_impl<tuple_size - 1>(0u,
                                                               decomposer(key),
                                                               static_cast<int>(start),
                                                               static_cast<int>(start + radix_bits),
                                                               0);
        }

        /// \brief Gives the default value for out-of-bound keys of type \p Key.
        ///
        /// \tparam Decomposer Decomposer functor type. Being \p Key a custom key type, the decomposer
        /// type must be other than the \p identity_decomposer.
        /// \param [in] decomposer \p Key is a custom key type, so a custom decomposer
        /// functor that returns a \p ::rocprim::tuple of references to fundamental types from a
        /// \p Key key is needed.
        /// \return Out-of-bound keys' value.
        template<class Decomposer>
    ROCPRIM_HOST_DEVICE
        static Key get_out_of_bounds_key(Decomposer decomposer)
        {
            static_assert(!std::is_same<Decomposer, ::rocprim::identity_decomposer>::value,
                          "The decomposer of a custom-type key cannot be the identity decomposer.");
            static_assert(std::is_default_constructible<Key>::value,
                          "The sorted Key type must be default constructible");
            Key key;
            ::rocprim::detail::for_each_in_tuple(
                decomposer(key),
                [](auto& element)
                {
                    using element_t    = std::decay_t<decltype(element)>;
                    using codec_t      = codec<element_t, Descending>;
                    using bit_key_type = typename codec_t::bit_key_type;
                    element            = codec_t::decode(static_cast<bit_key_type>(-1));
                });
            return key;
        }

    private:
        template<int ElementIndex, class... Args>
    ROCPRIM_HOST_DEVICE
        static auto extract_digit_from_key_impl(unsigned int                     digit,
                                                const ::rocprim::tuple<Args...>& key_tuple,
                                                const int                        start,
                                                const int                        end,
                                                const int                        previous_bits)
            -> std::enable_if_t<(ElementIndex >= 0), unsigned int>
        {
            using T
                = std::decay_t<::rocprim::tuple_element_t<ElementIndex, ::rocprim::tuple<Args...>>>;
            using bit_key_type                 = typename codec<T, Descending>::bit_key_type;
            constexpr int current_element_bits = 8 * sizeof(T);

            const int total_extracted_bits    = end - start;
            const int current_element_end_bit = previous_bits + current_element_bits;
            if(start < current_element_end_bit && end > previous_bits)
            {
                // unsigned integral representation of the current tuple element
                const auto element_value
                    = bit_cast<bit_key_type>(::rocprim::get<ElementIndex>(key_tuple));

                const int bits_extracted_previously = ::rocprim::max(0, previous_bits - start);

                // start of the bit range copied from the current tuple element
                const int current_start_bit = ::rocprim::max(0, start - previous_bits);

                // end of the bit range copied from the current tuple element
                const int current_end_bit = ::rocprim::min(current_element_bits,
                                                           current_start_bit + total_extracted_bits
                                                               - bits_extracted_previously);

                // number of bits extracted from the current tuple element
                const int current_length = current_end_bit - current_start_bit;

                // bits extracted from the current tuple element, aligned to LSB
                unsigned int current_extract = (element_value >> current_start_bit);

                if(current_length != 32)
                {
                    current_extract &= (1u << current_length) - 1;
                }

                digit |= current_extract << bits_extracted_previously;
            }
            return extract_digit_from_key_impl<ElementIndex - 1>(digit,
                                                                 key_tuple,
                                                                 start,
                                                                 end,
                                                                 previous_bits
                                                                     + current_element_bits);
        }

        ///
        template<int ElementIndex, class... Args>
    ROCPRIM_HOST_DEVICE
        static auto extract_digit_from_key_impl(unsigned int digit,
                                                const ::rocprim::tuple<Args...>& /*key_tuple*/,
                                                const int /*start*/,
                                                const int /*end*/,
                                                const int /*previous_bits*/)
            -> std::enable_if_t<(ElementIndex < 0), unsigned int>
        {
            return digit;
        }
    };

    /// \brief The getter of this trait
    /// \tparam Key type of the radix key
    /// \returns  The specialization of `rocprim::traits::radix_key_codec::codec`.
    template<class Key, bool Descending>
    static constexpr auto get()
    {
        return codec<Key, Descending>{};
    }
};

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
        return std::is_fundamental<T>::value || rocprim::traits::is_arithmetic{}.get<T>().value;
    };

    /// \brief Check if the type is a `build_in` type, this function is different from `is_fundamental`,
    /// because, by implementing traits, downstream code can "hack" into rocprim to let a type be `arithmetic`,
    /// and by following the rules of `std::is_fundamental`, `rocprim::is_fundamental` returns a union set of
    /// `std::is_fundamental` and `rocprim::is_arithmetic`. So, to check wether a type is a build-in type,
    /// please use this function.
    /// \returns `true` if `T` is a `build_in` type (that is, char, unsigned char, short, unsigned short, int
    /// unsigned int, long long, unsigned long long, rocprim::int128_t, rocprim::uint128_t, rocprim::half,
    /// float, double);
    constexpr bool is_build_in() const
    {
        return std::is_same<T, bool>::value || std::is_same<T, char>::value
               || std::is_same<T, unsigned char>::value || std::is_same<T, short>::value
               || std::is_same<T, unsigned short>::value || std::is_same<T, int>::value
               || std::is_same<T, unsigned int>::value || std::is_same<T, long long>::value
               || std::is_same<T, unsigned long long>::value
               || std::is_same<T, rocprim::int128_t>::value
               || std::is_same<T, rocprim::uint128_t>::value
               || std::is_same<T, rocprim::half>::value || std::is_same<T, float>::value
               || std::is_same<T, rocprim::bfloat16>::value || std::is_same<T, double>::value;
    }

    /// \brief If `T` is fundamental type, then returns `false`.
    /// \returns `false` if `T` is a fundamental type (that is, rocPRIM arithmetic type, void, or nullptr_t);
    /// otherwise, returns `true`.
    constexpr bool is_compound() const
    {
        return !is_fundamental();
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

    /// \brief Get trait `radix_key_codec`.
    /// \returns A constexpr instance of the specialization of `rocprim::traits::radix_key_codec::codec`
    /// as provided in the traits definition of type T.
    template<bool Descending = false>
    constexpr auto radix_key_codec() const
    {
        return rocprim::traits::radix_key_codec{}.get<T, Descending>();
    }
};

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

static_assert(::rocprim::traits::radix_key_codec::radix_key_fundamental<int>::value,
              "'int' should be fundamental");
static_assert(!::rocprim::traits::radix_key_codec::radix_key_fundamental<int*>::value,
              "'int*' should not be fundamental");
static_assert(::rocprim::traits::radix_key_codec::radix_key_fundamental<rocprim::int128_t>::value,
              "'rocprim::int128_t' should be fundamental");
static_assert(::rocprim::traits::radix_key_codec::radix_key_fundamental<rocprim::uint128_t>::value,
              "'rocprim::uint128_t' should be fundamental");
static_assert(!::rocprim::traits::radix_key_codec::radix_key_fundamental<rocprim::int128_t*>::value,
              "'rocprim::int128_t*' should not be fundamental");

END_ROCPRIM_NAMESPACE

#endif
