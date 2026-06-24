/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef ROCM_GUARD_ROCM_TYPE_TRAITS_HPP
#define ROCM_GUARD_ROCM_TYPE_TRAITS_HPP

#include <rocm/config.hpp>
#include <rocm/utility/declval.hpp>
#include <rocm/integral_constant.hpp>
#include <rocm/stddef.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T>
struct type_identity
{
    using type = T;
};

template <class T>
using type_identity_t = typename type_identity<T>::type;

template <bool B, class T = void>
struct enable_if
{
};

template <class T>
struct enable_if<true, T>
{
    using type = T;
};

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <bool B, class T, class F>
struct conditional
{
    using type = T;
};

template <class T, class F>
struct conditional<false, T, F>
{
    using type = F;
};

template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;

template <class T>
struct add_cv
{
    using type = const volatile T;
};
template <class T>
using add_cv_t = typename add_cv<T>::type;

template <class T>
struct add_const
{
    using type = const T;
};
template <class T>
using add_const_t = typename add_const<T>::type;

template <class T>
struct add_volatile
{
    using type = volatile T;
};
template <class T>
using add_volatile_t = typename add_volatile<T>::type;

template <class T>
struct remove_cv
{
    using type = T;
};

template <class T>
struct remove_cv<const T> : remove_cv<T>
{
};

template <class T>
struct remove_cv<volatile T> : remove_cv<T>
{
};

template <class T>
struct remove_cv<const volatile T> : remove_cv<T>
{
};

template <class T>
using remove_cv_t = typename remove_cv<T>::type;

template <class T>
struct remove_const
{
    using type = T;
};
template <class T>
struct remove_const<const T>
{
    using type = T;
};
template <class T>
using remove_const_t = typename remove_const<T>::type;

template <class T>
struct remove_reference
{
    using type = T;
};
template <class T>
struct remove_reference<T&>
{
    using type = T;
};
template <class T>
struct remove_reference<T&&>
{
    using type = T;
};

template <class T>
using remove_reference_t = typename remove_reference<T>::type;

template <class T>
struct remove_cvref : remove_cv<remove_reference_t<T>>
{
};
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <class T>
struct add_pointer : type_identity<remove_reference_t<T>*>
{
};

template <class T>
using add_pointer_t = typename add_pointer<T>::type;

template <class T>
struct remove_pointer
{
    using type = T;
};
template <class T>
struct remove_pointer<T*>
{
    using type = T;
};
template <class T>
struct remove_pointer<T* const>
{
    using type = T;
};
template <class T>
struct remove_pointer<T* volatile>
{
    using type = T;
};
template <class T>
struct remove_pointer<T* const volatile>
{
    using type = T;
};

template <class T>
using remove_pointer_t = typename remove_pointer<T>::type;

template <class... Ts>
struct common_type;

template <class T>
struct common_type<T>
{
    using type = typename common_type<T, T>::type;
};

template <class T, class U>
struct common_type<T, U>
{
    using type = remove_cv_t<remove_reference_t<decltype(true ? declval<T>() : declval<U>())>>;
};

template <class T, class U, class... Us>
struct common_type<T, U, Us...>
{
    using type = typename common_type<typename common_type<T, U>::type, Us...>::type;
};

template <class... Ts>
using common_type_t = typename common_type<Ts...>::type;

template <class...>
using void_t = void;

// NOLINTNEXTLINE
#define ROCM_BUILTIN_TYPE_TRAIT1(name)       \
    template <class T>                       \
    struct name : bool_constant<__##name(T)> \
    {                                        \
    };                                       \
    template <class T>                       \
    inline constexpr bool name##_v = __##name(T)

// NOLINTNEXTLINE
#define ROCM_BUILTIN_TYPE_TRAIT2(name)          \
    template <class T, class U>                 \
    struct name : bool_constant<__##name(T, U)> \
    {                                           \
    };                                          \
    template <class T, class U>                 \
    inline constexpr bool name##_v = __##name(T, U)

// NOLINTNEXTLINE
#define ROCM_BUILTIN_TYPE_TRAITN(name)           \
    template <class... Ts>                       \
    struct name : bool_constant<__##name(Ts...)> \
    {                                            \
    };                                           \
    template <class... Ts>                       \
    inline constexpr bool name##_v = __##name(Ts...)

// ROCM_BUILTIN_TYPE_TRAIT1(is_destructible);
// ROCM_BUILTIN_TYPE_TRAIT1(is_nothrow_destructible);
// ROCM_BUILTIN_TYPE_TRAIT1(is_scalar);
// ROCM_BUILTIN_TYPE_TRAIT1(is_signed);
// ROCM_BUILTIN_TYPE_TRAIT1(is_void);
ROCM_BUILTIN_TYPE_TRAIT1(is_abstract);
ROCM_BUILTIN_TYPE_TRAIT1(is_aggregate);
ROCM_BUILTIN_TYPE_TRAIT1(is_arithmetic);
ROCM_BUILTIN_TYPE_TRAIT1(is_array);
ROCM_BUILTIN_TYPE_TRAIT1(is_class);
ROCM_BUILTIN_TYPE_TRAIT1(is_compound);
ROCM_BUILTIN_TYPE_TRAIT1(is_const);
ROCM_BUILTIN_TYPE_TRAIT1(is_empty);
ROCM_BUILTIN_TYPE_TRAIT1(is_enum);
ROCM_BUILTIN_TYPE_TRAIT1(is_final);
ROCM_BUILTIN_TYPE_TRAIT1(is_floating_point);
ROCM_BUILTIN_TYPE_TRAIT1(is_function);
ROCM_BUILTIN_TYPE_TRAIT1(is_fundamental);
ROCM_BUILTIN_TYPE_TRAIT1(is_integral);
ROCM_BUILTIN_TYPE_TRAIT1(is_literal_type);
ROCM_BUILTIN_TYPE_TRAIT1(is_lvalue_reference);
ROCM_BUILTIN_TYPE_TRAIT1(is_member_function_pointer);
ROCM_BUILTIN_TYPE_TRAIT1(is_member_object_pointer);
ROCM_BUILTIN_TYPE_TRAIT1(is_member_pointer);
ROCM_BUILTIN_TYPE_TRAIT1(is_object);
ROCM_BUILTIN_TYPE_TRAIT1(is_pod);
ROCM_BUILTIN_TYPE_TRAIT1(is_pointer);
ROCM_BUILTIN_TYPE_TRAIT1(is_polymorphic);
ROCM_BUILTIN_TYPE_TRAIT1(is_reference);
ROCM_BUILTIN_TYPE_TRAIT1(is_rvalue_reference);
ROCM_BUILTIN_TYPE_TRAIT1(is_standard_layout);
ROCM_BUILTIN_TYPE_TRAIT1(is_trivial);
ROCM_BUILTIN_TYPE_TRAIT1(is_trivially_copyable);
ROCM_BUILTIN_TYPE_TRAIT1(is_trivially_destructible);
ROCM_BUILTIN_TYPE_TRAIT1(is_union);
ROCM_BUILTIN_TYPE_TRAIT1(is_unsigned);
ROCM_BUILTIN_TYPE_TRAIT1(is_volatile);
ROCM_BUILTIN_TYPE_TRAIT2(is_assignable);
ROCM_BUILTIN_TYPE_TRAIT2(is_base_of);
ROCM_BUILTIN_TYPE_TRAIT2(is_convertible);
ROCM_BUILTIN_TYPE_TRAIT2(is_nothrow_assignable);
ROCM_BUILTIN_TYPE_TRAIT2(is_same);
ROCM_BUILTIN_TYPE_TRAIT2(is_trivially_assignable);
ROCM_BUILTIN_TYPE_TRAITN(is_constructible);
ROCM_BUILTIN_TYPE_TRAITN(is_nothrow_constructible);
ROCM_BUILTIN_TYPE_TRAITN(is_trivially_constructible);

template <class T>
struct is_void : is_same<void, remove_cv_t<T>>
{
};
template <class T>
inline constexpr bool is_void_v = is_void<T>::value;

template <class T>
struct is_null_pointer : is_same<nullptr_t, remove_cv_t<T>>
{
};
template <class T>
inline constexpr bool is_null_pointer_v = is_null_pointer<T>::value;

#define ROCM_REQUIRES(...) class = enable_if_t<__VA_ARGS__>

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_TYPE_TRAITS_HPP
