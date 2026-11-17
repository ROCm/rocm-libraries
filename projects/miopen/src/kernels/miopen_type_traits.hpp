/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

// Minimal <type_traits> definitions for runtime compilation without a host C++ standard library.
#if defined(MIOPEN_HIP_RUNTIME_COMPILE) && !__has_include(<type_traits>)

namespace std {

template <class T>
struct remove_reference
{
    typedef T type;
};
template <class T>
struct remove_reference<T&>
{
    typedef T type;
};
template <class T>
struct remove_reference<T&&>
{
    typedef T type;
};

template <class T>
using remove_reference_t = typename remove_reference<T>::type;

template <class T>
struct remove_const
{
    typedef T type;
};
template <class T>
struct remove_const<const T>
{
    typedef T type;
};
template <class T>
using remove_const_t = typename remove_const<T>::type;

template <class T>
struct remove_volatile
{
    typedef T type;
};
template <class T>
struct remove_volatile<volatile T>
{
    typedef T type;
};
template <class T>
using remove_volatile_t = typename remove_volatile<T>::type;

template <class T>
struct remove_cv
{
    typedef typename remove_volatile<typename remove_const<T>::type>::type type;
};

template <class T>
using remove_cv_t = typename remove_cv<T>::type;

template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    using value_type         = T;
    using type               = integral_constant;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

using true_type  = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <class T, class U>
struct is_same : false_type
{
};

template <class T>
struct is_same<T, T> : true_type
{
};

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

template <class T>
struct is_pointer_helper : false_type
{
};

template <class T>
struct is_pointer_helper<T*> : true_type
{
};

template <class T>
struct is_pointer : is_pointer_helper<typename remove_cv<T>::type>
{
};

template <bool predicate, typename X, typename Y>
struct conditional;

template <typename X, typename Y>
struct conditional<true, X, Y>
{
    using type = X;
};

template <typename X, typename Y>
struct conditional<false, X, Y>
{
    using type = Y;
};

template <bool predicate, typename X, typename Y>
using conditional_t = typename conditional<predicate, X, Y>::type;

// clang provides this as a builtin in all modes (host STL not required).
template <class T>
struct is_trivially_copyable : integral_constant<bool, __is_trivially_copyable(T)>
{
};
template <class T>
inline constexpr bool is_trivially_copyable_v = __is_trivially_copyable(T);

} // namespace std

// Self-tests: compiled (and discarded) wherever the fallback is used.
static_assert(std::is_same<std::remove_reference_t<int&>, int>::value,
              "freestanding remove_reference_t<int&> must be int");
static_assert(std::is_same<std::remove_reference_t<int&&>, int>::value,
              "freestanding remove_reference_t<int&&> must be int");
static_assert(std::is_same<std::remove_cv_t<const volatile float>, float>::value,
              "freestanding remove_cv_t must strip const volatile");
static_assert(std::is_same<std::remove_const_t<const double>, double>::value,
              "freestanding remove_const_t must strip const");
static_assert(std::true_type::value && !std::false_type::value,
              "freestanding true_type/false_type values");
static_assert(std::is_same<std::conditional_t<true, char, long>, char>::value &&
                  std::is_same<std::conditional_t<false, char, long>, long>::value,
              "freestanding conditional_t must select on B");
static_assert(std::is_same<std::enable_if_t<true, unsigned>, unsigned>::value,
              "freestanding enable_if_t<true> must yield T");
static_assert(std::is_pointer<int*>::value && !std::is_pointer<int>::value &&
                  std::is_pointer<const int*>::value && std::is_pointer<int* const>::value &&
                  std::is_pointer<int* volatile>::value &&
                  std::is_pointer<int* const volatile>::value,
              "freestanding is_pointer, incl. pointee- and pointer-cv");
static_assert(std::integral_constant<int, 5>::value == 5, "freestanding integral_constant value");
namespace miopen_type_traits_selftest {
struct non_trivial
{
    non_trivial(const non_trivial&) {} // user copy ctor
    int x;
};
} // namespace miopen_type_traits_selftest
static_assert(std::is_trivially_copyable<float>::value &&
                  std::is_trivially_copyable<float[4]>::value &&
                  !std::is_trivially_copyable<miopen_type_traits_selftest::non_trivial>::value,
              "freestanding is_trivially_copyable sanity");
static_assert(
    std::is_same<std::enable_if_t<std::is_trivially_copyable_v<double>, char>, char>::value,
    "freestanding is_trivially_copyable_v usable in SFINAE");

#else

#include <type_traits> // std::remove_reference, std::remove_cv, is_pointer

#endif
