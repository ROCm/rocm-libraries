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

// Runtime compilation (MIOPEN_HIP_RUNTIME_COMPILE) may run where no host
// C++ standard library is installed (for example the AMD Windows ROCm
// wheels without an MSVC installation). Prefer the real <type_traits>
// whenever it is reachable; otherwise fall back to the minimal traits
// below so runtime-compiled kernels stay self-contained. The fallback
// definitions never coexist with a real standard library in one
// translation unit (that coexistence is the rocm-libraries#7718 failure
// class).
#if defined(MIOPEN_HIP_RUNTIME_COMPILE) && !__has_include(<type_traits>)

#if __has_include(<utility>)
#error \
    "inconsistent C++ standard library availability: <type_traits> is not reachable but <utility> is; provide a complete standard library (or none)"
#endif

namespace std {

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
struct remove_volatile
{
    using type = T;
};
template <class T>
struct remove_volatile<volatile T>
{
    using type = T;
};
template <class T>
using remove_volatile_t = typename remove_volatile<T>::type;

template <class T>
struct remove_cv
{
    using type = typename remove_volatile<typename remove_const<T>::type>::type;
};
template <class T>
using remove_cv_t = typename remove_cv<T>::type;

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

namespace detail {
template <class T>
struct is_pointer_helper : false_type
{
};
template <class T>
struct is_pointer_helper<T*> : true_type
{
};
} // namespace detail

template <class T>
struct is_pointer : detail::is_pointer_helper<typename remove_cv<T>::type>
{
};

template <bool B, class X, class Y>
struct conditional
{
    using type = X;
};
template <class X, class Y>
struct conditional<false, X, Y>
{
    using type = Y;
};
template <bool B, class X, class Y>
using conditional_t = typename conditional<B, X, Y>::type;

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
