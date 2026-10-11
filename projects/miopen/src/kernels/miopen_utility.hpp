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

#ifdef MIOPEN_HIP_RUNTIME_COMPILE
#if HIP_PACKAGE_VERSION_FLAT < 7000000000ULL

#include "miopen_type_traits.hpp" // std::remove_reference

namespace std {

template <typename T>
constexpr T&& forward(typename remove_reference<T>::type& t_) noexcept
{
    return static_cast<T&&>(t_);
}

template <typename T>
constexpr T&& forward(typename remove_reference<T>::type&& t_) noexcept
{
    return static_cast<T&&>(t_);
}

} // namespace std

#else

// HIP >= 7: runtime compilation may run where no host C++ standard library
// is installed (e.g. ROCm Windows wheels without MSVC). Prefer the real
// <utility> whenever it is reachable; otherwise fall back to MIOpen's
// freestanding forward so runtime-compiled kernels stay self-contained.
#if __has_include(<utility>)
#include <utility>
#else
#if __has_include(<type_traits>)
#error \
    "inconsistent C++ standard library availability: <utility> is not reachable but <type_traits> is (a partial freestanding override on the include path?); provide a complete standard library (or none)"
#endif
#include "miopen_freestanding_utility.hpp"
#endif

#endif
#else

#include <utility>

#endif
