//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types>
// constexpr bool
// operator==(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator!=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>=(variant<Types...> const&, variant<Types...> const&) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"

struct MyBoolExplicit
{
  bool value;
  constexpr explicit MyBoolExplicit(bool v)
      : value(v)
  {}
  constexpr explicit operator bool() const noexcept
  {
    return value;
  }
};

struct ComparesToMyBoolExplicit
{
  int value = 0;
};
inline constexpr MyBoolExplicit
operator==(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value == RHS.value);
}
inline constexpr MyBoolExplicit
operator!=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value != RHS.value);
}
inline constexpr MyBoolExplicit
operator<(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value < RHS.value);
}
inline constexpr MyBoolExplicit
operator<=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value <= RHS.value);
}
inline constexpr MyBoolExplicit
operator>(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value > RHS.value);
}
inline constexpr MyBoolExplicit
operator>=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value >= RHS.value);
}

int main(int, char**)
{
  using V = cuda::std::variant<int, ComparesToMyBoolExplicit>;
  V v1(42);
  V v2(101);
  // expected-error-re@variant:* 6 {{{{(static_assert|static assertion)}} failed{{.*}}the relational operator does not
  // return a type which is implicitly convertible to bool}} expected-error-re@variant:* 6 {{no viable conversion}}
  (void) (v1 == v2); // expected-note {{here}}
  (void) (v1 != v2); // expected-note {{here}}
  (void) (v1 < v2); // expected-note {{here}}
  (void) (v1 <= v2); // expected-note {{here}}
  (void) (v1 > v2); // expected-note {{here}}
  (void) (v1 >= v2); // expected-note {{here}}

  return 0;
}
