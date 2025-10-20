//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc, hiprtc

// cuda::mr::resource

#include <cuda/memory_resource>
#include <cuda/std/cstdint>

#include "test_macros.h"

struct invalid_argument
{};

struct valid_resource
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const valid_resource&) const
  {
    return true;
  }
  bool operator!=(const valid_resource&) const
  {
    return false;
  }
};
static_assert(cuda::mr::resource<valid_resource>, "");

struct invalid_allocate_argument
{
  void* allocate(invalid_argument, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const invalid_allocate_argument&)
  {
    return true;
  }
  bool operator!=(const invalid_allocate_argument&)
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_allocate_argument>, "");

struct invalid_allocate_return
{
  int allocate(std::size_t, std::size_t)
  {
    return 42;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const invalid_allocate_return&)
  {
    return true;
  }
  bool operator!=(const invalid_allocate_return&)
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_allocate_return>, "");

struct invalid_deallocate_argument
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, invalid_argument, std::size_t) noexcept {}
  bool operator==(const invalid_deallocate_argument&)
  {
    return true;
  }
  bool operator!=(const invalid_deallocate_argument&)
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_deallocate_argument>, "");

struct non_comparable
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
};
static_assert(!cuda::mr::resource<non_comparable>, "");

struct non_eq_comparable
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator!=(const non_eq_comparable&)
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<non_eq_comparable>, "");

#if TEST_STD_VER < 2020
struct non_neq_comparable
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const non_neq_comparable&)
  {
    return true;
  }
};
static_assert(!cuda::mr::resource<non_neq_comparable>, "");
#endif // TEST_STD_VER < 2020

int main(int, char**)
{
  return 0;
}
