//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

#ifndef __CUDA_SPACE_SELECTOR_H
#define __CUDA_SPACE_SELECTOR_H

#include <cuda/std/cassert>
#include <cuda/std/cstddef>

// NOTE(HIP/AMD): We need to include new header to get the correct device definitions.
#if defined(__HIPCC_RTC__)
  #if __has_include("new")
    #include <new>
  #endif
#endif

#include "concurrent_agents.h"
#include "test_macros.h"

#if TEST_CUDA_COMPILER(CLANG)
#  include <new>
#endif // TEST_CUDA_COMPILER(CLANG)

#if _CCCL_COMPILER(NVRTC) || defined(_CCCL_COMPILER_HIPRTC)
#  define LAMBDA [=]
#else
#  define LAMBDA [=] __host__ __device__
#endif

#if defined (__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#  define SHARED __shared__
#else
#  define SHARED
#endif

template <typename T, cuda::std::size_t SharedOffset>
struct malloc_memory_provider
{
  static const constexpr cuda::std::size_t prefix_size =
    SharedOffset % alignof(T*) == 0 ? SharedOffset : SharedOffset + (alignof(T*) - SharedOffset % alignof(T*));
  static const constexpr cuda::std::size_t shared_offset = prefix_size + sizeof(T*);

private:
  __host__ __device__ T*& get_pointer()
  {
    alignas(T*)
#if defined (__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      __shared__
#else
      static
#endif
      char storage[shared_offset];
    T*& allocated = *reinterpret_cast<T**>(storage + prefix_size);
    return allocated;
  }

public:
  __host__ __device__ T* get()
  {
    execute_on_main_thread([&] {
      get_pointer() = reinterpret_cast<T*>(malloc(sizeof(T) + alignof(T)));
    });

    auto ptr = reinterpret_cast<cuda::std::uintptr_t>(get_pointer());
    ptr += alignof(T) - ptr % alignof(T);
    return reinterpret_cast<T*>(ptr);
  }

  __host__ __device__ ~malloc_memory_provider()
  {
    execute_on_main_thread([&] {
      free((void*) get_pointer());
    });
  }
};

template <typename T, cuda::std::size_t SharedOffset>
struct local_memory_provider
{
  static const constexpr cuda::std::size_t prefix_size   = 0;
  static const constexpr cuda::std::size_t shared_offset = SharedOffset;

  alignas(T) char buffer[sizeof(T)];

  __host__ __device__ T* get()
  {
    return reinterpret_cast<T*>(&buffer);
  }
};

template <typename T, cuda::std::size_t SharedOffset>
struct device_shared_memory_provider
{
  static const constexpr cuda::std::size_t prefix_size =
    SharedOffset % alignof(T) == 0 ? SharedOffset : SharedOffset + (alignof(T) - SharedOffset % alignof(T));
  static const constexpr cuda::std::size_t shared_offset = prefix_size + sizeof(T);

  __device__ T* get()
  {
    alignas(T) __shared__ char buffer[shared_offset];
    return reinterpret_cast<T*>(buffer + prefix_size);
  }
};

struct init_initializer
{
  template <typename T, typename... Ts>
  __host__ __device__ static void construct(T& t, Ts&&... ts)
  {
    #if !defined(_CCCL_COMPILER_HIPRTC)
    t.init(std::forward<Ts>(ts)...);
    #else
    t.init(cuda::std::forward<Ts>(ts)...);
    #endif
  }
};

struct constructor_initializer
{
  template <typename T, typename... Ts>
  __host__ __device__ static void construct(T& t, Ts&&... ts)
  {
    #if !defined(_CCCL_COMPILER_HIPRTC)
    new ((void*) &t) T(std::forward<Ts>(ts)...);
    #else
    new ((void*) &t) T(cuda::std::forward<Ts>(ts)...);
    #endif
  }
};

struct default_initializer
{
  template <typename T>
  __host__ __device__ static void construct(T& t)
  {
    new ((void*) &t) T;
  }
};

template <typename T,
          template <typename, cuda::std::size_t> class Provider,
          typename Initializer           = constructor_initializer,
          cuda::std::size_t SharedOffset = 0>
class memory_selector
{
  Provider<T, SharedOffset> provider;
  T* ptr;

public:
  template <cuda::std::size_t SharedOffset_>
  using offsetted = memory_selector<T, Provider, Initializer, SharedOffset + SharedOffset_>;

  static const constexpr cuda::std::size_t prefix_size   = Provider<T, SharedOffset>::prefix_size;
  static const constexpr cuda::std::size_t shared_offset = Provider<T, SharedOffset>::shared_offset;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename... Ts>
  __host__ __device__ T* construct(Ts&&... ts)
  {
    ptr = provider.get();
    assert(ptr);

    execute_on_main_thread([&] {
      #if !defined(_CCCL_COMPILER_HIPRTC)
      Initializer::construct(*ptr, std::forward<Ts>(ts)...);
      #else
      Initializer::construct(*ptr, cuda::std::forward<Ts>(ts)...);
      #endif
    });
    return ptr;
  }

  _CCCL_EXEC_CHECK_DISABLE
  __host__ __device__ ~memory_selector()
  {
    execute_on_main_thread([&] {
      ptr->~T();
    });
  }
};

template <typename T, typename Initializer = constructor_initializer>
using local_memory_selector = memory_selector<T, local_memory_provider, Initializer>;

template <typename T, typename Initializer = constructor_initializer>
using shared_memory_selector = memory_selector<T, device_shared_memory_provider, Initializer>;

template <typename T, typename Initializer = constructor_initializer>
using global_memory_selector = memory_selector<T, malloc_memory_provider, Initializer>;

#endif // __CUDA_SPACE_SELECTOR_H
