//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: nvrtc, hiprtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "../helpers.h"

template <int Operand>
struct store_tester
{
  template <typename A>
  __host__ __device__ static void initialize(A& v)
  {
    cuda::std::atomic_ref<A> a(v);
    using T = decltype(a.load());
    a.store(static_cast<T>(Operand));
  }

  template <typename A>
  __host__ __device__ static void validate(A& v)
  {
    cuda::std::atomic_ref<A> a(v);
    using T = decltype(a.load());
    assert(a.load() == static_cast<T>(Operand));
  }
};

template <int PreviousValue, int Operand>
struct exchange_tester
{
  template <typename A>
  __host__ __device__ static void initialize(A& v)
  {
    cuda::std::atomic_ref<A> a(v);
    using T = decltype(a.load());
    assert(a.exchange(static_cast<T>(Operand)) == static_cast<T>(PreviousValue));
  }

  template <typename A>
  __host__ __device__ static void validate(A& v)
  {
    cuda::std::atomic_ref<A> a(v);
    using T = decltype(a.load());
    assert(a.load() == static_cast<T>(Operand));
  }
};

template <int PreviousValue, int Expected, int Desired, int Result>
struct strong_cas_tester
{
  enum
  {
    ShouldSucceed = (Expected == PreviousValue)
  };
  template <typename A>
  __host__ __device__ static void initialize(A& v)
  {
    cuda::std::atomic_ref<A> a(v);
    using T    = decltype(a.load());
    T expected = static_cast<T>(Expected);
    assert(a.compare_exchange_strong(expected, static_cast<T>(Desired)) == ShouldSucceed);
    assert(expected == static_cast<T>(PreviousValue));
  }

  template <typename A>
  __host__ __device__ static void validate(A& v)
  {
    cuda::std::atomic_ref<A> a(v);
    using T = decltype(a.load());
    assert(a.load() == static_cast<T>(Result));
  }
};

template <int PreviousValue, int Expected, int Desired, int Result>
struct weak_cas_tester
{
  enum
  {
    ShouldSucceed = (Expected == PreviousValue)
  };
  template <typename A>
  __host__ __device__ static void initialize(A& v)
  {
    cuda::std::atomic_ref<A> a(v);
    using T    = decltype(a.load());
    T expected = static_cast<T>(Expected);
    if (!ShouldSucceed)
    {
      assert(a.compare_exchange_weak(expected, static_cast<T>(Desired)) == false);
    }
    else
    {
      while (a.compare_exchange_weak(expected, static_cast<T>(Desired)) != ShouldSucceed)
        ;
    }
    assert(expected == static_cast<T>(PreviousValue));
  }

  template <typename A>
  __host__ __device__ static void validate(A& v)
  {
    cuda::std::atomic_ref<A> a(v);
    using T = decltype(a.load());
    assert(a.load() == static_cast<T>(Result));
  }
};

#define ATOMIC_TESTER(operation)                                     \
  template <int PreviousValue, int Operand, int ExpectedValue>       \
  struct operation##_tester                                          \
  {                                                                  \
    template <typename A>                                            \
    __host__ __device__ static void initialize(A& v)                 \
    {                                                                \
      cuda::std::atomic_ref<A> a(v);                                 \
      using T = decltype(a.load());                                  \
      assert(a.operation(Operand) == static_cast<T>(PreviousValue)); \
    }                                                                \
                                                                     \
    template <typename A>                                            \
    __host__ __device__ static void validate(A& v)                   \
    {                                                                \
      cuda::std::atomic_ref<A> a(v);                                 \
      using T = decltype(a.load());                                  \
      assert(a.load() == static_cast<T>(ExpectedValue));             \
    }                                                                \
  };

ATOMIC_TESTER(fetch_add);
ATOMIC_TESTER(fetch_sub);

ATOMIC_TESTER(fetch_and);
ATOMIC_TESTER(fetch_or);
ATOMIC_TESTER(fetch_xor);

using basic_testers =
  tester_list<store_tester<0>,
              store_tester<-1>,
              store_tester<17>,
              exchange_tester<17, 31>,
              /* *_cas_tester<PreviousValue, Expected, Desired, Result> */
              weak_cas_tester<31, 12, 13, 31>,
              weak_cas_tester<31, 31, -6, -6>,
              strong_cas_tester<-6, -6, -12, -12>,
              strong_cas_tester<-12, 31, 17, -12>,
              exchange_tester<-12, 17>>;

using arithmetic_atomic_testers =
  append<basic_testers, fetch_add_tester<17, 13, 30>
#ifdef LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH
         // NOTE(HIP/AMD): fetch_(sub/or/xor/and) are not supported with hipMallocManaged.
         // This issue is tracked internally in issue SWDEV-390383.
         , fetch_sub_tester<30, 21, 9>, fetch_sub_tester<9, 17, -8>
#endif
         >;

using bitwise_atomic_testers =
  append<arithmetic_atomic_testers,
#ifdef LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH
         fetch_add_tester<-8, 10, 2>,
         // NOTE(HIP/AMD): fetch_(sub/or/xor/and) are not supported with hipMallocManaged.
         // This issue is tracked internally in issue SWDEV-390383.
         fetch_or_tester<2, 13, 15>,
         fetch_and_tester<15, 8, 8>,
         fetch_and_tester<8, 13, 8>,
         fetch_xor_tester<8, 12, 4>
#else
         fetch_add_tester<30, 10, 40>
#endif
         >;

void kernel_invoker()
{
  validate_pinned<signed char, arithmetic_atomic_testers>();
  validate_pinned<signed short, arithmetic_atomic_testers>();
  validate_pinned<signed int, arithmetic_atomic_testers>();
  validate_pinned<signed long, arithmetic_atomic_testers>();
  validate_pinned<signed long long, arithmetic_atomic_testers>();

  validate_pinned<unsigned char, bitwise_atomic_testers>();
  validate_pinned<unsigned short, bitwise_atomic_testers>();
  validate_pinned<unsigned int, bitwise_atomic_testers>();
  validate_pinned<unsigned long, bitwise_atomic_testers>();
  validate_pinned<unsigned long long, bitwise_atomic_testers>();

#ifdef LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH
  validate_pinned<float, arithmetic_atomic_testers>();
  validate_pinned<double, arithmetic_atomic_testers>();
#else
  // NOTE(HIP/AMD): fetch_(sub/or/xor/and) are not supported with hipMallocManaged.
  // For float and double also fetch_add is not supported. This issue is tracked internally in issue SWDEV-390383.
  validate_pinned<float, basic_testers>();
  validate_pinned<double, basic_testers>();
#endif
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
