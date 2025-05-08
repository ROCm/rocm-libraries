//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// template <>
// struct atomic<integral>
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(integral desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(integral desr, memory_order m = memory_order_seq_cst);
//     integral load(memory_order m = memory_order_seq_cst) const volatile;
//     integral load(memory_order m = memory_order_seq_cst) const;
//     operator integral() const volatile;
//     operator integral() const;
//     integral exchange(integral desr,
//                       memory_order m = memory_order_seq_cst) volatile;
//     integral exchange(integral desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order s, memory_order f);
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order m = memory_order_seq_cst);
//
//     integral
//         fetch_add(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_add(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_sub(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_sub(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_and(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_and(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_or(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_or(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_xor(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_xor(integral op, memory_order m = memory_order_seq_cst);
//
//     atomic() = default;
//     constexpr atomic(integral desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     integral operator=(integral desr) volatile;
//     integral operator=(integral desr);
//
//     integral operator++(int) volatile;
//     integral operator++(int);
//     integral operator--(int) volatile;
//     integral operator--(int);
//     integral operator++() volatile;
//     integral operator++();
//     integral operator--() volatile;
//     integral operator--();
//     integral operator+=(integral op) volatile;
//     integral operator+=(integral op);
//     integral operator-=(integral op) volatile;
//     integral operator-=(integral op);
//     integral operator&=(integral op) volatile;
//     integral operator&=(integral op);
//     integral operator|=(integral op) volatile;
//     integral operator|=(integral op);
//     integral operator^=(integral op) volatile;
//     integral operator^=(integral op);
// };

#include "gpu/atomic"
#include "hip/hip_runtime.h"
#include <cassert>
#include <new>

#include <cmpxchg_loop.h>

#include "hip_helpers.h"
#include "test_macros.h"

template <class A, class T>
__device__ void do_test() {
    A obj(T(0));
    assert(obj == T(0));
    bool b0 = obj.is_lock_free();
    ((void)b0); // mark as unused
    obj.store(T(0));
    assert(obj == T(0));
    obj.store(T(1), gpu::memory_order_release);
    assert(obj == T(1));
    assert(obj.load() == T(1));
    assert(obj.load(gpu::memory_order_acquire) == T(1));
    assert(obj.exchange(T(2)) == T(1));
    assert(obj == T(2));
    assert(obj.exchange(T(3), gpu::memory_order_relaxed) == T(2));
    assert(obj == T(3));
    T x = obj;
    assert(cmpxchg_weak_loop(obj, x, T(2)) == true);
    assert(obj == T(2));
    assert(x == T(3));
    assert(obj.compare_exchange_weak(x, T(1)) == false);
    assert(obj == T(2));
    assert(x == T(2));
    x = T(2);
    assert(obj.compare_exchange_strong(x, T(1)) == true);
    assert(obj == T(1));
    assert(x == T(2));
    assert(obj.compare_exchange_strong(x, T(0)) == false);
    assert(obj == T(1));
    assert(x == T(1));
    assert((obj = T(0)) == T(0));
    assert(obj == T(0));
    assert(obj++ == T(0));
    assert(obj == T(1));
    assert(++obj == T(2));
    assert(obj == T(2));
    assert(--obj == T(1));
    assert(obj == T(1));
    assert(obj-- == T(1));
    assert(obj == T(0));
    obj = T(2);
    assert((obj += T(3)) == T(5));
    assert(obj == T(5));
    assert((obj -= T(3)) == T(2));
    assert(obj == T(2));
    assert((obj |= T(5)) == T(7));
    assert(obj == T(7));
    assert((obj &= T(0xF)) == T(7));
    assert(obj == T(7));
    assert((obj ^= T(0xF)) == T(8));
    assert(obj == T(8));

    {
        TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {23};
        A &zero = *new (storage) A();
        assert(zero == 0);
        zero.~A();
    }
}

template <class A, class T>
__device__ void test() {
    do_test<A, T>();
    do_test<volatile A, T>();
}

__global__ void gmain() {
    // This is a very long test, so we break it up a bit to speed things up
    switch (blockIdx.x) {
    case 0:
        test<gpu::atomic_char, char>();
        test<gpu::atomic_schar, signed char>();
        test<gpu::atomic_uchar, unsigned char>();
        break;

    case 1:
        test<gpu::atomic_short, short>();
        test<gpu::atomic_ushort, unsigned short>();
        test<gpu::atomic_int, int>();
        test<gpu::atomic_uint, unsigned int>();
        break;

    case 2:
        test<gpu::atomic_long, long>();
        test<gpu::atomic_ulong, unsigned long>();
        test<gpu::atomic_llong, long long>();
        test<gpu::atomic_ullong, unsigned long long>();
        break;

    case 3:
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
        test<gpu::atomic_char8_t, char8_t>();
#endif
        test<gpu::atomic_char16_t, char16_t>();
        test<gpu::atomic_char32_t, char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        test<gpu::atomic_wchar_t, wchar_t>();
#endif
        break;

    case 4:
        test<gpu::atomic_int8_t, int8_t>();
        test<gpu::atomic_uint8_t, uint8_t>();
        test<gpu::atomic_int16_t, int16_t>();
        test<gpu::atomic_uint16_t, uint16_t>();
        break;

    case 5:
        test<gpu::atomic_int32_t, int32_t>();
        test<gpu::atomic_uint32_t, uint32_t>();
        test<gpu::atomic_int64_t, int64_t>();
        test<gpu::atomic_uint64_t, uint64_t>();
        break;

    case 6:
        test<volatile gpu::atomic_char, char>();
        test<volatile gpu::atomic_schar, signed char>();
        test<volatile gpu::atomic_uchar, unsigned char>();
        break;

    case 7:
        test<volatile gpu::atomic_short, short>();
        test<volatile gpu::atomic_ushort, unsigned short>();
        test<volatile gpu::atomic_int, int>();
        test<volatile gpu::atomic_uint, unsigned int>();
        break;

    case 8:
        test<volatile gpu::atomic_long, long>();
        test<volatile gpu::atomic_ulong, unsigned long>();
        test<volatile gpu::atomic_llong, long long>();
        test<volatile gpu::atomic_ullong, unsigned long long>();
        break;

    case 9:
        test<volatile gpu::atomic_char16_t, char16_t>();
        test<volatile gpu::atomic_char32_t, char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        test<volatile gpu::atomic_wchar_t, wchar_t>();
#endif
        break;

    case 10:
        test<volatile gpu::atomic_int8_t, int8_t>();
        test<volatile gpu::atomic_uint8_t, uint8_t>();
        test<volatile gpu::atomic_int16_t, int16_t>();
        test<volatile gpu::atomic_uint16_t, uint16_t>();
        break;

    case 11:
        test<volatile gpu::atomic_int32_t, int32_t>();
        test<volatile gpu::atomic_uint32_t, uint32_t>();
        test<volatile gpu::atomic_int64_t, int64_t>();
        test<volatile gpu::atomic_uint64_t, uint64_t>();
        break;

    default:
        break;
    }
}

int main(int, char **) {
    hipLaunchKernelGGL(gmain, dim3(12), dim3(1), 0, nullptr);
    __HIP_CHECK__(hipGetLastError());
    __HIP_CHECK__(hipDeviceSynchronize());
    return 0;
}
