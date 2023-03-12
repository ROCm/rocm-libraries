//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// typedef atomic<int_least8_t>   atomic_int_least8_t;
// typedef atomic<uint_least8_t>  atomic_uint_least8_t;
// typedef atomic<int_least16_t>  atomic_int_least16_t;
// typedef atomic<uint_least16_t> atomic_uint_least16_t;
// typedef atomic<int_least32_t>  atomic_int_least32_t;
// typedef atomic<uint_least32_t> atomic_uint_least32_t;
// typedef atomic<int_least64_t>  atomic_int_least64_t;
// typedef atomic<uint_least64_t> atomic_uint_least64_t;
//
// typedef atomic<int_fast8_t>   atomic_int_fast8_t;
// typedef atomic<uint_fast8_t>  atomic_uint_fast8_t;
// typedef atomic<int_fast16_t>  atomic_int_fast16_t;
// typedef atomic<uint_fast16_t> atomic_uint_fast16_t;
// typedef atomic<int_fast32_t>  atomic_int_fast32_t;
// typedef atomic<uint_fast32_t> atomic_uint_fast32_t;
// typedef atomic<int_fast64_t>  atomic_int_fast64_t;
// typedef atomic<uint_fast64_t> atomic_uint_fast64_t;
//
// typedef atomic<intptr_t>  atomic_intptr_t;
// typedef atomic<uintptr_t> atomic_uintptr_t;
// typedef atomic<size_t>    atomic_size_t;
// typedef atomic<ptrdiff_t> atomic_ptrdiff_t;
// typedef atomic<intmax_t>  atomic_intmax_t;
// typedef atomic<uintmax_t> atomic_uintmax_t;

#include "gpu/atomic"
#include <type_traits>
#include <cstdint>

#include "test_macros.h"
#include "kernel_launcher.h"

__global__ void gmain()
{
    static_assert((std::is_same<gpu::atomic<  std::int_least8_t>,   gpu::atomic_int_least8_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::uint_least8_t>,  gpu::atomic_uint_least8_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::int_least16_t>,  gpu::atomic_int_least16_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::uint_least16_t>, gpu::atomic_uint_least16_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::int_least32_t>,  gpu::atomic_int_least32_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::uint_least32_t>, gpu::atomic_uint_least32_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::int_least64_t>,  gpu::atomic_int_least64_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::uint_least64_t>, gpu::atomic_uint_least64_t>::value), "");

    static_assert((std::is_same<gpu::atomic<  std::int_fast8_t>,   gpu::atomic_int_fast8_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::uint_fast8_t>,  gpu::atomic_uint_fast8_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::int_fast16_t>,  gpu::atomic_int_fast16_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::uint_fast16_t>, gpu::atomic_uint_fast16_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::int_fast32_t>,  gpu::atomic_int_fast32_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::uint_fast32_t>, gpu::atomic_uint_fast32_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::int_fast64_t>,  gpu::atomic_int_fast64_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::uint_fast64_t>, gpu::atomic_uint_fast64_t>::value), "");

    static_assert((std::is_same<gpu::atomic< std::intptr_t>,  gpu::atomic_intptr_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::uintptr_t>, gpu::atomic_uintptr_t>::value), "");
    static_assert((std::is_same<gpu::atomic<   std::size_t>,    gpu::atomic_size_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::ptrdiff_t>, gpu::atomic_ptrdiff_t>::value), "");
    static_assert((std::is_same<gpu::atomic< std::intmax_t>,  gpu::atomic_intmax_t>::value), "");
    static_assert((std::is_same<gpu::atomic<std::uintmax_t>, gpu::atomic_uintmax_t>::value), "");
}
