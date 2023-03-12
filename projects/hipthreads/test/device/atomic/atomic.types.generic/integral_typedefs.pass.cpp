//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// typedef atomic<char>               atomic_char;
// typedef atomic<signed char>        atomic_schar;
// typedef atomic<unsigned char>      atomic_uchar;
// typedef atomic<short>              atomic_short;
// typedef atomic<unsigned short>     atomic_ushort;
// typedef atomic<int>                atomic_int;
// typedef atomic<unsigned int>       atomic_uint;
// typedef atomic<long>               atomic_long;
// typedef atomic<unsigned long>      atomic_ulong;
// typedef atomic<long long>          atomic_llong;
// typedef atomic<unsigned long long> atomic_ullong;
// typedef atomic<char8_t>            atomic_char8_t; // C++20
// typedef atomic<char16_t>           atomic_char16_t;
// typedef atomic<char32_t>           atomic_char32_t;
// typedef atomic<wchar_t>            atomic_wchar_t;
//
// typedef atomic<intptr_t>           atomic_intptr_t;
// typedef atomic<uintptr_t>          atomic_uintptr_t;
//
// typedef atomic<int8_t>             atomic_int8_t;
// typedef atomic<uint8_t>            atomic_uint8_t;
// typedef atomic<int16_t>            atomic_int16_t;
// typedef atomic<uint16_t>           atomic_uint16_t;
// typedef atomic<int32_t>            atomic_int32_t;
// typedef atomic<uint32_t>           atomic_uint32_t;
// typedef atomic<int64_t>            atomic_int64_t;
// typedef atomic<uint64_t>           atomic_uint64_t;

#include "gpu/atomic"
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<gpu::atomic<char>, gpu::atomic_char>::value), "");
    static_assert((std::is_same<gpu::atomic<signed char>, gpu::atomic_schar>::value), "");
    static_assert((std::is_same<gpu::atomic<unsigned char>, gpu::atomic_uchar>::value), "");
    static_assert((std::is_same<gpu::atomic<short>, gpu::atomic_short>::value), "");
    static_assert((std::is_same<gpu::atomic<unsigned short>, gpu::atomic_ushort>::value), "");
    static_assert((std::is_same<gpu::atomic<int>, gpu::atomic_int>::value), "");
    static_assert((std::is_same<gpu::atomic<unsigned int>, gpu::atomic_uint>::value), "");
    static_assert((std::is_same<gpu::atomic<long>, gpu::atomic_long>::value), "");
    static_assert((std::is_same<gpu::atomic<unsigned long>, gpu::atomic_ulong>::value), "");
    static_assert((std::is_same<gpu::atomic<long long>, gpu::atomic_llong>::value), "");
    static_assert((std::is_same<gpu::atomic<unsigned long long>, gpu::atomic_ullong>::value), "");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    static_assert((std::is_same<gpu::atomic<wchar_t>, gpu::atomic_wchar_t>::value), "");
#endif
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    static_assert((std::is_same<gpu::atomic<char8_t>, gpu::atomic_char8_t>::value), "");
#endif
    static_assert((std::is_same<gpu::atomic<char16_t>, gpu::atomic_char16_t>::value), "");
    static_assert((std::is_same<gpu::atomic<char32_t>, gpu::atomic_char32_t>::value), "");

//  Added by LWG 2441
    static_assert((std::is_same<gpu::atomic<intptr_t>,  gpu::atomic_intptr_t>::value), "");
    static_assert((std::is_same<gpu::atomic<uintptr_t>, gpu::atomic_uintptr_t>::value), "");

    static_assert((std::is_same<gpu::atomic<int8_t>,    gpu::atomic_int8_t>::value), "");
    static_assert((std::is_same<gpu::atomic<uint8_t>,   gpu::atomic_uint8_t>::value), "");
    static_assert((std::is_same<gpu::atomic<int16_t>,   gpu::atomic_int16_t>::value), "");
    static_assert((std::is_same<gpu::atomic<uint16_t>,  gpu::atomic_uint16_t>::value), "");
    static_assert((std::is_same<gpu::atomic<int32_t>,   gpu::atomic_int32_t>::value), "");
    static_assert((std::is_same<gpu::atomic<uint32_t>,  gpu::atomic_uint32_t>::value), "");
    static_assert((std::is_same<gpu::atomic<int64_t>,   gpu::atomic_int64_t>::value), "");
    static_assert((std::is_same<gpu::atomic<uint64_t>,  gpu::atomic_uint64_t>::value), "");

  return 0;
}
