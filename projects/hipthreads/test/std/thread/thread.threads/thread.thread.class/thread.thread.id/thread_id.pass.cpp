//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// hipthreads doesn't yet have support hashing thread IDs
// XFAIL: *

// <thread>

// template <class T>
// struct hash
// {
//     size_t operator()(T val) const;
// };

// Not very portable

#include <cassert>
#include <functional>
#include <hip/thread>

#include "test_macros.h"
#include "force_include_hip.h"

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    hip::wthread::id id1;
    hip::wthread::id id2 = hip::this_thread::get_id();
    typedef ::std::hash<hip::wthread::id> H;
#if TEST_STD_VER <= 14
    static_assert((::std::is_same<typename H::argument_type, hip::wthread::id>::value), "" );
    static_assert((::std::is_same<typename H::result_type, ::std::size_t>::value), "" );
#endif
    ASSERT_NOEXCEPT(H()(id2));
    H h;
    assert(h(id1) != h(id2));
#endif

  return 0;
}
