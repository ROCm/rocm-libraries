//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

//  using id = thread::id;
//  using native_handle_type = thread::native_handle_type;

#include <hip/thread>
#include <type_traits>

static_assert(::std::is_same_v<hip::jthread::id, hip::thread::id>);

// Per the C++20 jthread synopsis, jthread::native_handle_type must be the same type
// as thread::native_handle_type (the standard defines it as a using-alias). Neither
// hip::thread nor hip::jthread exposes this member in this initial port; this check
// is dormant until both gain the alias, at which point it verifies consistency.
template <class JT, class T>
constexpr bool check_native_handle_type_same() {
    if constexpr (requires {
                      typename JT::native_handle_type;
                      typename T::native_handle_type;
                  }) {
        static_assert(::std::is_same_v<typename JT::native_handle_type, typename T::native_handle_type>);
    }
    return true;
}
static_assert(check_native_handle_type_same<hip::jthread, hip::thread>());
