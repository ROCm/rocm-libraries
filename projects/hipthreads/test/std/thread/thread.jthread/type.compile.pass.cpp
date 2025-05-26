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

#include <gpu/thread>
#include <type_traits>

static_assert(std::is_same_v<std::jthread::id, gpu::thread::id>);
static_assert(std::is_same_v<std::jthread::native_handle_type, gpu::thread::native_handle_type>);
