//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// ADDITIONAL_COMPILE_FLAGS: -DTEST_USE_GPU_THREADS
// gpulib doesn't yet have support hashing thread IDs
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
#include <gpu/thread>

#include "test_macros.h"
#include "force_include_hip.h"

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    gpu::thread::id id1;
    gpu::thread::id id2 = gpu::this_thread::get_id();
    typedef std::hash<gpu::thread::id> H;
#if TEST_STD_VER <= 14
    static_assert((std::is_same<typename H::argument_type, gpu::thread::id>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
#endif
    ASSERT_NOEXCEPT(H()(id2));
    H h;
    assert(h(id1) != h(id2));
#endif

  return 0;
}
