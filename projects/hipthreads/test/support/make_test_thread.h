//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -DTEST_PREP_DEVICE_FOR_THREADS

#ifndef TEST_SUPPORT_MAKE_TEST_THREAD_H
#define TEST_SUPPORT_MAKE_TEST_THREAD_H

#include <hip/thread>
#include <utility>

namespace support {

template <class F, class ...Args>
__host__ hip::wthread make_test_thread(F&& f, Args&& ...args) {
    return hip::wthread(::std::forward<F>(f), ::std::forward<Args>(args)...);
}
template <class F, class ...Args>
__device__ hip::wthread make_test_thread(F&& f, Args&& ...args) {
    return hip::wthread(::std::forward<F>(f), ::std::forward<Args>(args)...);
}

} // end namespace support

#endif // TEST_SUPPORT_MAKE_TEST_THREAD_H
