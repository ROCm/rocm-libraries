//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD
// XFAIL: *
// REASON: defer_lock_t, try_to_lock_t, adopt_lock_t not yet implemented in hipThreads

// <mutex>

// struct defer_lock_t { explicit defer_lock_t() = default; };
// struct try_to_lock_t { explicit try_to_lock_t() = default; };
// struct adopt_lock_t { explicit adopt_lock_t() = default; };
//
// constexpr defer_lock_t  defer_lock{};
// constexpr try_to_lock_t try_to_lock{};
// constexpr adopt_lock_t  adopt_lock{};

#include <hip/mutex>

#include "test_macros.h"

int main(int, char**)
{
    typedef hip::defer_lock_t T1;
    typedef hip::try_to_lock_t T2;
    typedef hip::adopt_lock_t T3;

    T1 t1 = hip::defer_lock; ((void)t1);
    T2 t2 = hip::try_to_lock; ((void)t2);
    T3 t3 = hip::adopt_lock; ((void)t3);

    return 0;
}
