//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD
// XFAIL: *
// REASON: defer_lock_t, try_to_lock_t, adopt_lock_t not yet implemented in hipThreads

// <mutex>

// struct defer_lock_t { explicit defer_lock_t() = default; };
// struct try_to_lock_t { explicit try_to_lock_t() = default; };
// struct adopt_lock_t { explicit adopt_lock_t() = default; };

// This test checks for https://wg21.link/LWG2510.

#include <hip/mutex>

hip::defer_lock_t f1() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
hip::try_to_lock_t f2() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
hip::adopt_lock_t f3() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
