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

// FLAKY: constructing a hip::wthread during global-variable destruction is undefined behavior.
// HIP runtime shutdown also runs via global destructors/atexit handlers, and the ordering between
// them is unspecified, so depending on teardown order this test nondeterministically passes OR
// crashes. That makes both a plain pass and an expected-failure unstable expectations: it was
// previously an expected failure, but a compiler promotion in TheRock flipped it to an unexpected
// pass and broke the suite. Marking it unsupported is the only option that tolerates a
// nondeterministic result (the test never runs, so it can neither fail nor unexpectedly pass).
// See https://amd-hub.atlassian.net/browse/LCOMPILER-2560.
// UNSUPPORTED: target={{.*}}

#include "make_test_thread.h"

__device__ void func() {}

struct T {
        ~T() {
            // __thread_local_data is expected to be destroyed as it was created
            // from the main(). Now trigger another access.
            support::make_test_thread([] __device__() { func(); }).join();
        }
} t;
// __device__ T t2;

int main(int, char **) {
    // Triggers construction of __thread_local_data.
    support::make_test_thread([] __device__() { func(); }).join();

    return 0;
}
