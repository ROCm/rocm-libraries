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

// <thread>

// class thread

// thread& operator=(thread&& t);

#include <gpu/thread>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <utility>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

struct G
{
    __device__ void operator()() { }
};

void f1()
{
    std::_Exit(0);
}

int main(int, char**)
{
#ifndef __HIP_DEVICE_COMPILE__
    std::set_terminate(f1);
#else
    {
        G g;
        gpu::thread t0 = support::make_test_thread(g);
        gpu::thread t1;
        t0 = std::move(t1);
        assert(false);
    }
#endif

    return 0;
}
