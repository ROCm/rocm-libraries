//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <thread>

// class wthread

// wthread& operator=(wthread&& t);

#include <hip/thread>
#include <hip/__support/misuse.h>
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

// The C++ standard says move-assigning onto a joinable thread calls
// std::terminate(). On the GPU there is no host terminate handler, and a device
// abort surfaces as an HSA hardware exception whose host exit code is not stable
// across runtimes, so the libc++ death-test form (set_terminate -> _Exit(0))
// cannot be verified by exit code here. Instead we put the library into nonfatal
// mode and verify that the misuse is *detected* (see hip/__thread/thread.h).
int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    {
        hip::__hipthreads_misuse_nonfatal = true;

        G g;
        hip::wthread t0 = support::make_test_thread(g);
        hip::wthread t1;

        unsigned int before = hip::__hipthreads_misuse_count;
        t0 = ::std::move(t1);                       // misuse: t0 is joinable
        assert(hip::__hipthreads_misuse_count == before + 1);

        // In nonfatal mode the assignment is skipped, so t0 still owns its
        // thread; join it for a clean teardown.
        t0.join();
    }
#endif

    return 0;
}
